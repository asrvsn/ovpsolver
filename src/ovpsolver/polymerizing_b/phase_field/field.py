"""A phase field whose monomers crosslink into a load-bearing gel.

Two velocities move the phase, and the split between them is itself a state: the
sol fraction is read off the moments of the generating function ``u`` of the
degree-of-polymerization distribution, which are advected with the phase and
advanced by an explicit chemical sub-step. All of it is declared, as one flux
partition and a list of transported states.

The declaration reaches its limit in one place, and it is a physical limit. The
gel's strain moment is Lie-transported: its stretching term is neither a flux
nor a source, so it writes its own row
(:class:`~ovpsolver.phase_field_system.transport.LieTransported`), and the same
term read from the velocity row is the elastic stress power
(:meth:`PolymerizingPhaseField.gel_stress_power`). The two are declared
separately and kept consistent by hand, which is what the semi-implicit stress
buys: the moment never becomes an unknown of the rate solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ufl

from ...fem.elements import (
    Fields,
    Solve,
    StateElement,
    VelocityElement,
)
from ...fem.elements.dg0 import (
    DISCONTINUOUS_LAGRANGE,
    add_scaled_identity,
    cell_scalars,
    cell_values,
)
from ...fem.saveable import ParametricSaveable, Saveable
from ...ovpsolver_ext import burgers_rhs_muscl_vanleer, max_cfl_timestep
from ...phase_field_system.energy import EnergyDensity
from ...phase_field_system.phase_field.field import PhaseField
from ...phase_field_system.transport import Flux, LieTransported, Transported
from ...solver.regularization import log_reg, log_reg_ratio
from .burgers import ssp_rk3
from .parameters import PolymerizingPhaseFieldParameters

if TYPE_CHECKING:
    from ufl.core.expr import Expr

    from ...fem.elements import ElementSpec


class PolymerizingPhaseField(
    PhaseField, ParametricSaveable[PolymerizingPhaseFieldParameters]
):
    """Sol/gel phase field with a finite-volume gelation state.

    The sol/gel split is carried by the moments ``(u_bar, u_trace)`` of the
    generating function ``u``. Those moments and ``c_x`` are ``ENERGY_LIVE``,
    unknowns of the rate solve, so the convex entropy is evaluated at ``k+1``;
    the distribution itself and the gel's elastic state are ``ENERGY_LAGGING``,
    advanced afterwards by the transport step.
    """

    parameters: PolymerizingPhaseFieldParameters

    _v_s_test: "Expr"
    _v_g_test: "Expr"

    ## Elements

    def declare_velocity_elements(self) -> list[VelocityElement]:
        """The sol and gel velocities, each with the multipliers it is
        constrained by."""

        shape = (self.solver_parameters.dolfinx_mesh.geometry.dim,)
        return [
            VelocityElement(
                "v_s",
                "_v_s_test",
                degree=2,
                shape=shape,
                boundary_multiplier="Lambda_s",
                crossing_multiplier="lambda_s",
            ),
            VelocityElement(
                "v_g",
                "_v_g_test",
                degree=2,
                shape=shape,
                boundary_multiplier="Lambda_g",
                crossing_multiplier="lambda_g",
            ),
        ]

    def declare_state_elements(self) -> list["ElementSpec"]:
        geometric_dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        num_z_cells = self.parameters.z_partition.num_cells

        return [
            # All non-negative, and declared so: ``u`` by its definition, the
            # moments being counts read off it, the modulus k_B_T times a
            # cycle-rank density, and the strain moment in the semidefinite
            # sense, which its square shape selects.
            StateElement(
                "c_x",
                "_c_x_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                positive=True,
                solve=Solve.ENERGY_LIVE,
            ),
            StateElement(
                "u_bar",
                "_u_bar_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                positive=True,
                solve=Solve.ENERGY_LIVE,
            ),
            StateElement(
                "u_trace",
                "_u_trace_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                positive=True,
                solve=Solve.ENERGY_LIVE,
            ),
            StateElement(
                "u",
                "_u_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                shape=(num_z_cells,),
                positive=True,
                solve=Solve.ENERGY_LAGGING,
            ),
            StateElement(
                "gel_shear_modulus",
                "_gel_shear_modulus_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                positive=True,
                solve=Solve.ENERGY_LAGGING,
            ),
            StateElement(
                "gel_strain_moment",
                "_gel_strain_moment_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                shape=(geometric_dim, geometric_dim),
                symmetry=True,
                positive=True,
                solve=Solve.ENERGY_LAGGING,
            ),
        ]

    def set_initial_conditions_on(self, fields: Fields, phi) -> None:
        super().set_initial_conditions_on(fields, phi)

        # All monomer and all unreacted (alpha = 0): c_x = f phi / nu,
        # a copy rather than a projection since the phase is cell-constant.
        cell_scalars(fields.c_x)[:] = cell_scalars(fields.phi)
        site_density = cell_scalars(fields.c_x)
        site_density *= self.parameters.monomer_functionality / float(
            self.parameters.monomer_volume
        )

        # The all-monomer u, whose profile in z is the partition's source weight.
        partition = self.parameters.z_partition
        monomers_only = partition.source_weight(
            self.parameters.monomer_functionality, np.arange(partition.num_cells)
        )
        cell_values(fields.u)[:] = site_density[:, None] * monomers_only[None, :]
        self.sync_moments(fields)

        cell_scalars(fields.gel_shear_modulus)[:] = 0.0
        fields.gel_strain_moment.x.array[:] = 0.0

    def declare_saveable(self):
        """The states, plus the sol/gel split and the stress read off them.

        Each is a pointwise function of cell-constant states, so it is saved
        exactly in one of their spaces -- ``phi``'s for a scalar,
        ``gel_strain_moment``'s for the tensor. ``phi_sol`` and ``phi_gel`` in
        particular stand beside the other phases' ``phi`` in a saturation sum at
        full precision.
        """

        stress = self.element("gel_strain_moment")
        cellwise = self.element("phi")
        return {
            **super().declare_saveable(),
            "reaction_extent": Saveable(cellwise, self.reaction_extent),
            "gel_stress": Saveable(stress, self.gel_stress),
            "gel_hydrostatic_stress": Saveable(cellwise, self.gel_hydrostatic_stress),
            "gel_deviatoric_stress": Saveable(cellwise, self.gel_deviatoric_stress),
            # At ``next``, against their default of ``prev``, which is lagged for
            # the transport rows: a saved frame should not report a step-old
            # value.
            "phi_sol": Saveable(cellwise, lambda: self.phi_sol(time_level="next")),
            "phi_gel": Saveable(cellwise, lambda: self.phi_gel(time_level="next")),
        }

    ## Sol/gel split

    def phi_sol(self, *, time_level: str = "prev") -> "Expr":
        """``phi_s = nu m_1``: the part of the phase still in the sol.

        A volume fraction: with :meth:`phi_gel` it partitions ``phi``, so the two
        stand beside the other phases' ``phi`` in the saturation sum.

        Read straight off the moments, with no clip and no division, and that is
        load-bearing. Being exactly affine in the moments is what makes
        ``phi - phi_s`` a transported quantity too: its flux is a difference of
        two fluxes whose sol parts cancel to the last bit, so the gel rides the
        gel velocity alone and cannot go negative from the sol's motion. A clip
        would leave a residue of the sol flux in the gel's balance, of the size
        of the clip, with nothing to bound it. What keeps the difference
        non-negative instead is ``nu m_1 <= phi``, which
        :meth:`check_sol_fraction` checks.

        Nothing bars ``m_1`` from zero directly: the live entropy takes the
        logarithm of ``m_0``, and reaches the sol only through ``m_0 <= m_1``,
        which the trace moment's ``O(dz)`` error makes inexact at pure monomer.
        The barrier protects the sol nonetheless, the error being proportional
        to the moments rather than added to them;
        :mod:`ovpsolver.phase_field_system.analyze.gelation` gives the argument
        and measures the link.
        """

        fields = self.fields_at(time_level)
        m_1 = (2.0 * fields.u_bar - fields.u_trace) / (
            self.parameters.monomer_functionality - 2
        )
        return self.parameters.monomer_volume * m_1

    def phi_gel(self, *, time_level: str = "prev") -> "Expr":
        """``phi - phi_s``: the part of the phase bound into the network.

        The difference and not ``(1 - theta) phi``, so that the two halves sum to
        the phase identically and the saturation constraint sees exactly the phase
        a species with no internal structure would give it. There is no ``phi_g``
        state; the network is whatever the moments leave over.
        """

        fields = self.fields_at(time_level)
        return fields.phi - self.phi_sol(time_level=time_level)

    ## Transport

    def sol_flux_carrying(
        self, lagged_density: "Expr | float", per_monomer: "Expr | float" = 0.0
    ) -> Flux:
        """The sol velocity carrying ``lagged_density``: the phase, or any state.

        ``per_monomer`` is how much of the variable arrives with each monomer
        injected -- one monomer volume for the phase, ``f`` sites for ``c_x`` --
        so every variable states its own stoichiometry while the velocity says
        where monomers are injected.
        """

        return Flux(
            live_velocity=self.rates.v_s,
            lagged_velocity=self.prev.v_s,
            lagged_density=lagged_density,
            surface_flux=per_monomer * self.surface_flux(),
            boundary_flux=per_monomer * self.parameters.boundary_flux_density,
        )

    def gel_flux_carrying(self, lagged_density: "Expr | float") -> Flux:
        """The gel velocity carrying ``lagged_density``, injecting nothing anywhere.

        The network crosses neither the inclusion surfaces nor the outer
        boundary, and its crosslinks are born by chemistry rather than influx.
        """

        return Flux(
            live_velocity=self.rates.v_g,
            lagged_velocity=self.prev.v_g,
            lagged_density=lagged_density,
        )

    def sol_flux(self) -> Flux:
        """The sol half of the phase, on the sol velocity."""

        return self.sol_flux_carrying(
            self.phi_sol(), self.parameters.monomer_volume
        )

    def gel_flux(self) -> Flux:
        """The gel half of the phase, on the gel velocity."""

        return self.gel_flux_carrying(self.phi_gel())

    def flux(self) -> list[Flux]:
        return [self.sol_flux(), self.gel_flux()]

    def sol_gel_fluxes(
        self,
        sol: "Expr",
        gel: "Expr",
        per_monomer: "Expr | float" = 0.0,
    ) -> list[Flux]:
        """One variable's flux, as the parts of it the two velocities carry.

        An injected monomer's sites all arrive unreacted and in the sol, so one
        ``per_monomer`` on the sol part states the variable's whole influx.
        """

        return [
            self.sol_flux_carrying(sol, per_monomer),
            self.gel_flux_carrying(gel),
        ]

    def transported(self) -> list[Transported]:
        parameters = self.parameters
        functionality = parameters.monomer_functionality
        partition = self.parameters.z_partition

        # A site in the gel is a site on a chain reaching z = 1, so the gel
        # velocity carries c_xg = u|_{z=1} of every site-counting state, weighted
        # as each counts a site.
        c_x_gel = self.prev.u_trace
        z_trace = partition.midpoint(-1)
        bins = np.arange(partition.num_cells)
        z_bin = ufl.as_vector([float(value) for value in partition.midpoint(bins)])
        source_weight = ufl.as_vector(
            [float(value) for value in partition.source_weight(functionality, bins)]
        )
        trace_weight = float(partition.source_weight(functionality, -1))

        return super().transported() + [
            # The states the entropy's moments are affine in, hence ENERGY_LIVE.
            Transported(
                self.element("c_x"),
                flux=self.sol_gel_fluxes(
                    self.prev.c_x - c_x_gel, c_x_gel, functionality
                ),
            ),
            Transported(
                self.element("u_bar"),
                flux=self.sol_gel_fluxes(
                    self.prev.u_bar - 0.5 * c_x_gel,
                    0.5 * c_x_gel,
                    0.5 * (functionality - 2),
                ),
            ),
            Transported(
                self.element("u_trace"),
                flux=self.sol_gel_fluxes(
                    self.prev.u_trace - z_trace * c_x_gel,
                    z_trace * c_x_gel,
                    functionality * trace_weight,
                ),
            ),
            # The distribution the moments summarize. The energy sees it only
            # through them, so it stays out of the rate solve, where it would
            # cost one unknown per gelation bin per cell.
            Transported(
                self.element("u"),
                flux=self.sol_gel_fluxes(
                    self.prev.u - z_bin * c_x_gel,
                    z_bin * c_x_gel,
                    functionality * source_weight,
                ),
            ),
            # The elastic state is a property of the network, already in
            # chi_eps-weighted form, so it carries no diffuse-domain weight.
            #
            # Both read the solved gel velocity to pick the upwind cell. Nothing
            # live is a function of either -- their stress reaches the velocity
            # rows lagged -- so the selector is free, and each carries only its
            # own previous value, so reading it genuinely upwind cancels the
            # density out of the step bound. A lagged selector divides the bound
            # by whatever ratio the gel front holds across a facet, which on a
            # threshold front with no diffusion to smooth it reaches 1e5 and
            # collapses the step.
            Transported(
                self.element("gel_shear_modulus"),
                flux=[self.gel_flux_carrying(self.prev.gel_shear_modulus)],
                chi_weighted=False,
                upwind_live=True,
            ),
            # Positive in the semidefinite sense: checked on its eigenvalues,
            # with no step bound of its own (Transported.entrywise_positive).
            LieTransported(
                self.element("gel_strain_moment"),
                flux=[self.gel_flux_carrying(self.prev.gel_strain_moment)],
                chi_weighted=False,
                upwind_live=True,
                velocity=self.rates.v_g,
            ),
        ]

    ## Free energy

    def reaction_extent(
        self,
        phi: "Expr | None" = None,
        c_x: "Expr | None" = None,
        *,
        time_level: str = "next",
    ) -> "Expr":
        """The conversion ``alpha``, for diagnostics only.

        The quotient is meaningless once the species has drained out of a region,
        and the floor only keeps it finite there. Called bare it reports on the
        state, which is what makes it saveable; the arguments ask the same of
        other expressions.
        """

        if phi is None or c_x is None:
            fields = self.fields_at(time_level)
            phi = fields.phi if phi is None else phi
            c_x = fields.c_x if c_x is None else c_x
        return 1.0 - (
            self.parameters.monomer_volume
            * c_x
            / (self.parameters.monomer_functionality * ufl.max_value(phi, 1.0e-14))
        )

    def bulk_affinity(self, phi: "Expr", c_x: "Expr") -> "Expr":
        """``k_B_T omega_i(alpha_i) phi_i``, the phase's uniform affinity.

        The affinity, linear in the conversion, written in the reacted-site
        fraction:

            omega_i phi_i = d^0 phi_i + d^1 r_i,
            r_i = phi_i alpha_i = phi_i - (nu_i / f_i) c_i^x,

        which is the definition of ``alpha_i`` cleared of its denominator. That
        form and not ``alpha_i`` itself, because ``alpha_i`` is a ratio against
        the phase and so is neither bounded nor differentiable where the species
        has drained out, while ``r_i`` is affine in the state everywhere. Both
        coefficients are needed; ``bulk_affinity_0`` on
        :class:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseFieldParameters`
        says why.

        Affine in both arguments, so each potential it contributes is a
        constant: it forces no integration rule, and can be declared alongside
        the moment entropy without the declaration having to be split.
        """

        parameters = self.parameters
        # Kept symbolic rather than evaluated: monomer_volume and the two
        # coefficients are fem.Constants, so a spec that changes any of them
        # reuses the compiled form.
        reacted_fraction = (
            phi
            - (parameters.monomer_volume / parameters.monomer_functionality) * c_x
        )
        return self.k_B_T * (
            parameters.bulk_affinity_0 * phi
            + parameters.bulk_affinity_1 * reacted_fraction
        )

    def convex_entropy(self, c_x: "Expr", u_bar: "Expr") -> "Expr":
        """``k_B_T ent^cvx`` of the moment entropy's convex split."""

        monomer_volume = self.parameters.monomer_volume
        m_0 = 0.5 * c_x - u_bar
        return (
            self.k_B_T
            * (monomer_volume / self.parameters.monomer_polymerization_degree)
            * m_0
            * log_reg(monomer_volume * m_0, self.solver_parameters.log_reg_delta)
        )

    def concave_entropy(self, c_x: "Expr", u_bar: "Expr", u_trace: "Expr") -> "Expr":
        """``k_B_T ent^ccv`` of the moment entropy's convex split."""

        monomer_volume = self.parameters.monomer_volume
        m_0 = 0.5 * c_x - u_bar
        m_1 = (2.0 * u_bar - u_trace) / (self.parameters.monomer_functionality - 2)
        # Kept as a single regularized ratio rather than a difference of logs:
        # m_0 / m_1 is the reciprocal number-average degree, an O(1) quantity
        # even where both moments vanish, and only the ratio form stays jointly
        # convex under regularization (see log_reg_ratio).
        return (
            self.k_B_T
            * (monomer_volume / self.parameters.monomer_polymerization_degree)
            * log_reg_ratio(m_0, m_1, self.solver_parameters.log_reg_delta)
        )

    def energy_density(self) -> list[EnergyDensity]:
        """Mixing entropy in the moments and the affinity, on the base terms.

        One declaration covers the ``c_x``, ``u_bar`` and ``u_trace`` rows at
        once. There is no entropy in ``phi``: the moments carry it.
        """

        phi_next = self.variable("phi", time_level="next")
        c_x_next = self.variable("c_x", time_level="next")
        c_x_prev = self.variable("c_x", time_level="prev")
        u_bar_next = self.variable("u_bar", time_level="next")
        u_bar_prev = self.variable("u_bar", time_level="prev")
        u_trace_prev = self.variable("u_trace", time_level="prev")

        return [
            EnergyDensity(
                convex=self.convex_entropy(c_x_next, u_bar_next),
                concave=self.concave_entropy(c_x_prev, u_bar_prev, u_trace_prev),
            ),
            EnergyDensity(convex=self.bulk_affinity(phi_next, c_x_next)),
            *super().energy_density(),
        ]

    ## Gel stress

    def gel_stress(self, *, time_level: str = "next") -> "Expr":
        """The Cauchy gel stress ``T = m - vartheta I``."""

        fields = self.fields_at(time_level)
        identity = ufl.Identity(self.solver_parameters.dolfinx_mesh.geometry.dim)
        return fields.gel_strain_moment - fields.gel_shear_modulus * identity

    def gel_hydrostatic_stress(self, *, time_level: str = "next") -> "Expr":
        """``tr(T) / d`` for the gel stress tensor."""

        geometric_dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        return ufl.tr(self.gel_stress(time_level=time_level)) / geometric_dim

    def gel_deviatoric_stress(self, *, time_level: str = "next") -> "Expr":
        """The Frobenius norm of the deviatoric gel stress."""

        identity = ufl.Identity(self.solver_parameters.dolfinx_mesh.geometry.dim)
        deviatoric = self.gel_stress(time_level=time_level) - (
            self.gel_hydrostatic_stress(time_level=time_level) * identity
        )
        return ufl.sqrt(ufl.inner(deviatoric, deviatoric))

    def gel_stress_power(self) -> ufl.Form:
        """Elastic stress power.

        The velocity-row counterpart of the stretching in
        :class:`~ovpsolver.phase_field_system.transport.LieTransported`, and the
        one thing in this field the declaration layer cannot derive.
        Substituting the strain moment's discrete update into the increment gives

            dE_el = dt T_k : grad(v) + (dt^2 / 2) m_k : grad(v)^T grad(v),

        whose second term is the ``dt^2`` term the transport row drops from
        forward Euler to stay in the PSD cone: the two change together, or the
        energy the step releases stops being what its transport released.

        Both terms are in the lagged moment, which is why the strain moment need
        not be live: the strain energy is linear in ``m``, so the update leaves
        ``m_k`` as a coefficient and nothing at ``k+1`` -- no Bregman divergence,
        and no convex split. The second term is ``m_k``'s quadratic form in the
        velocity gradient, non-negative for a PSD moment, so the rate problem
        stays convex.

        Returned as the rate ``dE_el / dt``, since the Rayleighian is a rate
        functional: hence no ``dt`` on the first term and one on the second.
        """

        dx = self.solver_parameters.get_mesh_dx()
        gradient = ufl.grad(self.rates.v_g)
        return (
            ufl.inner(self.gel_stress(time_level="prev"), gradient) * dx
            + 0.5
            * self.solver_parameters.dt_live
            * ufl.inner(
                self.prev.gel_strain_moment, ufl.dot(gradient.T, gradient)
            )
            * dx
        )

    def energy_rate(self, terms: list[EnergyDensity]) -> ufl.Form:
        return super().energy_rate(terms) + self.gel_stress_power()

    ## Dissipation

    def dissipation(self) -> ufl.Form:
        # The same partition the transport rows carry, unclipped: its halves
        # are non-negative by the chain analyze.gelation checks rather than by a
        # max, and drag_measure floors whatever reaches it, as it does a bare phi.
        phi_sol = self.phi_sol()
        phi_gel = self.phi_gel()
        parameters = self.parameters
        # The sol's diffusivity scales the gel's terms too (see sol_diffusivity).
        diffusivity = parameters.sol_diffusivity
        sol_eta = parameters.sol_bulk_viscous_screening_length**2 / diffusivity
        gel_eta = parameters.gel_bulk_viscous_screening_length**2 / diffusivity
        return (
            super().dissipation()
            + self.darcy_dissipation(phi_sol, self.rates.v_s, diffusivity)
            + self.diffuse_domain.drag(
                self.parameters.surface_sol_drag * phi_sol, self.rates.v_s
            )
            + self.diffuse_domain.drag(
                self.parameters.surface_gel_drag * phi_gel, self.rates.v_g
            )
            + self.bulk_viscosity(phi_sol, self.rates.v_s, sol_eta)
            + self.bulk_viscosity(phi_gel, self.rates.v_g, gel_eta)
            # The sol drag measure is floored inside darcy_dissipation; the gel
            # has no bulk Darcy term of its own, so it takes the deficit up to
            # the same drag_measure.
            + self.drag_floor_deficit(phi_gel, self.rates.v_g, diffusivity)
            + self.solver_parameters.boundary_drag(
                self.parameters.boundary_gel_drag * phi_gel, self.rates.v_g
            )
            + self.surface_viscosity(
                phi_sol, self.rates.v_s, parameters.sol_surface_viscous_screening_length
            )
            + self.surface_viscosity(
                phi_gel, self.rates.v_g, parameters.gel_surface_viscous_screening_length
            )
        )

    ## Chemical sub-step

    def gelation_max_dt(self) -> float:
        """The longest chemistry sub-step this phase can presently take.

        The Burgers step's CFL bound with this phase's own ``CFL_u`` applied, so
        a caller sub-cycling the irreversible half-step need not know what
        bounds it. Infinite where ``u`` vanishes everywhere: there is then no
        wave to resolve.
        """

        bound = self.parameters.gelation_burgers_cfl * max_cfl_timestep(
            cell_values(self.next.u),
            self.parameters.z_partition.edges,
            self.parameters.crosslinking_rate,
        )
        return bound if np.isfinite(bound) else float("inf")

    def shear_birth_rate(self) -> np.ndarray:
        """Per-cell shear-modulus birth rate ``b^theta``, scaled by
        ``cycle_shear_modulus``.

        The zero right-boundary slope makes the left trace ``u|_{z=1}`` exactly
        the last cell average, so it is read directly.

        Every coefficient is taken through ``float``: a declared number reads
        back as a ``fem.Constant`` once the run has a mesh, and numpy multiplied
        by one produces an object array rather than failing, which the compiled
        kernels and the in-place updates cannot take.
        """

        left_trace = cell_values(self.next.u)[:, -1]
        return (
            0.5
            * float(self.parameters.cycle_shear_modulus)
            * float(self.k_B_T)
            * float(self.parameters.crosslinking_rate)
            * left_trace**2
        )

    def sync_moments(self, fields: Fields) -> None:
        """Recompute ``u_bar`` and ``u_trace`` from the ``u`` cells."""

        values = cell_values(fields.u)
        cell_scalars(fields.u_bar)[:] = values @ self.parameters.z_partition.widths
        cell_scalars(fields.u_trace)[:] = values[:, -1]

    def irreversible_timestep(
        self, dt: float, *, chi_eps_finite_volume: np.ndarray
    ) -> None:
        """Advance this phase's crosslinking by one sub-step of the mixture's loop.

        Three updates, each in the form that keeps its own field admissible: the
        exact Riccati step for ``c_x``, positive at any step; forward Euler for
        the elastic state, whose source only adds; and one SSP-RK3 MUSCL/van
        Leer step for ``u``, which is what :meth:`gelation_max_dt` bounds.

        The birth rate is read off ``u`` before the Burgers step advances it,
        and the moments are re-synced after, since the rate solve reads them.
        """

        gamma = self.parameters.crosslinking_rate
        fields = self.next

        weighted_birth_rate = chi_eps_finite_volume * self.shear_birth_rate()
        cell_scalars(fields.gel_shear_modulus)[:] += dt * weighted_birth_rate
        add_scaled_identity(fields.gel_strain_moment, dt * weighted_birth_rate)

        concentration = cell_scalars(fields.c_x)
        concentration[:] = concentration / (gamma * concentration * dt + 1.0)

        edges = self.parameters.z_partition.edges

        def burgers_rhs(state: np.ndarray, rate: np.ndarray) -> None:
            burgers_rhs_muscl_vanleer(state, rate, edges, gamma)

        sol_before = self.sol_fraction_values(fields)
        ssp_rk3(cell_values(fields.u), burgers_rhs, dt)
        self.sync_moments(fields)
        self.check_sol_fraction(fields, sol_before)

    def sol_fraction_values(self, fields: Fields) -> np.ndarray:
        """``nu m_1`` per cell: :meth:`phi_sol` read off the moment arrays.

        As numbers rather than UFL because the caller is a numpy sub-step, which
        would otherwise interpolate an expression per sub-step to compare two
        arrays.
        """

        functionality = self.parameters.monomer_functionality
        m_1 = (
            2.0 * cell_scalars(fields.u_bar) - cell_scalars(fields.u_trace)
        ) / (functionality - 2)
        return float(self.parameters.monomer_volume) * m_1

    def check_sol_fraction(self, fields: Fields, before: np.ndarray) -> None:
        """Check the two conditions that keep ``phi_gel = phi - nu m_1`` non-negative.

        ``phi_gel`` is a difference of transported quantities, so no step bound
        reaches it: :meth:`Transported.cfl_limit` bounds the declared rows, and a
        difference of two has none. What stands in is that each half of the step
        preserves the ordering for a reason of its own, and both are checked
        here, where the arrays are in hand.

        ``nu m_1 <= phi`` is the ordering itself, the sol being part of the
        phase. The chemistry does not touch ``phi``, so a violation is the
        mechanical step's.

        ``nu m_1`` non-increasing is why the chemistry cannot break it. The wave
        velocity ``gamma u`` is non-negative, so ``z = 1`` is an outflow: sol
        leaves into the gel and nothing returns. Integrating the Burgers equation
        for ``u`` over the partition,

            d(m_1)/dt = -[gamma (u|_{z=1})^2 + d(u|_{z=1})/dt] / (f - 2),

        so the sol falls unless the gel's unreacted-site concentration
        ``c_xg = u|_{z=1}`` drains faster than ``gamma c_xg^2``. That holds
        while the network grows, but as a property of the limiter at the last
        cell rather than an identity, hence measured.

        Together they close the induction the mechanical step needs: its flux
        densities are ``phi_sol`` and ``phi_gel`` at ``k``, so a step that starts
        from a valid split and lands on one leaves the next a valid start.
        """

        floor = self.solver_parameters.timestepping.positivity_floor_throw
        if floor is None:
            return

        after = self.sol_fraction_values(fields)
        gel = cell_scalars(fields.phi) - after
        violations = []
        if gel.size and float(gel.min()) < floor:
            violations.append(
                f"phi - nu m_1 min={float(gel.min()):.6e}, so the gel fraction is "
                f"negative in {int((gel < floor).sum())} cell(s)"
            )
        # Against the pre-step value and not zero: the sub-step converts sol to
        # gel, and a rise means the limiter put mass back across z = 1.
        rise = after - before
        tolerance = -floor * max(1.0, float(np.abs(before).max(initial=0.0)))
        if rise.size and float(rise.max()) > tolerance:
            violations.append(
                f"nu m_1 rose by {float(rise.max()):.6e} over the sub-step in "
                f"{int((rise > tolerance).sum())} cell(s)"
            )
        if violations:
            raise ValueError(
                f"{self.name()}: the sol/gel split left the admissible set during "
                f"the crosslinking sub-step; " + "; ".join(violations)
            )
