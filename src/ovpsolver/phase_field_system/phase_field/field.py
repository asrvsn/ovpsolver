"""One phase field of an incompressible mixture.

A :class:`PhaseField` owns one conserved volume fraction ``phi`` and the
potential its transport is paired against, which has to be an unknown because a
cell-constant phase has no pointwise derivative. Which rates carry the phase, and
how much each carries, is the concrete field's
(:meth:`PhaseField.declare_velocity_elements`, :meth:`PhaseField.flux`): a
species that flows as a whole declares one velocity and one flux, one with a
load-bearing network two, carrying its sol/gel split.

What follows from a velocity merely existing is here. Each is constrained the
same two ways whatever it carries -- no crossing of the diffuse inclusion
surfaces except by the declared injection, and a pinned normal flux at ``Sigma``
-- so this class declares the multipliers and writes both terms. Likewise what a
velocity costs: the floored Darcy measure, the Brinkman term and the membrane
resistance are written against a velocity and a carried density, and the
concrete field states which of them each velocity pays.

Velocities and fluxes are declared separately because a flux is an expression in
bound functions, which do not exist until every element has been declared. The
fluxes answer the velocities one for one and in order; a velocity two pieces
would ride is declared once carrying their sum, so each ``Sigma`` condition is a
statement about a single carried and a single imposed value.

The field assembles its share of ``R = dE/dt + Psi + C`` and stops short of
differentiating it: the pressure enforcing ``sum_i phi_i = 1`` couples every
phase, so a per-field derivative would drop exactly the cross terms. The system
sums the contributions and differentiates once. Flory-Huggins cross terms are
the system's too, declared in the phases' own
:meth:`~ovpsolver.fem.elements.ElementOwner.variable` handles so they reach the
right rows unannounced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import ufl

from ...fem.elements import (
    ElementDomain,
    ElementOwner,
    Fields,
    Solve,
    StateElement,
    VelocityElement,
)
from ...fem.elements.dg0 import (
    DISCONTINUOUS_LAGRANGE,
    two_point_stiffness,
)
from ...fem.saveable import ParametricSaveable, Saveable, own_elements
from ...solver.regularization import smooth_max
from ..dissipative import Dissipative
from ..energy import EnergyDensity
from ..transport import Flux, Transported
from .parameters import PhaseFieldParameters

if TYPE_CHECKING:
    from dolfinx.fem import Function
    from ufl.core.expr import Expr

    from ...diffuse_domain import DiffuseDomain
    from ...fem.elements import ElementSpec
    from ...solver.parameters import SolverParameters


class PhaseField(
    ParametricSaveable[PhaseFieldParameters], Dissipative, ElementOwner, ABC
):
    """Abstract conserved phase field of an incompressible mixture."""

    # Bound by the solver from the ``test_attr`` of each element spec.
    _phi_test: "Expr"
    _aux_potential_test: "Expr"

    ## Identity and elements

    def __init__(
        self,
        solver_parameters: "SolverParameters",
        parameters: PhaseFieldParameters,
        diffuse_domain: "DiffuseDomain",
        k_B_T: float,
    ) -> None:
        super().__init__(solver_parameters, parameters, diffuse_domain)
        # A property of the mixture, not of this phase, so it arrives from the
        # system rather than from the material parameters.
        self.k_B_T = k_B_T

    def name(self) -> str:
        return self.parameters.name

    @abstractmethod
    def declare_velocity_elements(self) -> list[VelocityElement]:
        """The rates that carry this phase through the mixture.

        One per independent velocity: a phase that moves as a whole declares one,
        a phase with a load-bearing network declares the two its content is
        partitioned between. Each names the multipliers its constraints are paired
        with, and this class builds them.

        :meth:`flux` answers this list one for one, in order.
        """

    def declare_state_elements(self) -> list["ElementSpec"]:
        """Internal variables carried along by the phase. None by default."""

        return []

    def declare_elements(self) -> list["ElementSpec"]:
        """``phi``, the potential built on its derivatives, every velocity with
        its multipliers, and the states.

        ``phi`` is live because the convex part of the energy is evaluated at
        ``k+1``; ``aux_potential`` is live because its row reads ``phi`` at
        ``k+1`` (:meth:`aux_residual`). The multipliers follow from the
        velocities, which is why a subclass writes
        :meth:`declare_velocity_elements` and not this.
        """

        specs: list["ElementSpec"] = [
            StateElement(
                "phi",
                "_phi_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                positive=True,
                solve=Solve.ENERGY_LIVE,
            ),
            # Not positive: the Laplacian of a phase is negative wherever the
            # phase peaks, which is most of the support of a droplet.
            StateElement(
                "aux_potential",
                "_aux_potential_test",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                solve=Solve.ENERGY_LIVE,
            ),
        ]
        for velocity in self.declare_velocity_elements():
            specs.append(velocity)
            specs.extend(
                velocity.multipliers(has_inclusions=not self.diffuse_domain.uniform)
            )
        return specs + self.declare_state_elements()

    def declare_saveable(self) -> dict[str, Saveable]:
        """Every bulk state this phase holds, plus :meth:`drag_floor_indicator`.

        The floor that indicator reports is applied on this class, so every phase
        has one whatever it carries. Concrete phases add their own diagnostics.
        """

        return {
            **own_elements(self),
            "drag_floor_indicator": Saveable(
                self.element("phi"), self.drag_floor_indicator
            ),
        }

    def set_initial_conditions(self, phi: "Function") -> None:
        """Start both time levels at ``phi``, from rest."""

        for fields in (self.prev, self.next):
            self.set_initial_conditions_on(fields, phi)
        self.zero_lagged_rates()

    def set_initial_conditions_on(self, fields: Fields, phi: "Function") -> None:
        """One time level's initial state; a phase with internal state extends it."""

        fields.phi.interpolate(phi)
        fields.phi.x.scatter_forward()
        # Solved for, so this is only a starting iterate.
        fields.aux_potential.x.array[:] = 0.0
        fields.aux_potential.x.scatter_forward()

    ## Transport declaration

    @abstractmethod
    def flux(self) -> list[Flux]:
        """How much of the phase each declared velocity carries.

        One :class:`Flux` per entry of :meth:`declare_velocity_elements`, in that
        order, summing to the transport of the whole phase. The piece riding a
        rate supplies the density its ``Sigma`` multiplier pins and the value it
        is pinned to.
        """

    def surface_flux(self) -> "Expr":
        """The number flux of this phase injected at the inclusion surfaces.

        A bare number flux: each flux piece scales it by the stoichiometry of the
        variable it moves -- one monomer volume of phase per monomer, ``f``
        crosslinkable sites per monomer. Already smeared to a flux per unit
        volume, ``q sum_a |grad phi_a|``, which is what a transport row over
        ``Omega`` integrates; the spec's per-unit-area number is
        ``parameters.surface_flux_density``.
        """

        return self.diffuse_domain.q_eps_ufl(self.name())

    def transported_phase(self) -> Transported:
        """``phi`` itself, out of everything this field advects."""

        return self.transported()[0]

    def transported(self) -> list[Transported]:
        """Everything this field advects: ``phi``, and whatever rides along.

        The phase comes first and is always live, the free energy being a
        functional of ``phi`` at ``k+1``. A field with internal structure appends
        its states via ``super()``: quantities carried *by* the phase, and
        indistinguishable from it downstream.
        """

        return [Transported(self.element("phi"), flux=self.flux())]

    ## Variational declaration

    def energy_density(self) -> list[EnergyDensity]:
        """This field's free energy, already split into convex and concave parts.

        Written in the :meth:`variable` wrappers, so varying a term against any
        variable gives that variable's chemical potential. Bulk and
        diffuse-surface terms are declared together and told apart by their
        :class:`~ovpsolver.fem.elements.ElementDomain`. The Flory-Huggins cross
        terms are not here: they couple fields and belong to a coupling.

        What every phase has: the gradient penalty and the wetting affinity. A
        concrete field adds its own terms to these (``super()``), and may change
        or drop them.

        The gradient penalty ``(kappa / 2) |grad phi|^2`` is declared by its
        tangent ``aux_potential phi``. A cell-constant phase has no gradient to
        write the penalty in, so its derivative is an unknown of its own, defined
        by :meth:`aux_residual`; differentiated against ``phi``, the tangent hands
        that derivative to the rows exactly as every other term hands over its
        own. Its value is not the penalty's, and nothing reads it.
        """

        phi = self.variable("phi", time_level="next")
        return [
            EnergyDensity(convex=self.next.aux_potential * phi),
            EnergyDensity(
                convex=self.surface_affinity(phi),
                domain=ElementDomain.DIFFUSE_SURFACE,
            ),
        ]

    def surface_affinity(self, phi: "Expr") -> "Expr":
        """Wetting energy per unit area of diffuse inclusion surface.

        Linear in the phase, so its potential is a constant, which the initial
        wetting-equilibrium solve requires.
        """

        return self.k_B_T * self.parameters.surface_affinity * phi

    ## Dissipation

    def dissipation(self) -> ufl.Form:
        """The dissipation potential this field contributes, in the live rates.

        Only what no phase can decline and no coefficient of its own enters: the
        membrane resistance every velocity meets at the inclusion surfaces
        (:meth:`surface_crossing_penalty`), and the drag holding it to the
        inclusions' own velocity inside them (:meth:`inclusion_drag`). Everything
        material -- which velocity has a Darcy mobility and which only feels the
        floor, which drags against what -- is written against a velocity and a
        carried density, and the concrete field sums the terms it pays on top of
        this. Drag *against another phase* is a coupling's.

        The membrane resistance is a constraint only in the impermeable limit; at
        finite permeability it is a genuine dissipation, written as the supremum
        over its own cell-constant multiplier. Concave in that multiplier, it is
        worth the dissipation only where the multiplier is stationary, which a
        converged solve is and an intermediate iterate need not be.
        """

        total = self.nothing()
        for velocity, flux in zip(
            self.declare_velocity_elements(), self.flux(), strict=True
        ):
            total += self.surface_crossing_penalty(velocity, flux)
            total += self.inclusion_drag(flux.lagged_density, flux.live_velocity)
        return total

    def drag_measure(self, chi_eps: "Expr", carried: "Expr") -> "Expr":
        """The floored Darcy measure ``sqrt((chi_eps phi)^2 + (chi phi)_*^2)``.

        The Darcy dissipation's ``chi_eps phi``, floored at ``(chi phi)_*`` by
        the smooth maximum (:func:`~ovpsolver.solver.regularization.smooth_max`),
        so without a corner where the floor takes over. Below the floor the
        coefficient stops tracking the density, which caps the mobility at
        ``D / (chi phi)_*`` and leaves the velocity row solvable where the phase
        vanishes -- inside inclusions, and in the interior of a fully expelled
        wall. Every velocity's drag is measured by this, whether it has a Darcy
        term of its own (:meth:`darcy_dissipation`) or only the lift to it
        (:meth:`drag_floor_deficit`), so the floor a velocity sees does not
        depend on which dissipations its phase declares.

        A floor and not an additive off-support ramp, which would be non-monotone
        in sum with the physical term: a ramp decays quadratically to zero at the
        floor while ``chi_eps phi`` grows only linearly, so the total dips to
        roughly the floor value at ``phi = (chi phi)_*`` and recovers to the ramp
        amplitude at ``phi = 0``. An interface expelling the species crosses that
        well over one element, swinging the coefficient by two orders of magnitude
        between neighbours.
        """

        return smooth_max(chi_eps * carried, self.solver_parameters.drag_measure_floor)

    def drag_floor_indicator(self) -> "Expr":
        """How much of the Darcy measure here is floor rather than phase.

        ``1 - chi_eps phi / drag_measure``: one where the phase is absent and the
        floor carries the whole drag, falling as ``(chi phi)_*^2 / 2 (chi_eps
        phi)^2`` where the phase carries its own. The deficit
        :meth:`drag_floor_deficit` supplies, as a fraction of the measure.

        A drawable cell representative, not the floor itself: :meth:`drag_measure`
        floors the *tabulated* ``chi_eps`` at the bulk quadrature points, which
        varies across a cell in the band, exactly where the floor is interesting.
        ``analyze.drag_floor`` takes the integral that has to be faithful against
        the saved quadrature copy.
        """

        measure = self.diffuse_domain.chi_eps_analytic_ufl() * self.prev.phi
        return 1.0 - measure / smooth_max(measure, self.solver_parameters.drag_measure_floor)

    def darcy_dissipation(
        self, carried: "Expr", velocity: "Expr", diffusivity: "Expr | float"
    ) -> ufl.Form:
        """Darcy dissipation potential, lagged density against the live rate.

        ``diffusivity`` is an argument because it is a property of the velocity
        rather than of the phase: a phase with a sol and a gel velocity has one of
        each.
        """

        if float(diffusivity) <= 0.0:
            raise ValueError("Darcy dissipation requires a positive diffusivity")

        dx = self.solver_parameters.get_mesh_dx()
        return (
            self.drag_measure(self.diffuse_domain.chi_eps, carried)
            * ufl.dot(velocity, velocity)
            / (2.0 * diffusivity)
        ) * dx

    def drag_floor_deficit(
        self, carried: "Expr", velocity: "Expr", diffusivity: "Expr | float"
    ) -> ufl.Form:
        """Lift a velocity's drag measure to the floor where its phase is absent.

        A velocity whose dissipation is only interpenetration drag, surface drag
        and Brinkman viscosity has all three vanish with its density. This supplies
        the deficit ``drag_measure - chi_eps phi``, so the total measure it sees is
        the floored :meth:`drag_measure` a Darcy velocity gets, falling off as
        ``(chi phi)_*^2 / 2 chi_eps phi`` where the density carries its own drag.
        ``diffusivity`` is the velocity's, as in :meth:`darcy_dissipation`.
        """

        dx = self.solver_parameters.get_mesh_dx()
        if float(diffusivity) <= 0.0:
            return ufl.as_ufl(0.0) * dx

        chi_eps = self.diffuse_domain.chi_eps
        deficit = self.drag_measure(chi_eps, carried) - chi_eps * carried
        return 0.5 * deficit / diffusivity * ufl.dot(velocity, velocity) * dx

    def bulk_viscosity(
        self, carried: "Expr", velocity: "Expr", eta: "Expr | float"
    ) -> ufl.Form:
        """Brinkman screening viscosity for grid-scale velocity modes.

        Measured by the floored :meth:`drag_measure` rather than by the bare
        ``chi_eps phi``, so that it and the Darcy term reach their floor
        together. The screening length the pair defines,

            sqrt(eta (chi phi)_* / [(chi phi)_* / D]) = sqrt(eta D) = l_eta,

        is then the one they define where the phase carries them: the measure
        cancels, so the regularized region is a Brinkman problem at the physical
        screening length. Floor only the Darcy term and it is not a Brinkman
        problem at all: an ``L^2`` coefficient with no ``H^1`` term to screen
        against it, so the velocity off the support answers its forcing pointwise
        and grid-scale modes go undamped.

        ``eta`` is the viscosity itself, an argument for the reason
        :meth:`darcy_dissipation`'s diffusivity is; a concrete field forms it as
        ``l_eta^2 / D`` from that velocity's own screening length.
        """

        dx = self.solver_parameters.get_mesh_dx()
        domain = self.diffuse_domain
        return (
            0.5
            * (eta * self.drag_measure(domain.chi_eps, carried))
            * ufl.inner(ufl.grad(velocity), ufl.grad(velocity))
            * dx
        )

    def inclusion_drag(self, carried: "Expr", velocity: "Expr") -> ufl.Form:
        """Drag holding a velocity to the inclusions' own, inside them.

            (n_s / 2) int phi_incl phi |v|^2

        The diffuse-domain equations never say what the velocity is inside the
        inclusions: every term there carries ``chi_eps``, so energy and
        dissipation vanish together and the velocity their ratio determines is the
        one it would be outside. The interior is then a second, unphysical copy of
        the mixture, free to separate with only the mesh to seed it. This pins the
        extension at what the geometry states -- the inclusions are stationary, so
        zero. Being a dissipation it changes no equilibrium and cannot move the
        phases by itself: at ``v = 0`` it and its row vanish.

        Measured by ``phi_incl``
        (:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.inclusion_phase_ufl`)
        against the carried phase, as every other drag on the velocity is: what it
        resists is a *flux*, so a piece carrying nothing pays nothing and the term
        cannot make a velocity expensive where its Darcy drag is free. Unfloored,
        since the vanishing it covers is ``chi_eps``, not ``phi``.

        ``phi_incl`` is a half at the level set, so this adds roughly ``n_s eps``
        of tangential resistance per unit area, which at finite ``eps`` has to
        stay well under the slip the surfaces state through ``surface_drag``. From
        below it must beat the Darcy drag the interior copy runs on, which it does
        by ``n_s D / chi_eps``: parity at the level set for ``n_s D`` of one,
        growing without bound inwards.
        """

        drag = self.diffuse_domain.parameters.inclusion_drag
        dx = self.solver_parameters.get_mesh_dx()
        return (
            0.5
            * drag
            * self.diffuse_domain.inclusion_phase_ufl()
            * carried
            * ufl.dot(velocity, velocity)
            * dx
        )

    def surface_viscosity(
        self, carried: "Expr", velocity: "Expr", length: "Expr | float"
    ) -> ufl.Form:
        """Viscous resistance to a normal velocity varying along the surface.

            (eta_s / 2) int phi |P_eps grad(v.n_eps)|^2 (eps dGamma_eps)

        with ``eta_s = length^2``, stated as a length so that it compares directly
        with the bulk screening length (:meth:`bulk_viscosity`); ``length`` is an
        argument for the reason :meth:`darcy_dissipation`'s diffusivity is.

        The ``H^1`` companion of the membrane's normal drag, as the Brinkman term
        is of the bulk Darcy drag. The crossing dissipation
        (:meth:`surface_crossing_penalty`) costs only the *size* of the normal
        velocity, so a normal velocity alternating in sign along a surface is no
        dearer than a uniform one and a discrete surface is free to take the mesh's
        pattern into it. This costs exactly that alternation. No boundary
        condition meets it: a uniform crossing has no along-surface gradient, and a
        slip has no normal component on any level set. ``P_eps`` discards the
        normal derivative, so the profile a crossing takes through the band is
        unresisted and the contact angle stays a condition on the phase.

        Measured by ``eps dGamma_eps`` rather than ``dGamma_eps``, so ``eta_s`` is
        a viscosity in the units of the bulk ``eta`` acting through a
        dimensionless profile of the band. At the level set it resists a normal
        mode of along-surface wavenumber ``q`` ``3 eta_s D q^2`` times as hard as
        the Darcy drag and ``3 eta_s / eta`` times as hard as the Brinkman term,
        neither carrying the interface width or the mesh spacing, so one value
        means the same on any ``(h, eps)``. Per unit area it is ``eta_s eps``,
        vanishing with the interface width.

        It cannot be raised without bound. Flow approaching a curved impermeable
        surface has a normal component varying along every level set in the band,
        at the curvature scale ``q = 1 / R``; above about ``R^2 / 3D`` the term
        drags flow past the inclusion rather than filtering a mode. An injection
        varying along a surface asks for exactly the variation this resists.

        The density is floored by ``solver.drag_measure_floor``; the surface
        density is not, since a floor on the product would put the term in the
        bulk. The gradient comes from
        :meth:`~ovpsolver.diffuse_domain.DiffuseDomain.normal_rate_surface_gradient`,
        which leaves no division to the form.
        """

        dx = self.solver_parameters.get_mesh_dx()
        # Gates work and not just algebra: a zero coefficient annihilates the
        # form, but only after the gradient below has tabulated three quadrature
        # fields off the analytic geometry. Zero surface viscosity should cost
        # nothing, including at startup.
        if length == 0.0:
            return ufl.as_ufl(0.0) * dx
        eta = length**2

        gradient = self.diffuse_domain.normal_rate_surface_gradient(velocity)
        return (
            0.5
            * eta
            * self.diffuse_domain.parameters.domain_eps
            * smooth_max(carried, self.solver_parameters.drag_measure_floor)
            * ufl.dot(gradient, gradient)
            * dx
        )

    def surface_crossing_penalty(
        self, velocity: VelocityElement, flux: Flux
    ) -> ufl.Form:
        """Normal poroelastic drag of the phase crossing a diffuse surface.

        The crossing per unit volume is

            h = phi v.grad(phi_incl) + nu q_eps

        -- the achieved crossing less the injection the spec permits, so ``h = 0``
        is the flux condition holding, and a piece that injects nothing states
        non-crossing without being asked. It costs ``Psi_n(h) = int h^2 / 2 M_n``
        against the membrane mobility ``M_n = varsigma phi``, linear in the density
        that moves as the Darcy mobility ``D phi`` is; ``varsigma -> 0`` is the
        impermeable limit.

        Written as its Legendre transform,

            <lambda, h> - Psi_n^*(lambda),    Psi_n^*(lambda) = (1/2) int M_n lambda^2,

        concave in the cell-constant ``lambda`` this velocity declares and worth
        ``Psi_n`` wherever that multiplier is stationary. The supremum over
        cell-constant multipliers is the point: stationarity pins ``int_T h`` one
        cell at a time -- the crossing a cell-constant transport row conserves --
        whereas ``Psi_n`` assembled directly would hold ``h`` at every quadrature
        point, conditions finer than any row of the transport can see. The
        quadrature rule then decides how faithfully each cell's integral is
        taken, not how many conditions a cell imposes. ``M_n`` is floored by
        ``solver.drag_measure_floor``, because ``h`` carries the phase too: where a
        piece carries none of it (a gel velocity before anything has gelled), an
        unfloored mobility and ``h`` vanish together and the multiplier's row is
        empty.

        The gradient is asked of
        :meth:`~ovpsolver.diffuse_domain.DiffuseDomain.grad_phi_incl_degree` at
        ``flux_bc_quadrature``, a quadrature coefficient being readable only at
        the rule it was tabulated on; the surface density enters only through
        ``grad_phi_incl = n_eps dGamma_eps``, never as a divisor. Nothing on a
        domain without inclusions, which declares no multiplier for one.
        """

        if self.diffuse_domain.uniform:
            return self.nothing()

        domain = self.diffuse_domain.parameters
        degree = domain.flux_bc_quadrature
        dx = ufl.Measure(
            "dx",
            domain=self.solver_parameters.dolfinx_mesh,
            metadata={"quadrature_degree": degree},
        )
        v_dot_grad = ufl.dot(
            flux.live_velocity, self.diffuse_domain.grad_phi_incl_degree(degree)
        )
        # The injection per unit volume, smeared per inclusion, which is the
        # ``nu q_eps`` of the defect and not its per-unit-area counterpart.
        source = ufl.as_ufl(flux.surface_flux)
        injects = not isinstance(source, ufl.constantvalue.Zero)
        if injects and degree != domain.indicator_quadrature_degree:
            raise ValueError(
                f"{self.name()} injects at the inclusion surfaces, and that "
                "injection is tabulated on the bulk quadrature rule (degree "
                f"{domain.indicator_quadrature_degree}), which a flux condition "
                f"assembled at flux_bc_quadrature (degree {degree}) cannot read. A "
                "spec that injects needs the two degrees equal"
            )

        multiplier = getattr(self.rates, velocity.crossing_multiplier)
        h = flux.lagged_density * v_dot_grad
        if injects:
            h += source
        mobility = domain.surface_permeability * smooth_max(
            flux.lagged_density, self.solver_parameters.drag_measure_floor
        )
        integrand = multiplier * h - 0.5 * mobility * multiplier**2
        return integrand * dx

    ## Constraints

    def constraints(self) -> ufl.Form:
        """Each velocity's normal flux through ``Sigma``, against its multiplier.

        The only hard constraint a phase carries on its own. Crossing the diffuse
        inclusion surfaces is relaxed to a finite permeability and so is declared
        as a dissipation; incompressibility couples every phase and its multiplier
        is the pressure the system owns.
        """

        velocities, fluxes = self.declare_velocity_elements(), self.flux()
        if len(velocities) != len(fluxes):
            raise ValueError(
                f"{self.name()} declares {len(velocities)} velocities and states "
                f"{len(fluxes)} fluxes. Each velocity is pinned on the one flux "
                "that answers it, so a velocity with no flux would carry a "
                "multiplier and no constraint, and a flux on no velocity would "
                "leave a rate nothing pins"
            )

        total = self.nothing()
        for velocity, flux in zip(velocities, fluxes):
            total += self.boundary_constraint(
                flux, getattr(self.rates, velocity.boundary_multiplier)
            )
        return total

    def boundary_constraint(self, flux: Flux, multiplier: "Expr") -> ufl.Form:
        """Pin one velocity's normal flux through ``Sigma`` to its imposed value.

        Reads the target off the flux, the same number the transport row
        substitutes for the boundary integral. A flux that imposes nothing states
        non-crossing.

        Pinned in the multiplier's space and no more finely, which is not pointwise:
        with the P1 multiplier declared here, a run imposing nothing still has a
        small ``v.n`` on ``Sigma``, orthogonal to P1 rather than absent. See
        :meth:`~ovpsolver.fem.elements.VelocityElement.multipliers`.
        """

        ds = self.solver_parameters.get_mesh_ds()
        normal = ufl.FacetNormal(self.solver_parameters.dolfinx_mesh)
        return (
            multiplier
            * (
                flux.lagged_density * ufl.dot(flux.live_velocity, normal)
                + flux.boundary_flux
            )
            * ds
        )

    ## Rows

    def energy_rate(self, terms: list[EnergyDensity]) -> ufl.Form:
        """``dE/dt`` for this field: the release-rate term of the Rayleighian.

        Every live variable's transport paired against its potential, in the
        live velocities. ``terms`` is the whole energy of the mixture, not just
        :meth:`energy_density`, because the Flory-Huggins couplings that reach
        this field are declared by the system against the same
        :meth:`~ovpsolver.fem.elements.ElementOwner.variable` handles.

        A field whose energy changes for a reason that is not the transport of a
        declared variable adds that term by overriding this. Elastic stress power
        is the one case: the strain moment it is stored in is not an unknown of
        the rate solve, so there is no ``k+1`` value to differentiate a declared
        energy against.
        """

        rate = self.nothing()
        for variable in self.transported():
            rate += variable.energy_rate(terms)
        return rate

    def surface_potential(self) -> "Expr":
        """Derivative of the declared surface energy with respect to ``phi``.

        The wetting affinity of the diffuse inclusions: one of the two halves of
        the contact-angle condition, read by the initial wetting equilibrium and
        by whatever enforces that condition on a step.
        """

        potential = ufl.as_ufl(0.0)
        for term in self.energy_density():
            if term.domain is ElementDomain.DIFFUSE_SURFACE:
                potential = potential + term.potential(
                    self.variable("phi", time_level="next"),
                    self.variable("phi", time_level="prev"),
                )
        return potential

    def aux_residual(self) -> ufl.Form:
        """Weak definition of ``aux_potential``, the potential the flux is paired
        against, which has no pointwise value.

        In the continuum this is the variation of the ``chi_eps``-weighted
        Dirichlet energy ``int |grad phi|^2 chi_eps dx``, giving the standard
        diffuse-domain form (https://pmc.ncbi.nlm.nih.gov/articles/PMC3171464/)

            chi_eps aux_potential = -kappa div(chi_eps grad phi)
                                    + (dg/dphi) dGamma_eps.

        The Korteweg stress splits as

            -kappa chi_eps laplace(phi) - kappa grad(phi).grad(chi_eps),

        and what the flux is paired against carries ``1 / chi_eps``, since the flux
        is ``chi_eps``-weighted and :meth:`energy_rate` therefore needs
        ``(dE/dphi) / chi_eps`` for any energy. Against that factor the first
        term's ``chi_eps`` cancels *exactly*, leaving the Laplacian with no diffuse
        weight. The second does not cancel, and together with the diffuse-surface
        affinity is exactly ``C / chi_eps``, ``C`` being the contact-angle defect
        (:meth:`contact_angle_defect`). So

            aux_potential = -kappa laplace(phi) + C / chi_eps,

        exactly and with no penalty: the wetting condition is this energy's natural
        boundary condition at the diffuse surfaces, so writing the potential
        exactly is what imposes it.

        Cancelling a discrete ``chi_eps`` against a discrete ``chi_eps`` is the
        pitfall avoided, and why the Laplacian's weight is ``1``. The surviving
        ``1 / chi_eps`` cannot amplify: ``C`` carries a factor of ``dGamma_eps``,
        which decays into an inclusion at the same exponential rate ``chi_eps``
        does, so the quotient rises to ``6 / eps`` and falls once the indicator
        reaches its floor. A facet sum carrying no such factor, divided by a
        floored indicator, would make the row's conditioning depend on where the
        geometry sits.

        A cell-constant phase has no interior gradient, so the stiffness is the
        two-point sum over neighbouring cells, and ``C`` reaches the row as a form
        rather than a value.
        """

        test = self._aux_potential_test
        stiffness = two_point_stiffness(
            1.0,
            self.next.phi,
            test,
            self.solver_parameters.cell_centroids(),
            self.solver_parameters.get_mesh_dS(),
        )
        cells = (
            self.next.aux_potential * test * self.solver_parameters.get_mesh_dx()
        )
        return (
            cells
            - self.contact_angle_defect(test)
            - self.parameters.kappa * stiffness
        )

    def contact_angle_defect(self, test: "Expr") -> ufl.Form:
        r"""The wetting defect ``C``, as it reaches the potential row.

        ``C = kappa grad(phi).grad(phi_incl) + (dg/dphi) dGamma_eps`` is the
        residual of the wetting condition, zero exactly when the mixture meets
        the inclusions at the angle the affinity asks for. It carries one factor
        of the diffuse surface measure, ``grad(phi_incl) = n_eps dGamma_eps``.

        Returns ``int (C / chi_eps) w dx``, the contribution to
        :meth:`aux_residual`, rather than ``C``: with a cell-constant phase the
        Korteweg half is a facet sum in either of its writings
        (:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.contact_angle`) and has no
        cell value at all.

        The affinity stays a cell integral, carrying no derivative of the phase
        and so nothing on the facets; a facet sum is not a volume integral. Under
        the fitted contact angle the Korteweg half reproduces its cell integral
        exactly for any quadratic profile of the distance, so for such a profile
        the two halves cancel cell by cell where the condition holds.

        ``kappa`` is applied here and not in the geometry's half, that half being
        a quadrature of the inclusion geometry against a phase while the
        stiffness is the phase's own.
        """

        domain = self.diffuse_domain
        affinity = (
            self.surface_potential()
            * domain.dgamma_eps_over_chi_eps_analytic_ufl()
            * test
            * self.solver_parameters.get_mesh_dx()
        )
        korteweg = self.parameters.kappa * domain.contact_angle(
            self.next.phi, test
        )
        return korteweg + affinity

