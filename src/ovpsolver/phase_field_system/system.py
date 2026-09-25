"""An incompressible mixture of phase fields, and the step it assembles.

:class:`PhaseFieldSystem` is the discrete-time Rayleighian of the mixture,

    R = dE/dt + Psi + C

It owns the two things no single phase can: the diffuse geometry they all live
on, and the pressure enforcing ``sum_i phi_i = 1``. Everything else it gets by
asking.

Which way things flow
---------------------
Declarations come *up*. Each phase declares its energy, its friction, its
impermeability and what carries it, knowing nothing about the others, and the
assembled Rayleighian is differentiated once at the top: the pressure couples
every phase, so differentiating per phase would drop exactly the cross terms.

The free energy goes back *down*, because a mixture has one free energy: a phase
declares terms of it, the system adds the couplings, and the whole list is handed
back so each variable's potential is the derivative of all of it. Letting a phase
differentiate only its own terms is what would lose Flory-Huggins.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from ufl.constantvalue import Zero

from ..diffuse_domain import DiffuseDomain
from ..fem.elements import ElementDomain, ElementOwner, StaticElement
from ..fem.elements.dg0 import (
    DISCONTINUOUS_LAGRANGE,
    dg0_upwind_ibp,
    two_point_stiffness,
)
from ..fem.reduce import assemble_scalar, function_min_max
from ..fem.saveable import (
    ParametricSaveable,
    Saveable,
    own_elements,
    qualified,
)
from . import cli
from . import run as running
from .couplings import Coupling
from .dissipative import Dissipative
from .entry import EntryPoint
from .parameters import PhaseFieldSystemParameters
from .phase_field import PhaseField
from .visualize.style import PLOT_PARAMS

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from ufl.core.expr import Expr

    from ..fem.elements import ElementSpec
    from .energy import EnergyDensity
    from .transport import Transported


class PhaseFieldSystem(
    ParametricSaveable[PhaseFieldSystemParameters], Dissipative, ElementOwner, ABC
):
    """Saturated mixture of phase fields sharing a diffuse domain."""

    # Bound by the solver from the ``test_attr`` of the pressure spec.
    _pressure_test: "Expr"

    #: How this mixture's figures look: type sizes, colourbar geometry, panel
    #: resolution, the background. Not in the spec, because none of it is a
    #: statement about the material or the run. A subclass restates only the
    #: keys it changes::
    #:
    #:     plot_params = {**PolymerizingB.plot_params, "dpi": 600.0}
    #:
    #: See :data:`~ovpsolver.phase_field_system.visualize.style.PLOT_PARAMS`.
    plot_params = PLOT_PARAMS

    ## Construction

    def __init__(self, parameters: PhaseFieldSystemParameters) -> None:
        solver_parameters = parameters.solver
        # Before anything is declared: every bulk measure a declaration is
        # written against has to be the rule ``chi_eps`` has values at, and it
        # is the diffuse geometry that decides that rule.
        solver_parameters.integrate_against(
            parameters.solver.diffuse_domain.indicator_quadrature_degree
        )
        domain = DiffuseDomain(solver_parameters, parameters.solver.diffuse_domain)
        super().__init__(solver_parameters, parameters, domain)
        self.phase_fields = tuple(self.make_phase_fields())
        self.couplings = tuple(self.make_couplings())

        declared = tuple(field.name() for field in self.phase_fields)
        if declared != self.parameters.phase_field_names:
            raise ValueError(
                f"{type(self).__name__}.make_phase_fields built {declared}, which "
                f"does not match the roster {self.parameters.phase_field_names}"
            )
        self.check_saved_names()

    ## Entry points

    # Classmethods, because the class settles which parameters classes a
    # document is read against: a spec never names a type, and a subclass
    # inherits all of this without writing any of it.

    @classmethod
    def run(
        cls,
        spec: "str | Path",
        *,
        n_steps: int | None = None,
        overwrite: bool = False,
        resume: bool = False,
    ) -> float:
        """Read ``spec`` against this mixture, build it, and step it.

        Returns the time reached, which is not simply ``n_steps * dt``: a macro
        step that fails its checks is retried shorter, and one that cannot be
        made at all stops the run.

        Refuses an output directory that already holds anything, since a second
        run into one merges with the first rather than replacing it.
        ``overwrite`` clears it instead, and ``resume`` continues the run that
        is there from the end of one of its own macro steps.
        """

        return running.run(
            cls, spec, n_steps=n_steps, overwrite=overwrite, resume=resume
        )

    @classmethod
    def plot(cls, spec: "str | Path | Sequence[str | Path]", **options: Any) -> Any:
        """Read ``spec``, open the run it produced, and draw its saved frames.

        The counterpart of :meth:`run`: the spec says what the mixture is made
        of, so a figure of it needs nothing else specified. What the picture is
        *of* comes from
        :meth:`~ovpsolver.phase_field_system.parameters.PhaseFieldSystemParameters.phase_layers`,
        and what is *available* from the saved run's own metadata.

        Takes the options of ``visualize.phases``, and returns the matplotlib
        figure so that a script can go on decorating it.
        """

        # Inline, to keep matplotlib and pyvista off the solver's import path.
        from .visualize import plot

        return plot(cls, spec, **options)

    @classmethod
    def entry_points(cls) -> "tuple[type[EntryPoint], ...]":
        """The alternatives of this mixture's command line: every registered one.

        An :class:`~ovpsolver.phase_field_system.entry.EntryPoint` subclass
        registers itself, so a new diagnostic appears here by existing. A mixture
        with an alternative that only makes sense for it overrides this and
        appends; one that wants to withhold an alternative filters it out.
        """

        return EntryPoint.all()

    @classmethod
    def main(cls, argv: "list[str] | None" = None) -> Any:
        """Dispatch a command line to whichever entry point it names.

        The whole of what a user's module needs at the bottom of it::

            if __name__ == "__main__":
                MyMixture.main()
        """

        return cli.main(cls, argv)

    ## What the mixture is made of

    @abstractmethod
    def make_phase_fields(self) -> Sequence[PhaseField]:
        """Instantiate one phase field per entry of the parameter roster.

        The only place the kinds of phase are distinguished; everything after
        treats them uniformly, which is what lets the saturation constraint and
        the flux balance be written once. Pass each field :attr:`diffuse_domain`
        and :attr:`k_B_T`, which exist by the time this is called.
        """

    @abstractmethod
    def make_couplings(self) -> Sequence[Coupling]:
        """Instantiate one coupling per block of the ``couplings`` spec section.

        Written out by every mixture, as :meth:`make_phase_fields` is, so that
        where a term of the Rayleighian comes from can be read off the class that
        has it. A mixture with nothing coupling its phases returns an empty
        tuple.

        Called after :meth:`make_phase_fields`, because a coupling is about the
        phases: a cross interaction writes in their differentiation handles and a
        drag reads their velocities. Pass each coupling :attr:`phase_fields` along
        with what a phase is passed.
        """

    @property
    def k_B_T(self) -> float:
        """The mixture's thermal scale, which every energy in it is measured in."""

        return self.parameters.k_B_T

    def name(self) -> str:
        return "phase_field_system"

    def save_qualifier(self) -> str:
        """``system``, whatever the mixture is called.

        What the mixture's own saved fields are filed under, so what anything
        reading a run names them by -- ``system.pressure``. Not :meth:`name`,
        which differs between mixtures and would put the same field at
        ``model_b.pressure`` in one run directory and ``polymerizing_b.pressure``
        in another. ``system`` is the spec block these fields are declared in, as
        ``diffuse_domain.indicator`` and ``water.phi`` are named for theirs.
        """

        return "system"

    def declare_elements(self) -> list["ElementSpec"]:
        """The pressure, and nothing else.

        A multiplier rather than a state: it is conjugate to the volumetric rate
        of the mixture, not to its composition, so it appears in the Legendre
        form of the compression dissipation (:meth:`compression_penalty`) and
        never in ``E``.

        Cell-constant, like the phases whose sum it holds: the rate it is paired
        with is a per-cell balance of facet fluxes, so there is one multiplier
        per cell. That is what makes the degree-2 velocities load-bearing -- P2
        against DG0 is an inf-sup stable pair for this divergence, and P1 against
        DG0 is not.
        """

        return [
            StaticElement("pressure", "_pressure_test", family=DISCONTINUOUS_LAGRANGE, degree=0)
        ]

    def element_owners(self) -> tuple[ElementOwner, ...]:
        """The mixture, then every phase, then every coupling.

        The solver walks this to size the mixed space and to advance the time
        levels.
        """

        return (self, *self.phase_fields, *self.couplings)

    ## What a run writes out

    def savers(self) -> tuple["ParametricSaveable", ...]:
        """Everything of this run that has a ``save`` list.

        The mixture, the geometry it lives on, and its phases, each resolving
        what its spec block names under ``save`` against itself. The diffuse
        domain is one of them because its indicator is a field of the run like
        any other, which an analysis may want to mask by.

        Couplings are not: no block of the spec names their fields, since a
        coupling is built by the mixture rather than declared. What a coupling
        has worth saving, it publishes through a phase.
        """

        return (self, self.diffuse_domain, *self.phase_fields)

    def saved_names(self) -> tuple[str, ...]:
        """What this run will write, by qualified name.

        The names without the values, which is all there is before the solver
        binds the functions the values close over, and all that is wanted when
        comparing what a mixture declares against what a finished directory
        holds.
        """

        return tuple(
            qualified(owner, name)
            for owner in self.savers()
            for name in owner.requested()
        )

    def saved_fields(self) -> dict[str, "tuple[Any, Saveable]"]:
        """Every declared save field of the run, by qualified name.

        Qualified by owner, since two phases may publish diagnostics of the same
        name. Each comes with its owner's
        :class:`~ovpsolver.fem.saveable.Saveable`, so the saver records what the
        mixture declared rather than deciding anything itself.
        """

        return {
            qualified(owner, name): owner.saved(name)
            for owner in self.savers()
            for name in owner.requested()
        }

    def declare_saveable(self) -> dict[str, Saveable]:
        """The pressure, and the saturation residual read off the phases.

        ``total_phase`` is a sum of the cell-constant phases, so it is stored
        exactly in the pressure's cell-constant space.
        """

        return {
            **own_elements(self),
            "total_phase": Saveable(self.element("pressure"), self.total_phase),
        }

    def check_saved_names(self) -> None:
        """Every save name in the spec answers to something, checked at build.

        The names, not the values, which close over functions the solver has
        not built yet; this is the earliest a misspelling can be reported.
        """

        for owner in self.savers():
            owner.check_saveable_names()

    ## Declarations

    def contributors(self) -> tuple["Dissipative", ...]:
        """Everything with a term in ``R``, this mixture's own terms aside.

        The phases and the couplings it built. The mixture is not in its own
        list: it is the sum, and the one term it owns is
        :meth:`compression_penalty`, the pressure's. There is no third category:
        a cross interaction the mixture assembled itself would be a term the
        declaration layer could not see.
        """

        return (*self.phase_fields, *self.couplings)

    def energy_density(self) -> list["EnergyDensity"]:
        """``E``: every contributor's free energy, in one list.

        Handed to every variable that needs a potential, because a variable's
        potential is the derivative of all of it, and a term that does not
        depend on that variable differentiates to zero and drops out.
        """

        terms: list["EnergyDensity"] = []
        for contributor in self.contributors():
            terms += contributor.energy_density()
        return terms

    def energy_rate(self, terms: list["EnergyDensity"]) -> ufl.Form:
        """``dE/dt`` in the live rates, summed over the contributors.

        Nothing of the mixture's own. A coupling's cross terms are declared in
        the *phases'* differentiation handles, so they are released by the phases'
        transport once each has read the whole list; what a coupling adds here
        is only the power it has no declared energy for.
        """

        rate = self.nothing()
        for contributor in self.contributors():
            rate += contributor.energy_rate(terms)
        return rate

    def dissipation(self) -> ufl.Form:
        """``Psi``: the mixture's compression, and each contributor's friction."""

        total = self.compression_penalty()
        for contributor in self.contributors():
            total += contributor.dissipation()
        return total

    def constraints(self) -> ufl.Form:
        """``C``: each contributor's."""

        total = self.nothing()
        for contributor in self.contributors():
            total += contributor.constraints()
        return total

    ## The rate problem

    def stationarity(self, rayleighian: ufl.Form) -> ufl.Form:
        """``dR/dS``: differentiate a Rayleighian against every rate unknown.

        One pass, over this mixture's rates and every phase's and coupling's at
        once: the pressure and the drag couple every phase to every other, so
        differentiating phase by phase would drop exactly the cross terms.
        """

        residual = self.nothing()
        for owner in self.element_owners():
            for spec in owner.element_specs():
                if isinstance(spec, StaticElement):
                    residual += ufl.derivative(
                        rayleighian,
                        getattr(owner.rates, spec.name),
                        getattr(owner, spec.test_attr),
                    )
        return residual

    def rate_residual(self) -> ufl.Form:
        """The nonlinear rate solve.

        Stationarity of the Rayleighian, plus the rows stated outright rather
        than obtained by differentiating it: the auxiliary unknowns the phases
        define their potentials in, and the transport of every live variable.

        The rule for which rows belong here: a ``k+1`` value is live if the
        energy is evaluated at it, and a live value's row has to be solved with
        the rates because the rates cannot be found without it. Everything else
        the energy sees only lagged, so once the rates are known each such row is
        linear in its own unknown and independent of the rest, and the variable
        solves itself afterwards
        (:meth:`~ovpsolver.phase_field_system.transport.Transported.solve`).
        """

        residual = self.stationarity(self.rayleighian())
        for field in self.phase_fields:
            residual += field.aux_residual()
        for variable in self.transported():
            if variable.energy_live:
                residual += variable.transport_residual()
        return residual

    def bregman_deficit(self) -> ufl.Form:
        """``int D_ccv dx_eps``: the dissipation the *step* adds.

        The companion the excess Rayleighian needs to be read correctly. That
        one goes to zero when Newton converges, the convex-split Rayleighian being
        what the solve is stationary for; this is how much more energy the split
        increment gives up than the flow it approximates would, which no amount
        of solving reduces and only a shorter step does.

        Summed over the declared terms, each in the measure it was declared on,
        so a mixture whose energy is entirely convex reports zero.
        """

        lag_to_live = self.lag_to_live()
        domain = self.diffuse_domain
        dx = self.solver_parameters.get_mesh_dx()
        measures = {
            ElementDomain.BULK: domain.chi_eps,
            ElementDomain.DIFFUSE_SURFACE: domain.dgamma_eps,
        }

        total = self.nothing()
        for term in self.energy_density():
            density = term.bregman_divergence(lag_to_live)
            if isinstance(density, Zero):
                continue
            total += density * measures[term.domain] * dx
        return total

    ## Pressure

    def total_phase(self, *, time_level: str = "next") -> "Expr":
        """``sum_i phi_i``, which saturation says is one.

        The residual of that constraint, and worth writing out: its correct
        value is known in advance, so any departure from it reads directly as
        error.
        """

        total = None
        for field in self.phase_fields:
            phi = field.fields_at(time_level).phi
            total = phi if total is None else total + phi
        return total

    def compression_penalty(self) -> ufl.Form:
        """The pressure's share of ``Psi``: compressible motion against a viscosity.

        The pressure is paired with the volumetric rate
        ``f = -sum_i div(chi_eps J_i)``, ``C_p = <p, f>``, and relaxed by the
        concave ``Psi_p^*(p) = (1/2) int M_p p^2`` with ``M_p = Phi / eta_b`` and
        ``Phi = chi_eps sum_i phi_i``. Their difference is the Legendre transform
        of ``Psi_p(f) = int f^2 / 2 M_p``, the power of volumetric change, and is
        worth it wherever ``p`` is stationary -- a dissipation that takes a
        multiplier, like the membrane's, hence part of :meth:`dissipation`.
        Saturation is therefore held up to the compression rate
        ``f = Phi p / eta_b``, and a large ``dilatational_viscosity`` is a nearly
        incompressible mixture.

        ``f`` is written on the same upwind facet fluxes, weighted by the same
        facet indicator, as the phases' transport and energy rows. Each facet flux
        is one number entering one cell's balance with each sign, so the rate the
        pressure is paired with is exactly the rate a cell's phase sum changes at,
        which the continuous divergence would only approximate. Read off the flux
        pieces rather than their sum, because the upwind selector is per piece
        and a summed flux has thrown it away.

        The injections are not in ``f``: what the spec puts in at the inclusion
        surfaces and on ``Sigma`` is volume-neutral over the phases, and each
        phase's transport row carries its own share.

        ``Phi`` is lagged, as every mobility in the mixture is, and uses the
        floored indicator: the floor is what keeps ``M_p`` positive inside an
        inclusion, where there is no free fluid and nothing else determines
        ``p``.
        """

        viscosity = self.parameters.dilatational_viscosity

        # The triples the phases' own rows are built from, so this pairs against
        # the facet numbers those rows apply: there is no selector, carrier or
        # weight to state here, hence none to state differently.
        pieces: list[tuple] = []
        for field in self.phase_fields:
            pieces += field.transported_phase().upwind()
        pressure = self.rates.pressure
        mobility = (
            self.diffuse_domain.chi_eps
            * self.total_phase(time_level="prev")
            / viscosity
        )
        return (
            -dg0_upwind_ibp(self.solver_parameters, pressure, pieces)
            - 0.5 * mobility * pressure**2 * self.solver_parameters.get_mesh_dx()
        )

    def transported(self) -> list["Transported"]:
        """Every variable the mixture advects: each phase's, then each coupling's.

        A coupling's own fields are carried the same way a phase's internal
        states are, so they go through the same rows, bounds and checks.
        """

        return [
            variable
            for owner in (*self.phase_fields, *self.couplings)
            for variable in owner.transported()
        ]

    ## The irreversible half-step

    def irreversible_timestep(self, dt: float) -> None:
        """Advance everything that is not the Onsager problem, over ``dt``.

        The other half of the Lie split. The rate step descends the free energy
        by construction; this one may do anything to it, which is why the two are
        split rather than solved together. Empty here, since a mixture need not
        have such a part; a concrete system that does sub-cycles it at its own
        stability limit, which is why it is given the whole step.
        """

    ## Initial conditions

    def set_initial_conditions(self) -> None:
        """The wetting equilibrium, roughened, for the phases; then the couplings."""

        self.rates.pressure.x.array[:] = 0.0
        initial = self.add_phase_noise(self.wetting_equilibrium())
        for field, phi in zip(self.phase_fields, initial, strict=True):
            field.set_initial_conditions(phi)
        for coupling in self.couplings:
            coupling.set_initial_conditions()

    def add_phase_noise(
        self, initial: tuple[fem.Function, ...]
    ) -> tuple[fem.Function, ...]:
        """Roughen the initial composition, and put it back on the simplex.

        A smooth state on the unstable side of the spinodal decomposes on
        whatever asymmetries the discretization happens to have, which is a slow
        and mesh-dependent way to start. So a run that means to demix seeds the
        instability: independent Gaussian noise in every degree of freedom,
        clipped at zero because a negative volume fraction is not a composition,
        then rescaled by the pointwise sum so the mixture is saturated again.

        Rescaling rather than subtracting the mean, because saturation is a
        pointwise constraint and division restores it exactly everywhere. It
        biases each phase towards its neighbours' noise, which is harmless: what
        matters is that the state is rough, not that its roughness has a
        particular spectrum.
        """

        sigma = self.parameters.initial_phase_noise_sigma
        if sigma == 0.0:
            return initial

        generator = self.parameters.rng
        total = None
        for phi in initial:
            values = phi.x.array
            values += generator.normal(0.0, sigma, size=values.shape)
            np.clip(values, 0.0, None, out=values)
            total = values.copy() if total is None else total + values
        if np.any(total <= 0.0):
            raise ValueError(
                f"initial_phase_noise_sigma={sigma:.6g} left every phase at zero "
                f"somewhere, so there is nothing to renormalize; the noise is "
                f"comparable to the composition itself"
            )
        for phi in initial:
            phi.x.array[:] /= total
            phi.x.scatter_forward()
        self.report_initial_conditions(initial, "perturbed and renormalized")
        return initial

    def wetting_equilibrium(self) -> tuple[fem.Function, ...]:
        """Solve for the composition at which the run's own driving force vanishes.

        The step should start from a state that already satisfies the constraint
        the pressure enforces and already wets the inclusions at the declared
        contact angle, rather than from a discontinuity the first step has to
        resolve. Away from the diffuse surface the declared volume fractions are
        soft far-field targets rather than integral constraints, which keeps the
        problem local and linear.

        It has to be the *run's* equilibrium. The stationarity of a discrete
        energy is an equilibrium for whatever operator that discretisation
        defines, and if the rate step uses a different one, the zeroth step opens
        on the difference between two discretisations of the same force -- a
        term living in the diffuse band that depends on how the facets there lie,
        and so has a preferred direction on a mesh with any lattice coherence.

        So this solves in the run's own unknowns: ``aux_potential`` defined term
        for term as :meth:`PhaseField.aux_residual` defines it, with the
        contact-angle term in the operator the run uses
        (:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.contact_angle`), and the
        balance

            chi_eps aux_potential + off_surface (phi - phi_0) + pressure = 0

        which is the discrete force per unit volume the rate step would see with
        the bulk energy switched off. The solve is direct, so that combination
        vanishes to the factorisation's accuracy and the zeroth step is driven by
        the bulk energy alone. Not the ``chi_eps``-weighted two-point stiffness,
        which places the indicator on the facet where the potential places it on
        the cell; the two differ by about a third across the band, where
        ``chi_eps`` turns over within a cell or two.

        The contact-angle defect ``C`` does not vanish on its own, and should
        not: the wetting condition holds weakly, so the defect is balanced against
        the Korteweg term, exactly as during the run. This is the equilibrium of
        the dynamics, not the minimum of an energy -- the operator is the
        symmetric ``chi_eps``-weighted Korteweg operator composed with a cell-wise
        scaling -- and it is the first of those the zeroth step cares about.
        """

        if len(self.phase_fields) == 1:
            # One phase in a saturated mixture is the mixture; no solve needed.
            field = self.phase_fields[0]
            phi = fem.Function(
                field.next.phi.function_space, name=f"{field.name()}_phi_initial"
            )
            phi.x.array[:] = 1.0
            phi.x.scatter_forward()
            self.report_initial_conditions([phi], "saturated by the only phase")
            return (phi,)

        mesh = self.solver_parameters.dolfinx_mesh
        dx = self.solver_parameters.get_mesh_dx()
        dS = self.solver_parameters.get_mesh_dS()
        centroids = self.solver_parameters.cell_centroids()
        count = len(self.phase_fields)
        dg0 = element(DISCONTINUOUS_LAGRANGE, mesh.basix_cell(), 0)
        space = fem.functionspace(mesh, mixed_element([dg0, dg0] * count + [dg0]))
        trials = ufl.TrialFunctions(space)
        tests = ufl.TestFunctions(space)
        pressure_trial, pressure_test = trials[-1], tests[-1]

        chi_eps = self.diffuse_domain.chi_eps_analytic_ufl()
        dgamma_over_chi = self.diffuse_domain.dgamma_eps_over_chi_eps_analytic_ufl()
        off_surface = self.diffuse_domain.off_surface_weight_ufl()

        bilinear = ufl.as_ufl(0.0) * dx
        linear = ufl.as_ufl(0.0) * dx
        for index, field in enumerate(self.phase_fields):
            phi, potential = trials[2 * index : 2 * index + 2]
            test, potential_test = tests[2 * index : 2 * index + 2]
            kappa = field.parameters.kappa

            # The force balance, weighted as the rate step weights it.
            bilinear += (
                chi_eps * potential * test
                + off_surface * phi * test
                + pressure_trial * test
            ) * dx
            linear += (
                off_surface * float(field.parameters.initial_volume_fraction) * test
            ) * dx

            # aux_potential = -kappa laplace(phi) + C / chi_eps, term for term as
            # PhaseField.aux_residual writes it, with the Korteweg half of C in
            # the operator the run itself uses.
            bilinear += (
                potential * potential_test * dx
                - kappa * self.diffuse_domain.contact_angle(phi, potential_test)
                - kappa * two_point_stiffness(1.0, phi, potential_test, centroids, dS)
            )
            linear += (
                self._wetting_force(field) * dgamma_over_chi * potential_test
            ) * dx

        bilinear += sum(trials[0 : 2 * count : 2]) * pressure_test * dx
        linear += pressure_test * dx

        solution = fem.Function(space, name="wetting_equilibrium")
        # Factored directly, once, and on nothing the spec says about the rate
        # solve: this is a coupled system of its own, not a block of that one.
        problem = fem_petsc.LinearProblem(
            bilinear,
            linear,
            u=solution,
            petsc_options_prefix="phase_field_initial_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        problem.solve()

        initial = []
        tolerance = 1.0e-12
        for index, field in enumerate(self.phase_fields):
            phi = fem.Function(
                field.next.phi.function_space, name=f"{field.name()}_phi_initial"
            )
            phi.interpolate(solution.sub(2 * index).collapse())
            phi.x.scatter_forward()
            low, high = function_min_max(phi)
            if low < -tolerance or high > 1.0 + tolerance:
                raise ValueError(
                    f"the wetting equilibrium for {field.name()} left the "
                    f"simplex: range=[{low:.6g}, {high:.6g}]"
                )
            initial.append(phi)
        self.report_initial_conditions(initial, "solved wetting equilibrium")
        return tuple(initial)

    def report_initial_conditions(
        self, initial: Sequence[fem.Function], how: str
    ) -> None:
        """Log each phase's mean and range, and ``how`` the state was reached."""

        dx = self.solver_parameters.get_mesh_dx()
        volume = assemble_scalar(1.0 * dx)
        pieces = []
        for field, phi in zip(self.phase_fields, initial, strict=True):
            mean = assemble_scalar(phi * dx) / volume
            low, high = function_min_max(phi)
            pieces.append(
                f"{field.name()}: mean={mean:.6g}, range=[{low:.6g}, {high:.6g}]"
            )
        logger.info("initial %s; %s", how, "; ".join(pieces))

    ## Private helpers

    def _wetting_force(self, field: PhaseField) -> float:
        """The constant ``dE_surface/dphi`` the initial-condition problem needs.

        Taken from the field's declared surface energy rather than its
        parameters, so a field that declares no wetting term contributes none.
        A nonlinear affinity would make the problem nonlinear and is rejected
        instead of silently linearized about an uninitialized phase.
        """

        potential = field.surface_potential()
        if ufl.algorithms.extract_coefficients(potential):
            raise NotImplementedError(
                f"{field.name()} declares a surface energy that is nonlinear in "
                "phi; the wetting equilibrium is a linear problem and has no "
                "state to linearize about"
            )
        return potential
