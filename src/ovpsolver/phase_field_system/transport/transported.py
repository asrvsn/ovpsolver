"""What an element owner advects, and the rows that move it.

:class:`Transported` is one advected variable: a phase's own ``phi``, or any of
the internal states something carries -- a crosslink concentration, the moments
of a degree-of-polymerization distribution, the elastic state of a gel network,
the locking modulus between two of them. One type for all of them lets the rows
be a loop rather than a block per variable.

A variable is declared once, as a :class:`~ovpsolver.fem.elements.StateElement`,
whose spec already knows its name, its test function, its degree, and --
through :class:`~ovpsolver.fem.elements.Solve` -- whether the free energy sees
its ``k+1`` value. :class:`Transported` holds the spec and adds what the element
cannot know: which velocities carry the variable, and whether it is a density of
the free fluid or an intrinsic property of a phase.

The transport row
-----------------
Every variable has one, stating its conservation law: by default the law its
flux list states. A variable whose evolution is not of that form overrides
:meth:`Transported.transport_residual` -- an objective rate of a tensorial state
has no useful closure to declare. An override owes the shape of every state
update of the scheme: explicit in the state, so its own next value appears in
the accumulation and nowhere else, and some route by which the work it does
against the energy reaches the velocity rows. Couplings across components or
cells are free as long as they read the lagged value. A live variable's row is
part of the rate problem and must also be affine in the rates, or that problem
stops being convex; a lagging one is solved once the rates are numbers and need
not be (the Lie stretching is quadratic in the velocity).

A lagging variable is solved by its own :meth:`Transported.solve`, cell by cell
and with no matrix, which is where what its row owes is checked.
:func:`transport_step` collects them and checks what no single row can: that
none reads another's next value.

Every variable is cell-constant, so the row has one rule: the cell-wise
divergence theorem against a single-valued upwind facet flux. A cell-constant
variable under an upwind flux is monotone under a step bound, which a nodal one
is not under any; a nodal variable is refused
(:meth:`Transported._require_cell_constant`).

The energy row
--------------
What moving the variable does to the free energy. Only
:attr:`~ovpsolver.fem.elements.Solve.ENERGY_LIVE` variables have one, and they
are exactly the ones the rate solve carries. It uses the same facet rule, a
potential of cell-constant states being cell-constant too, which
:func:`~ovpsolver.phase_field_system.energy.require_facet_potential` checks.

Both rows pair against the same facet numbers, and so does the pressure
(:meth:`~ovpsolver.phase_field_system.PhaseFieldSystem.compression_penalty`), so
the flux that leaves one cell enters its neighbour by construction and the phase
sum in each cell changes by exactly the compression the dilatational viscosity
permits, whatever each piece's selector chose.

The rows differ at ``Sigma``. The transport row *substitutes* the imposed normal
flux, so the influx it accumulates is exactly the spec's number. The energy row
*integrates* the live one, because the velocity row has to feel the energy that
influx releases and a prescribed number is not differentiable in the velocity.
The outer boundary constraint makes the two agree wherever it is satisfied.

Positivity is the part that cannot be declared. The facet rule preserves it only
under a step bound, which depends on the solved velocities:
:meth:`Transported.cfl_limit` is assembled *after* the rate solve and tells the
driver whether the step it just took was admissible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import fem
from ufl.algorithms import expand_derivatives

from ...fem.elements import ElementDomain, Solve, StateElement
from ...fem.elements.dg0 import (
    DISCONTINUOUS_LAGRANGE,
    dg0_emptying_times,
    dg0_outflow,
    dg0_upwind_ibp,
    positive_part,
)
from ...fem.reduce import (
    assemble_nodal_values,
    global_all,
    global_max,
    global_min,
    owned_dofs,
)
from ...fem.save.element import interpolation_expression
from ...solver.utils import SolveDiverged
from ..energy import require_facet_potential

if TYPE_CHECKING:
    from ufl.core.expr import Expr

    from ...diffuse_domain import DiffuseDomain
    from ...fem.elements import ElementOwner
    from ...solver.parameters import SolverParameters
    from ..energy import EnergyDensity
    from .flux import Flux


def injected(pieces: Iterable["Flux"], *, boundary: bool = False) -> "Expr":
    """Everything the pieces put into the variable, on one of the two surfaces.

    A flag rather than a sum, because the two are integrated differently: the
    inclusion injection is a flux per unit volume and belongs under ``dx``;
    ``boundary=True`` gives the flux per unit area through ``Sigma``, which
    belongs under ``ds``.

    A piece that injects nothing says so with a scalar zero whatever the shape of
    the variable, so absent terms are dropped rather than added: only the sol
    carries influx, and the variable it feeds may be a vector of gelation bins.
    """

    total = None
    for piece in pieces:
        term = ufl.as_ufl(
            piece.boundary_flux if boundary else piece.surface_flux
        )
        if _is_zero(term):
            continue
        total = term if total is None else total + term
    return ufl.as_ufl(0.0) if total is None else total


@dataclass(frozen=True)
class Transported:
    """A variable advected by the velocities of the owner that declares it.

    Parameters
    ----------
    element : the variable's own declaration, through which everything else is
        reached. Its :class:`~ovpsolver.fem.elements.Solve` says whether the
        energy sees the ``k+1`` value, and so whether the rate solve carries it;
        its owner -- usually a phase field, sometimes a coupling -- supplies the
        two time levels, the test function, the mesh and the diffuse domain.
    flux : the pieces of ``sum_alpha v_alpha rho_alpha``, with their injection.
        Required of a live variable, whose energy row pairs each piece against
        the potential. A lagging variable may leave it empty and override
        :meth:`transport_residual` instead.
    upwind_live : whether the facet rule reads the *solved* velocity's sign to
        pick the upwind cell, rather than the lagged one. Only for an
        ``ENERGY_LAGGING`` variable, whose row is assembled after the rate solve,
        when the sign is a number and the row still linear in its own unknown; a
        live variable's rate residual would stop being affine in the rate.

        What it buys is the step bound. A lagged selector that disagrees with the
        live velocity takes the density from the *downstream* cell, which is
        monotone at no step at all, and the bound shortens the step to match. A
        self-carrying field read genuinely upwind has ``rho_up = rho_K`` on every
        outflow facet, so the density cancels against the cell's own content and
        the bound is the kinematic ``|K| / sum_F (v.n)^+``.

        Off by default: a lagging variable that live variables are *functions of*
        has to share their selector, or the relation does not survive transport.
        The generating function is the case in point -- its moments are live,
        ``u_bar = u . widths`` exactly, and that holds after a step only because
        advection is linear in the density and both read the same sign.
    chi_weighted : whether the variable is a density of the free fluid, whose
        accumulation and flux carry ``chi_eps``, or an intrinsic property of a
        phase, which does not. A gel's elastic moments are already resolved in
        ``chi_eps``-weighted form and so set this false.
    """

    element: StateElement
    flux: Sequence["Flux"] = field(default=())
    upwind_live: bool = False
    chi_weighted: bool = True
    _assembled: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.element.solve is Solve.NONE:
            raise ValueError(
                f"{self.name!r} is transported, so some solve has to determine "
                "its next value; declare its element ENERGY_LIVE or ENERGY_LAGGING"
            )
        if self.energy_live and not self.flux:
            raise ValueError(
                f"{self.name!r} is ENERGY_LIVE, so its energy row pairs the "
                "potential against each velocity's density and it must declare "
                "flux pieces; a variable with a law of its own has to be lagging"
            )
        states_its_own_law = (
            type(self).transport_residual is not Transported.transport_residual
        )
        if not self.flux and not states_its_own_law:
            raise ValueError(
                f"{self.name!r} declares no flux pieces and does not override "
                "transport_residual, so nothing determines its next value"
            )
        if self.upwind_live and self.energy_live:
            raise ValueError(
                f"{self.name!r} is ENERGY_LIVE and asks for a live upwind "
                "selector, which would put sign(v_{k+1}.n) in the rate residual "
                "and cost it the affineness in the rate the convex split needs. "
                "A live selector is only available to a row assembled after the "
                "rate solve; declare the element ENERGY_LAGGING or leave the "
                "selector lagged"
            )

    ## Everything the element and the owner already know

    @property
    def owner(self) -> "ElementOwner":
        return self.element.owner

    @property
    def name(self) -> str:
        return self.element.name

    @property
    def label(self) -> str:
        """Qualified by owner, since two phases declare a ``phi`` each."""

        return self.element.label

    @property
    def energy_live(self) -> bool:
        """Whether the energy sees the ``k+1`` value, so the rate solve carries it."""

        return self.element.solve is Solve.ENERGY_LIVE

    @property
    def cell_constant(self) -> bool:
        """Whether the variable's own row admits the cell-wise facet rule.

        True of everything the mixture declares, and checked rather than assumed
        because an element is a declaration a subclass writes.
        """

        return self.element.degree == 0

    @property
    def next_value(self) -> "Expr":
        return self.element.function("next")

    @property
    def prev_value(self) -> "Expr":
        return self.element.function("prev")

    @property
    def next_handle(self) -> "Expr":
        """The wrapper an energy term is differentiated against, at ``k+1``."""

        return self.owner.variable(self.name, time_level="next")

    @property
    def prev_handle(self) -> "Expr":
        return self.owner.variable(self.name, time_level="prev")

    @property
    def test(self) -> "Expr":
        return self.element.test

    @property
    def solver_parameters(self) -> "SolverParameters":
        return self.owner.solver_parameters

    @property
    def domain(self) -> "DiffuseDomain":
        return self.owner.diffuse_domain

    ## Rows

    def upwind(self, *, boundary: bool = False) -> list[tuple["Expr", "Expr", "Expr"]]:
        """This variable's flux as ``(selector, carrier, density)`` facet triples.

        Every row that reads this variable across a facet reads it from here. The
        transport row, the energy rate and the mixture's saturation constraint
        are separate residuals against separate test functions, and the scheme's
        exactness rests on all three pairing against the *same* facet number. A
        caller that could choose the triple could choose differently in one of
        them, which is easy to introduce and hard to see.

        So none of the three is a parameter. The carrier is always the live
        velocity, since a lagged one would make the flux explicit and cost the
        step its no-motion property. The weight is the variable's own: the
        analytic indicator for a density of the free fluid, nothing for an
        intrinsic property of a phase. The selector is :attr:`upwind_live`,
        decided at the declaration.

        ``boundary`` drops the weight, for the one term that lives on ``Sigma``:
        ``chi_eps`` is approximately one there, the inclusions being interior,
        and taking it as one keeps a quadrature-element coefficient, which has
        no facet value, off ``ds``.
        """

        weight = 1.0 if boundary else self._chi(self.domain.chi_eps_analytic_ufl())
        return [
            (
                # Only the sign is read, so this chooses which cell's density
                # crosses the facet and nothing else.
                piece.live_velocity if self.upwind_live else piece.lagged_velocity,
                piece.live_velocity,
                weight * piece.lagged_density,
            )
            for piece in self.flux
        ]

    def transport_residual(self) -> ufl.Form:
        """``F(chi rho) + div(chi sum_a v_a rho_a) = sum_a s_a`` over one step.

        Accumulation implicit, flux explicit in the density and live in the
        velocity: that combination makes the row a positivity-preserving
        transport under a step bound rather than an unconditionally stable but
        sign-indefinite one. The divergence is the cell-wise theorem against a
        single-valued upwind facet flux.

        The flux through ``Sigma`` is *substituted*, not integrated. Each
        velocity's normal flux there is pinned to its imposed value by the outer
        boundary constraint, so the row carries the injection directly and would
        count it twice if it also integrated the flux.

        Override to state a law this does not cover. That is the one place in the
        declaration layer where a residual is written by hand.
        """

        solver_parameters = self.solver_parameters
        domain = self.domain
        dt = solver_parameters.dt_live
        dx = solver_parameters.get_mesh_dx()
        ds = solver_parameters.get_mesh_ds()

        self._require_cell_constant()
        weight = self._chi(domain.chi_eps)
        residual = (
            ufl.inner(
                weight * self.next_value - weight * self.prev_value, self.test
            )
            * dx
            # The accumulation weight is the quadrature field, since the cell term
            # can have it; the facet weight is the same indicator as an expression,
            # since a facet term cannot. Two evaluations of one analytic function,
            # differing by a quadrature rule rather than by an interpolation.
            + dt
            * dg0_upwind_ibp(
                solver_parameters,
                self.test,
                self.upwind(),
            )
        )
        # A variable with no injection declares a scalar zero whatever its shape,
        # so pair only what is actually there.
        source = injected(self.flux)
        if not _is_zero(source):
            residual -= dt * ufl.inner(source, self.test) * dx
        boundary_flux = injected(self.flux, boundary=True)
        if not _is_zero(boundary_flux):
            residual -= dt * ufl.inner(boundary_flux, self.test) * ds
        return residual

    def energy_rate(self, terms: Iterable["EnergyDensity"]) -> ufl.Form:
        """``-int mu div J`` for this variable, in the live velocities.

        A term of the Rayleighian, not a row: it is linear in every velocity it
        carries, so differentiating it puts this variable's share of the release
        into each of those velocities' rows without naming them.

        Unlike :meth:`transport_residual` this *integrates* the flux through
        ``Sigma``, because only the integrated form is differentiable in the
        velocity and the velocity row has to feel the energy an influx releases.
        Each velocity's multiplier on ``Sigma`` pins the two together at the
        solution.

        Nothing for a lagging variable, which is refused if the free energy
        depends on its next value.
        """

        solver_parameters = self.solver_parameters
        domain = self.domain
        rate = ufl.as_ufl(0.0) * solver_parameters.get_mesh_dx()
        potential = potential_of(terms, self, domain=ElementDomain.BULK)
        if not self.energy_live:
            if not _is_zero(potential):
                raise ValueError(
                    f"{self.name!r} is ENERGY_LAGGING but the free energy depends "
                    "on its next value, so the rate solve would be stationary for "
                    "an energy the transport step only delivers afterwards; declare "
                    "the element live or write the term at the lagged level"
                )
            return rate

        require_facet_potential(potential)
        return rate - dg0_upwind_ibp(
            solver_parameters,
            potential,
            self.upwind(),
            boundary_flux_pieces=self.upwind(boundary=True),
        )

    ## The transport step

    def solve(self) -> None:
        """Take this lagging variable across the step whose rates were just solved.

        Exactly, in one update. With the rates known, the row, explicit in the
        state, has only this variable's next value left to find, and it enters
        through the accumulation alone. So the row is affine in it, its
        derivative is a mass, and for a cell-constant variable that mass is
        diagonal -- each component in each cell against its own test function,
        the coupling across the gelation coordinate belonging to the irreversible
        step. The update

            x <- x - F(x) / diag(M)

        therefore lands on the solution from any ``x``, with ``F`` the row at the
        current value and ``diag(M)`` the mass's action on ones. No matrix is
        built: assembled, the mass of a blocked element stores every component of
        a cell against every other -- 4096 entries a cell for the sixty-four-bin
        generating function, of which 64 are nonzero.

        A mass that vanishes in some cell leaves the value there undetermined,
        which is a declaration or a geometry rather than a step size, so it stops
        the run. A non-finite value is a failed solve, retried smaller like any
        other.
        """

        residual, mass, _ = self.transport_forms()
        comm = self.solver_parameters.dolfinx_mesh.comm
        function = self.element.function("next")
        owned = owned_dofs(function.function_space)
        diagonal = assemble_nodal_values(mass)[:owned]
        if not global_all(comm, np.all(np.isfinite(diagonal) & (diagonal != 0.0))):
            raise RuntimeError(
                f"the accumulation of {self.label!r} vanishes or is not finite in "
                "some cell, so its row does not determine its next value there"
            )
        values = function.x.array
        values[:owned] -= assemble_nodal_values(residual)[:owned] / diagonal
        if not global_all(comm, np.all(np.isfinite(values[:owned]))):
            raise SolveDiverged(
                f"the transport step of {self.label!r} produced a non-finite value",
                cause="transport",
            )
        function.x.scatter_forward()

    def transport_forms(self) -> tuple[fem.Form, fem.Form, fem.Function]:
        """The row, and its mass's action on ones, compiled once and checked.

        What :meth:`solve` assembles, and where its exactness is established,
        since a row that broke it would be solved wrongly rather than slowly:

        * the variable is lagging; a live one is solved with the rates;
        * the row is affine in its own next value, which the second derivative
          vanishing says symbolically;
        * that value enters through a diagonal mass and nowhere else, which the
          mass's action on a random vector matching its diagonal times that
          vector says to roundoff -- of the largest entry, not of each, since the
          indicator's floor can put entries many decades below the rest.

        These rule out a state whose own law is implicit: a diffusion of it at
        ``k+1``, a reaction among its components inside this step rather than in
        the irreversible one, a consistent mass on a continuous element.

        The third entry is the vector of ones the mass form reads, which has to
        live as long as the form does.
        """

        cached = self._assembled.get("transport_forms")
        if cached is not None:
            return cached
        if self.energy_live:
            raise ValueError(
                f"{self.label!r} is ENERGY_LIVE, so the rate problem solves its row "
                "together with the rates; only a lagging variable is solved by the "
                "transport step"
            )

        row = self.transport_residual()
        unknown = self.element.function("next")
        mass = ufl.derivative(row, unknown)
        if not expand_derivatives(ufl.derivative(mass, unknown)).empty():
            raise ValueError(
                f"the transport row of {self.label!r} is not affine in its own next "
                "value, so no single update solves it"
            )
        space = unknown.function_space
        ones = fem.Function(space)
        ones.x.array[:] = 1.0
        forms = (fem.form(row), fem.form(ufl.action(mass, ones)), ones)

        probe = fem.Function(space)
        probe.x.array[:] = np.random.default_rng(0).uniform(
            1.0, 2.0, probe.x.array.size
        )
        probe.x.scatter_forward()
        owned = owned_dofs(space)
        applied = assemble_nodal_values(fem.form(ufl.action(mass, probe)))[:owned]
        expected = assemble_nodal_values(forms[1])[:owned] * probe.x.array[:owned]
        comm = self.solver_parameters.dolfinx_mesh.comm
        error = global_max(comm, np.abs(applied - expected).max(initial=0.0))
        if error > 1.0e-10 * global_max(comm, np.abs(expected).max(initial=0.0)):
            raise ValueError(
                f"the transport row of {self.label!r} is not its accumulation "
                "against a lagged flux: its own next value enters the row somewhere "
                "other than a diagonal mass, so the transport step cannot solve it "
                "cell by cell"
            )
        self._assembled["transport_forms"] = forms
        return forms

    ## Step bound

    def cfl_limit(self, cfl: float) -> float:
        """Largest step this variable's row stays non-negative over.

        Assembled from the *solved* velocities, so it is a verdict on the step
        just taken rather than a prediction: the driver compares it against the
        step it used and retries smaller if it fell short. The velocity that
        advects is an unknown of the very solve the step size bounds, so there
        is no other order.

        The upwind CFL condition, made fully sufficient. The row is explicit in
        the density and implicit only in the velocity, so a cell's next value is
        its content less what the facets take,

            rho^{k+1}_K = rho^k_K - (dt / int_K chi) sum_F Phi_F,

        and bounding ``dt`` by the content over ``sum_F max(Phi_F, 0)``, with the
        actual ``Phi_F`` from the helper the row uses, makes that non-negative
        with nothing else assumed. The neighbour-independent form of the
        condition takes each facet's *own* cell density instead, which is only
        partially sufficient: it leaves the off-diagonals to the lagged selector
        agreeing in sign with the live velocity. After the solve every
        neighbour's density is a number in hand, and the two bounds agree exactly
        on every facet where the selector was right.

        A *negative* source drains a cell without any flux doing it, so its
        positive part joins the outflow; an influx needs no bound. Content and
        outflow are assembled per component, so a vector variable -- the binned
        generating function -- is bounded component by component, which is what
        its positivity means. Out of reach are a quantity defined as a difference
        of two rows and a tensor whose positivity is its spectrum; both are
        checked after the fact instead (:meth:`floor_value`).

        ``inf`` for anything with no entrywise positivity to lose.
        """

        self._require_cell_constant()
        if not self.entrywise_positive:
            return float("inf")
        content, outflow = self._dg0_cfl_forms()
        owned = owned_dofs(self._dg0_space())
        held = assemble_nodal_values(content)[:owned]
        lost = assemble_nodal_values(outflow)[:owned]
        times = dg0_emptying_times(held, lost)
        local = float(times.min()) if times.size else float("inf")
        return cfl * global_min(self.solver_parameters.dolfinx_mesh.comm, local)

    @property
    def positive(self) -> bool:
        """Whether this variable claims to be non-negative, per its element."""

        return self.element.positive

    @property
    def entrywise_positive(self) -> bool:
        """Whether non-negativity is a statement about the stored numbers.

        For a scalar or a vector it is, and the upwind outflow bound controls it
        directly. For a square tensor positivity is the semidefinite cone, which
        a bound on each entry's transport says nothing about: a rotation changes
        every entry of a PSD matrix and none of its spectrum. So a tensor is
        checked as positive but imposes no step of its own.
        """

        return self.positive and len(self.element.shape) < 2

    ## Step verdict

    def floor_value(self) -> float:
        """The smallest value this variable takes, as its positivity means it.

        Entry by entry for a scalar or a vector, and eigenvalue by eigenvalue for
        a square tensor: a strain moment can hold a negative off-diagonal entry
        and still be positive semidefinite, and can hold none while having left
        the cone. Like :meth:`cfl_limit`, a verdict on the state the step
        produced, which the driver decides whether to keep.
        """

        values = self.element.function("next").x.array
        owned = owned_dofs(self.element.function("next").function_space)
        if not owned:
            local = float("inf")
        elif self._is_square_tensor:
            local = float(np.min(self._eigenvalues(values, owned)))
        else:
            local = float(np.min(values[:owned]))
        return global_min(self.solver_parameters.dolfinx_mesh.comm, local)

    ## Private helpers

    def _require_cell_constant(self) -> None:
        """Refuse a nodal variable: both rules here are cell-constant ones.

        A nodal phase would need a lumped accumulation against a Galerkin flux,
        whose off-diagonals are of either sign, so no step bound makes it
        monotone and the positivity the solver checks for would have nothing
        behind it.
        """

        if not self.cell_constant:
            raise ValueError(
                f"{self.label!r} is transported but is not cell-constant. Every "
                "state of the mixture is DG0: the transport row is the cell-wise "
                "divergence theorem with an upwind facet flux, and its step "
                "bound is the outflow bound that rule makes exact. Declare the "
                "element family DG0 with degree 0, or write the row by hand by "
                "overriding transport_residual"
            )

    @property
    def _is_square_tensor(self) -> bool:
        shape = self.element.shape
        if len(shape) < 2:
            return False
        if len(shape) > 2 or shape[0] != shape[1]:
            raise NotImplementedError(
                f"{self.name!r} is declared positive with shape {shape}, and "
                f"non-negativity of anything but a scalar, a vector or a square "
                f"tensor is not defined here"
            )
        return True

    def _eigenvalues(self, values: np.ndarray, owned: int) -> np.ndarray:
        """The spectrum in every cell, from the stored components.

        Symmetrized first: the field is symmetric only by construction, a sum of
        outer products carried by a velocity gradient, and nothing in the
        discrete step enforces the symmetry the continuous equation has.
        ``eigvalsh`` reads one triangle and would otherwise answer for a tensor
        nobody computed. A batched two- or three-by-three decomposition over the
        cells, cheap against a rate solve.
        """

        if self.element.symmetry:
            # Symmetric storage keeps one triangle, so the stored numbers are
            # not the matrix. Let dolfinx write it out rather than guessing at
            # the packing order, which is basix's business and not ours.
            dense = self._dense_tensor(self.element.function("next"))
            values = dense.x.array
            owned = owned_dofs(dense.function_space)
        shape = self.element.shape
        matrices = np.asarray(values[:owned], dtype=float).reshape(-1, *shape)
        return np.linalg.eigvalsh(0.5 * (matrices + np.swapaxes(matrices, -1, -2)))

    def _dense_tensor(self, function: fem.Function) -> fem.Function:
        """``function`` with its symmetry written out, into :meth:`_dg0_space`,
        interpolated by an expression compiled once."""

        cached = self._assembled.get("dense_tensor")
        if cached is None:
            space = self._dg0_space()
            cached = (fem.Function(space), interpolation_expression(function, space))
            self._assembled["dense_tensor"] = cached
        dense, expression = cached
        dense.interpolate(expression)
        return dense

    def _dg0_space(self) -> fem.FunctionSpace:
        """DG0 of this variable's shape, with no symmetry: where the step bound
        is assembled, and where a symmetric tensor is written out in full."""

        space = self._assembled.get("dg0_space")
        if space is None:
            mesh = self.solver_parameters.dolfinx_mesh
            space = fem.functionspace(
                mesh, (DISCONTINUOUS_LAGRANGE, 0, self.element.shape)
            )
            self._assembled["dg0_space"] = space
        return space

    def _dg0_cfl_forms(self) -> tuple[fem.Form, fem.Form]:
        """What each cell holds, and the rate at which it stands to lose it.

        The two halves of :meth:`cfl_limit`, compiled once and kept: the bound is
        evaluated at every attempt of every step, and recompiling a form per
        attempt would cost more than the solve it is protecting.
        """

        forms = self._assembled.get("cfl_forms")
        if forms is not None:
            return forms

        solver_parameters = self.solver_parameters
        domain = self.domain
        mesh = solver_parameters.dolfinx_mesh
        normal = ufl.FacetNormal(mesh)
        test = ufl.TestFunction(self._dg0_space())

        content = (
            ufl.inner(self._chi(domain.chi_eps) * self.prev_value, test)
            * solver_parameters.get_mesh_dx()
        )

        # The flux the transport row applies, with the same lagged selector and
        # live magnitude.
        lost = dg0_outflow(
            self.upwind(),
            test,
            normal,
            solver_parameters.get_mesh_dS(),
        )

        # Sources, where they drain rather than inject. The row carries these on
        # dx and ds respectively instead of integrating a boundary flux, so this is
        # where the bound has to meet them; an influx needs no bound, since adding
        # to a cell cannot empty it.
        for source, measure in (
            (injected(self.flux), solver_parameters.get_mesh_dx()),
            (injected(self.flux, boundary=True), solver_parameters.get_mesh_ds()),
        ):
            if not _is_zero(source):
                lost += ufl.inner(positive_part(-source), test) * measure

        forms = (fem.form(content), fem.form(lost))
        self._assembled["cfl_forms"] = forms
        return forms

    def _chi(self, weight: "Expr") -> "Expr | float":
        return weight if self.chi_weighted else 1.0


@dataclass(frozen=True)
class LieTransported(Transported):
    """A tensor carried by a deforming network, which conservation alone misses.

    A scalar riding a velocity satisfies ``d_t f + div(f v) = 0``, and its flux
    list says everything. A gel's strain moment also stretches with the network,

        d_t M + div(M v) = grad(v) M + M grad(v)^T

    whose right-hand side is neither a flux nor a source. So the conservation part
    is still declared as flux, and only the stretching is added here.

    The moment is ``ENERGY_LAGGING``: the energy it stores reaches the velocity
    rows as a semi-implicit stress power, declared by the phase that owns it.
    Keeping the two consistent is that phase's job, including the ``dt^2`` term
    below.
    """

    velocity: "Expr | None" = None

    ## Overrides

    def transport_residual(self) -> ufl.Form:
        """The conservation row, plus the stretching no flux expresses.

        Forward Euler on the stretching does not preserve positive
        semidefiniteness: over random PSD moments and velocity gradients it
        leaves the cone most of the time. The congruence

            m_{k+1} = G m G^T,   G = I + dt L,   L = grad(v)

        does, for *any* ``G``, since
        ``x^T G m G^T x = (G^T x)^T m (G^T x) >= 0``, so the bound is
        unconditional rather than a restriction on the step. The conservation
        row already carries ``m_k``, so what belongs here is the increment,
        written as its expansion

            G m G^T - m = dt(L m + m L^T) + dt^2 L m L^T.

        Expanded rather than formed as the product less ``m``, which differences
        two quantities agreeing to ``O(dt)`` and cancels the leading digits: at
        ``dt = 1e-9`` the difference holds seven digits of the increment where
        the expansion holds sixteen.

        The ``dt^2`` term is taken only where the variable declares itself
        positive; a sign-indefinite moment has no cone to leave and keeps the
        plain Euler stretching. It is the ``dt^2`` viscosity the stress power
        carries, and the two change together or the energy the step releases
        stops matching the transport that released it.

        Quadratic in the live velocity, which is admissible because the row is
        lagging and so solved once the rates are known.
        """

        dx = self.solver_parameters.get_mesh_dx()
        dt = self.solver_parameters.dt_live
        moment = self.prev_value
        gradient = ufl.grad(self.velocity)
        stretched = dt * (
            ufl.dot(gradient, moment) + ufl.dot(moment, gradient.T)
        )
        if self.positive:
            stretched += dt * dt * ufl.dot(
                gradient, ufl.dot(moment, gradient.T)
            )
        return (
            super().transport_residual()
            - ufl.inner(stretched, self.test) * dx
        )

    def cfl_limit(self, cfl: float) -> float:
        """The step over which the explicit stretching cannot run away.

        The moment's entries take either sign, so the entrywise outflow bound
        says nothing about it, and where it is declared positive the congruence
        of :meth:`transport_residual` keeps it semidefinite at any step. What
        remains is growth: the stretching is explicit in the lagged moment and
        amplifies it by ``1 + 2 dt |grad v|`` per step, and holding that below
        the DG0 factor keeps the growth per step comparable to what an advected
        density is allowed.
        """

        stretch = self._max_velocity_gradient()
        if stretch <= 0.0:
            return float("inf")
        return cfl / (2.0 * stretch)

    ## Private helpers

    def _max_velocity_gradient(self) -> float:
        """``max |grad v|`` over the cells, interpolated by an expression
        compiled once."""

        buffer = self._assembled.get("stretch_buffer")
        if buffer is None:
            space = fem.functionspace(
                self.solver_parameters.dolfinx_mesh, (DISCONTINUOUS_LAGRANGE, 0)
            )
            gradient = ufl.grad(self.velocity)
            buffer = (
                fem.Function(space),
                interpolation_expression(
                    ufl.sqrt(ufl.inner(gradient, gradient)), space
                ),
            )
            self._assembled["stretch_buffer"] = buffer

        function, expression = buffer
        function.interpolate(expression)
        values = function.x.array[: owned_dofs(function.function_space)]
        local = float(np.abs(values).max()) if values.size else 0.0
        return global_max(self.solver_parameters.dolfinx_mesh.comm, local)


def transport_step(
    variables: Iterable[Transported], lagging: Iterable[StateElement]
) -> list[Transported]:
    """The variables the transport step solves, each compiled and checked.

    Every lagging variable, each by its own :meth:`Transported.solve`, one at a
    time in no particular order. That equals solving them together only because
    no row reads another's next value -- the scheme is explicit in the state --
    and a row written by hand could break that without looking wrong, leaving
    an answer that depends on the order. It is a fact about the rows together,
    which no single one can check, so it is checked here.

    ``lagging`` is every element declared ``ENERGY_LAGGING``, which the step has
    to cover exactly: an element nothing transports would have no next value
    determined at all, and one transported twice would have two.

    Compiled here rather than at the first step, so that a row the update cannot
    solve is refused before the first rate solve is paid for.
    """

    step = [variable for variable in variables if not variable.energy_live]
    for element in lagging:
        carriers = [variable for variable in step if variable.element is element]
        if len(carriers) != 1:
            raise ValueError(
                f"{element.label!r} is ENERGY_LAGGING and transported by "
                f"{len(carriers)} variables; the transport step determines its "
                "next value, so exactly one has to carry it"
            )
    next_values = {id(variable.element.function("next")): variable for variable in step}
    for variable in step:
        for coefficient in variable.transport_residual().coefficients():
            other = next_values.get(id(coefficient))
            if other is not None and other is not variable:
                raise ValueError(
                    f"the transport row of {variable.label!r} reads the next value "
                    f"of {other.label!r}, which the transport step solves "
                    "separately; read its lagged value, as the scheme does"
                )
    for variable in step:
        variable.transport_forms()
    return step


def potential_of(
    terms: Iterable["EnergyDensity"],
    variable: Transported,
    *,
    domain: ElementDomain,
) -> "Expr":
    """Sum every potential conjugate to one variable over the terms on ``domain``.

    The potential has the shape of the variable, so a term that does not depend
    on it contributes a scalar zero of the wrong shape and is dropped rather than
    added. That is what lets a vector-valued state -- the binned generating
    function, say -- sit in the same list as the scalars.
    """

    contributions = [
        term.potential(variable.next_handle, variable.prev_handle)
        for term in terms
        if term.domain is domain
    ]
    nonzero = [term for term in contributions if not _is_zero(term)]
    if not nonzero:
        return ufl.as_ufl(0.0)
    return sum(nonzero[1:], nonzero[0])


def _is_zero(expression: "Expr") -> bool:
    """Whether ``expression`` is UFL's symbolic zero, which is how an absent term
    of any shape is declared."""

    return isinstance(ufl.as_ufl(expression), ufl.constantvalue.Zero)
