"""The driver: build the mixture's finite elements, then advance it.

A :class:`~ovpsolver.phase_field_system.PhaseFieldSystem` says what the material
is; this says how a step is taken. Everything here is determined by the
declarations -- there is no physics in this module, and a mixture cannot change
any of it.

Building
--------
One pass over :meth:`~ovpsolver.fem.elements.ElementOwner.element_owners`
creates a function space per declared element and binds the functions onto the
owner by object reference, which is how a declaration written as
``self.rates.v`` becomes a coefficient of a form. The pass also sorts each
element into the solve that determines it, and that sorting is the whole content
of the rate step:

* every rate, and every ``ENERGY_LIVE`` variable, goes into one nonlinear
  problem, because the energy is evaluated at their ``k+1`` values and the
  rates cannot be found without them;
* every ``ENERGY_LAGGING`` variable is left to the transport step, because by
  the time the rates are known its row is linear in its own unknown and
  independent of the others. Each solves itself, exactly, cell by cell and with
  no matrix (:meth:`~ovpsolver.phase_field_system.transport.Transported.solve`).

How the nonlinear problem's linearization is solved is ``ovp_step``'s to say:
factored whole, or split along the same declarations into its live states and
its static unknowns (:meth:`PhaseFieldSolver.fieldsplit_index_sets`). The
partition is the one thing the solver supplies to a split, because it is the
one thing a spec cannot state.

Stepping
--------
Irreversible half-step, snapshot, rate solve, transport step. Then the step is
*checked*: the transport rows only preserve positivity below a bound that
depends on the velocities they were solved with, so the bound cannot be known
until after the solve it constrains. A step that violates it -- or that Newton
could not solve at all, which an oversized step is a routine way to cause -- is
rolled back and retaken smaller, and the bound an accepted step reports is what
sizes the next one. The first sub-step of a run has nothing to be sized from and
uses ``warm_start_dt``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from ..fem.elements import ElementDomain, Solve, StaticElement
from ..phase_field_system.transport import transport_step
from . import progress
from .instrument import Instrument
from .onsager import ExcessRayleighian
from .parameters import TimesteppingParameters
from .utils import (
    PositivityViolation,
    SolveDiverged,
    check_converged,
    enable_convergence_history,
    failures_inside,
    reset_convergence_history,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dolfinx.mesh import Mesh

    from ..fem.elements import ElementOwner, ElementSpec
    from ..phase_field_system import PhaseFieldSystem
    from ..phase_field_system.resume import Resume
    from ..phase_field_system.transport import Transported


#: How far above its reported transport bound a step may still be accepted.
#:
#: The reported bound is ``cfl_positive_transport`` times the outflow bound that
#: actually preserves positivity, so a spec asking for ``0.7`` has already set
#: three tenths aside. The slack is headroom against the bound *moving*: an
#: accepted step proposes the bound it just measured, the next solve measures a
#: slightly different one from slightly different velocities, and without it a
#: step a hundredth of a per cent over is thrown away along with its solve.
#:
#: Capped so it never reaches past the real bound: the allowance, in fractions of
#: the outflow bound, is ``min(1, slack * cfl)``, so a spec that sets
#: ``cfl_positive_transport`` close to one gets no slack rather than a step past
#: the condition it asked for.
CFL_ACCEPTANCE_SLACK = 1.1


@dataclass(frozen=True)
class Unknown:
    """One declared element, and the function the solve writes into."""

    spec: "ElementSpec"
    space: "fem.FunctionSpace"
    function: fem.Function

    @property
    def owner(self) -> "ElementOwner":
        return self.spec.owner

    @property
    def label(self) -> str:
        return self.spec.label


@dataclass
class BuiltProblem:
    """What the mixture's declarations were turned into, once, at construction."""

    #: The rate problem's unknowns, every rate and every ``ENERGY_LIVE`` state,
    #: in the order its vector is laid out.
    mechanical: list[Unknown] = field(default_factory=list)
    #: The ``ENERGY_LAGGING`` states, which the transport step solves.
    state: list[Unknown] = field(default_factory=list)
    transported: list["Transported"] = field(default_factory=list)
    #: The transport step: every lagging variable of :attr:`transported`, each
    #: solving itself. See
    #: :func:`~ovpsolver.phase_field_system.transport.transport_step`.
    lagging: list["Transported"] = field(default_factory=list)
    test_space: "ufl.MixedFunctionSpace | None" = None
    entity_maps: tuple | None = None


class PhaseFieldSolver:
    """Advance one phase-field system through the Lie-split OVP timestep."""

    def __init__(
        self,
        system: "PhaseFieldSystem",
        parameters: TimesteppingParameters | None = None,
        *,
        resume: "Resume | None" = None,
    ) -> None:
        self.system = system
        self.parameters = parameters or TimesteppingParameters()
        self.built = self._build(resume)
        self.instrument = Instrument(self)
        # Replaced for the duration of :meth:`watching`. A solver driven a step
        # at a time from a harness has no bar and no need of one, and this way
        # the calls to it work either way.
        self.progress = progress.Progress(0, enabled=False)
        self.rate_problem = self._build_rate_problem()
        self.instrument.newton(self.rate_problem, "rate")
        # After the residual is built, since it compiles the same declarations.
        self.excess_rayleighian = ExcessRayleighian(
            system, system.solver_parameters.ovp_step, self.built.entity_maps
        )
        self.excess_rayleighian.attach(self.rate_problem)
        # The bound the last accepted step reported, which is what sizes the
        # next one. ``None`` until a step has been taken and checked.
        self._proposal: float | None = None

    ## Public

    def fieldsplit_index_sets(
        self, problem: fem_petsc.NonlinearProblem
    ) -> list[tuple[str, PETSc.IS]]:
        """The rate problem's unknowns as ``state``, then ``static``.

        Read off the declarations and nothing else. A :class:`StaticElement` is an
        unknown of the Onsager principle itself -- a velocity, the pressure, a
        multiplier a velocity carries -- and every other unknown of the rate
        problem is a live state, a ``StateElement`` declared ``ENERGY_LIVE``,
        there because the energy rate reads its ``k+1`` value. The halves are
        named after the element types, so a spec's ``fieldsplit_state_*`` and
        ``fieldsplit_static_*`` options say which declarations they solve.

        ``state`` first, because it is the half a Schur factorization eliminates,
        and eliminating it costs nothing. A live state's row is either its
        transport, whose only term in the live states is its accumulation -- the
        flux is explicit in the density -- or the definition of an auxiliary
        potential, which reads its own phase. So the block is triangular with
        diagonal blocks, and LU factors it without fill. The static block left
        behind is the Hessian of the Rayleighian in the rates at fixed states,
        symmetric because every one of its rows is a derivative of that one
        functional.

        The ranges are the owned ones dolfinx recorded when it laid out the rate
        problem's vector, one per unknown in the order of
        :attr:`BuiltProblem.mechanical`, which the matrix rows follow.
        """

        owned, _ = problem.x.getAttr("_blocks")
        start, _ = problem.x.getOwnershipRange()
        halves: dict[str, list[np.ndarray]] = {"state": [], "static": []}
        for unknown, first, last in zip(
            self.built.mechanical, owned[:-1], owned[1:], strict=True
        ):
            name = "static" if isinstance(unknown.spec, StaticElement) else "state"
            halves[name].append(np.arange(start + first, start + last))
        return [
            (
                name,
                PETSc.IS().createGeneral(
                    np.concatenate(ranges).astype(PETSc.IntType),
                    comm=problem.x.getComm(),
                ),
            )
            for name, ranges in halves.items()
            if ranges
        ]

    @contextmanager
    def watching(self, steps: int) -> "Iterator[progress.Progress]":
        """Draw a progress bar over ``steps`` macro steps, for as long as asked.

        Opened by whoever is driving -- :meth:`run` here, or
        :mod:`ovpsolver.phase_field_system.run`, which also writes frames -- and
        installed on the solver rather than passed around, so that the sub-step
        and the rate solve report into it without knowing whether anybody is
        watching.
        """

        self.progress = progress.Progress(steps)
        try:
            yield self.progress
        finally:
            self.progress.close()
            self.progress = progress.Progress(0, enabled=False)

    def step(self, dt: float | None = None) -> float:
        """Advance one macro step, sub-cycled at whatever the transport bound allows.

        Returns the time actually advanced, which is the macro step unless a
        trailing remainder was too small to be worth a nonlinear solve.
        """

        target = self.parameters.dt if dt is None else dt
        if target <= 0.0:
            raise ValueError("dt must be positive")

        remaining = target
        while remaining > self.parameters.min_remainder_fraction * target:
            remaining -= self._substep(min(remaining, self._propose()))
        return target - remaining

    def cfl_allowance(self, limit: float) -> float:
        """The largest step a reported bound is enforced at.

        :data:`CFL_ACCEPTANCE_SLACK` above the bound, and never past the outflow
        bound it is a fraction of. The step *proposed* is still the bound itself,
        so the slack absorbs the bound moving rather than inviting a larger step;
        and positivity is not relaxed, since an accepted step still goes through
        :meth:`_check_nonnegative`.
        """

        cfl = float(self.parameters.cfl_positive_transport)
        if cfl <= 0.0:
            return limit
        return limit * min(CFL_ACCEPTANCE_SLACK, 1.0 / cfl)

    def retry_step_size(self, dt: float, limit: float) -> float:
        """How small to go after a step was rejected for violating its bound.

        The reported bound, backed off by ``retry_cfl_timestep_fraction``, which
        lands inside the admissible range in one attempt rather than bisecting
        towards it. A bound at or near zero means the step was overshot badly
        enough to empty a cell, and then measures the damage rather than the step
        that would have avoided it; with nothing to aim at, the attempt itself is
        shrunk. The same fallback catches a bound that would not shrink the step,
        which would otherwise retry the rejected step forever.
        """

        backoff = self.parameters.retry_cfl_timestep_fraction
        proposed = backoff * limit
        return proposed if 0.0 < proposed < dt else backoff * dt

    ## Private helpers

    def _build(self, resume: "Resume | None") -> BuiltProblem:
        built = BuiltProblem()

        for owner in self.system.element_owners():
            for spec in owner.element_specs():
                mesh = self._mesh_for(spec.domain)
                space = fem.functionspace(mesh, spec.make_element(mesh.basix_cell()))
                if isinstance(spec, StaticElement):
                    self._bind_rate(built, spec, space)
                else:
                    self._bind_variable(built, spec, space)

        built.test_space = (
            ufl.MixedFunctionSpace(*[u.space for u in built.mechanical])
            if built.mechanical
            else None
        )
        tests = ufl.TestFunctions(built.test_space)
        for unknown, test in zip(built.mechanical, tests, strict=True):
            setattr(unknown.owner, unknown.spec.test_attr, test)
        for unknown in built.state:
            setattr(
                unknown.owner,
                unknown.spec.test_attr,
                ufl.TestFunction(unknown.space),
            )

        # The outer boundary's entity map, required as soon as an unknown lives
        # there: ``Lambda`` is a function of the boundary submesh and its
        # constraint is integrated on the parent mesh's ``ds``, so the form has
        # cells of two meshes in it and cannot be compiled without the map.
        if any(
            unknown.spec.domain is ElementDomain.SURFACE
            for unknown in built.mechanical + built.state
        ):
            built.entity_maps = (
                self.system.solver_parameters.get_outer_boundary_entity_maps()
            )

        # The two ways a run can begin, and the only place that chooses between
        # them. Both fill the same functions; a mixture declares its state and
        # is resumable for having declared it, without mentioning resuming.
        if resume is None:
            self.system.set_initial_conditions()
        else:
            self.system.resume_from(resume.read)

        built.transported = list(self.system.transported())
        built.lagging = transport_step(
            built.transported, [unknown.spec for unknown in built.state]
        )
        return built

    def _bind_rate(
        self, built: BuiltProblem, spec: StaticElement, space: "fem.FunctionSpace"
    ) -> None:
        owner = spec.owner
        function = fem.Function(space, name=f"{owner.name()}_{spec.name}")
        setattr(owner.rates, spec.name, function)
        if spec.store_lagged:
            setattr(
                owner.prev,
                spec.name,
                fem.Function(space, name=f"{owner.name()}_prev_{spec.name}"),
            )
        built.mechanical.append(Unknown(spec, space, function))

    def _bind_variable(
        self, built: BuiltProblem, spec: "ElementSpec", space: "fem.FunctionSpace"
    ) -> None:
        owner = spec.owner
        next_function = fem.Function(space, name=f"{owner.name()}_next_{spec.name}")
        setattr(
            owner.prev,
            spec.name,
            fem.Function(space, name=f"{owner.name()}_prev_{spec.name}"),
        )
        setattr(owner.next, spec.name, next_function)
        unknown = Unknown(spec, space, next_function)
        if spec.solve is Solve.ENERGY_LIVE:
            built.mechanical.append(unknown)
        elif spec.solve is Solve.ENERGY_LAGGING:
            built.state.append(unknown)
        elif spec.solve is not Solve.NONE:
            raise ValueError(f"unknown solve target {spec.solve!r}")

    def _mesh_for(self, domain: ElementDomain) -> "Mesh":
        parameters = self.system.solver_parameters
        if domain is ElementDomain.BULK:
            return parameters.dolfinx_mesh
        if domain is ElementDomain.SURFACE:
            return parameters.get_outer_boundary_mesh()
        raise ValueError(
            f"the phase-field layer declares nothing on {domain!r}; only the "
            "outer boundary is a submesh here"
        )

    def _build_rate_problem(self) -> fem_petsc.NonlinearProblem:
        """The Newton problem for the rates and the live states, as the spec solves it.

        Under ``pc_type: fieldsplit`` it is split into the declared halves
        (:meth:`fieldsplit_index_sets`) and every ``fieldsplit_*`` option goes to
        them; anything else factors the linearization whole, and a
        ``fieldsplit_*`` option given there is refused rather than ignored.
        """

        residual = self.system.rate_residual()
        options = dict(self.system.solver_parameters.petsc_options)
        # Held back from what dolfinx is handed, and put in the database
        # afterwards instead. dolfinx sets every option it is given, calls
        # ``setFromOptions`` and then deletes them again; a fieldsplit's sub-KSPs
        # are not created until ``PCSetUp``, at the first solve, and options they
        # never saw are options silently ignored.
        splits = {
            key: options.pop(key)
            for key in list(options)
            if key.startswith("fieldsplit_")
        }
        problem = fem_petsc.NonlinearProblem(
            ufl.extract_blocks(residual),
            [unknown.function for unknown in self.built.mechanical],
            kind="mpi",
            petsc_options_prefix="ovp_rate_",
            petsc_options=options,
            entity_maps=self.built.entity_maps,
        )
        preconditioner = problem.solver.getKSP().getPC()
        database = PETSc.Options()
        prefix = f"{problem.solver.getOptionsPrefix()}fieldsplit_"
        # Kept for as long as the problem is, rather than deleted once read,
        # because they are read late: a half's solver reads its options when the
        # split first sets up, and a factorization package reads its own
        # (``mat_mumps_*``) whenever it analyses the matrix. What goes is
        # whatever an earlier problem left under the same prefix, so that one
        # spec's options cannot quietly apply to another's split.
        for stale in [key for key in database.getAll() if key.startswith(prefix)]:
            database.delValue(stale)
        if preconditioner.getType() == PETSc.PC.Type.FIELDSPLIT:
            preconditioner.setFieldSplitIS(*self.fieldsplit_index_sets(problem))
            for key, value in splits.items():
                database[f"{prefix}{key.removeprefix('fieldsplit_')}"] = value
        elif splits:
            raise ValueError(
                f"the spec sets {sorted(splits)} but pc_type is "
                f"{preconditioner.getType()!r}; those options only mean "
                "anything under pc_type: fieldsplit"
            )
        ksp = problem.solver.getKSP()
        ksp.setPreSolve(lambda *_: self.instrument.linear_solve_started())
        ksp.setPostSolve(self._finish_linear_solve)
        enable_convergence_history(problem)
        return problem

    def _finish_linear_solve(self, ksp: PETSc.KSP, *_) -> None:
        """Count a linear solve, and fail it if a solver inside it did not converge.

        PETSc takes an inner solve that ran out of iterations as merely inexact:
        only its other failures fail the preconditioner around it. Under an outer
        ``preonly`` that inner solve is the whole of the static half's, and its
        tolerance is what holds the static rows, the constraints among them -- so
        a truncated one would loosen the constraints without anyone deciding to.
        Failing the linear solve instead, with PETSc's own reason for a nested
        failure, makes Newton report it and the driver retry smaller, naming the
        inner solver in the cause.
        """

        self.instrument.linear_solve_finished(ksp)
        if ksp.getConvergedReason() > 0 and failures_inside(ksp):
            ksp.setConvergedReason(PETSc.KSP.ConvergedReason.DIVERGED_PCSETUP_FAILED)

    def _propose(self) -> float:
        """The sub-step to attempt next: the bound the last accepted step reported.

        A good prediction, because the velocities move little over a step that
        satisfied it. Before the first step no velocity has been solved, so
        ``warm_start_dt`` stands in, and is checked like any other.
        """

        if self._proposal is None:
            return self.parameters.warm_start_dt
        return self._proposal

    def _substep(self, dt: float) -> float:
        """Take one Lie-split sub-step, retrying smaller until it is admissible.

        A step is refused for violating the bound it reports, or because the rate
        solve failed to converge, which an oversized step is a routine way to
        cause: the convex split makes a step unconditionally descending, not
        unconditionally solvable. Either way the state goes back and a smaller
        step is tried.

        Until the selector is frozen, a rejection keeps the attempt's rates. The
        upwind selectors are lagged from them, and the velocity a rejected attempt
        solved is the closest estimate of the one the retry will find, so the
        retry selects on it. The first rejection for the bound re-measures the
        bound with that selector and then freezes it -- the lagged selector
        ``sigma_k`` replaced by the attempt's ``sigma_{k+1}``, once rather than
        iterated -- and every later rejection restores it with the state.
        """

        saved = self.system.checkpoint()
        #: Whether the selector has been settled for this sub-step. Until it is,
        #: one rejection is allowed to improve it; after, it is held.
        frozen = False
        limit, who = float("inf"), "nothing transported"
        for attempt in range(self.parameters.max_retries + 1):
            self.progress.substep(dt)
            try:
                self._advance(dt)
            except SolveDiverged as diverged:
                self.instrument.retry(attempt, dt, diverged.cause)
                self.instrument.diverged(diverged)
                self.progress.retry()
                self.system.restore(saved, keep_rates=not frozen)
                dt *= self.parameters.retry_diverged_timestep_fraction
                continue

            with self.progress.during(progress.POST):
                limit, who = self._cfl_limit()
                if dt <= self.cfl_allowance(limit):
                    self._check_nonnegative()
                    self._proposal = limit
                    self.instrument.accepted(dt, limit, who)
                    return dt
            self.instrument.retry(attempt, dt, "cfl", limit=limit, binding=who)
            self.progress.retry()
            if not frozen:
                # The one update worth having. This attempt solved from the
                # same pre-step state the retry will, so its velocity is a
                # better selector than the one carried in from whenever a step
                # was last accepted. Lagging it and re-assembling the bound
                # against the state this attempt consumed measures what the
                # retry will actually face: same content, same magnitudes, the
                # selector alone changed.
                self.system.lag_rates()
                updated, updated_who = self._cfl_limit()
                self.instrument.selector_updated(
                    limit, who, updated, updated_who
                )
                limit, who = updated, updated_who
            self.system.restore(saved, keep_rates=not frozen)
            if not frozen:
                # Held from here on. The bound is a discontinuous function of
                # the step while the selector moves with it, and the acceptance
                # test can oscillate against a bound that jumps indefinitely --
                # every attempt reporting a limit that the next attempt's
                # selector invalidates. Fixed, the bound varies continuously and
                # tends to a positive value as the step shortens, so the backoff
                # reaches an admissible step in finitely many attempts.
                # Re-checkpointed, so that a later rejection restores this
                # velocity and not the one before it.
                saved = self.system.checkpoint()
                frozen = True
            if limit <= 0.0:
                break
            dt = self.retry_step_size(dt, limit)

        if limit > 0.0:
            raise RuntimeError(
                f"no admissible sub-step after {self.parameters.max_retries} "
                f"retries, down to dt={dt:.6e}, with {who} bounding it at "
                f"{limit:.6e}"
            )
        raise RuntimeError(
            f"{who} admits no positive step: a cell holds nothing while still "
            "losing to outflow, so no reduction in dt keeps it non-negative"
        )

    def _advance(self, dt: float) -> None:
        """The irreversible half-step, the rate solve, then the transport step."""

        parameters = self.system.solver_parameters
        parameters.set_timestep(dt)

        if not parameters.irreversible_step.skip:
            self.system.irreversible_timestep(dt)
        # After the irreversible half-step, so the rate solve lags at k+1/2 and
        # the convex split brackets the increment it is about to make.
        self.system.snapshot()

        if parameters.ovp_step.skip:
            # Every rate stays where it was. The transport step below still
            # runs, which is the point: it makes the transport rows a test
            # against a frozen flow rather than against no step at all.
            self.instrument.rate_solve_skipped(dt)
        else:
            with self.progress.during(progress.RATE):
                reset_convergence_history(self.rate_problem)
                self.excess_rayleighian.reset()
                started = time.perf_counter()
                self.rate_problem.solve()
                wall = time.perf_counter() - started
            self.instrument.timing(wall)
            self.instrument.rate_solve(dt, wall)
            self.instrument.excess_rayleighian(self.excess_rayleighian)
            self.progress.rate_solve(self.excess_rayleighian.iterations)
            self.progress.excess_rayleighian(self.excess_rayleighian)
            check_converged(
                self.rate_problem,
                "mechanical OVP",
                unknowns=[unknown.label for unknown in self.built.mechanical],
            )
            # After the check, so describing a solve can never hide its failure.
            self.instrument.linear_algebra()

        with self.progress.during(progress.STATE):
            for variable in self.built.lagging:
                started = time.perf_counter()
                variable.solve()
                self.instrument.state_solve(
                    variable.label, time.perf_counter() - started
                )

    def _cfl_limit(self) -> tuple[float, str]:
        """The tightest step bound any transported variable reports, and whose.

        Assembled from the velocities just solved, so it is the verdict on the
        step just taken rather than a prediction about it.
        """

        cfl = self.parameters.cfl_positive_transport
        bounds = [(v.cfl_limit(cfl), v.label) for v in self.built.transported]
        return min(bounds, default=(float("inf"), "nothing transported"))

    def _check_nonnegative(self) -> None:
        """Stop the run if anything declared non-negative came out negative.

        Every transported variable that declares itself positive, in the sense
        that variable means it
        (:meth:`~ovpsolver.phase_field_system.transport.Transported.floor_value`):
        the phases, the crosslink moments and the modulus entry by entry, the
        strain moment by its eigenvalues. Read off the declarations, so a phase
        adding a state it says is positive is checked for saying so.

        For an entrywise variable this is meant to be unreachable: its step bound
        is evaluated after the solve on the flux its row actually applies, and so
        is sufficient outright
        (:meth:`~ovpsolver.phase_field_system.transport.Transported.cfl_limit`).
        A tensor has no such bound, its positivity being its spectrum, which no
        bound on the transport of each entry controls; for the strain moment this
        is the check. A quantity defined as a difference of transported ones,
        such as a polymerizing phase's gel, has no row of its own and is not
        checked here.

        Only the lower bound: a phase slightly over one is out of range but
        harmless to evaluate, and saturation is the constraint's to hold. Not a
        retry, unlike the bound, since a violation is not a step that was too
        long and a shorter one is a guess rather than a remedy. Continuing is
        worse than stopping: the floors that keep the divisions finite would hide
        it until it surfaced as a divergence with no visible cause.
        """

        floor = self.parameters.positivity_floor_throw
        if floor is None:
            return

        values = [
            (variable, variable.floor_value())
            for variable in self.built.transported
            if variable.positive
        ]
        self.instrument.floors([(v.label, minimum) for v, minimum in values])
        violations = [
            f"{variable.label} min={minimum:.6e}"
            for variable, minimum in values
            if minimum < floor
        ]
        if violations:
            raise PositivityViolation(
                f"undershot positivity_floor_throw={floor:.3e}: "
                + "; ".join(violations)
            )
