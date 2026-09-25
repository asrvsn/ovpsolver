"""What a run says about itself while it takes a step.

Kept out of the driver (:mod:`ovpsolver.solver.solver`), which says what a step
*is*; this says what happened in one. Like the solve, it is built from the
declarations: there is no list here of which quantities a model has, only a walk
over the unknowns the declarations produced with a statistic chosen by each
one's shape (:func:`summarize`), so a phase somebody else writes is described
without this module learning anything about it.

Three levels:

``INFO``
    the shape of the run and the fate of every step: what was built, what each
    macro step advanced, and every retry with its reason. The default, and small
    enough to read end to end afterwards.
``DEBUG``
    the numbers behind that: per-unknown statistics on each accepted step, and
    per-Newton residuals when ``snes_monitor`` asks. The level a run behaving
    strangely is re-read at.
failure
    the full PETSc report (:func:`~ovpsolver.solver.utils.failure_report`),
    carried by the exception.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from petsc4py import PETSc
from tqdm.auto import tqdm

from ..fem.reduce import global_max, global_min, global_sum, owned_function_values
from ..mesh.utils import lattice_coherence
from .utils import factorizations, krylov_iterations, relative_asymmetry, solvers

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem
    from mpi4py import MPI

    from ..fem.save import Saver
    from .onsager import ExcessRayleighian
    from .solver import PhaseFieldSolver

#: Root of the package's logger tree. Every module logs beneath it, so one
#: handler on this catches the driver and the mixture alike.
LOGGER_NAME = "ovpsolver"

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

logger = logging.getLogger(f"{LOGGER_NAME}.ovp")

#: Relative asymmetry of the static block above which a symmetric factorization
#: of it is reported as reading the wrong matrix. Far above the roundoff a
#: Hessian assembles to, which is around ``1e-16``.
STATIC_ASYMMETRY_TOLERANCE = 1.0e-10


def configure(
    log_path: "str | Path | None",
    *,
    comm: "MPI.Comm",
    debug: bool = False,
    append: bool = False,
) -> None:
    """Point the package's logger at the console and the run's own file.

    Rank-aware, because a log written by every rank of an MPI run is unreadable
    and, if they share a path, corrupt. Ranks other than zero get a null handler
    rather than a filter, so their formatting work is never done at all. The
    file, when a spec names one, takes the whole trace; the console keeps the
    INFO summary.

    ``append`` keeps what is already in the file, which a resumed run wants: a
    trajectory assembled from two sittings is only readable if the first is
    still there to say where the second started.
    """

    root = logging.getLogger(LOGGER_NAME)
    root.handlers.clear()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.propagate = False

    if getattr(comm, "rank", 0) != 0:
        root.addHandler(logging.NullHandler())
        return

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    console = _ConsoleHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    # The console is a subset of the file by level, and by this one record as
    # well: what ends a run reaches the terminal as Python's traceback, and
    # printing it twice makes the second copy look like a second failure.
    console.addFilter(lambda record: not getattr(record, "already_on_stderr", False))
    root.addHandler(console)

    if log_path is not None:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            path, mode="a" if append else "w", encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.info("logging to %s", path)


@contextmanager
def log_failures() -> "Iterator[None]":
    """Put whatever ends a run in that run's own log, then let it end the run.

    A raised exception is otherwise the one thing a run says that its log does
    not hear, and it carries what the log is read for afterwards: the PETSc
    diagnostic (:func:`~ovpsolver.solver.utils.failure_report`), or the cells a
    state left its admissible set in. ``KeyboardInterrupt`` is recorded too,
    since it is how a long run usually ends and a reader wants to know whether
    the run was stopped or stopped itself; ``SystemExit`` and ``GeneratorExit``
    are neither, and a traceback for one would report a failure that did not
    happen.

    Nothing is swallowed. The exception is re-raised unchanged, so the exit
    status and the console traceback are the caller's, which is why the record
    is marked for the file alone.
    """

    try:
        yield
    except (Exception, KeyboardInterrupt):
        logger.exception("the run stopped here", extra={"already_on_stderr": True})
        raise


class _ConsoleHandler(logging.StreamHandler):
    """Log lines that step around the progress bar instead of through it.

    Anything printed while a bar is up lands on its line and is overwritten by
    the next redraw. ``tqdm.write`` clears the bar, writes the line and puts the
    bar back below it; with no bar up it is an ordinary write. It never fails
    over a bar -- the log is the thing that matters.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            super().emit(record)


class Instrument:
    """The running commentary on one solver.

    Held by the driver and called at its seams. The per-step methods check their
    level before doing any work, so switched-off instrumentation costs a
    comparison per step.
    """

    def __init__(self, solver: "PhaseFieldSolver") -> None:
        self.solver = solver
        self.system = solver.system
        self.comm = solver.system.solver_parameters.dolfinx_mesh.comm
        self._elapsed = 0.0
        self._steps = 0
        self._retries = 0
        self._described_linear_algebra = False
        #: Krylov iterations since the last rate solve was reported, and since the
        #: last Newton line, wherever in the solver tree they were taken; and the
        #: seconds spent in linear solves since that line. See
        #: :meth:`linear_solve_finished`.
        self._krylov = 0
        self._newton_krylov = 0
        self._linear_wall = 0.0
        self._linear_started = 0.0

    ## What was built

    def header(self) -> None:
        """Say what the declarations produced, before anything expensive runs.

        Nearly every mistake in a spec shows here, as a missing unknown or a
        variable on the wrong side of the split, and far more cheaply than in the
        residual.
        """

        system, solver = self.system, self.solver
        parameters = system.parameters
        stepping = parameters.solver.timestepping

        logger.info("spec %s", getattr(parameters, "source", "<built in Python>"))
        logger.info(
            "mixture %s: phases %s on %d inclusions from %s",
            system.name(),
            ", ".join(phase.name() for phase in system.phase_fields),
            len(system.diffuse_domain.signed_distances),
            parameters.solver.diffuse_domain.geometry_path.name,
        )
        logger.info(
            "rate solve unknowns %s", [u.label for u in solver.built.mechanical]
        )
        logger.info(
            "transport step unknowns %s", [u.label for u in solver.built.state]
        )
        logger.info("transported %s", [v.label for v in solver.built.transported])
        problem = solver.rate_problem
        if problem.solver.getKSP().getPC().getType() == PETSc.PC.Type.FIELDSPLIT:
            logger.info(
                "rate solve split %s",
                ", ".join(
                    f"{name}={indices.getSize()}"
                    for name, indices in solver.fieldsplit_index_sets(problem)
                ),
            )
        logger.info(
            "stepping dt=%.3e warm_start=%.3e n_steps=%d irreversible=%s ovp=%s",
            stepping.dt,
            stepping.warm_start_dt,
            stepping.n_steps,
            not parameters.solver.irreversible_step.skip,
            not parameters.solver.ovp_step.skip,
        )
        mesh = system.solver_parameters.dolfinx_mesh
        cells = mesh.topology.index_map(mesh.topology.dim).size_local
        vertices = mesh.topology.index_map(0).size_local
        coherence, axis = lattice_coherence(mesh)
        logger.info(
            "mesh cells=%.0f vertices=%.0f lengthscale=%.6e "
            "lattice_coherence=%.3f axis=%.1fdeg",
            global_sum(mesh.comm, cells),
            global_sum(mesh.comm, vertices),
            system.solver_parameters.mesh_lengthscale,
            coherence,
            np.degrees(axis),
        )

    def saving(self, saver: "Saver") -> None:
        """What the run will write, and on what element, once the saver is open.

        Not with the rest of the header, because an expression has no element
        until the saver has found one it can be evaluated on, and which one is
        worth recording: a diagnostic that lands on the quadrature space is
        written at every integration point of every cell.
        """

        logger.info("saving %d field(s) to %s", len(saver.fields), saver.path)
        for name, series in sorted(saver.fields.items()):
            logger.info(
                "  %s -> %s %s n=%d",
                name,
                series.describe(),
                series.mode,
                series.n_dofs,
            )

    ## Per-solve

    def linear_solve_started(self) -> None:
        self._linear_started = time.perf_counter()

    def linear_solve_finished(self, ksp: PETSc.KSP) -> None:
        """One linear solve done: add its wall time and its Krylov iterations.

        The iterations wherever in the solver tree they were taken
        (:func:`~ovpsolver.solver.utils.krylov_iterations`). Under an outer
        ``preonly`` each linear solve is one solve of the static half, so that is
        all of it, and the number that moves when a split's options change.
        Counted whether or not Newton is traced, since :meth:`rate_solve`
        reports the total.
        """

        self._linear_wall += time.perf_counter() - self._linear_started
        try:
            iterations = sum(krylov_iterations(ksp).values())
        except PETSc.Error:
            # A split that failed to set up has no halves to count, and the
            # failure is the solve's to report rather than this one's.
            iterations = 0
        self._newton_krylov += iterations
        self._krylov += iterations

    def newton(self, problem: "NonlinearProblem", label: str) -> None:
        """Attach a per-iteration residual, timing and Krylov trace to a SNES.

        A callback per Newton iteration, so only when the spec's ``snes_monitor``
        asks. When a step is failing, this is what distinguishes converging slowly
        from not converging.

        ``ksp`` is the time in linear solves and ``its`` their Krylov iterations,
        both as :meth:`linear_solve_finished` accumulated them since the last
        line; ``wall`` less ``ksp`` is assembly and linesearch. The split says
        what to reach for: the preconditioner and how often the Jacobian is
        rebuilt improve the first, the quadrature and the linesearch the second.
        A factorization happens inside ``KSPSolve`` and so counts as ``ksp``:
        under ``snes_lag_preconditioner`` the difference between a rebuilding
        iteration's ``ksp`` and a reusing one's is what the factorization costs.
        """

        if not self.system.solver_parameters.ovp_step.snes_monitor:
            return
        rank = getattr(self.comm, "rank", 0)
        previous: list[float | None] = [None]

        def monitor(_snes, iteration: int, residual_norm: float) -> None:
            now = time.perf_counter()
            last, previous[0] = previous[0], now
            spent, self._linear_wall = self._linear_wall, 0.0
            iterations, self._newton_krylov = self._newton_krylov, 0
            if rank != 0:
                return
            gap = 0.0 if last is None or iteration == 0 else now - last
            logger.debug(
                "newton [%s] it=%3d |F|=%.6e wall=%.3f ksp=%.3f its=%d",
                label,
                iteration,
                residual_norm,
                gap,
                spent,
                iterations,
            )

        problem.solver.setMonitor(monitor)

    def selector_updated(
        self, before: float, before_who: str, after: float, after_who: str
    ) -> None:
        """The step bound, re-read once the solved velocity became the selector.

        Both numbers describe the *same* attempt -- same content, same flux
        magnitudes -- so their ratio is the price of having selected on a stale
        velocity and nothing else. Near one, the step is genuinely bounded by the
        flow. Large, the bound is being divided by a density the row imported from
        the wrong side of a facet, a statement about how sharp a field is across a
        cell rather than about how fast anything moves. Logged on every update,
        because the trend over a run is the diagnosis.
        """

        logger.debug(
            "selector updated: limit %.6e (%s) -> %.6e (%s), x%.3g",
            before,
            before_who,
            after,
            after_who,
            (after / before) if before > 0.0 else float("inf"),
        )

    def rate_solve(self, dt: float, wall: float) -> None:
        """One rate solve: its step, wall time, and Newton and Krylov iterations.

        The Krylov count is summed over the solve, wherever in the solver tree
        the iterations were taken (:meth:`linear_solve_finished`): zero under a
        direct solve, and under a split the number to watch, since what
        eliminating the states adds to the static block is left to the Krylov
        iteration and grows with the step. Logged whether or not the solve
        converged, since a count that ran out is the first thing to read about
        one that did not.
        """

        logger.info(
            "rate_solve dt=%.6e wall=%.3f newton=%d krylov=%d",
            dt,
            wall,
            self.solver.rate_problem.solver.getIterationNumber(),
            self._krylov,
        )
        # All three, because a solve that failed in a linear solve never reached
        # the Newton line that would have reset the other two, and the next solve
        # would open by reporting this one's leftovers.
        self._krylov = self._newton_krylov = 0
        self._linear_wall = 0.0

    def linear_algebra(self) -> None:
        """What the rate solve's factorizations cost, and what the static one reads.

        Once, after the first rate solve that converged: PETSc sets the
        preconditioner up at its first use, and describing a failed solve is not
        worth risking the report of its failure.

        Every solver in the tree as PETSc actually set it up: its type, and for
        one that iterates its tolerance and iteration cap. PETSc ignores an
        option it does not recognize without a word, so this is where a
        misspelled one shows, as a setting that did not change. Factorization
        memory as MUMPS counts it, used and estimated at analysis, because a
        process's resident size misreads a factorization on a machine that
        compresses memory; entries for anything else factored.

        Under a split, also how symmetric the static block is. A symmetric
        factorization of it (``fieldsplit_static_pc_type: cholesky``) reads the
        upper triangle alone, which is exact because the block is a Hessian
        (:meth:`~ovpsolver.solver.solver.PhaseFieldSolver.fieldsplit_index_sets`)
        and a slowly converging preconditioner if it is not. So this warns rather
        than stops: the Krylov iteration still converges on the true operator.
        """

        if self._described_linear_algebra:
            return
        self._described_linear_algebra = True
        ksp = self.solver.rate_problem.solver.getKSP()
        for solver in solvers(ksp):
            if solver.getType() == PETSc.KSP.Type.PREONLY:
                settings = ""
            else:
                rtol, _, _, max_it = solver.getTolerances()
                settings = f" rtol={rtol:.1e} max_it={max_it}"
            logger.info(
                "solver %s %s%s pc=%s",
                solver.getOptionsPrefix(),
                solver.getType(),
                settings,
                solver.getPC().getType(),
            )
        for prefix, factor in factorizations(ksp):
            if factor.getType() == "mumps":
                logger.info(
                    "factorization %s mumps: %d MB used, %d MB estimated",
                    prefix,
                    factor.getMumpsInfog(22),
                    factor.getMumpsInfog(17),
                )
            else:
                logger.info(
                    "factorization %s %s: %d entries",
                    prefix,
                    factor.getType(),
                    int(factor.getInfo()["nz_used"]),
                )

        preconditioner = ksp.getPC()
        if preconditioner.getType() != PETSc.PC.Type.FIELDSPLIT:
            return
        static = preconditioner.getFieldSplitSubKSP()[-1]
        _, matrix = static.getOperators()
        asymmetry = relative_asymmetry(matrix)
        logger.info("static block asymmetry %.1e", asymmetry)
        if (
            static.getPC().getType() == PETSc.PC.Type.CHOLESKY
            and asymmetry > STATIC_ASYMMETRY_TOLERANCE
        ):
            logger.warning(
                "the static block is not symmetric (%.1e) but is factored "
                "symmetrically, which reads its upper triangle alone; expect the "
                "Krylov iteration to converge slowly",
                asymmetry,
            )

    def excess_rayleighian(self, measure: "ExcessRayleighian") -> None:
        """How converged the rate solve was, in the metric the problem supplies
        (:mod:`ovpsolver.solver.onsager`).

        Every step, not only when it is stopped on: ``L`` says whether Newton
        finished, ``bregman`` whether the step was short enough to be worth
        finishing, and the two are worth reading against each other long before
        either is trusted with the decision.
        """

        logger.info("excess_rayleighian %s", measure.summary())

    def rate_solve_skipped(self, dt: float) -> None:
        logger.info("rate_solve dt=%.6e skipped", dt)

    def state_solve(self, label: str, wall: float) -> None:
        logger.debug("state_solve %s wall=%.3f", label, wall)

    ## Per-step

    def accepted(self, dt: float, limit: float, binding: str) -> None:
        """One sub-step passed its checks. Report it, and what nearly stopped it.

        The margin is the number to watch. A step running at the bound is one the
        driver is sizing, the intended regime; one far below it is being sized by
        something else, usually the macro step or a retry that overshot downwards.
        """

        self._steps += 1
        margin = dt / limit if 0.0 < limit < np.inf else 0.0
        logger.debug(
            "substep dt=%.6e limit=%.6e margin=%.3f binding=%s",
            dt,
            limit,
            margin,
            binding,
        )
        if logger.isEnabledFor(logging.DEBUG):
            self.fields()

    def floors(self, values: "Sequence[tuple[str, float]]") -> None:
        """How much room every non-negative variable has left, on a step that passed.

        The step bound is meant to make the floor check unreachable, and this says
        how close it came. The trend is what to read: a margin that erodes
        steadily is a scheme running out of room, and one that falls off a cliff
        in a single step is a bound that failed to see something. Every step
        rather than every frame, since the values are already in hand from the
        check.
        """

        if not values or not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "floors %s",
            " ".join(f"{label}={minimum:+.6e}" for label, minimum in values),
        )

    def retry(
        self,
        attempt: int,
        dt: float,
        cause: str,
        *,
        limit: float | None = None,
        binding: str | None = None,
    ) -> None:
        """A sub-step was refused. Always INFO: retries are how the run reports
        that its step size is wrong, and a run spending most of its wall time on
        them looks fine at every other level of the log."""

        self._retries += 1
        if limit is None:
            logger.info("step_retry attempt=%d dt=%.6e cause=%s", attempt, dt, cause)
            return
        logger.info(
            "step_retry attempt=%d dt=%.6e cause=%s limit=%.6e binding=%s",
            attempt,
            dt,
            cause,
            limit,
            binding,
        )

    def diverged(self, failure: BaseException) -> None:
        """The full PETSc report behind a retry, which INFO only names.

        DEBUG because it is many lines and the run recovers from it; the same
        report is raised, and so printed in full, if the retries run out.
        """

        logger.debug("%s", failure)

    def macro_step(self, index: int, advanced: float, reached: float) -> None:
        logger.info(
            "macro_step index=%d advanced=%.6e time=%.6e substeps=%d retries=%d",
            index,
            advanced,
            reached,
            self._steps,
            self._retries,
        )
        self._steps = self._retries = 0

    def finished(self, steps: int, reached: float) -> None:
        logger.info(
            "run finished steps=%d time=%.6e wall=%.1fs", steps, reached, self._elapsed
        )

    def timing(self, wall: float) -> None:
        self._elapsed += wall

    ## The numbers behind it

    def fields(self) -> None:
        """One statistic line per unknown, over whatever the mixture declared."""

        for unknown in self.solver.built.mechanical + self.solver.built.state:
            logger.debug("field %s %s", unknown.label, summarize(unknown.function))


def summarize(function: "fem.Function") -> str:
    """The range and mean of a scalar, or of the magnitude of anything with
    components, which for a velocity is its speed.

    Rank decides, being the only classification available without knowing what
    the field means. Components are the *stored* ones, per the dofmap, so a
    symmetric tensor's magnitude counts its off-diagonal once: not the Frobenius
    norm, and not worth reconstructing the layout for in a number that is
    watched for its size between steps rather than computed with. Only owned
    values count, since a ghost is another rank's copy of the same degree of
    freedom.
    """

    space = function.function_space
    components = space.dofmap.index_map_bs
    comm = space.mesh.comm
    values = owned_function_values(function)
    label = ""
    if components > 1:
        values = np.linalg.norm(values.reshape(-1, components), axis=1)
        label = "|v| " if len(function.ufl_shape) == 1 else f"|.|({components}) "

    # A rank owning nothing contributes the identity of each reduction.
    low = global_min(comm, float(values.min()) if values.size else np.inf)
    high = global_max(comm, float(values.max()) if values.size else -np.inf)
    count = global_sum(comm, values.size)
    mean = global_sum(comm, float(values.sum())) / count if count else float("nan")
    return f"{label}min={low:.6e} max={high:.6e} mean={mean:.6e}"
