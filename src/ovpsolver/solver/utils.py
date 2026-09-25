"""PETSc diagnostics, and the failures a step can end in."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from petsc4py import PETSc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from dolfinx.fem.petsc import NonlinearProblem


class PositivityViolation(RuntimeError):
    """Something declared non-negative came out negative on an admissible step.

    Its own type, and deliberately not one the driver retries: a shorter step is
    a guess rather than a remedy
    (:meth:`~ovpsolver.solver.solver.PhaseFieldSolver._check_nonnegative`).
    """


class SolveDiverged(RuntimeError):
    """A solve failed in a way a smaller step might fix, so the driver retries.

    Its own type so that the retry does not swallow unrelated failures.
    ``cause`` says which solve failed and how, and is what the retry is logged
    under, because the same response covers different stories: Newton failing
    is the step being too long for the nonlinearity, while a linear solve inside
    it running out of Krylov iterations is the preconditioner losing to the step
    -- what a split leaves to the iteration grows with ``dt``.
    """

    def __init__(self, message: str, *, cause: str = "diverged") -> None:
        super().__init__(message)
        self.cause = cause


def check_converged(
    problem: "NonlinearProblem",
    name: str,
    *,
    unknowns: list[str] | None = None,
) -> None:
    """Raise :class:`SolveDiverged` with the full PETSc report if a solve failed.

    The cause is ``newton:<reason>``, or ``linear:<reason>`` when Newton stopped
    because a linear solve inside it did, with that solve's own reason and
    whatever failed inside it.
    """

    solver = problem.solver
    reason = int(solver.getConvergedReason())
    if reason > 0:
        return
    if reason == PETSc.SNES.ConvergedReason.DIVERGED_LINEAR_SOLVE:
        ksp = solver.getKSP()
        cause = f"linear:{ksp_reason_name(int(ksp.getConvergedReason()))}"
        inside = failures_inside(ksp)
        if inside:
            cause += f"[{'; '.join(inside)}]"
    else:
        cause = f"newton:{snes_reason_name(reason)}"
    raise SolveDiverged(failure_report(problem, name, unknowns=unknowns), cause=cause)


def enable_convergence_history(problem: "NonlinearProblem") -> None:
    """Ask the SNES and its KSP to keep per-iteration convergence histories."""

    solver = problem.solver
    _try_call(solver, "setConvergenceHistory")
    ksp = _try_call(solver, "getKSP")
    if ksp is not None:
        _try_call(ksp, "setConvergenceHistory")


def reset_convergence_history(problem: "NonlinearProblem") -> None:
    """Clear the retained convergence histories before one solve attempt."""

    solver = problem.solver
    _try_call(solver, "resetConvergenceHistory")
    ksp = _try_call(solver, "getKSP")
    if ksp is not None:
        _try_call(ksp, "resetConvergenceHistory")


def solvers(ksp: "PETSc.KSP") -> "Iterator[PETSc.KSP]":
    """A solver and every solver inside it, outermost first.

    Down through fieldsplits, the one preconditioner here that holds solvers of
    its own. Meaningful only once the preconditioner has been set up, which PETSc
    does at its first use; before that a split has no halves to walk.
    """

    yield ksp
    preconditioner = ksp.getPC()
    if preconditioner.getType() == PETSc.PC.Type.FIELDSPLIT:
        for sub in preconditioner.getFieldSplitSubKSP():
            yield from solvers(sub)


def factorizations(ksp: "PETSc.KSP") -> "Iterator[tuple[str, PETSc.Mat]]":
    """Every matrix factored inside a solver, by options prefix, so a split
    reports each factored half separately."""

    for solver in solvers(ksp):
        preconditioner = solver.getPC()
        if preconditioner.getType() in (PETSc.PC.Type.LU, PETSc.PC.Type.CHOLESKY):
            yield solver.getOptionsPrefix(), preconditioner.getFactorMatrix()


def snes_reason_name(reason: int) -> str:
    """A nonlinear solve's converged reason, by name."""

    return _enum_name(PETSc.SNES.ConvergedReason, reason)


def ksp_reason_name(reason: int) -> str:
    """A linear solve's converged reason, by the name PETSc gives it now.

    Never named against the SNES enum, which shares its values: -3 is
    ``DIVERGED_LINEAR_SOLVE`` to a SNES and ``DIVERGED_MAX_IT`` to a KSP. And
    petsc4py still calls -11 by its name from before PETSc 3.11,
    ``DIVERGED_PCSETUP_FAILED``, which misleads: it is raised when the
    preconditioner fails to set up *or to apply*, and under a split the second
    includes a solver inside it failing to converge.
    """

    if reason == -11:
        return "DIVERGED_PC_FAILED"
    return _enum_name(PETSc.KSP.ConvergedReason, reason)


def failures_inside(ksp: "PETSc.KSP") -> list[str]:
    """What failed inside a solver: nested solvers, then factorizations.

    What a ``DIVERGED_PC_FAILED`` does not say. The common case is a solver
    inside a split that ran out of iterations -- under an outer ``preonly``, the
    static half's Krylov iteration is inside the preconditioner. A failed MUMPS
    factorization is named by ``INFOG(1)`` and its detail ``INFOG(2)``: -10 is a
    numerically singular matrix, -13 a failed allocation, and most other codes a
    workspace sized at analysis that proved too small, which
    ``mat_mumps_icntl_14`` relaxes.

    Empty when there is nothing to name, including when the tree cannot be
    walked because the split never finished setting up.
    """

    try:
        found = []
        for solver in list(solvers(ksp))[1:]:
            reason = int(solver.getConvergedReason())
            if reason < 0:
                found.append(
                    f"{solver.getOptionsPrefix()} {ksp_reason_name(reason)} "
                    f"after {solver.getIterationNumber()}"
                )
        found += [
            f"{prefix} mumps INFOG(1)={factor.getMumpsInfog(1)} "
            f"INFOG(2)={factor.getMumpsInfog(2)}"
            for prefix, factor in factorizations(ksp)
            if factor.getType() == "mumps" and factor.getMumpsInfog(1) < 0
        ]
    except PETSc.Error:
        return []
    return found


def krylov_iterations(ksp: "PETSc.KSP") -> dict[str, int]:
    """Iterations of the last solve, by options prefix, of every solver that iterates.

    ``preonly`` applies its preconditioner once and is left out: under a direct
    solve this is empty, under a split whose outer solver iterates it is that
    solver's count, and under an outer ``preonly`` it is the static half's,
    which is where the iterating moved. A solver iterating inside another that
    also iterates reports only its own last solve.
    """

    return {
        solver.getOptionsPrefix(): solver.getIterationNumber()
        for solver in solvers(ksp)
        if solver.getType() != PETSc.KSP.Type.PREONLY
    }


def relative_asymmetry(matrix: "PETSc.Mat", samples: int = 3) -> float:
    """``max |y'Ax - x'Ay| / (|y| |Ax| + |x| |Ay|)`` over random pairs ``x, y``.

    Zero to roundoff for a symmetric matrix, whatever its scaling. A probe rather
    than ``A - A'``, which would copy the matrix -- on a block of the rate problem
    a sizeable share of the memory a split exists to save -- while this costs two
    products per pair.
    """

    right, left = matrix.createVecs()
    x, y = right.duplicate(), right.duplicate()
    image_x, image_y = left.duplicate(), left.duplicate()
    # One generator for every draw. Without it each ``setRandom`` seeds a fresh
    # one identically, so ``x`` and ``y`` come out equal and the difference below
    # is ``x'Ax - x'Ax``, zero for any matrix at all.
    draw = PETSc.Random().create(comm=matrix.getComm())
    draw.setType(PETSc.Random.Type.RAND)
    worst = 0.0
    for _ in range(samples):
        x.setRandom(draw)
        y.setRandom(draw)
        matrix.mult(x, image_x)
        matrix.mult(y, image_y)
        scale = y.norm() * image_x.norm() + x.norm() * image_y.norm()
        if scale > 0.0:
            worst = max(worst, abs(y.dot(image_x) - x.dot(image_y)) / scale)
    return worst


def failure_report(
    problem: "NonlinearProblem",
    name: str,
    *,
    unknowns: list[str] | None = None,
) -> str:
    """Everything the SNES and its KSP can say about a failed solve, as one message.

    Each query goes through :func:`_try_call`, so a solver left in a state that
    refuses one still reports the rest.
    """

    snes = problem.solver
    ksp = snes.getKSP()
    line_search = _try_call(snes, "getLineSearch")
    preconditioner = _try_call(ksp, "getPC")
    lines = [f"{name} solve failed to converge."]
    if unknowns:
        lines.append("Unknown blocks:")
        lines.extend(f"  - {unknown}" for unknown in unknowns)
    lines += [
        "SNES/KSP diagnostics:",
        f"  type: {_try_call(snes, 'getType')!r}",
        f"  options_prefix: {_try_call(snes, 'getOptionsPrefix')!r}",
        f"  reason: {_reason(snes, snes_reason_name)}",
        f"  iterations: {_try_call(snes, 'getIterationNumber')!r}",
        f"  tolerances: {_try_call(snes, 'getTolerances')!r}",
        f"  function_norm: {_try_call(snes, 'getFunctionNorm')!r}",
        "  linear_solve_iterations: "
        f"{_try_call(snes, 'getLinearSolveIterations')!r}",
        f"  convergence_history: {_history(snes)}",
        f"  line_search_type: {_try_call(line_search, 'getType')!r}",
        "  nested KSP diagnostics:",
        f"    type: {_try_call(ksp, 'getType')!r}",
        f"    options_prefix: {_try_call(ksp, 'getOptionsPrefix')!r}",
        f"    reason: {_reason(ksp, ksp_reason_name)}",
        f"    iterations: {_try_call(ksp, 'getIterationNumber')!r}",
        f"    tolerances: {_try_call(ksp, 'getTolerances')!r}",
        f"    residual_norm: {_try_call(ksp, 'getResidualNorm')!r}",
        f"    convergence_history: {_history(ksp)}",
        f"    pc_type: {_try_call(preconditioner, 'getType')!r}",
    ]
    lines.extend(f"  failed inside: {failure}" for failure in failures_inside(ksp))
    return "\n".join(lines)


def _enum_name(enum_type: type, value: int) -> str:
    """The first name, alphabetically, that a petsc4py enum gives ``value``."""

    for name in dir(enum_type):
        if not name.startswith("_") and getattr(enum_type, name) == value:
            return name
    return str(value)


def _reason(solver: object, name_of: "Callable[[int], str]") -> str:
    """A solver's converged reason as ``value (NAME)``, named against its own enum."""

    reason = _try_call(solver, "getConvergedReason")
    try:
        value = int(reason)
    except (TypeError, ValueError):
        return repr(reason)
    return f"{value} ({name_of(value)})"


def _try_call(obj: object, method_name: str) -> object:
    """``obj.method_name()``, ``None`` if there is no such method, or the error it
    raised as a string: a report on a failed solver must not fail itself."""

    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"


def _history(solver: object) -> str:
    """A solver's retained convergence history, with arrays as plain lists."""

    history = _try_call(solver, "getConvergenceHistory")
    if not isinstance(history, tuple):
        return repr(history)
    pieces = []
    for entry in history:
        try:
            pieces.append(np.asarray(entry).tolist())
        except Exception:
            pieces.append(repr(entry))
    return repr(tuple(pieces))
