"""How converged a rate solve is, measured in the metric the problem comes with.

A Newton solve is normally stopped on ``|F|``, the Euclidean norm of the
residual, which treats the rate problem as an unweighted least squares: a
velocity error in a stiff region weighs exactly as much as the same error in a
slack one, and a change of mesh or basis changes the number while the physics
does not. The problem already carries a better norm. The excess Rayleighian,

    L = R - R_min = dE/dt + Psi + Psi*,

is the Fenchel-Young gap between the Rayleighian at the current iterate and its
minimum: nonnegative, zero exactly at force balance, and the squared rate error
in the Onsager metric that ``Psi`` defines.

Why this is not an extra solve
------------------------------
``Psi*`` has an infimum over the pressure inside it, and no closed form a
declaration could write in general: the Darcy drag's dual is the pointwise
inverse of the drag matrix contracted with the force, but the Brinkman
viscosity's
(:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.bulk_viscosity`) is
an ``H^-1``-type norm, nonlocal, and the dual of their sum is an infimal
convolution rather than a sum. None of it is needed, because ``R`` is convex in
the rates and Newton already computes what the gap needs. For a functional
stationary at ``S*``,

    L = R(S) - R(S*) = -(1/2) <r, d> + O(|d|^3),   J d = -r,

the classical Newton decrement, exact when ``R`` is quadratic in the rates. The
infimum over the pressure and the boundary multipliers is realized by the linear
solve, since ``J`` is the saddle operator. The cost is one dot product of two
vectors PETSc already holds, whatever dissipation a mixture declares; what makes
``R`` non-quadratic is the energy's dependence on the ``k+1`` state, not ``Psi``.

Relative to what
----------------
``L`` is a power and needs a scale, and the scale is the depth of the minimum.
``R`` is a form the mixture already declares, so the dual potential is measured
rather than derived, ``Psi* = L - R``, and

    L / Psi* = (R - R_min) / |R_min|

is the relative suboptimality of the iterate. Both ends are pinned: at rest
nothing moves, so ``R(0) = 0`` and the ratio is exactly one; at force balance
it is zero. Being quadratic in the rate error, the ratio is the *square* of the
relative error in the rates, so rates good to a part in ``10^n`` want an
``excess_rayleighian_rtol`` of ``10^-2n``.

``Psi/Psi*`` is reported beside it. For a quadratic dissipation the two coincide
at the solution, so it approaching one checks that the measured dual belongs to
the dissipation actually declared; drifting from one says ``R`` and the residual
are not the same functional. It is also the check on a cold start, where
``L/Psi*`` is one identically and says nothing about the decrement's accuracy.

What it is not
--------------
``L`` is a verdict on the *solve* and nothing else. It does not floor at the
convex-splitting deficit, because the Rayleighian it is the excess of is built
from the same split potential as the residual. That deficit is ``bregman``, a
verdict on ``dt``: the split increment majorizes the true one by
``int D_ccv dx_eps``, the Bregman divergence of the energy's concave part
between the two states, numerical dissipation that no amount of Newton
iteration removes. A step whose ``L`` is tiny and whose ``bregman`` is a large
fraction of ``Psi`` is converged and over-damped -- solved well, asked badly.

Sign
----
``<r, d>`` is sign-definite only once the constraint residual is small, since
``d^T J d`` reduces to the positive definite ``d^T A d`` only for a ``d`` in the
null space of the constraint block. An early iterate can therefore report a
negative gap, and with it a non-positive ``Psi*``; both are recorded as they come
out, and a solve is stopped on neither.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from dolfinx import fem
from petsc4py import PETSc

from ..fem.reduce import assemble_scalar
from .instrument import LOGGER_NAME

if TYPE_CHECKING:
    from dolfinx.fem.petsc import NonlinearProblem

    from ..phase_field_system import PhaseFieldSystem
    from .parameters import OVPStepParameters


logger = logging.getLogger(f"{LOGGER_NAME}.onsager")


class ExcessRayleighian:
    """The Onsager-metric convergence measure of one rate problem.

    Attached to a SNES, which then reports the gap at every iterate and, when the
    spec asks, stops on the gap instead of on ``|F|``.
    """

    def __init__(
        self,
        system: "PhaseFieldSystem",
        parameters: "OVPStepParameters",
        entity_maps: tuple | None = None,
    ) -> None:
        self.rtol = parameters.excess_rayleighian_rtol
        self._timestep = system.solver_parameters.dt_live
        # With the solver's entity maps, because these are the declarations the
        # residual compiles: a boundary multiplier lives on the submesh while its
        # constraint is integrated on the parent, so the forms span two meshes.
        #
        # ``R`` is assembled once per Newton iteration, and only to scale the
        # gap, so a run that is not stopping on the gap neither compiles nor
        # assembles it (see :meth:`attach`). ``Psi`` and the deficit cost one
        # assembly each per step and answer a question about ``dt``.
        self._rayleighian = (
            fem.form(system.rayleighian(), entity_maps=entity_maps)
            if self.rtol is not None
            else None
        )
        self._dissipation = fem.form(system.dissipation(), entity_maps=entity_maps)
        self._bregman = fem.form(system.bregman_deficit(), entity_maps=entity_maps)

        #: The gap at the last iterate it could be measured at, and the ``Psi*``
        #: it was measured against. ``None`` before a solve has produced either.
        self.gap: float | None = None
        self.scale: float | None = None
        #: Which iterate :attr:`gap` belongs to, and how many the solve took. A
        #: gap trails the solve by one iterate (see :meth:`_monitor`), so a solve
        #: that converged in a single Newton step reports the gap it started from,
        #: and these two say so.
        self.measured_at = 0
        self.iterations = 0
        #: Every gap of the last solve, oldest first, which is what shows whether
        #: it was converging or wandering.
        self.history: list[float] = []

        self._previous: PETSc.Vec | None = None
        self._previous_rayleighian: float | None = None
        self._stopped = False
        self._floor = 0.0

    ## Attachment

    def attach(self, problem: "NonlinearProblem") -> None:
        """Watch a nonlinear problem's SNES, and stop it on the gap if asked to.

        Always a monitor, since the iteration count is wanted whatever else is.
        The gap itself is a dot product beside the Jacobian that produced the
        vectors, but its scale ``Psi*`` needs ``R`` assembled at every iterate but
        the last, a global assembly per Newton step. So the gap is measured only
        when the spec sets ``excess_rayleighian_rtol``, which ties reporting it to
        stopping on it: a run that will not act on the answer should not pay for
        it every iteration. Reporting only the final ``Psi*`` would still need
        ``R`` at an iterate the solve has already left.
        """

        snes = problem.solver
        snes.setMonitor(self._monitor)
        if self.rtol is not None:
            snes.setConvergenceTest(self._converged)

    def reset(self) -> None:
        """Forget the last solve, so a retry is not read as a continuation."""

        self.history.clear()
        self.gap = self.scale = None
        self.measured_at = self.iterations = 0
        self._stopped = False
        self._previous = None
        self._previous_rayleighian = None

    ## Measurement

    def rayleighian(self) -> float:
        """``R`` at the current rates: the objective the gap is a gap in."""

        return assemble_scalar(self._rayleighian)

    def dissipation(self) -> float:
        """``Psi`` at the current rates: the primal the measured dual is checked
        against."""

        return assemble_scalar(self._dissipation)

    def bregman_deficit(self) -> float:
        """``int D_ccv dx_eps / dt``: the *rate* at which the split over-dissipates.

        ``int D_ccv dx_eps`` is an energy, a divergence between two states;
        divided by the step it is comparable with ``Psi``, and the fraction says
        how much of the dissipation a step reports is the discretization's
        rather than the material's.
        """

        return assemble_scalar(self._bregman) / float(self._timestep.value)

    ## PETSc callbacks

    def _monitor(self, snes: PETSc.SNES, iteration: int, residual_norm: float) -> None:
        """Count the iteration and, when measuring, the gap at the iterate just left.

        PETSc stores the update as ``Y`` with ``X <- X - Y``, so ``Y = J^-1 F``
        and the decrement is ``+(1/2) <F, Y>`` -- with ``Y`` at iteration ``k``
        computed from ``F`` at ``k-1``, which is why the previous residual is
        kept rather than the current one used. The result is the gap at the
        iterate just left, exact rather than approximate and one step stale;
        since Newton is converging, stopping on it errs towards one iteration too
        many. ``R`` is carried across the same way: it is the ``R`` of the
        iterate the gap belongs to that makes ``L - R`` the dual potential, and
        pairing this iteration's with the previous gap would report a scale too
        large by one Newton step's descent -- the direction that stops a solve
        early.
        """

        self.iterations = iteration
        if self._rayleighian is None:
            return
        if iteration == 0:
            self._previous = None
            self._previous_rayleighian = None

        measured = None
        if self._previous is not None and self._previous_rayleighian is not None:
            gap = 0.5 * self._previous.dot(snes.getSolutionUpdate())
            measured = (gap, gap - self._previous_rayleighian)
        # Carried forward for the next call alone, and ``R`` costs a global
        # assembly. PETSc settles the convergence reason before it calls the
        # monitor, so a call that sees one is the last and nothing reads what it
        # would leave behind.
        if snes.getConvergedReason() == 0:
            residual = snes.getFunction()[0]
            self._previous_rayleighian = self.rayleighian()
            if self._previous is None:
                self._previous = residual.duplicate()
            residual.copy(self._previous)
        if measured is None:
            return

        self.gap, self.scale = (float(value) for value in measured)
        self.measured_at = iteration - 1
        self.history.append(self.gap)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "excess_rayleighian it=%3d L=%+.6e Psi*=%.6e L/Psi*=%s "
                "Psi/Psi*=%s |F|=%.6e",
                iteration - 1,
                self.gap,
                self.scale,
                _ratio(self.gap, self.scale),
                _ratio(self.dissipation(), self.scale),
                residual_norm,
            )

    def _converged(
        self, snes: PETSc.SNES, iteration: int, norms: tuple[float, float, float]
    ) -> int:
        """Stop on the excess Rayleighian, or on what PETSc would have stopped on.

        PETSc's default test is reproduced rather than delegated to, since
        petsc4py cannot call it: the NaN guard that keeps a diverging solve from
        running to its iteration cap, and the absolute, relative and step tests.
        ``max_it`` is not here: the SNES loop enforces that itself whatever this
        returns.
        """

        reason = PETSc.SNES.ConvergedReason
        solution_norm, step_norm, residual_norm = norms
        rtol, atol, stol, _ = snes.getTolerances()

        if not np.isfinite(residual_norm):
            return reason.DIVERGED_FUNCTION_NANORINF
        if iteration == 0:
            self._floor = residual_norm * rtol
            return reason.ITERATING
        if residual_norm < atol:
            return reason.CONVERGED_FNORM_ABS
        if residual_norm < self._floor:
            return reason.CONVERGED_FNORM_RELATIVE
        if step_norm < stol * solution_norm:
            return reason.CONVERGED_SNORM_RELATIVE

        if self.gap is not None and self.gap >= 0.0 and (self.scale or 0.0) > 0.0:
            if self.gap <= self.rtol * self.scale:
                self._stopped = True
                logger.info(
                    "excess_rayleighian converged it=%d L=%.6e Psi*=%.6e L/Psi*=%s",
                    iteration - 1,
                    self.gap,
                    self.scale,
                    _ratio(self.gap, self.scale),
                )
                return reason.CONVERGED_FNORM_RELATIVE
        return reason.ITERATING

    ## Reporting

    @property
    def ratio(self) -> float | None:
        """``L/Psi*`` as a number, for a caller that will format it itself.

        ``None`` where :func:`_ratio` prints ``n/a``: no gap measured yet, or a
        measured ``Psi*`` that came out non-positive. Both mean there is no scale
        to divide by, which is different from a ratio of zero.
        """

        if self.gap is None or not (self.scale or 0.0) > 0.0:
            return None
        return self.gap / self.scale

    def summary(self) -> str:
        """One line for the step log, in the ratios that mean something.

        ``L/Psi*`` is how far from stationary the rates are, as a fraction of the
        descent that was available; ``Psi/Psi*`` should approach one; and
        ``bregman/Psi`` is how much of the dissipation the step reports is the
        discretization's rather than the material's, taken against the primal,
        the dissipation the step actually did. The iterate the gap was measured
        at is stated because it trails the solve by one: a large ``L/Psi*`` at
        ``it 0/1`` is a step solved in one Newton iteration, not one that failed.
        """

        dissipation = self.dissipation()
        deficit = _ratio(self.bregman_deficit(), dissipation)
        if self.gap is None or self.scale is None:
            # No ``R`` was assembled to scale a gap against. What is left still
            # answers the question about ``dt``.
            return (
                f"L=unmeasured it {self.iterations} Psi={dissipation:.6e} "
                f"bregman/Psi={deficit}"
            )
        stopped = " (stopped on L)" if self._stopped else ""
        return (
            f"L={self.gap:+.3e} L/Psi*={_ratio(self.gap, self.scale)} "
            f"it {self.measured_at}/{self.iterations} "
            f"Psi*={self.scale:.6e} Psi/Psi*={_ratio(dissipation, self.scale)} "
            f"bregman/Psi={deficit}{stopped}"
        )


def _ratio(value: float, scale: float) -> str:
    """``value / scale``, or ``n/a`` when there is no scale to divide by.

    None at rest, where nothing is dissipating and a ratio would report the
    floating-point noise in a quotient of two zeros; nor for a measured ``Psi*``
    that came out non-positive, which the saddle structure allows on an iterate
    far from feasible.
    """

    return f"{value / scale:.3e}" if scale > 0.0 else "n/a"
