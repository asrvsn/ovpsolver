"""Do the phases still fill the space, and is the departure the one we expect?

Saturation is the constraint the pressure is there to enforce, and the DG0 scheme
makes a specific claim about it. The constraint row is written on the *same* upwind
facet fluxes as the phases' transport rows, so a cell's phase sum changes only by
the net of numerical fluxes across its own facets, each one number shared with the
neighbour it leaves into. The sum is therefore preserved identically -- not to the
order of a quadrature rule, which is all a formulation in the continuous divergence
could offer.

The failure this is here to see: the transport rows substitute the prescribed
injection for the flux on the outer boundary rather than integrating one, so a
constraint row that integrates a live flux there disagrees with them on exactly the
cells touching the wall, and the defect spreads inward, growing geometrically and
indifferent to ``snes_rtol``. The tolerance catches it once it has grown past the
compression a finite dilatational viscosity permits each frame.

Three things keep a run from reporting exactly one. The pressure holds saturation
through a finite dilatational viscosity
(:meth:`~ovpsolver.phase_field_system.PhaseFieldSystem.compression_penalty`), so the
volumetric rate is ``Phi p / eta_b`` rather than zero and every step leaves that
times the step behind in each cell: a property of the spec, shrinking as
``dilatational_viscosity`` grows. Inside an inclusion there is no free fluid for
saturation to be a statement about, so the residual there is reported and never
checked. And in the free fluid each step leaves whatever the solve left, which
accumulates, making the running total a function of how long the run is rather
than of how well it works -- so the check is on what a single frame *adds*,
measured against the frame before it whatever frames are walked, and the total is
reported beside it.

The compression can also be taken out. Summing the phases' transport rows over a
cell against the pressure's own row, both weighted by the same indicator and the
injections being volume-neutral, every sub-step changes a cell's phase sum by
exactly

    dt (sum phi_prev) p / eta_b,

so whatever of a frame's increment that does not account for is the constraint row
not being held. The ``unexplained`` column is that remainder as a fraction of the
frame's compression, with the saved end-of-frame pressure standing in for every
sub-step's. On a frame of one sub-step the stand-in is exact and the column is the
row's own residual, about ``1e-7`` on a healthy split solve. On a frame of several
it also carries how far the pressure moved within the frame, which it need not do
monotonically, so a few per cent there is the saved data's resolution rather than
the solve's; closing it would need the pressure's time integral over the frame.

Cells and not vertices, throughout: every phase is cell-constant and the constraint
is imposed cell by cell, so the cell values *are* the residual. Sampling onto P1
would L2-fit a step function and spread the inclusions' relaxed residual into the
free fluid around them, which reads as a failure of the very thing being checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    FREE_FLUID,
    Diagnostic,
    Unanswerable,
    declare_frames,
    frames_of,
    inclusion_indicator,
)
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--tol``, applied in the free fluid to what one frame adds. Set for
#: the compression ``dt Phi p / eta_b`` a finite dilatational viscosity leaves each
#: frame -- about ``1e-9 p`` at ``dt = 1e-3`` and ``eta_b = 1e6`` -- with room for
#: pressures of order a hundred on top of the solve's rounding. A run whose
#: viscosity leaves more than this per frame fails here, which is the check saying
#: the mixture is compressing rather than that the solve is wrong.
TOLERANCE = 1.0e-6

#: Default for ``--exact-tol``: what counts as identically saturated, for the
#: column that counts cells past it. Loose by a couple of orders over machine
#: epsilon, since a cell's sum is a few floating-point additions of numbers of
#: order one and the initial condition is projected before it is summed.
EXACT_TOLERANCE = 1.0e-12


class Saturation(Diagnostic):
    """The saturation residual per frame, inside and outside the inclusions."""

    name = "analyze.saturation"
    summary = "check the phases sum to one where the constraint is imposed"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--tol",
            type=float,
            default=TOLERANCE,
            help="how much one frame may add to the free-fluid residual",
        )
        parser.add_argument(
            "--exact-tol",
            type=float,
            dest="exact_tol",
            default=EXACT_TOLERANCE,
            help="what counts as identically saturated, for the cell count",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        tol: float = TOLERANCE,
        exact_tol: float = EXACT_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        phases = [phase.name for phase in parameters.phase_field_parameters]
        missing = [name for name in phases if not run.has(f"{name}.phi")]
        if missing:
            # The sum of a subset is not near one, and its distance from one would
            # be a failure the solve did not commit.
            raise Unanswerable(
                f"{', '.join(missing)} did not save 'phi', and saturation is a "
                f"statement about the whole mixture; add it to every phase"
            )

        # Where the constraint is imposed, on cells so the mask lines up with the
        # cell-constant residual it selects from.
        free_fluid = inclusion_indicator(run) >= FREE_FLUID
        report.field("phases", ", ".join(phases))
        report.field(
            "constraint imposed in",
            f"{int(free_fluid.sum())} of {free_fluid.size} cells, "
            f"chi_eps >= {FREE_FLUID}",
        )
        report.note()

        times = np.asarray(run.times, dtype=float)
        pressure = "system.pressure" if run.has("system.pressure") else None
        table = report.table(
            Table(
                "frame",
                "time",
                "max",
                "rms",
                "added",
                "unexplained",
                "cells past exact",
                "inclusions",
                formats={"unexplained": ".3e"},
            )
        )
        closest = float("inf")
        worst = added_worst = 0.0
        worst_frame = added_frame = 0
        relaxed = 0.0

        for frame in frames_of(run, frames, stride=stride):
            signed = self._residual(run, phases, frame)
            outside = signed[free_fluid]
            inside = signed[~free_fluid] if (~free_fluid).any() else None
            largest = float(np.abs(outside).max()) if outside.size else 0.0
            before = self._residual(run, phases, frame - 1) if frame > 0 else None
            added = 0.0 if before is None else self._added(signed, before, free_fluid)
            unexplained = (
                self._unexplained(
                    run, frame, signed, before, pressure, parameters, free_fluid
                )
                if before is not None and pressure is not None
                else None
            )
            if unexplained is not None and not np.isnan(unexplained):
                closest = min(closest, unexplained)
            if largest > worst:
                worst, worst_frame = largest, frame
            if added > added_worst:
                added_worst, added_frame = added, frame
            if inside is not None:
                relaxed = max(relaxed, float(np.abs(inside).max()))
            table.add(
                frame,
                float(times[frame]),
                largest,
                float(np.sqrt(np.mean(outside**2))) if outside.size else 0.0,
                added,
                unexplained,
                int((np.abs(outside) > exact_tol).sum()),
                float(np.abs(inside).max()) if inside is not None else float("nan"),
            )

        # Frame zero whatever frames were walked, since this is a statement about
        # the state before any step was taken.
        initial_inexact = (
            int((np.abs(self._residual(run, phases, 0)[free_fluid]) > exact_tol).sum())
            if run.n_frames
            else 0
        )
        report.check(
            added_worst <= tol,
            "no frame adds to the saturation residual where it is imposed",
            f"worst frame adds {added_worst:.3e} against {tol:.1e}, at frame "
            f"{added_frame}; accumulated to {worst:.3e} by frame {worst_frame}",
        )
        report.check(
            initial_inexact == 0,
            "the initial condition saturates identically",
            f"{initial_inexact} cell(s) already past {exact_tol:.1e} before any "
            f"step was taken, so the departure is not the rate solve's",
        )
        report.note(
            f"inside the inclusions the sum departs by up to {relaxed:.3e}, "
            f"which is the relaxed constraint and not a failure of it"
        )
        if closest < float("inf"):
            report.note(
                f"the compression the pressure permits accounts for a frame's "
                f"increment to within {closest:.3e} at best: the constraint row's "
                f"residual, on the frames taken in one sub-step"
            )

    ## Private helpers

    def _residual(self, run: "Run", phases: "list[str]", frame: int) -> np.ndarray:
        """``sum(phi) - 1`` in every cell, summed from the phases themselves.

        From each phase's own ``phi`` rather than from a published ``total_phase``,
        so there is nothing between the check and the thing being checked, and a
        mixture which never published a total can still be asked.
        """

        total = None
        for name in phases:
            values = run.field(f"{name}.phi").cell_values(frame)
            total = values if total is None else total + values
        return total - 1.0

    def _added(
        self, residual: np.ndarray, previous: np.ndarray, free_fluid: np.ndarray
    ) -> float:
        """The most any free-fluid cell's signed residual moved since the frame
        before.

        Signed and differenced per cell, so a residual that merely moved around the
        domain does not read as one that grew, and a cancellation between two cells
        does not hide a step that spoiled both.
        """

        change = np.abs(residual - previous)[free_fluid]
        return float(change.max()) if change.size else 0.0

    def _unexplained(
        self,
        run: "Run",
        frame: int,
        residual: np.ndarray,
        before: np.ndarray,
        pressure: str,
        parameters: "PhaseFieldSystemParameters",
        free_fluid: np.ndarray,
    ) -> float:
        """What of a frame's phase-sum increment the permitted compression misses.

        As a fraction of the largest compression where the constraint is imposed,
        so it reads the same whatever the pressure's scale.
        """

        values = run.field(pressure).cell_values(frame)
        dt = float(run.times[frame]) - float(run.times[frame - 1])
        permitted = (
            dt * (before + 1.0) * values / float(parameters.dilatational_viscosity)
        )
        missed = residual - before - permitted
        permitted, missed = permitted[free_fluid], missed[free_fluid]
        scale = float(np.abs(permitted).max()) if permitted.size else 0.0
        return float(np.abs(missed).max()) / scale if scale > 0.0 else float("nan")
