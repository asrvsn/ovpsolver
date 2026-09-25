"""How much of the mixture is the Darcy floor carrying, rather than the phase?

Every phase's drag measure is floored at the run's ``(chi phi)_*`` by a smoothed
maximum, so the Darcy coefficient is
``sqrt((chi_eps phi)^2 + (chi phi)_*^2) / D``. The floor keeps the velocity row
solvable where the carrier vanishes -- inside an inclusion, or in a wall the phase
has been expelled from -- and there it is free. Where the phase is still present
and moving it is a cap ``D / (chi phi)_*`` on a mobility that ought to be larger,
and a cap that switches on over part of a surface is a coefficient with the shape
of that region, which the velocity inherits. So the question is not whether the
floor fires, which it always does somewhere, but how far it reaches and whether
the reach grows as the phases separate.

Both fractions are weighted by ``chi_eps``. The floor's indicator is one
throughout every inclusion's interior, where there is no fluid to move, so an
unweighted integral would mostly measure the inclusions:

``of free``
    ``int chi g / int chi``, the share of the free fluid dragged by the floor
    rather than by itself.
``of phase``
    the same numerator against the phase's initial amount ``int chi phi``.
    Larger than ``of free`` by roughly the reciprocal of the phase fraction, and
    the one to read for how much of *this phase* is under its own floor.

``phase left`` is beside them because the floor follows the phase: a rising
floored fraction against a falling amount is depletion, and against a flat amount
redistribution.

The integrals are against the saved quadrature ``chi_eps``, the weight the rows
used (:func:`~ovpsolver.phase_field_system.analyze.base.quadrature_indicator`).
The per-cell picture is the phase's own ``drag_floor_indicator``, a drawable cell
representative of the same quantity; add it to a phase's ``save`` list to see
where the reach sits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import ufl

from ...fem.reduce import assemble_scalar
from ...solver.regularization import smooth_max
from .base import (
    INDICATOR,
    Diagnostic,
    Unanswerable,
    declare_frames,
    frames_of,
    quadrature_indicator,
)
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from dolfinx import fem

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--reach-tol``: the chi-weighted share of the free region a phase's
#: drag floor may carry before the cap is a shape in the coefficient rather than a
#: guard on an empty row. A few per cent is the expelled wall and the far side of
#: the inclusions; a tenth is the floor reaching into the band the surfaces act in.
REACH_TOLERANCE = 0.05


class DragFloor(Diagnostic):
    """Track how far each phase's drag measure floor reaches over a run."""

    name = "analyze.drag_floor"
    summary = "per-phase reach of the Darcy drag measure floor"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        parser.add_argument(
            "--reach-tol",
            type=float,
            default=REACH_TOLERANCE,
            dest="reach_tol",
            help="floored share of the free region that still counts as a guard",
        )
        declare_frames(parser)

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        reach_tol: float = REACH_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        floor = float(parameters.solver.drag_measure_floor)
        phases = [
            phase
            for phase in parameters.phase_field_parameters
            if run.has(f"{phase.name}.phi")
        ]
        if not (phases and floor > 0.0):
            raise Unanswerable(
                "no phase saved 'phi', or solver.drag_measure_floor is zero; "
                "either way there is no floor to measure the reach of. Add 'phi' "
                "to each phase's save list."
            )

        indicator = quadrature_indicator(run)
        if indicator is None:
            raise Unanswerable(
                f"'{INDICATOR}' was not saved, so the free region cannot be "
                "weighed and the floored share would be the inclusions. Add "
                "'indicator' to the diffuse_domain save list."
            )
        measure = ufl.dx(domain=run.mesh)
        free = assemble_scalar(indicator * measure)

        report.field("phases", ", ".join(phase.name for phase in phases))
        report.field("floor", f"{floor:.1e}, solver.drag_measure_floor")
        report.field("free region", f"{free:.4g}, as int chi_eps")
        report.note()

        walk = frames_of(run, frames, stride=stride)
        for phase in phases:
            self._one_phase(
                report, run, phase, floor, indicator, measure, free, reach_tol, walk
            )

    ## Private helpers

    def _one_phase(
        self,
        report: "Report",
        run: "Run",
        phase: Any,
        floor: float,
        indicator: "fem.Function",
        measure: ufl.Measure,
        free: float,
        tolerance: float,
        walk: tuple[int, ...],
    ) -> None:
        times = np.asarray(run.times, dtype=float)
        composition = run.field(f"{phase.name}.phi")
        nominal = assemble_scalar(indicator * composition.sample(0) * measure)

        table = report.table(
            Table(
                f"{phase.name} frame",
                "time",
                "of free",
                "of phase",
                "phase left",
                formats={"of free": ".3%", "of phase": ".3%", "phase left": ".2%"},
            )
        )

        reach: list[float] = []
        for frame in walk:
            values = composition.sample(frame)
            # The phase's drag_floor_indicator, against the quadrature weight.
            deficit = 1.0 - indicator * values / smooth_max(indicator * values, floor)
            floored = assemble_scalar(indicator * deficit * measure)
            amount = assemble_scalar(indicator * values * measure)
            reach.append(floored / free if free else float("nan"))
            table.add(
                frame,
                float(times[frame]),
                reach[-1],
                floored / nominal if nominal else float("nan"),
                amount / nominal if nominal else float("nan"),
            )

        if not reach:
            return
        worst = max(reach)
        report.check(
            worst <= tolerance,
            f"{phase.name}: the floor is guarding an empty row, not capping a live one",
            f"reaches {worst:.2%} of the free region against {tolerance:.0%}, "
            f"from {reach[0]:.2%} at the first frame measured",
        )
