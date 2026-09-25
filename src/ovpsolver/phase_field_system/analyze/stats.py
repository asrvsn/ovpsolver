"""Summary statistics of one saved field, frame by frame.

The general-purpose diagnostic, for when the question is still being formed.

Statistics are taken where the mixture is, not over the whole square. A field
inside an inclusion is whatever the diffuse-domain regularization left there, and
averaging it in makes every number smaller for no reason, so by default the
indicator masks the sample. ``--raw`` counts everything, and the report always
states which it did: a statistic whose support is ambiguous is worse than none.
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


class Stats(Diagnostic):
    """Per-frame min, mean, max and RMS of a saved field, one value per cell.

    A vector or tensor field is reported by its magnitude, since its components
    depend on the frame they are written in and the magnitude does not.
    """

    name = "analyze.stats"
    summary = "summary statistics of one saved field over the run"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        parser.add_argument(
            "field", help="the saved field, by its full name, e.g. water.phi"
        )
        parser.add_argument(
            "--raw",
            action="store_true",
            help="count every cell, including those inside the inclusions",
        )
        declare_frames(parser)

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        field: str,
        raw: bool = False,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        if not run.has(field):
            raise Unanswerable(
                f"{field!r} was not saved; this run has {', '.join(run.names)}"
            )
        saved = run.field(field)
        report.field("field", f"{field}, {saved.element.describe()}")

        indicator = inclusion_indicator(run)
        mask = np.ones(indicator.size, dtype=bool) if raw else indicator > FREE_FLUID
        report.field("counted", f"{int(mask.sum())} of {mask.size} cells")
        report.field(
            "support",
            "every cell" if raw else f"outside the inclusions (chi > {FREE_FLUID})",
        )
        report.note()

        times = np.asarray(run.times, dtype=float)
        table = report.table(
            Table("frame", "time", "n", "min", "mean", "max", "rms")
        )
        for frame in frames_of(run, frames, stride=stride):
            values = saved.cell_values(frame)
            if saved.shape:
                values = np.linalg.norm(values.reshape(len(values), -1), axis=1)
            sample = values[mask]
            table.add(
                frame,
                float(times[frame]),
                int(sample.size),
                float(sample.min()),
                float(sample.mean()),
                float(sample.max()),
                float(np.sqrt(np.mean(sample**2))),
            )
