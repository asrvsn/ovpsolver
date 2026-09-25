"""Is each phase still all there?

Conservation says a phase's amount changes only by what crossed a boundary, which
for a closed spec means it does not change at all. The transport row is
conservative by construction -- one shared facet number leaves one cell and enters
the next -- so a drift here is the linear solves inside it and not the formulation.

The amount is the saved field integrated over the run's own mesh, against the
saved quadrature ``chi_eps`` where there is one: the weight the row's accumulation
term carries
(:func:`~ovpsolver.phase_field_system.analyze.base.quadrature_indicator`), and the
one that confines the total to the region the mixture lives in.

Saturation, the mixture's other invariant, is ``analyze.saturation``: a statement
about a constraint row rather than a transport row, imposed only where there is
free fluid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import ufl

from ...fem.reduce import assemble_scalar
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

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--conservation-tol``: relative drift in a phase's total amount,
#: over the whole run, that still counts as conserved. Transport is conservative
#: by construction and then only as exact as the linear solves inside it.
CONSERVATION_TOLERANCE = 1.0e-6


class Conservation(Diagnostic):
    """Track each phase's total amount over the saved frames."""

    name = "analyze.conservation"
    summary = "track per-phase mass over a run"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        parser.add_argument(
            "--conservation-tol",
            type=float,
            default=CONSERVATION_TOLERANCE,
            dest="conservation_tol",
            help="relative drift in a phase's total that still counts as conserved",
        )
        declare_frames(parser)

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        conservation_tol: float = CONSERVATION_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        phases = [phase.name for phase in parameters.phase_field_parameters]
        available = [name for name in phases if run.has(f"{name}.phi")]
        if not available:
            raise Unanswerable(
                f"none of {', '.join(phases)} saved 'phi'; there is nothing to "
                f"integrate. Add it to each phase's save list."
            )

        weight = quadrature_indicator(run)
        report.field("phases", ", ".join(available))
        report.field(
            "integrated against",
            "the whole mesh (no indicator saved)"
            if weight is None
            else f"chi_eps, as {INDICATOR}",
        )
        report.note()

        times = np.asarray(run.times, dtype=float)
        measure = ufl.dx(domain=run.mesh)
        table = report.table(Table("frame", "time", *available))
        totals: dict[str, list[float]] = {name: [] for name in available}
        for frame in frames_of(run, frames, stride=stride):
            amounts = []
            for name in available:
                field = run.field(f"{name}.phi").sample(frame)
                integrand = field if weight is None else weight * field
                amounts.append(assemble_scalar(integrand * measure))
                totals[name].append(amounts[-1])
            table.add(frame, float(times[frame]), *amounts)

        for name, series in totals.items():
            values = np.asarray(series, dtype=float)
            if values.size < 2 or values[0] == 0.0:
                continue
            drift = float(abs(values[-1] - values[0]) / abs(values[0]))
            report.check(
                drift <= conservation_tol,
                f"{name} is conserved across the run",
                f"relative drift {drift:.3e} against {conservation_tol:.1e}",
            )
