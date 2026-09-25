"""What is in this run: the fields, their elements, and the times they cover.

The first thing to ask of an output directory, and what says which of the other
diagnostics can be asked at all: one screen, no computation, and the difference
between "the plot says that field is missing" and "the spec did not save it".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...fem.reduce import owned_cells, owned_dofs
from .base import Diagnostic
from .report import Table

if TYPE_CHECKING:
    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report


class Fields(Diagnostic):
    """List the saved fields, the element each is dofs of, and the frame times.

    Doubles as the cheapest check that the directory is intact: every field named
    in the metadata is opened, and its dof count is compared against the space its
    element rebuilds to.
    """

    name = "analyze.fields"
    summary = "list what a run saved, and check each field still loads"

    ## Overrides

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        **options: Any,
    ) -> None:
        times = np.asarray(run.times, dtype=float)
        mesh = run.mesh
        report.field("mesh", f"{mesh.topology.dim}D, {owned_cells(mesh)} cells")
        if times.size:
            report.field("time", f"{times[0]:.6g} to {times[-1]:.6g}")
        report.note()

        table = report.table(
            Table("field", "element", "dofs/frame", "how", "min", "max")
        )
        for name in run.names:
            field = run.field(name)
            values = np.asarray(field.values, dtype=float)
            saved, expected = int(values.shape[1]), owned_dofs(field.space)
            report.check(
                saved == expected,
                f"{name} fills its space",
                f"{saved} saved into {expected} of {field.element.describe()}",
            )
            table.add(
                name,
                field.element.describe(),
                saved,
                field.mode,
                float(values.min()) if values.size else None,
                float(values.max()) if values.size else None,
            )
