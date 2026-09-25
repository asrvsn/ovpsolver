"""Is the saved run the same object the solver held?

The whole offline pipeline rests on one claim: a dof vector written during the
solve, read back later, and put on a space rebuilt from the saved element is the
same function it was. If that fails, every plot and every other diagnostic is
quietly wrong rather than loudly broken, because a permuted dof vector still looks
like a field.

Three things have to line up, and each is checked separately. The mesh in the
directory has to be the mesh the solve ran on, cell for cell and vertex for vertex,
which is why it is written before the solve and read back to define the numbering
rather than generated twice. Each field's element description has to rebuild the
space it came from, dof for dof. And each field the mixture declares has to be
findable under the name the mixture calls it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...fem.reduce import owned_dofs
from .base import Diagnostic
from .report import Table

if TYPE_CHECKING:
    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from ..system import PhaseFieldSystem
    from .report import Report

#: Coordinates and dof values are compared exactly, not to a tolerance. They
#: went to disk as float64 and came back as float64, and anything other than
#: equality means something reordered them.
EXACT = 0.0


class Roundtrip(Diagnostic):
    """Check the saved mesh, elements and names against a freshly built system.

    Builds the mixture from the same spec the run used and compares what it gets
    against what is on disk. Serial only: the saved arrays are indexed by local
    dof, so the question has no meaning across ranks.
    """

    name = "analyze.roundtrip"
    summary = "check that a saved run reloads as the mesh and fields it was"

    ## Overrides

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        system_class: "type[PhaseFieldSystem]",
        **options: Any,
    ) -> None:
        self._mesh(report, run, parameters)
        self._elements(report, run)
        self._declarations(report, run, system_class, parameters)

    ## Private helpers

    def _mesh(
        self, report: "Report", run: "Run", parameters: "PhaseFieldSystemParameters"
    ) -> None:
        """The mesh on disk against the one the parameters hand the solver."""

        saved, live = run.mesh, parameters.solver.dolfinx_mesh
        report.field("mesh file", run.path / "mesh.xdmf")

        dimensions = (saved.topology.dim, saved.geometry.dim)
        if not report.check(
            dimensions == (live.topology.dim, live.geometry.dim),
            "mesh dimensions agree",
            f"saved {dimensions}, built "
            f"{(live.topology.dim, live.geometry.dim)}",
        ):
            return

        saved_x, live_x = saved.geometry.x, live.geometry.x
        if not report.check(
            saved_x.shape == live_x.shape,
            "vertex counts agree",
            f"saved {saved_x.shape[0]}, built {live_x.shape[0]}",
        ):
            return
        drift = float(np.abs(saved_x - live_x).max())
        report.check(
            drift <= EXACT,
            "vertices are in the same order, at the same coordinates",
            f"largest disagreement {drift:.3e}",
        )

        saved_cells = _cells(saved)
        live_cells = _cells(live)
        report.check(
            saved_cells.shape == live_cells.shape
            and bool(np.array_equal(saved_cells, live_cells)),
            "cells are in the same order, on the same vertices",
            f"saved {saved_cells.shape}, built {live_cells.shape}",
        )

    def _elements(self, report: "Report", run: "Run") -> None:
        """Every saved element rebuilds a space of exactly its own size."""

        table = report.table(Table("field", "element", "saved", "rebuilt", "frames"))
        for name in run.names:
            field = run.field(name)
            saved = int(field.values.shape[1])
            rebuilt = owned_dofs(field.space)
            report.check(
                saved == rebuilt,
                f"{name} rebuilds to its saved size",
                f"{saved} on disk, {rebuilt} in {field.element.describe()}",
            )
            table.add(
                name,
                field.element.describe(),
                saved,
                rebuilt,
                # Spelled out rather than shown as its one row, which beside a
                # column of hundreds reads as a run that died immediately.
                "static" if field.static else field.n_frames,
            )

    def _declarations(
        self,
        report: "Report",
        run: "Run",
        system_class: "type[PhaseFieldSystem]",
        parameters: "PhaseFieldSystemParameters",
    ) -> None:
        """Everything the spec asked to save is in the directory under that name.

        Asked of the built system rather than of the spec text, so that a name
        the mixture resolves differently from how it is written is caught here
        rather than by whichever plot went looking for it.
        """

        declared = system_class(parameters).saved_names()
        missing = [name for name in declared if not run.has(name)]
        extra = [name for name in run.names if name not in declared]
        report.check(
            not missing,
            "every field the mixture saves is in the directory",
            ", ".join(missing) if missing else f"{len(declared)} fields",
        )
        report.check(
            not extra,
            "the directory holds nothing the mixture does not declare",
            ", ".join(extra) if extra else "",
        )


def _cells(mesh: Any) -> np.ndarray:
    """The cell-to-vertex connectivity, as a dense array of vertex indices."""

    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, 0)
    connectivity = mesh.topology.connectivity(dim, 0)
    n_cells = mesh.topology.index_map(dim).size_local
    return np.asarray(connectivity.array, dtype=np.int64).reshape(n_cells, -1)
