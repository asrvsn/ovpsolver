"""Collapsing distributed objects to numbers every rank agrees on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dolfinx import fem, la
from mpi4py import MPI

if TYPE_CHECKING:
    import ufl
    from dolfinx.mesh import Mesh

    from .types import Function, FunctionSpace


def assemble_nodal_values(form: "ufl.Form | fem.Form") -> np.ndarray:
    """A linear form's assembled vector, as a detached array.

    A rank assembles only its own cells' share of each row, so a row is only
    complete once the shared entries are summed (``scatter_reverse``); every
    caller wants the complete row, so the two steps are one.
    """

    vector = fem.assemble_vector(form if isinstance(form, fem.Form) else fem.form(form))
    vector.scatter_reverse(la.InsertMode.add)
    return np.asarray(vector.array, dtype=np.float64).copy()


def assemble_scalar(form: "ufl.Form | fem.Form") -> float:
    """A scalar form's value over the whole mesh, summed across ranks.

    dolfinx assembles a rank's own cells and no more, so an integral is only an
    integral once the ranks have been added up.
    """

    compiled = form if isinstance(form, fem.Form) else fem.form(form)
    return global_sum(compiled.mesh.comm, fem.assemble_scalar(compiled))


def global_min(comm: MPI.Comm, value: float) -> float:
    """Minimum scalar value over all MPI ranks."""

    return float(comm.allreduce(float(value), op=MPI.MIN))


def global_max(comm: MPI.Comm, value: float) -> float:
    """Maximum scalar value over all MPI ranks."""

    return float(comm.allreduce(float(value), op=MPI.MAX))


def global_sum(comm: MPI.Comm, value: float) -> float:
    """Sum scalar value over all MPI ranks."""

    return float(comm.allreduce(float(value), op=MPI.SUM))


def global_all(comm: MPI.Comm, value: bool) -> bool:
    """Whether a statement holds on every MPI rank.

    What a check that raises has to agree on first: a rank that raised alone
    would leave the others waiting in the next collective.
    """

    return bool(comm.allreduce(bool(value), op=MPI.LAND))


def owned_cells(mesh: "Mesh") -> int:
    """Cells this rank owns, ghosts excluded."""

    return int(mesh.topology.index_map(mesh.topology.dim).size_local)


def owned_dofs(space: "FunctionSpace") -> int:
    """Degrees of freedom this rank owns, ghosts excluded and blocks counted."""

    index_map = space.dofmap.index_map
    return int(index_map.size_local * space.dofmap.index_map_bs)


def owned_function_values(function: "Function") -> np.ndarray:
    """A view of a function's owned, flattened coefficient values."""

    return function.x.array[: owned_dofs(function.function_space)]


def function_min_max(function: "Function") -> tuple[float, float]:
    """Global min and max over a function's owned coefficient values."""

    values = owned_function_values(function)
    local_min = float(values.min()) if values.size else np.inf
    local_max = float(values.max()) if values.size else -np.inf
    comm = function.function_space.mesh.comm
    return global_min(comm, local_min), global_max(comm, local_max)


def function_max(function: "Function") -> float:
    """Global max over a function's owned coefficient values."""

    values = owned_function_values(function)
    local_max = float(values.max()) if values.size else -np.inf
    return global_max(function.function_space.mesh.comm, local_max)
