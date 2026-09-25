"""Measurements of a mesh."""

from __future__ import annotations

import numpy as np
from dolfinx import cpp
from dolfinx import mesh as dmesh

from ..fem.reduce import global_max


def max_cell_diameter(mesh: dmesh.Mesh) -> float:
    """The largest cell diameter, which is the ``h`` a mesh bound is stated in.

    The whole mesh's, not a rank's: an ``h``-dependent stability threshold is a
    property of the discretization and every rank has to compute the same one.
    """

    dim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
    local = float(cpp.mesh.h(mesh._cpp_object, dim, cells).max()) if cells.size else 0.0
    return global_max(mesh.comm, local)


def lattice_coherence(mesh: dmesh.Mesh) -> tuple[float, float]:
    """How much of one global orientation a triangulation has, and along which axis.

    A facet taken either way round is the same facet, so facet directions live on
    a half turn, and a triangular lattice puts them in three directions sixty
    degrees apart -- a six-fold pattern. The mean of ``exp(-6i theta)`` over the
    interior facets' directions is therefore near zero for a mesh whose
    irregularity has no direction to it, and near one for a single oriented
    lattice. Returned as that magnitude and the lattice's facet axis, in radians.

    Worth reporting because the upwinded cell-constant transport is sensitive to
    it in a way a continuous scheme is not: its O(h) numerical diffusion depends
    on the flow direction, which on a mesh with no global orientation acts as
    incoherent noise, and on an oriented one adds up into a standing six-fold
    forcing, phase-locked to this axis, that any instability then amplifies.
    ``frontal_delaunay`` reaches about 0.97 here where every other gmsh
    algorithm stays under 0.09.
    """

    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim - 1, dim)
    mesh.topology.create_connectivity(dim - 1, 0)
    facet_cells = mesh.topology.connectivity(dim - 1, dim)
    facet_vertices = mesh.topology.connectivity(dim - 1, 0)
    points = mesh.geometry.x

    interior = [
        facet
        for facet in range(mesh.topology.index_map(dim - 1).size_local)
        if len(facet_cells.links(facet)) == 2
    ]
    if not interior or mesh.geometry.dim != 2:
        return 0.0, 0.0
    ends = np.array([points[facet_vertices.links(facet)][:, :2] for facet in interior])
    edges = ends[:, 1] - ends[:, 0]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    coefficient = np.mean(np.exp(-6j * angles))
    return float(np.abs(coefficient)), float(np.angle(coefficient) / 6.0)
