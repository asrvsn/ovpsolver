"""Build the run's mesh from its geometry, graded across the diffuse band.

The mesh is generated rather than given because ``domain_eps`` decides what it
has to be. How well the diffuse surface terms are resolved is not a detail of
accuracy: on a fixed mesh with only ``eps`` varying, the normal-velocity leak
through an inclusion carries an ``m = 6`` modulation of 8 to 12 per cent, seven
times its neighbouring modes, while ``eps`` spans four cells (eight across the
``2 eps`` width), and 0.5 per cent, no more than its neighbours, once it spans
eight. Aland, Lowengrub & Voigt (2010) sec. 3.2 report the same threshold at
about five points across the layer. A mesh handed over as a file cannot be
checked against a width it was built without knowing, so the width and the mesh
are decided together.

Graded, because the criterion is local. Eight cells per ``eps`` everywhere costs
four times the cells for a condition that only bites within a couple of widths
of a surface, and grading reaches the same resolution in the band for about 38
per cent of that count. The inclusions are stationary, so the grading is built
once, before the run.

The size field takes two passes. It has to exist before the mesh does, but the
geometry is UFL in a mesh's spatial coordinate, and UFL evaluated off an abstract
``ufl.Mesh`` costs about 220 microseconds a point -- minutes for a background
grid. So the domain is first meshed uniformly at ``mesh_h``, the target size is
interpolated onto that mesh through the compiled expression path, exactly at its
nodes, and the nodal values go back to gmsh as a background view for the second
pass. The first pass only has to place the fine cells, and at ``mesh_h`` it
already has a few cells across the band.

One trap fails silently: ``Mesh.MeshSizeMin`` and ``Mesh.MeshSizeMax`` are
global options, and the uniform pass sets both to ``mesh_h``. Left alone they
clamp the background field flat, and the second pass returns a mesh identical to
the first, with no error anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI

from ..fem.reduce import global_sum

if TYPE_CHECKING:
    from types import ModuleType

    from dolfinx.mesh import Mesh
    from ufl.core.expr import Expr

    from .shapes import OuterDomain

logger = logging.getLogger(__name__)

#: gmsh's 2D algorithm numbers, by the name a spec uses.
#:
#: ``frontal_delaunay`` makes triangles so regular that the mesh is an oriented
#: lattice (:func:`~ovpsolver.mesh.utils.lattice_coherence` near one, against a
#: few hundredths for the others), and a noiseless quench grows the lattice's
#: error into hexagonal droplet patterns which are entirely an artefact. It is
#: here to provoke the artefact deliberately, and is the wrong default for a real
#: run.
MESHERS = {"delaunay": 5, "frontal_delaunay": 6, "mesh_adapt": 1}

#: Full refinement within ``NEAR_WIDTHS * eps`` of a surface, the bulk size beyond
#: ``FAR_WIDTHS * eps``, graded logarithmically between, as a mesher grades anyway.
#: In ``eps`` because ``dGamma_eps`` goes as ``sech^2(3 r / eps)``, down to one
#: per cent of its peak at one ``eps`` and under a part in ten thousand at two.
#: The far figure is how far out the diagnostics read shells, so that what is
#: measured is resolved as well as what is solved. Not spec parameters.
NEAR_WIDTHS = 1.5
FAR_WIDTHS = 4.0


def target_size(
    distances: Sequence[Expr],
    domain_eps: float,
    fine: float,
    bulk: float,
) -> Expr:
    """The requested cell size, as UFL in whatever coordinate ``distances`` use.

    ``min_a |r_a|`` is the distance to the nearest inclusion surface, so
    overlapping bands, unequal shapes and any number of inclusions need no
    special case.
    """

    if not distances:
        raise ValueError("a size field needs at least one signed distance")
    if not 0.0 < fine <= bulk:
        raise ValueError(
            f"the refined size {fine} has to be positive and no larger than the "
            f"bulk size {bulk}"
        )
    if fine == bulk:
        raise ValueError(
            "a size field with nothing to grade towards is a uniform mesh; ask "
            "graded_mesh for one rather than building a constant field"
        )

    nearest = reduce(ufl.min_value, [abs(one) for one in distances])
    ramp = (nearest / float(domain_eps) - NEAR_WIDTHS) / (
        FAR_WIDTHS - NEAR_WIDTHS
    )
    clamped = ufl.max_value(0.0, ufl.min_value(1.0, ramp))
    return fine * ufl.exp(clamped * ufl.ln(bulk / fine))


def graded_mesh(
    *,
    outer: OuterDomain,
    distances_of: Callable[[Mesh], Sequence[Expr]],
    domain_eps: float,
    mesh_h: float,
    min_elements_across_interface_width: int,
    mesh_algorithm: str = "delaunay",
    comm: MPI.Comm | None = None,
) -> Mesh:
    """The run's mesh: ``mesh_h`` everywhere, refined across the diffuse band.

    The interface is ``2 domain_eps`` wide, so the band is refined to
    ``2 eps / min_elements_across_interface_width``, and never coarser than
    ``mesh_h``.

    ``distances_of`` returns a mesh's signed distances, as
    :func:`~ovpsolver.diffuse_domain.sdf.signed_distances` bound to a geometry
    file does. A callable rather than expressions, because the two passes are two
    meshes and the expressions belong to whichever one is in hand.
    """

    # Not at the top: diffuse_domain.parameters imports this module.
    from ..diffuse_domain.sdf import evaluate

    if mesh_algorithm not in MESHERS:
        raise ValueError(
            f"mesh_algorithm must be one of {sorted(MESHERS)}, not "
            f"{mesh_algorithm!r}"
        )
    if mesh_h <= 0.0:
        raise ValueError(f"mesh_h must be positive, not {mesh_h}")
    if min_elements_across_interface_width < 1:
        raise ValueError(
            f"min_elements_across_interface_width must be at least one, not "
            f"{min_elements_across_interface_width}"
        )
    if comm is None:
        comm = MPI.COMM_WORLD

    interface_width = 2.0 * float(domain_eps)
    asked = int(min_elements_across_interface_width)
    fine = min(interface_width / asked, float(mesh_h))
    supplied = interface_width / float(mesh_h)
    with gmsh_session() as gmsh:
        coarse = _uniform(gmsh, outer, mesh_h, mesh_algorithm, comm)
        if fine >= mesh_h:
            # ``mesh_h`` already holds the asked-for number of cells across the
            # width, which is how a uniform mesh is requested.
            logger.info(
                "mesh not graded: mesh_h=%.4e already spans 2*eps=%.4e with %.1f "
                "cells, which meets the %d asked for (%d cells)",
                float(mesh_h),
                interface_width,
                supplied,
                asked,
                _cells(coarse),
            )
            return coarse
        distances = distances_of(coarse)
        if not distances:
            # Nothing inside the outer shape, so no diffuse surface to resolve.
            logger.info(
                "mesh not graded: this geometry declares no inclusions, so "
                "there is no surface to refine towards (%d cells at mesh_h=%.4e)",
                _cells(coarse),
                float(mesh_h),
            )
            return coarse
        # At the vertices, because gmsh reads the background view as piecewise
        # linear: a cell-constant field would step the size at every facet
        # rather than ramp it.
        sizes = evaluate(
            target_size(distances, domain_eps, fine, mesh_h),
            fem.functionspace(coarse, ("Lagrange", 1)),
        )
        graded = _graded(gmsh, outer, sizes, coarse, mesh_h, mesh_algorithm, comm)
        logger.info(
            "mesh graded to %.4e in the band: %d cells across 2*eps=%.4e, where "
            "mesh_h=%.4e alone spans it with %.1f. %d cells, up from %d uniform",
            fine,
            asked,
            interface_width,
            float(mesh_h),
            supplied,
            _cells(graded),
            _cells(coarse),
        )
        return graded


@contextmanager
def gmsh_session() -> Iterator[ModuleType]:
    """gmsh, silenced, and finalized afterwards only if this call initialized it."""

    # Not at the top: gmsh loads the CAD kernel, and this module is imported by
    # everything that reads a spec.
    import gmsh

    opened = not gmsh.isInitialized()
    if opened:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        yield gmsh
    finally:
        if opened:
            gmsh.finalize()


def write(mesh: Mesh, output: str | Path) -> None:
    """Archive a generated mesh where a reader will find it."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(mesh.comm, output, "w") as file:
        file.write_mesh(mesh)


def read(path: str | Path) -> Mesh:
    """Read an archived mesh, which is the only way a reader gets one.

    A reader never regenerates the mesh: gmsh is not promised to be reproducible
    across versions or platforms, and every saved dof vector is indexed by the
    archived mesh's cell and node numbering, so a regenerated mesh could silently
    misalign them.
    """

    with XDMFFile(MPI.COMM_WORLD, str(path), "r") as file:
        return file.read_mesh()


def _cells(mesh: Mesh) -> int:
    """How many cells a mesh has, summed over ranks, for a line of log."""

    index = mesh.topology.index_map(mesh.topology.dim)
    return int(global_sum(mesh.comm, index.size_local))


def _uniform(
    gmsh: ModuleType,
    outer: OuterDomain,
    lengthscale: float,
    mesh_algorithm: str,
    comm: MPI.Comm,
) -> Mesh:
    """A uniform mesh of the outer domain, the pass that carries the size field."""

    if comm.rank == 0:
        gmsh.model.add("ovpsolver_uniform")
        outer.build(gmsh)
        gmsh.option.setNumber("Mesh.Algorithm", MESHERS[mesh_algorithm])
        gmsh.option.setNumber("Mesh.MeshSizeMin", lengthscale)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lengthscale)
        gmsh.model.mesh.generate(outer.dimension)
    comm.Barrier()
    mesh = dolfinx_gmsh.model_to_mesh(
        gmsh.model, comm, 0, gdim=outer.dimension
    ).mesh
    if comm.rank == 0:
        gmsh.model.remove()
    return mesh


def _graded(
    gmsh: ModuleType,
    outer: OuterDomain,
    sizes: npt.NDArray[np.float64],
    sampled: Mesh,
    mesh_h: float,
    mesh_algorithm: str,
    comm: MPI.Comm,
) -> Mesh:
    """Mesh the outer domain again, against ``sizes`` as a background field.

    ``sizes`` are nodal values on ``sampled``, indexed by its vertices.
    """

    if comm.rank == 0:
        dim = sampled.topology.dim
        sampled.topology.create_connectivity(dim, 0)
        cells = np.asarray(
            sampled.topology.connectivity(dim, 0).array, dtype=np.int64
        ).reshape(-1, dim + 1)
        points = sampled.geometry.x

        # "ST" is a scalar on a triangle: nine coordinates, then three values.
        data = np.empty((len(cells), 12), dtype=float)
        for axis in range(3):
            data[:, 3 * axis : 3 * axis + 3] = points[cells, axis]
        data[:, 9:12] = sizes[cells]

        gmsh.model.add("ovpsolver_graded")
        outer.build(gmsh)

        view = gmsh.view.add("ovpsolver_target_size")
        gmsh.view.addListData(view, "ST", len(cells), data.reshape(-1))
        field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field, "ViewIndex", gmsh.view.getIndex(view))
        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        # Every other source of size off, or gmsh takes the smaller of the field
        # and its own guess at the boundary and refines the whole outer edge.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # The trap in the module docstring: both are global, and the uniform pass
        # set them to its own lengthscale.
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(sizes.min()))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(max(sizes.max(), mesh_h)))
        gmsh.option.setNumber("Mesh.Algorithm", MESHERS[mesh_algorithm])
        gmsh.model.mesh.generate(outer.dimension)
    comm.Barrier()
    mesh = dolfinx_gmsh.model_to_mesh(
        gmsh.model, comm, 0, gdim=outer.dimension
    ).mesh
    if comm.rank == 0:
        gmsh.view.remove(gmsh.view.getTags()[-1])
        gmsh.model.remove()
    return mesh


__all__ = [
    "FAR_WIDTHS",
    "MESHERS",
    "NEAR_WIDTHS",
    "graded_mesh",
    "read",
    "target_size",
    "write",
]
