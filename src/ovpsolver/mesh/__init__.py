"""The run's mesh: the domain it covers, and how it is built.

A run builds its own mesh, graded across the diffuse band, because ``domain_eps``
decides what the mesh has to be (:mod:`ovpsolver.mesh.generate`); it is archived,
and every reader afterwards reads the archive. The outer boundary is one of the
closed set of shapes in :mod:`ovpsolver.mesh.shapes`, since ``Sigma`` carries real
boundary conditions and the mesh has to end exactly on it, while the inclusions
are signed distances and may be anything. :mod:`.point_configurations` places
equal inclusions for a geometry file.
"""

from __future__ import annotations

from .generate import MESHERS, graded_mesh, read, target_size, write
from .point_configurations import (
    lloyd,
    log_gas_hard_sphere,
    thinned_ginibre,
)
from .shapes import (
    DOMAIN_TAG,
    OUTER_BOUNDARY_TAG,
    Disk,
    OuterDomain,
    Rectangle,
    RoundedRectangle,
    disk,
    rectangle,
    rounded_rectangle,
)
from .utils import max_cell_diameter

__all__ = [
    "DOMAIN_TAG",
    "Disk",
    "MESHERS",
    "OUTER_BOUNDARY_TAG",
    "OuterDomain",
    "lloyd",
    "log_gas_hard_sphere",
    "thinned_ginibre",
    "Rectangle",
    "RoundedRectangle",
    "disk",
    "graded_mesh",
    "max_cell_diameter",
    "read",
    "rectangle",
    "rounded_rectangle",
    "target_size",
    "write",
]
