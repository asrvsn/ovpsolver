"""The outer domain a geometry file declares, and how gmsh builds it.

The inclusions are signed distances, because an arbitrary shape has no other
description and the diffuse-domain method needs none (see
:mod:`ovpsolver.diffuse_domain.sdf`). The *outer* boundary is the opposite case:
it is a real boundary with real conditions on it, ``Sigma``, and the mesh has
to end exactly there, which an implicit function cannot give without marching
it first. So it is one of a closed set of shapes, each an :class:`OuterDomain`
that adds itself to a live gmsh model::

    from ovpsolver.mesh import OuterDomain, disk

    def outer_domain(rng_seed: int) -> OuterDomain:
        return disk(radius=1.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType

#: The physical groups a built shape registers, which are what the mesh is read
#: out of gmsh by. The second is ``Sigma``: the curves bounding the domain, not
#: the domain itself.
DOMAIN_TAG = 1
OUTER_BOUNDARY_TAG = 2


class OuterDomain(ABC):
    """A shape gmsh can mesh, whose boundary is ``Sigma``.

    What a geometry file's ``outer_domain()`` returns. Subclass to add a shape;
    the constructors below build the ones this package ships.
    """

    #: Geometric dimension of the shape, which gmsh meshes to and dolfinx reads
    #: back as ``gdim``. Every shape here is planar.
    dimension: int = 2

    @abstractmethod
    def build(self, gmsh: ModuleType) -> int:
        """Add this shape to gmsh's current model, under the physical groups.

        Returns the tag of the domain entity added -- the region gmsh will mesh,
        whose boundary carries :data:`OUTER_BOUNDARY_TAG`.
        """


def _tag_groups(gmsh: ModuleType, domain: int) -> int:
    """Register a built domain entity and its boundary as physical groups."""

    gmsh.model.occ.synchronize()
    curves = [edge for _, edge in gmsh.model.getBoundary([(2, domain)])]
    gmsh.model.addPhysicalGroup(2, [domain], DOMAIN_TAG, name="domain")
    gmsh.model.addPhysicalGroup(
        1, [abs(curve) for curve in curves], OUTER_BOUNDARY_TAG, name="outer_boundary"
    )
    return domain


@dataclass(frozen=True)
class Disk(OuterDomain):
    """A disk of the given radius about its centre."""

    radius: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if float(self.radius) <= 0.0:
            raise ValueError(f"a disk's radius must be positive, not {self.radius}")

    def build(self, gmsh: ModuleType) -> int:
        x, y = self.center
        return _tag_groups(
            gmsh, gmsh.model.occ.addDisk(x, y, 0.0, self.radius, self.radius)
        )


@dataclass(frozen=True)
class Rectangle(OuterDomain):
    """An axis-aligned rectangle, given by its extents about its centre."""

    width: float
    height: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if float(value) <= 0.0:
                raise ValueError(f"a rectangle's {name} must be positive, not {value}")

    def build(self, gmsh: ModuleType) -> int:
        x, y = self.center
        return _tag_groups(
            gmsh,
            gmsh.model.occ.addRectangle(
                x - 0.5 * self.width, y - 0.5 * self.height, 0.0,
                self.width, self.height,
            ),
        )


@dataclass(frozen=True)
class RoundedRectangle(Rectangle):
    """A rectangle whose corners are arcs of the given radius.

    Each arc is tangent to the two sides it joins, so the extents and the centre
    mean what they do on a :class:`Rectangle`. OpenCASCADE fillets the corners
    itself, as exact arcs, and the outline comes out as eight curves, all of
    which :func:`_tag_groups` registers as ``Sigma``.
    """

    #: Keyword-only so that it can be required after the defaulted ``center``.
    corner_radius: float = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        radius = float(self.corner_radius)
        limit = 0.5 * min(float(self.width), float(self.height))
        if radius <= 0.0:
            raise ValueError(
                f"a rounded rectangle's corner_radius must be positive, not "
                f"{self.corner_radius}; a square-cornered rectangle is a "
                f"Rectangle"
            )
        if radius >= limit:
            raise ValueError(
                f"a rounded rectangle's corner_radius must be less than half "
                f"its shorter side ({limit}), not {self.corner_radius}. At half "
                f"the shorter side the two corners along it meet and the side "
                f"vanishes, and beyond it the arcs would have to overlap"
            )

    def build(self, gmsh: ModuleType) -> int:
        x, y = self.center
        return _tag_groups(
            gmsh,
            gmsh.model.occ.addRectangle(
                x - 0.5 * self.width, y - 0.5 * self.height, 0.0,
                self.width, self.height,
                roundedRadius=self.corner_radius,
            ),
        )


def disk(radius: float, center: Sequence[float] = (0.0, 0.0)) -> Disk:
    """A disk of the given radius, centred at the origin unless told otherwise."""

    x, y = (float(one) for one in center)
    return Disk(radius=float(radius), center=(x, y))


def rectangle(
    width: float, height: float, center: Sequence[float] = (0.0, 0.0)
) -> Rectangle:
    """An axis-aligned rectangle of the given extents."""

    x, y = (float(one) for one in center)
    return Rectangle(width=float(width), height=float(height), center=(x, y))


def rounded_rectangle(
    width: float,
    height: float,
    corner_radius: float,
    center: Sequence[float] = (0.0, 0.0),
) -> RoundedRectangle:
    """An axis-aligned rectangle of the given extents, with rounded corners."""

    x, y = (float(one) for one in center)
    return RoundedRectangle(
        width=float(width),
        height=float(height),
        center=(x, y),
        corner_radius=float(corner_radius),
    )


__all__ = [
    "DOMAIN_TAG",
    "Disk",
    "OUTER_BOUNDARY_TAG",
    "OuterDomain",
    "Rectangle",
    "RoundedRectangle",
    "disk",
    "rectangle",
    "rounded_rectangle",
]
