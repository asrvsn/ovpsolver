"""Three equal inclusions spread evenly through a disk of radius 1.2.

A geometry file defines ``outer_domain``, giving the shape the mesh is built on,
and ``signed_distance_functions``, giving one signed distance per inclusion in
the mesh's spatial coordinate. Both are handed the spec's ``system.rng_seed``,
which makes an arrangement drawn from a rule reproducible from the spec, and
varied by changing one number in it.

The centres are generated rather than written out, so the file records which
pattern was wanted instead of one sample of it. Swap
:func:`~ovpsolver.mesh.lloyd` for :func:`~ovpsolver.mesh.thinned_ginibre` to ask
the same question of a disordered arrangement, or for
:func:`~ovpsolver.mesh.log_gas_hard_sphere` to ask it at a stated regularity.
"""

import ufl
from ufl.core.expr import Expr

from ovpsolver.diffuse_domain import sphere
from ovpsolver.mesh import OuterDomain, disk
import ovpsolver.mesh.point_configurations.disk as pcs

#: How many inclusions, and how big each one is.
COUNT = 3
RADIUS = 0.2

#: The disk they are spread through, and so the domain to mesh.
DOMAIN_RADIUS = 1.2

#: Room kept from the boundary, and twice that between inclusion surfaces, in
#: interface widths ``2 eps``: the scale that decides whether two surfaces can
#: be resolved apart at all. Measured against ``eps`` rather than the inclusion
#: radius, so that a spec changing its smearing gets an arrangement that still
#: resolves. The centres are placed for inclusions larger by the gap, since a
#: rule only keeps its spheres inside the domain and clear of each other.
#:
#: This ties the arrangement to the spec: at these extents the relaxed centres
#: stop clearing the boundary between ``eps = 0.07`` and ``0.08``, and the
#: geometry refuses rather than place a surface its band cannot be resolved
#: against. At this count the boundary binds and not the neighbours, which are
#: still more than five interface widths apart there.
GAP_IN_INTERFACE_WIDTHS = 2.0


def outer_domain(rng_seed: int) -> OuterDomain:
    """The domain the mixture occupies, whose boundary is Sigma."""

    return disk(radius=DOMAIN_RADIUS)


def signed_distance_functions(
    rng_seed: int, x: ufl.SpatialCoordinate, eps: float
) -> list[Expr]:
    """One signed distance per inclusion, negative inside it.

    ``eps`` sets only the clearance (:data:`GAP_IN_INTERFACE_WIDTHS`): a sphere
    is exact, so there is nothing to approximate.
    """

    effective_radius = RADIUS + GAP_IN_INTERFACE_WIDTHS * 2.0 * eps
    centers = pcs.lloyd(
        COUNT,
        disk_radius=DOMAIN_RADIUS,
        inclusion_radius=effective_radius,
        rng_seed=rng_seed,
    )
    return [sphere(center, RADIUS)(x) for center in centers]
