"""The letters OVP, drawn in CAD, inside a 2.5 x 1.1 rounded rectangle.

Three separately connected bodies, two of them with a hole -- which is bulk like
any other part of the domain, since an inclusion is whatever the signed distance
says is inside it and a hole is simply not inside. The source is ``ovp.step``,
exported from Fusion as three solids extruded 0.1 from their footprints; only the
footprints are read and the height is discarded.

The file is refitted before it is read. A STEP export of drawn letterforms is
mostly B-splines, and a B-spline has no closed-form distance function -- its
closest point is a root find per query point, which cannot be written as UFL. So
:func:`~ovpsolver.diffuse_domain.sdf.fit_step_to_arcs` rewrites every boundary
piece as a line or a circular arc, both of which do have one, and puts the result
in ``ovp_fit.step`` beside the source. That file is what the distances are built
from, and it is a STEP file like any other: open it in the modeller and check the
fit against the original rather than taking it on trust.

Each body then becomes one signed distance, exact in the pieces it is built from:
the distance to a bounded segment or arc is a closed form, and the sign is a
crossing count, so a hole is just an inner loop that counts the other way and no
boolean appears anywhere.
"""

from pathlib import Path

import ufl
from ufl.core.expr import Expr

from ovpsolver.diffuse_domain.sdf import fit_outputs, sdfs_from_step
from ovpsolver.mesh import OuterDomain, rounded_rectangle

#: Extents of the domain, about its centre at the origin. The letters span about
#: 2.10 x 0.67, so they clear the boundary by about 0.19 on every side.
WIDTH = 2.5
HEIGHT = 1.1

#: Radius of the four corner arcs. Under half the shorter side, so the short
#: sides keep a straight stretch between their two corners.
CORNER_RADIUS = 0.25

#: How far the refit may move the boundary, as a fraction of the run's smearing
#: parameter. The diffuse method resolves a boundary over ``2 eps``, so this is
#: the only scale the tolerance means anything against -- and ``eps`` arrives
#: from the spec rather than being restated here, so the two cannot drift apart.
FIT_TOLERANCE_IN_EPS = 1.0 / 20.0

#: Beside this file rather than relative to wherever a run was launched from,
#: since the two travel together and the spec names only this file.
SOURCE = Path(__file__).with_name("ovp.step")


def assets() -> list[Path]:
    """The files this geometry reads, so a run archives them beside itself.

    The CAD source, and the refit derived from it where that already exists. The
    fit is deterministic, so archiving it is not what makes the run reproducible:
    it is what keeps reading a run from writing into it, as refitting on first
    plot would.
    """

    return [SOURCE, *fit_outputs(SOURCE)]


def outer_domain(rng_seed: int) -> OuterDomain:
    """The domain the mixture occupies, whose boundary is Sigma."""

    return rounded_rectangle(
        width=WIDTH, height=HEIGHT, corner_radius=CORNER_RADIUS
    )


def signed_distance_functions(
    rng_seed: int, x: ufl.SpatialCoordinate, eps: float
) -> list[Expr]:
    """One signed distance per body of the CAD file, negative inside it."""

    return [
        distance(x)
        for distance in sdfs_from_step(
            SOURCE, eps=eps, tolerance_in_eps=FIT_TOLERANCE_IN_EPS
        )
    ]
