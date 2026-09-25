"""The signed distance to a region bounded by line segments and circular arcs.

Exact, and written in UFL: both pieces have a closed-form distance, so the field
is the true distance and ``|grad r| = 1`` holds to round-off. Two separate
problems have to be got right.

**How far.** The distance to a *bounded* piece, never to the line or circle it
lies on. A minimum over unbounded continuations is right for a convex region and
wrong everywhere else, and the error sits exactly at the corners, where a wetting
boundary condition is most interesting. A segment clamps the projection to its
own span, an arc tests whether the point falls within its own sweep, and beyond
either the nearest point is an endpoint.

**Which side.** A crossing count: a ray from the point crosses the boundary an odd
number of times exactly when the point is inside, whatever the shape and however
many loops it has. Holes need no special case, since an inner loop contributes
its crossings like any other.

The crossings are counted against the *chords* and corrected once per arc. A
chord's crossing test is two comparisons on numbers known when the form is built;
an arc's would be a quadratic. The region bounded by the arcs differs from the
polygon through their endpoints by one circular segment per arc, and a point
inside such a segment is on the other side of the boundary from what the polygon
said, whichever way the arc bulges. So each arc contributes one more flip.

Parity is accumulated as a product of ``±1`` rather than a sum modulo two, since
UFL has no modulo and a product of exact signs is exact.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import ufl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ufl.core.expr import Expr

    from .step import Piece


def signed_distance(loops: Sequence[Sequence[Piece]]) -> Callable[[Expr], Expr]:
    """One body's signed distance, as a callable of the spatial coordinate.

    ``loops`` are the body's boundary loops, as
    :func:`~ovpsolver.diffuse_domain.sdf.step.read_pieces` returns them. Their
    orientation is not read, since a crossing count does not care, but each has to
    be closed: a ray escaping through a gap counts one crossing too few and turns
    the body inside out. Returns the same kind of callable
    :func:`~ovpsolver.diffuse_domain.sdf.base.sphere` does.
    """

    pieces = [piece for loop in loops for piece in loop]
    if not pieces:
        raise ValueError("a signed distance needs at least one boundary piece")

    def build(x):
        distance = None
        sign = None
        for piece in pieces:
            here = (
                _arc_distance(piece, x) if piece.is_arc else _segment_distance(piece, x)
            )
            distance = here if distance is None else ufl.min_value(distance, here)
            for flip in _flips(piece, x):
                sign = flip if sign is None else sign * flip
        return sign * distance

    return build


## How far


def _segment_distance(piece: Piece, x) -> Expr:
    """Distance to a bounded segment: the projection, clamped to its own span."""

    start = [float(one) for one in piece.start]
    edge = [float(b - a) for a, b in zip(piece.start, piece.end)]
    squared = edge[0] * edge[0] + edge[1] * edge[1]
    offset = [x[0] - start[0], x[1] - start[1]]
    along = (offset[0] * edge[0] + offset[1] * edge[1]) / squared
    clamped = ufl.max_value(0.0, ufl.min_value(1.0, along))
    gap = [offset[0] - clamped * edge[0], offset[1] - clamped * edge[1]]
    return ufl.sqrt(gap[0] * gap[0] + gap[1] * gap[1])


def _arc_distance(piece: Piece, x) -> Expr:
    """Distance to a bounded arc: the radial gap within its sweep, else an endpoint.

    Taken in the arc's own frame, folded about its bisector, so "within the
    sweep" is one comparison against the half-angle and needs no inverse tangent.
    """

    center = [float(one) for one in piece.center]
    half = 0.5 * piece.sweep()
    bisector, normal = _frame(piece)
    offset = [x[0] - center[0], x[1] - center[1]]

    along = offset[0] * bisector[0] + offset[1] * bisector[1]
    across = abs(offset[0] * normal[0] + offset[1] * normal[1])
    radius = ufl.sqrt(offset[0] * offset[0] + offset[1] * offset[1])

    to_start = _to_point(x, piece.start)
    to_end = _to_point(x, piece.end)
    return ufl.conditional(
        ufl.ge(along * math.sin(half) - across * math.cos(half), 0.0),
        abs(radius - float(piece.radius)),
        ufl.min_value(to_start, to_end),
    )


def _frame(piece: Piece) -> tuple[list[float], list[float]]:
    """The arc's bisector and the perpendicular to it, as plain numbers."""

    middle = piece.point_at(0.5)
    bisector = [float(middle[0] - piece.center[0]), float(middle[1] - piece.center[1])]
    length = math.hypot(*bisector)
    bisector = [one / length for one in bisector]
    return bisector, [-bisector[1], bisector[0]]


def _to_point(x, point) -> Expr:
    gap = [x[0] - float(point[0]), x[1] - float(point[1])]
    return ufl.sqrt(gap[0] * gap[0] + gap[1] * gap[1])


## Which side


def _flips(piece: Piece, x) -> list[Expr]:
    """Every ``±1`` this piece contributes to the parity."""

    flips = [_chord_flip(piece, x)]
    if piece.is_arc:
        flips.append(_segment_flip(piece, x))
    return flips


def _chord_flip(piece: Piece, x) -> Expr:
    """``-1`` where a ray from the point crosses this piece's chord.

    The ray runs along ``+x``, so a chord with no vertical extent never crosses it
    and is dropped while the form is built. That also keeps the chord's inverse
    slope a number known here, never a division by zero.
    """

    start_y, end_y = float(piece.start[1]), float(piece.end[1])
    if start_y == end_y:
        return ufl.as_ufl(1.0)

    start_x, end_x = float(piece.start[0]), float(piece.end[0])
    inverse_slope = (end_x - start_x) / (end_y - start_y)
    crossing_x = start_x + (x[1] - start_y) * inverse_slope

    below_start = ufl.conditional(ufl.le(start_y, x[1]), 1.0, 0.0)
    below_end = ufl.conditional(ufl.le(end_y, x[1]), 1.0, 0.0)
    straddles = abs(below_start - below_end)
    to_the_right = ufl.conditional(ufl.lt(x[0], crossing_x), 1.0, 0.0)
    return 1.0 - 2.0 * straddles * to_the_right


def _segment_flip(piece: Piece, x) -> Expr:
    """``-1`` inside the circular segment this arc cuts off from its chord.

    The correction that turns the polygon's answer into the region's: inside the
    circle *and* on the arc's side of the chord, that side being read off the
    arc's own midpoint.
    """

    center = [float(one) for one in piece.center]
    start = [float(one) for one in piece.start]
    chord = [float(b - a) for a, b in zip(piece.start, piece.end)]
    middle = piece.point_at(0.5)
    outward = math.copysign(
        1.0,
        chord[0] * float(middle[1] - start[1]) - chord[1] * float(middle[0] - start[0]),
    )

    offset = [x[0] - center[0], x[1] - center[1]]
    inside_circle = ufl.conditional(
        ufl.lt(
            offset[0] * offset[0] + offset[1] * offset[1],
            float(piece.radius) ** 2,
        ),
        1.0,
        0.0,
    )
    side = chord[0] * (x[1] - start[1]) - chord[1] * (x[0] - start[0])
    arc_side = ufl.conditional(ufl.gt(outward * side, 0.0), 1.0, 0.0)
    return 1.0 - 2.0 * inside_circle * arc_side


__all__ = ["signed_distance"]
