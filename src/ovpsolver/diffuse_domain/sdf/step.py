"""A CAD solid's 2D footprint, refitted so every boundary piece has a closed form.

A geometry can come from CAD so long as what arrives can have a signed distance
written against it in UFL. That constrains the *curve types*, not the shape: the
distance to a bounded line segment or circular arc is a closed form of a few
operations, so a boundary made of those gives an exact signed distance with one
term per piece. A B-spline has no closed form -- its closest point is a root find
per query point -- so a boundary carrying one cannot be written at all.

STEP is a boundary representation, not a CSG tree: the booleans that built the
shape are not exported. None are needed. A body is its loops, the distance is a
minimum over all their pieces whichever loop they came from, and the side is a
crossing count, in which an inner loop needs no special case
(:mod:`ovpsolver.diffuse_domain.sdf.arcs`).

Splines are refitted rather than sampled. A chord approximation converges like
``h^2`` and puts a term in the expression per vertex; a biarc -- two circular arcs
meeting at a tangent -- matches position *and* direction at both ends, converges
far faster, and stays inside the closed forms. The fit is ``G1`` and no more: the
curvature jumps at each joint, which the diffuse phase smears over ``2 eps``
anyway, and is invisible wherever the fit tolerance is small against ``eps``.

Watertightness is by construction and not by tolerance. Consecutive pieces are
built on shared point tags, so they end on the same point; and each original edge
is fitted between its own endpoints with its own end tangents, so the junctions the
CAD had are reproduced exactly. What the fit can move is the interior of an edge,
which :attr:`FitReport.deviation` measures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...mesh.generate import gmsh_session
from . import arcs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from ufl.core.expr import Expr

logger = logging.getLogger(__name__)

#: Curve types whose distance function is already a closed form, so a boundary
#: made only of these is ready to be written as UFL and needs no fitting.
CLOSED_FORM_TYPES = ("Line", "Circle")

#: Below this the fit treats a piece as straight rather than as an arc of vast
#: radius, whose centre is far away and numerically meaningless. A length in the
#: model's own units (how far the far end sits off the tangent line at the start),
#: so it compares directly with a tolerance and does not tighten with the piece.
STRAIGHT_SAGITTA = 1.0e-12


@dataclass(frozen=True)
class Piece:
    """One boundary piece: a segment if ``center`` is None, else a circular arc.

    Endpoints are carried even for an arc, rather than an angular span, because
    they are what the next piece shares and what closes the loop. ``ccw`` is the
    sweep direction: an arc is a *bounded* piece of its circle, and the two ways
    round it are different pieces.
    """

    start: np.ndarray
    end: np.ndarray
    center: np.ndarray | None = None
    radius: float = 0.0
    ccw: bool = True

    @property
    def is_arc(self) -> bool:
        return self.center is not None

    def sweep(self) -> float:
        """How much angle this arc turns through, in radians; zero for a segment."""

        if not self.is_arc:
            return 0.0
        start = math.atan2(*(self.start - self.center)[::-1])
        end = math.atan2(*(self.end - self.center)[::-1])
        turn = (end - start) % (2.0 * math.pi)
        return turn if self.ccw else (2.0 * math.pi - turn)

    def point_at(self, fraction: float) -> np.ndarray:
        """The point ``fraction`` of the way along the piece."""

        if not self.is_arc:
            return self.start + fraction * (self.end - self.start)
        start = math.atan2(*(self.start - self.center)[::-1])
        turn = self.sweep() * (1.0 if self.ccw else -1.0)
        angle = start + fraction * turn
        return self.center + self.radius * np.array(
            [math.cos(angle), math.sin(angle)]
        )


@dataclass(frozen=True)
class FitReport:
    """What the fit did, in the terms a reader would check it by.

    ``deviation`` is the thing to look at: the largest distance from a sample of
    any original curve to the fitted boundary, in the model's own units. It bounds
    how far the zero level set moved, which is the only error the diffuse method
    sees.
    """

    source: Path
    output: Path
    bodies: int
    pieces: int
    original_curves: int
    deviation: float
    area_error: float
    fitted: bool

    def __str__(self) -> str:
        if not self.fitted:
            return (
                f"{self.source.name}: every boundary piece is already a line or "
                f"an arc ({self.pieces} of them across {self.bodies} bodies); "
                f"nothing to fit"
            )
        return (
            f"{self.source.name} -> {self.output.name}: {self.bodies} bodies, "
            f"{self.original_curves} curves refitted as {self.pieces} arcs and "
            f"lines, deviation {self.deviation:.3e}, area error "
            f"{self.area_error:.3e}"
        )


## Reading a footprint out of a solid


@contextmanager
def _silenced_stdout() -> Iterator[None]:
    """Point file descriptor 1 at the null device for the duration of a call.

    The STEP writer reports its transfer statistics from inside OpenCASCADE, below
    what gmsh's verbosity setting reaches, so only the file descriptor stops it.
    Scoped to the one call that needs it, since anything that goes wrong in there
    is silenced too.
    """

    sys.stdout.flush()
    saved = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)


def footprint_face(gmsh: ModuleType, solid: int) -> int:
    """The planar face a solid was extruded from, which is the 2D shape wanted.

    Found as the thin planar face at the bottom of the solid, since a boundary
    representation does not record which face the modeller called a profile.
    Kernel bounding boxes carry a small padding, so "thin" and "at the bottom" are
    both measured against the solid's own height rather than against zero.
    """

    box = gmsh.model.getBoundingBox(3, solid)
    height = box[5] - box[2]
    candidates = []
    for _, signed in gmsh.model.getBoundary([(3, solid)], oriented=False):
        face = abs(signed)
        if gmsh.model.getType(2, face) != "Plane":
            continue
        face_box = gmsh.model.getBoundingBox(2, face)
        thin = (face_box[5] - face_box[2]) < 0.5 * height
        at_bottom = face_box[2] < box[2] + 0.5 * height
        if thin and at_bottom:
            candidates.append(face)
    if len(candidates) != 1:
        raise ValueError(
            f"solid {solid} has {len(candidates)} planar faces at its base, and a "
            f"footprint has to be exactly one. A body extruded along z from a "
            f"single closed profile gives one; a body built some other way needs "
            f"its 2D shape stated some other way"
        )
    return candidates[0]


def oriented_loops(gmsh: ModuleType, face: int) -> list[list[tuple[int, bool]]]:
    """The face's wires, each a list of ``(curve, reversed)`` in traversal order.

    The kernel lists a wire's curves in order, but the sign it gives them does not
    reliably say which way each is traversed: a wire whose curves were drawn
    outward from a corner comes back with every tag positive and every other curve
    stored backwards. So each curve is taken in whichever orientation starts where
    the last one ended.

    This is also the watertightness check: a wire that cannot be chained is one
    whose curves do not meet, and it fails here naming the gap.
    """

    _, curves = gmsh.model.occ.getCurveLoops(face)
    box = gmsh.model.getBoundingBox(2, face)
    diagonal = math.dist(box[:3], box[3:])
    tolerance = 1.0e-6 * diagonal

    loops = []
    for wire in curves:
        tags = [abs(int(tag)) for tag in wire]
        ends = {}
        for tag in tags:
            point, _ = curve_geometry(gmsh, tag, reverse=False)
            ends[tag] = (point(0.0), point(1.0))
        for first_reversed in (False, True):
            chain = _chain(tags, ends, first_reversed, tolerance)
            if chain is not None:
                loops.append(chain)
                break
        else:
            raise ValueError(
                f"the wire {tags} of face {face} does not close: its curves were "
                f"returned in order but consecutive ones do not share an endpoint "
                f"to within {tolerance:.2e}. The footprint is not watertight"
            )
    return loops


def _chain(tags, ends, first_reversed: bool, tolerance: float):
    """Walk a wire, orienting each curve to start where the last one ended."""

    chain = [(tags[0], first_reversed)]
    start, finish = ends[tags[0]]
    here = start if first_reversed else finish
    opening = finish if first_reversed else start
    for tag in tags[1:]:
        low, high = ends[tag]
        if float(np.linalg.norm(low - here)) <= tolerance:
            chain.append((tag, False))
            here = high
        elif float(np.linalg.norm(high - here)) <= tolerance:
            chain.append((tag, True))
            here = low
        else:
            return None
    if float(np.linalg.norm(here - opening)) > tolerance:
        return None
    return chain


def curve_geometry(gmsh: ModuleType, tag: int, *, reverse: bool):
    """One oriented curve's point and unit tangent, as callables over ``[0, 1]``.

    They run along the curve in the direction asked for, so a caller walking a
    loop never has to think about which way an edge was stored.
    """

    curve = abs(tag)
    lower, upper = gmsh.model.getParametrizationBounds(1, curve)
    low, high = float(lower[0]), float(upper[0])

    def parameter(fraction: float) -> float:
        fraction = 1.0 - fraction if reverse else fraction
        return low + fraction * (high - low)

    def point(fraction: float) -> np.ndarray:
        value = gmsh.model.getValue(1, curve, [parameter(fraction)])
        return np.array(value[:2], dtype=float)

    def tangent(fraction: float) -> np.ndarray:
        value = gmsh.model.getDerivative(1, curve, [parameter(fraction)])
        direction = np.array(value[:2], dtype=float)
        if reverse:
            direction = -direction
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError(f"curve {curve} has a vanishing tangent")
        return direction / norm

    return point, tangent


## Fitting


def biarc(
    start: np.ndarray,
    start_tangent: np.ndarray,
    end: np.ndarray,
    end_tangent: np.ndarray,
) -> list[Piece] | None:
    """Two arcs from ``start`` to ``end`` matching both tangents, or None.

    The joint is placed by the equal-tangent-length construction: both arcs get
    the same tangent length ``d``, which fixes the one free parameter of the biarc
    family and stays well behaved as the two tangents approach each other. ``d``
    solves the quadratic that makes the two arcs share a tangent at the joint, so
    the result is ``G1`` there by construction.

    None where the construction degenerates -- a non-positive ``d``, or endpoints
    on top of each other -- which is the caller's signal to subdivide instead.
    """

    chord = end - start
    if float(chord @ chord) <= 0.0:
        return None
    projected = float(chord @ (start_tangent + end_tangent))
    squared = float(chord @ chord)
    opening = 2.0 * (1.0 - float(start_tangent @ end_tangent))
    if opening < 1.0e-12:
        # Parallel tangents: the quadratic degenerates to a linear equation.
        along = float(chord @ end_tangent)
        if abs(along) < 1.0e-14:
            return None
        length = squared / (4.0 * along)
    else:
        length = (
            -projected + math.sqrt(projected * projected + opening * squared)
        ) / opening
    if not math.isfinite(length) or length <= 0.0:
        return None
    joint = 0.5 * ((start + length * start_tangent) + (end - length * end_tangent))
    first = arc_through(start, start_tangent, joint)
    second = arc_through(end, -end_tangent, joint)
    if first is None or second is None:
        return None
    return [first, _reversed(second)]


def arc_through(
    start: np.ndarray, tangent: np.ndarray, end: np.ndarray
) -> Piece | None:
    """The arc leaving ``start`` along ``tangent`` and reaching ``end``.

    A segment where ``end`` lies within :data:`STRAIGHT_SAGITTA` of the tangent
    line; None where ``end`` is ``start``.
    """

    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if chord_length == 0.0:
        return None
    normal = np.array([-tangent[1], tangent[0]])
    denominator = 2.0 * float(chord @ normal)
    sagitta = abs(float(chord @ normal)) / chord_length
    if abs(denominator) < 1.0e-30 or sagitta * chord_length < STRAIGHT_SAGITTA:
        return Piece(start=start, end=end)
    signed_radius = float(chord @ chord) / denominator
    center = start + signed_radius * normal
    return Piece(
        start=start,
        end=end,
        center=center,
        radius=abs(signed_radius),
        ccw=signed_radius > 0.0,
    )


def _reversed(piece: Piece) -> Piece:
    """The same piece traversed from its end to its start."""

    return Piece(
        start=piece.end,
        end=piece.start,
        center=piece.center,
        radius=piece.radius,
        ccw=not piece.ccw,
    )


def distance_to_piece(piece: Piece, point: np.ndarray) -> float:
    """Distance from a point to one *bounded* piece.

    The nearest point of the infinite line or circle is not the nearest point of
    the piece unless it falls inside it; taking the unbounded one is what makes a
    naive fit of a corner wrong.
    """

    if not piece.is_arc:
        edge = piece.end - piece.start
        squared = float(edge @ edge)
        if squared == 0.0:
            return float(np.linalg.norm(point - piece.start))
        along = float((point - piece.start) @ edge) / squared
        along = min(1.0, max(0.0, along))
        return float(np.linalg.norm(point - (piece.start + along * edge)))

    radial = point - piece.center
    length = float(np.linalg.norm(radial))
    if length == 0.0:
        return piece.radius
    start = math.atan2(*(piece.start - piece.center)[::-1])
    here = math.atan2(*(radial)[::-1])
    turn = (here - start) % (2.0 * math.pi)
    if not piece.ccw:
        turn = (2.0 * math.pi - turn) % (2.0 * math.pi)
    if turn <= piece.sweep():
        return abs(length - piece.radius)
    return min(
        float(np.linalg.norm(point - piece.start)),
        float(np.linalg.norm(point - piece.end)),
    )


def fit_loop(
    edges: Sequence[tuple[bool, Callable, Callable]],
    *,
    tolerance: float,
    samples: int,
    corner_degrees: float,
) -> list[Piece]:
    """Refit a whole wire, merging across the edge divisions the CAD happened to use.

    ``edges`` are ``(exact, point, tangent)`` per edge in traversal order. The
    divisions in a STEP file are an artefact of how the shape was drawn: a
    letterform arrives as ten spline edges meeting smoothly where one curve would
    do, and fitting each separately floors the piece count at one biarc per edge
    however loose the tolerance. So consecutive spline edges meeting smoothly are
    merged into one span and fitted as a single curve.

    Nothing is merged across a *corner*, where the boundary genuinely turns, or
    across an edge that is already a line or an arc, which is kept exactly as it
    is. So the fit never rounds off a feature the CAD was explicit about.
    """

    pieces: list[Piece] = []
    for exact, run in _runs(edges, math.radians(corner_degrees)):
        if exact:
            for index in run:
                pieces.extend(_exact_piece(edges[index][1], edges[index][2]))
        else:
            point, tangent = _span(edges, run)
            # Per *edge* of the run, so that a span covering ten edges is
            # checked as densely as ten spans covering one each. A fixed budget
            # would thin out exactly as merging made the spans longer, and the
            # fit would drift past the tolerance without ever measuring it.
            pieces.extend(
                fit_span(
                    point,
                    tangent,
                    tolerance=tolerance,
                    samples=samples * len(run),
                )
            )
    return pieces


def fit_span(point, tangent, *, tolerance: float, samples: int) -> list[Piece]:
    """Cover one smooth span in as few biarcs as the tolerance allows.

    Greedy and not recursive: each biarc is stretched as far along the span as it
    can reach within tolerance, and the next starts where it stopped. Bisection
    would halve regardless of how much curve a piece could cover, making the piece
    count depend on the subdivision tree rather than on the shape.

    How far a biarc reaches is found by bisecting on its far end, which assumes
    the deviation grows with the length of the span -- true of a curve of slowly
    varying curvature, and the assumption every arc-spline fitter makes. Where it
    fails the result is a shorter piece than necessary, never one that exceeds the
    tolerance, because what is accepted is always measured.
    """

    pieces: list[Piece] = []
    start = 0.0
    while start < 1.0 - 1.0e-12:
        end, fitted = _longest_biarc(
            point, tangent, start, tolerance=tolerance, samples=samples
        )
        pieces.extend(fitted)
        start = end
    return pieces


def _runs(edges, corner: float) -> list[tuple[bool, list[int]]]:
    """Group a wire's edges into exact ones and maximal smooth spline runs."""

    groups: list[tuple[bool, list[int]]] = []
    for index, (exact, _, tangent) in enumerate(edges):
        previous = edges[index - 1]
        joins = (
            groups
            and not exact
            and not groups[-1][0]
            and not previous[0]
            and _break_angle(previous[2](1.0), tangent(0.0)) <= corner
        )
        if joins:
            groups[-1][1].append(index)
        else:
            groups.append((exact, [index]))
    # A wire whose first and last runs are one smooth spline run split by the
    # arbitrary place the kernel started listing it.
    if (
        len(groups) > 1
        and not groups[0][0]
        and not groups[-1][0]
        and _break_angle(
            edges[groups[-1][1][-1]][2](1.0), edges[groups[0][1][0]][2](0.0)
        )
        <= corner
    ):
        merged = groups.pop(0)
        groups[-1][1].extend(merged[1])
    return groups


def _break_angle(outgoing, incoming) -> float:
    """The angle between two unit tangents."""

    return math.acos(min(1.0, max(-1.0, float(outgoing @ incoming))))


def _span(edges, run: Sequence[int]):
    """One continuous parametrization over ``[0, 1]`` across several edges."""

    count = len(run)

    def locate(fraction: float):
        scaled = min(1.0, max(0.0, fraction)) * count
        index = min(int(scaled), count - 1)
        return run[index], scaled - index

    def point(fraction: float):
        index, local = locate(fraction)
        return edges[index][1](local)

    def tangent(fraction: float):
        index, local = locate(fraction)
        return edges[index][2](local)

    return point, tangent


def _longest_biarc(point, tangent, start: float, *, tolerance: float, samples: int):
    """The furthest a single biarc reaches from ``start``, and the biarc itself."""

    def attempt(end: float):
        if end <= start:
            return None
        fitted = biarc(point(start), tangent(start), point(end), tangent(end))
        if fitted is None:
            return None
        worst = max(
            min(
                distance_to_piece(piece, point(start + (end - start) * k / samples))
                for piece in fitted
            )
            for k in range(samples + 1)
        )
        return fitted if worst <= tolerance else None

    whole = attempt(1.0)
    if whole is not None:
        return 1.0, whole

    low, high = start, 1.0
    best = None
    for _ in range(24):
        middle = 0.5 * (low + high)
        fitted = attempt(middle)
        if fitted is None:
            high = middle
        else:
            low, best = middle, fitted
    if best is None or low - start < 1.0e-9:
        raise ValueError(
            f"no biarc from {start:.6f} reaches the tolerance {tolerance:.3e}; the "
            f"span is probably not smooth, or the tolerance is below the model's "
            f"own precision"
        )
    return low, best


def split_wide_arcs(pieces: Sequence[Piece], limit: float) -> list[Piece]:
    """Halve any arc turning through more than ``limit``.

    Not about accuracy: STEP writes an arc by its two endpoints and its centre,
    which does not say which way round the circle was meant. Keeping every sweep
    under half a turn makes the short way the right way, so the file reads back as
    the shape that was written.
    """

    out: list[Piece] = []
    for piece in pieces:
        if not piece.is_arc or piece.sweep() <= limit:
            out.append(piece)
            continue
        middle = piece.point_at(0.5)
        out.append(
            Piece(piece.start, middle, piece.center, piece.radius, piece.ccw)
        )
        out.append(Piece(middle, piece.end, piece.center, piece.radius, piece.ccw))
    return out


## The whole pass


def read_bodies(
    gmsh: ModuleType, path: Path
) -> list[tuple[int, int, list[list[tuple[int, bool]]]]]:
    """Every solid in the file as ``(solid, footprint face, oriented loops)``."""

    gmsh.model.add(path.stem)
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    solids = gmsh.model.getEntities(3)
    if not solids:
        raise ValueError(
            f"{path} holds no solid bodies. Each inclusion has to be one solid, "
            f"so a sketch has to be extruded before it is exported"
        )
    bodies = []
    for _, solid in solids:
        face = footprint_face(gmsh, solid)
        bodies.append((solid, face, oriented_loops(gmsh, face)))
    return bodies


def fit_step_to_arcs(
    path: str | Path,
    *,
    tolerance: float,
    samples: int = 64,
    corner_degrees: float = 5.0,
    max_sweep_degrees: float = 120.0,
    force: bool = False,
) -> FitReport:
    """Rewrite a STEP file with every footprint spline refitted as circular arcs.

    The output goes beside the input as ``<stem>_fit.step``, with a ``.json``
    record of the fit. It is a STEP file like any other, so the fit can be opened
    in the modeller against the original rather than taken on trust, and it is
    what later stages read. It is reused while the record's key, a hash of the
    source and the fit parameters, still matches.

    ``tolerance`` is the largest distance the boundary may move, in the model's
    own units, and the only accuracy knob. It belongs beside the smearing
    parameter, not the CAD: the method resolves a boundary over ``2 eps``, so a
    tolerance well under ``eps`` is invisible and one above it is not.

    A file already made only of lines and arcs is left alone and its own path
    comes back, so a geometry that needs no fitting gains no second copy.
    """

    path = Path(path).expanduser().resolve()
    output = path.with_name(f"{path.stem}_fit.step")
    record = output.with_suffix(".json")
    key = _fit_key(
        path, tolerance, samples, corner_degrees, max_sweep_degrees
    )

    if not force and output.exists() and record.exists():
        try:
            stored = json.loads(record.read_text())
            if stored["key"] == key:
                logger.info(
                    "arc fit is current for %s; reading %s (%d piece(s) in %d "
                    "bod(ies), deviation %.3e, area error %.3e)",
                    path,
                    output,
                    stored["pieces"],
                    stored["bodies"],
                    stored["deviation"],
                    stored["area_error"],
                )
                return FitReport(
                    source=path,
                    output=output,
                    bodies=stored["bodies"],
                    pieces=stored["pieces"],
                    original_curves=stored["original_curves"],
                    deviation=stored["deviation"],
                    area_error=stored["area_error"],
                    fitted=True,
                )
        except (ValueError, KeyError):
            pass

    limit = math.radians(max_sweep_degrees)
    with gmsh_session() as gmsh:
        bodies = read_bodies(gmsh, path)
        whole = gmsh.model.getBoundingBox(-1, -1)
        tolerance_junction = 1.0e-6 * math.dist(whole[:3], whole[3:])
        if not force and not any(
            gmsh.model.getType(1, tag) not in CLOSED_FORM_TYPES
            for _, _, loops in bodies
            for loop in loops
            for tag, _ in loop
        ):
            pieces_total = sum(
                len(loop) for _, _, loops in bodies for loop in loops
            )
            logger.info(
                "%s is already lines and arcs; reading it directly, with no fit "
                "and no derived copy (%d piece(s) in %d bod(ies))",
                path,
                pieces_total,
                len(bodies),
            )
            return FitReport(
                source=path,
                output=path,
                bodies=len(bodies),
                pieces=pieces_total,
                original_curves=pieces_total,
                deviation=0.0,
                area_error=0.0,
                fitted=False,
            )

        # Said before the work, since this is the one step of building a
        # geometry that takes seconds rather than milliseconds.
        logger.info(
            "refitting %s to lines and arcs at tolerance %.3e%s; %d bod(ies)",
            path,
            tolerance,
            " (forced)" if force else "",
            len(bodies),
        )

        fitted: list[tuple[float, list[list[Piece]]]] = []
        original_areas: list[float] = []
        worst = 0.0
        curve_count = 0
        for solid, face, loops in bodies:
            box = gmsh.model.getBoundingBox(3, solid)
            original_areas.append(gmsh.model.occ.getMass(2, face))
            body_loops = []
            for loop in loops:
                edges = []
                for tag, reverse in loop:
                    curve_count += 1
                    point, tangent = curve_geometry(gmsh, tag, reverse=reverse)
                    exact = gmsh.model.getType(1, tag) in CLOSED_FORM_TYPES
                    edges.append((exact, point, tangent))
                pieces = fit_loop(
                    edges,
                    tolerance=tolerance,
                    samples=samples,
                    corner_degrees=corner_degrees,
                )
                worst = max(worst, _deviation(edges, pieces, samples))
                pieces = split_wide_arcs(pieces, limit)
                _require_closed(pieces, tolerance_junction)
                body_loops.append(pieces)
            fitted.append((box[5] - box[2], orient_body(body_loops)))

    with gmsh_session() as gmsh:
        gmsh.model.add(f"{path.stem}_fit")
        surfaces = []
        construction = []
        for height, body_loops in fitted:
            loop_tags = []
            for pieces in body_loops:
                loop_tag, centers = _build_loop(gmsh, pieces)
                loop_tags.append(loop_tag)
                construction.extend(centers)
            surfaces.append(gmsh.model.occ.addPlaneSurface(loop_tags))
        gmsh.model.occ.remove([(0, tag) for tag in construction])
        for surface, (height, _) in zip(surfaces, fitted):
            gmsh.model.occ.extrude([(2, surface)], 0.0, 0.0, height)
        gmsh.model.occ.synchronize()
        _require_only_solids(gmsh, len(fitted))
        with _silenced_stdout():
            gmsh.write(str(output))

    fitted_areas = [sum(signed_area(loop) for loop in body) for _, body in fitted]
    _verify(output, fitted_areas)

    area_error = max(
        abs(new - old) / abs(old) for new, old in zip(fitted_areas, original_areas)
    )
    pieces_total = sum(len(loop) for _, body in fitted for loop in body)
    record.write_text(
        json.dumps(
            {
                "key": key,
                "source": str(path),
                "bodies": len(fitted),
                "pieces": pieces_total,
                "original_curves": curve_count,
                "deviation": worst,
                "area_error": area_error,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "fitted %s: %d curve(s) -> %d piece(s) in %d bod(ies), deviation %.3e "
        "(tolerance %.3e), area error %.3e; wrote %s and %s",
        path,
        curve_count,
        pieces_total,
        len(fitted),
        worst,
        tolerance,
        area_error,
        output,
        record,
    )
    return FitReport(
        source=path,
        output=output,
        bodies=len(fitted),
        pieces=pieces_total,
        original_curves=curve_count,
        deviation=worst,
        area_error=area_error,
        fitted=True,
    )


def signed_area(pieces: Sequence[Piece]) -> float:
    """The enclosed area, signed: positive counter-clockwise.

    Exact, not sampled: the shoelace over the chords plus, for every arc, the
    circular segment it bulges by, ``(R^2/2)(theta - sin theta)``, added when the
    arc turns counter-clockwise and subtracted when it does not. Exact because it
    both decides which loop of a body is the outside and checks the fit, where a
    mis-oriented piece or a lost hole shows up as an area that moved.
    """

    total = 0.0
    for piece in pieces:
        start, end = piece.start, piece.end
        total += 0.5 * float(start[0] * end[1] - end[0] * start[1])
        if piece.is_arc:
            turn = piece.sweep()
            segment = 0.5 * piece.radius**2 * (turn - math.sin(turn))
            total += segment if piece.ccw else -segment
    return total


def orient_body(loops: Sequence[Sequence[Piece]]) -> list[list[Piece]]:
    """The outside loop counter-clockwise and every hole clockwise.

    So a body's area is the sum of its loops' :func:`signed_area`, which is what
    the fit is checked by. The kernel's own sense is not this: a footprint is read
    off the *bottom* face of the extrusion, whose normal points away from the
    solid, so its outer wire arrives clockwise. The outside is the loop of largest
    area, since nothing promises the order of the list.
    """

    areas = [signed_area(loop) for loop in loops]
    outer = max(range(len(loops)), key=lambda index: abs(areas[index]))
    oriented = []
    for index, loop in enumerate(loops):
        wanted_ccw = index == outer
        oriented.append(
            list(loop) if (areas[index] > 0.0) == wanted_ccw else reverse_loop(loop)
        )
    return oriented


def reverse_loop(pieces: Sequence[Piece]) -> list[Piece]:
    """The same closed loop traversed the other way round."""

    return [_reversed(piece) for piece in reversed(pieces)]


def _deviation(edges, pieces: Sequence[Piece], samples: int) -> float:
    """How far the original wire strays from the fitted one, at its worst.

    Over the whole wire, because after merging a fitted piece may span several
    original edges and an edge may be covered by several pieces. Every sample of
    the original is measured against all the pieces: the one-sided Hausdorff
    distance the tolerance bounds.
    """

    worst = 0.0
    for _, point, _ in edges:
        for step in range(samples + 1):
            here = point(step / samples)
            worst = max(worst, min(distance_to_piece(p, here) for p in pieces))
    return worst


def _require_closed(pieces: Sequence[Piece], tolerance: float) -> None:
    """Every piece has to start where the last one ended, or the loop has a slit.

    Checked on the fitted pieces too, because a fit that lost an endpoint would
    still produce a plausible-looking wire with a sliver missing from it.
    """

    for index, piece in enumerate(pieces):
        following = pieces[(index + 1) % len(pieces)]
        gap = float(np.linalg.norm(piece.end - following.start))
        if gap > tolerance:
            raise ValueError(
                f"the fitted boundary has a gap of {gap:.3e} between pieces "
                f"{index} and {(index + 1) % len(pieces)}, which is above the "
                f"{tolerance:.3e} a junction is allowed"
            )


def _exact_piece(point, tangent) -> list[Piece]:
    """A line or arc kept as it already is, read off three of its own points."""

    start, middle, end = point(0.0), point(0.5), point(1.0)
    # A closed circle has coincident ends, and no single arc between them; it is
    # split at its midpoint before anything else is asked of its endpoints.
    if float(np.linalg.norm(end - start)) < STRAIGHT_SAGITTA:
        first = arc_through(start, tangent(0.0), middle)
        second = arc_through(middle, tangent(0.5), end)
        return [p for p in (first, second) if p is not None]
    piece = arc_through(start, tangent(0.0), end)
    if piece is None or not piece.is_arc:
        return [Piece(start=start, end=end)]
    return [piece]


def _build_loop(gmsh: ModuleType, pieces: Sequence[Piece]) -> tuple[int, list[int]]:
    """One closed wire, on point tags shared between consecutive pieces.

    Sharing the tags makes the result watertight without relying on the kernel's
    sewing tolerance: consecutive pieces end at the same point, not at equal
    coordinates.

    An arc also needs a point at its centre, which is construction geometry. Those
    tags come back so the caller can delete them once the arcs exist: left in
    place they are free vertices, which STEP records and a modeller opens as a
    stray sketch beside the bodies.
    """

    starts = [
        gmsh.model.occ.addPoint(float(piece.start[0]), float(piece.start[1]), 0.0)
        for piece in pieces
    ]
    tags = []
    centers = []
    for index, piece in enumerate(pieces):
        first = starts[index]
        second = starts[(index + 1) % len(pieces)]
        if piece.is_arc:
            center = gmsh.model.occ.addPoint(
                float(piece.center[0]), float(piece.center[1]), 0.0
            )
            centers.append(center)
            tags.append(gmsh.model.occ.addCircleArc(first, center, second))
        else:
            tags.append(gmsh.model.occ.addLine(first, second))
    return gmsh.model.occ.addCurveLoop(tags), centers


def _require_only_solids(gmsh: ModuleType, expected: int) -> None:
    """Nothing in the written model but the bodies themselves.

    A free point or dangling curve is a root shape of its own, which STEP records
    and a modeller opens as a stray object; the output is a CAD file so that a
    reader can open it and see exactly the inclusions.
    """

    solids = gmsh.model.getEntities(3)
    if len(solids) != expected:
        raise ValueError(
            f"the fitted model holds {len(solids)} solids where {expected} were "
            f"built; a body was lost or split"
        )
    used_points = {
        abs(tag)
        for _, solid in solids
        for _, tag in gmsh.model.getBoundary(
            [(3, solid)], oriented=False, recursive=True
        )
    }
    stray = [tag for _, tag in gmsh.model.getEntities(0) if tag not in used_points]
    if stray:
        raise ValueError(
            f"the fitted model holds {len(stray)} free points that belong to no "
            f"body; they would be written as a stray sketch"
        )


def _verify(output: Path, expected: Sequence[float]) -> None:
    """Read the written file back and check each body encloses the area it should.

    End to end and not on the model in memory, because the file is what a modeller
    opens and every later stage reads, and the kernel's account of a half-built
    face is not a reliable stand-in: the area it reports for a face with holes adds
    the holes, while the same face written out and read back is right. A hole
    added instead of subtracted, or a loop reversed, moves the area by a large
    fraction of itself.
    """

    with gmsh_session() as gmsh:
        gmsh.model.add(f"{output.stem}_check")
        gmsh.model.occ.importShapes(str(output))
        gmsh.model.occ.synchronize()
        solids = gmsh.model.getEntities(3)
        if len(solids) != len(expected):
            raise ValueError(
                f"{output.name} reads back as {len(solids)} bodies, not "
                f"{len(expected)}"
            )
        for (_, solid), wanted in zip(solids, expected):
            box = gmsh.model.getBoundingBox(3, solid)
            area = gmsh.model.occ.getMass(3, solid) / (box[5] - box[2])
            # Loose on purpose: it must not trip on the kernel's own
            # integration tolerance, or on the round trip through the file.
            if abs(area - wanted) > 1.0e-4 * abs(wanted):
                raise ValueError(
                    f"{output.name}: a body encloses {area:.6f} where its pieces "
                    f"enclose {wanted:.6f}. A hole was added rather than "
                    f"subtracted, or a loop came out the wrong way round"
                )


def _fit_key(
    path: Path, tolerance: float, samples: int, corner: float, sweep: float
) -> str:
    """Hash the source *and* the parameters, so a changed tolerance misses."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(repr((tolerance, samples, corner, sweep, 2)).encode())
    return digest.hexdigest()


## Reading a fitted file back as pieces


def piece_of_curve(gmsh: ModuleType, tag: int, *, reverse: bool) -> Piece:
    """One line or arc of a fitted file, as the piece it stands for.

    Read off the curve by evaluation, since the kernel's API offers no centre or
    radius for an edge. Three points determine a circle, and the middle one also
    says which way round the arc goes.
    """

    point, tangent = curve_geometry(gmsh, tag, reverse=reverse)
    start, middle, end = point(0.0), point(0.5), point(1.0)
    if gmsh.model.getType(1, tag) == "Line":
        return Piece(start=start, end=end)

    piece = arc_through(start, tangent(0.0), end)
    if piece is None:
        raise ValueError(f"curve {tag} is an arc of no extent")
    if not piece.is_arc:
        return piece
    # arc_through takes the short way; the stored arc may be the long one, which
    # its own midpoint settles.
    if float(np.linalg.norm(piece.point_at(0.5) - middle)) > 1.0e-7 * piece.radius:
        piece = Piece(
            start=start,
            end=end,
            center=piece.center,
            radius=piece.radius,
            ccw=not piece.ccw,
        )
    return piece


def read_pieces(path: str | Path) -> list[list[list[Piece]]]:
    """Every body of a fitted file, as oriented loops of lines and arcs.

    Refuses a file that still holds a curve with no closed form, so that the one
    thing the distance needs of its input is checked where it can be said plainly.
    """

    path = Path(path)
    bodies = []
    with gmsh_session() as gmsh:
        for _, _, loops in read_bodies(gmsh, path):
            body = []
            for loop in loops:
                pieces = []
                for tag, reverse in loop:
                    kind = gmsh.model.getType(1, tag)
                    if kind not in CLOSED_FORM_TYPES:
                        raise ValueError(
                            f"{path} still holds a {kind} curve, which has no "
                            f"closed-form distance. Fit it first"
                        )
                    pieces.append(piece_of_curve(gmsh, tag, reverse=reverse))
                _require_closed(pieces, 1.0e-9 * max(1.0, abs(signed_area(pieces))))
                body.append(pieces)
            bodies.append(orient_body(body))
    return bodies


def sdfs_from_step(
    path: str | Path,
    *,
    eps: float,
    tolerance_in_eps: float = 1.0 / 20.0,
    **fit,
) -> list[Callable[[Expr], Expr]]:
    """One signed distance per body of a CAD file, ready to call on a coordinate.

    Refits the file (:func:`fit_step_to_arcs`, cached against the source and the
    parameters), reads the result back, and builds a distance from each body. The
    fit tolerance is a fraction of the run's ``eps``, the only scale it means
    anything against.
    """

    report = fit_step_to_arcs(path, tolerance=tolerance_in_eps * eps, **fit)
    return [arcs.signed_distance(body) for body in read_pieces(report.output)]


def fit_outputs(path: str | Path) -> list[Path]:
    """The files :func:`fit_step_to_arcs` derives from a source, where they exist.

    For a caller that has to move a geometry around, such as archiving a run
    beside its output, without knowing how the fit names things.
    """

    path = Path(path).expanduser().resolve()
    derived = [
        path.with_name(f"{path.stem}_fit.step"),
        path.with_name(f"{path.stem}_fit.json"),
    ]
    return [one for one in derived if one.is_file()]


__all__ = [
    "CLOSED_FORM_TYPES",
    "FitReport",
    "Piece",
    "arc_through",
    "biarc",
    "distance_to_piece",
    "fit_loop",
    "fit_outputs",
    "fit_span",
    "fit_step_to_arcs",
    "orient_body",
    "piece_of_curve",
    "read_pieces",
    "reverse_loop",
    "sdfs_from_step",
    "signed_area",
    "split_wide_arcs",
]
