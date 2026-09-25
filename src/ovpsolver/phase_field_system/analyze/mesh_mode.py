"""Is a field carrying the mesh's symmetry rather than the geometry's?

An inclusion is a circle, and nothing in the spec distinguishes one direction around
it from another, so azimuthal structure in a field about one is either the physics,
which has no preferred direction either and shows up broadband, or the mesh, which
does. These are the measurements that tell the two apart, for
:mod:`~ovpsolver.phase_field_system.analyze.wetting` and any other diagnostic asking
the question of a saved field.

Three things have to agree before the mesh is named. *Amplitude* alone says nothing,
since a rough field carries as much at ``m = 6`` as anywhere, so the mode is held
against its neighbours and the *ratio* is the measurement. *Phase*: a triangular mesh
has one six-fold axis, and lobes sitting on it are its, so a crest is compared with
the axis of the mesh's own facet normals (:func:`lattice_axis`). *Agreement between
inclusions*: they sit at unrelated places, so lobe patterns at one common angle are
copies of one field's symmetry, and the only field they have in common is the mesh.

The angle is the direction the nearest surface faces, not an angle about a centre:
an arbitrary shape has no centre, and for a circle the two coincide. Shells are in
multiples of ``eps`` outward from the zero level set, the scale the surface terms
decay on, since the structure is layered and one band averaged over several would
report the difference of large numbers. Each shell is detrended radially first
(:func:`detrend`), or a radial profile crossing it leaks into every mode at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Unanswerable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...fem.save import Run

#: The mode this is about. Six, because a triangular mesh has a six-fold axis and
#: imprints it on anything the mesh is asked to resolve.
MODE = 6

#: The modes :data:`MODE` is held against, to tell a six-fold structure from a
#: field that is merely rough. Its immediate neighbours either side, skipping none.
NEIGHBOURS = (2, 3, 4, 5, 7, 8)

#: The fewest cells a shell may have and still be asked for a mode: ten per lobe
#: of :data:`MODE`, enough that a single cell cannot move the answer.
LEAST_CELLS = 60


def angular_mode(
    values: np.ndarray,
    angles: np.ndarray,
    mode: int,
    weights: "np.ndarray | None" = None,
) -> tuple[float, float]:
    """Amplitude and crest of ``A cos(mode (theta - crest))`` in ``values``.

    The amplitude is in the units of ``values`` and the crest in degrees, modulo
    one period of the mode. One convention, so that a field's crest and a mesh's
    axis are the same kind of number and may be subtracted.
    """

    if weights is None:
        weights = np.ones(len(angles))
    total = float(weights.sum()) or 1.0
    coefficient = 2.0 * np.sum(weights * values * np.exp(-1j * mode * angles)) / total
    crest = float(np.degrees(-np.angle(coefficient) / mode) % (360.0 / mode))
    return float(np.abs(coefficient)), crest


def circular_mean(crests: "Sequence[float]", mode: int) -> float:
    """Mean of angles that live modulo ``360 / mode``, in degrees.

    :func:`angular_mode` reduces a crest modulo the mode's period, so for
    ``m = 6`` the angles 59.9 and 0.1 degrees are a fifth of a degree apart, not
    59.8. Their arithmetic mean is 30, the furthest possible value from both,
    which would report inclusions all lobed on the mesh axis as maximally off it.
    Averaged instead as unit vectors at ``mode`` times the angle, the only mean
    that respects the wrap.
    """

    angles = np.deg2rad(np.asarray(list(crests), dtype=float)) * mode
    resultant = np.mean(np.exp(1j * angles))
    period = 360.0 / mode
    return float(np.rad2deg(np.angle(resultant)) / mode % period)


def detrend(values: np.ndarray, coordinate: np.ndarray, degree: int = 3) -> np.ndarray:
    """``values`` with a polynomial in ``coordinate`` taken out.

    A radial profile crossing a shell is not azimuthal structure, but it is not
    orthogonal to any of the modes either, so it has to come out before they are
    measured rather than being divided out after.
    """

    design = np.column_stack([coordinate**power for power in range(degree + 1)])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def lattice_axis(mesh: Any, mode: int = MODE) -> tuple[float, float]:
    """The mesh's own ``mode``-fold axis, off its interior facet normals.

    Normals rather than edge directions because the scheme is written in the
    normals: every upwind flux, the two-point Korteweg stiffness and the
    least-squares gradient are sums over facets weighted by them. For a triangle
    the two differ by ninety degrees, half a period at ``m = 6``, so the choice
    moves the answer by thirty degrees.

    Length weighted, since a long facet carries more of every facet sum, and
    undirected: a normal's sign is arbitrary, and the statistic is invariant under
    reversing it only for even ``mode``. The amplitude is the peak-to-mean
    modulation of the direction density, so two hundred per cent is a perfect
    triangular lattice and zero no preference at all.
    """

    if mode % 2:
        raise ValueError(
            f"a facet normal has an arbitrary sign, so an m={mode} statistic on "
            "one is not well defined; ask for an even mode"
        )
    dim = mesh.topology.dim
    if dim != 2:
        raise Unanswerable(
            f"the lattice axis is read off facet normals, which is a "
            f"two-dimensional idea; this mesh is {dim}-dimensional"
        )

    mesh.topology.create_connectivity(dim - 1, dim)
    mesh.topology.create_connectivity(dim - 1, 0)
    facet_cells = mesh.topology.connectivity(dim - 1, dim)
    facet_nodes = mesh.topology.connectivity(dim - 1, 0)
    points = mesh.geometry.x[:, :dim]

    angles, lengths = [], []
    for facet in range(mesh.topology.index_map(dim - 1).size_local):
        if len(facet_cells.links(facet)) != 2:
            continue
        first, second = facet_nodes.links(facet)[:2]
        edge = points[second] - points[first]
        lengths.append(float(np.linalg.norm(edge)))
        angles.append(np.arctan2(-edge[0], edge[1]))
    if not angles:
        raise Unanswerable("this mesh has no interior facets to read an axis off")
    return angular_mode(
        np.ones(len(angles)), np.asarray(angles), mode, np.asarray(lengths)
    )


def axis_offset(crest: float, axis: float, mode: int = MODE) -> float:
    """How far a crest is from an axis, the short way round, in degrees."""

    period = 360.0 / mode
    gap = abs(crest - axis) % period
    return min(gap, period - gap)


class Shells:
    """The cells of each shell about each inclusion, and their angles.

    Built once from the geometry and reused for every field and every frame: the
    signed distances and the surface normals cost more to evaluate than the modes
    cost to measure, and do not depend on what is being measured.
    """

    def __init__(self, run: "Run", domain: Any, edges: "tuple[float, ...]") -> None:
        dim = run.mesh.topology.dim
        signed = domain.signed_distances_on(run.cell_space())
        normals = domain.surface_normals_on(run.cell_space((dim,)))
        which = np.argmin(np.abs(signed), axis=0)
        columns = np.arange(signed.shape[1])
        # Unit, pointing into the nearest inclusion.
        normal = normals[which, columns]

        self.eps = float(domain.domain_eps)
        self.inclusions = int(which.max()) + 1
        self.which = which
        #: Signed distance to the nearest surface, in multiples of ``eps``.
        self.gap = signed[which, columns] / self.eps
        #: The direction the nearest surface faces, outward: for a circle the
        #: azimuth about its centre, and for anything else what replaces it.
        self.angle = np.arctan2(-normal[:, 1], -normal[:, 0])
        self.bands = tuple(zip(edges[:-1], edges[1:]))

    ## Public

    def masks(self, low: float, high: float) -> "list[np.ndarray | None]":
        """One mask per inclusion, ``None`` where a shell is too thin to measure."""

        band = (self.gap >= low) & (self.gap < high)
        out = []
        for inclusion in range(self.inclusions):
            mask = band & (self.which == inclusion)
            out.append(mask if mask.sum() >= LEAST_CELLS else None)
        return out

    def modes(
        self, values: np.ndarray, low: float, high: float, mode: int
    ) -> "list[tuple[float, float]] | None":
        """``(relative amplitude, crest)`` per measurable inclusion in one shell.

        Relative to the field's own mean magnitude in that shell, so the number is
        a fractional modulation, comparable between frames, fields and runs, which
        the absolute amplitude of a potential growing by orders of magnitude is not.
        """

        out = []
        for mask in self.masks(low, high):
            if mask is None:
                continue
            chosen = values[mask]
            scale = float(np.abs(chosen).mean())
            if scale == 0.0:
                # Identically zero here: no modulation and no crest, and a crest
                # of zero degrees beside the mesh's axis would mean nothing.
                continue
            amplitude, crest = angular_mode(
                detrend(chosen, self.gap[mask]) / scale, self.angle[mask], mode
            )
            out.append((amplitude, crest))
        return out or None

    def innermost_structure(self, defect: np.ndarray) -> tuple[float, float, float]:
        """``(amplitude, selectivity, crest)`` of :data:`MODE` where it is imposed.

        The innermost shell, where a condition holding on the surface acts. The
        amplitude comes back with its ratio against the mean of :data:`NEIGHBOURS`,
        which says whether the structure is six-fold or merely rough, and the crest
        averaged over the inclusions, to compare against the mesh's own axis.
        ``nan`` throughout where the shell holds too few cells to read a mode from.
        """

        low, high = self.bands[0]
        found = self.modes(defect, low, high, MODE)
        if found is None:
            return float("nan"), float("nan"), float("nan")
        background = [
            one[0]
            for other in NEIGHBOURS
            for one in (self.modes(defect, low, high, other) or ())
        ]
        amplitude = float(np.mean([one[0] for one in found]))
        mean = float(np.mean(background)) if background else 0.0
        return (
            amplitude,
            amplitude / mean if mean else float("nan"),
            circular_mean([one[1] for one in found], MODE),
        )


__all__ = [
    "MODE",
    "NEIGHBOURS",
    "Shells",
    "angular_mode",
    "axis_offset",
    "circular_mean",
    "detrend",
    "lattice_axis",
]
