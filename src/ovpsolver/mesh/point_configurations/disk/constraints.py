"""The two conditions every arrangement in the disk has to answer for.

Every inclusion of the stated radius lies wholly inside the disk, and no two
centres are closer than twice that radius, which is exactly non-overlap with
tangency as the worst case. One radius says both. A caller wanting clearance
between surfaces asks for a radius larger than the inclusions it will place, and
places the smaller ones at the centres it gets back, so nothing here has a
separate clearance to keep consistent. How much to ask for stays with the caller
that knows the scale: for a diffuse-domain run, the interface width ``2 eps``.

The rules differ in when they meet the conditions, not in what the conditions
are. :func:`reach` is the containment half alone, for a rule that samples inside
it; :func:`require_room` checks both after the fact, for a rule that settles its
arrangement before any radius is named, and for one that imposes them while
sampling but asserts the result rather than trusting it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reach(disk_radius: float, inclusion_radius: float) -> float:
    """How far a centre may sit from the origin and still hold its inclusion.

    A zero radius is allowed and means there is nothing to contain: the rule is
    then about points rather than inclusions, and the whole disk is within reach.
    """

    disk_radius = float(disk_radius)
    inclusion_radius = float(inclusion_radius)
    if disk_radius <= 0.0:
        raise ValueError(f"disk_radius must be positive, not {disk_radius}")
    if inclusion_radius < 0.0:
        raise ValueError(
            f"inclusion_radius must not be negative, not {inclusion_radius}"
        )

    room = disk_radius - inclusion_radius
    if room <= 0.0:
        raise ValueError(
            f"an inclusion of radius {inclusion_radius:g} does not fit in a disk "
            f"of radius {disk_radius:g}"
        )
    return room


def require_room(
    points: npt.NDArray[np.float64],
    *,
    disk_radius: float,
    inclusion_radius: float,
    pattern: str,
    advice: str = "",
) -> npt.NDArray[np.float64]:
    """Refuse an arrangement that cannot hold inclusions of the stated size.

    ``advice`` is what the rule has to say about its own refusal, since why an
    arrangement fell short, and what to do about it, belong to the rule and not
    to the conditions.
    """

    room = reach(disk_radius, inclusion_radius)
    furthest = float(np.linalg.norm(points, axis=1).max())
    if furthest > room:
        raise ValueError(
            f"{pattern} left a centre {furthest:.4g} from the origin, past the "
            f"{room:.4g} at which an inclusion of radius {inclusion_radius:g} "
            f"still clears the boundary of a disk of radius {disk_radius:g}. Use "
            f"fewer inclusions, a smaller radius, or a larger disk"
        )

    required = 2.0 * float(inclusion_radius)
    if len(points) < 2 or required <= 0.0:
        return points

    offsets = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(offsets, axis=2)
    np.fill_diagonal(distances, np.inf)
    closest = float(distances.min())
    if closest < required:
        raise ValueError(
            f"{pattern} left two of {len(points)} centres {closest:.4g} apart, "
            f"short of the {required:.4g} that inclusions of radius "
            f"{inclusion_radius:g} need to stay clear of one another. {advice}Use "
            f"fewer inclusions, a smaller radius, or a larger disk"
        )
    return points
