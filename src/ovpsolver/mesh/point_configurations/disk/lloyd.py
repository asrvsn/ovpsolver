"""Centres spread as evenly through the disk as it allows.

The most regular arrangement here, and the one the others are measured against:
a centroidal tessellation is locally hexagonal away from the boundary, so it has
its own lattice directions. That is what makes a disordered rule worth having
beside it, and what makes it the natural starting configuration for one -- see
:mod:`.log_gas`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import constraints

#: Points standing in for the disk's area when relaxing. They only integrate each
#: Voronoi cell's centroid, so they want to be numerous enough that the answer
#: stops depending on them, not tuned.
SAMPLES_PER_INCLUSION = 128
MINIMUM_SAMPLES = 4096


def lloyd(
    count: int,
    *,
    disk_radius: float,
    inclusion_radius: float,
    rng_seed: int,
    iterations: int = 500,
) -> npt.NDArray[np.float64]:
    """``count`` centres spread as evenly through the disk as it allows.

    Lloyd relaxation: start from a uniform random sample and repeatedly move each
    generator to the centroid of its Voronoi cell, which converges towards a
    centroidal tessellation -- locally hexagonal away from the boundary, and as
    close to equally spaced as a disk permits. The tessellation is of the whole
    disk, so the spacing reflects the domain the inclusions are spread through.

    Seeded rather than deterministic, so that a spec can ask for several
    arrangements of the same kind and see which of its conclusions depend on one.
    The relaxation contracts, so the seed's influence fades with ``iterations``,
    and two seeds give arrangements alike in spacing but not in orientation.

    ``inclusion_radius`` does not reach the arrangement: it only decides
    afterwards whether inclusions of that size fit in what came out
    (:func:`~.constraints.require_room`). A refusal is therefore a statement
    about the size rather than the pattern, and means relaxation will not reach
    the separation asked for -- not that no arrangement could. Lloyd fills the
    area evenly rather than packing densely, reaching about two thirds of the
    spacing a triangular lattice of the same density would have, and relaxing
    longer does not help; fewer inclusions, a smaller radius or a larger disk do.

    No centre can leave the disk: a Voronoi centroid lies in the convex hull of
    the cloud points assigned to it, and the cloud is drawn inside the domain.
    """

    if count <= 0:
        raise ValueError(f"count must be positive, not {count}")
    generator = np.random.default_rng(rng_seed)

    # Over the whole domain, which is the region being divided. A cloud over the
    # smaller disk the centres have to fit inside would tessellate *that* disk,
    # pulling every centre inward and leaving the inclusions clustered near the
    # origin inside an empty annulus.
    points = _disk_sample(generator, count, float(disk_radius))
    cloud = _disk_sample(
        generator,
        max(MINIMUM_SAMPLES, SAMPLES_PER_INCLUSION * count),
        float(disk_radius),
    )
    for _ in range(int(iterations)):
        # Nearest generator per sample, then the mean of each generator's own
        # samples. A cell with no sample keeps its point rather than vanishing.
        owner = np.argmin(
            np.linalg.norm(cloud[:, None, :] - points[None, :, :], axis=2), axis=1
        )
        for index in range(count):
            assigned = cloud[owner == index]
            if assigned.size:
                points[index] = assigned.mean(axis=0)

    return constraints.require_room(
        np.ascontiguousarray(points),
        disk_radius=disk_radius,
        inclusion_radius=inclusion_radius,
        pattern="Lloyd relaxation",
        advice=(
            "Relaxation spreads points to fill the area rather than to maximize "
            "the smallest gap, so a tighter arrangement may exist -- but not one "
            "it will find. "
        ),
    )


def _disk_sample(
    generator: np.random.Generator, count: int, radius: float
) -> npt.NDArray[np.float64]:
    """``count`` points uniform over the disk, by area rather than by radius."""

    angle = generator.uniform(0.0, 2.0 * np.pi, size=count)
    distance = radius * np.sqrt(generator.uniform(0.0, 1.0, size=count))
    return np.column_stack((distance * np.cos(angle), distance * np.sin(angle)))
