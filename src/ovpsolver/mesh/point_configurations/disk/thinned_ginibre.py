"""Centres at Ginibre eigenvalues, thinned until none of them overlap.

Sampled from a random matrix rather than from a Markov chain, which makes this
the one rule here that is exact at finite count, and so the reference for the
sampled log-gas of :mod:`.log_gas`. The price is that thinning fixes neither the
count nor the statistics: what survives is a different point process from what
was drawn, closer to Poisson the harder it has to thin.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import constraints


def thinned_ginibre(
    matrix_size: int,
    *,
    disk_radius: float,
    inclusion_radius: float,
    rng_seed: int,
) -> npt.NDArray[np.float64]:
    """Centres at Ginibre eigenvalues, thinned until none of them overlap.

    The eigenvalues of a complex Gaussian matrix scaled by ``1/sqrt(n)`` fill the
    unit disk with a *repulsive* point pattern: unlike Poisson points they avoid
    one another at short range, and unlike a relaxed arrangement they are
    genuinely random. Disordered without being clumped, which makes it a fair
    test of whether a result depends on the lattice-like regularity of an evenly
    spread arrangement.

    Repulsion is not a minimum separation, so the pattern is thinned, Matern
    type II: a point is kept only if no *earlier* point in an independent random
    order lies within the exclusion distance. The independence of that order is
    what makes the result a thinning of the pattern rather than an artefact of
    the order the eigensolver returned. Points too near the boundary are dropped
    outright rather than thinned, since an inclusion sticking out of the domain
    is not one the wetting condition can be posed on.

    How many survive is not the caller's choice -- that is what thinning means.
    ``matrix_size`` sets the density and the count follows, so a geometry wanting
    a particular number has to look at what it got. Asking for a separation the
    pattern rarely offers thins most of it away and drives what is left towards
    Poisson, the opposite of the reason to use it;
    :func:`~.log_gas.log_gas_hard_sphere` keeps the count and conditions the
    measure instead.
    """

    if matrix_size < 1:
        raise ValueError(f"matrix_size must be at least 1, not {matrix_size}")
    room = constraints.reach(disk_radius, inclusion_radius)
    separation = 2.0 * float(inclusion_radius)
    generator = np.random.default_rng(rng_seed)

    # Standard complex Gaussian entries: real and imaginary parts of variance
    # one half, the normalization under which the spectrum fills the unit disk.
    real = generator.normal(scale=np.sqrt(0.5), size=(matrix_size, matrix_size))
    imaginary = generator.normal(scale=np.sqrt(0.5), size=(matrix_size, matrix_size))
    spectrum = np.linalg.eigvals((real + 1j * imaginary) / np.sqrt(matrix_size))
    points = np.column_stack((spectrum.real, spectrum.imag)) * float(disk_radius)

    # Inside first, so that outliers do not exclude interior neighbours they were
    # never going to be kept alongside. The circular law is asymptotic, so a
    # modest matrix leaves a few outside.
    points = points[np.linalg.norm(points, axis=1) <= room]
    if len(points) == 0:
        raise ValueError(
            f"no eigenvalue of a {matrix_size}x{matrix_size} matrix sits far "
            f"enough inside a disk of radius {disk_radius:g} to hold an "
            f"inclusion of radius {inclusion_radius:g}; the inclusions are too "
            f"big for the pattern"
        )

    kept: list[int] = []
    for candidate in generator.permutation(len(points)):
        position = points[candidate]
        if all(
            np.linalg.norm(position - points[other]) >= separation for other in kept
        ):
            kept.append(int(candidate))
    return np.ascontiguousarray(points[sorted(kept)])
