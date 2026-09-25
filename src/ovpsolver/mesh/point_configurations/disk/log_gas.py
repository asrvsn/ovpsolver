"""Centres from a hard-core log-gas in the unit disk, at a tunable temperature.

The rule with a knob on regularity. The measure is the two-dimensional log-gas,
equivalently the one-component plasma or the beta-Ginibre ensemble, conditioned on
the particles not overlapping::

    P(x) proportional to  1{admissible} exp(-beta_n H(x)),

    H(x) = (1/N) sum_i V(x_i) + (1/(2 N^2)) sum_{i != j} W(x_i - x_j),

    V(x) = |x|^2 / 2,    W(x) = -log|x|,    beta_n = N^2 beta.

The prefactors are the nondimensionalization of Chafai and Ferre
(arXiv:1806.05985, eq. 1.1): they make ``H`` a functional of the empirical measure
alone, so it stays order one as the count grows and one ``beta`` means the same
thing at every count.

``beta`` is an inverse temperature, and the reason to parameterize regularity by
it is that it leaves the density alone. The equilibrium measure minimizes
``int V dmu + int int W dmu dmu`` and so has density proportional to the
Laplacian of ``V``: uniform on the unit disk, for every ``beta``. What ``beta``
sets is the microstructure, exactly: the pair correlation vanishes as ``r^beta``
at short range, which decides how often two inclusions leave a tight channel
between them, and the structure factor goes as ``|k|^2 / (2 pi beta rho)`` at long
wavelength, so ``1/beta`` is the amplitude of density fluctuations. Landmarks:
``beta = 2`` is the complex Ginibre ensemble, which :mod:`.thinned_ginibre`
samples exactly; the plasma freezes near ``beta = 140``; a relaxed arrangement
(:mod:`.lloyd`) sits at an effective ``beta`` of order a hundred; and large
``beta`` crystallizes to the triangular lattice, which the relaxed arrangement
shares as its limit.

Unlike the beta-Ginibre parameterization of the reference, whose equilibrium
measure is the disk of radius ``sqrt(beta/2)``, the confinement here carries no
``beta``. Putting ``beta`` in the temperature alone pins the support to the unit
disk, so changing it gives a different pattern at the same density rather than
the same pattern at a different scale, and nothing needs rescaling afterwards.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import constraints
from .lloyd import lloyd

#: Sweeps of Metropolis-adjusted Langevin before the sample is taken, one sweep
#: being a proposal for every particle in a fresh random order.
#:
#: Measured: the integrated autocorrelation time of ``H`` and of the radial second
#: moment stays under twenty-five sweeps over counts to 32, ``beta`` from a
#: quarter to 140, and separations up to the largest a relaxed start admits, so
#: this is some eighty times the slowest mode, and a few tenths of a second at the
#: counts a geometry asks for. Mixing does degrade towards jamming, but the
#: relaxed start refuses before that is reached.
BURN_IN_SWEEPS = 2000

#: Proposal size as a fraction of the proposal scale (see
#: :func:`log_gas_hard_sphere`). Deliberately below the fraction that maximizes
#: displacement per sweep: a short step costs more sweeps, which are cheap, and a
#: long one a chain that stops moving at close packing, which is silent.
PROPOSAL_STEP = 0.6

#: Floor on the free room in the proposal scale, as a fraction of the spacing.
#: Only reached when the particles are within a percent of touching at that
#: spacing, where the point is to keep proposing something rather than to
#: propose well.
MINIMUM_ROOM_FRACTION = 1.0e-2


def triangular_spacing(count: int) -> float:
    """Nearest-neighbour distance of a triangular lattice of ``count`` in the disk.

    The reference length for everything microscopic here: the densest arrangement
    of a given number has this spacing, so a separation is tight or loose relative
    to it and not to the disk.
    """

    return float(np.sqrt(2.0 * np.pi / np.sqrt(3.0) / count))


def particle_potential(
    points: npt.NDArray[np.float64], index: int, position: npt.NDArray[np.float64]
) -> tuple[float, npt.NDArray[np.float64], float]:
    """The part of ``H`` one particle contributes at ``position``, and its gradient.

    Everything the move of a single particle needs and nothing more: the other
    pairs' terms are common to both sides of the acceptance ratio and cancel.

    The distance to the nearest other particle comes back too, because the same
    pass over the pairs produces it and the hard core is tested against it. It is
    infinite for a lone particle, which has no pair to violate.
    """

    count = len(points)
    offsets = position - np.delete(points, index, axis=0)
    distances = np.linalg.norm(offsets, axis=1)
    if distances.size == 0:
        return 0.5 * float(position @ position) / count, position / count, np.inf

    energy = (
        0.5 * float(position @ position) / count
        - float(np.log(distances).sum()) / count**2
    )
    gradient = (
        position / count
        - (offsets / (distances * distances)[:, None]).sum(axis=0) / count**2
    )
    return energy, gradient, float(distances.min())


def log_gas_hard_sphere(
    count: int,
    *,
    disk_radius: float,
    inclusion_radius: float,
    beta: float,
    rng_seed: int,
    sweeps: int = BURN_IN_SWEEPS,
    step: float = PROPOSAL_STEP,
) -> npt.NDArray[np.float64]:
    """``count`` centres from the hard-core, hard-wall log-gas at inverse temperature ``beta``.

    ``inclusion_radius`` is the hard-sphere radius, in the units of
    ``disk_radius``; :mod:`.constraints` states the two conditions it imposes and
    how to ask for clearance. The measure is defined on the unit disk and the
    chain runs there, both radii divided by ``disk_radius`` going in and the
    centres multiplied by it coming back. That is done here and not by the
    caller, so that this rule takes the same arguments as the two beside it, and a
    forgotten scaling cannot silently mis-space an arrangement.

    A radius of zero asks for the log-gas itself, points with nothing to contain,
    whose edge fluctuations then reach outside the disk as they should: the disk
    is the support of the limiting measure, not a bound on a sample.

    Sampled by Metropolis-adjusted Langevin: propose one particle displaced along
    the gradient of the energy with Gaussian noise of the matching variance, then
    accept or reject against the exact ratio, so the chain leaves the measure
    invariant with no discretization bias. The target is zero on overlapping
    configurations, so rejecting such a proposal outright *is* the Metropolis step
    for the hard core, and no repulsive potential has to stand in for it.

    The hard core also tames the singular interaction. The force is bounded by
    ``1 / (2 inclusion_radius)`` on every configuration the chain can reach, so
    the plain overdamped proposal is enough, without the tamed or kinetic schemes
    the reference needs for particles that approach arbitrarily closely.

    One particle moves per proposal rather than the whole configuration. The cost
    is the same -- a sweep of single moves and one move of everything both touch
    every pair once -- but the admissible step is set by that particle's own free
    room rather than by the closest pair anywhere, which tightens with the count.
    Over a binding hard core the difference is two orders of magnitude in
    autocorrelation time.

    The chain starts from the relaxed arrangement of :mod:`.lloyd`, which does two
    jobs. It is the large-``beta`` end of this same family, so the chain melts a
    crystal rather than freezing a gas, the fast direction. And it is the
    feasibility gate: if the most even arrangement of this count cannot hold
    inclusions of this radius, no configuration the chain samples will either,
    and the refusal comes from there with its own diagnosis rather than from a
    chain that quietly never moves.
    """

    if beta <= 0.0:
        raise ValueError(f"beta must be positive, not {beta}")
    if float(step) <= 0.0:
        raise ValueError(f"step must be positive, not {step}")
    if float(disk_radius) <= 0.0:
        raise ValueError(f"disk_radius must be positive, not {disk_radius}")

    radius = float(inclusion_radius) / float(disk_radius)
    # Relaxed in the caller's units and then normalized, so that a refusal from
    # the feasibility gate quotes the radii that were asked for. The relaxation
    # is linear in the radius, so which side of the division it runs on changes
    # nothing else.
    points = (
        lloyd(
            count,
            disk_radius=float(disk_radius),
            inclusion_radius=float(inclusion_radius),
            rng_seed=rng_seed,
        )
        / disk_radius
    )

    generator = np.random.default_rng(rng_seed)
    beta_n = count**2 * float(beta)
    # Nothing to contain at zero radius, so nothing to keep inside either.
    wall = 1.0 - radius if radius > 0.0 else np.inf
    separation = 2.0 * radius
    # The proposal scale is the smaller of two lengths. The thermal one is how far
    # a particle sits from where it would at zero temperature: matching the
    # structure factor against an independently displaced lattice's gives a mean
    # square displacement ``1 / (pi beta rho)``, with ``rho = count / pi`` on the
    # unit disk. The geometric one is the room between two surfaces at the
    # crystalline spacing, which binds when a hot gas would propose further than
    # the hard core allows, since an overlapping proposal is rejected whatever its
    # energy. The smaller keeps one step size right from a dilute gas to close
    # packing without tuning the acceptance rate, which near jamming cannot be
    # raised anyway: pairs resting at contact reject a fixed share of moves
    # however short.
    spacing = triangular_spacing(count)
    thermal = 1.0 / np.sqrt(float(beta) * count)
    room = max(spacing - 2.0 * radius, MINIMUM_ROOM_FRACTION * spacing)
    tau = 0.5 * float(step) * min(thermal, room) ** 2
    spread = np.sqrt(2.0 * tau)

    for _ in range(int(sweeps)):
        for index in generator.permutation(count):
            energy, gradient, _ = particle_potential(points, index, points[index])
            drift = -tau * beta_n * gradient
            proposal = points[index] + drift + spread * generator.normal(size=2)
            if proposal @ proposal > wall * wall:
                continue
            moved, moved_gradient, closest = particle_potential(
                points, index, proposal
            )
            if closest < separation:
                continue

            # Asymmetric proposal, so the kernel does not cancel: the reverse move
            # drifts along the gradient at the proposed position, not this one.
            forward = proposal - points[index] - drift
            reverse = points[index] - proposal + tau * beta_n * moved_gradient
            log_ratio = -beta_n * (moved - energy) - (
                reverse @ reverse - forward @ forward
            ) / (4.0 * tau)
            if np.log(generator.uniform()) <= log_ratio:
                points[index] = proposal

    # Asserted rather than trusted, the loop above having imposed both conditions
    # already -- except at zero radius, where there is nothing to assert and the
    # edge fluctuations are outside the unit disk on purpose.
    points *= float(disk_radius)
    if radius == 0.0:
        return points
    return constraints.require_room(
        points,
        disk_radius=float(disk_radius),
        inclusion_radius=float(inclusion_radius),
        pattern=f"the log-gas at beta = {beta:g}",
    )
