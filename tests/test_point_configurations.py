import numpy as np
import pytest

from ovpsolver.mesh import lloyd, log_gas_hard_sphere
from ovpsolver.mesh.point_configurations.disk.log_gas import triangular_spacing

# Short chains: enough to leave the crystalline start, since these check the
# invariants and the trend and not the fine structure of the measure.
SWEEPS = 200


def closest_pair(points):
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


@pytest.mark.parametrize("beta", [0.5, 2.0, 140.0])
def test_log_gas_hard_sphere_holds_both_conditions(beta):
    radius = 0.25 * triangular_spacing(8)
    points = log_gas_hard_sphere(
        8,
        disk_radius=1.0,
        inclusion_radius=radius,
        beta=beta,
        rng_seed=0,
        sweeps=SWEEPS,
    )
    assert len(points) == 8
    assert np.linalg.norm(points, axis=1).max() <= 1.0 - radius
    assert closest_pair(points) >= 2.0 * radius


def test_log_gas_hard_sphere_reproduces_from_the_seed():
    draw = lambda seed: log_gas_hard_sphere(
        6,
        disk_radius=1.0,
        inclusion_radius=0.1,
        beta=2.0,
        rng_seed=seed,
        sweeps=SWEEPS,
    )
    assert np.array_equal(draw(0), draw(0))
    assert not np.allclose(draw(0), draw(1))


def test_log_gas_hard_sphere_regularity_increases_with_beta():
    """The one thing ``beta`` is for: more of it, more even spacing."""

    spacings = [
        np.mean(
            [
                closest_pair(
                    log_gas_hard_sphere(
                        12,
                        disk_radius=1.0,
                        inclusion_radius=0.0,
                        beta=beta,
                        rng_seed=seed,
                        sweeps=SWEEPS,
                    )
                )
                for seed in range(3)
            ]
        )
        for beta in (0.5, 8.0, 140.0)
    ]
    assert spacings[0] < spacings[1] < spacings[2]
    # And the cold end approaches the relaxed arrangement it shares a limit with.
    relaxed = closest_pair(
        lloyd(12, disk_radius=1.0, inclusion_radius=0.0, rng_seed=0)
    )
    assert spacings[2] == pytest.approx(relaxed, rel=0.2)


def test_log_gas_hard_sphere_density_fills_the_unit_disk():
    """The confinement is normalized so the support is the unit disk at any beta.

    Checked against the complex Ginibre ensemble, which this is at ``beta = 2``: its
    mean radius at eight points is 0.6974 by the exact finite-count density.
    """

    radii = np.concatenate(
        [
            np.linalg.norm(
                log_gas_hard_sphere(
                    8,
                    disk_radius=1.0,
                    inclusion_radius=0.0,
                    beta=2.0,
                    rng_seed=seed,
                    sweeps=SWEEPS,
                ),
                axis=1,
            )
            for seed in range(12)
        ]
    )
    assert radii.mean() == pytest.approx(0.6974, abs=0.08)


def test_log_gas_hard_sphere_refuses_what_no_arrangement_holds():
    radius = 0.45 * triangular_spacing(10)
    with pytest.raises(ValueError, match="Lloyd relaxation"):
        log_gas_hard_sphere(
            10, disk_radius=1.0, inclusion_radius=radius, beta=2.0, rng_seed=0
        )


def test_log_gas_hard_sphere_rejects_a_zero_temperature():
    with pytest.raises(ValueError, match="beta must be positive"):
        log_gas_hard_sphere(
            4, disk_radius=1.0, inclusion_radius=0.1, beta=0.0, rng_seed=0
        )
