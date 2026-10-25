from types import SimpleNamespace

import pytest

from ovpsolver.solver import PhaseFieldSolver
from ovpsolver.solver.parameters import TimesteppingParameters


def solver_with(min_dt, proposal=None):
    """Just enough of a solver for the step floor, which reads nothing else."""

    fake = SimpleNamespace(
        parameters=TimesteppingParameters(min_dt=min_dt, warm_start_dt=1.0e-4),
        _proposal=proposal,
        _binding="water.phi",
    )
    fake._require_step = lambda dt, what: PhaseFieldSolver._require_step(fake, dt, what)
    return fake


def test_a_floor_at_or_above_the_first_sub_step_is_refused():
    with pytest.raises(ValueError, match="min_dt"):
        TimesteppingParameters(min_dt=1.0e-4, warm_start_dt=1.0e-4)


def test_zero_floor_lets_any_step_through():
    PhaseFieldSolver._require_step(solver_with(0.0), 1.0e-300, "after 1 retry")


def test_a_collapsed_bound_stops_the_run_instead_of_being_proposed():
    with pytest.raises(RuntimeError, match=r"bound water\.phi reported is dt=1\.0+e-09, below min_dt"):
        PhaseFieldSolver._propose(solver_with(1.0e-8, proposal=1.0e-9))
    assert PhaseFieldSolver._propose(solver_with(1.0e-8, proposal=1.0e-7)) == 1.0e-7
    # Before any step there is no bound, and warm_start_dt is above the floor.
    assert PhaseFieldSolver._propose(solver_with(1.0e-8)) == 1.0e-4


def test_a_retry_below_the_floor_stops_the_run():
    with pytest.raises(RuntimeError, match="after 3 retries is dt="):
        PhaseFieldSolver._require_step(solver_with(1.0e-8), 1.0e-9, "after 3 retries")
