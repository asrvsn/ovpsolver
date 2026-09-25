"""The time integrator of the gelation step's Burgers transport in ``z``.

The spatial operator is the compiled MUSCL/van Leer kernel
(``ovpsolver_ext.burgers_rhs_muscl_vanleer``); this advances it in time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ssp_rk3(
    state: np.ndarray,
    rhs: Callable[[np.ndarray, np.ndarray], None],
    timestep: float,
) -> np.ndarray:
    """Advance ``state`` one step with the Shu-Osher SSP-RK3 scheme.

    The RHS callable must have signature ``rhs(state, dstate_dt)`` and fill
    ``dstate_dt`` in-place. The input ``state`` is updated in-place and returned.
    """
    if timestep < 0.0:
        raise ValueError("timestep must be non-negative")
    if not np.issubdtype(state.dtype, np.floating):
        raise TypeError("state must have a floating-point dtype")

    state_0 = state.copy()
    dstate_dt = np.empty_like(state)

    rhs(state, dstate_dt)
    state[...] = state + timestep * dstate_dt

    rhs(state, dstate_dt)
    state[...] = 0.75 * state_0 + 0.25 * (state + timestep * dstate_dt)

    rhs(state, dstate_dt)
    state[...] = (1.0 / 3.0) * state_0 + (2.0 / 3.0) * (
        state + timestep * dstate_dt
    )

    return state
