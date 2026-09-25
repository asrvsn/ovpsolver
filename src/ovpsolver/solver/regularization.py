"""Regularizations that keep the rate problem's coefficients finite and convex."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ufl

if TYPE_CHECKING:
    from ufl.core.expr import Expr


def log_reg(value: "Expr", delta: float) -> "Expr":
    """``log_delta(value)``: ``ln(value)``, continued linearly below ``delta``.

    Bare ``ln`` returns NaN on a Newton step to ``phi <= 0`` and the solve is
    over. The continuation stays finite with its derivative bounded by
    ``1 / delta``, which is also why ``delta`` cannot be arbitrarily small: at
    ``1e-12`` a Newton overshoot into the continued region produces an enormous
    residual and backtracking stalls, while too large a value blunts the entropy
    near vacuum. Values around ``1e-4`` sit between the two.
    """

    if float(delta) <= 0.0:
        raise ValueError("log_reg delta must be positive")

    return ufl.conditional(
        ufl.gt(value, delta),
        ufl.ln(value),
        ufl.ln(delta) + (value - delta) / delta,
    )


def smooth_max(value: "Expr", delta: float) -> "Expr":
    """``max(value, delta)``, smoothed: ``sqrt(value^2 + delta^2)``.

    For a coefficient that has to stay bounded away from zero where what it
    measures vanishes, as the drag floor on the Darcy measure ``chi_eps phi``. A
    ``max`` puts a corner in the coefficient where it switches, and the velocity
    row inherits that corner as a jump in its derivative; this has none. It is
    ``delta`` at zero, within ``(sqrt(2) - 1) delta`` of the ``max`` everywhere,
    and above the floor exceeds ``value`` only by about ``delta^2 / 2 value``.

    Monotone in a non-negative ``value``, so it does not reintroduce the well an
    additive ramp makes against a term growing linearly. Even in ``value``, so a
    carrier a roundoff below zero still gets a positive coefficient.
    """

    if float(delta) <= 0.0:
        raise ValueError("smooth_max delta must be positive")

    return ufl.sqrt(value * value + delta * delta)


def log_reg_ratio(numerator: "Expr", denominator: "Expr", delta: float) -> "Expr":
    """Regularized relative entropy ``numerator ln(numerator / denominator)``.

    ``x ln(x / y)`` is the perspective ``y h(x / y)`` of the convex
    ``h(t) = t ln t``, and so jointly convex. Regularizing the *ratio* keeps it
    so: ``t log_delta(t)`` is still convex, and shifting the denominator by
    ``delta`` is an affine precomposition. Expanded into
    ``x log_delta(x) - x log_delta(y)`` instead, the exact Hessian is only
    positive *semi*definite -- rank one, degenerate along the scaling ray -- so
    it has no margin to absorb an independent perturbation of each argument and
    goes indefinite for ``y < delta`` or ``x < delta / 2``.

    Inside the log ``delta`` caps the resolvable ratio; in the denominator it
    floors a vanishing density.
    """

    return numerator * log_reg(numerator / (denominator + delta), delta)
