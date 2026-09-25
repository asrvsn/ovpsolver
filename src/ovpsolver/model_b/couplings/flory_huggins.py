"""Flory-Huggins between phases that only ever flow: one coefficient per pair.

Nothing in the mixture reacts, so a pair interaction is a number and the state
the quadratic form acts on is the composition itself,

    y = (phi_1, phi_2, ...),      H[i, j] = chi_ij.

Building ``H``, splitting it convex-concave and expanding the quadratic into rows
are :mod:`ovpsolver.phase_field_system.couplings.flory_huggins`'s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...parametric import Number
from ...phase_field_system.couplings.flory_huggins import (
    FloryHugginsCoupling,
    FloryHugginsPairParameters,
    FloryHugginsParameters,
)

if TYPE_CHECKING:
    from ufl.core.expr import Expr


class CHFloryHugginsPairParameters(FloryHugginsPairParameters):
    """The mixing coefficient between two phases.

    The constant term of ``chi`` as a function of the reaction extents, which is
    all of it when neither phase reacts: a crosslinking mixture's ``chi_00``,
    named without an index because here there is nothing to index against.
    """

    chi: float = Number(0.0)


class CHFloryHugginsParameters(FloryHugginsParameters):
    """Mixing coefficients, one per unordered pair of phases."""

    item = CHFloryHugginsPairParameters


class CHFloryHugginsCoupling(FloryHugginsCoupling[CHFloryHugginsParameters]):
    """``(k_B_T/2) y^T H y`` over the mixture's compositions."""

    def hessian(self) -> np.ndarray:
        """``H``, which here is the matrix of ``chi`` itself.

        Zero on the diagonal, since no pair names a phase twice, so the quadratic
        expands to ``sum_{i<j} chi_ij phi_i phi_j``: a crosslinking mixture's
        pair interaction with every reacted-site fraction dropped.
        """

        return self.parameters.matrices["chi"]

    def slots(self, time_level: str) -> "list[Expr | None]":
        """``y``: each phase's composition, at one time level."""

        return [
            field.variable("phi", time_level=time_level) for field in self.phase_fields
        ]
