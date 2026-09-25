"""Flory-Huggins on the extended state space of a crosslinking mixture.

Model B's mixing energy with the reaction coordinate added. The interaction
between two phases is a bilinear function of how far each has reacted,

    chi_ij(a_i, a_j) = chi_00 + chi_10 a_i + chi_01 a_j + chi_11 a_i a_j,

and what that costs is

    g_ij = chi_00 phi_i phi_j + chi_10 r_i phi_j + chi_01 r_j phi_i + chi_11 r_i r_j,

in the reacted-site fraction ``r_i = phi_i alpha_i = phi_i - (nu_i/f_i) c_x_i``.
Since ``r_i`` is affine in the state, ``g_ij`` is still a quadratic form, on the
larger space ``y = (phi_1, c_x_1, phi_2, c_x_2, ...)`` and with a constant
Hessian there (the "FH subspace"), which is what makes the convex split a fixed
matrix. Each phase gets a two-slot block spanned by the composition direction
``(1, 0)`` and its own reaction direction ``(1, -nu/f)``, and the four
coefficients weight the four outer products of the two.

A two-slot block is a different space from Model B's composition, not an
extension of it, so this and :mod:`ovpsolver.model_b.couplings.flory_huggins`
both derive from :mod:`ovpsolver.phase_field_system.couplings.flory_huggins` and
neither from the other.
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
from ..phase_field import PolymerizingPhaseField, PolymerizingPhaseFieldParameters

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ufl.core.expr import Expr

    from ...phase_field_system.phase_field import PhaseFieldParameters

#: The composition direction in a phase's two-slot block: ``phi`` itself.
COMPOSITION = np.array([1.0, 0.0])


class PolymerizingFloryHugginsPairParameters(FloryHugginsPairParameters):
    """The four coefficients of the bilinear ``chi_ij`` between two phases.

    ``chi_ab`` multiplies ``alpha_i^a alpha_j^b``: the first index is the first
    phase's and the second the second's. ``chi_00`` is the whole of the
    interaction when neither phase reacts, and is what Model B calls ``chi``.
    """

    chi_00: float = Number(0.0)
    chi_10: float = Number(0.0)
    chi_01: float = Number(0.0)
    chi_11: float = Number(0.0)


class PolymerizingFloryHugginsParameters(FloryHugginsParameters):
    """Mixing coefficients, one entry per unordered pair, split by gelation state."""

    item = PolymerizingFloryHugginsPairParameters

    def applicable(
        self, phases: "Sequence[PhaseFieldParameters]", first: int, second: int
    ) -> frozenset[str]:
        """Which coefficients a pair may carry, given which sides react.

        A ``1`` index multiplies a reaction extent, which a phase that does not
        crosslink does not have.
        """

        names = {"chi_00"}
        reacts = tuple(
            isinstance(phases[position], PolymerizingPhaseFieldParameters)
            for position in (first, second)
        )
        if reacts[0]:
            names.add("chi_10")
        if reacts[1]:
            names.add("chi_01")
        if all(reacts):
            names.add("chi_11")
        return frozenset(names)

    def values_of(
        self, block: PolymerizingFloryHugginsPairParameters, *, flipped: bool
    ) -> dict[str, float]:
        """One pair's coefficients, oriented so the first index is the lower one.

        Writing the pair the other way round swaps ``chi_10`` and ``chi_01``;
        the other two are symmetric in the pair.
        """

        values = super().values_of(block, flipped=flipped)
        if flipped:
            values["chi_10"], values["chi_01"] = values["chi_01"], values["chi_10"]
        return values


class PolymerizingFloryHugginsCoupling(
    FloryHugginsCoupling[PolymerizingFloryHugginsParameters]
):
    """``(k_B_T/2) y^T H y`` over compositions *and* crosslink concentrations."""

    def hessian(self) -> np.ndarray:
        """``H`` on the two-slot state, from the four coefficients per pair.

        Symmetrized on the diagonal blocks and mirrored off them, since ``H`` is
        a second derivative while the coefficients are given once per unordered
        pair.
        """

        directions = self.reaction_directions()
        matrices = self.parameters.matrices
        chi_00, chi_10 = matrices["chi_00"], matrices["chi_10"]
        chi_01, chi_11 = matrices["chi_01"], matrices["chi_11"]

        count = len(directions)
        matrix = np.zeros((2 * count, 2 * count))
        for i in range(count):
            rows = slice(2 * i, 2 * i + 2)
            for j in range(i, count):
                columns = slice(2 * j, 2 * j + 2)
                block = (
                    chi_00[i, j] * np.outer(COMPOSITION, COMPOSITION)
                    + chi_10[i, j] * np.outer(directions[i], COMPOSITION)
                    + chi_01[i, j] * np.outer(COMPOSITION, directions[j])
                    + chi_11[i, j] * np.outer(directions[i], directions[j])
                )
                if i == j:
                    block = block + block.T
                matrix[rows, columns] = block
                if i != j:
                    matrix[columns, rows] = block.T
        return matrix

    def slots(self, time_level: str) -> list["Expr | None"]:
        """``y``: each phase's ``phi``, then its ``c_x``, or ``None`` for a phase
        that does not crosslink."""

        slots: list["Expr | None"] = []
        for field in self.phase_fields:
            slots.append(field.variable("phi", time_level=time_level))
            slots.append(
                field.variable("c_x", time_level=time_level)
                if isinstance(field, PolymerizingPhaseField)
                else None
            )
        return slots

    def reaction_directions(self) -> list[np.ndarray]:
        """``r_i`` in the phase's own two slots: ``(1, -nu/f)`` per gel, and zero
        for a phase that does not react, the algebra's counterpart of
        :meth:`PolymerizingFloryHugginsParameters.applicable`."""

        directions = []
        for field in self.phase_fields:
            if isinstance(field, PolymerizingPhaseField):
                parameters = field.parameters
                directions.append(
                    np.array(
                        [
                            1.0,
                            -float(parameters.monomer_volume)
                            / int(parameters.monomer_functionality),
                        ]
                    )
                )
            else:
                directions.append(np.zeros(2))
        return directions
