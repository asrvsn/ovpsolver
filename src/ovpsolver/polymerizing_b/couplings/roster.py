"""Which entries of a mixture's roster are networks.

What the gel couplings share. Reading pair-keyed declarations is not about gels
and lives in :mod:`ovpsolver.parametric.pairs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..phase_field.parameters import PolymerizingPhaseFieldParameters

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...phase_field_system.phase_field import PhaseFieldParameters


def gel_positions(phases: "Sequence[PhaseFieldParameters]") -> tuple[int, ...]:
    """Roster positions of the networks, in the order gel-only matrices index them.

    Read off the roster rather than assumed contiguous, so nothing indexed by it
    depends on gels being declared last. The type is what says a phase is a
    network (see ``PolymerizingPhaseFieldParameters.crosslinking_rate``).
    """

    return tuple(
        position
        for position, parameters in enumerate(phases)
        if isinstance(parameters, PolymerizingPhaseFieldParameters)
    )
