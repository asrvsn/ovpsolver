"""A phase field whose monomers crosslink into a gel network.

Two velocities' worth of one conserved species, with the gelation state that
decides how much of it is which. A sibling of
:class:`~ovpsolver.model_b.phase_field.CHPhaseField`, not a refinement of it:
both derive from :class:`~ovpsolver.phase_field_system.phase_field.PhaseField`
directly.
"""

from .field import PolymerizingPhaseField
from .parameters import (
    PolymerizingPhaseFieldParameters,
    ZPartitionParameters,
)

__all__ = [
    "PolymerizingPhaseField",
    "PolymerizingPhaseFieldParameters",
    "ZPartitionParameters",
]
