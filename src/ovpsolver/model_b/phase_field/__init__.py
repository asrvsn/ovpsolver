"""The Cahn-Hilliard species: a phase field carried by a single Darcy velocity.

Here rather than in the core, because which kinds of phase exist is a statement
about a mixture. The polymerizing mixture uses it too, for its species that do not
crosslink (its ``sol`` roster).

A sibling of :class:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseField`,
not its base: both derive from
:class:`~ovpsolver.phase_field_system.phase_field.PhaseField` directly.
"""

from .field import CHPhaseField
from .parameters import CHPhaseFieldParameters

__all__ = ["CHPhaseField", "CHPhaseFieldParameters"]
