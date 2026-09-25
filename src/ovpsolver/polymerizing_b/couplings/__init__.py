"""What couples the phases of a polymerizing mixture.

Three couplings, each a different kind of statement. :mod:`.flory_huggins` is
energetic, a mixing energy on the state space the reaction coordinate extends.
:mod:`.gel_entanglements` is energetic and mechanical, fields two networks share
and push back on each other through. :mod:`.drag` is a dissipation, the friction
a network exerts on everything moving past it.
"""

from .drag import Drag, DragParameters
from .flory_huggins import (
    PolymerizingFloryHugginsCoupling,
    PolymerizingFloryHugginsPairParameters,
    PolymerizingFloryHugginsParameters,
)
from .gel_entanglements import GelEntanglements, GelEntanglementsParameters

__all__ = [
    "Drag",
    "DragParameters",
    "GelEntanglements",
    "GelEntanglementsParameters",
    "PolymerizingFloryHugginsCoupling",
    "PolymerizingFloryHugginsPairParameters",
    "PolymerizingFloryHugginsParameters",
]
