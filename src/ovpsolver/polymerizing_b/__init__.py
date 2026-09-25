"""Polymerizing Model B(+), the mixture whose phases crosslink as they demix.

    python -m ovpsolver.polymerizing_b run spec.yaml

Model B with a reaction coordinate added to every polymerizing species, and with
the couplings that follow from having networks in the mixture: a Flory-Huggins
interaction that depends on how far each side has reacted, the drag a network
exerts on everything moving past it, and the entanglement of any two networks
that have grown through each other.
"""

from .couplings import (
    Drag,
    DragParameters,
    GelEntanglements,
    GelEntanglementsParameters,
    PolymerizingFloryHugginsCoupling,
    PolymerizingFloryHugginsPairParameters,
    PolymerizingFloryHugginsParameters,
)
from .parameters import (
    PolymerizingBCouplingsParameters,
    PolymerizingBParameters,
    PolymerizingRosterParameters,
)
from .system import PolymerizingB

__all__ = [
    "Drag",
    "DragParameters",
    "GelEntanglements",
    "GelEntanglementsParameters",
    "PolymerizingB",
    "PolymerizingBCouplingsParameters",
    "PolymerizingBParameters",
    "PolymerizingFloryHugginsCoupling",
    "PolymerizingFloryHugginsPairParameters",
    "PolymerizingFloryHugginsParameters",
    "PolymerizingRosterParameters",
]
