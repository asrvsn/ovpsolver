"""What couples phase fields to each other, in the abstract.

:class:`Coupling` is the base every cross-phase interaction derives from, and
:mod:`.flory_huggins` is the one kind of coupling the core knows the shape of --
a quadratic form over whatever state a mixture's pairs couple -- without knowing
what that state is. The concrete couplings live with the mixtures that have them,
each in a ``couplings`` package of its own, laid out exactly as this one is.
"""

from .base import Coupling
from .flory_huggins import (
    FloryHugginsCoupling,
    FloryHugginsPairParameters,
    FloryHugginsParameters,
)

__all__ = [
    "Coupling",
    "FloryHugginsCoupling",
    "FloryHugginsPairParameters",
    "FloryHugginsParameters",
]
