"""Advection of the mixture's variables, and the rows that realize it.

More abstract than a phase field, and deliberately so: a :class:`Transported`
variable belongs to an :class:`~ovpsolver.fem.elements.ElementOwner`, which is
as much as it needs to know about whom it rides. A phase carries its own volume
fraction this way, and so does the locking between two of them.
"""

from .flux import Flux
from .transported import (
    LieTransported,
    Transported,
    injected,
    potential_of,
    transport_step,
)

__all__ = [
    "Flux",
    "LieTransported",
    "Transported",
    "injected",
    "potential_of",
    "transport_step",
]
