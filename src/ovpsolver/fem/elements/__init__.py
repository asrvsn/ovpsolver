"""Declaring finite elements, and the discretization they are realized in.

:mod:`.spec` is the declaration layer: what an element is, and which solve
determines it. :mod:`.owner` holds the objects that declare elements, and the
functions the solver binds to them. :mod:`.dg0` is the cell-constant
discretization every state of the mixture uses, whose monotonicity is a facet
rule expressible in UFL.
"""

from .owner import ElementOwner, Fields
from .spec import (
    ElementDomain,
    ElementSpec,
    Solve,
    StateElement,
    StaticElement,
    VelocityElement,
)

__all__ = [
    "ElementDomain",
    "ElementOwner",
    "ElementSpec",
    "Fields",
    "Solve",
    "StateElement",
    "StaticElement",
    "VelocityElement",
]
