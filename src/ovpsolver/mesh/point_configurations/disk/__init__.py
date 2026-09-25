"""Rules for placing equal inclusions in a disk.

Three of them, differing in how regular what they produce is, which is the one
difference that matters for a result about phase separation in a complex domain.
:func:`lloyd` is as even as a disk allows, regular enough to have its own lattice
directions. :func:`thinned_ginibre` is random yet still repulsive, so a result
surviving both does not depend on the regularity of the first.
:func:`log_gas_hard_sphere` is the family containing both: one inverse
temperature running from a hard-core fluid through the Ginibre ensemble to the
triangular lattice, at fixed count and density, so that the other two are points
on one scale rather than unrelated choices.
"""

from __future__ import annotations

from .lloyd import lloyd
from .log_gas import log_gas_hard_sphere
from .thinned_ginibre import thinned_ginibre

__all__ = [
    "lloyd",
    "log_gas_hard_sphere",
    "thinned_ginibre",
]
