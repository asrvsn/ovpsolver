"""The geometry a run is posed on: one signed distance per inclusion, in UFL.

:mod:`.base` is the interface -- what a geometry file defines, and how a signed
distance becomes a diffuse phase. :mod:`.step` and :mod:`.arcs` are one way of
writing such a file for a shape drawn in CAD: :mod:`.step` refits a solid's 2D
footprint so that every boundary piece is a line or a circular arc, the two curves
whose distance can be written in UFL, and :mod:`.arcs` writes that distance.
Nothing in :mod:`.base` knows about CAD.
"""

from __future__ import annotations

from .base import (
    ASSETS_ENTRY_POINT,
    ENTRY_POINT,
    OUTER_DOMAIN_ENTRY_POINT,
    evaluate,
    load,
    load_assets,
    load_outer_domain,
    phase,
    signed_distances,
    sphere,
)
from .arcs import signed_distance
from .step import (
    FitReport,
    Piece,
    fit_outputs,
    fit_step_to_arcs,
    read_pieces,
    sdfs_from_step,
)

__all__ = [
    "ASSETS_ENTRY_POINT",
    "ENTRY_POINT",
    "FitReport",
    "OUTER_DOMAIN_ENTRY_POINT",
    "Piece",
    "evaluate",
    "fit_outputs",
    "fit_step_to_arcs",
    "load",
    "load_assets",
    "load_outer_domain",
    "phase",
    "read_pieces",
    "sdfs_from_step",
    "signed_distance",
    "signed_distances",
    "sphere",
]
