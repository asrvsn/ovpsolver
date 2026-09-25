"""Measurements of a finished run, each one an alternative of the command line.

    python -m <mixture> analyze.roundtrip spec.yaml
    python -m <mixture> analyze.conservation spec.yaml
    python -m <mixture> analyze.stats spec.yaml water.phi

All of them read the run the spec names through the same reader the plots use, so
a diagnostic sees the functions the solver held, on the mesh it held them on. Each
returns a shell exit code, non-zero if a check failed, so an integration test is a
spec run followed by diagnostics over its output with nothing in between.

Importing this module registers every diagnostic, which is why the submodules are
imported here eagerly: the registry is what the CLI enumerates, and one that
filled on first use would list nothing.
"""

from __future__ import annotations

from .base import (
    Diagnostic,
    Unanswerable,
    declare_frames,
    frames_of,
)
from .conservation import Conservation
from .drag_floor import DragFloor
from .emptying import Emptying
from .fields import Fields
from .gelation import Gelation
from .mesh_correlation import MeshCorrelation
from .moments import Moments
from .positivity import Positivity
from .report import Report, Table
from .roundtrip import Roundtrip
from .saturation import Saturation
from .stats import Stats
from .surface_leakage import SurfaceLeakage
from .wetting import Wetting

__all__ = [
    "Conservation",
    "Diagnostic",
    "DragFloor",
    "Emptying",
    "Fields",
    "Gelation",
    "MeshCorrelation",
    "Moments",
    "Positivity",
    "Report",
    "Roundtrip",
    "Saturation",
    "Stats",
    "SurfaceLeakage",
    "Table",
    "Unanswerable",
    "Wetting",
    "declare_frames",
    "frames_of",
]
