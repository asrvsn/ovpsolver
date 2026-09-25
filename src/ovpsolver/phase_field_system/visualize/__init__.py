"""Drawing a mixture: the domain it is posed on, and the output it produced.

Under ``phase_field_system`` because a figure is made of what a run is made of:
the spec says what the mixture has, the parameters objects say what a picture of
it shows, and the saved run says what there is to show. A plotter outside it
would have to re-derive all three from raw YAML, and so know about particular
mixtures.

The pieces, in the order data moves through them:

:mod:`.layers`
    what a figure is made of. Vocabulary only, so a parameters class can build a
    layer without importing anything heavy.
:mod:`.style`
    the numbers it is drawn to, which a mixture carries as ``plot_params``.
:mod:`.values`
    one frame of one layer, sampled onto something drawable by dolfinx.
:mod:`.render`
    values to colours, and colours to pixels, through pyvista.
:mod:`.figure`
    pixels to a page, through matplotlib.
:mod:`.domain`, :mod:`.phases`, :mod:`.phases_movie`, :mod:`.gelation`
    the figures and the movie, each an entry point. :mod:`.domain` needs no run:
    it draws the geometry a spec declares.

Only :mod:`.layers` and :mod:`.style` are imported with the package. Parameters
classes import from here, and reading a spec should not cost matplotlib and
pyvista; everything else is fetched on the attribute that names it. The
:func:`~.domain.domain` figure is not among those attributes, because
``domain`` is its module's name.
"""

from __future__ import annotations

import importlib
from typing import Any

from .layers import CHI_MODES, COMPLEMENT, MASK, OCCUPY, PLAIN, Layer
from .style import BESIDE_THE_RUN, PLOT_PARAMS, font_size

#: Attribute name to the module defining it, for the lazy fetch below.
_LAZY = {
    "CELL": "values",
    "Domain": "values",
    "Frame": "values",
    "POINT": "values",
    "VisualizeDomain": "domain",
    "VisualizeGelation": "gelation",
    "VisualizePhases": "phases",
    "VisualizePhasesMovie": "phases_movie",
    "movie": "phases_movie",
    "plot": "phases",
    "plot_gelation": "gelation",
}

__all__ = [
    "BESIDE_THE_RUN",
    "CELL",
    "CHI_MODES",
    "COMPLEMENT",
    "Domain",
    "Frame",
    "Layer",
    "MASK",
    "OCCUPY",
    "PLAIN",
    "PLOT_PARAMS",
    "POINT",
    "VisualizeDomain",
    "VisualizeGelation",
    "VisualizePhases",
    "VisualizePhasesMovie",
    "font_size",
    "movie",
    "plot",
    "plot_gelation",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
