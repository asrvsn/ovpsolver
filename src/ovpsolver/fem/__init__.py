"""The finite-element layer: what a discretization is, beneath any particular mixture.

``elements``
    what an object declares when it wants a function space, the
    ``prev``/``next``/rate bundles the solver binds those declarations to, and
    the cell-constant (DG0) facet rules every transport and energy row shares.
``reduce``
    the collective reductions a residual needs -- global minima, maxima and
    sums over the mesh partition -- so that a bound computed in parallel is the
    same number as the one computed in serial.
``saveable``, ``save``
    what an owner offers to write, and the format a run is written in and read
    back from.
``jit``
    clearing the form-compilation cache of claims an interrupted run abandoned.
``types``
    the dolfinx aliases the annotations are written in.
"""

from __future__ import annotations

from .reduce import (
    assemble_nodal_values,
    function_max,
    function_min_max,
    global_max,
    global_min,
    global_sum,
)

__all__ = [
    "assemble_nodal_values",
    "function_max",
    "function_min_max",
    "global_max",
    "global_min",
    "global_sum",
]
