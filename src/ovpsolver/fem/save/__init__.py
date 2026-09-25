"""Storing a run's fields, and reading them back as the functions they were.

What a run saves is whatever its mixture declared, and the mixture is the
user's, so the saver is abstract over it: it takes a mapping of name to value and
:class:`~ovpsolver.fem.saveable.Saveable`, and every decision after that follows
from the declaration.

The format keeps as little judgement as it can. The mesh goes out once, each
field's dof vector goes out once a frame, and the
:class:`~ovpsolver.fem.elements.ElementSpec` that makes sense of the numbers goes
into the metadata beside them -- the same type the solve declared the field with,
so there is one vocabulary for elements and not two. Nothing is resampled or
reduced, so a reader gets the solver's own functions back and can ask dolfinx for
whatever it actually wanted.
"""

from __future__ import annotations

from .element import (
    check_saveable,
    describe,
    interpolation_expression,
    space_of,
)
from .field import DIRECT, INTERPOLATED, Series, series_for
from .read import DEFAULT_QUADRATURE_DEGREE, Field, Run, project
from .saver import (
    MESH_FILE,
    GEOMETRY_FILE,
    META_FILE,
    SPEC_FILE,
    STEPS_FILE,
    TIMES_FILE,
    Saver,
    describe_version,
    source_version,
)

__all__ = [
    "DEFAULT_QUADRATURE_DEGREE",
    "DIRECT",
    "INTERPOLATED",
    "MESH_FILE",
    "GEOMETRY_FILE",
    "META_FILE",
    "SPEC_FILE",
    "STEPS_FILE",
    "TIMES_FILE",
    "Field",
    "Run",
    "Saver",
    "Series",
    "check_saveable",
    "describe",
    "describe_version",
    "interpolation_expression",
    "project",
    "series_for",
    "source_version",
    "space_of",
]
