"""Declarative parameters, and the objects they describe.

An experiment is built from a prototype -- a parameters class naming, for each
block, which concrete class reads it -- and a document supplying the numbers.
The two have the same shape, so no reader has to know the model's vocabulary:
what a block means is settled by the class the prototype put there, and a class
defined outside this package is read exactly as one defined inside it.
"""

from . import pairs
from .parameter import (
    REQUIRED,
    ZERO,
    Boolean,
    bind_form_mesh,
    Count,
    Fraction,
    Index,
    Integer,
    Names,
    Nonnegative,
    Number,
    Options,
    Pairs,
    Parameter,
    Positive,
    Text,
)
from .parameters import (
    PairsParameters,
    Parameters,
    ParametersList,
    Parametric,
)

__all__ = [
    "ZERO",
    "bind_form_mesh",
    "Boolean",
    "Count",
    "Fraction",
    "Index",
    "Integer",
    "Names",
    "Nonnegative",
    "Number",
    "Options",
    "Pairs",
    "pairs",
    "PairsParameters",
    "Parameter",
    "Parameters",
    "ParametersList",
    "Parametric",
    "Positive",
    "REQUIRED",
    "Text",
]
