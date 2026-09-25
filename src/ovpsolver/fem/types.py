from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from dolfinx import fem

    Function: TypeAlias = fem.Function
    FunctionSpace: TypeAlias = fem.FunctionSpace
else:
    Function: TypeAlias = Any
    FunctionSpace: TypeAlias = Any
