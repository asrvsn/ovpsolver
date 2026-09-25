"""Turning one thing a mixture wants saved into one array of dofs per frame.

A value that is already a :class:`dolfinx.fem.Function` is written as its dof
vector, untouched: that vector *is* the run's state, so a reader can put it back
into the same space and hold the object the solver held. It is saved under the
mixture's own declaration of it -- the
:class:`~ovpsolver.fem.elements.ElementSpec` the solve was built from -- so the
description on disk is the declaration rather than a reconstruction of it.

An expression has no dofs of its own and is interpolated each frame into the
element its owner declared for it in its
:class:`~ovpsolver.fem.saveable.Saveable`. Nothing here inspects the expression
to choose that element: an expression of cell-constant states reads as
discontinuous of degree one, and would be stored as three values per triangle
for a quantity with one. Either way what lands on disk is dofs plus the element
they belong to, so reading is the same operation for a state and for a
diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx import fem

from ..elements import ElementSpec
from ..reduce import owned_dofs
from .element import check_saveable, describe, interpolation_expression

if TYPE_CHECKING:
    from collections.abc import Callable

    from dolfinx.mesh import Mesh

    from ..saveable import Saveable
    from ..types import Function

#: How a field's values were arrived at, recorded so a reader need not guess.
DIRECT = "direct"
INTERPOLATED = "interpolated"


@dataclass(slots=True)
class Series:
    """One saved field: where its values come from, and where they go."""

    name: str
    function: "Function"
    array: np.ndarray
    element: ElementSpec
    mode: str
    #: What an interpolated field is refreshed from each frame; ``None`` for a
    #: field written as its own dofs.
    expression: "fem.Expression | None" = None
    static: bool = False

    def write(self, frame: int) -> None:
        """This frame's values, unless the field already said them once.

        A static field holds one row, which frame zero fills. The caller writes
        every frame as usual, and later frames of a static field are no-ops.
        """

        if self.static and frame > 0:
            return
        if self.expression is not None:
            self.function.interpolate(self.expression)
            self.function.x.scatter_forward()
        row = 0 if self.static else frame
        self.array[row, :] = self.function.x.array[: self.array.shape[1]]

    @property
    def n_dofs(self) -> int:
        return int(self.array.shape[1])

    def describe(self) -> str:
        """This field's element in the notation a person writes it in."""

        return self.element.describe()


def series_for(
    name: str,
    expression: Any,
    entry: "Saveable",
    *,
    mesh: "Mesh",
    open_array: "Callable[[str, tuple[int, int]], np.ndarray]",
    n_frames: int,
) -> Series:
    """Everything needed to write ``expression`` every frame, opened and ready.

    ``entry`` is the owner's declaration and the only source of what the
    expression cannot say: the element to store the values in, and whether they
    change. A static field is allocated one row rather than ``n_frames``, so its
    array is the size of the field rather than of the run.
    """

    declared = entry.element
    direct = hasattr(expression, "function_space") and hasattr(expression, "x")
    if direct:
        function = expression
    else:
        function = fem.Function(
            fem.functionspace(mesh, declared.make_element(mesh.basix_cell()))
        )
    space = function.function_space

    # The saved element is read off the space rather than copied from the
    # declaration, which is written the way a person writes it -- ``Discontinuous
    # Lagrange``, no variant -- where basix reports the same element as family
    # ``P``, discontinuous, ``gll_warped``. Only the second is what a reader gets
    # when it describes the space it rebuilt, and a saved identity that does not
    # match that is a cache miss at best and a spurious interpolation at worst.
    # The declaration supplies what the space cannot report, the mesh domain and
    # the solve, and is otherwise checked by comparing built elements, which is
    # exact and needs no table of spellings.
    built = describe(space, name=name)
    if declared.make_element(space.mesh.basix_cell()) != space.ufl_element():
        raise ValueError(
            f"{name} is declared as {declared.describe()} but its function is on "
            f"{built.describe()}; the saved description would not rebuild it"
        )
    element = replace(built, solve=declared.solve, domain=declared.domain)
    check_saveable(name, element)

    array = open_array(name, (1 if entry.static else n_frames, owned_dofs(space)))
    return Series(
        name=name,
        function=function,
        array=array,
        element=element,
        mode=DIRECT if direct else INTERPOLATED,
        expression=None if direct else interpolation_expression(expression, space),
        static=entry.static,
    )


__all__ = ["DIRECT", "INTERPOLATED", "Series", "series_for"]
