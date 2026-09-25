"""Building a saved field's space, and checking it is the space that was saved.

A saved field is a block of numbers, and the numbers alone are meaningless: the
same three hundred doubles are one per cell, one per vertex or one per quadrature
point depending on an element nobody wrote down. Resampling onto a point cloud at
save time would be a lossy answer to a question that has an exact one, which is
to write the element down.

It is written down as an :class:`~ovpsolver.fem.elements.ElementSpec` -- the
type the solve declares its unknowns with, with no owner -- so a saved run and a
running solve describe their elements in one vocabulary, and an analysis asking a
saved field what it is gets the answer in the terms the mixture declared it in.

The round trip is checked rather than trusted: :func:`describe` rebuilds the
element and refuses a description whose dof count is not the space's, so an
element option the description does not carry fails when the run is written
rather than when it is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ufl
from dolfinx import fem

from ..elements import ElementDomain, ElementSpec
from ..reduce import owned_dofs

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh


def space_of(mesh: "Mesh", spec: ElementSpec) -> fem.FunctionSpace:
    """The function space ``spec`` names, on ``mesh``."""

    return fem.functionspace(mesh, spec.make_element(mesh.basix_cell()))


def interpolation_expression(
    expression: "ufl.core.expr.Expr | float", space: fem.FunctionSpace
) -> fem.Expression:
    """``expression`` compiled for interpolation into ``space``.

    Evaluated at the space's own interpolation points, so the interpolation is
    exact there. On the space's communicator, so that an expression with no mesh
    of its own -- a constant, such as the indicator of a geometry without
    inclusions -- compiles too.
    """

    return fem.Expression(
        ufl.as_ufl(expression),
        space.element.interpolation_points,
        comm=space.mesh.comm,
    )


def describe(space: fem.FunctionSpace, **attributes) -> ElementSpec:
    """The spec for ``space``, checked to rebuild the space it came from.

    The check is on the dof count, which is what a reader will index by and what
    goes wrong when an attribute is missed: an unnoticed symmetry, or a quadrature
    scheme that is not the default, shows up as the wrong number of values per
    cell.
    """

    spec = ElementSpec.of(space, **attributes)
    try:
        rebuilt = space_of(space.mesh, spec)
    except Exception as error:  # pragma: no cover - depends on basix internals
        raise ValueError(
            f"cannot describe this element well enough to rebuild it: {error}"
        ) from error

    if owned_dofs(rebuilt) != owned_dofs(space):
        raise ValueError(
            f"the description of this element rebuilds to {owned_dofs(rebuilt)} "
            f"dofs but the space has {owned_dofs(space)}; it has an attribute "
            f"{spec.to_dict()} does not carry"
        )
    return spec


def check_saveable(name: str, spec: ElementSpec) -> None:
    """Refuse a field whose space a reader would have no way to rebuild.

    Only the bulk mesh is saved. The outer boundary is a submesh derived from it,
    and deriving it again on the way in would renumber it -- the hazard the bulk
    mesh avoids by being shipped rather than regenerated -- so a field there would
    be a dof vector nothing can index.
    """

    if spec.domain is not ElementDomain.BULK:
        raise NotImplementedError(
            f"cannot save {name!r}: it lives on {spec.domain.name.lower()}, and "
            f"only the bulk mesh is written, so nothing could index its dofs on "
            f"the way back in; ship the submesh too if this is wanted"
        )
