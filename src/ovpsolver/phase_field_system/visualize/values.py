"""One frame of one layer, sampled onto something drawable.

A picture needs one array per layer, all on the same points. Saved dofs come back
as the functions they were, and dolfinx puts them on one of two sets of points:

* cells, for nearly everything. Every phase, chemical state and the pressure is
  cell-constant, so one space holds them all indexed the same way, and
  compositing two layers is adding two vectors. Interpolating them to P1 would
  smooth the steps they exist to show.
* P1 vertices, for the continuous fields: the velocities, and the inclusion
  indicator when the domain alone is drawn.

Which one a field gets is read off its saved element, and the inclusion
indicator is evaluated on whichever points the layer it weights is on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from dolfinx import fem

from .layers import COMPLEMENT, INDICATOR, MASK, OCCUPY

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .layers import Layer

#: What a value array is attached to, in the vocabulary pyvista uses for it.
POINT = "point"
CELL = "cell"


class Frame:
    """One saved run at one frame, with the sampling it implies cached.

    A figure asks for the same few fields several times over -- for the
    composite, a detail row, a colourbar range -- and each ask is an
    interpolation or a projection, so each is done once per frame.
    """

    def __init__(self, run: Run, frame: int, *, with_ddm_floor: bool = False) -> None:
        if frame < 0 or frame >= run.n_frames:
            raise ValueError(
                f"frame {frame} is outside the {run.n_frames} frame(s) saved in "
                f"{run.path}"
            )
        self.run = run
        self.frame = frame
        #: Weight by the indicator the solve carried, floor included, rather
        #: than by the indicator of the domain. Off by default: the floor is a
        #: regularization keeping a row solvable where there is no material, not
        #: part of the picture. On, it shows what the solve did inside the
        #: inclusions.
        self.with_ddm_floor = with_ddm_floor
        self._values: dict[tuple[str, str | None], tuple[np.ndarray, str]] = {}
        self._chi: dict[str, np.ndarray] = {}

    @property
    def time(self) -> float:
        return float(self.run.times[self.frame])

    def values(
        self, field: str, association: str | None = None
    ) -> tuple[np.ndarray, str]:
        """This frame of ``field``, and whether it is per point or per cell.

        ``association`` asks for one of the two. By default it is whichever the
        saved element makes natural, and cells for :data:`~.layers.INDICATOR`,
        which is evaluated rather than read.
        """

        key = (field, association)
        if key not in self._values:
            if field == INDICATOR:
                association = association or CELL
                values = self.chi(association)
            else:
                saved = self.run.field(field)
                if association is None:
                    association = CELL if saved.is_cellwise else POINT
                space = (
                    self.run.cell_space(saved.shape)
                    if association == CELL
                    else self.run.vertex_space(saved.shape)
                )
                function = saved.sample(self.frame, space)
                values = np.asarray(function.x.array, dtype=float)
                components = int(function.function_space.dofmap.index_map_bs)
                if components > 1:
                    # The magnitude: unlike any one component, it does not
                    # depend on the order the components are stored in.
                    values = np.linalg.norm(values.reshape(-1, components), axis=1)
            # Filed under the association it resolved to as well, so that a
            # caller naming that association shares the sample.
            self._values[key] = self._values[(field, association)] = (
                values,
                association,
            )
        return self._values[key]

    def common(self, layers: tuple[Layer, ...]) -> str:
        """Where a set of layers can be added together: cells, if any lives there.

        Compositing is arithmetic between arrays on the same points, and it is
        the continuous member of a mixed set that moves: bringing a cell-constant
        phase to the vertices would smooth the steps the picture is of, while
        bringing a smooth field to the cells only averages it.
        """

        if any(self.values(layer.field)[1] == CELL for layer in layers):
            return CELL
        return POINT

    def layer(
        self, layer: Layer, association: str | None = None
    ) -> tuple[np.ndarray, str]:
        """This frame of a layer, weighted by the inclusions as it asks to be."""

        values, association = self.values(layer.field, association)
        return weighted(values, layer, lambda: self.chi(association)), association

    def chi(self, association: str) -> np.ndarray:
        """The inclusion indicator on ``association``'s points.

        Evaluated from the spec rather than read, so weighting does not depend
        on the run having saved it. One everywhere for a :class:`Run` opened
        without its parameters, so that its phases draw unweighted rather than
        not at all.
        """

        if association not in self._chi:
            if association == CELL:
                space = self.run.cell_space()
            else:
                space = self.run.vertex_space()
            solver = getattr(self.run.parameters, "solver", None)
            geometry = getattr(solver, "diffuse_domain", None)
            if geometry is None:
                # An array rather than the scalar 1, since callers index it per
                # point.
                chi = np.ones_like(fem.Function(space).x.array, dtype=float)
            elif self.with_ddm_floor:
                chi = geometry.chi_eps_on(space)
            else:
                chi = geometry.bulk_indicator_on(space)
            self._chi[association] = chi
        return self._chi[association]


def weighted(
    values: np.ndarray, layer: Layer, chi: Callable[[], np.ndarray]
) -> np.ndarray:
    """One layer's values against the inclusions, the way the layer asks.

    ``chi`` is called rather than passed, because evaluating the indicator is an
    interpolation and only two of the four modes use it.
    """

    if layer.chi == COMPLEMENT:
        return 1.0 - values
    if layer.chi == OCCUPY:
        return values * chi()
    if layer.chi == MASK:
        # Held at the material's minimum inside the inclusions, which takes the
        # values there out of the colour range without a hole in the picture.
        outside = np.broadcast_to(chi(), values.shape) > 0.5
        if not outside.any():
            return values
        return np.where(outside, values, float(values[outside].min()))
    return values


class Domain:
    """The geometry alone, answering as a :class:`Frame` does, with no run behind it.

    For a picture of where the inclusions landed and how far the smearing
    reaches before anything is solved; :mod:`.render` cannot tell it from a
    frame. Only the indicator has a value: every other field is something a
    solve produces.

    Drawn on P1 by default, because the indicator is smooth and cell averages
    would draw its diffuse profile as a staircase. ``CELL`` shows the profile as
    the cells resolve it, which is the question
    ``min_elements_across_interface_width`` is about.
    """

    def __init__(
        self, parameters: PhaseFieldSystemParameters, *, association: str = POINT
    ) -> None:
        self.parameters = parameters
        self.association = association
        self.mesh = parameters.solver.dolfinx_mesh

    def values(
        self, field: str, association: str | None = None
    ) -> tuple[np.ndarray, str]:
        if field != INDICATOR:
            raise ValueError(
                f"a figure of the domain has only its indicator to draw; "
                f"{field!r} is a saved field, which needs a run"
            )
        association = association or self.association
        return self.chi(association), association

    def layer(
        self, layer: Layer, association: str | None = None
    ) -> tuple[np.ndarray, str]:
        values, association = self.values(layer.field, association)
        return weighted(values, layer, lambda: values), association

    def common(self, layers: tuple[Layer, ...]) -> str:
        return self.association

    def chi(self, association: str) -> np.ndarray:
        """The indicator of the simulated domain, without the solve's floor."""

        element = (
            ("Discontinuous Lagrange", 0) if association == CELL else ("Lagrange", 1)
        )
        return self.parameters.solver.diffuse_domain.bulk_indicator_on(
            fem.functionspace(self.mesh, element)
        )
