"""A picture of the domain, before there is a run to draw on it.

What a spec fixes about the geometry -- where the inclusions are, how far the
smearing reaches, how the mesh grades around them -- is settled before the first
step, and starting a run to see it would build and compile every form. This takes
a mesh, one interpolation of the indicator the spec defines, and the renderer the
other figures use: no system is built and no form compiled.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from ..entry import EntryPoint
from ..run import read
from . import figure as figures
from . import render
from .style import BESIDE_THE_RUN, draw_without_a_window, resolved
from .values import CELL, POINT, Domain

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ..system import PhaseFieldSystem

def domain(
    system_class: type[PhaseFieldSystem],
    spec: str | Path,
    *,
    save: str | Path | None = None,
    show: bool = False,
    cellwise: bool = False,
    outline: bool = True,
    joint_colorbars: bool = True,
    **style_overrides: Any,
) -> plt.Figure:
    """Draw ``spec``'s inclusions and return the figure.

    The mesh comes from the spec as a run's would -- generated if the geometry
    has not been meshed, read back from the archive if it has -- so this is the
    mesh a run would use. The indicator is evaluated by
    :meth:`~ovpsolver.diffuse_domain.DiffuseDomainParameters.bulk_indicator_on`,
    the expression the solver interpolates into its quadrature space, sampled
    where the picture needs it instead.

    ``outline`` draws the mesh boundary, Sigma. On by default here only: this
    figure is asked what the geometry is, and an inclusion against an undrawn
    edge looks like one against nothing.
    """

    style = resolved(system_class, style_overrides)
    parameters = read(system_class.Parameters, Path(spec))
    layer = parameters.inclusion_layer()
    if layer is None:
        raise ValueError(
            f"{spec} declares no inclusions, so its domain is the mesh and "
            f"there is no indicator to draw"
        )

    source = Domain(parameters, association=CELL if cellwise else POINT)
    view = render.camera(source.mesh, style)
    rgba, association = render.composite(source, (layer,), style)
    image = render.panel(
        render.grids(source.mesh)[association],
        rgba,
        association=association,
        view=view,
        style=style,
        outline=outline,
    )

    figure = figures.compose(
        [[image]],
        titles=(None,),
        summary_layers=(layer,),
        detail_layers=(),
        detail_titles=(),
        detail_ranges=(),
        joint_colorbars=joint_colorbars,
        aspect=view["aspect"],
        style=style,
    )

    if show:
        plt.show()
    elif save is not None:
        beside = str(save) == BESIDE_THE_RUN
        figures.save(
            figure,
            parameters.solver.save.directory / "domain.pdf" if beside else Path(save),
            dpi=style["dpi"],
        )
    return figure


class VisualizeDomain(EntryPoint):
    """Draw the inclusions a spec declares, without running it."""

    name = "visualize.domain"
    summary = "draw a spec's diffuse domain, without running it"

    #: The figure is on disk by the time this returns, and the exit hooks
    #: dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: ArgumentParser) -> None:
        parser.add_argument("spec", type=Path, help="path to the experiment YAML")
        parser.add_argument(
            "--save",
            nargs="?",
            const=BESIDE_THE_RUN,
            help=(
                "where to write the figure. Bare, it writes domain.pdf into the "
                "run's own output folder. Absent, the figure opens in a window "
                "instead."
            ),
        )
        parser.add_argument(
            "--cellwise",
            action="store_true",
            help="draw the indicator as the cells average it, showing how much "
            "of the diffuse profile the mesh actually resolves, rather than as "
            "the smooth field it is",
        )
        parser.add_argument(
            "--no-outline",
            dest="outline",
            action="store_false",
            help="leave off the line along the outer boundary Sigma",
        )
        parser.add_argument(
            "--no-colorbar",
            dest="joint_colorbars",
            action="store_false",
            help="leave off the zero-to-one ramp beside the panel",
        )
        parser.add_argument(
            "--panel-pixels",
            type=int,
            dest="panel_pixels",
            help="the height of the rendered panel, in pixels",
        )
        parser.add_argument(
            "--font-scaling",
            type=float,
            dest="font_scaling",
            metavar="SCALE",
            help="multiply every font size on the figure by this",
        )
        parser.add_argument("--dpi", type=float, help="resolution of the figure")

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Left off the command line means the mixture's own plot_params value.
        for key in ("panel_pixels", "dpi", "font_scaling"):
            if arguments.get(key) is None:
                arguments.pop(key, None)
        arguments["show"] = arguments["save"] is None
        return arguments

    def __call__(
        self, system_class: type[PhaseFieldSystem], *, spec: Path, **options: Any
    ) -> plt.Figure:
        if not options.get("show"):
            draw_without_a_window()
        return domain(system_class, spec, **options)
