"""A mixture's saved frames, rendered into one publication figure.

:func:`plot` is what
:meth:`PhaseFieldSystem.plot <ovpsolver.phase_field_system.PhaseFieldSystem.plot>`
calls, and like ``run`` it takes a spec and works out the rest: which layers to
draw comes from the spec's parameters object, and which fields exist from the
saved run's metadata.

The layout is read off the arguments:

*one spec, several frames*
    a time series: a column per moment, titled with its time unless told
    otherwise.
*several specs, one frame*
    a comparison: a column per run, untitled unless told otherwise, since what
    distinguishes them is not the time.

The columns are checked for sharing a mesh and a set of phases before anything
is drawn, since a figure whose columns differ in either means nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ...fem.reduce import owned_cells
from ...fem.save import Run
from ..entry import EntryPoint
from ..run import read
from . import figure as figures
from . import render
from .layers import COMPLEMENT, INDICATOR, PLAIN, Layer
from .style import BESIDE_THE_RUN, draw_without_a_window, resolved
from .values import Frame

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

    from ..system import PhaseFieldSystem


def plot(
    system_class: type[PhaseFieldSystem],
    spec: str | Path | Sequence[str | Path],
    *,
    frames: Sequence[int],
    save: str | Path | None = None,
    show: bool = False,
    split_fields: bool = False,
    override_colors: bool = False,
    group_polymerizing: bool = False,
    diffuse_domain: bool = True,
    with_ddm_floor: bool = False,
    joint_colorbars: bool = False,
    aux_fields: Sequence[str] = (),
    aux_cmap: str | None = None,
    titles: Sequence[str] | None = None,
    split_titles: Sequence[str] | None = None,
    transpose: bool = False,
    n_rows: int = 1,
    **style_overrides: Any,
) -> plt.Figure:
    """Render ``spec``'s saved output and return the figure.

    The look is the mixture's
    :data:`plot_params <ovpsolver.phase_field_system.visualize.style.PLOT_PARAMS>`,
    whose keys ``style_overrides`` overrides.
    """

    style = resolved(system_class, style_overrides)
    if isinstance(spec, (str, Path)):
        specs: tuple[Path, ...] = (Path(spec),)
    else:
        specs = tuple(Path(one) for one in spec)
    frames = tuple(int(frame) for frame in frames)
    if not frames:
        raise ValueError("at least one frame must be asked for")
    if any(frame < 0 for frame in frames):
        raise ValueError("frame indices are non-negative")
    if aux_cmap is not None and not (aux_fields or override_colors):
        raise ValueError("a colormap for the auxiliary rows needs auxiliary rows")
    if override_colors and not split_fields:
        raise ValueError(
            "--override-colors recolours the per-phase rows, which only --split-fields "
            "draws; the composite keeps its hues either way"
        )

    by_run = len(specs) > 1
    if by_run and len(frames) > 1:
        raise ValueError(
            "a figure is either one run over several frames or several runs at "
            "one frame; this asks for both"
        )
    pairs = (
        tuple((path, frames[0]) for path in specs)
        if by_run
        else tuple((specs[0], frame) for frame in frames)
    )
    sources = tuple(
        Source(system_class, path, frame, with_ddm_floor=with_ddm_floor)
        for path, frame in pairs
    )
    _check_comparable(sources)

    summary = _summary_layers(
        sources[0],
        group_polymerizing=group_polymerizing,
        diffuse_domain=diffuse_domain,
    )
    split = summary if split_fields else ()
    if override_colors:
        # Drawn by value, as a diagnostic is, rather than by opacity in the
        # phase's hue: read against the page, opacity cannot tell a volume
        # fraction of a tenth from a fifth, and loses the structure inside a
        # layer. The composite keeps its hues, being several phases in one.
        split = tuple(
            replace(layer, color=None, colormap=aux_cmap) if layer.is_phase else layer
            for layer in split
        )
    details = split + _aux_layers(sources[0], tuple(aux_fields), aux_cmap)
    if split_fields and diffuse_domain:
        # The inclusions are context in a composite, and a row of their own
        # would say nothing: chi does not change over a run.
        details = tuple(layer for layer in details if layer.chi != COMPLEMENT)
    if not diffuse_domain:
        summary, details = _unweighted(summary), _unweighted(details)

    mesh = sources[0].run.mesh
    view = render.camera(mesh, style)
    rows, ranges = _render(
        sources, summary, details, grids=render.grids(mesh), view=view, style=style
    )

    if by_run:
        times: tuple[str | None, ...] = (None,) * len(sources)
    else:
        times = tuple(f"t = {source.frame.time:.3g}" for source in sources)
    figure = figures.compose(
        rows,
        titles=_titles(titles, times, "title"),
        summary_layers=summary,
        detail_layers=details,
        detail_titles=_titles(
            split_titles, tuple(layer.name for layer in details), "row title"
        ),
        detail_ranges=ranges,
        joint_colorbars=joint_colorbars,
        transpose=transpose,
        n_rows=n_rows,
        aspect=view["aspect"],
        style=style,
    )

    if show:
        plt.show()
    elif save is not None:
        beside = str(save) == BESIDE_THE_RUN
        if beside and by_run:
            raise ValueError(
                "a bare --save has nowhere to put a figure of several runs; name a path"
            )
        figures.save(
            figure,
            sources[0].run.path / "figure.pdf" if beside else Path(save),
            dpi=style["dpi"],
        )
    return figure


## What a column is


class Source:
    """One spec at one frame: its parameters, the run it saved, and the frame."""

    def __init__(
        self,
        system_class: type[PhaseFieldSystem],
        spec: Path,
        frame: int,
        *,
        with_ddm_floor: bool = False,
    ) -> None:
        self.spec = spec
        self.parameters = read(system_class.Parameters, spec)
        self.run = Run.of(self.parameters)
        self.frame = Frame(self.run, frame, with_ddm_floor=with_ddm_floor)


def _check_comparable(sources: tuple[Source, ...]) -> None:
    """Refuse to put runs side by side that are not the same picture twice."""

    reference = sources[0]
    cells = owned_cells(reference.run.mesh)
    for source in sources[1:]:
        if owned_cells(source.run.mesh) != cells:
            raise ValueError(f"{source.spec} is not on the same mesh as {reference.spec}")
        if (
            source.parameters.phase_field_names
            != reference.parameters.phase_field_names
        ):
            raise ValueError(f"{source.spec} does not have the same phases")


## Layers


def _summary_layers(
    source: Source, *, group_polymerizing: bool, diffuse_domain: bool
) -> tuple[Layer, ...]:
    layers = list(
        source.parameters.phase_layers(group_polymerizing=group_polymerizing)
    )
    inclusion = source.parameters.inclusion_layer()
    if diffuse_domain and inclusion is not None:
        # First, so it is under the phases rather than over them.
        layers.insert(0, inclusion)
    if not layers:
        raise ValueError(f"{source.spec} describes no phases to draw")
    for layer in layers:
        _require(source, layer.field)
    return tuple(layers)


def _aux_layers(
    source: Source, names: tuple[str, ...], colormap: str | None
) -> tuple[Layer, ...]:
    """A detail row per named saved field, coloured by value.

    How a field relates to the inclusions is a question about the quantity, so
    its owning phase is asked; a field no phase owns is drawn as saved.
    """

    layers = []
    for name in names:
        _require(source, name)
        owner, _, attribute = name.partition(".")
        chi = next(
            (
                phase.plot_chi_mode(attribute)
                for phase in source.parameters.phase_field_parameters
                if phase.name == owner
            ),
            PLAIN,
        )
        layers.append(Layer(name=name, field=name, colormap=colormap, chi=chi))
    return tuple(layers)


def _unweighted(layers: tuple[Layer, ...]) -> tuple[Layer, ...]:
    """Every layer as its field was saved, and none that draws the inclusions.

    What leaving the diffuse domain out means past dropping the inclusions: a
    volume fraction is not multiplied by ``chi`` and a ratio not masked, so the
    mixture's extension into the inclusions is what shows there. A field saved
    already weighted is drawn as saved.
    """

    return tuple(
        replace(layer, chi=PLAIN) for layer in layers if layer.chi != COMPLEMENT
    )


def _require(source: Source, field: str) -> None:
    # The geometry is computed from the spec rather than read, so it is there
    # whether or not the run saved any of it.
    if field == INDICATOR:
        return
    if not source.run.has(field):
        raise ValueError(
            f"{source.spec} needs {field!r} plotted but its run did not save it; "
            f"it has {', '.join(source.run.names)}"
        )


## Rendering


def _render(
    sources: tuple[Source, ...],
    summary: tuple[Layer, ...],
    details: tuple[Layer, ...],
    *,
    grids: dict[str, Any],
    view: dict[str, Any],
    style: dict[str, Any],
) -> tuple[list[list[np.ndarray]], tuple[tuple[tuple[float, float], ...], ...]]:
    """The composite row and a row per detail layer, with each detail panel's range."""

    rows = []
    row = []
    for source in sources:
        rgba, association = render.composite(source.frame, summary, style)
        row.append(
            render.panel(
                grids[association],
                rgba,
                association=association,
                view=view,
                style=style,
            )
        )
    rows.append(row)

    ranges: list[tuple[tuple[float, float], ...]] = []
    for layer in details:
        row, spans = [], []
        for source in sources:
            values, association = source.frame.layer(layer)
            span = (float(np.min(values)), float(np.max(values)))
            if layer.is_phase:
                rgba = render.phase_rgba(values, layer.color, span)
            else:
                rgba = render.scalar_rgba(
                    values, span, colormap=layer.colormap, style=style
                )
            row.append(
                render.panel(
                    grids[association],
                    rgba,
                    association=association,
                    view=view,
                    style=style,
                )
            )
            spans.append(span)
        rows.append(row)
        ranges.append(tuple(spans))
    return rows, tuple(ranges)


def _titles(
    given: Sequence[str] | None, default: tuple[str | None, ...], what: str
) -> tuple[str | None, ...]:
    """``given`` if there is one, checked to name as many as ``default`` does."""

    if given is None:
        return default
    given = tuple(given)
    if len(given) != len(default):
        raise ValueError(f"expected {len(default)} {what}(s), got {len(given)}")
    return given


## The command line


class VisualizePhases(EntryPoint):
    """Render a mixture's saved phase fields as one publication figure.

    Give one spec and several frames for a time series, or several specs and one
    frame to compare runs. Which layers are drawn comes from the mixture's own
    parameters, and how they are drawn from its ``plot_params``; what is on the
    command line is only what changes between one figure and the next.
    """

    name = "visualize.phases"
    summary = "render saved phase fields as a composite figure"

    #: The figure is on disk by the time this returns, and the exit hooks
    #: dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "items",
            nargs="+",
            metavar="SPEC ... FRAME ...",
            help="the specs to read, then the frames to draw",
        )
        parser.add_argument(
            "--save",
            nargs="?",
            const=BESIDE_THE_RUN,
            help=(
                "where to write the figure. Bare, it writes figure.pdf into the "
                "run's own output folder. Absent, the figure opens in a window "
                "instead."
            ),
        )
        parser.add_argument(
            "--split-fields",
            "--split",
            dest="split_fields",
            action="store_true",
            help="give every phase a row of its own under the composite",
        )
        parser.add_argument(
            "--override-colors",
            "--override-colours",
            dest="override_colors",
            action="store_true",
            help=(
                "draw those per-phase rows by value, as the diagnostic rows are, "
                "instead of by opacity in the phase's own hue"
            ),
        )
        parser.add_argument(
            "--group-polymerizing",
            "--group-poly",
            dest="group_polymerizing",
            action="store_true",
            help="draw a crosslinking phase as one layer rather than as sol and gel",
        )
        parser.add_argument(
            "--no-diffuse-domain",
            dest="diffuse_domain",
            action="store_false",
            help=(
                "leave the fixed inclusions out, and draw every field unweighted by "
                "them: phases not multiplied by chi, ratios not masked inside the "
                "inclusions. A field saved already weighted is drawn as saved."
            ),
        )
        parser.add_argument(
            "--with-ddm-floor",
            action="store_true",
            help=(
                "weight fields by the indicator the solve carried, including "
                "indicator_floor, instead of by the indicator of the domain. Shows "
                "what the scheme did inside the inclusions, which is regularization "
                "rather than physics; off by default, when phases vanish there"
            ),
        )
        parser.add_argument(
            "--show-joint-colorbars",
            dest="joint_colorbars",
            action="store_true",
            help="add the right-hand column of zero-to-one phase colourbars",
        )
        parser.add_argument(
            "--aux-fields",
            nargs="+",
            default=(),
            dest="aux_fields",
            metavar="FIELD",
            help="saved fields to add as diagnostic rows, by their saved name",
        )
        parser.add_argument(
            "--aux-cmap",
            dest="aux_cmap",
            metavar="NAME",
            help="a colorcet colormap for the diagnostic rows, e.g. isolum",
        )
        parser.add_argument(
            "--panel-pixels",
            type=int,
            dest="panel_pixels",
            help="the height of each rendered panel, in pixels",
        )
        parser.add_argument(
            "--font-scaling",
            type=float,
            dest="font_scaling",
            metavar="SCALE",
            help="multiply every font size on the figure by this, to read type "
            "that is too small or too large for the size it is shown at",
        )
        parser.add_argument("--dpi", type=float, help="resolution of the saved figure")
        parser.add_argument(
            "--titles",
            nargs="+",
            help="a title per column, replacing the times a time series would use",
        )
        parser.add_argument(
            "--split-titles",
            nargs="+",
            dest="split_titles",
            help="a title per detail row, replacing the layer names",
        )

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments["spec"], arguments["frames"] = split_items(arguments.pop("items"))
        # Derived rather than asked for: a figure nobody named a path for is one
        # to look at, so a flag could only contradict the path.
        arguments["show"] = arguments["save"] is None
        # Left off the command line means the mixture's own plot_params value,
        # which is not the same as being passed None.
        for key in ("panel_pixels", "dpi", "font_scaling"):
            if arguments.get(key) is None:
                arguments.pop(key, None)
        return arguments

    def __call__(
        self,
        system_class: type[PhaseFieldSystem],
        *,
        spec: tuple[Path, ...],
        **options: Any,
    ) -> plt.Figure:
        if not options.get("show"):
            draw_without_a_window()
        return system_class.plot(spec, **options)


def split_items(items: list[str]) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    """Separate ``SPEC ... FRAME ...`` into the two, by what parses as a number.

    Positional rather than two flags, because the common invocation is one spec
    and one frame. Specs come first, and everything from the first number on is
    a frame.
    """

    specs: list[Path] = []
    frames: list[int] = []
    for item in items:
        try:
            frame = int(item)
        except ValueError:
            if frames:
                raise ValueError(
                    f"specs come before frames, but {item!r} follows one"
                ) from None
            specs.append(Path(item))
            continue
        if frame < 0:
            raise ValueError(f"frame indices are non-negative, got {frame}")
        frames.append(frame)

    if not specs:
        raise ValueError("name at least one spec to draw")
    if not frames:
        raise ValueError("name at least one frame to draw")
    return tuple(specs), tuple(frames)
