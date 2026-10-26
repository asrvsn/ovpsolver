"""A sweep of two axes as one figure: a saved field of every run, on the sweep's grid.

    python -m ovpsolver.polymerizing_b visualize.sweep examples/pips_coarse.yaml \\
        --sweep-zip 'phase_fields.sol[0].kappa=[1e-3, 2e-3, 4e-3]' \\
                    'phase_fields.polymerizing[0].kappa=[1e-3, 2e-3, 4e-3]' \\
        --sweep 'phase_fields.polymerizing[0].crosslinking_rate=[1.56, 3.125, 6.25]' \\
        --x 'phase_fields.polymerizing[0].crosslinking_rate' --x-label '$\\gamma$' \\
        --y 'phase_fields.sol[0].kappa' --y-label '$\\kappa$' \\
        --field 'protein.phi_gel=$\\phi_{\\rm gel}$' --save

The spec and the ``--sweep`` and ``--sweep-zip`` options are the ones the sweep
was run with, so the runs are found by the sweep's own naming
(:meth:`~ovpsolver.phase_field_system.sweep.Sweep.directories`) and nothing has
to say where they went. Each run is read from the spec archived in its
directory, which is the document that run was made from, whatever has become
of the original since.

The grid
--------
The sweep has two axes, a ``--sweep-zip`` counting as one. ``--x`` names an
address on one of them, whose values run along the columns, and ``--y`` one on
the other, whose values run up the rows: the figure reads as a plot of the two
does, the first values at the bottom left. Either may be left off, for the first
address of the axis the other did not take; with neither, x is the last axis
given, the one varying fastest through the runs, and y the first.

Each row is labelled with its y value left of its first panel, and each column
with its x value under its bottom panel; every panel is titled with the time of
its frame. ``--x-label`` and ``--y-label`` name the two as matplotlib text, so
``'$\\kappa$'`` labels a row ``$\\kappa = 0.001$``; by default each is its
address's label in the run directories.

A label reaches matplotlib exactly as the shell hands it over, since nothing in
Python unescapes the command line, so there is no ``r''`` to write. The shell's
equivalent is single quotes, inside which it leaves every character alone; in
double quotes it expands ``$t`` as a variable.

The panels
----------
``--field`` is any saved field ``visualize.phases --aux-fields`` takes, drawn as
``visualize.phases --split-fields`` draws its rows: a field that is one of the
mixture's phase layers -- a phase's volume fraction, a polymerizing phase's sol
or gel -- in that layer's colour from the spec, by opacity, and anything else by
value through a colormap. ``--cmap`` draws a phase by value too, through the
colorcet map it names. ``=LABEL`` after the field names it beside each
colourbar, by default the field's saved name.

Each panel has its own colourbar over its own range, as each split row does:
the runs of a sweep can differ by orders of magnitude, and one range would draw
all but the largest flat. ``--range`` gives every panel the same one instead.

Each panel is its run's last saved frame, so the figure is what the sweep ended
at; ``--frame`` picks another. The times are not one time across the figure,
since a run that failed ended early. A run with nothing to draw -- not run yet,
or failed before its first frame -- leaves its cell empty under a title saying
so, rather than refusing the figure: a sweep goes on past a failed run, and its
figure is of what the sweep did.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from ...fem.save import META_FILE, SPEC_FILE, Run
from .. import sweep as sweeps
from ..entry import EntryPoint
from . import figure as figures
from . import render
from .layers import PLAIN
from .phases import Source, aux_layers
from .style import BESIDE_THE_RUN, draw_without_a_window, resolved

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ..system import PhaseFieldSystem


def plot_sweep(
    system_class: type[PhaseFieldSystem],
    spec: str | Path,
    sweep: sweeps.Sweep,
    *,
    field: str,
    field_label: str | None = None,
    x: str | None = None,
    y: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    frame: int = -1,
    cmap: str | None = None,
    value_range: tuple[float, float] | None = None,
    diffuse_domain: bool = True,
    with_ddm_floor: bool = False,
    save: str | Path | None = None,
    show: bool = False,
    **style_overrides: Any,
) -> plt.Figure:
    """Draw ``field`` for every run of ``sweep`` over ``spec``, and return the figure.

    ``x`` and ``y`` are swept addresses as the command line spells them.
    ``field_label`` names the field beside each colourbar, by default its saved
    name. ``frame`` counts from each run's end when negative.
    """

    style = resolved(system_class, style_overrides)
    (across, columns), (up, rows) = grid_axes(sweep, x, y)
    directories = sweep.directories(spec)
    if value_range is not None and not value_range[0] < value_range[1]:
        raise ValueError(f"a range is VMIN VMAX, the first below the second; got {value_range}")

    # The runs are every combination of the axes, the last varying fastest, so
    # a run's place on the grid is its index along each. Each panel is drawn as
    # its run is read, and only the image kept: the run's mesh and spaces, held
    # for every run at once, are the most memory a sweep figure could take.
    n_columns, n_rows = len(columns.values), len(rows.values)
    images: list[list[np.ndarray | None]] = [[None] * n_columns for _ in range(n_rows)]
    titles: list[list[str | None]] = [[None] * n_columns for _ in range(n_rows)]
    ranges: list[list[tuple[float, float] | None]] = [[None] * n_columns for _ in range(n_rows)]
    layer, aspect, sizes, missing = None, None, set(), []
    places = itertools.product(*(range(len(axis[0].values)) for axis in sweep.axes))
    for indices, directory in zip(places, directories, strict=True):
        # Rows are laid out from the top, and y runs up.
        row, column = n_rows - 1 - indices[up], indices[across]
        source = open_run(system_class, directory, frame, with_ddm_floor=with_ddm_floor)
        if isinstance(source, str):
            titles[row][column] = source
            missing.append(f"  {directory}: {source}")
            continue
        # The --aux-fields layer, which checks the run saved the field, unless
        # the field is a phase layer and so has a colour of its own.
        (layer,) = aux_layers(source, (field,), cmap)
        if cmap is None:
            layer = next(
                (one for one in source.parameters.phase_layers() if one.field == field),
                layer,
            )
        if not diffuse_domain:
            layer = replace(layer, chi=PLAIN)
        values, association = source.frame.layer(layer)
        span = value_range or (float(np.min(values)), float(np.max(values)))
        mesh = source.run.mesh
        view = render.camera(mesh, style)
        images[row][column] = render.panel(
            render.grids(mesh)[association],
            render.layer_rgba(values, layer, span, style),
            association=association,
            view=view,
            style=style,
        )
        titles[row][column] = f"$t = {source.frame.time:.3g}$"
        ranges[row][column] = span
        aspect = view["aspect"]
        sizes.add(view["window_size"])

    if layer is None:
        raise ValueError(
            f"none of the {len(directories)} runs of this sweep has a frame to draw:\n"
            + "\n".join(missing)
        )
    if missing:
        print(
            f"{len(missing)} of {len(directories)} runs have nothing to draw, so their "
            f"cells are left empty:",
            *missing,
            sep="\n",
            file=sys.stderr,
        )
    if len(sizes) > 1:
        raise ValueError(
            "the runs' meshes are different shapes, so their panels do not tile one grid"
        )

    labels = sweep.labels()
    x_label = labels[columns.address] if x_label is None else x_label
    y_label = labels[rows.address] if y_label is None else y_label
    figure = figures.grid(
        images,
        titles=titles,
        row_labels=tuple(equation(y_label, value) for value in reversed(rows.values)),
        column_labels=tuple(equation(x_label, value) for value in columns.values),
        ranges=ranges,
        layer=layer,
        colorbar_label=field if field_label is None else field_label,
        title=title,
        aspect=aspect,
        style=style,
    )

    if show:
        plt.show()
    elif save is not None:
        if str(save) == BESIDE_THE_RUN:
            base = sweeps.base_directory(spec)
            path = base.with_name(f"{base.name}-sweep-{field}.pdf")
        else:
            path = Path(save)
        figures.save(figure, path, dpi=style["dpi"])
        print(f"wrote {path}")
    return figure


def equation(label: str, value: Any) -> str:
    """``label = value``, set inside the label's mathtext when it is all mathtext.

    So ``'$\\kappa$'`` and 0.001 make ``'$\\kappa = 0.001$'``, set as a panel's
    ``'$t = ...$'`` title is, rather than a mathtext kappa beside an equals sign
    and a number in the text face.
    """

    text = sweeps.flow(value)
    if len(label) > 1 and label[0] == label[-1] == "$" and label.count("$") == 2:
        return f"{label[:-1]} = {text}$"
    return f"{label} = {text}"


def grid_axes(
    sweep: sweeps.Sweep, x: str | None, y: str | None
) -> tuple[tuple[int, sweeps.Assignment], tuple[int, sweeps.Assignment]]:
    """Which axis of ``sweep`` runs across the figure and which up it.

    Each as its index among the sweep's axes and the assignment whose values
    label it, which on a ``--sweep-zip`` is the one ``x`` or ``y`` named.
    """

    if len(sweep.axes) != 2:
        raise sweeps.SweepError(
            f"a sweep figure is the grid of two axes, and this sweep has "
            f"{len(sweep.axes)}: each --sweep assignment is an axis of its own, and "
            f"each --sweep-zip one axis between its assignments"
        )

    def find(text: str, flag: str) -> tuple[int, sweeps.Assignment]:
        address = sweeps.parse_address(text)
        for index, axis in enumerate(sweep.axes):
            for assignment in axis:
                if assignment.address == address:
                    return index, assignment
        swept = ", ".join(sweeps.render(one) for one in sweep.addresses())
        raise sweeps.SweepError(f"{flag} {text} is not swept; the swept addresses are {swept}")

    across = None if x is None else find(x, "--x")
    up = None if y is None else find(y, "--y")
    if across is not None and up is not None and across[0] == up[0]:
        raise sweeps.SweepError(
            f"--x {x} and --y {y} step together, on one axis, so they cannot be the "
            f"two sides of a grid"
        )
    if across is None:
        index = 1 if up is None else 1 - up[0]
        across = (index, sweep.axes[index][0])
    if up is None:
        index = 1 - across[0]
        up = (index, sweep.axes[index][0])
    return across, up


def open_run(
    system_class: type[PhaseFieldSystem],
    directory: Path,
    frame: int,
    *,
    with_ddm_floor: bool = False,
) -> Source | str:
    """The run in ``directory`` at ``frame``, or why there is nothing there to draw.

    Read from the spec the run archived, which is the document it was made from.
    ``frame`` counts from the end when negative, as an index does: the runs of a
    sweep end at different frames when one stops early, and a figure of the
    sweep wants each one's last.
    """

    if not directory.is_dir():
        return "not run"
    if not ((directory / META_FILE).is_file() and (directory / SPEC_FILE).is_file()):
        return "nothing saved"
    count = Run(directory).n_frames
    index = count + frame if frame < 0 else frame
    if not 0 <= index < count:
        return "nothing saved" if count == 0 else f"no frame {frame}"
    return Source(system_class, directory / SPEC_FILE, index, with_ddm_floor=with_ddm_floor)


class VisualizeSweep(EntryPoint):
    """Render one saved field of every run of a two-axis sweep, as a grid of panels.

    Give the spec and the --sweep and --sweep-zip options exactly as the sweep
    was run, then which swept address runs across the figure and which up it,
    and the field to draw:

        visualize.sweep pips_coarse.yaml \\
            --sweep-zip 'phase_fields.sol[0].kappa=[1e-3, 2e-3, 4e-3]' \\
                        'phase_fields.polymerizing[0].kappa=[1e-3, 2e-3, 4e-3]' \\
            --sweep 'phase_fields.polymerizing[0].crosslinking_rate=[1.56, 3.125, 6.25]' \\
            --x 'phase_fields.polymerizing[0].crosslinking_rate' --x-label '$\\gamma$' \\
            --y 'phase_fields.sol[0].kappa' --y-label '$\\kappa$' \\
            --field 'protein.phi_gel=$\\phi_{\\rm gel}$' --save

    Each panel is its run's last saved frame, drawn as --split-fields draws the
    field, with its time above it and its own colourbar to its right; each row
    is labelled with its y value on the left, and each column with its x value
    along the bottom. A run with nothing saved leaves its cell empty, titled
    with why. Put labels in single quotes, which the shell passes
    on untouched, as Python's r'' does; in double quotes it would expand $t as a
    variable.
    """

    name = "visualize.sweep"
    summary = "render one saved field of every run of a sweep, on the sweep's two axes"

    #: The figure is on disk by the time this returns, and the exit hooks
    #: dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: ArgumentParser) -> None:
        parser.add_argument("spec", type=Path, help="the spec the sweep was run from")
        sweeps.add_options(parser)
        parser.add_argument(
            "--x",
            metavar="ADDRESS",
            help="the swept address whose values run along the columns; by default "
            "the first of the last axis given",
        )
        parser.add_argument(
            "--x-label",
            dest="x_label",
            metavar="TEXT",
            help="the x value's name under each column, as matplotlib text, so "
            "'$\\gamma$' is a gamma; by default the address's label in the run "
            "directories",
        )
        parser.add_argument(
            "--y",
            metavar="ADDRESS",
            help="the swept address whose values run up the rows, on the other axis; "
            "by default the first of the first axis given",
        )
        parser.add_argument(
            "--y-label",
            dest="y_label",
            metavar="TEXT",
            help="the y value's name beside each row, as --x-label",
        )
        parser.add_argument(
            "--field",
            required=True,
            metavar="FIELD[=LABEL]",
            help="the saved field to draw, by its saved name as --aux-fields takes "
            "it, and optionally the label to name it by beside each colourbar, as "
            "'protein.phi_gel=$\\phi_{\\rm gel}$'; by default the saved name. A "
            "phase layer is drawn in its colour from the spec, as --split-fields "
            "draws it, and anything else by value",
        )
        parser.add_argument("--title", metavar="TEXT", help="a title above the grid; none by default")
        parser.add_argument(
            "--frame",
            type=int,
            default=-1,
            metavar="N",
            help="the saved frame each panel draws, negative counting from each "
            "run's end (default: -1, its last)",
        )
        parser.add_argument(
            "--cmap",
            metavar="NAME",
            help="draw the field by value through this colorcet colormap, e.g. "
            "isolum, even a phase, which is otherwise drawn in its own colour",
        )
        parser.add_argument(
            "--range",
            type=float,
            nargs=2,
            dest="value_range",
            metavar=("VMIN", "VMAX"),
            help="one colour range for every panel, instead of each panel's own extent",
        )
        parser.add_argument(
            "--no-diffuse-domain",
            dest="diffuse_domain",
            action="store_false",
            help="draw the field unweighted by the inclusions, as visualize.phases "
            "--no-diffuse-domain does",
        )
        parser.add_argument(
            "--with-ddm-floor",
            action="store_true",
            help="weight the field by the indicator the solve carried, floor included, "
            "as visualize.phases --with-ddm-floor does",
        )
        parser.add_argument(
            "--save",
            nargs="?",
            const=BESIDE_THE_RUN,
            help="where to write the figure. Bare, it writes BASE-sweep-FIELD.pdf "
            "beside the runs, BASE being the output directory the spec names. "
            "Absent, the figure opens in a window instead.",
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
            help="multiply every font size on the figure by this",
        )
        parser.add_argument("--dpi", type=float, help="resolution of the saved figure")

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        groups = arguments.pop("sweep")
        if not groups:
            raise sweeps.SweepError(
                "visualize.sweep finds the runs by the sweep that made them; give it "
                "the --sweep and --sweep-zip options the sweep was run with"
            )
        arguments["sweep"] = sweeps.Sweep.from_groups(groups)
        # A saved name has no '=' in it, so the first one starts the label, which
        # may hold more: '$t=0$'.
        field, _, label = arguments["field"].partition("=")
        arguments["field"], arguments["field_label"] = field, label or None
        arguments["show"] = arguments["save"] is None
        # Left off the command line means the mixture's own plot_params value.
        for key in ("panel_pixels", "dpi", "font_scaling"):
            if arguments.get(key) is None:
                arguments.pop(key, None)
        return arguments

    def __call__(
        self,
        system_class: type[PhaseFieldSystem],
        *,
        spec: Path,
        sweep: sweeps.Sweep,
        **options: Any,
    ) -> plt.Figure:
        if not options.get("show"):
            draw_without_a_window()
        return plot_sweep(system_class, spec, sweep, **options)
