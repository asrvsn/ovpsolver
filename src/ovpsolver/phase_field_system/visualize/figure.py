"""Arranging rendered panels into a figure.

Pure matplotlib and pure layout: everything here takes images already drawn and
decides where on the page they go. The composite row has a column per time or
run, each panel every phase at once, with colourbars down the right saying which
hue was which. Split fields add a row per layer under it, each with a colourbar
of its own, because each layer has its own range. A sweep is laid out as a grid
instead, a panel per run on the sweep's two axes, over one colourbar.

Every size is read from the ``style`` a mixture carries; see :mod:`.style`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .render import alpha_ramp, scalar_rgba
from .style import configure_tex_like_fonts

if TYPE_CHECKING:
    from .layers import Layer


def compose(
    rows: list[list[np.ndarray]],
    *,
    titles: tuple[str | None, ...],
    summary_layers: tuple[Layer, ...],
    detail_layers: tuple[Layer, ...],
    detail_titles: tuple[str, ...],
    detail_ranges: tuple[tuple[tuple[float, float], ...], ...],
    joint_colorbars: bool,
    transpose: bool = False,
    n_rows: int = 1,
    aspect: float,
    style: dict[str, Any],
) -> plt.Figure:
    """The composite row of ``rows``, then its detail rows, one per detail layer.

    ``transpose`` lays a single frame's panels out as one strip instead, and
    ``n_rows`` folds that strip into several, for a frame naming enough fields
    to be unreadably wide as one.

    ``aspect`` is one panel's width over its height, as
    :func:`~ovpsolver.phase_field_system.visualize.render.camera` fitted it.
    Every width below is a multiple of it and every height a multiple of
    ``panel_pixels``, so each grid cell has the shape of its image: an axis
    shaped otherwise pads its image with whitespace.
    """

    configure_tex_like_fonts()
    summary, details = rows[0], rows[1:]
    n_details, columns = len(details), len(summary)
    if len(titles) != columns:
        raise ValueError("one panel title per rendered column")
    if not len(detail_layers) == len(detail_titles) == len(detail_ranges) == n_details:
        raise ValueError("one layer, title and range row per rendered detail row")
    if n_rows < 1:
        raise ValueError("a figure has at least one row of panels")
    if n_rows > 1 and not (transpose and detail_layers):
        raise ValueError(
            "n_rows folds the strip of panels a transposed split layout draws; "
            "this figure has no such strip, so there is nothing to fold"
        )
    # Without detail rows the strip is the composite alone, laid out as one.
    transpose = transpose and n_details > 0
    if transpose and columns != 1:
        raise ValueError("a transposed split layout needs exactly one frame")

    pixels, dpi = style["panel_pixels"], style["dpi"]
    panel_width = pixels * aspect
    # Furniture holds type, so it is sized in points. The colourbar column holds
    # the rotated layer names, and sized against the panel instead it would
    # widen for a wide mesh and crush for a tall one.
    bar_width = (
        pixels
        * _in_panels(
            style["colorbar_label_font_size"]
            * max(
                style["colorbar_column_min_width"],
                style["colorbar_column_width"] * len(summary_layers),
            ),
            style,
        )
        if joint_colorbars
        else 0.0
    )
    # A title's row: one line of title type plus matplotlib's gap between a
    # title and its axes. Reserved in the grid rather than left to
    # bbox_inches="tight", because a movie's frames have to come out the same
    # size whatever their titles say.
    band = _in_panels(mpl.rcParams["axes.titlepad"] + style["title_font_size"], style)
    # The depth of the colourbar under a detail panel.
    ratio = _in_panels(
        style["split_colorbar_height"] * style["colorbar_tick_font_size"], style
    )
    panels = 1 + n_details
    panel_columns = -(-panels // n_rows) if transpose else columns

    # Rows are laid out first and filled second, because each title needs a row
    # of its own above the panels it names.
    heights: list[float] = []
    detail_rows: list[tuple[int, int]] = []
    if transpose:
        # Strips of ``panel_columns`` panels, each titled above and ramped below,
        # filled row-major with the composite first; the last is short by
        # whatever does not divide, and its empty cells are left unfilled.
        strips: list[int] = []
        for _ in range(-(-panels // panel_columns)):
            heights.append(band)
            strips.append(len(heights))
            heights.extend((1.0, ratio))
        summary_row = strips[0]
        detail_rows = [
            (strips[place // panel_columns], strips[place // panel_columns] + 1)
            for place in range(1, panels)
        ]
    else:
        if any(title is not None for title in titles):
            heights.append(band)
        summary_row = len(heights)
        heights.append(1.0)
        for _ in range(n_details):
            heights.append(band)
            detail_rows.append((len(heights), len(heights) + 1))
            heights.extend((1.0, ratio))

    figure = plt.figure(
        figsize=(
            (panel_width * panel_columns + bar_width) / dpi,
            pixels * sum(heights) / dpi,
        ),
        dpi=dpi,
    )
    grid = figure.add_gridspec(
        len(heights),
        panel_columns + int(joint_colorbars),
        width_ratios=[panel_width] * panel_columns
        + ([bar_width] if joint_colorbars else []),
        height_ratios=heights,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        wspace=0.0,
        hspace=0.0,
    )

    for column, image in enumerate(summary):
        axis = figure.add_subplot(grid[summary_row, 0 if transpose else column])
        axis.imshow(image)
        _panel_title(axis, titles[column], style)
        axis.set_axis_off()
    if joint_colorbars:
        _joint_colorbars(
            figure.add_subplot(grid[summary_row, panel_columns]),
            summary_layers,
            style,
        )

    for row, (images, layer, title) in enumerate(
        zip(details, detail_layers, detail_titles, strict=True)
    ):
        if len(detail_ranges[row]) != columns:
            raise ValueError("one colourbar range per rendered column")
        panel_row, colorbar_row = detail_rows[row]
        for column, image in enumerate(images):
            panel_column = (1 + row) % panel_columns if transpose else column
            axis = figure.add_subplot(grid[panel_row, panel_column])
            axis.imshow(image)
            if transpose or column == 0:
                _panel_title(axis, title, style)
            axis.set_axis_off()
            _range_colorbar(
                figure.add_subplot(grid[colorbar_row, panel_column]),
                layer,
                detail_ranges[row][column],
                style,
            )

    return figure


def grid(
    images: list[list[np.ndarray | None]],
    *,
    titles: list[list[str | None]],
    column_labels: tuple[str, ...],
    row_labels: tuple[str, ...],
    x_label: str,
    y_label: str,
    title: str | None,
    layer: Layer,
    value_range: tuple[float, float],
    colorbar_label: str,
    aspect: float,
    style: dict[str, Any],
) -> plt.Figure:
    """Panels on the grid of two swept values, over one colourbar naming what they show.

    ``images`` is rows from the top, each a list of columns from the left, and
    ``None`` leaves a cell empty; ``titles`` is shaped alike, and a row of them is
    given room above its panels only if any is set. Each column's label goes
    under the bottom row with ``x_label`` under those, and each row's label to
    the left with ``y_label`` beside those, as a plot's axes are labelled.

    One colourbar for the grid rather than one per panel, as a detail row has:
    every panel is drawn against ``value_range``, so a colour means one value
    anywhere on the figure. ``colorbar_label`` goes under it, where a horizontal
    colourbar's label goes, and ``title``, if any, above the grid. Sized as
    :func:`compose` sizes, in panel heights, so a panel's cell has its image's
    shape.
    """

    configure_tex_like_fonts()
    n_rows, n_columns = len(images), len(column_labels)
    if len(row_labels) != n_rows or len(titles) != n_rows:
        raise ValueError("one row label and one row of titles per row of panels")
    if any(len(row) != n_columns for row in (*images, *titles)):
        raise ValueError("one column label, image and title per column of panels")

    pixels, dpi = style["panel_pixels"], style["dpi"]
    label_size = style["title_font_size"]
    value_size = style["colorbar_label_font_size"]
    pad = mpl.rcParams["axes.titlepad"]
    label_band = _in_panels(pad + label_size, style)
    value_band = _in_panels(pad + value_size, style)
    titled = any(text is not None for row in titles for text in row)

    heights: list[float] = []
    if title is not None:
        heights.append(label_band)
    panel_rows = []
    for _ in range(n_rows):
        if titled:
            heights.append(value_band)
        panel_rows.append(len(heights))
        heights.append(1.0)
    value_row = len(heights)
    heights.extend((value_band, label_band))
    heights.append(
        _in_panels(style["split_colorbar_height"] * style["colorbar_tick_font_size"], style)
    )
    heights.append(label_band)
    widths = [label_band, value_band] + [aspect] * n_columns
    # Panels start after the y label's column and the row labels'.
    first = 2

    figure = plt.figure(
        figsize=(pixels * sum(widths) / dpi, pixels * sum(heights) / dpi), dpi=dpi
    )
    cells = figure.add_gridspec(
        len(heights),
        len(widths),
        width_ratios=widths,
        height_ratios=heights,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        wspace=0.0,
        hspace=0.0,
    )
    if title is not None:
        _centred_text(figure.add_subplot(cells[0, first:]), title, label_size)
    for row, panel_row in enumerate(panel_rows):
        _centred_text(
            figure.add_subplot(cells[panel_row, 1]), row_labels[row], value_size, rotation=90
        )
        for column in range(n_columns):
            axis = figure.add_subplot(cells[panel_row, first + column])
            if images[row][column] is not None:
                axis.imshow(images[row][column])
            if titles[row][column] is not None:
                axis.set_title(titles[row][column], fontsize=value_size, color="black")
            axis.set_axis_off()
    for column, label in enumerate(column_labels):
        _centred_text(figure.add_subplot(cells[value_row, first + column]), label, value_size)
    _centred_text(figure.add_subplot(cells[value_row + 1, first:]), x_label, label_size)
    _centred_text(
        figure.add_subplot(cells[panel_rows[0] : panel_rows[-1] + 1, 0]),
        y_label,
        label_size,
        rotation=90,
    )
    _range_colorbar(
        figure.add_subplot(cells[value_row + 2, first:]), layer, value_range, style
    )
    _centred_text(figure.add_subplot(cells[value_row + 3, first:]), colorbar_label, label_size)
    return figure


def save(figure: plt.Figure, path: Path, *, dpi: float) -> None:
    """Write ``figure`` to ``path``, cropped to what is drawn, and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


## Furniture


def _in_panels(points: float, style: dict[str, Any]) -> float:
    """A length in points, as a multiple of panel height.

    Type is quoted in the first unit and the grid in the second.
    """

    return points / 72.0 * style["dpi"] / style["panel_pixels"]


def _panel_title(axis: plt.Axes, title: str | None, style: dict[str, Any]) -> None:
    """A panel's title, as an axes title so matplotlib centres and pads it.

    It survives ``set_axis_off``, which turns off only the frame and ticks.
    """

    if title is None:
        return
    axis.set_title(title, fontsize=style["title_font_size"], color="black")


def _centred_text(
    axis: plt.Axes, text: str, size: float, *, rotation: float = 0.0
) -> None:
    """``text`` in the middle of a cell of its own, which is all the cell holds."""

    axis.text(
        0.5,
        0.5,
        text,
        color="black",
        fontsize=size,
        ha="center",
        va="center",
        rotation=rotation,
        transform=axis.transAxes,
    )
    axis.set_axis_off()


def _range_colorbar(
    axis: plt.Axes,
    layer: Layer,
    value_range: tuple[float, float],
    style: dict[str, Any],
) -> None:
    """A horizontal bar under one detail panel, labelled with its two ends."""

    vmin, vmax = value_range
    if layer.is_phase:
        gradient = np.swapaxes(alpha_ramp(layer.color), 0, 1)
    else:
        gradient = scalar_rgba(
            np.linspace(vmin, vmax, 256),
            value_range,
            colormap=layer.colormap,
            style=style,
        ).reshape(1, 256, 4)
    axis.imshow(gradient, origin="lower", extent=(0.08, 0.92, 0.45, 0.78), aspect="auto")
    for x, value, align in ((0.08, vmin, "left"), (0.92, vmax, "right")):
        axis.text(
            x,
            0.10,
            f"{value:.{style['colorbar_decimals']}f}",
            color="black",
            fontsize=style["colorbar_tick_font_size"],
            ha=align,
            va="bottom",
            transform=axis.transAxes,
        )
    axis.set_axis_off()


def _joint_colorbars(
    axis: plt.Axes, layers: tuple[Layer, ...], style: dict[str, Any]
) -> None:
    """The right-hand column: one zero-to-one ramp per composited phase."""

    axis.set_xlim(0.0, float(len(layers)))
    axis.set_ylim(0.0, 1.0)
    for index, layer in enumerate(layers):
        left, right = index + 0.16, index + 0.84
        axis.imshow(
            alpha_ramp(layer.color),
            origin="lower",
            extent=(left, right, 0.05, 0.95),
            aspect="auto",
        )
        middle = 0.5 * (left + right)
        axis.text(
            middle,
            0.5,
            layer.name,
            color="white",
            fontsize=style["colorbar_label_font_size"],
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )
        for y, text, va in ((0.08, "0", "bottom"), (0.92, "1", "top")):
            axis.text(
                middle,
                y,
                text,
                color="white",
                fontsize=style["colorbar_tick_font_size"],
                fontweight="bold",
                ha="center",
                va=va,
            )
    axis.set_axis_off()
