"""Turning arrays of values into arrays of pixels, in two steps kept apart.

Colour first: a phase layer becomes its own hue at an opacity given by its value,
and several composite into one image the way overlapping stains do; a diagnostic
layer becomes a colormap of its value. Then rasterization: pyvista draws the
coloured mesh off screen and hands back a bitmap, which matplotlib arranges into
a figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx import fem, plot
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb

from .values import CELL, POINT

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh

    from .layers import Layer
    from .values import Domain, Frame

#: The widest panel an off-screen render returns intact: past the largest
#: framebuffer the OpenGL implementation allocates, the screenshot comes back
#: truncated rather than failing.
MAX_PANEL_PIXELS = 16384


## The grid


def grids(mesh: Mesh) -> dict[str, Any]:
    """A pyvista grid per association, both on ``mesh`` so that they overlay.

    Two because they are indexed differently: the cell grid's cells are the mesh
    cells, where nearly every layer lives, and the point grid's points are the
    dofs of a P1 space, where the velocities live. Built from the mesh rather
    than a run, so the domain can be drawn before there is one; the P1 space
    built here numbers its dofs as a run's own does.
    """

    # Inline, here and in panel: the command line imports this module for every
    # command, and pyvista is slow to import.
    import pyvista as pv

    vertices = fem.functionspace(mesh, ("Lagrange", 1))
    return {
        POINT: pv.UnstructuredGrid(*plot.vtk_mesh(vertices)),
        CELL: pv.UnstructuredGrid(*plot.vtk_mesh(mesh, mesh.topology.dim)),
    }


def camera(mesh: Mesh, style: dict[str, Any]) -> dict[str, Any]:
    """An orthographic view fitted to the mesh, and the panel shape that suits it.

    The panel is cut to the mesh's bounding box rather than left square, where a
    wide domain would be drawn small between bands of background: so
    ``panel_pixels`` sets the height and the width follows the mesh. Planar
    meshes only, since the view looks down z.
    """

    if mesh.topology.dim != 2:
        raise ValueError(
            f"phase-field figures are drawn looking down at a plane, and this "
            f"mesh is {mesh.topology.dim}-dimensional"
        )
    points = np.asarray(mesh.geometry.x)[:, :2]
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)
    x_center, y_center = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    width, height = float(xmax - xmin), float(ymax - ymin)
    if not (width > 0.0 and height > 0.0):
        raise ValueError("the mesh has no extent in x or in y to fit a view to")

    pixels = style["panel_pixels"]
    width_pixels = max(1, round(pixels * width / height))
    # Fitted to the panel as rounded rather than as asked for, or a panel
    # rounded narrower than the mesh would clip it at the sides.
    aspect = width_pixels / pixels
    if width_pixels > MAX_PANEL_PIXELS:
        raise ValueError(
            f"this mesh is {width / height:.1f} times wider than it is tall, so "
            f"a panel {pixels} pixels high is {width_pixels} wide, past the "
            f"{MAX_PANEL_PIXELS} an off-screen render returns intact; draw it "
            f"with a smaller panel_pixels"
        )
    return {
        "position": [
            (x_center, y_center, 1.0),
            (x_center, y_center, 0.0),
            (0.0, 1.0, 0.0),
        ],
        "parallel_scale": style["camera_margin"] * max(height, width / aspect),
        "aspect": aspect,
        "window_size": (width_pixels, pixels),
    }


## Colour


def composite(
    frame: Frame | Domain, layers: tuple[Layer, ...], style: dict[str, Any]
) -> tuple[np.ndarray, str]:
    """Several phase layers over the background, as one RGBA array.

    Each layer contributes its colour with its value as opacity, and the result
    is the opacity-weighted mean of their hues over a background that shows
    through wherever the total is under one: a phase alone shows its colour, two
    overlapping show their mixture, and thinning material fades to the
    background.
    """

    if not layers:
        raise ValueError("cannot composite an empty layer list")

    association = frame.common(layers)
    premultiplied, total = 0.0, 0.0
    for layer in layers:
        alpha = np.clip(frame.layer(layer, association)[0], 0.0, 1.0)
        premultiplied = (
            premultiplied + alpha[:, None] * np.asarray(to_rgb(layer.color))[None, :]
        )
        total = total + alpha

    rgb = np.tile(np.asarray(style["background"], dtype=float), (total.size, 1))
    painted = total > 1.0e-14
    rgb[painted] = premultiplied[painted] / total[painted, None]
    return rgba_uint8(np.clip(rgb, 0.0, 1.0), np.clip(total, 0.0, 1.0)), association


def phase_rgba(
    values: np.ndarray, color: str, value_range: tuple[float, float]
) -> np.ndarray:
    """One phase layer alone, its range stretched over the full opacity.

    A detail row shows one layer, and one that only reaches a tenth would be a
    faint smudge drawn as the composite draws it; so the row is normalized, and
    its colourbar says what the ends were.
    """

    vmin, vmax = value_range
    if is_degenerate(vmin, vmax):
        alpha = np.ones_like(values) if vmax > 0.0 else np.zeros_like(values)
    else:
        alpha = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return rgba_uint8(
        np.tile(np.asarray(to_rgb(color), dtype=float), (values.size, 1)), alpha
    )


def scalar_rgba(
    values: np.ndarray,
    value_range: tuple[float, float],
    *,
    colormap: str | None,
    style: dict[str, Any],
) -> np.ndarray:
    """A diagnostic layer, coloured by value.

    ``colormap`` names a colorcet map, stretched over the range. Without one the
    map is diverging and centred on zero when the range straddles it, since a
    sequential map hides the sign change somewhere in the middle of its ramp.
    """

    vmin, vmax = value_range
    sequential = colormaps.get_cmap(style["sequential_colormap"])
    if colormap is not None:
        # Inline: colorcet takes seconds to import, and only a named map needs it.
        import colorcet as cc

        try:
            cmap = cc.cm[colormap]
        except KeyError:
            raise ValueError(f"unknown colorcet colormap {colormap!r}") from None
        if is_degenerate(vmin, vmax):
            scaled = np.full(values.shape, 0.5)
        else:
            scaled = Normalize(vmin=vmin, vmax=vmax)(values)
    elif is_degenerate(vmin, vmax):
        cmap, scaled = sequential, np.full(values.shape, 0.5)
    elif vmin < 0.0 < vmax:
        limit = max(abs(vmin), abs(vmax))
        cmap = colormaps.get_cmap(style["diverging_colormap"])
        scaled = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)(values)
    else:
        cmap, scaled = sequential, Normalize(vmin=vmin, vmax=vmax)(values)
    return np.asarray(
        np.round(255.0 * cmap(np.clip(scaled, 0.0, 1.0))), dtype=np.uint8
    )


def is_degenerate(vmin: float, vmax: float) -> bool:
    """Whether a range has no width to normalize over.

    Exact rather than ``isclose``: a field spanning ``[0, 1e-9]``, such as a gel
    fraction where it first appears, is a real range, and a relative tolerance
    would paint it flat under a colourbar reading two different numbers. ``nan``
    counts as degenerate, so it is drawn at the midpoint: normalizing over it
    would make every entry ``nan``, which a colormap sends to a transparent
    pixel, losing the whole panel.
    """

    return not vmax > vmin


def rgba_uint8(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.round(255.0 * np.clip(np.column_stack((rgb, alpha)), 0.0, 1.0)),
        dtype=np.uint8,
    )


def alpha_ramp(color: str) -> np.ndarray:
    """A one-pixel-wide colourbar image for a phase: its hue, fading in."""

    alpha = np.linspace(0.0, 1.0, 256)
    rgb = np.tile(np.asarray(to_rgb(color), dtype=float), (alpha.size, 1))
    return np.column_stack((rgb, alpha)).reshape(alpha.size, 1, 4)


## Rasterization


def panel(
    grid: Any,
    rgba: np.ndarray,
    *,
    association: str,
    view: dict[str, Any],
    style: dict[str, Any],
    outline: bool = False,
) -> np.ndarray:
    """One rendered panel, as an image array shaped by ``view``.

    ``outline`` draws the mesh's boundary, which is Sigma: the inclusions are
    diffuse rather than cut out, so the only edges belonging to one cell are the
    outer shape's. Taken from the grid rather than from gmsh's physical groups,
    which the archived mesh is read without.
    """

    import pyvista as pv

    plotter = pv.Plotter(
        off_screen=True, window_size=view["window_size"], border=False
    )
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        plotter.enable_anti_aliasing()

    drawn = grid.copy(deep=True)
    if association == POINT:
        drawn.point_data["rgba"] = rgba
    elif association == CELL:
        drawn.cell_data["rgba"] = rgba
    else:
        raise ValueError(f"unknown render association {association!r}")

    plotter.set_background(tuple(float(c) for c in style["background"]))
    plotter.add_mesh(
        drawn,
        scalars="rgba",
        rgb=True,
        preference=association,
        show_edges=False,
        lighting=False,
        smooth_shading=False,
        interpolate_before_map=True,
    )
    if outline:
        # Edges used by exactly one cell, which on a mesh with no holes is its
        # outer boundary and nothing else.
        plotter.add_mesh(
            drawn.extract_surface(
                # Named rather than defaulted: pyvista warns that it intends to
                # change which algorithm this picks.
                algorithm="dataset_surface"
            ).extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            ),
            color=style["boundary_color"],
            line_width=max(
                1.0, style["boundary_line_width"] * view["window_size"][1]
            ),
            lighting=False,
        )
    plotter.camera_position = view["position"]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = view["parallel_scale"]
    plotter.reset_camera_clipping_range()
    image = np.asarray(plotter.screenshot(return_img=True))
    plotter.close()
    return image
