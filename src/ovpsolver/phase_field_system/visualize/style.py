"""The numbers a figure is drawn to, and the one place a subclass changes them.

Every constant a figure is drawn to is a key of :data:`PLOT_PARAMS`, which a
mixture carries as its ``plot_params`` class attribute. A subclass overrides a
key without restating the rest::

    class MyMixture(PolymerizingB):
        plot_params = {**PolymerizingB.plot_params, "dpi": 600.0}

Not a :class:`~ovpsolver.parametric.Parameters` block: these are not statements
about the material or the run, nothing validates them against each other, and in
the spec every document would carry a typography section.

Three of them -- ``panel_pixels``, ``dpi`` and ``font_scaling`` -- are also
command-line options, which start from the values here.
"""

from __future__ import annotations

from typing import Any

#: ``--save`` with no path: write the figure into the run's output folder, under
#: a name the figure chooses. Shared, so every figure taking the flag agrees on
#: what a bare one means.
BESIDE_THE_RUN = "__beside_the_run__"

#: Everything on a figure is sized off the body text, so one number moves all of
#: it: type is quoted at its natural size and scaled through here.
FONT_SCALE = 2.0


def font_size(base_size: float) -> int:
    """A point size, scaled by :data:`FONT_SCALE`."""

    return int(round(FONT_SCALE * base_size))


#: The defaults, and the full list of what a figure can be told.
#:
#: Furniture is quoted in multiples of the type it holds, so it keeps fitting its
#: text when ``font_scaling`` changes. Panels are quoted in pixels of height, the
#: width following the mesh, so a figure drawn at twice the pixels keeps its
#: proportions.
PLOT_PARAMS: dict[str, Any] = {
    ## The page
    #: Height of each rendered panel, in pixels.
    "panel_pixels": 900,
    #: Resolution the composed figure is saved at.
    "dpi": 300.0,
    ## Type
    #: Multiplies every ``*_font_size`` below. :data:`FONT_SCALE` is the same
    #: idea fixed at import; this is the one a particular figure turns.
    "font_scaling": 1.0,
    "title_font_size": font_size(12),
    "colorbar_label_font_size": font_size(6),
    "colorbar_tick_font_size": font_size(4.5),
    #: Decimal places on the two ends of a detail row's colourbar.
    "colorbar_decimals": 3,
    ## Furniture, in multiples of the type each piece holds
    #: Width of the right-hand colourbar column per phase, and its floor, in
    #: multiples of ``colorbar_label_font_size``: the rotated layer names have to
    #: fit across it.
    "colorbar_column_width": 2.34,
    "colorbar_column_min_width": 5.40,
    #: Height of the colourbar under a detail row, in multiples of
    #: ``colorbar_tick_font_size``. Its two numbers sit on the ramp rather than
    #: below it, so this is the depth that clears them.
    "split_colorbar_height": 1.92,
    ## Rendering
    #: What an unpainted pixel is, and so what a composite fades towards.
    "background": (1.0, 1.0, 1.0),
    #: The outer boundary, where a figure draws it. The width is a fraction of
    #: panel height, so the line keeps its weight at any ``panel_pixels``.
    "boundary_color": "black",
    "boundary_line_width": 0.0025,
    #: Half the view's height, as a fraction of the mesh's: 0.5 fits the mesh
    #: exactly, and more leaves a margin.
    "camera_margin": 0.52,
    #: For a diagnostic row whose range straddles zero, and for one that does
    #: not. A signed quantity gets a diverging map centred on zero, since which
    #: sign it has is usually the point.
    "diverging_colormap": "coolwarm",
    "sequential_colormap": "viridis",
}


def resolved(
    system_class: type, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The mixture's plot parameters, with ``overrides`` on top and type scaled.

    Checked against :data:`PLOT_PARAMS` rather than merged blindly, so a
    misspelled key is an error rather than a setting that silently did nothing.
    """

    params = dict(getattr(system_class, "plot_params", PLOT_PARAMS))
    for key, value in (overrides or {}).items():
        if key not in PLOT_PARAMS:
            raise ValueError(
                f"unknown plot parameter {key!r}; this mixture takes "
                f"{', '.join(sorted(PLOT_PARAMS))}"
            )
        params[key] = value
    missing = set(PLOT_PARAMS) - set(params)
    if missing:
        raise ValueError(
            f"{system_class.__name__}.plot_params is missing "
            f"{', '.join(sorted(missing))}; override it as "
            f"{{**super_class.plot_params, ...}} so the rest is inherited"
        )
    if params["panel_pixels"] <= 0 or params["dpi"] <= 0.0:
        raise ValueError("panel size and dpi are positive")

    # Scaled here, the one place every size passes through, so that a size added
    # later cannot be missed.
    scaling = float(params["font_scaling"])
    if scaling <= 0.0:
        raise ValueError(f"font_scaling is positive, got {scaling}")
    return {
        key: max(1, round(value * scaling)) if key.endswith("_font_size") else value
        for key, value in params.items()
    }


def draw_without_a_window() -> None:
    """Non-interactive matplotlib and off-screen pyvista, for figures bound for files.

    Otherwise both hold a window-server context until the interpreter exits, so a
    process that has written its figure can sit there instead of returning the
    prompt. It happens only where there is a display to connect to, not over ssh.
    Called by the command line when a figure is saved and by
    :func:`~.phases_movie.movie`, which only writes files; a library caller's
    backend is otherwise theirs to choose.
    """

    # Inline, here and below: this module is imported on the solver's run path,
    # which has no use for matplotlib or pyvista.
    import matplotlib
    import pyvista

    matplotlib.use("Agg", force=True)
    pyvista.OFF_SCREEN = True


def configure_tex_like_fonts() -> None:
    """Serif fonts and STIX mathtext, so labels match the surrounding document."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
