"""The chemistry-validation figure: does the gelation solver follow the exact solution?

For a run whose only interesting axis is ``z``. The left panel is the sol/gel
split at one cell against the characteristic solution of the gelation kinetics;
the right is the generating function ``u(t, z)`` in that cell, as the MUSCL
reconstruction the solver took, against the characteristics it approximates.
Together they show whether the finite-volume partition in ``z`` resolves the gel
point.

The ``z`` partition comes from the spec's parameters object, so its edges are the
ones the run computed rather than a copy that could drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ... import burgers_upwind_values_muscl_vanleer
from ...fem.save import Run
from ..entry import SpecEntryPoint
from ..run import read
from .style import (
    BESIDE_THE_RUN,
    configure_tex_like_fonts,
    draw_without_a_window,
    font_size,
    resolved,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...polymerizing_b.phase_field.parameters import (
        PolymerizingPhaseFieldParameters,
    )
    from ..parameters import PhaseFieldSystemParameters
    from ..system import PhaseFieldSystem

#: The saved names this figure is made of: ``phi`` picks the cell, the two
#: volume fractions are the split being validated, and ``u`` is the binned
#: generating function.
REQUIRED = ("phi", "phi_sol", "phi_gel", "u")

#: Layout of the two panels. Not in ``plot_params``, which is the look of the
#: phase figures; the type scale and fonts are shared.
FIGURE_SIZE = (17.0, 4.8)
SUBPLOT_ADJUST = {
    "left": 0.055,
    "right": 0.985,
    "bottom": 0.18,
    "top": 0.9,
    "wspace": 0.15,
}
TITLE_FONT_SIZE = 24


def plot_gelation(
    system_class: type[PhaseFieldSystem],
    spec: str | Path,
    *,
    phase: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    characteristics_every: int = 1,
    **style_overrides: Any,
) -> plt.Figure:
    """Draw the gelation figure for one polymerizing phase of ``spec``'s run.

    ``phase`` names it, and can be left out when there is only one.
    ``characteristics_every`` thins the frames the right-hand panel draws.
    """

    # Inline, here and below: seaborn takes over a second to import, and the
    # command line imports this module for every command.
    import seaborn as sns

    if characteristics_every <= 0:
        raise ValueError("characteristics_every is a positive number of frames")

    style = resolved(system_class, style_overrides)
    parameters = read(system_class.Parameters, Path(spec))
    phase_parameters = _phase(parameters, phase)
    run = Run.of(parameters)
    tracked = Tracked(run, phase_parameters)

    sns.set_theme(style="whitegrid", context="talk")
    configure_tex_like_fonts()
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, constrained_layout=False)
    _sol_gel_panel(axes[0], tracked)
    _characteristics_panel(axes[1], tracked, every=characteristics_every)
    figure.subplots_adjust(**SUBPLOT_ADJUST)

    if save is not None:
        path = run.path / "gelation.pdf" if str(save) == BESIDE_THE_RUN else Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=style["dpi"], bbox_inches="tight", pad_inches=0.02)
    elif show:
        plt.show()
    return figure


## What is being compared


class Tracked:
    """One cell of the run, and the exact solution it should be following.

    The cell holding the most of the phase at the start: for a homogeneous run
    any cell, and otherwise the one worth looking at. Every field is read per
    cell, which for these cell-constant fields is the dof itself.
    """

    def __init__(self, run: Run, parameters: PolymerizingPhaseFieldParameters) -> None:
        self.run = run
        self.parameters = parameters
        self.times = np.asarray(run.times, dtype=float)
        if self.times.size == 0:
            raise ValueError(f"{run.path} has no frames to plot")

        phi = self._cells("phi")
        self.cell = int(np.argmax(phi[0]))
        self.initial_phi = float(phi[0, self.cell])
        if self.initial_phi <= 0.0:
            raise ValueError(
                f"{parameters.name} starts at zero everywhere, so there is no "
                f"gelation to plot"
            )
        self.sol = self._cells("phi_sol")[:, self.cell]
        self.gel = self._cells("phi_gel")[:, self.cell]
        #: The bin averages of ``u`` at the cell, as ``(frame, bin)``.
        self.u_b = self._cells("u", components=self.bins)[:, self.cell, :]

    ## Public

    @property
    def edges(self) -> np.ndarray:
        """The ``z`` partition, as the phase's parameters compute it."""

        return np.asarray(self.parameters.z_partition.edges, dtype=float)

    @property
    def bins(self) -> int:
        return int(self.edges.size - 1)

    @property
    def gamma(self) -> float:
        return float(self.parameters.crosslinking_rate)

    @property
    def functionality(self) -> int:
        return int(self.parameters.monomer_functionality)

    @property
    def monomer_volume(self) -> float:
        return float(self.parameters.monomer_volume)

    @property
    def c_0(self) -> float:
        """Monomer concentration at the tracked cell, at the start."""

        return self.initial_phi / self.monomer_volume

    @property
    def gel_time(self) -> float:
        """The gel point, ``nu / (f (f - 2) gamma phi(0))``."""

        if self.functionality <= 2:
            raise ValueError("gelation needs a functionality above two")
        return self.monomer_volume / (
            self.functionality * (self.functionality - 2) * self.gamma * self.initial_phi
        )

    ## Private helpers

    def _cells(self, name: str, *, components: int = 1) -> np.ndarray:
        """One saved field as ``(frame, cell)``, or ``(frame, cell, component)``."""

        qualified = f"{self.parameters.name}.{name}"
        if not self.run.has(qualified):
            raise ValueError(
                f"the gelation figure needs {qualified!r}; this run saved "
                f"{', '.join(self.run.names)}. Add "
                f"{', '.join(REQUIRED)} to {self.parameters.name}'s save list."
            )
        field = self.run.field(qualified)
        shape = (components,) if components > 1 else ()
        if field.is_cellwise and field.shape == shape:
            values = np.asarray(field.values, dtype=float)
        else:
            space = self.run.cell_space(shape)
            values = np.stack(
                [
                    np.asarray(field.sample(frame, space).x.array, dtype=float)
                    for frame in range(field.n_frames)
                ]
            )
        if components == 1:
            return values
        return values.reshape(values.shape[0], -1, components)


## The panels


def _sol_gel_panel(axis: plt.Axes, tracked: Tracked) -> None:
    """Simulated sol and gel against the characteristic split, with the gel point."""

    import seaborn as sns

    exact_sol = _characteristic_sol(tracked)
    for values, label in (
        (exact_sol, r"$\phi^{\rm sol}$ [char. sol.]"),
        (tracked.initial_phi - exact_sol, r"$\phi^{\rm gel}$ [char. sol.]"),
    ):
        axis.plot(
            tracked.times,
            values,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=label,
            zorder=2.6,
        )

    for values, color, label in (
        (tracked.sol, tracked.parameters.color_sol, r"$\phi^{\rm sol}$ [f.v.]"),
        (tracked.gel, tracked.parameters.color_gel, r"$\phi^{\rm gel}$ [f.v.]"),
    ):
        _smooth(axis, tracked.times, values, color)
        sns.scatterplot(
            x=tracked.times,
            y=values,
            ax=axis,
            color=color,
            label=label,
            s=12,
            edgecolor="none",
            zorder=5,
        )

    gel_time = tracked.gel_time
    axis.axvline(gel_time, color="0.25", linestyle="--", linewidth=0.9, zorder=1)
    axis.annotate(
        r"$\tau_{\rm gel}=\frac{\nu}{f(f-2)\gamma\phi(0)}" + f"={gel_time:.3g}$",
        xy=(gel_time, 0.5),
        xycoords=("data", "axes fraction"),
        xytext=(-7, 0),
        textcoords="offset points",
        rotation=90,
        ha="right",
        va="center",
        fontsize=font_size(12),
    )
    axis.set_xlabel(r"$t$", fontsize=font_size(11))
    axis.set_ylabel(r"$\phi$", fontsize=font_size(11))
    axis.set_title(r"Sol $\to$ gel reaction", fontsize=TITLE_FONT_SIZE)
    axis.legend(
        title=None,
        fontsize=font_size(8),
        loc="center left",
        bbox_to_anchor=(0.6, 0.5),
        frameon=True,
        markerscale=2.2,
    )
    axis.tick_params(axis="both", which="both", labelsize=font_size(8))


def _characteristics_panel(axis: plt.Axes, tracked: Tracked, *, every: int) -> None:
    """The binned ``u(t, z)`` the solver took, over the characteristics it follows."""

    # Inline: colorcet takes seconds to import, and the command line imports
    # this module for every command.
    import colorcet as cc

    times, edges = tracked.times, tracked.edges
    frames = np.arange(0, times.size, every, dtype=int)
    if frames[-1] != times.size - 1:
        frames = np.append(frames, times.size - 1)

    colormap = cc.cm.CET_C8
    norm = mpl.colors.Normalize(vmin=float(times[0]), vmax=float(times[-1]))
    scalars = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)

    z_max = max(1.4, float(np.max(edges)))
    xi = np.linspace(0.0, 1.0, max(4 * tracked.bins, 512))
    u_initial = _monomer_initial_value(xi, tracked)
    upwind_z = np.empty(tracked.bins)
    upwind_u = np.empty((1, tracked.bins))

    axis.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    axis.scatter(
        edges,
        np.zeros_like(edges),
        color="0.25",
        marker="|",
        s=35,
        linewidths=0.8,
        label="z-partition edges",
        zorder=4,
    )
    for index, frame in enumerate(frames):
        time = float(times[frame])
        axis.plot(
            xi + tracked.gamma * u_initial * time,
            u_initial,
            color=colormap(norm(time)),
            zorder=1,
        )
        state = np.ascontiguousarray(tracked.u_b[frame][None, :])
        burgers_upwind_values_muscl_vanleer(state, upwind_z, upwind_u, edges)
        axis.scatter(
            upwind_z,
            upwind_u[0],
            facecolors="none",
            edgecolors="black",
            linewidths=0.4,
            marker="o",
            s=5,
            label="MUSCL upwind values" if index == 0 else None,
            zorder=3,
        )

    axis.set_xlim(0.0, z_max)
    axis.set_xlabel(r"$z$", fontsize=font_size(11))
    axis.set_ylabel(r"$u(t, z)$", fontsize=font_size(11))
    axis.set_title("Upwind-$z$ vs. characteristic solution", fontsize=TITLE_FONT_SIZE)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5 / z_max, 0.08),
        fontsize=font_size(8),
        markerscale=2.2,
    )
    axis.tick_params(axis="both", which="both", labelsize=font_size(8))
    colorbar = axis.figure.colorbar(scalars, ax=axis)
    colorbar.set_label(r"$t$", fontsize=font_size(10))
    colorbar.ax.tick_params(labelsize=font_size(8))


def _smooth(axis: plt.Axes, times: np.ndarray, values: np.ndarray, color: str) -> None:
    # Inline: scipy.interpolate is slow to import, and the command line imports
    # this module for every command.
    from scipy.interpolate import make_interp_spline

    if times.size < 2:
        return
    dense = np.linspace(float(times[0]), float(times[-1]), max(256, 4 * times.size))
    try:
        smoothed = make_interp_spline(times, values, k=min(3, times.size - 1))(dense)
    except ValueError:
        smoothed = np.interp(dense, times, values)
    axis.plot(dense, smoothed, color=color, linewidth=1.2, alpha=0.85, zorder=4)


## The exact solution


def _monomer_initial_value(xi: np.ndarray, tracked: Tracked) -> np.ndarray:
    """The generating function of pure monomer, ``f c_0 (xi - xi^(f-1))``."""

    degree = tracked.functionality - 1
    return tracked.functionality * tracked.c_0 * (xi - xi**degree)


def _characteristic_sol(tracked: Tracked) -> np.ndarray:
    """The sol fraction the characteristics give, at every saved time.

    ``nu (2 u_bar - u(t, 1)) / (f - 2)``, with ``u_bar`` the integral of ``u``
    over ``[0, 1]``. Along the characteristics ``z = xi + gamma u_0(xi) t`` that
    is the integral of ``u_0 dz`` over the preimage of ``[0, 1]``: a polynomial in
    ``xi``, whose antiderivative is written out.
    """

    functionality, gamma, c_0 = tracked.functionality, tracked.gamma, tracked.c_0
    degree = functionality - 1
    amplitude = functionality * c_0
    values = []
    for time in tracked.times:
        time = float(time)
        xi = _unit_interval_endpoint(time, tracked)
        coefficient = gamma * functionality * c_0 * time
        u_bar = amplitude * (
            (1.0 + coefficient) * xi**2 / 2.0
            - (1.0 + coefficient + coefficient * degree)
            * xi ** (degree + 1)
            / (degree + 1)
            + coefficient * degree * xi ** (2 * degree) / (2 * degree)
        )
        u_trace = _monomer_initial_value(np.asarray(xi), tracked).item()
        values.append(
            tracked.monomer_volume * ((2.0 * u_bar - u_trace) / (functionality - 2))
        )
    return np.clip(np.array(values, dtype=float), 0.0, tracked.initial_phi)


def _unit_interval_endpoint(time: float, tracked: Tracked) -> float:
    """Where the characteristic reaching ``z = 1`` at ``time`` started.

    One until the gel point. Past it ``xi -> z`` folds, so that ``z = 1`` has
    several preimages, and this is the one below the fold.
    """

    # Inline: scipy.optimize is slow to import, and the command line imports
    # this module for every command.
    from scipy.optimize import brentq

    if time <= tracked.gel_time or np.isclose(time, tracked.gel_time):
        return 1.0

    functionality, gamma, c_0 = tracked.functionality, tracked.gamma, tracked.c_0
    degree = functionality - 1
    coefficient = gamma * functionality * c_0 * time
    fold = ((1.0 + coefficient) / (coefficient * degree)) ** (1.0 / (degree - 1))

    def offset(xi: float) -> float:
        u = _monomer_initial_value(np.asarray(xi), tracked).item()
        return xi + gamma * u * time - 1.0

    return float(brentq(offset, 0.0, fold))


def _phase(
    parameters: PhaseFieldSystemParameters, name: str | None
) -> PolymerizingPhaseFieldParameters:
    """The polymerizing phase to plot: the one named, or else the only one.

    A phase crosslinks by being of the polymerizing type, the only one that
    declares a crosslinking rate.
    """

    # Inline: this is the one figure in the core about a particular mixture, and
    # at the top it would pull the polymerizing package into every mixture's
    # command line.
    from ...polymerizing_b.phase_field.parameters import (
        PolymerizingPhaseFieldParameters,
    )

    candidates = [
        phase
        for phase in parameters.phase_field_parameters
        if isinstance(phase, PolymerizingPhaseFieldParameters)
    ]
    if not candidates:
        raise ValueError(
            "this mixture has no polymerizing phase, so there is no gelation to plot"
        )
    if name is None:
        if len(candidates) > 1:
            raise ValueError(
                f"name a phase with --phase; this mixture crosslinks "
                f"{', '.join(phase.name for phase in candidates)}"
            )
        return candidates[0]
    for phase in candidates:
        if phase.name == name:
            return phase
    raise ValueError(
        f"{name!r} is not a polymerizing phase of this mixture; it has "
        f"{', '.join(phase.name for phase in candidates)}"
    )


## The command line


class VisualizeGelation(SpecEntryPoint):
    """Validate the crosslinking chemistry against its characteristic solution.

    For a run whose only interesting axis is the gelation coordinate. The left
    panel is the sol/gel split at the busiest cell against the exact split; the
    right is the binned generating function against the characteristics it
    approximates, so the resolution of the gel point is visible directly.

    Needs the phase to have saved phi, phi_sol, phi_gel and u.
    """

    name = "visualize.gelation"
    summary = "compare the gelation solver against its exact solution"

    #: The figure is on disk by the time this returns, and the exit hooks
    #: dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: ArgumentParser) -> None:
        super().declare(parser)
        parser.add_argument(
            "--phase",
            help="which polymerizing phase to plot; needed only if there are two",
        )
        parser.add_argument(
            "--save",
            nargs="?",
            const=BESIDE_THE_RUN,
            help=(
                "where to write the figure. Bare, it writes gelation.pdf into "
                "the run's own output folder. Absent, the figure opens in a "
                "window instead."
            ),
        )
        parser.add_argument(
            "--characteristics-every",
            type=int,
            default=1,
            dest="characteristics_every",
            help="draw every Nth frame in the right-hand panel",
        )

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments["show"] = arguments["save"] is None
        return arguments

    def __call__(
        self, system_class: type[PhaseFieldSystem], *, spec: Path, **options: Any
    ) -> plt.Figure:
        if not options.get("show"):
            draw_without_a_window()
        return plot_gelation(system_class, spec, **options)
