"""Which cells set the transport step bound, and do they empty on content or on velocity?

A transported variable may step only as far as its fastest-emptying cell can sustain
its outflow
(:meth:`~ovpsolver.phase_field_system.transport.Transported.cfl_limit`),

    dt <= cfl min_K held_K / O_K,    held_K = int_K chi phi_K,

so a run whose step collapses has cells making that ratio small, in one of two ways
that call for different fixes.

On its own velocity. Where the upwind selector agrees with the velocity, a cell loses
its own phase, so ``phi_K`` cancels from the ratio: what is left is the cell's size
over the chi-weighted speed it is emptied at, and only a large outward velocity makes
the bound small.

On its content. Where the lagged selector disagrees with the solved velocity, a cell
loses its *neighbour's* phase, and the ratio picks up ``phi_K / phi_neighbour``. A cell
depleted against the cells around it binds there however slowly it is emptied.

A frame is written once per macro step, after its sub-steps, so the selector a run
used is not saved and is not guessed at. Both bounds are assembled instead, from the
frame's own phase and velocity -- the content the next sub-step starts from and the
velocity its selector is frozen at:

``own``
    every selector agreeing with the velocity: the bound on velocity alone, and the
    one the next sub-step meets if it solves for this velocity again.
``flipped``
    every selector reversed, so each facet a cell loses across carries the
    neighbour's phase: the most a lagging selector can make of the cell's content.

For a phase on one velocity a facet loses one or the other, so no selector takes the
bound below half the smaller of the two, and whichever is smaller says what the cell
is exposed to. Both are the solve's own facet forms
(:func:`~ovpsolver.fem.elements.dg0.dg0_outflow`), reported as limits -- ``cfl``
times the emptying time -- so they read directly against the ``limit=`` of a run's
``step_retry`` lines.

The bounds weigh content and facets by the analytic indicator, where the solve weighs
the content by its tabulated copy at ``indicator_quadrature_degree``: the same
function under a different rule, which can move a band cell's content by a few per
cent and does not change which cells bind. A source that drains a cell -- extraction
at the inclusion surfaces or through ``Sigma`` -- is not counted; the row carries
those on other measures, and a spec imposing one would want it added here.

``--variable`` reads one of a polymerizing phase's site-counting states instead
(:data:`SITE_PIECES`). Its sol piece carries what is left once the gel's share is
taken out, a difference of two states that can be far more depleted against its
neighbours than either state is, and a lagging selector prices exactly that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import ufl
from dolfinx import fem

from ...fem.elements.dg0 import dg0_emptying_times, dg0_outflow
from ...fem.reduce import assemble_nodal_values
from .base import (
    FREE_FLUID,
    Diagnostic,
    Unanswerable,
    declare_frames,
    flux_pieces,
    frames_of,
)
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Callable

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--cells``: how many of the tightest cells the anatomy lists.
CELLS = 12

#: Default for ``--step-tol``: how far below the macro step either bound may fall, as a
#: fraction of it. A hundredth is a hundred sub-steps to every step the spec asked for,
#: which is a run spending its time on a handful of cells.
STEP_TOLERANCE = 1.0e-2

#: How much faster a reversed selector must drain a cell than its own velocity before
#: the cell is said to empty on its content. The ratio is the depletion ``phi_K``
#: against the phase beside it, and any cell of a moving interface is within a factor
#: of two of its neighbours: its bound is still its speed.
DEPLETION = 2.0

#: How a polymerizing phase's site-counting states split their flux, as
#: :meth:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseField.transported`
#: declares it, written against the saved fields since a finished run has no phase to
#: ask. The gel velocity carries the sites on chains reaching ``z = 1`` -- the trace
#: moment, weighted as each state counts a site, with ``z_B`` the midpoint of the
#: partition's last cell -- and the sol velocity carries the rest.
SITE_PIECES = {
    "c_x": lambda f, z_B: (("v_s", f["c_x"] - f["u_trace"]), ("v_g", f["u_trace"])),
    "u_bar": lambda f, z_B: (
        ("v_s", f["u_bar"] - 0.5 * f["u_trace"]),
        ("v_g", 0.5 * f["u_trace"]),
    ),
    "u_trace": lambda f, z_B: (
        ("v_s", (1.0 - z_B) * f["u_trace"]),
        ("v_g", z_B * f["u_trace"]),
    ),
}

#: The saved fields each of :data:`SITE_PIECES` reads.
SITE_FIELDS = {
    "c_x": ("c_x", "u_trace", "v_s", "v_g"),
    "u_bar": ("u_bar", "u_trace", "v_s", "v_g"),
    "u_trace": ("u_trace", "v_s", "v_g"),
}


class Emptying(Diagnostic):
    """Find the cells that bound the transport step, and whether content or speed does it."""

    name = "analyze.emptying"
    summary = "find the cells that set the step bound, and why they empty"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--phase",
            help="only this phase; default is every phase that saved its flux",
        )
        parser.add_argument(
            "--variable",
            default="phi",
            choices=("phi", *SITE_PIECES),
            help="the transported variable whose bound is read; default the phase",
        )
        parser.add_argument(
            "--frame",
            type=int,
            default=-1,
            help="the frame whose tightest cells are listed; default is the last",
        )
        parser.add_argument(
            "--cells",
            type=int,
            default=CELLS,
            help="how many of the tightest cells to list",
        )
        parser.add_argument(
            "--step-tol",
            type=float,
            dest="step_tol",
            default=STEP_TOLERANCE,
            help="how far below the macro step a bound may fall, as a fraction",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        phase: str | None = None,
        variable: str = "phi",
        frame: int = -1,
        cells: int = CELLS,
        step_tol: float = STEP_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        chosen = []
        for one in parameters.phase_field_parameters:
            if phase not in (None, one.name):
                continue
            if variable == "phi":
                pieces = flux_pieces(run, one.name)
                if pieces is None:
                    continue
                fields = tuple(
                    dict.fromkeys(("phi", *(n for piece in pieces for n in piece)))
                )
                rules = lambda f, z_B, pieces=pieces: tuple(
                    (velocity, f[carried]) for velocity, carried in pieces
                )
                z_B = 0.0
            else:
                partition = getattr(one, "z_partition", None)
                fields = SITE_FIELDS[variable]
                if partition is None or not all(
                    run.has(f"{one.name}.{n}") for n in fields
                ):
                    continue
                rules = SITE_PIECES[variable]
                z_B = float(partition.midpoint(-1))
            chosen.append((one.name, fields, rules, z_B))
        if not chosen:
            raise Unanswerable(
                f"no phase saved what the outflow of {variable} is a product of; "
                "add the variable and the velocities that carry it (and, for a "
                "site-counting state, 'u_trace') to the phase's 'save' list"
            )

        timestepping = parameters.solver.timestepping
        chi = parameters.solver.diffuse_domain.bulk_indicator_on(run.cell_space())
        report.field("cfl", f"{timestepping.cfl_positive_transport:g}")
        report.field("macro step", f"{timestepping.dt:g}")
        report.field(
            "limits",
            "cfl times the emptying time; 'own' with every selector agreeing with "
            "the velocity, 'flipped' with every one reversed",
        )
        report.field(
            "indicator",
            "at the centroid, floor subtracted: 1 in the fluid, 0.5 on the level set, "
            "0 inside an inclusion",
        )
        report.note()

        walked = frames_of(run, frames, stride=stride)
        for name, fields, rules, z_B in chosen:
            self._one_phase(
                report,
                run,
                parameters,
                name,
                variable,
                fields,
                rules,
                z_B,
                chi,
                frame=frame % run.n_frames,
                cells=cells,
                step_tol=step_tol,
                walked=walked,
            )

    ## Private helpers

    def _one_phase(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        name: str,
        variable: str,
        fields: tuple[str, ...],
        rules: "Callable[[dict[str, Any], float], tuple[tuple[str, Any], ...]]",
        z_B: float,
        chi: np.ndarray,
        *,
        frame: int,
        cells: int,
        step_tol: float,
        walked: tuple[int, ...],
    ) -> None:
        timestepping = parameters.solver.timestepping
        cfl = float(timestepping.cfl_positive_transport)
        solver = run.solver_parameters
        mesh = run.mesh
        space = run.cell_space()
        test = ufl.TestFunction(space)
        normal = ufl.FacetNormal(mesh)
        weight = parameters.solver.diffuse_domain.chi_eps_ufl(mesh)
        dS = solver.get_mesh_dS()

        # Built once against functions refilled per frame: the content, and the
        # outflow under both selectors.
        saved = {field: run.field(f"{name}.{field}") for field in fields}
        functions = {field: one.function(0) for field, one in saved.items()}
        pieces = rules(functions, z_B)
        held_form = fem.form(weight * functions[variable] * test * solver.get_mesh_dx())
        own_form, flipped_form = (
            fem.form(
                dg0_outflow(
                    [
                        (sign * functions[velocity], functions[velocity], weight * part)
                        for velocity, part in pieces
                    ],
                    test,
                    normal,
                    dS,
                )
            )
            for sign in (1.0, -1.0)
        )

        def read(index: int) -> dict[str, np.ndarray]:
            for field, one in saved.items():
                one.load_into(functions[field], index)
            held = assemble_nodal_values(held_form)
            own = assemble_nodal_values(own_form)
            flipped = assemble_nodal_values(flipped_form)
            own_limit = cfl * dg0_emptying_times(held, own)
            flipped_limit = cfl * dg0_emptying_times(held, flipped)
            return {
                "value": saved[variable].cell_values(index),
                "own": own,
                "flipped": flipped,
                "own limit": own_limit,
                "flipped limit": flipped_limit,
                "tightest": np.minimum(own_limit, flipped_limit),
                "drains by": np.where(
                    flipped > DEPLETION * own, "content", "velocity"
                ),
            }

        report.note(
            f"{name}.{variable}: flux read as "
            + ", ".join(f"{part!s} on {velocity}" for velocity, part in pieces)
        )
        history = report.table(
            Table(
                f"{name} frame",
                "time",
                "own",
                "flipped",
                "cell",
                "indicator",
                variable,
                "drains by",
                formats={"indicator": ".3g", variable: ".3g"},
            )
        )
        times = np.asarray(run.times, dtype=float)
        worst, worst_frame = float("inf"), 0
        for index in walked:
            one = read(index)
            cell = int(np.argmin(one["tightest"]))
            if not np.isfinite(one["tightest"][cell]):
                history.add(index, float(times[index]), *([None] * 6))
                continue
            if one["tightest"][cell] < worst:
                worst, worst_frame = float(one["tightest"][cell]), index
            history.add(
                index,
                float(times[index]),
                float(one["own limit"].min()),
                float(one["flipped limit"].min()),
                cell,
                float(chi[cell]),
                float(one["value"][cell]),
                str(one["drains by"][cell]),
            )

        # The tightest cells of one frame. ``nbr`` is the variable a reversed
        # selector carries out of the cell, averaged over the facets it loses across
        # by how much each takes -- its value times the ratio of the two outflows --
        # so against the cell's own value it is the depletion ``flipped`` prices in.
        # ``|v|`` is the fastest piece's speed at the centroid.
        one = read(frame)
        order = np.argsort(one["tightest"])[:cells]
        order = order[np.isfinite(one["tightest"][order])]
        speed = np.max(
            [
                np.linalg.norm(saved[velocity].cell_values(frame), axis=1)
                for velocity in dict.fromkeys(velocity for velocity, _ in pieces)
            ],
            axis=0,
        )
        fluid = chi >= FREE_FLUID
        report.note()
        report.note(
            f"{name}.{variable}: the {len(order)} tightest cells at frame {frame}, "
            f"t = {float(times[frame]):.6g}; median |v| in the fluid "
            f"{float(np.median(speed[fluid])) if fluid.any() else float('nan'):.4g}"
        )
        table = report.table(
            Table(
                f"{name} cell",
                "indicator",
                variable,
                "nbr",
                "own",
                "flipped",
                "|v|",
                "drains by",
                formats={
                    "indicator": ".3g",
                    variable: ".3g",
                    "nbr": ".3g",
                    "|v|": ".4g",
                },
            )
        )
        for cell in order:
            own = one["own"][cell]
            table.add(
                int(cell),
                float(chi[cell]),
                float(one["value"][cell]),
                float(one["value"][cell] * one["flipped"][cell] / own)
                if own > 0.0
                else None,
                float(one["own limit"][cell]),
                float(one["flipped limit"][cell]),
                float(speed[cell]),
                str(one["drains by"][cell]),
            )

        report.note()
        report.check(
            worst >= step_tol * float(timestepping.dt),
            f"{name}.{variable}: no cell holds the step far below the macro step",
            f"a limit of {worst:.3e} at frame {worst_frame}, against {step_tol:g} of "
            f"the macro step {float(timestepping.dt):g}; the tables below say which "
            f"cells, and whether their content or their speed does it",
        )
        report.note()


