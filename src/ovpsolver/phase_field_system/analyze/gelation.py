"""Does the chain of inequalities the gelation scheme rests on still hold?

A polymerizing phase never evolves a gel volume fraction. It evolves the phase
``phi`` and the moments of the degree distribution, and reads the split off them:

    phi_s = nu m_1,   phi_g = phi - phi_s,
    m_0   = c_x / 2 - u_bar,
    m_1   = (2 u_bar - u_trace) / (f - 2),
    c_xg  = u_trace,  c_xs = c_x - u_trace.

So the gel is a *difference*, and the discretization owes an argument that it stays
non-negative. The argument is the chain

    0 <= m_0 <= m_1 <= phi / nu,

and this checks it link by link. ``m_0 >= 0`` because a chain count is a count; it
is what the live half of the convex-split entropy takes the logarithm of, so it is
the one link the scheme actively defends.
``nu m_1 <= phi`` because the sol is part of the phase: it is ``phi_g >= 0`` one
level down, and why nothing in the split is clipped.

``m_0 <= m_1``, the number-average degree ``m_1 / m_0`` being at least one, does not
hold exactly and cannot. The trace moment samples the generating function at
``z = 1`` through the last cell of the gelation partition, with an ``O(dz)`` error
that enters ``m_1`` and not ``m_0``: ``m_1`` comes out low by ``u_trace / (f - 2)``,
which at pure monomer, where the true ratio is exactly one, is the whole of the
ordering. The deficit is largest at the initial condition and shrinks as
polymerization raises the real degree past it. That is the resolution of ``z`` and
not a broken scheme, so it is measured against ``dz`` of the last cell rather than
against zero, and a deficit much larger than the partition explains is the failure.
Nor does it endanger the barrier's protection of ``phi_s``: the residue is
proportional to the moments, so ``m_0`` held off zero holds ``m_1`` off zero at a
fixed ratio.

Margins are reported absolutely, since a violation is a violation, and against
``phi / nu``, the top of the chain, since the quantities have different units and a
drained region has small margins everywhere without anything being wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Diagnostic, Unanswerable, declare_frames, frames_of
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: The saved states the chain is built from, all cell-constant primitives of the
#: scheme. ``phi_sol`` and ``phi_gel`` are derived from these, so reading the moments
#: themselves is what makes this a check rather than a restatement of the split.
REQUIRED = ("phi", "c_x", "u_bar", "u_trace")

#: The links that hold exactly, in chain order: the lowest to break says where the
#: scheme lost its footing. The last two are the site split, the same statement one
#: level down.
EXACT_LINKS = ("m_0", "phi/nu - m_1", "c_xs", "c_xg")

#: The ordering link, checked against the resolution of ``z`` rather than against
#: zero, for the reason the module docstring gives.
ORDERING = "m_1 - m_0"

#: Every margin the table reports, ordering included.
LINKS = ("m_0", ORDERING, "phi/nu - m_1", "c_xs", "c_xg")

#: Default for ``--tol``: how far an exact margin may go negative, as a fraction
#: of the largest ``phi / nu`` at the same frame. Relative because the moments
#: scale with how much material there is. At this level a violation is arithmetic;
#: well above it, a link has broken.
MARGIN_TOLERANCE = 1.0e-9

#: How many widths of the partition's last cell the ordering deficit may reach,
#: relative to ``m_0``. The deficit is ``u_trace / (f - 2)`` and ``u_trace`` is
#: ``O(dz)`` times a moment, so one width is the scale and this is the slack on
#: top of it. Generous, because the constant in front of ``dz`` depends on the shape
#: of the generating function near ``z = 1``; a deficit an order past this is an
#: ordering that has inverted, not a partition that needs refining.
ORDERING_WIDTHS = 4.0

#: How far the saved split may depart from the one rebuilt from the moments, over
#: ``phi``. Tighter than the margins because the saver evaluates the same
#: expression, so only rounding lies between them.
SPLIT_TOLERANCE = 1.0e-12


class Gelation(Diagnostic):
    """The gelation chain of inequalities, link by link and frame by frame.

    Fails if an exact link goes negative by more than ``--tol``, or the ordering
    past what the partition explains. Reported per link because they break for
    different reasons and the fixes differ: the lowest is the log barrier or the
    moment rows' step bound, and the upper ones are the trace moment's resolution
    in ``z``.
    """

    name = "analyze.gelation"
    summary = "check the sol/gel chain of inequalities holds cell by cell"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--phase",
            help="which polymerizing phase; default is every one that saved the moments",
        )
        parser.add_argument(
            "--tol",
            type=float,
            default=MARGIN_TOLERANCE,
            help=(
                "how far a margin may go negative, as a fraction of phi / nu "
                "at the same frame"
            ),
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        phase: str | None = None,
        tol: float = MARGIN_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        phases = self._phases(run, parameters, phase)
        report.field("chain", "0 <= m_0 <= m_1 <= phi / nu")
        report.note()

        times = np.asarray(run.times, dtype=float)
        walk = frames_of(run, frames, stride=stride)
        # Floor subtracted, so zero inside an inclusion, where the phase is a
        # regularization: a link broken there is still broken, since the entropy
        # takes the logarithm of m_0 in every cell, but it is a different story.
        indicator = parameters.solver.diffuse_domain.bulk_indicator_on(
            run.cell_space()
        )
        for name, phase_parameters in phases:
            nu = float(phase_parameters.monomer_volume)
            functionality = float(phase_parameters.monomer_functionality)
            last_width = float(phase_parameters.z_partition.widths[-1])
            report.field(
                name,
                f"nu = {nu:.6e}, f = {functionality:g}, "
                f"last z cell = {last_width:.4e}",
            )
            table = report.table(
                Table(
                    "frame",
                    "time",
                    *LINKS,
                    "deficit/m_0",
                    "indicator there",
                    "split as saved",
                    formats={"indicator there": ".3g"},
                )
            )
            worst = {link: (0.0, 0, 0.0, 0) for link in LINKS}
            # None until a frame carries the split: seeded with a number, a run
            # that saved no phi_sol would pass with a departure of zero.
            drift: float | None = None
            for frame in walk:
                margins, scale, deficit, deficit_cell, split_error = self._frame(
                    run, name, frame, nu, functionality
                )
                if not np.isnan(split_error):
                    drift = split_error if drift is None else max(drift, split_error)
                for link, values in margins.items():
                    cell = int(np.argmin(values))
                    smallest = float(values[cell])
                    reference = scale if link in EXACT_LINKS else 1.0
                    relative = (
                        max(0.0, -smallest / reference) if reference > 0.0 else 0.0
                    )
                    if link == ORDERING:
                        relative, cell = deficit, deficit_cell
                    if relative > worst[link][0]:
                        worst[link] = (relative, frame, smallest, cell)
                table.add(
                    frame,
                    float(times[frame]),
                    *(float(margins[link].min()) for link in LINKS),
                    deficit,
                    float(indicator[deficit_cell]) if deficit > 0.0 else None,
                    split_error,
                )

            for link in EXACT_LINKS:
                relative, frame, smallest, cell = worst[link]
                report.check(
                    relative <= tol,
                    f"{name}: {link} >= 0",
                    f"reaches {smallest:.4e} at frame {frame}, which is "
                    f"{relative:.2e} of phi / nu there"
                    + (f"; cell {cell}, indicator {indicator[cell]:.3g}" if relative else ""),
                )
            deficit, frame, smallest, cell = worst[ORDERING]
            allowed = ORDERING_WIDTHS * last_width
            report.check(
                deficit <= allowed,
                f"{name}: m_1 >= m_0 to the resolution of the z partition",
                f"worst deficit {deficit:.2e} of m_0 at frame {frame}, against "
                f"{ORDERING_WIDTHS:g} x dz = {allowed:.2e} allowed"
                + (f"; cell {cell}, indicator {indicator[cell]:.3g}" if deficit else ""),
            )
            if drift is not None:
                report.check(
                    drift <= SPLIT_TOLERANCE,
                    f"{name}: the saved phi_sol and phi_gel partition the saved phi",
                    f"worst departure {drift:.4e} of phi",
                )

    ## Private helpers

    def _frame(
        self, run: "Run", name: str, frame: int, nu: float, functionality: float
    ) -> "tuple[dict[str, np.ndarray], float, float, int, float]":
        """One frame's margins, their scale, the worst ordering deficit and its cell,
        and how far the saved split departs from the moments.

        Arithmetic on aligned arrays: every field involved is cell-constant on one
        space, so nothing is interpolated on the way into a check whose subject is
        exact cancellation.
        """

        cells = {
            state: run.field(f"{name}.{state}").cell_values(frame)
            for state in REQUIRED
        }
        phi, c_x = cells["phi"], cells["c_x"]
        u_bar, u_trace = cells["u_bar"], cells["u_trace"]

        m_0 = 0.5 * c_x - u_bar
        m_1 = (2.0 * u_bar - u_trace) / (functionality - 2.0)
        capacity = phi / nu
        margins = {
            "m_0": m_0,
            ORDERING: m_1 - m_0,
            "phi/nu - m_1": capacity - m_1,
            "c_xs": c_x - u_trace,
            "c_xg": u_trace,
        }
        # Per cell and then reduced, not a ratio of reductions: the deficit and
        # the moment it is measured against have to come from the same cell, or a
        # drained corner divides a small deficit by a large moment elsewhere.
        holds = m_0 > 0.0
        ratio = np.zeros_like(m_0)
        ratio[holds] = np.maximum(0.0, (m_0[holds] - m_1[holds]) / m_0[holds])
        cell = int(np.argmax(ratio)) if ratio.size else 0

        # A check on the save path rather than on the scheme: whether what a plot
        # or a saturation sum reads back is this partition. Both halves are
        # compared, so a phi_gel that arrived other than as the difference shows
        # up; nan when the split was not saved, which leaves the check off.
        split = float("nan")
        if run.has(f"{name}.phi_sol") and run.has(f"{name}.phi_gel"):
            phi_sol = run.field(f"{name}.phi_sol").cell_values(frame)
            phi_gel = run.field(f"{name}.phi_gel").cell_values(frame)
            split = max(
                float(np.abs(phi_sol - nu * m_1).max()),
                float(np.abs(phi_sol + phi_gel - phi).max()),
            ) / (float(np.abs(phi).max()) or 1.0)
        return (
            margins,
            float(capacity.max()),
            float(ratio[cell]) if ratio.size else 0.0,
            cell,
            split,
        )

    def _phases(
        self, run: "Run", parameters: "PhaseFieldSystemParameters", name: str | None
    ) -> "tuple[tuple[str, Any], ...]":
        saved = tuple(
            (phase.name, phase)
            for phase in parameters.phase_field_parameters
            if all(run.has(f"{phase.name}.{state}") for state in REQUIRED)
        )
        if not saved:
            raise Unanswerable(
                f"no phase saved all of {', '.join(REQUIRED)}; a polymerizing "
                f"phase needs them in its save list for the chain to be read off"
            )
        if name is None:
            return saved
        matching = tuple(entry for entry in saved if entry[0] == name)
        if not matching:
            raise ValueError(
                f"{name!r} did not save the moments; "
                f"{', '.join(entry[0] for entry in saved)} did"
            )
        return matching

