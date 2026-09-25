"""Is the interfacial tail resolved, and what does an unresolved one cost?

Not whether the phase went negative. It cannot: the phase is cell-constant, its
transport row is bounded after the fact by the outflow it actually applied, and a
step inside that bound keeps every cell non-negative outright
(:meth:`~ovpsolver.phase_field_system.transport.Transported.cfl_limit`). A run that
does go negative stops on the spot.

What is left is a resolution question, and it is the one that sets the cost of a
run. A quench of depth ``chi`` puts the minority plateau at ``phi_min``, approached
over a tail of width ``xi = sqrt(kappa phi_min)`` -- the square root because the
regularized logarithm has ``g'' ~ 1/phi`` on the entropic branch. A cell wider than
``xi`` does not resolve that tail, and the comparison

    m := (h / xi)^2 = h^2 / (kappa phi_min)

is what the table reports. The threshold of six it is read against is where a P1
Galerkin tail starts alternating in sign, which is not a mechanism this scheme has;
it is kept because the *scale* is still the right one: ``m`` of order one is a
resolved tail either way.

An unresolved cell costs something specific. On the linearized branch of the
regularized logarithm ``g'' = 1/delta``, which manufactures a large spurious
potential difference across the cell's facets and a ghost velocity with it, and the
step is bounded by the largest velocity anywhere. So a handful of unresolved cells
can set the cost of the whole simulation, and the speed-by-composition table shows
whether they did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...mesh.utils import max_cell_diameter
from .base import (
    FREE_FLUID,
    Diagnostic,
    Unanswerable,
    inclusion_indicator,
)
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: The ``m`` past which a P1 Galerkin tail alternates in sign; see the module
#: docstring for why it is still the scale to read against.
MONOTONE_LIMIT = 6.0

#: Composition bands the report bins by. Chosen to separate the undershoot from
#: the shallow tail that is merely close to it.
BANDS = (
    ("< 0 (undershoot)", -np.inf, 0.0),
    ("0 - 1e-3", 0.0, 1.0e-3),
    ("1e-3 - 5e-3", 1.0e-3, 5.0e-3),
    ("5e-3 - 2e-2", 5.0e-3, 2.0e-2),
    ("2e-2 - 0.1", 2.0e-2, 1.0e-1),
    ("0.1 - 0.3", 1.0e-1, 3.0e-1),
    ("> 0.3", 3.0e-1, np.inf),
)


class Positivity(Diagnostic):
    """Find undershoot in a phase and test its tail against ``(h/xi)^2 < 6``.

    Reports the history of the minimum, the composition below which this mesh
    cannot resolve the tail, and the speed carried by each composition band --
    which says whether the unresolved cells are the ones forcing the timestep down.
    """

    name = "analyze.positivity"
    summary = "locate undershoot and attribute the timestep to it"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        parser.add_argument(
            "--phase",
            help="which phase to examine; default is the one that goes lowest",
        )
        parser.add_argument(
            "--frame",
            type=int,
            default=-1,
            help="the frame to anatomize; default is the last",
        )
        parser.add_argument(
            "--velocity",
            default=None,
            help="the phase's saved velocity, if it is not called 'v'",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        phase: str | None = None,
        frame: int = -1,
        velocity: str | None = None,
        **options: Any,
    ) -> None:
        frame = frame % run.n_frames
        chosen = self._phase(run, parameters, phase)
        values = run.field(f"{chosen.name}.phi").cell_values(frame)
        # The mixture only lives outside the inclusions, so everything is
        # counted there.
        outside = inclusion_indicator(run) > FREE_FLUID

        report.field("phase", f"{chosen.name} (kappa = {float(chosen.kappa):g})")
        report.field("frame", f"{frame} at t = {float(run.times[frame]):.6g}")
        report.field(
            "cells", f"{int(outside.sum())} of {values.size} outside the inclusions"
        )
        report.note()

        self._history(report, run, chosen.name, outside)
        resolvable = self._resolution(report, run, chosen)
        self._speeds(report, run, chosen, frame, values, outside, velocity)

        minimum = float(values[outside].min())
        report.note()
        report.check(
            minimum >= 0.0,
            f"{chosen.name} stays non-negative outside the inclusions",
            f"minimum {minimum:.4e} at frame {frame}",
        )
        # Whether the mesh could have resolved the tail is an attribution, not a
        # verdict: a coarse mesh that never undershot has not failed anything, and
        # saying so would make every quick run red.
        if minimum < 0.0:
            report.note(
                f"the minimum is below phi_crit = {resolvable:.4e}, so this mesh "
                f"cannot resolve that tail monotonically: the undershoot is the "
                f"discretization, not the physics. Refine, or raise kappa."
            )
        elif minimum < resolvable:
            report.note(
                f"nothing is negative, but the minimum {minimum:.4e} is below "
                f"phi_crit = {resolvable:.4e}: this mesh is at the edge of "
                f"resolving its own tail, and a deeper quench would undershoot."
            )

    ## Private helpers

    def _history(
        self, report: "Report", run: "Run", name: str, outside: np.ndarray
    ) -> None:
        """The minimum over the run, which says whether it is growing."""

        times = np.asarray(run.times, dtype=float)
        stride = max(1, run.n_frames // 12)
        frames = sorted({*range(0, run.n_frames, stride), run.n_frames - 1})
        table = report.table(
            Table("frame", "time", "min", "p01", "n < 0", "n < 5e-3")
        )
        for index in frames:
            values = run.field(f"{name}.phi").cell_values(index)[outside]
            table.add(
                index,
                float(times[index]),
                float(values.min()),
                float(np.percentile(values, 1)),
                int((values < 0.0).sum()),
                int((values < 5.0e-3).sum()),
            )

    def _resolution(self, report: "Report", run: "Run", phase: Any) -> float:
        """What this mesh can resolve: the composition ``phi_crit`` below which it
        cannot."""

        h = max_cell_diameter(run.mesh)
        threshold = h * h / (MONOTONE_LIMIT * float(phase.kappa))
        report.note()
        report.note(
            f"P1 monotonicity: phi_crit = h^2 / (6 kappa) = {threshold:.3e}  "
            f"(h = {h:.4g}, kappa = {float(phase.kappa):g})"
        )
        table = report.table(Table("phi", "xi", "h/xi", "m", "verdict"))
        for value in (1e-4, 1e-3, 5e-3, 1e-2, 3e-2, 1e-1):
            xi = float(np.sqrt(float(phase.kappa) * value))
            m = (h / xi) ** 2
            table.add(
                float(value),
                xi,
                h / xi,
                m,
                "monotone" if m < MONOTONE_LIMIT else "OSCILLATORY",
            )
        return threshold

    def _speeds(
        self,
        report: "Report",
        run: "Run",
        phase: Any,
        frame: int,
        values: np.ndarray,
        outside: np.ndarray,
        velocity: str | None,
    ) -> None:
        """Speed binned by composition: does the undershoot carry the timestep?"""

        candidates = [velocity] if velocity else ["v", "v_s"]
        name = next(
            (
                f"{phase.name}.{candidate}"
                for candidate in candidates
                if run.has(f"{phase.name}.{candidate}")
            ),
            None,
        )
        if name is None:
            report.note()
            report.note(
                f"no velocity saved for {phase.name}, so the timestep cannot be "
                f"attributed; add 'v' (or 'v_s') to its save list"
            )
            return

        # On cells, where the phase is, so a band selects the same entries in
        # both. The speed at the centroid is the right comparison anyway: the
        # potential difference a badly resolved cell manufactures acts across that
        # cell's facets, so a band should report the speed the cell carries and
        # not the speed at a vertex it shares with its neighbours.
        speed = np.linalg.norm(
            run.field(name).cell_values(frame).reshape(values.size, -1), axis=1
        )
        table = report.table(
            Table(f"{phase.name}.phi band", "n", "median |v|", "p99 |v|", "max |v|")
        )
        for label, low, high in BANDS:
            selected = outside & (values >= low) & (values < high)
            count = int(selected.sum())
            if count == 0:
                table.add(label, 0, None, None, None)
                continue
            band = speed[selected]
            table.add(
                label,
                count,
                float(np.median(band)),
                float(np.percentile(band, 99)),
                float(band.max()),
            )

    def _phase(
        self, run: "Run", parameters: "PhaseFieldSystemParameters", name: str | None
    ) -> Any:
        """The named phase, or the one whose fraction goes lowest."""

        phases = [
            phase
            for phase in parameters.phase_field_parameters
            if run.has(f"{phase.name}.phi")
        ]
        if not phases:
            raise Unanswerable(
                "no phase saved 'phi'; there is nothing to check for positivity"
            )
        if name is not None:
            for phase in phases:
                if phase.name == name:
                    return phase
            raise ValueError(
                f"{name!r} did not save 'phi'; this run has "
                f"{', '.join(phase.name for phase in phases)}"
            )
        return min(
            phases, key=lambda phase: float(run.field(f"{phase.name}.phi").values.min())
        )
