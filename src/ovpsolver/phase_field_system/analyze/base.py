"""What every diagnostic has in common: a spec, the run it names, and a verdict.

The reading is the same every time -- parse the spec against the mixture's own
parameters, open the output directory it names -- so it is here, and a subclass
writes only :meth:`Diagnostic.measure`. That is handed the parameters the run was
built from and a :class:`~ovpsolver.fem.save.Run` whose fields come back as
``dolfinx`` functions on the run's own mesh, so an integral is a form assembled
against the solve's own measure and a value at a point is an evaluation.

Every diagnostic returns an exit code, non-zero if any check failed, so a spec can
be run and then interrogated from a shell script or a test with no glue between.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from ...fem.save import Run
from ..entry import SpecEntryPoint
from ..run import read
from .report import Report

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from pathlib import Path

    from dolfinx import fem

    from ..parameters import PhaseFieldSystemParameters
    from ..system import PhaseFieldSystem

#: The name a run saves the diffuse domain's quadrature ``chi_eps`` under.
INDICATOR = "diffuse_domain.indicator"

#: A cell is in the free fluid when the indicator at its centroid is past this.
FREE_FLUID = 0.5

#: How a phase's row splits its flux, as the saved names of each piece's velocity
#: and of the part of the phase it carries: the whole phase on one velocity, or a
#: polymerizing phase's sol and gel halves on theirs.
#:
#: The first piece of a split is the injecting one. A phase is fed as free
#: monomer, so the sol carries whatever the spec puts in at the surfaces and the
#: network carries none of it, which is why
#: :meth:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseField.gel_flux_carrying`
#: takes no source at all.
PIECES = (
    (("v", "phi"),),
    (("v_s", "phi_sol"), ("v_g", "phi_gel")),
)


class Unanswerable(Exception):
    """The run did not save what the diagnostic needs to answer its question.

    Reported as neither a pass nor a failure: a spec that never saved the gel
    strain moment has no moment rather than a bad one, and an integration test
    running every diagnostic over every spec should not go red because of it.
    """


class Diagnostic(SpecEntryPoint):
    """One measurement of a saved run, reported as a table and a verdict."""

    #: The table is printed and the CSV closed by the time this returns, and
    #: the exit hooks dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        parser.add_argument(
            "--save",
            metavar="CSV",
            help="also write the table to this path, as CSV",
        )

    def __call__(
        self,
        system_class: "type[PhaseFieldSystem]",
        *,
        spec: "Path",
        save: str | None = None,
        **options: Any,
    ) -> int:
        parameters = read(system_class.Parameters, spec)
        run = Run.of(parameters)
        report = Report(f"{self.name}: {run.path}")
        report.field("spec", spec)
        report.field("frames", f"{run.n_frames} written")
        report.note()
        try:
            self.measure(report, run, parameters, system_class=system_class, **options)
        except Unanswerable as reason:
            # Said once and plainly, because the fix is a line in the spec and a
            # traceback would not say so.
            report.note(f"not applicable: {reason}")
        return report.emit(save)

    ## Public

    @abstractmethod
    def measure(
        self,
        report: Report,
        run: Run,
        parameters: "PhaseFieldSystemParameters",
        *,
        system_class: "type[PhaseFieldSystem]",
        **options: Any,
    ) -> None:
        """Ask the question and write the answer into ``report``."""


def flux_pieces(run: Run, name: str) -> "tuple[tuple[str, str], ...] | None":
    """The first split of :data:`PIECES` this phase saved every name of, if any."""

    if not run.has(f"{name}.phi"):
        return None
    for pieces in PIECES:
        if all(
            run.has(f"{name}.{velocity}") and run.has(f"{name}.{carried}")
            for velocity, carried in pieces
        ):
            return pieces
    return None


def frames_of(
    run: Run, frames: "list[int] | None", *, stride: int = 1
) -> tuple[int, ...]:
    """The frames a diagnostic walks: the ones named, or every ``stride``-th.

    A frame out of range is an error rather than a silent clamp: it asks about a
    run other than the one the caller meant.
    """

    if frames:
        for frame in frames:
            if not 0 <= frame < run.n_frames:
                raise ValueError(
                    f"frame {frame} is outside this run, which wrote "
                    f"{run.n_frames}"
                )
        return tuple(frames)
    return tuple(range(0, run.n_frames, max(1, stride)))


def declare_frames(parser: "ArgumentParser") -> None:
    """The ``--frames``/``--stride`` pair, for diagnostics that walk a run."""

    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="only these frames, by index; default is all of them",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="walk every Nth frame instead, for a long run",
    )


def inclusion_indicator(run: Run) -> np.ndarray:
    """The free-region indicator per cell, for a mask over a cell-constant field.

    Computed from the run's own spec rather than read back from it: the geometry
    is analytic and static, so evaluating it is cheap and exact, while the saved
    quadrature indicator would have to be projected to be used per cell, which
    overshoots at a step.

    At cell centroids, the value a cell-constant mask has, which lines the mask
    up with the residuals it selects from. Not the cell average the solve
    integrates -- a mask is thresholded far from its cutoff, where the two agree.
    """

    return run.parameters.solver.diffuse_domain.chi_eps_on(run.cell_space())


def quadrature_indicator(run: Run) -> "fem.Function | None":
    """The saved quadrature ``chi_eps``, or ``None`` if the run did not save it.

    The weight an integral has to carry to match one the transport rows took. What
    a row conserves is its own accumulation term, ``sum_K phi_K int_K chi_eps``,
    with this field on this rule; any other weight -- the analytic indicator at
    centroids, a nodal copy -- measures a quantity nothing conserves, which reads
    as a drift growing as the phases move through the band where the two differ.
    """

    if not run.has(INDICATOR):
        return None
    return run.field(INDICATOR).sample(0)


__all__ = [
    "Diagnostic",
    "Unanswerable",
    "declare_frames",
    "frames_of",
    "inclusion_indicator",
]
