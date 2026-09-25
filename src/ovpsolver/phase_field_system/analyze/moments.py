"""Is the gel strain moment still a strain moment?

``M`` is a second moment of the network's chain end-to-end vectors, so it is
symmetric and positive semi-definite by construction, and its trace is the mean
squared stretch. A negative eigenvalue is not a soft network: it is a state the
tensor cannot represent, and it means the moment row has been integrated past
where its own transport stayed monotone.

The spectrum is the diagnostic rather than the components, because the components
depend on how the tensor was stored and the eigenvalues do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Diagnostic, Unanswerable, declare_frames, frames_of
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Field, Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: The field a polymerizing phase saves its moment under.
MOMENT = "gel_strain_moment"

#: Default for ``--tol``: how negative the smallest eigenvalue may be, as a
#: fraction of the largest at the same frame. Relative because the moment starts
#: at exactly zero and grows with the crosslinking, so an absolute floor would
#: pass everything early and nothing late. A violation at this level is
#: arithmetic; one far above it is the Lie transport of the tensor having left the
#: cone, which is a statement about the step and not about noise.
NEGATIVE_TOLERANCE = 1.0e-6


class Moments(Diagnostic):
    """Per-frame extreme eigenvalues of each gel's strain moment tensor.

    Fails if an eigenvalue goes negative by more than ``--tol`` of the largest,
    which is the one thing the tensor is not allowed to do.
    """

    name = "analyze.moments"
    summary = "check the gel strain moment stays positive semi-definite"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--phase",
            help="which polymerizing phase; default is every one that saved a moment",
        )
        parser.add_argument(
            "--tol",
            type=float,
            default=NEGATIVE_TOLERANCE,
            help=(
                "how negative an eigenvalue may go, as a fraction of the "
                "largest at the same frame"
            ),
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        phase: str | None = None,
        tol: float = NEGATIVE_TOLERANCE,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        names = self._phases(run, parameters, phase)
        report.field("phases", ", ".join(names))
        report.note()

        times = np.asarray(run.times, dtype=float)
        walk = frames_of(run, frames, stride=stride)
        for name in names:
            field = run.field(f"{name}.{MOMENT}")
            report.field(f"{name}.{MOMENT}", field.element.describe())
            table = report.table(
                Table(
                    "frame",
                    "time",
                    "min",
                    "at cell",
                    "max",
                    "at cell",
                    "negative fraction",
                    "min trace",
                )
            )
            worst, worst_frame, worst_absolute = 0.0, 0, 0.0
            for frame in walk:
                spectrum = _eigenvalues(field, frame)
                low = int(np.unravel_index(np.argmin(spectrum), spectrum.shape)[0])
                high = int(np.unravel_index(np.argmax(spectrum), spectrum.shape)[0])
                smallest, largest = float(spectrum.min()), float(spectrum.max())
                scale = max(abs(largest), abs(smallest))
                # How far below zero the smallest eigenvalue reaches, against the
                # size of the tensor at the same frame. Clamped at zero, so the
                # column is what the check is and a run that never goes negative
                # reads as zeros rather than as small numbers to squint at.
                relative = max(0.0, 0.0 if scale == 0.0 else -smallest / scale)
                if relative > worst:
                    worst, worst_frame, worst_absolute = relative, frame, smallest
                table.add(
                    frame,
                    float(times[frame]),
                    smallest,
                    low,
                    largest,
                    high,
                    relative,
                    float(spectrum.sum(axis=1).min()),
                )
            report.check(
                worst <= tol,
                f"{name}'s moment stays positive semi-definite",
                f"most negative eigenvalue {worst_absolute:.4e}, which is "
                f"{worst:.2e} of the largest at frame {worst_frame}",
            )

    ## Private helpers

    def _phases(
        self, run: "Run", parameters: "PhaseFieldSystemParameters", name: str | None
    ) -> tuple[str, ...]:
        saved = tuple(
            phase.name
            for phase in parameters.phase_field_parameters
            if run.has(f"{phase.name}.{MOMENT}")
        )
        if not saved:
            raise Unanswerable(
                f"no phase saved {MOMENT!r}; add it to a polymerizing phase's "
                f"save list to measure its spectrum"
            )
        if name is None:
            return saved
        if name not in saved:
            raise ValueError(
                f"{name!r} did not save {MOMENT!r}; {', '.join(saved)} did"
            )
        return (name,)


def _eigenvalues(field: "Field", frame: int) -> np.ndarray:
    """``(cell, eigenvalue)`` for one frame of a tensor field.

    Read on the full tensor space of the same value shape, which lays out a
    symmetric element's packed components, rather than reshaped in place: a
    symmetric 2x2 stores three dofs, which do not reshape to the value shape at
    all. Symmetrized anyway before the decomposition, because nothing in the
    discrete step enforces the symmetry the continuous equation has, and
    ``eigvalsh`` reads one triangle and would quietly answer for a tensor nobody
    computed.
    """

    shape = field.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"{field.name} is {field.element.describe()}; a strain moment is a "
            f"square tensor"
        )
    matrices = field.cell_values(frame)
    return np.linalg.eigvalsh(0.5 * (matrices + np.swapaxes(matrices, -1, -2)))
