"""The bar the person who started the run watches.

The log (:mod:`ovpsolver.solver.instrument`) is the record, everything at full
precision; this is the view, a handful of numbers overwritten in place and
formatted to be glanced at rather than grepped. What it shows is health more
than progress: how far along, whether the driver is sub-cycling under the macro
step (the sub-step count and ``dt``), whether Newton is comfortable (its
iterations and ``L/Psi*``), and which part of a step is running, so that a run
that has stopped moving says what it stopped inside of.
"""

from __future__ import annotations

import shutil
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .onsager import ExcessRayleighian

#: The parts of a sub-step worth naming while they run.
RATE = "rate"
STATE = "state"
POST = "post"
SAVE = "save"

#: Narrower than this and the postfix is worth more than the bar, so the width
#: is treated as unmeasured rather than as a very small terminal.
MINIMUM_WIDTH = 40
DEFAULT_WIDTH = 110


class Progress:
    """A tqdm bar over the macro steps, or nothing at all.

    Nothing when disabled or when stderr is not a terminal -- a batch job, a
    test, a pipe -- since a bar in a captured stream is noise in a file someone
    later reads past. Every method is then a no-op, so the solver calls them
    unconditionally.
    """

    def __init__(self, total: int, *, enabled: bool = True) -> None:
        self.bar = None
        if enabled and sys.stderr.isatty():
            # The width is given rather than measured by tqdm: a tty with no
            # window size set (a captured pty, some CI terminals) measures zero
            # columns, which tqdm turns into ``ncols=-1`` and a bar that
            # silently draws the empty string.
            columns = shutil.get_terminal_size(fallback=(0, 0)).columns
            self.bar = tqdm(
                total=total,
                desc="solving",
                unit="step",
                ncols=columns if columns >= MINIMUM_WIDTH else DEFAULT_WIDTH,
            )
        self._substeps = 0
        self._retries = 0
        self._info: dict[str, str] = {}

    ## What the solver reports

    @contextmanager
    def during(self, part: str) -> "Iterator[None]":
        """Name the part of the step being run, for as long as it runs."""

        if self.bar is None:
            yield
            return
        self.bar.set_description_str(f"solving [{part}]")
        try:
            yield
        finally:
            self.bar.set_description_str("solving")

    def substep(self, dt: float) -> None:
        self._substeps += 1
        self._info["dt"] = f"{dt:.1e}"

    def retry(self) -> None:
        self._retries += 1

    def rate_solve(self, iterations: int) -> None:
        """How hard the rate solve was. Not how long: tqdm's own ``s/step`` says
        that."""

        self._info["newton"] = str(iterations)

    def excess_rayleighian(self, measure: "ExcessRayleighian") -> None:
        """``L/Psi*``, the one of the log's three numbers that is a verdict.

        Zero is converged and one a standing start, so ``1e-9`` is a solve that
        finished and ``1e-2`` one that stopped early -- readable at a glance in a
        way ``|F|`` is not.
        """

        ratio = measure.ratio
        if ratio is not None:
            self._info["L/Psi*"] = f"{ratio:.1e}"

    def macro_step(self, reached: float) -> None:
        """One macro step is done: publish the step's numbers and reset them."""

        if self.bar is not None:
            self._info["t"] = f"{reached:.4g}"
            # Only when not one, the ordinary case: a column that always reads
            # the same is a column nobody reads.
            if self._substeps > 1:
                self._info["sub"] = str(self._substeps)
            else:
                self._info.pop("sub", None)
            if self._retries:
                self._info["retry"] = str(self._retries)
            else:
                self._info.pop("retry", None)
            self.bar.set_postfix(self._info, refresh=False)
            self.bar.update(1)
        self._substeps = self._retries = 0

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
