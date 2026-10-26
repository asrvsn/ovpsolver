"""Reading a spec against a mixture, and advancing what comes out.

The functions behind :meth:`~ovpsolver.phase_field_system.system.PhaseFieldSystem.run`,
kept apart from the class because reading a document, opening a log and opening
a saver are not statements about the physics. :func:`read` is shared with the
other entry points, since drawing or measuring a run starts by finding out what
the run was.

A document is two blocks, ``solver`` and ``system``, whatever mixture it is for.
``solver`` is always read against the same class; ``system`` against the mixture
named on the command line, so the word that picks the class is in the invocation
and not in the file. There is no builder and no registry: a mixture class names
its parameters class, that class names the class of every block under it, and
reading is one recursive walk that checks values against declarations::

    class MyMixtureParameters(PhaseFieldSystemParameters):
        phase_fields = ParametersList(MyPhaseFieldParameters())

    class MyMixture(PhaseFieldSystem, ParametricSaveable[MyMixtureParameters]):
        def make_phase_fields(self): ...

    MyMixture.run("mine.yaml")

Nor is there any migration. A spec is read strictly, so a document written
against an older shape of the schema is rejected with the name of what it got
wrong rather than translated. An archived run therefore reads back only under the
version that wrote it, which its ``meta.json`` names, and in exchange a spec in
hand means exactly what it says.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from mpi4py import MPI

from ..fem.jit import release_abandoned_forms
from ..fem.save import META_FILE, SPEC_FILE, Saver, describe_version, source_version
from ..solver.instrument import configure, log_failures
from ..solver.progress import SAVE
from .entry import EntryPoint
from .parameters import SpecParameters
from .resume import Resume
from . import sweep as sweeps

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

    from ..solver.parameters import SaveParameters
    from .parameters import PhaseFieldSystemParameters
    from .system import PhaseFieldSystem


def load(path: str | Path) -> dict[str, Any]:
    """Read one YAML document, refusing anything that is not a mapping."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"{source} must contain a mapping at the top level")
    return document


def read(
    prototype: "type[PhaseFieldSystemParameters] | PhaseFieldSystemParameters",
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> "PhaseFieldSystemParameters":
    """Read the document at ``path``, its ``system`` block against ``prototype``.

    Returns the ``system`` block, which is what a mixture is built from; the
    ``solver`` block beside it is reached through
    :attr:`~ovpsolver.phase_field_system.parameters.PhaseFieldSystemParameters.solver`.
    Strict, so the document has to answer every declaration: a spec is a
    complete record of the run it produced, and nothing here fills a silence.

    The paths the document names are relative to it, or to ``relative_to``: a
    sweep writes its variants elsewhere, each meaning its paths as the spec it
    varies does (:mod:`~ovpsolver.phase_field_system.sweep`).
    """

    source = Path(path).expanduser().resolve()
    blank = prototype() if isinstance(prototype, type) else prototype
    spec = SpecParameters(system=blank).read(load(source), source.name, strict=True)
    spec.resolve_against(
        source if relative_to is None else Path(relative_to).expanduser().resolve()
    )
    parameters = spec.system
    parameters.source = source
    return parameters


def check_serial() -> None:
    """Stop before anything is built if this was launched on several ranks.

    Distributed parallelism is not a feature of this package. Several pieces
    assume one rank -- the saver indexes arrays by local dof, the diffuse
    geometry is built from the whole inclusion list on every rank, the reader
    maps frames on ``COMM_SELF`` -- and a run under ``mpirun`` would not fail,
    it would produce a directory in which each rank had overwritten the last.
    """

    if MPI.COMM_WORLD.size > 1:
        raise NotImplementedError(
            f"this was launched on {MPI.COMM_WORLD.size} MPI ranks, and nothing "
            f"here is distributed: the saved frames are indexed by local dof, so "
            f"the ranks would overwrite each other's output rather than fail. "
            f"Run it on one rank."
        )


def claim(
    save: "SaveParameters",
    *,
    overwrite: bool = False,
    resuming: bool = False,
    check_only: bool = False,
) -> None:
    """Take the output directory, refusing one that already holds something.

    A second run into the same place lands on top of the first rather than
    replacing it: frames go into arrays named after the fields, so the fields
    the two share are overwritten, those only the first saved are left, and
    ``meta.json`` describes arrays that were never written together. The result
    reads back without complaint and is not a run, and there is no cheap way to
    notice afterwards. Checked here rather than in the saver, because by then the
    log file has been opened and the directory is no longer as it was found.

    ``overwrite`` clears the directory first, but only one that looks like a
    previous run; anything else is somebody's own. ``resuming`` inverts the
    question: a resume continues the run that is there, so an empty directory is
    the failure, and nothing is cleared. ``check_only`` asks every question and
    clears nothing, so a queue can be checked whole before its first run starts.
    """

    directory = save.directory
    if resuming:
        if not directory.is_dir() or not any(directory.iterdir()):
            raise FileNotFoundError(
                f"--resume continues the run in solver.save.path, and there is "
                f"nothing in {directory} to continue"
            )
        return
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError(f"solver.save.path is not a directory: {directory}")

    entries = sorted(entry.name for entry in directory.iterdir())
    if not entries:
        return
    if not overwrite:
        raise FileExistsError(
            f"solver.save.path already holds {len(entries)} entries: "
            f"{directory}. A run cannot be started into it without merging "
            f"into whatever is there. Move it aside, point the spec somewhere "
            f"else, or pass --overwrite to clear it first."
        )

    # The log counts: it is opened before the mesh is built or a form compiled,
    # so it is the whole of what a run that died before its first frame leaves
    # behind, and without it such a directory could never be cleared.
    written_by_a_run = {META_FILE, SPEC_FILE}
    if save.logfile:
        written_by_a_run.add(save.logfile)
    if not (written_by_a_run.intersection(entries) or any(
        name.endswith((".npy", ".xdmf")) for name in entries
    )):
        raise FileExistsError(
            f"{directory} is not empty and does not look like a saved run, so "
            f"--overwrite will not clear it: {', '.join(entries[:8])}. Empty it "
            f"by hand if that is really what is wanted."
        )
    if not check_only:
        shutil.rmtree(directory)


def run(
    system_class: "type[PhaseFieldSystem]",
    path: str | Path,
    *,
    n_steps: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    relative_to: str | Path | None = None,
) -> float:
    """Build the mixture the document describes and step it; returns the time reached.

    ``resume`` continues the run already in the output directory from the
    restart point it last committed; a run keeps one, overwritten every
    ``resumable_every`` macro steps. Frames after it are rewritten and
    everything up to it is kept. Everything else happens as for a fresh run --
    the same mesh, read from the archive it is already in, the same forms, the
    same warm start -- except where the first step starts from. See
    :mod:`~ovpsolver.phase_field_system.resume`.

    ``relative_to`` is what the document's paths are relative to, if not the
    document itself; see :func:`read`.
    """

    check_serial()
    if resume and overwrite:
        raise ValueError(
            "--resume continues a run and --overwrite deletes one; they cannot "
            "both be what was meant"
        )
    parameters = read(system_class.Parameters, path, relative_to=relative_to)
    save = parameters.solver.save
    claim(save, overwrite=overwrite, resuming=resume)
    configure(
        save.log_path,
        comm=parameters.solver.mesh_comm,
        debug=save.debug,
        append=resume,
    )

    # Everything from here is in the run's log, a failure included. One while
    # building the mesh or compiling the forms ends a run as surely as one in a
    # step, and a sweep that goes on past it leaves the log as its only record.
    # Outermost, so a saver or a progress bar that fails on the way out fails
    # after the loop has finished with it, and that is the failure a reader of
    # the archive needs told about.
    with log_failures():
        return _build_and_step(parameters, n_steps=n_steps, resume=resume)


def _build_and_step(
    parameters: "PhaseFieldSystemParameters",
    *,
    n_steps: int | None,
    resume: bool,
) -> float:
    """What :func:`run` does once the run's log is open."""

    # Inline: the solver module imports this package's transport at import time.
    from ..solver import PhaseFieldSolver

    # Before anything compiles, and after the log exists to say so. A run
    # interrupted during compilation leaves claims in the FFCx cache that stop
    # every later run until they are cleared.
    for released in release_abandoned_forms():
        logger.info("released abandoned form cache entry %s", released)
    # Also before anything compiles, since the source is read as it is at the
    # first call, and a long compile is when someone edits it.
    code = source_version()
    logger.info("code: %s", describe_version(code))

    system = parameters.build()

    # Before the solver compiles anything and before the saver opens the arrays,
    # so a resume that cannot be honoured costs nothing and changes nothing.
    plan = None
    if resume:
        plan = Resume.last(system)
        logger.info(
            "RESUMED from the restart point in slot %s: %d step(s) taken, "
            "t=%.6e; its %d frame(s) are kept and later ones rewritten",
            plan.slot,
            plan.next_step,
            plan.time,
            plan.frame,
        )
        # Resuming after a change to the code is the point of resuming, so a
        # change is announced rather than refused.
        written = (plan.code or {}).get("digest")
        if written is not None and written == code["digest"]:
            logger.info("code unchanged since the frames kept were written")
        else:
            logger.warning(
                "code changed since the frames kept were written: they were "
                "written by %s",
                describe_version(plan.code),
            )

    steps = parameters.solver.timestepping.n_steps if n_steps is None else n_steps
    first = 0 if plan is None else plan.next_step
    if first >= steps:
        raise ValueError(
            f"the restart point is at step {first}, and this spec asks for "
            f"{steps}, so there is nothing left to take"
        )

    solver = PhaseFieldSolver(
        system, parameters.solver.timestepping, resume=plan
    )
    solver.instrument.header()

    reached = 0.0 if plan is None else plan.time
    with (
        Saver.of(
            system,
            n_steps=steps,
            resume_frame=None if plan is None else plan.frame,
        ) as saver,
        solver.watching(steps - first) as bar,
    ):
        solver.instrument.saving(saver)
        if plan is None:
            # Frame zero is the initial condition, and every later frame is read
            # as a change from it. A resumed run starts from a frame already
            # written.
            saver.frame(reached)
        for index in range(first, steps):
            advanced = solver.step()
            reached += advanced
            solver.instrument.macro_step(index, advanced, reached)
            with bar.during(SAVE):
                saver.frame(reached)
                saver.resume_point(reached)
            bar.macro_step(reached)
        # The cadence may have last fired several steps back; keep the work this
        # run actually finished.
        with bar.during(SAVE):
            saver.resume_point(reached, force=True)
    solver.instrument.finished(steps - first, reached)
    return reached


# An EntryPoint rather than a SpecEntryPoint, whose single positional this
# replaces: only the alternative that produces runs has a use for several specs.
class RunSpec(EntryPoint):
    """Build the mixture each spec describes and step it, writing frames as it goes.

    Everything a run needs is in its document: the mesh, the material, how long a
    step is, and where the output goes. The options say how many steps to take,
    so that a long spec can be checked in a minute without editing it, whether an
    existing output directory may be cleared, and whether the run already in it
    is continued.

    Several specs may be named, and are run one after another in the order given,
    each writing to the directory its own spec names, in one process rather than
    re-importing everything per run. A sweep runs each spec many times over,
    varying values at addresses into its system block:

        run pips.yaml \\
            --sweep 'couplings.flory_huggins.(water, protein).chi_01=[1, 2, 3]' \\
            --sweep-zip 'phase_fields.sol[0].kappa=[0.5e-3, 1.0e-3]' \\
                        'phase_fields.polymerizing[0].kappa=[0.5e-3, 1.0e-3]'

    is six runs: each chi_01 against each pair of kappas, the two stepping
    together. Each saves to the spec's solver.save.path with -label=value
    appended per address, as pips-chi_01=2-sol.0.kappa=0.0005-..., and archives
    the spec with its values in place. A change no address reaches is still a
    spec of its own.
    """

    name = "run"
    summary = "step a mixture through its spec, several in sequence, or a sweep"

    def declare(self, parser: "ArgumentParser") -> None:
        parser.add_argument(
            "specs",
            type=Path,
            nargs="+",
            metavar="spec",
            help="path to an experiment YAML; several run in the order given",
        )
        parser.add_argument(
            "--n-steps",
            type=int,
            default=None,
            dest="n_steps",
            help="override the macro step count, for a short check of a long run",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="clear the output directory first, instead of refusing to run",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="continue the run already in the output directory from the "
            "restart point it last committed. Frames after it are rewritten; "
            "everything up to it is kept",
        )
        sweeps.add_options(parser)

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        groups = arguments.pop("sweep")
        return {**arguments, "sweep": sweeps.Sweep.from_groups(groups) if groups else None}

    def __call__(
        self,
        system_class: "type[PhaseFieldSystem]",
        *,
        specs: "list[Path]",
        n_steps: int | None = None,
        overwrite: bool = False,
        resume: bool = False,
        sweep: "sweeps.Sweep | None" = None,
    ) -> "list[float | None]":
        """Each run in turn; returns what each reached, in order.

        Every run is read, and its output directory checked, before the first
        starts. A queue is started and left, and a typo in its last spec, two
        runs writing to one directory, or a directory left from an earlier
        attempt should stop it now rather than hours in.

        A queue of specs stops at the first run that fails: its specs are
        variations written by hand, so the second failure is nearly always the
        first one again. A sweep goes on to the next run, since where a sweep
        fails depends on the values -- the deepest quench diverging while the
        shallow ones do not -- and the runs that would succeed are the point of
        it. Each failure is in its run's log, and a :class:`~.sweep.SweepFailed` naming
        them is raised once the last run is over. An interrupt stops either.

        ``--resume`` continues one run from its own restart point, so it is
        refused for more than one. A run of a sweep is resumed from the spec
        archived in its directory.
        """

        scratch = (
            tempfile.TemporaryDirectory(prefix="ovpsolver-sweep-")
            if sweep is not None
            else nullcontext()
        )
        with scratch as directory:
            runs = (
                [sweeps.Variant(Path(one), Path(one), ()) for one in specs]
                if sweep is None
                else [
                    variant
                    for one in specs
                    for variant in sweep.variants(one, Path(directory))
                ]
            )
            if resume and len(runs) > 1:
                raise ValueError(
                    "--resume continues one run from its own restart point; name a "
                    "single spec, or resume a run of a sweep from the spec archived "
                    "in its directory"
                )
            directories = self.check(system_class, runs, overwrite=overwrite, resume=resume)
            if len(runs) > 1:
                print(
                    f"{len(runs)} runs, in this order:",
                    *(f"  {os.path.relpath(path)}" for path in directories),
                    sep="\n",
                    flush=True,
                )
            options = {"n_steps": n_steps, "overwrite": overwrite, "resume": resume}
            if sweep is None:
                return [
                    system_class.run(run.path, relative_to=run.origin, **options)
                    for run in runs
                ]

            reached: list[float | None] = []
            failures: list[tuple[Path, str]] = []
            for number, (run, output) in enumerate(zip(runs, directories), start=1):
                try:
                    reached.append(
                        system_class.run(run.path, relative_to=run.origin, **options)
                    )
                except Exception as error:
                    lines = str(error).splitlines()
                    why = type(error).__name__ + (f": {lines[0]}" if lines else "")
                    reached.append(None)
                    failures.append((output, why))
                    print(
                        f"run {number} of {len(runs)} failed, going on to the next: {why}",
                        file=sys.stderr,
                        flush=True,
                    )
                    # The traceback holds the failed run's mesh, forms and
                    # matrices through its frames; let them go before the next
                    # run builds its own.
                    del error
                    gc.collect()
            if failures:
                raise sweeps.SweepFailed(reached, failures)
            return reached

    @staticmethod
    def check(
        system_class: "type[PhaseFieldSystem]",
        runs: "Sequence[sweeps.Variant]",
        *,
        overwrite: bool,
        resume: bool,
    ) -> list[Path]:
        """Read every run and check its directory, clearing nothing; returns the directories."""

        claimed: dict[Path, sweeps.Variant] = {}
        for run in runs:
            save = read(system_class.Parameters, run.path, relative_to=run.origin).solver.save
            directory = Path(os.path.abspath(save.directory))
            if directory in claimed:
                raise ValueError(
                    f"two runs would write to {directory}: "
                    f"{_named(claimed[directory])} and {_named(run)}"
                )
            claimed[directory] = run
            claim(save, overwrite=overwrite, resuming=resume, check_only=True)
        return list(claimed)


def _named(run: sweeps.Variant) -> str:
    return f"{run.origin} with {run.describe()}" if run.settings else str(run.origin)
