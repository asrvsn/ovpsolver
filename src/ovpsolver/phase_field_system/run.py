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

import logging
import shutil
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from argparse import ArgumentParser

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
) -> "PhaseFieldSystemParameters":
    """Read the document at ``path``, its ``system`` block against ``prototype``.

    Returns the ``system`` block, which is what a mixture is built from; the
    ``solver`` block beside it is reached through
    :attr:`~ovpsolver.phase_field_system.parameters.PhaseFieldSystemParameters.solver`.
    Strict, so the document has to answer every declaration: a spec is a
    complete record of the run it produced, and nothing here fills a silence.
    """

    source = Path(path).expanduser().resolve()
    blank = prototype() if isinstance(prototype, type) else prototype
    spec = SpecParameters(system=blank).read(load(source), source.name, strict=True)
    spec.resolve_against(source)
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
    save: "SaveParameters", *, overwrite: bool = False, resuming: bool = False
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
    the failure, and nothing is cleared.
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
    shutil.rmtree(directory)


def run(
    system_class: "type[PhaseFieldSystem]",
    path: str | Path,
    *,
    n_steps: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> float:
    """Build the mixture the document describes and step it; returns the time reached.

    ``resume`` continues the run already in the output directory from the
    restart point it last committed; a run keeps one, overwritten every
    ``resumable_every`` macro steps. Frames after it are rewritten and
    everything up to it is kept. Everything else happens as for a fresh run --
    the same mesh, read from the archive it is already in, the same forms, the
    same warm start -- except where the first step starts from. See
    :mod:`~ovpsolver.phase_field_system.resume`.
    """

    # Inline: the solver module imports this package's transport at import time.
    from ..solver import PhaseFieldSolver

    check_serial()
    if resume and overwrite:
        raise ValueError(
            "--resume continues a run and --overwrite deletes one; they cannot "
            "both be what was meant"
        )
    parameters = read(system_class.Parameters, path)
    save = parameters.solver.save
    claim(save, overwrite=overwrite, resuming=resume)
    configure(
        save.log_path,
        comm=parameters.solver.mesh_comm,
        debug=save.debug,
        append=resume,
    )

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
        # First, so it is the last to be left: a saver or a progress bar that
        # fails on the way out fails after the loop has finished with it, and
        # that is the failure a reader of the archive needs told about.
        log_failures(),
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
    each writing to the directory its own spec names. A sweep is a set of
    documents rather than a flag, since what varies between two runs is not
    always one number, and a queue of them runs in one process rather than
    re-importing everything per run.
    """

    name = "run"
    summary = "step a mixture through its spec, or several in sequence"

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

    def __call__(
        self,
        system_class: "type[PhaseFieldSystem]",
        *,
        specs: "list[Path]",
        n_steps: int | None = None,
        overwrite: bool = False,
        resume: bool = False,
    ) -> list[float]:
        """Each spec in turn, stopping at the first that fails.

        Stopping rather than reporting at the end: the specs in a queue are
        usually variations on one another, so the second failure is nearly
        always the first one again. Returns what each run reached, in order.

        ``--resume`` continues one run from its own restart point, so it is
        refused for a queue.
        """

        if resume and len(specs) > 1:
            raise ValueError(
                "--resume continues one run from its own restart point; name a "
                "single spec"
            )
        return [
            system_class.run(
                one, n_steps=n_steps, overwrite=overwrite, resume=resume
            )
            for one in specs
        ]
