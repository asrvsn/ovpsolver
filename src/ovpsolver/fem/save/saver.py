"""Writing a run's frames, and everything needed to read them back.

The saver is given a mapping of name to value and declaration and never asks
where it came from: the mixture decides what is worth saving, because only it
knows, and the saver decides how. Adding a diagnostic to a phase is a method and
a line of YAML, and nothing here changes.

It writes the mesh once, then each field's dof vector once per frame -- or once
in total, for a field its owner declares static, such as the geometry, built from
inclusions that do not move. Nothing is resampled, averaged or projected on the
way out: those decisions depend on what the data will be used for, which the
saver cannot know, and made at save time they cannot be undone. A reader rebuilds
the spaces on the saved mesh instead: :class:`~ovpsolver.fem.save.read.Run` hands
back :class:`dolfinx.fem.Function` objects, so an analysis of a saved run and one
written inside the solve are the same code.

The directory is self-describing: the frames, the mesh, the spec that produced
them and a ``meta.json`` naming all of it, so a run can be moved, published or
read years later without the code that wrote it -- and names that code too
(:func:`source_version`), for when the run has to be reproduced rather than read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any

import numpy as np

from ..saveable import Saveable, qualified
from .field import Series, series_for

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...phase_field_system import PhaseFieldSystem
    from ...solver.parameters import SaveParameters, SolverParameters
    from ..types import Function

#: The fixed names in a run directory. Fixed rather than recorded, because a
#: reader that cannot find ``meta.json`` cannot be told where ``meta.json`` is.
META_FILE = "meta.json"
SPEC_FILE = "spec.yaml"
TIMES_FILE = "times.npy"
STEPS_FILE = "steps.npy"

#: Where a run archives its mesh: a byte copy of the file the run read it from,
#: since a mesh written out and read back is renumbered and every saved dof
#: vector is indexed by the run's own numbering (see
#: :attr:`~ovpsolver.solver.parameters.SolverParameters.mesh_source_path`).
MESH_FILE = "mesh.xdmf"

#: The geometry file, archived under this name whatever it was called.
GEOMETRY_FILE = "geometry.py"

#: Where the restart points live, two slots deep. A directory rather than a
#: suffix beside the frames, so the set can be dropped or excluded from a sync
#: in one move.
RESUME_DIR = "resume"

#: The source the running code is built from: the package and the C++ beside it.
#: Found from this file rather than the working directory, so a run launched from
#: anywhere records the code it ran.
SOURCE_DIR = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


class Saver:
    """Append every saved field's dofs, one frame at a time.

    Serial only, and it says so rather than writing something wrong: the arrays
    are indexed by local dof, so under MPI each rank would write a different
    partition's values into the same slots. Making it parallel means gathering
    to rank zero, which is a real piece of work and not one this needs yet.
    """

    def __init__(
        self,
        fields: "Mapping[str, tuple[Any, Saveable]]",
        parameters: "SaveParameters",
        solver_parameters: "SolverParameters",
        *,
        n_steps: int,
        resumable_every: int = 1,
        resume_values: "Mapping[str, Function] | None" = None,
        resume_frame: int | None = None,
        source: Path | None = None,
        geometry: Path | None = None,
        geometry_assets: "Sequence[Path]" = (),
    ) -> None:
        mesh = solver_parameters.dolfinx_mesh
        if mesh.comm.size != 1:
            raise NotImplementedError(
                "saving gathers nothing yet, so a parallel run would write each "
                "rank's partition over the last; run it on one rank or add the "
                "gather"
            )

        self.solver_parameters = solver_parameters
        self.geometry = geometry
        self.geometry_assets = tuple(geometry_assets)
        self.path = parameters.directory
        self.every_k = parameters.every_k
        self.dtype = np.dtype(parameters.dtype)
        #: The frames a resumed run keeps, so every array is reopened in place
        #: rather than created. ``None`` for a run that starts from nothing,
        #: which is the ordinary case.
        self._resume_frame = resume_frame
        #: What a restart point holds, by the name the archive files it under.
        #: Not the save lists: what a run needs to be put back as it was is a
        #: different set from what a reader wants to look at.
        self._resume_values = dict(resume_values or {})
        self.resumable_every = int(resumable_every)
        #: What this directory already records, read back rather than passed in:
        #: this run is about to rewrite that file, and a restart point it did not
        #: carry forward would be erased on the very first frame, before a
        #: replacement exists.
        committed = {} if resume_frame is None else self._committed_metadata()
        #: The committed restart point's record, and the slot holding it; the
        #: next is written into the other.
        self._resume_record: dict | None = committed.get("resume")
        self._resume_slot: str | None = (
            None if self._resume_record is None else self._resume_record.get("slot")
        )
        #: Which frame to write next, and the saver's own step counter, which
        #: counts the initial condition as step zero and so runs one ahead of
        #: the macro steps taken. A resumed run takes both exactly as the
        #: restart point recorded them, so the two cadences carry on as if the
        #: run had never stopped.
        self._frame = 0 if resume_frame is None else resume_frame
        self._step = 0 if resume_frame is None else int(self._resume_record["step"])
        #: The code that wrote each stretch of frames, one entry per process
        #: from the frame it started at. A resumed run keeps the entries for the
        #: frames it keeps, since resuming after a change to the code is what
        #: resuming is for.
        self._code = [
            *(
                entry
                for entry in committed.get("code", ())
                if entry["first_frame"] < self._frame
            ),
            {"first_frame": self._frame, **source_version()},
        ]
        self._closed = False

        self.path.mkdir(parents=True, exist_ok=True)

        # One spare, so a frame written off the normal cadence still lands.
        self._n_frames = 2 + n_steps // self.every_k
        #: Each saved field's opened array and how it gets filled. Public
        #: because the run logs what the fields turned out to be.
        self.fields: dict[str, Series] = {
            name: series_for(
                name,
                expression,
                entry,
                mesh=mesh,
                open_array=self._open,
                n_frames=self._n_frames,
            )
            for name, (expression, entry) in fields.items()
        }
        self._times = self._sidecar(TIMES_FILE, np.float64)
        self._steps = self._sidecar(STEPS_FILE, np.int64)

        self._copy_inputs()
        self._archive(source)
        self._write_metadata()

    @classmethod
    def of(
        cls,
        system: "PhaseFieldSystem",
        *,
        n_steps: int | None = None,
        resume_frame: int | None = None,
    ) -> "Saver":
        """Open the saver a mixture's own declarations describe."""

        parameters = system.parameters
        return cls(
            system.saved_fields(),
            parameters.solver.save,
            system.solver_parameters,
            n_steps=(
                parameters.solver.timestepping.n_steps if n_steps is None else n_steps
            ),
            resumable_every=parameters.solver.save.resumable_every,
            # What a resume has to read back, each at the function holding it --
            # a state at ``next``, a lagged rate at ``rates`` -- and filed under
            # the archive's own name for it, so what a run writes and what a
            # resume looks for cannot drift apart.
            resume_values={
                qualified(spec.owner, spec.name): spec.function()
                for spec in system.resume_specs()
            },
            resume_frame=resume_frame,
            source=getattr(parameters, "source", None),
            geometry=parameters.solver.diffuse_domain.geometry_path,
            geometry_assets=parameters.solver.diffuse_domain.geometry_assets(),
        )

    def __enter__(self) -> "Saver":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

    ## Frames

    def frame(self, time: float) -> None:
        """Write the current values, if this step is one the cadence wants.

        Step zero always is. It is the initial condition, which is what every
        later frame is a change from, and a run saved without it cannot be
        differenced or normalized.
        """

        if self._closed:
            raise RuntimeError("cannot write to a closed saver")
        if self._step == 0 or self._step % self.every_k == 0:
            self.write(time)
        self._step += 1

    def write(self, time: float) -> None:
        """Write a frame regardless of cadence, for a step worth keeping anyway."""

        if self._frame >= self._n_frames:
            raise RuntimeError(
                f"more frames than the {self._n_frames} preallocated; the saver "
                "sizes itself from n_steps and this run took more"
            )

        for series in self.fields.values():
            series.write(self._frame)
            series.array.flush()
        self._times[self._frame] = time
        self._steps[self._frame] = self._step
        self._times.flush()
        self._steps.flush()
        self._frame += 1
        # Every frame, not only at close, so a run can be read while it runs: the
        # values are already flushed, and the frame count is the only thing
        # between a reader and them. A few kilobytes of JSON against a step that
        # takes seconds.
        self._write_metadata()

    def resume_point(self, time: float, *, force: bool = False) -> None:
        """Overwrite the run's restart point, if this step is one the cadence wants.

        ``force`` writes one regardless, which is what a run stopping cleanly
        does: the cadence may have last fired several steps ago, and there is no
        reason to make someone redo work the run actually completed.
        """

        if not force and (
            self.resumable_every <= 0 or self._step % self.resumable_every != 0
        ):
            return
        self.write_resume_point(time)

    def write_resume_point(self, time: float) -> None:
        """Write the restart point into the slot not in use, then commit it.

        A restart point is a dozen or more arrays, and writing them is not one
        operation: a run stopped partway would leave some new and some old -- a
        state that looks valid, describes no instant, and would be restored
        without complaint. Writing into the *unused* slot leaves the committed
        one untouched until every array is on disk, and the commit is one atomic
        rewrite of ``meta.json``. So an interruption leaves either the new
        restart point committed or the previous one, and nothing else.
        """

        if not self._resume_values:
            return
        with uninterrupted():
            slot = "b" if self._resume_slot == "a" else "a"
            directory = self.path / RESUME_DIR / slot
            directory.mkdir(parents=True, exist_ok=True)
            fields = {}
            for name, function in self._resume_values.items():
                values = np.asarray(function.x.array, dtype=np.float64)
                path = directory / f"{safe_filename(name)}.resume.npy"
                with open(path, "wb") as handle:
                    np.save(handle, values)
                    handle.flush()
                    os.fsync(handle.fileno())
                fields[name] = {"dofs": int(values.shape[0])}
            self._resume_record = {
                "slot": slot,
                # Both counters, so the cadences need not divide one another:
                # the run seeks to exactly the frame and step this was taken at
                # and rewrites whatever came after.
                "step": self._step,
                "frame": self._frame,
                "time": float(time),
                "fields": fields,
            }
            self._resume_slot = slot
            self._write_metadata()
        logger.debug(
            "restart point written to %s at step %d (t=%.6e)",
            slot,
            self._step,
            time,
        )

    def close(self) -> None:
        """Flush, and record how many of the preallocated frames were used.

        Without the count a reader cannot tell a run that stopped early from one
        that wrote zeros. It is written every frame as well, so that a reader can
        follow a run in progress; this is the last, and the one that is right if
        the run ended between frames.
        """

        if self._closed:
            return
        for series in self.fields.values():
            series.array.flush()
        self._times.flush()
        self._steps.flush()
        self._closed = True
        self._write_metadata()

    ## Setting up

    def _committed_metadata(self) -> dict:
        """The metadata already in this directory, or nothing if it has none."""

        try:
            return json.loads((self.path / META_FILE).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def _open(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        return self._array(f"{safe_filename(name)}.npy", self.dtype, shape)

    def _sidecar(self, filename: str, dtype) -> np.ndarray:
        return self._array(filename, np.dtype(dtype), (self._n_frames,))

    def _array(self, filename: str, dtype, shape: tuple[int, ...]) -> np.ndarray:
        """One preallocated array, created or -- when resuming -- reopened in place.

        A reopened array is checked against what this spec would have allocated,
        because the frames already in it only mean something if they are the same
        frames: a differing length means the step count changed, a differing
        width that the field or the mesh did. Either is refused here, before
        anything is written, rather than found as a shape error partway through.
        """

        path = self.path / filename
        if self._resume_frame is None:
            return np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot resume: {path} is not there, so the run in {self.path} "
                f"did not save it"
            )
        array = np.lib.format.open_memmap(path, mode="r+")
        if tuple(array.shape) != tuple(shape) or array.dtype != dtype:
            raise ValueError(
                f"cannot resume into {path}: it holds {array.shape} of "
                f"{array.dtype} where this run wants {shape} of {dtype}. The "
                f"preallocated length is set by n_steps and every_k, so resuming "
                f"needs the step count the run was started with"
            )
        return array

    def _copy_inputs(self) -> None:
        """The mesh and the geometry the run was posed on, copied in beside the dofs.

        The mesh is the file the run read, not a fresh write of the one in memory:
        see :data:`MESH_FILE`. XDMF is a pair -- an XML file naming its arrays and
        an HDF5 file holding them, referenced by bare filename -- so the companion
        is renamed to match the fixed name and the XML keeps pointing at something
        that is there.

        The geometry comes too, with whatever it reads
        (:func:`~ovpsolver.diffuse_domain.sdf.base.load_assets`): a reader
        recomputes the indicator from it, and the mesh was built from it, so a run
        that lost it could say neither where its inclusions were nor how it was
        meshed.
        """

        source = Path(self.solver_parameters.mesh_source_path)
        destination = self.path / MESH_FILE
        if source.resolve() != destination.resolve():
            copyfile(source, destination)
            companion = source.with_suffix(".h5")
            if companion.is_file():
                copyfile(companion, destination.with_suffix(".h5"))

        if self.geometry is not None:
            # Guarded like the mesh above: resuming from a run's own archived
            # spec names the geometry already sitting in this directory.
            geometry = Path(self.geometry)
            destination = self.path / GEOMETRY_FILE
            if geometry.resolve() != destination.resolve():
                copyfile(geometry, destination)
        # Whatever the geometry reads comes too, under its own name, because the
        # archived copy looks for it beside itself by that name.
        for asset in self.geometry_assets:
            destination = self.path / Path(asset).name
            if Path(asset).resolve() != destination.resolve():
                copyfile(asset, destination)

    def _archive(self, source: Path | None) -> None:
        """Copy the spec in beside its output, with its paths pointed inside it.

        A spec names two paths -- the output directory and the geometry -- both
        right at launch and wrong once the directory moves. The archived copy
        names ``.`` and the geometry copied in beside it, both resolved relative
        to the spec, so a finished run reads back from inside itself wherever it
        is taken. The mesh needs no entry: it is archived under a fixed name,
        where :attr:`~ovpsolver.solver.parameters.SolverParameters.mesh_path`
        finds it.

        Rewritten as text rather than re-dumped from the parsed document, because
        a spec is written to be read, and its comments saying why each number is
        what it is would not survive the round trip.
        """

        if source is None:
            return
        source = Path(source).expanduser()
        if not source.is_file():
            raise ValueError(f"the spec to archive does not exist: {source}")

        wanted = {
            ("solver", "save", "path"): ".",
            ("solver", "diffuse_domain", "geometry"): f"./{GEOMETRY_FILE}",
        }
        rewritten: set[tuple[str, ...]] = set()
        output: list[str] = []
        trail: list[str] = []

        for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            content = stripped.rstrip("\r\n")
            if not content or content.startswith("#"):
                output.append(line)
                continue

            # A mapping key at this indent replaces everything at or below it.
            del trail[indent // 2 :]
            key = content.split(":", 1)[0].strip()
            path = (*trail, key)
            trail.append(key)

            replacement = wanted.get(path)
            if replacement is None:
                output.append(line)
                continue
            ending = "\n" if line.endswith("\n") else ""
            output.append(f"{' ' * indent}{key}: {replacement}{ending}")
            rewritten.add(path)

        missing = set(wanted) - rewritten
        if missing:
            raise ValueError(
                f"{source} names no "
                f"{', '.join('.'.join(path) for path in sorted(missing))} to rewrite, "
                f"so the archived spec would point outside its own directory"
            )
        (self.path / SPEC_FILE).write_text("".join(output), encoding="utf-8")

    def _write_metadata(self) -> None:
        """The few things about this run that its spec does not already say.

        Everything else a reader might want -- the step, the cadence, the
        precision, the quadrature degree -- is in the archived spec, which parses
        against the same classes the run was built from; a copy here would be a
        second one free to disagree. What the spec cannot say is how many frames
        were written (without which a short run is indistinguishable from a run
        of zeros), which element each field turned out to be, whether a field is
        static (a one-row array cannot tell a field whose one row answers every
        frame from a run that died after one frame), the committed restart
        point, and which code wrote which frames.
        """

        metadata = {
            "frames_written": self._frame,
            "resume": self._resume_record,
            "code": self._code,
            "fields": {
                name: {
                    "filename": f"{safe_filename(name)}.npy",
                    "mode": series.mode,
                    "element": series.element.to_dict(),
                    "static": series.static,
                }
                for name, series in self.fields.items()
            },
        }
        # Staged and moved into place, because this runs after every frame and a
        # run is expected to be interrupted. A plain write truncates first, and a
        # process killed in that window leaves an unreadable file -- the only
        # record of how many frames are real, so the whole directory goes with
        # it. ``os.replace`` is atomic within a filesystem, and the fsync before
        # it makes that survive the machine going down and not just the process.
        document = json.dumps(metadata, indent=2, sort_keys=True)
        target = self.path / META_FILE
        staged = target.with_name(f"{META_FILE}.writing")
        with open(staged, "w", encoding="utf-8") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)


@contextmanager
def uninterrupted():
    """Hold a keyboard interrupt until the block finishes, then deliver it.

    Ctrl-C landing between a restart point's files and the write that commits
    them is the likeliest way anyone will interrupt a run, and deferring the
    signal closes that window. It closes only that one: a kill, a crash or the
    machine going down cannot be caught, which is what the unused slot and the
    atomic commit are for.

    A second interrupt is delivered immediately. Someone pressing it twice has
    decided not to wait, and a program that cannot be stopped is worse than a
    restart point that has to be written again. Off the main thread there is
    nothing to install, and the block runs unprotected rather than failing.
    """

    caught = False

    def handler(signum, frame):
        nonlocal caught
        if caught:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        caught = True
        logger.warning(
            "interrupt held until the restart point is written; press again to "
            "stop now and leave the previous one in place"
        )

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    # Outside the finally, so an exception from the block is the one that
    # propagates and the interrupt does not mask it.
    if caught:
        raise KeyboardInterrupt


def safe_filename(name: str) -> str:
    """``name`` with every character but alphanumerics and ``._-`` made ``_``."""

    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )


@cache
def source_version() -> dict[str, Any]:
    """The code this process runs: its git commit, and a digest of its source.

    Asked of git in :data:`SOURCE_DIR`, so the answer does not depend on where
    the run was launched from. The commit alone does not say what ran when the
    tree has uncommitted changes -- the usual state of a run resumed after a
    fix -- so ``digest`` hashes every file git sees there, committed or not,
    and skips what it ignores (the build, the bytecode). Equal digests ran the
    same source.

    Cached, so a process reports its source as it was at the first call, which
    :func:`~ovpsolver.phase_field_system.run.run` makes before compiling
    anything: an edit made during a long compile changes the files, not the
    code already imported. All ``None`` outside a git checkout of the package,
    as for an installed wheel -- including one inside some other repository,
    whose commit would otherwise be reported as this code's.
    """

    def git(*arguments: str) -> bytes:
        return subprocess.run(
            # No optional locks: a status that refreshed the index could make a
            # concurrent commit of the user's fail on the lock.
            ["git", "--no-optional-locks", "-C", str(SOURCE_DIR), *arguments],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout

    try:
        git("ls-files", "--error-unmatch", "--", str(Path(__file__).resolve()))
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain", "--", "."))
        names = git(
            "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "."
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "digest": None}
    digest = hashlib.sha256()
    for name in sorted(set(names.split(b"\0")) - {b""}):
        path = SOURCE_DIR / os.fsdecode(name)
        # Skips a tracked file deleted from the tree, which the name set
        # records anyway by lacking it.
        if path.is_file():
            digest.update(name + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return {"commit": commit, "dirty": dirty, "digest": digest.hexdigest()}


def describe_version(version: "Mapping[str, Any] | None") -> str:
    """A :func:`source_version`, or a recorded one, in a line for the log."""

    if not version or version.get("commit") is None:
        return "an unrecorded version"
    changes = " with uncommitted changes" if version["dirty"] else ""
    return f"commit {version['commit'][:12]}{changes} (source {version['digest'][:12]})"
