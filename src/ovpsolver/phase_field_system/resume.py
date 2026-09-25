"""Restarting a run from a frame it already wrote.

When something in the scheme turns out to be wrong at step 130 of 400, the state
at step 129 is still the state the corrected code would have reached, and
re-deriving it buys nothing.

What a resume has to put back is exactly enumerable, and the element
declarations say what it is:
:meth:`~ovpsolver.fem.elements.ElementOwner.resume_specs` is the whole
requirement, and :meth:`~ovpsolver.fem.elements.ElementOwner.resume_from` is the
whole restoration. Nothing in this module walks a mixture or knows what a phase
is; a mixture that gains state becomes resumable by declaring it.

A resumed run is identical to a continued one but for a single number. The
solver proposes the next step from the bound its last accepted step reported; a
resumed run has none, so :meth:`~ovpsolver.solver.PhaseFieldSolver._propose`
falls back to the spec's ``warm_start_dt`` and the estimate re-establishes from
the first step. Everything else -- the upwind selectors, the convex split, the
saturation pairing, the step bound -- reads state that is on disk.

Nothing here writes. Everything is checked before the run opens a single array
for writing, so a resume that cannot be honoured leaves the directory as it
found it and says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..fem.save import Run
from ..fem.save.saver import RESUME_DIR, safe_filename
from ..fem.saveable import qualified

if TYPE_CHECKING:
    from dolfinx.fem import Function

    from ..fem.elements import ElementSpec
    from .system import PhaseFieldSystem


@dataclass(frozen=True)
class Resume:
    """One saved frame, and the reading of it into a built system."""

    run: "Run"
    #: Which of the two restart slots holds it.
    slot: str
    #: How many frames the run had written when this was taken: the resumed run
    #: writes its first frame at that index, over whatever came after.
    frame: int
    #: The macro step the resumed run takes first, which is how many had been
    #: taken. One less than the saver's own step counter, which counts the
    #: initial condition as a step.
    next_step: int
    #: The time reached at that frame, which the resumed run counts on from.
    time: float
    #: The code that wrote the frames the resumed run keeps, as
    #: :func:`~ovpsolver.fem.save.saver.source_version` recorded it; ``None``
    #: for a run that recorded none.
    code: dict | None

    @classmethod
    def last(cls, system: "PhaseFieldSystem") -> "Resume":
        """The restart point this run last committed, or a refusal saying why not.

        There is one, overwritten in place: the purpose is to carry a solve past
        a step it could not take, not to revisit a trajectory, and keeping the
        history of a state like the generating function, sixty-four values per
        cell per step, would cost gigabytes.

        What is on disk is checked against what the built mixture declares it
        needs rather than trusted: a spec edited between the two runs can have
        changed what the state consists of, and the arrays would not report that
        themselves.
        """

        run = Run.of(system.parameters)
        record = run.metadata.get("resume")
        if not record:
            raise ValueError(
                f"{run.path} has no restart point to resume from. A run writes "
                f"one every solver.save.resumable_every macro steps, so either "
                f"it stopped before the first was due or it was written by a "
                f"version that did not keep one"
            )

        held = tuple(record.get("fields", {}))
        missing = [
            name
            for name in map(cls.saved_name, system.resume_specs())
            if name not in held
        ]
        if missing:
            raise ValueError(
                f"the restart point in {run.path} is missing {', '.join(missing)}, "
                f"which this mixture needs to be put back as it was. It was "
                f"written for a different set of states than the spec now "
                f"declares"
            )
        frame = int(record["frame"])
        written = [
            entry
            for entry in run.metadata.get("code", ())
            if entry["first_frame"] < frame
        ]
        return cls(
            run=run,
            slot=str(record["slot"]),
            frame=frame,
            next_step=int(record["step"]) - 1,
            time=float(record["time"]),
            code=written[-1] if written else None,
        )

    @staticmethod
    def saved_name(spec: "ElementSpec") -> str:
        """What the archive files this declaration under.

        The owner's save qualifier and not its
        :attr:`~ovpsolver.fem.elements.ElementSpec.label`. The two differ for the
        mixture itself, whose fields go under ``system`` so a reader finds them
        at one name whichever mixture wrote them.

        An owner that is not saveable at all -- a coupling, which holds state and
        has no way to write it -- falls back to its name, which nothing on disk
        matches. Its state is then reported missing, correctly: a run of one
        cannot be resumed, and silently resetting it would be worse.
        """

        return qualified(spec.owner, spec.name)

    def read(self, function: "Function", spec: "ElementSpec") -> None:
        """Fill one function from this frame of the field that ``spec`` declares.

        The callable :meth:`~ovpsolver.fem.elements.ElementOwner.resume_from` is
        given, which is why it takes a declaration rather than a name: what a
        field is called on disk is this archive's convention, not the element's.

        Read out of the restart slot, whose row is the dof vector this very
        function held, and assigned straight into its dofs rather than put
        through :meth:`~ovpsolver.fem.save.Field.sample`. Sampling moves a field
        onto a different space, and would be wrong here rather than merely
        wasteful: a reader opens the mesh on its own communicator, so the run's
        space and the solver's are two objects over two meshes read from one
        file, ``sample`` compares spaces by identity, never matches, and falls
        through to an interpolation between meshes that dolfinx may refuse.
        Copying the dofs is exact and has no mesh in it to disagree about.
        """

        name = self.saved_name(spec)
        path = (
            self.run.path
            / RESUME_DIR
            / self.slot
            / f"{safe_filename(name)}.resume.npy"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"the restart point claims to hold {name!r} but {path} is not "
                f"there; the slot is incomplete and cannot be restored from"
            )
        values = np.load(path)
        dofs = function.x.array
        if values.shape[0] != dofs.shape[0]:
            raise ValueError(
                f"the restart point holds {values.shape[0]} dofs for {name!r} "
                f"where this run's {spec.name!r} has {dofs.shape[0]}; it was "
                f"written on a different mesh"
            )
        dofs[:] = values
        function.x.scatter_forward()


__all__ = ["Resume"]
