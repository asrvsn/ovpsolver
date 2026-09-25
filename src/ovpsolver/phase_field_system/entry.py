"""One thing you can do to a mixture, and how it is asked for at a prompt.

An :class:`EntryPoint` is a name, a parser and a function. The name is what the
first word of the command line has to be -- ``run``, ``visualize.phases``,
``analyze.positivity`` -- the parser says what may follow it, and the function
does the work. Subclassing registers it, so adding a diagnostic is writing a
class and nothing else: no table to extend, no dispatch to add a branch to, and
:mod:`.cli` never learns its name.

The dotted names are a hierarchy in the same sense conda's are: a common prefix
for things of a kind, with no machinery behind it beyond grouping in the help.

What an entry point is *given* is the mixture class, so a subclass inherits
every alternative and overriding one classmethod changes what the corresponding
alternative does. What it must not do is know which mixture: an entry point that
only works for one kind of phase belongs on that mixture, which can add it to
:meth:`~ovpsolver.phase_field_system.system.PhaseFieldSystem.entry_points`.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from typing import NoReturn

    from .system import PhaseFieldSystem

#: Every entry point that has been defined, by name, in definition order, so the
#: CLI can enumerate what exists without importing a list of it.
REGISTRY: dict[str, "type[EntryPoint]"] = {}


class EntryPoint(ABC):
    """One alternative of ``python -m <mixture> <name> ...``.

    Subclass it, give it a :attr:`name` and a :meth:`__call__`, and it appears
    in the help of every mixture. Leave the name off to write an intermediate
    base -- :class:`SpecEntryPoint` below is one -- which is how a family of
    entry points shares its arguments without being one itself.
    """

    #: The word that selects this, dotted for grouping. Empty on a base class.
    name: ClassVar[str] = ""

    #: One line, for the list of alternatives. The class docstring is used for
    #: the fuller description under ``<name> --help``.
    summary: ClassVar[str] = ""

    #: Whether the process should end the moment this finishes, rather than
    #: unwind (see :func:`exit_without_teardown`). Off by default, so that
    #: everything's cleanup runs unless an entry point says otherwise.
    exits_without_teardown: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name"):
            return
        if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
            raise ValueError(
                f"two entry points are called {cls.name!r}: "
                f"{REGISTRY[cls.name].__qualname__} and {cls.__qualname__}"
            )
        REGISTRY[cls.name] = cls

    ## What it takes

    def declare(self, parser: "ArgumentParser") -> None:
        """Add this entry point's own arguments. Nothing, by default."""

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Last chance to turn parsed arguments into call arguments.

        For the few cases argparse cannot express -- a positional list that
        splits into two by what parses as a number, a default that depends on
        another flag. Most entry points leave it alone.
        """

        return arguments

    ## What it does

    @abstractmethod
    def __call__(self, system_class: "type[PhaseFieldSystem]", **arguments: Any) -> Any:
        """Do the thing, and return whatever a script would want back."""

    ## Reading the registry

    @classmethod
    def all(cls) -> tuple["type[EntryPoint]", ...]:
        """Every registered entry point, grouped by prefix and named in order.

        ``run`` first wherever it is, because it is the one that has to happen
        before any of the others mean anything.
        """

        load()
        return tuple(
            sorted(
                REGISTRY.values(),
                key=lambda entry: (entry.name != "run", entry.name),
            )
        )


class SpecEntryPoint(EntryPoint, ABC):
    """An entry point that acts on one spec, which is nearly all of them.

    The spec says what the mixture is and where the run went, so finding the
    saved output is not a second argument.
    """

    def declare(self, parser: "ArgumentParser") -> None:
        parser.add_argument("spec", type=Path, help="path to the experiment YAML")


def load() -> None:
    """Import the modules that define the entry points, so the registry is full.

    A registry filled by subclassing is only as complete as the imports, and
    nothing else imports these modules. So the CLI imports them here, once: a
    list of modules rather than a list of names and signatures.
    """

    for module in (
        ".run",
        ".visualize.domain",
        ".visualize.phases",
        ".visualize.phases_movie",
        ".visualize.gelation",
        ".analyze",
    ):
        import_module(module, __package__)


def exit_without_teardown() -> "NoReturn":
    """End the process now, skipping the exit hooks the imports installed.

    Loading a saved run brings in dolfinx, and with it MPI, PETSc and VTK, each
    of which registers work for interpreter shutdown -- ``MPI_Finalize``,
    ``PetscFinalize``, the teardown of graphics contexts -- that can block for a
    long time, or forever, on a machine with a display attached or a particular
    MPI build. By then the output is written and closed, but from outside a
    finalizer that never returns looks like a command still working, and killing
    it looks like losing the output.

    Nothing skipped is needed: the command is single-rank (``run`` refuses
    anything else), so no communicator is owed a collective call, the saver
    flushes each frame as it writes it, and every file has been closed on the way
    out of the call. Python's own buffers are flushed below, since
    :func:`os._exit` would drop them.

    Called only from :func:`~ovpsolver.phase_field_system.cli.main`: a library
    caller's process is theirs to end.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
