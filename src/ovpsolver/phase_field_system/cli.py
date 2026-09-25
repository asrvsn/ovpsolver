"""One command line per mixture, assembled from whatever entry points exist.

    python -m ovpsolver.polymerizing_b run spec.yaml
    python -m ovpsolver.polymerizing_b visualize.phases spec.yaml 0 40
    python -m ovpsolver.polymerizing_b analyze.saturation spec.yaml

The first word chooses an alternative and everything after it is parsed against
that choice, the shape ``conda`` and ``git`` have.

Nothing here is about any particular alternative. Each one is an
:class:`~ovpsolver.phase_field_system.entry.EntryPoint` subclass that declares
its own arguments, and this module asks the mixture which it has, gives each a
subparser, and calls the one that was named. Adding a diagnostic does not touch
this file.

Nor does anything here know what a mixture is. The mixture class is passed to
the entry point, so a user's subclass gets the same command line by inheriting
it, and one that overrides ``plot`` gets its own behaviour behind the same word::

    class MyMixture(PolymerizingB):
        ...

    if __name__ == "__main__":
        MyMixture.main()
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

from .entry import exit_without_teardown

if TYPE_CHECKING:
    from .entry import EntryPoint
    from .system import PhaseFieldSystem


def main(system_class: "type[PhaseFieldSystem]", argv: "list[str] | None" = None) -> Any:
    """Parse ``argv`` and run the entry point it names against ``system_class``."""

    argv = list(sys.argv[1:] if argv is None else argv)
    entries = {entry.name: entry() for entry in system_class.entry_points()}
    parser = build(system_class, entries)
    if not argv:
        parser.print_help()
        raise SystemExit(2)

    arguments = vars(parser.parse_args(argv))
    entry = entries[arguments.pop("entry_point")]
    result = entry(system_class, **entry.prepare(arguments))
    if entry.exits_without_teardown:
        exit_without_teardown()
    return result


def build(
    system_class: "type[PhaseFieldSystem]", entries: "dict[str, EntryPoint]"
) -> argparse.ArgumentParser:
    """The parser, one subcommand per entry point the mixture offers."""

    name = system_class.__name__
    parser = argparse.ArgumentParser(
        prog=f"python -m {system_class.__module__.rpartition('.')[0]}",
        description=f"Run, draw or measure a {name} mixture from a spec.",
    )
    alternatives = parser.add_subparsers(
        dest="entry_point",
        metavar=" | ".join(entries),
        required=True,
    )
    for entry in entries.values():
        subparser = alternatives.add_parser(
            entry.name,
            help=entry.summary,
            description=(type(entry).__doc__ or entry.summary),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        entry.declare(subparser)
    return parser
