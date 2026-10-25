"""Sweeps: one spec run many times, with values along addresses into ``system`` varied.

::

    python -m ovpsolver.polymerizing_b run examples/pips.yaml \\
        --sweep-zip 'phase_fields.sol[0].kappa=[0.5e-3, 1.0e-3, 1.5e-3]' \\
                    'phase_fields.polymerizing[0].kappa=[0.5e-3, 1.0e-3, 1.5e-3]' \\
        --sweep 'couplings.flory_huggins.(water, protein).chi_01=[1, 2, 3, 4, 5]'

is fifteen runs. The two ``kappa`` step together through their three values, and
each of those runs against all five ``chi_01``. Every ``--sweep`` assignment is an
axis of its own; a ``--sweep-zip`` is one axis, whose assignments step in lockstep
and so need as many values each. The runs are every combination of the axes, in
the order the axes were given, the last varying fastest.

Addresses
---------
An address is a path into the spec's ``system`` block, read off the document
rather than off the parameter classes, so it reaches anything a document holds
and knows nothing of what the mixture declares: ``.key`` into a mapping, ``[n]``
into a list. A key is everything up to the next ``.``, ``[`` or ``]``, spaces and
parentheses included, and is quoted to hold one of those. It has to exist
already. A spec is read strictly, so it already has every key there is, and a
missing one is a typo, reported with the keys that are there.

The values are a YAML list, each element one run's value, so a list-valued
parameter sweeps as a list of lists. A bare comma-separated list, or a single
value, is read as that list. Values read as they would in the spec, except that
``1e-3`` is a number. Quote each assignment in the shell: brackets glob, and
spaces split it.

Variants
--------
Each run is a variant: the spec's own text, with the swept values replaced where
they stand and ``solver.save.path`` replaced by the run's own directory. Text,
not a re-dumped document, for the reason the archive is text: a spec carries the
reasons for its numbers as comments, and a run archived without them would not
say why the numbers it did not sweep are what they are. A header comment records
what was swept, so every run's archive says what it was a run of. Each value is
found by the YAML parser's own node positions rather than by matching lines, so
any layout the parser reads can be rewritten. Every variant is read back, and
compared with the document it should be, before it is written.

The variants are written to a scratch directory and read against the spec they
vary, so a relative geometry or output path means what it means in the original.

A run's directory is the original's with ``-label=value`` appended for each swept
address, where ``label`` is the shortest trailing part of the address that no
other swept address ends in: ``scratch/pips-sol.0.kappa=0.0005-polymerizing.0.kappa=0.0005-chi_01=3``.
Not ``label:value``, since a colon is the path separator of the macOS Finder and
is refused by Windows, and with it by a synced tree.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: A path into a document: mapping keys and list indices, from the top.
Address = tuple["str | int", ...]

#: The block every address starts in. A sweep varies the physics. The solver
#: block says how a run is taken and where it goes, and where it goes is the one
#: thing the sweep decides for itself.
ROOT = "system"

#: Where a run's output goes, which every variant replaces with its own.
SAVE_PATH: Address = ("solver", "save", "path")

#: How a group of assignments becomes axes: one axis each, or one between them.
PRODUCT, ZIP = "product", "zip"

#: The longest file name most filesystems allow, in bytes, less room for the
#: variant's ``.yaml``.
NAME_MAX = 250

#: Characters a key can only hold quoted, and an index's brackets.
_SPECIAL = frozenset(".[]\"'")


class SweepError(ValueError):
    """A sweep that cannot be laid out: an address, a list of values, or a name."""


class SweepFailed(RuntimeError):
    """Some runs of a sweep failed; the others ran to the end.

    Raised once the last run is over, so a command line that swept exits
    unsuccessfully without any run having been skipped for another's failure.
    """

    def __init__(
        self, reached: "list[float | None]", failures: "list[tuple[Path, str]]"
    ) -> None:
        #: What each run reached, in run order, ``None`` for one that failed.
        self.reached = reached
        #: Each failed run's directory, and the first line of what stopped it.
        self.failures = failures
        lines = "".join(f"\n  {directory}: {why}" for directory, why in failures)
        super().__init__(
            f"{len(failures)} of {len(reached)} runs failed, each recorded in the "
            f"log in its directory:{lines}"
        )


class _ValueLoader(yaml.SafeLoader):
    """The spec's own YAML, with ``1e-3`` read as the number it is.

    PyYAML reads YAML 1.1, whose floats need a dot and a signed exponent, so
    ``1e-3`` is a string there and a number parameter refuses it. A spec writes
    ``1.0e-3``. A command line should not have to.
    """


_ValueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+.0123456789"),
)


## Addresses


def parse_address(text: str) -> Address:
    """``phase_fields.sol[0].kappa`` as ``("phase_fields", "sol", 0, "kappa")``."""

    def fail(why: str) -> SweepError:
        return SweepError(f"address {text!r}: {why}")

    steps: list[str | int] = []
    position = 0
    # A key is wanted at the start and after every '.'; anything else is wanted
    # after a key or an index.
    wanted = True
    while position < len(text):
        char = text[position]
        if char == "[":
            if wanted:
                raise fail("an index follows a key, as in sol[0]")
            close = text.find("]", position)
            if close < 0:
                raise fail("a '[' is never closed")
            index = text[position + 1 : close].strip()
            if not index.isdigit():
                raise fail(f"[{index}] is not a list index; a key follows a '.'")
            steps.append(int(index))
            position = close + 1
        elif char == ".":
            if wanted:
                raise fail("a key is empty")
            wanted = True
            position += 1
        elif not wanted:
            raise fail(f"expected '.' or '[' before {text[position:]!r}")
        else:
            if char in "\"'":
                close = text.find(char, position + 1)
                if close < 0:
                    raise fail(f"a {char} is never closed")
                key, position = text[position + 1 : close], close + 1
            else:
                end = position
                while end < len(text) and text[end] not in _SPECIAL:
                    end += 1
                key, position = text[position:end].strip(), end
            if not key:
                raise fail("a key is empty")
            steps.append(key)
            wanted = False
    if wanted:
        raise fail("it ends in a '.'" if steps else "it is empty")
    return tuple(steps)


def render(address: Address) -> str:
    """An address as it is written, quoting a key only where it has to be."""

    text = ""
    for step in address:
        if isinstance(step, int):
            text += f"[{step}]"
            continue
        key = step
        if step != step.strip() or _SPECIAL.intersection(step):
            quote = "'" if '"' in step else '"'
            key = f"{quote}{step}{quote}"
        text += f".{key}" if text else key
    return text


def lookup(document: Any, address: Address, where: str) -> Any:
    """The value at ``address``, or why there is none, naming what is there."""

    node: Any = document
    for depth, step in enumerate(address):
        here = render(address[:depth]) or "the document"
        if isinstance(step, int):
            if not isinstance(node, list):
                raise SweepError(f"{where}: {here} is {_kind(node)}, so [{step}] indexes nothing")
            if step >= len(node):
                raise SweepError(
                    f"{where}: {here} has {len(node)} entries, so [{step}] is past its end"
                )
            node = node[step]
            continue
        if not isinstance(node, dict):
            raise SweepError(f"{where}: {here} is {_kind(node)}, so .{step} names nothing")
        key = _key(node, step)
        if key is None:
            present = ", ".join(str(name) for name in node) or "nothing"
            raise SweepError(f"{where}: {here} has no {step!r}; it has {present}")
        node = node[key]
    return node


def assign(document: Any, address: Address, value: Any) -> None:
    """Set the value at ``address``, which has to be there already."""

    parent = lookup(document, address[:-1], "the variant")
    step = address[-1]
    parent[step if isinstance(step, int) else _key(parent, step)] = value


## What a command line asks for


@dataclass(frozen=True)
class Assignment:
    """One address, and the value it takes in each run along its axis."""

    address: Address
    values: tuple[Any, ...]

    @classmethod
    def parse(cls, text: str) -> "Assignment":
        """``ADDRESS=VALUES``, split at the first ``=`` outside a quoted key."""

        split = _split(text)
        if split is None:
            raise SweepError(f"{text!r} is not ADDRESS=VALUES")
        address, values = split
        return cls(parse_address(address), _read_values(values, text))


def assignments(words: Iterable[str]) -> list[Assignment]:
    """The assignments a command line spelled, rejoining any the shell split.

    An unquoted ``chi_01=[1, 2, 3]`` reaches the program as three words, and only
    the first has an ``=`` in it. A word without one continues the assignment
    before it.
    """

    pieces: list[str] = []
    for word in words:
        if pieces and _split(word) is None:
            pieces[-1] += f" {word}"
        else:
            pieces.append(word)
    return [Assignment.parse(piece) for piece in pieces]


@dataclass(frozen=True)
class Variant:
    """One run of a sweep: the spec it varies, the file written for it, what it sets.

    A spec run as written is the variant that sets nothing, its own origin.
    """

    origin: Path
    path: Path
    settings: tuple[tuple[Assignment, Any], ...]

    def describe(self) -> str:
        return ", ".join(
            f"{render(assignment.address)}={_flow(value)}"
            for assignment, value in self.settings
        )


@dataclass(frozen=True)
class Sweep:
    """Axes of assignments, and the runs that are every combination of them."""

    #: Each axis a tuple of assignments that step together.
    axes: tuple[tuple[Assignment, ...], ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise SweepError("a sweep needs at least one assignment")
        for axis in self.axes:
            if len({len(assignment.values) for assignment in axis}) > 1:
                counts = ", ".join(
                    f"{len(assignment.values)} for {render(assignment.address)}"
                    for assignment in axis
                )
                raise SweepError(
                    f"--sweep-zip steps its assignments together, so each needs as "
                    f"many values as the others; got {counts}"
                )
        # Disjoint, or a variant would set one value twice, or set a block and
        # then a value inside the block it just replaced.
        addresses = self.addresses()
        for index, first in enumerate(addresses):
            for second in addresses[index + 1 :]:
                shared = min(len(first), len(second))
                if first[:shared] == second[:shared]:
                    raise SweepError(
                        f"{render(first)} and {render(second)} are swept both, and "
                        f"one of them is {'the other' if first == second else 'inside the other'}"
                    )

    @classmethod
    def from_groups(cls, groups: Sequence[tuple[str, Sequence[str]]]) -> "Sweep":
        """The sweep a command line's groups make, in the order they were given.

        Each group is the kind of option that opened it, :data:`PRODUCT` or
        :data:`ZIP`, and the words that followed.
        """

        axes: list[tuple[Assignment, ...]] = []
        for kind, words in groups:
            parsed = assignments(words)
            if kind == ZIP:
                axes.append(tuple(parsed))
            else:
                axes.extend((assignment,) for assignment in parsed)
        return cls(tuple(axes))

    def addresses(self) -> list[Address]:
        return [assignment.address for axis in self.axes for assignment in axis]

    def __len__(self) -> int:
        return math.prod(len(axis[0].values) for axis in self.axes)

    def points(self) -> list[tuple[tuple[Assignment, Any], ...]]:
        """Every run's settings, as ``(assignment, value)`` pairs, in run order."""

        steps = [
            [
                tuple((assignment, assignment.values[index]) for assignment in axis)
                for index in range(len(axis[0].values))
            ]
            for axis in self.axes
        ]
        return [
            tuple(pair for step in combination for pair in step)
            for combination in itertools.product(*steps)
        ]

    def labels(self) -> dict[Address, str]:
        """Each swept address by its shortest trailing part no other one ends in."""

        addresses = self.addresses()
        labels: dict[Address, str] = {}
        for address in addresses:
            for length in range(1, len(address) + 1):
                tail = address[-length:]
                if not any(
                    other != address and other[-length:] == tail for other in addresses
                ):
                    break
            labels[address] = ".".join(_safe(str(step)) for step in tail)
        return labels

    def variants(self, spec: "str | Path", scratch: Path) -> list[Variant]:
        """Write one variant of ``spec`` per run under ``scratch``, in run order.

        Each is named for the directory its run writes to, so a message about a
        variant names the run it is about.
        """

        origin = Path(spec).expanduser().resolve()
        where = origin.name
        text = origin.read_text(encoding="utf-8")
        document = yaml.safe_load(text)
        if not isinstance(document, dict):
            raise SweepError(f"{where} must contain a mapping at the top level")
        for address in self.addresses():
            lookup(document, (ROOT, *address), where)
        saved = lookup(document, SAVE_PATH, where)
        if not isinstance(saved, str):
            raise SweepError(f"{where}: {render(SAVE_PATH)} is not a path: {saved!r}")
        # Absolute, so that a spec saving to '.' -- one archived by a run --
        # still has a name to extend.
        base = Path(os.path.abspath(origin.parent / Path(saved).expanduser()))

        own = Path(tempfile.mkdtemp(prefix=f"{origin.stem}-", dir=scratch))
        labels = self.labels()
        points = self.points()
        variants = []
        for number, settings in enumerate(points, start=1):
            name = base.name + "".join(
                f"-{labels[assignment.address]}={_name_of(value)}"
                for assignment, value in settings
            )
            if len(name.encode()) > NAME_MAX:
                raise SweepError(
                    f"{where}: the run directory {name!r} is longer than a file name "
                    f"can be; sweep fewer addresses at once, or shorten solver.save.path"
                )
            edits = {(ROOT, *assignment.address): value for assignment, value in settings}
            edits[SAVE_PATH] = str(base.with_name(name))
            path = own / f"{name}{origin.suffix}"
            path.write_text(
                _header(origin, number, len(points), settings)
                + rewrite(text, document, edits, where),
                encoding="utf-8",
            )
            variants.append(Variant(origin, path, settings))
        return variants


## Rewriting a spec's text


def rewrite(text: str, document: Any, edits: dict[Address, Any], where: str) -> str:
    """``text`` with the value at each address replaced where it stands.

    ``document`` is ``text`` as read. The result is read back and has to be that
    with ``edits`` applied, so whatever the YAML does -- an alias, a merge key --
    a variant is exactly what was asked for or is not written at all.
    """

    root = yaml.compose(text, Loader=yaml.SafeLoader)
    spans = sorted(
        (*_span(_node_at(root, address, where)), _flow(value))
        for address, value in edits.items()
    )
    for (_, end, _), (start, _, _) in zip(spans, spans[1:]):
        if start < end:
            raise SweepError(f"{where}: two of the swept values overlap in the text")
    for start, end, replacement in reversed(spans):
        text = text[:start] + replacement + text[end:]

    expected = copy.deepcopy(document)
    for address, value in edits.items():
        assign(expected, address, value)
    if _canonical(yaml.safe_load(text)) != _canonical(expected):
        raise SweepError(
            f"{where} could not be rewritten in place to set "
            f"{', '.join(render(address) for address in edits)}; write that variant "
            f"out as a spec of its own"
        )
    return text


def _node_at(root: yaml.Node, address: Address, where: str) -> yaml.Node:
    node = root
    for step in address:
        if isinstance(node, yaml.MappingNode) and isinstance(step, str):
            matches = [
                value
                for key, value in node.value
                if isinstance(key, yaml.ScalarNode) and key.value == step
            ]
            if matches:
                # The last, as the loader keeps the last of a repeated key.
                node = matches[-1]
                continue
        elif isinstance(node, yaml.SequenceNode) and isinstance(step, int):
            if step < len(node.value):
                node = node.value[step]
                continue
        raise SweepError(f"{where}: {render(address)} is not in the text where the document has it")
    return node


def _span(node: yaml.Node) -> tuple[int, int]:
    """Where a node's text starts and ends.

    A block collection ends where the next thing starts, past any comments in
    between, so its text ends where its last element's does.
    """

    if isinstance(node, yaml.ScalarNode) or node.flow_style or not node.value:
        return node.start_mark.index, node.end_mark.index
    last = node.value[-1]
    return node.start_mark.index, _span(last[1] if isinstance(last, tuple) else last)[1]


def _header(
    origin: Path, number: int, total: int, settings: tuple[tuple[Assignment, Any], ...]
) -> str:
    lines = [
        f"# Run {number} of {total} of a sweep over the spec",
        f"#   {origin}",
        "# with its own solver.save.path, and these values under system:",
        *(f"#   {render(assignment.address)} = {_flow(value)}" for assignment, value in settings),
        "#",
        "",
    ]
    return "\n".join(lines)


## Values


def _split(text: str) -> tuple[str, str] | None:
    """``text`` at its first ``=`` outside a quoted key, if it has one."""

    quote = None
    for index, char in enumerate(text):
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "=":
            return text[:index], text[index + 1 :]
    return None


def _read_values(text: str, assignment: str) -> tuple[Any, ...]:
    try:
        values = yaml.load(text, Loader=_ValueLoader)
        if not isinstance(values, list):
            values = yaml.load(f"[{text}]", Loader=_ValueLoader)
    except yaml.YAMLError as error:
        raise SweepError(
            f"{assignment!r}: the values are not YAML ({' '.join(str(error).split())}). "
            f"Name the specs before any sweep, or they are read as values"
        ) from None
    if not values:
        raise SweepError(f"{assignment!r} sweeps over no values")
    return tuple(values)


def _flow(value: Any) -> str:
    """``value`` as YAML on one line, as a spec would write it."""

    text = yaml.safe_dump(
        value,
        default_flow_style=True,
        width=float("inf"),
        allow_unicode=True,
        sort_keys=False,
    ).rstrip("\n")
    return text.removesuffix("\n...")


def _name_of(value: Any) -> str:
    """A value as a directory name reads it: a float shortest-exact, the rest as YAML."""

    return _safe(repr(value) if isinstance(value, float) else _flow(value))


def _safe(text: str) -> str:
    """``text`` with every run of characters a file name should not hold made ``_``."""

    return re.sub(r"[^A-Za-z0-9._+-]+", "_", text).strip("_") or "_"


def _canonical(document: Any) -> str:
    return yaml.safe_dump(document, sort_keys=True)


def _key(mapping: dict, step: str) -> Any:
    """The key of ``mapping`` an address step names, or ``None``."""

    for key in mapping:
        if key == step or str(key) == step:
            return key
    return None


def _kind(node: Any) -> str:
    if isinstance(node, dict):
        return "a mapping"
    if isinstance(node, list):
        return "a list"
    return f"the value {_flow(node)}"
