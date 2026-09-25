"""Reading coefficients stated between two named things.

Two mechanisms key a declaration by a pair of names:
:class:`~ovpsolver.parametric.parameters.PairsParameters`, a block whose every
entry is itself a block, and the :class:`~ovpsolver.parametric.parameter.Pairs`
declaration, one field holding a pair-keyed dict of numbers. Both read their
pairs, and the blocks using them resolve, enumerate and name pairs, the same
way. That is all done here, against a plain name-to-position map, since none of
it depends on what is being paired.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def parse_pair(key: Any, where: str) -> tuple[str, str]:
    """The two names a key states: ``(first, second)`` in a document, or a tuple."""

    if isinstance(key, tuple) and len(key) == 2:
        names = [str(name).strip() for name in key]
    elif isinstance(key, str) and key.startswith("(") and key.endswith(")"):
        names = [name.strip() for name in key[1:-1].split(",")]
    else:
        raise ValueError(
            f"{where} keys must be written '(first, second)'; got {key!r}"
        )
    if len(names) != 2 or not all(names):
        raise ValueError(f"{where} key {key!r} must name exactly two phases")
    return (names[0], names[1])


def read_pairs(
    node: Mapping, where: str, read: Callable[[Any, str], Any]
) -> dict[tuple[str, str], Any]:
    """Each entry of a pair-keyed mapping, read by ``read``, in document order.

    A pair is unordered in that naming it the other way round names it again,
    which is refused. The order it was written in is kept, since some of what a
    pair indexes is not symmetric in it.
    """

    entries: dict[tuple[str, str], Any] = {}
    seen: set[frozenset[str]] = set()
    for key, value in node.items():
        pair = parse_pair(key, where)
        if frozenset(pair) in seen:
            raise ValueError(f"{where} names the pair ({show_pair(pair)}) twice")
        seen.add(frozenset(pair))
        entries[pair] = read(value, f"{where}[{key}]")
    return entries


def pair_positions(
    where: str, pair: tuple[str, str], index: dict[str, int]
) -> tuple[int, int]:
    """The roster positions of a pair's two names.

    ``where`` includes the field within the block, if any, so that an error
    says which of several pair-keyed fields held the bad pair.
    """

    try:
        return index[pair[0]], index[pair[1]]
    except KeyError:
        raise ValueError(
            f"{where}[{show_pair(pair)}] names a phase this "
            f"mixture does not have; expected some of {', '.join(index)}"
        ) from None


def distinct_pairs(indices: Iterable[int]) -> set[frozenset[int]]:
    """Every unordered pair of distinct positions among ``indices``.

    What a coefficient stated between two different things has to cover. Taken
    over given positions, since what a coefficient applies to need not be the
    whole roster nor contiguous in it.
    """

    ordered = list(indices)
    return {
        frozenset((first, second))
        for position, first in enumerate(ordered)
        for second in ordered[position + 1 :]
    }


def show_pair(pair: tuple[str, str]) -> str:
    """A pair as the document wrote it, for an error about what it says."""

    return f"{pair[0]}, {pair[1]}"


def show_positions(pair: frozenset[int], names: Sequence[str]) -> str:
    """A pair the roster knows but the document did not write, for an error.

    Ordered by position, so that a list of missing pairs reads in roster order.
    A self-pair arrives as a one-element frozenset and is shown as the pair it
    stands for.
    """

    ordered = sorted(pair)
    if len(ordered) == 1:
        ordered = ordered * 2
    return f"({names[ordered[0]]}, {names[ordered[1]]})"
