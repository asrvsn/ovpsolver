"""What an object offers to write, and which space each of those goes in.

The object being saved is the only thing that knows either answer, so it says:
:meth:`ParametricSaveable.declare_saveable` returns a name to :class:`Saveable`
mapping, and that mapping is the whole contract. It is what ``save:`` is checked
against when the spec is read, what the value comes from when a frame is written,
and what element the run is described by when it is read back. A name absent from
it cannot be saved, which is an error at spec-reading time rather than at the
first frame.

The space is declared rather than guessed from the expression, because the guess
goes wrong: an expression built from cell-constant states reads as discontinuous
of degree one, and written as DG1 it stores three values per triangle for a
quantity that has one. Most entries are one-to-one -- a declared element's own
dofs, with nothing resampled. The exceptions are the genuinely mixed expressions,
which belong to no single space and have to be told which one to land in, so
reading a ``declare_saveable`` shows which are which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..parametric import Names, Parameters, Parametric
from ..parametric.parameters import ParametersT
from .elements import ElementDomain, ElementSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .elements import ElementOwner

#: What :attr:`Saveable.value` is when the values are a declared element's own,
#: which is the ordinary case and the one worth having a word for.
OWN_DOFS = None


@dataclass(frozen=True)
class Saveable:
    """One thing that can be written, and the space it is written in.

    Parameters
    ----------
    element : the space the values are stored in. For a declared element this is
        that element, and the dofs go to disk untouched; for a derived quantity it
        is the owner's choice, made here and nowhere else.
    value : how to get the values, called with no arguments each frame.
        :data:`OWN_DOFS` for a declared element, whose function the owner
        already holds.
    static : whether the values are the same at every frame, so that one copy is
        the whole series -- the geometry, built once from positions that do not
        move. A claim about the object rather than about this run's length, so
        the owner makes it, and making it wrongly loses the history: a field that
        does change and is declared static is saved as its first frame.
    """

    element: ElementSpec
    value: "Callable[[], Any] | None" = OWN_DOFS
    static: bool = False

    @property
    def derived(self) -> bool:
        return self.value is not OWN_DOFS


class SaveableParameters(Parameters):
    """The ``save`` list, for any block that describes something saveable.

    Declared once here rather than in each such block, which also says at the
    level of the types that a block describes something with state to publish:
    the parameters side of :class:`ParametricSaveable`.

    Parameters
    ----------
    save : which of the object's saveable names this run should write, by their
        unqualified names. Checked against
        :meth:`ParametricSaveable.declare_saveable` when the document is read,
        so a name nothing answers to is an error before anything is built.
        Empty writes nothing, and has to be said: a spec records what a run did,
        and "nothing" is as much a choice as a list.
    """

    save: tuple[str, ...] = Names()


class ParametricSaveable(Parametric[ParametersT]):
    """A parametric object that publishes some of its state for saving.

    What may be saved and what the spec asked to save are two halves of one
    question, answered here so the check happens once, when the document is read.
    The other half is :class:`SaveableParameters`, which every concrete owner's
    parameters derive from, so ``self.parameters.save`` is available here.
    """

    def declare_saveable(self) -> "Mapping[str, Saveable]":
        """Everything this object can be asked to write.

        Nothing, by default: an object that publishes something says so. An
        owner of elements starts from :func:`own_elements` and adds what it
        derives::

            def declare_saveable(self):
                return {
                    **super().declare_saveable(),
                    "reaction_extent": Saveable(
                        self.element("phi"), self.reaction_extent
                    ),
                }

        Reusing a declared element as the target is the usual way to name a
        space, since a derived quantity is nearly always a function of states
        this owner already has and belongs in one of their spaces.
        """

        return {}

    ## What the spec asked for

    def saveable(self) -> "Mapping[str, Saveable]":
        """:meth:`declare_saveable`, worked out once."""

        cache = self.__dict__.get("_saveable")
        if cache is None:
            cache = self.__dict__["_saveable"] = dict(self.declare_saveable())
        return cache

    def save_qualifier(self) -> str:
        """What this object's saved fields are filed under: ``<qualifier>.<field>``.

        Its :meth:`name` by default, which for a phase is the name the spec gave
        it and for the diffuse domain is the block it is declared in -- so the
        prefix on disk is the word a reader of the spec would look for. An owner
        whose name is not that word says what is.
        """

        return self.name()

    def requested(self) -> tuple[str, ...]:
        """What this object's own block asked to have written."""

        return tuple(self.parameters.save)

    def check_saveable_names(self, names: "Sequence[str] | None" = None) -> None:
        """Reject a save name nothing will answer to, before anything is built.

        Before, because otherwise it is found at the first frame, which can be an
        hour into a run. Checks this object's own request unless given another
        list.
        """

        offered = self.saveable()
        names = self.requested() if names is None else names
        unknown = [name for name in names if name not in offered]
        if unknown:
            raise ValueError(
                f"{self.name()} cannot save {', '.join(map(repr, unknown))}: "
                f"it offers {', '.join(sorted(offered)) or 'nothing'}"
            )

    def saved(self, name: str) -> "tuple[Any, Saveable]":
        """One save name's current value, and the declaration describing it.

        The whole declaration, so that what an owner can say about a field grows
        without changing this signature and every hop between here and the
        array. The value is the declared element's own function, unresampled, or
        whatever ``value`` returns for a derived entry.
        """

        entry = self.saveable()[name]
        if not entry.derived:
            return entry.element.function("next"), entry
        return entry.value(), entry


def own_elements(owner: "ElementOwner") -> dict[str, Saveable]:
    """Every bulk element an owner declares, stored as its own dofs.

    The one-to-one part of a declaration: the values are already in a space, so
    that space is where they go.

    A free function rather than a base-class default because the owners are
    parametric and element-owning by two separate inheritances, and a method
    resolved across both would depend on which one a subclass happened to list
    first. Called explicitly, it does not.

    Surface elements are left out: they are the boundary multipliers, which live
    on a submesh the saved mesh does not include, so there is nothing to index
    their dofs by.
    """

    return {
        spec.name: Saveable(spec)
        for spec in owner.element_specs()
        if spec.domain is ElementDomain.BULK
    }


def qualified(owner: "ElementOwner | ParametricSaveable", name: str) -> str:
    """How the archive files one of an owner's values: ``<qualifier>.<name>``.

    One function because what a run writes, what a resume demands and what a
    resume reads back have to agree exactly. The qualifier is not necessarily
    the owner's :meth:`name`: the mixture files its own fields under ``system``,
    so a reader finds them at one name whichever mixture wrote them, and a caller
    reaching for the name instead looks for ``polymerizing_b.pressure`` and finds
    nothing.

    An owner with no qualifier falls back to its name, which nothing on disk will
    match. That is the right answer rather than a guess: an owner that cannot be
    saved cannot be restored either, and reporting its fields as missing is how a
    resume says so.
    """

    qualifier = getattr(owner, "save_qualifier", None)
    prefix = qualifier() if callable(qualifier) else owner.name()
    return f"{prefix}.{name}"
