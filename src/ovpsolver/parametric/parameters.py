"""Parameter blocks, and the objects built from them.

A :class:`Parameters` class is a schema and a value at once. Written in a class
body it declares a sub-block; read from a document it comes back populated, with
every declaration resolved to the value behind it. It derives from
:class:`~ovpsolver.parametric.parameter.Parameter` so that nesting needs no
separate concept: a block is one more kind of declared value.

So the document and the parameters tree have the same shape, and there is no
table from a word in the spec to a class to instantiate, because the class was
chosen when the class body was written::

    class MyMixtureParameters(ModelBParameters):
        phase_fields = ParametersList(MyPhaseFieldParameters(), minimum=1)

reads every entry of the document's ``phase_fields`` against exactly that class,
which is what lets a phase field written elsewhere work without this package
knowing its name.

The geometry is the one thing not declared this way: an arrangement of
inclusions is not a claim about dynamics, so it arrives as a file the spec names
rather than as a class.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Generic, TypeVar, get_args, get_origin

from . import pairs
from .parameter import REQUIRED, Parameter


class Parameters(Parameter):
    """A named group of declarations, and the values read into them."""

    kind = "block"

    #: The :class:`Parametric` class this block describes, set by the class
    #: statement that names both. It is the whole of the registry.
    parametric: type | None = None

    def __init__(self, **values: Any) -> None:
        self._initialize(type(self).__name__, frozenset(values), type(self).__name__)

        declarations = self.declarations()
        unknown = set(values) - set(declarations)
        if unknown:
            raise TypeError(
                f"{self._where} got unexpected parameter(s) "
                f"{', '.join(sorted(unknown))}; expected "
                f"{', '.join(sorted(declarations))}"
            )
        for field_name, declaration in declarations.items():
            if field_name in values:
                supplied = values[field_name]
                if isinstance(supplied, Parameter):
                    # A declaration in place of a value: a prototype naming the
                    # concrete class for this slot.
                    self._values[field_name] = supplied
                else:
                    setattr(self, field_name, supplied)
            elif isinstance(declaration, Parameters):
                if declaration.defaultable:
                    self._values[field_name] = declaration.read(
                        {}, f"{self._where}.{field_name}"
                    )
            elif not declaration.required:
                self._values[field_name] = declaration.fresh_default()
        # Only a block that was told something settles. ``Cls()`` is the
        # prototype form a class body writes, and settling one would run a
        # mixture's checks at import time.
        if values and self.complete:
            self.settle()

    ## Overrides

    def read(self, node: Any, where: str, *, strict: bool = False) -> Parameters:
        """Read a mapping into a new block of this class, with this one as prototype.

        ``strict`` demands that the document answer every declaration, defaults
        included, and is how a spec is read off disk: the file should be a
        complete record of its run, readable without the version of the source
        whose defaults would otherwise fill its silences, so that an unchanged
        file cannot change meaning when a default does. Blocks built in Python
        are read leniently, since there the code is its own record.
        """

        if isinstance(node, Parameters):
            # A block handed in replaces this declaration outright: that is how a
            # prototype names the concrete class for a slot. It need not subclass
            # the class body's default; the two may be siblings.
            return node
        if node is None:
            node = {}
        if not isinstance(node, MappingABC):
            raise TypeError(f"{where} must be a mapping; got {node!r}")

        declarations = self.declarations()
        unknown = set(node) - set(declarations)
        if unknown:
            raise ValueError(
                f"{where} has unknown parameter(s) {', '.join(sorted(unknown))}; "
                f"expected {', '.join(sorted(declarations))}"
            )

        block = object.__new__(type(self))
        block._initialize(where, frozenset(node), self.name)
        for field_name in declarations:
            child_where = f"{where}.{field_name}"
            declaration = self.declaration(field_name)
            if isinstance(declaration, Parameters):
                if strict and field_name not in node:
                    raise ValueError(
                        f"{child_where} is missing. Every block has to appear in "
                        "a spec, so that the file says what was configured even "
                        "where the answer is the default. Write it out, as "
                        f"`{field_name}:` with its own parameters under it"
                    )
                # Read whether or not the document has it: leniently its defaults
                # apply, and strictly it is present.
                value = declaration.read(
                    node.get(field_name), child_where, strict=strict
                )
            elif field_name in node:
                value = declaration.read(
                    node[field_name], child_where, strict=strict
                )
            elif _answered(self._values.get(field_name, _MISSING)) and (
                not strict or field_name in self._given
            ):
                # Answered by the prototype. Strictly, only if the prototype was
                # *given* it: a blank prototype holds the default of every slot
                # that has one, and honouring those would make strict lenient.
                value = self._values[field_name]
            elif declaration.required:
                raise ValueError(
                    f"{child_where} is required: no default for this "
                    f"{declaration.kind}"
                )
            elif strict:
                raise ValueError(
                    f"{child_where} is missing. Every parameter has to appear in "
                    "a spec, so that the file is a complete record of the run "
                    f"and not a partial one completed by this version's "
                    f"defaults. Write `{field_name}: "
                    f"{_as_written(declaration.fresh_default())}` to keep the "
                    "behaviour it would have had"
                )
            else:
                value = declaration.fresh_default()
            _adopt(value, block)
            block._values[field_name] = value
        block.settle()
        return block

    def coerce(self, node: Any, where: str) -> Parameters:
        return self.read(node, where)

    def fresh_default(self) -> Any:
        return self.read({}, self._where)

    def __repr__(self) -> str:
        if not self.complete:
            return f"{type(self).__name__}(<prototype>)"
        shown = ", ".join(
            f"{name}={value!r}"
            for name, value in list(self._values.items())[:4]
            if not isinstance(value, Parameters)
        )
        return f"{type(self).__name__}({shown})"

    ## Public

    @classmethod
    def declarations(cls) -> dict[str, Parameter]:
        """Every declaration this class has, with subclasses overriding bases.

        A subclass that rebinds an inherited name to something that is not a
        declaration withdraws it, which is how a mixture that sorts its phases
        into typed rosters drops the untyped one it inherited.
        """

        found: dict[str, Parameter] = {}
        for ancestor in reversed(cls.__mro__):
            for field_name, value in vars(ancestor).items():
                if isinstance(value, Parameter):
                    found[field_name] = value
                else:
                    found.pop(field_name, None)
        return found

    @property
    def complete(self) -> bool:
        """Whether every declaration has been answered.

        An incomplete block is a prototype, as a class body holds it, and is not
        usable as a value.
        """

        return all(
            _answered(self._values.get(field_name, _MISSING))
            for field_name in self.declarations()
        )

    @property
    def defaultable(self) -> bool:
        """Whether this block could be read from a document that omits it.

        True when the prototype has already answered everything required.
        """

        for field_name in self.declarations():
            declaration = self.declaration(field_name)
            if isinstance(declaration, Parameters):
                if not declaration.defaultable:
                    return False
            elif _answered(self._values.get(field_name, _MISSING)):
                continue
            elif declaration.required:
                return False
        return True

    def declaration(self, field_name: str) -> Parameter:
        """What this block reads ``field_name`` against.

        Whatever the prototype was handed for the slot, if that was itself a
        declaration; otherwise what the class body said.
        """

        override = self._values.get(field_name)
        if isinstance(override, Parameter):
            return override
        return self.declarations()[field_name]

    @property
    def where(self) -> str:
        """Where in the document this block came from, for error messages."""

        return self._where

    @property
    def parent(self) -> Parameters | None:
        """The block this one was read as part of, if any."""

        return self._parent

    @property
    def given(self) -> frozenset[str]:
        """Which names the document actually supplied, as against defaulted.

        For where a default is indistinguishable from an answer and the
        difference matters: a coefficient omitted because it does not apply,
        against one set to zero on purpose.
        """

        return self._given

    def settle(self: ParametersT) -> ParametersT:
        """Finish the block once every value is in.

        Runs :meth:`derive` and then :meth:`validate`, so that a check may look
        at anything derived; subclasses override those two rather than this.
        Reading settles the block it returns, and so does construction with
        keywords that complete it. A block completed any other way is settled by
        calling this.
        """

        self.derive()
        self.validate()
        return self

    def derive(self) -> None:
        """Compute whatever this block knows how to work out for itself."""

    def validate(self) -> None:
        """Reject combinations no single declaration could have caught."""

    def build(self, **context: Any) -> Any:
        """Construct the object these parameters describe."""

        target = type(self).parametric
        if target is None:
            raise TypeError(
                f"{type(self).__name__} describes no object: no class declares "
                f"Parametric[{type(self).__name__}]"
            )
        return target(parameters=self, **context)

    ## Private helpers

    def _initialize(self, where: str, given: frozenset[str], name: str) -> None:
        """Set the state every block carries.

        :meth:`read` makes its blocks without ``__init__``, which would fill in
        every default the document is about to answer, and sets this directly.
        """

        self._values: dict[str, Any] = {}
        self._where = where
        self._parent: Parameters | None = None
        self._given = given
        self.optional = False
        self.default = None
        self.name = name


class PairsParameters(Parameters):
    """A block whose entries are keyed by a pair of names, not by declarations.

    For what a mixture states per unordered pair of phases -- a Flory-Huggins
    interaction, a drag -- where how many entries there are is a property of the
    roster rather than of the schema::

        flory_huggins:
          (water, lipid):
            chi: 5.0

    A block, unlike a :class:`~ovpsolver.parametric.parameter.Pairs`
    declaration, so that a coupling configured by pairs describes and builds a
    :class:`Parametric` the way every other block does.

    Subclasses name :attr:`item`, and check the pairs against the roster
    themselves, since the roster is not in this block.
    """

    kind = "block keyed by pairs"

    #: The class each pair's value is read against. The class and not an
    #: instance, because an instance in a class body is how a declaration is
    #: written: it would be read as a key of this block named ``item``.
    item: type[Parameters]

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        #: Pair to value, in the order the document wrote them.
        self.pairs: dict[tuple[str, str], Any] = {}

    ## Overrides

    def read(self, node: Any, where: str, *, strict: bool = False) -> Parameters:
        if isinstance(node, Parameters):
            return node
        if node is None:
            node = {}
        if not isinstance(node, MappingABC):
            raise TypeError(
                f"{where} must be a mapping keyed by pairs, written `(a, b):`; "
                f"got {node!r}"
            )

        block = object.__new__(type(self))
        block._initialize(where, frozenset(), self.name)
        # Entries are read leniently even from a strict spec. Strictly, every pair
        # would owe every coefficient, including those that cannot apply to it,
        # and which can is a question about the roster. So completeness is
        # checked by the block that has the roster; see the ``applicable`` hook of
        # the Flory-Huggins block.
        block.pairs = pairs.read_pairs(node, where, self.item().read)
        block.settle()
        return block

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.pairs)} pairs)"


#: What ``_values.get`` answers for a declaration nothing is stored under.
_MISSING = object()


def _answered(value: Any) -> bool:
    """Whether a stored entry is a value rather than a declaration awaiting one."""

    if value is _MISSING:
        return False
    if isinstance(value, Parameters):
        return value.complete
    return not isinstance(value, Parameter)


def _as_written(value: Any) -> str:
    """A default rendered as the YAML that would reproduce it, to paste.

    ``None`` is the case worth naming: for an optional parameter it is how the
    document says *off*, and YAML spells it ``null``.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_as_written(entry) for entry in value) + "]"
    if isinstance(value, MappingABC):
        return "{}" if not value else "{...}"
    return repr(value)


def _adopt(value: Any, block: Parameters) -> None:
    """Make ``block`` the parent of ``value``, or of each block in a list of them."""

    if isinstance(value, Parameters):
        value._parent = block
    elif isinstance(value, tuple):
        for entry in value:
            if isinstance(entry, Parameters):
                entry._parent = block


class ParametersList(Parameter):
    """An ordered list of blocks, all read against the same class.

    How the tree gets its breadth: a mixture declares its list of phases once,
    and the document supplies as many as it likes, each an entry naming itself::

        phase_fields:
          - name: water
    """

    kind = "list of blocks"

    def __init__(
        self,
        item: Parameters,
        *,
        minimum: int = 0,
        default: Any = REQUIRED,
    ) -> None:
        #: The prototype every entry is read against.
        self.item = item
        self.minimum = minimum
        super().__init__(() if default is REQUIRED and minimum == 0 else default)

    ## Overrides

    def read(
        self, node: Any, where: str, *, strict: bool = False
    ) -> tuple[Parameters, ...] | None:
        """Read each entry against :attr:`item`, as strictly as the list itself."""

        if node is None:
            return super().read(node, where)
        if isinstance(node, tuple) and all(
            isinstance(entry, Parameters) for entry in node
        ):
            entries = node
        else:
            if isinstance(node, (str, MappingABC)) or not isinstance(
                node, SequenceABC
            ):
                raise TypeError(
                    f"{where} must be a list of {type(self.item).__name__} "
                    f"blocks; got {node!r}"
                )
            entries = tuple(
                self.item.read(entry, f"{where}[{position}]", strict=strict)
                for position, entry in enumerate(node)
            )
        if len(entries) < self.minimum:
            raise ValueError(
                f"{where} needs at least {self.minimum} "
                f"{'entry' if self.minimum == 1 else 'entries'}; got {len(entries)}"
            )
        return entries

    def coerce(self, node: Any, where: str) -> tuple[Parameters, ...]:
        return self.read(node, where)

    def fresh_default(self) -> Any:
        return () if self.default is REQUIRED else self.default


ParametersT = TypeVar("ParametersT", bound=Parameters)


class Parametric(Generic[ParametersT]):
    """An object described by a parameters block.

    Naming the schema among the bases ties the two together in both directions:
    the class gains a ``Parameters`` attribute naming its schema, and the schema
    a ``parametric`` attribute naming what to build. Nothing else is registered
    anywhere, so a class defined in a user's own script is as visible to the
    reader as one defined in this package.

    At the root of a hierarchy it comes first, and in a subclass of an already
    parametric class it comes last, because the subclass has to precede what it
    derives from::

        class PhaseField(Parametric[PhaseFieldParameters], ElementOwner, ABC):
        class CHPhaseField(PhaseField, Parametric[CHPhaseFieldParameters]):

    A subclass that names no schema of its own keeps its parent's, which is
    right for a variant that adds behaviour but no new numbers.
    """

    #: The block this class is described by, set by ``__init_subclass__``.
    Parameters: type[Parameters]

    parameters: ParametersT

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only the bases this class statement listed: ``__orig_bases__`` is an
        # ordinary attribute, so a subclass naming no schema would otherwise be
        # handed its parent's and look like a rival claim on it.
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            # Any parametric base, not ``Parametric`` alone: a refinement such as
            # ``ParametricSaveable[P]`` names a schema the same way.
            if not (isinstance(origin, type) and issubclass(origin, Parametric)):
                continue
            (declared,) = get_args(base)
            if isinstance(declared, TypeVar):
                # An intermediate generic class, passing the variable along.
                return
            if not (isinstance(declared, type) and issubclass(declared, Parameters)):
                raise TypeError(
                    f"{cls.__name__} declares Parametric[{declared!r}], which is "
                    f"not a Parameters subclass"
                )
            claimed = declared.__dict__.get("parametric")
            if claimed is not None and claimed is not cls:
                raise TypeError(
                    f"{declared.__name__} is already described by "
                    f"{claimed.__name__}; give {cls.__name__} its own parameters "
                    f"class so a document can tell them apart"
                )
            cls.Parameters = declared
            declared.parametric = cls
            return
