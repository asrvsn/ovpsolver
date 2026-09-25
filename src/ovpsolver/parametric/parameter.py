"""What a single parameter is allowed to be.

Each class here is one kind of declaration. Writing ``kappa: float = Positive()``
in a parameters class says three things at once: the spec must supply a kappa,
it has to be a positive number, and reading ``self.kappa`` later gives back the
number rather than the declaration. The last is the descriptor protocol, and it
is what lets a parameters class carry ordinary properties and methods computed
from its own values.

Numbers are coefficients, not literals
--------------------------------------
Once the run has a mesh (:func:`bind_form_mesh`), reading ``self.kappa`` gives
back a :class:`dolfinx.fem.Constant` rather than a Python float. FFCx caches
compiled kernels by form signature, and a literal float is part of the
signature: with literals, changing one number in a spec recompiles every form
that number reaches -- for the numbers in the diffuse geometry, every form there
is -- and a run that should start in seconds spends minutes in the C compiler.

A number the spec sets to exactly zero reads back as UFL ``Zero`` instead, which
annihilates the product it is written into while the form is being built, so
the term never reaches the compiler or a quadrature point. The price is one
recompile when a coefficient is set to or from zero. Such a term needs no
``if coefficient == 0.0`` guard, and one does no harm: ``Zero == 0.0`` is true
and a live ``Constant`` compares false.

What this asks of a reader is that a number used as a *number* -- a mesh length,
a step, anything that reaches Python arithmetic or a comparison -- be spelled
``float(...)``, which works on a ``Constant`` and on ``Zero`` alike. Ordered
comparisons need it most: UFL reads ``<`` as building a condition and refuses to
collapse it to a bool, so ``if float(x) <= 0.0`` is the spelling, while ``==``
against zero needs nothing.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import ufl
from dolfinx import fem

from . import pairs

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh

T = TypeVar("T")

#: What a zero-valued number reads back as. One instance serves, since UFL's zero
#: is immutable.
ZERO = ufl.as_ufl(0.0)

#: The mesh numeric parameters bind their constants to, or ``None`` before the
#: run has one. Module-level because the parameters tree is read before any mesh
#: exists -- the mesh is built *from* some of these numbers -- so a parameter
#: cannot ask the tree for one.
_form_mesh: Mesh | None = None


def bind_form_mesh(mesh: Mesh | None) -> None:
    """Bind numeric parameters to the mesh whose forms will carry them.

    Before this, a number reads back as a plain float, which is what building
    the mesh from ``mesh_h`` and ``domain_eps`` needs. Binding a second mesh -- a
    reader opening a finished run beside a live one -- makes each block rebuild
    its constants the first time it is read against it, so a form is never
    handed a coefficient from another mesh.
    """

    global _form_mesh
    _form_mesh = mesh


class _Required:
    """Sentinel for a declaration with no default: the spec has to say."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<required>"


REQUIRED = _Required()


class Parameter(Generic[T]):
    """One declared value.

    Subclasses implement :meth:`coerce`, which checks and converts a value and is
    handed its dotted path in the spec, so that an error names where it came from.
    """

    #: What this kind of value is, for error messages.
    kind = "value"

    def __init__(self, default: Any = REQUIRED, *, optional: bool = False) -> None:
        if optional and default is REQUIRED:
            default = None
        self.optional = optional
        self.name = type(self).__name__
        self.default = (
            default
            if default is REQUIRED or default is None
            else self.coerce(default, f"the default for this {self.kind}")
        )

    ## Descriptor protocol

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        try:
            return instance._values[self.name]
        except KeyError:
            raise AttributeError(
                f"{instance.where}.{self.name} is required and was never given "
                f"a value"
            ) from None

    def __set__(self, instance: Any, value: Any) -> None:
        instance._values[self.name] = self.read(
            value, f"{instance.where}.{self.name}"
        )

    ## Public

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def read(self, node: Any, where: str, *, strict: bool = False) -> Any:
        """Turn one node of a spec document into the value to store.

        ``strict``, whether a document must mention every declaration, is settled
        by the block doing the reading before it gets here. It is accepted so that
        a block can pass it down uniformly, and matters only to declarations that
        contain other declarations.
        """

        if node is None:
            if self.optional:
                return None
            raise ValueError(f"{where} must not be null")
        return self.coerce(node, where)

    def fresh_default(self) -> Any:
        """The default, copied so that no two blocks share a mutable one."""

        return copy.copy(self.default)

    def coerce(self, node: Any, where: str) -> Any:
        """Check a non-null node and convert it to the value to store."""

        raise NotImplementedError


class Number(Parameter[float]):
    """A real number, optionally confined to an interval.

    Reads back as a coefficient rather than as the number itself; see the module
    docstring for why, and for what that asks of a call site.
    """

    kind = "number"

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        minimum: float = -math.inf,
        maximum: float = math.inf,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
        optional: bool = False,
        coefficient: bool = True,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.exclusive_maximum = exclusive_maximum
        #: Whether this number is a coefficient of a form -- the default, a
        #: material parameter -- or a number the program computes with: a step
        #: length, a tolerance, a mesh spacing. Those are read by Python and never
        #: by a form, so they stay floats that can be compared and computed with
        #: as they are.
        self.coefficient = coefficient
        super().__init__(default, optional=optional)

    ## Overrides

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        """This number as a form reads it: ``Zero``, a ``Constant``, or a float.

        The ``Constant`` is built once per block and mesh and reused, so that
        every form written against this parameter shares one coefficient and one
        signature. Before a mesh is bound there is nothing to build it on, and the
        float is the right answer: the only readers that early are the ones
        building the geometry.
        """

        if instance is None:
            return self
        value = super().__get__(instance, owner)
        # A prototype's slot still holding a declaration, and an optional number
        # the spec left null, are returned as they are.
        if value is None or isinstance(value, Parameter) or not self.coefficient:
            return value
        if value == 0.0:
            return ZERO
        mesh = _form_mesh
        if mesh is None:
            return value
        cache = instance.__dict__.setdefault("_constants", {})
        bound, constant = cache.get(self.name, (None, None))
        if bound is not mesh:
            constant = fem.Constant(mesh, np.float64(value))
            cache[self.name] = (mesh, constant)
        return constant

    def __set__(self, instance: Any, value: Any) -> None:
        super().__set__(instance, value)
        # A number set again -- a default settled by ``derive``, say -- must not
        # keep the constant built from its previous value, or a form would
        # silently carry a stale number.
        instance.__dict__.get("_constants", {}).pop(self.name, None)

    def coerce(self, node: Any, where: str) -> float:
        if isinstance(node, bool) or not isinstance(node, (int, float, np.floating)):
            raise TypeError(f"{where} must be a number; got {node!r}")
        value = float(node)
        if not math.isfinite(value):
            raise ValueError(f"{where} must be finite; got {value!r}")
        low, high = self.minimum, self.maximum
        if (
            value < low
            or value > high
            or (self.exclusive_minimum and value == low)
            or (self.exclusive_maximum and value == high)
        ):
            raise ValueError(f"{where} must be {self.interval()}; got {value!r}")
        return value

    ## Public

    def interval(self) -> str:
        """The interval this number is confined to, in words, for an error."""

        low, high = self.minimum, self.maximum
        if low == 0.0 and self.exclusive_minimum and high == math.inf:
            return "positive"
        if low == 0.0 and not self.exclusive_minimum and high == math.inf:
            return "non-negative"
        pieces = []
        if low != -math.inf:
            pieces.append(f"{'>' if self.exclusive_minimum else '>='} {low:g}")
        if high != math.inf:
            pieces.append(f"{'<' if self.exclusive_maximum else '<='} {high:g}")
        return " and ".join(pieces) if pieces else "finite"


class Positive(Number):
    """A number strictly greater than zero."""

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        optional: bool = False,
        coefficient: bool = True,
    ) -> None:
        super().__init__(
            default,
            minimum=0.0,
            exclusive_minimum=True,
            optional=optional,
            coefficient=coefficient,
        )


class Nonnegative(Number):
    """A number at or above zero."""

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        optional: bool = False,
        coefficient: bool = True,
    ) -> None:
        super().__init__(
            default, minimum=0.0, optional=optional, coefficient=coefficient
        )


class Fraction(Number):
    """A number strictly between zero and one."""

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        optional: bool = False,
        coefficient: bool = True,
    ) -> None:
        super().__init__(
            default,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
            exclusive_maximum=True,
            optional=optional,
            coefficient=coefficient,
        )


class Integer(Parameter[int]):
    """A whole number, optionally confined to an interval."""

    kind = "integer"

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        optional: bool = False,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(default, optional=optional)

    def coerce(self, node: Any, where: str) -> int:
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
            raise TypeError(f"{where} must be an integer; got {node!r}")
        value = int(node)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{where} must be at least {self.minimum}; got {value}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{where} must be at most {self.maximum}; got {value}")
        return value


class Count(Integer):
    """A whole number of things, at least one."""

    def __init__(self, default: Any = REQUIRED, *, optional: bool = False) -> None:
        super().__init__(default, minimum=1, optional=optional)


class Index(Integer):
    """A whole number of things, possibly none."""

    def __init__(self, default: Any = REQUIRED, *, optional: bool = False) -> None:
        super().__init__(default, minimum=0, optional=optional)


class Boolean(Parameter[bool]):
    """A flag."""

    kind = "flag"

    def coerce(self, node: Any, where: str) -> bool:
        if not isinstance(node, bool):
            raise TypeError(f"{where} must be true or false; got {node!r}")
        return node


class Text(Parameter[str]):
    """A non-empty string, optionally from a fixed set."""

    kind = "string"

    def __init__(
        self,
        default: Any = REQUIRED,
        *,
        choices: SequenceABC[str] | None = None,
        optional: bool = False,
    ) -> None:
        self.choices = tuple(choices) if choices is not None else None
        super().__init__(default, optional=optional)

    def coerce(self, node: Any, where: str) -> str:
        if not isinstance(node, str) or not node:
            raise ValueError(f"{where} must be a non-empty string; got {node!r}")
        if self.choices is not None and node not in self.choices:
            raise ValueError(
                f"{where} must be one of {', '.join(self.choices)}; got {node!r}"
            )
        return node


class Names(Parameter[tuple]):
    """An ordered list of distinct names, such as the fields a phase saves."""

    kind = "list of names"

    def __init__(self, default: Any = (), *, optional: bool = False) -> None:
        super().__init__(default, optional=optional)

    def coerce(self, node: Any, where: str) -> tuple[str, ...]:
        if isinstance(node, str) or not isinstance(node, SequenceABC):
            raise TypeError(f"{where} must be a list of names; got {node!r}")
        names = []
        for position, entry in enumerate(node):
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"{where}[{position}] must be a non-empty string; got {entry!r}"
                )
            names.append(entry)
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"{where} repeats {', '.join(sorted(duplicates))}")
        return tuple(names)


class Options(Parameter[dict]):
    """A free-form mapping passed straight through, such as PETSc options.

    Nothing is checked beyond its keys being strings, because what is valid is
    decided by the library it is passed to.
    """

    kind = "mapping"

    def __init__(self, default: Any = None, *, optional: bool = False) -> None:
        super().__init__({} if default is None else default, optional=optional)

    def coerce(self, node: Any, where: str) -> dict[str, Any]:
        if not isinstance(node, MappingABC):
            raise TypeError(f"{where} must be a mapping; got {node!r}")
        for key in node:
            if not isinstance(key, str):
                raise TypeError(f"{where} keys must be strings; got {key!r}")
        return dict(node)


class Pairs(Parameter[dict]):
    """A mapping keyed by a pair of names, written ``(a, b)``.

    Read by :func:`~ovpsolver.parametric.pairs.read_pairs`. Whether the names are
    phases of the mixture is checked by whoever knows the roster, which is not
    this declaration.
    """

    kind = "mapping keyed by pairs"

    def __init__(
        self,
        value: Parameter,
        default: Any = None,
        *,
        optional: bool = False,
    ) -> None:
        #: What each entry is read against.
        self.value = value
        super().__init__({} if default is None else default, optional=optional)

    def coerce(self, node: Any, where: str) -> dict[tuple[str, str], Any]:
        if not isinstance(node, MappingABC):
            raise TypeError(f"{where} must be a mapping; got {node!r}")
        return pairs.read_pairs(node, where, self.value.read)
