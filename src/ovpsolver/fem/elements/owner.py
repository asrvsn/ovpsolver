"""The objects that declare finite elements, and hold what the solver binds.

An :class:`ElementOwner` states its elements in :meth:`ElementOwner.declare_elements`
and is handed back their functions: the solver builds each space once and binds
the functions onto the owner's ``rates``/``prev``/``next`` bundles, and the test
function of each row onto the attribute the element names. Everything that needs
those functions in hand -- looking a declaration up by name, the differentiation
handles energy terms are written in, rotating the time levels, rolling a step
back, resuming -- is here, so a phase, a coupling and the mixture itself do it the
same way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

import ufl

from .spec import ElementSpec, StaticElement

if TYPE_CHECKING:
    import numpy as np
    from dolfinx.fem import Function
    from ufl.core.expr import Expr

    from ...diffuse_domain import DiffuseDomain
    from ...parametric import Parameters
    from ...solver.parameters import SolverParameters


class Fields:
    """Mutable namespace holding one time level, or the rates, by attribute."""


#: What :meth:`ElementOwner.checkpoint` hands back and :meth:`ElementOwner.restore`
#: takes: per owner, a detached copy of every function at every time level.
#: Opaque to whoever holds one, whose only use for it is to give it back.
Checkpoint = list[dict[str, dict[str, "np.ndarray"]]]


class ElementOwner(ABC):
    """Something that declares finite elements and holds what the solver binds.

    The shared machinery and deliberately no more. There is no tree and no
    recursion: the solver knows a mixture is a system, its phases and their
    couplings, so the structure is in the types, and an owner is responsible for
    its own functions and for nothing else's.

    It holds the context every declaration is written against -- the mesh and
    step (``solver_parameters``) and the diffuse geometry -- and adds the
    bookkeeping that needs the functions in hand: looking a declaration up by
    name, handing out differentiation handles, rotating the time levels. What the
    material is belongs to :class:`~ovpsolver.parametric.Parametric`, which every
    concrete owner also derives from; nothing abstract here reads
    ``self.parameters``.
    """

    def __init__(
        self,
        solver_parameters: "SolverParameters",
        parameters: "Parameters",
        diffuse_domain: "DiffuseDomain",
    ) -> None:
        self.solver_parameters = solver_parameters
        self.parameters = parameters
        self.diffuse_domain = diffuse_domain
        self.prev = Fields()
        self.next = Fields()
        self.rates = Fields()

    @abstractmethod
    def name(self) -> str:
        """Unique within the mixture; names this owner's functions."""

    @abstractmethod
    def declare_elements(self) -> list[ElementSpec]:
        """Every finite element this owner declares, static and dynamic.

        Written without naming the owner: :meth:`element_specs` stamps that on.
        """

    def element_specs(self) -> tuple[ElementSpec, ...]:
        """The declarations, stamped with this owner and built once.

        Once, so that the spec the solver builds a function space from is the
        same object a transported variable holds, and identity is a usable
        question. A declaration is a statement about the material, not the state,
        so nothing here can change after construction anyway.
        """

        cached = self.__dict__.get("_element_specs")
        if cached is None:
            cached = tuple(
                replace(spec, owner=self) for spec in self.declare_elements()
            )
            self.__dict__["_element_specs"] = cached
        return cached

    def element_owners(self) -> tuple["ElementOwner", ...]:
        """Everything the solver builds functions for, this owner first.

        Just this owner, unless it holds others. A flat list rather than a tree:
        an owner that holds others knows exactly which.
        """

        return (self,)

    ## Reading the declarations back

    def element(self, name: str) -> ElementSpec:
        """This owner's declaration of one variable, by name.

        The spec rather than the function, because a caller holding it can reach
        the functions, the test function and the mesh, and can also ask what was
        declared -- which degree, which domain, which solve.
        """

        cache = self.__dict__.setdefault("_elements", {})
        if not cache:
            cache.update({spec.name: spec for spec in self.element_specs()})
        if name not in cache:
            raise KeyError(f"{self.name()} declares no element {name!r}")
        return cache[name]

    def fields_at(self, time_level: str) -> Fields:
        if time_level == "next":
            return self.next
        if time_level == "prev":
            return self.prev
        raise ValueError("time_level must be 'prev' or 'next'")

    def variable(self, name: str, *, time_level: str = "next") -> "Expr":
        """A differentiation handle on one of this owner's variables.

        Energy terms are declared as expressions in these wrappers so their
        potentials can be recovered by :func:`ufl.diff`. The wrapper is memoized
        per ``(name, time_level)``, which is what makes that work across
        declarations: a Flory-Huggins term written by the system and a bulk term
        written by the owner differentiate against the same object, and so
        contribute to the same row.
        """

        cache = self.__dict__.setdefault("_variables", {})
        key = (name, time_level)
        if key not in cache:
            cache[key] = ufl.variable(getattr(self.fields_at(time_level), name))
        return cache[key]

    ## Time levels

    def lagged_rate_specs(self) -> tuple[StaticElement, ...]:
        """Rates that also keep their last solved value as a coefficient."""

        return tuple(
            spec
            for spec in self.element_specs()
            if isinstance(spec, StaticElement) and spec.store_lagged
        )

    def snapshot(self) -> None:
        """``prev <- next``, and the lagged rates <- the solved rates.

        After this, ``prev`` holds the pre-step state while the step advances
        ``next`` in place, so the two bracket the increment the convex-split
        energy consumes.
        """

        for owner in self.element_owners():
            for name, value in vars(owner.next).items():
                getattr(owner.prev, name).x.array[:] = value.x.array
        self.lag_rates()

    def lag_rates(self) -> None:
        """The lagged rates <- the solved rates, leaving the state alone.

        The half of :meth:`snapshot` that moves the upwind selectors, and what a
        rejected attempt wants: its velocity is the best estimate of the one the
        retry will find, while the state it produced is being thrown away. It
        also makes the step bound select on the solved velocity, which is the
        bound a genuinely upwinded row would have reported.
        """

        for owner in self.element_owners():
            for spec in owner.lagged_rate_specs():
                getattr(owner.prev, spec.name).x.array[:] = getattr(
                    owner.rates, spec.name
                ).x.array

    def zero_lagged_rates(self) -> None:
        """The lagged rates <- zero, for a run that starts from rest."""

        for owner in self.element_owners():
            for spec in owner.lagged_rate_specs():
                getattr(owner.prev, spec.name).x.array[:] = 0.0

    def lag_to_live(self) -> "dict[Function, Function]":
        """Each variable's ``k`` function paired with the ``k+1`` one it becomes.

        The pairing :meth:`snapshot` will make, named before it is made: an
        expression written in lagged handles holds no reference to the level it
        is compared against, so the map has to come from whoever knows both. Read
        off the declarations, so it is the same map whatever a caller mentions.
        Rates have one time level and are not in it.
        """

        return {
            getattr(owner.prev, spec.name): getattr(owner.next, spec.name)
            for owner in self.element_owners()
            for spec in owner.element_specs()
            if not isinstance(spec, StaticElement)
        }

    def checkpoint(self) -> "Checkpoint":
        """Detached copy of every function held, for rolling back a step."""

        return [
            {
                level: {
                    name: value.x.array.copy()
                    for name, value in vars(getattr(owner, level)).items()
                }
                for level in ("prev", "next", "rates")
            }
            for owner in self.element_owners()
        ]

    def restore(self, checkpoint: "Checkpoint", *, keep_rates: bool = False) -> None:
        """Put every function back as it was, optionally sparing the rates.

        ``keep_rates`` is what a rejected step wants: its state was computed at
        an inadmissible step size and has to go, but its rates are the best
        estimate of the velocities the retry will find. Leaving them lets the
        next :meth:`snapshot` lag them, so the retry selects on the attempt's own
        answer rather than on the state before it.
        """

        levels = ("prev", "next") if keep_rates else ("prev", "next", "rates")
        for owner, saved in zip(self.element_owners(), checkpoint, strict=True):
            for level in levels:
                bundle = getattr(owner, level)
                for name, values in saved[level].items():
                    getattr(bundle, name).x.array[:] = values

    ## Resuming

    def resume_specs(self) -> tuple[ElementSpec, ...]:
        """Every declaration a resumed run has to read back, over the whole tree.

        The state elements, and the rates that keep a lagged copy -- the
        velocities: :meth:`snapshot` opens a step by lagging them into the bundle
        the upwind selectors read, and a velocity left at zero would make every
        facet select ``ge(0, 0)``, true, so every one takes its ``+`` side. A rate
        with no lagged copy is a Newton iterate the next solve replaces, and a
        fresh function already holds the zero it would be given. Nor is the
        ``prev`` half of anything here: the snapshot overwrites it before a row
        reads it.

        Over :meth:`element_owners`, like :meth:`snapshot`, so asking the mixture
        gives the whole run's requirement in one call.
        """

        return tuple(
            spec
            for owner in self.element_owners()
            for spec in owner.element_specs()
            if not isinstance(spec, StaticElement) or spec.store_lagged
        )

    def resume_from(self, read) -> None:
        """Put every value in :meth:`resume_specs` back from a saved frame.

        ``read(function, spec)`` fills one function from the saved value that
        declaration stands for; where that is kept and what it is called on disk
        are the caller's business. The alternative to
        :meth:`~ovpsolver.phase_field_system.PhaseFieldSystem.set_initial_conditions`
        and not a correction applied after it, so a mixture that gains state is
        resumable by having declared it.

        Both time levels are filled: ``prev`` is overwritten by the next
        :meth:`snapshot` before any row reads it, so writing it costs a copy and
        spares a reader wondering which half is live.
        """

        for spec in self.resume_specs():
            owner = spec.owner
            if isinstance(spec, StaticElement):
                read(getattr(owner.rates, spec.name), spec)
                continue
            for level in ("prev", "next"):
                read(getattr(getattr(owner, level), spec.name), spec)
