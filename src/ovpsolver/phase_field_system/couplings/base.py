"""What couples phase fields to each other, declared the way a phase field is.

A :class:`Coupling` is a term of the Rayleighian that belongs to no single phase.
Flory-Huggins is the plain case -- a bilinear form in every phase's composition,
owning no state at all -- and inter-gel locking is the other extreme: two networks
that have hooked into each other share a modulus and a relative strain moment that
belong to neither of them, evolve by their own law, and push back on both.

Nothing structural separates the two, so they are the same kind of object. A
coupling may declare an energy, a dissipation, a constraint, fields of its own and
a transport law for them, each the way a phase does: as a statement, never as a
row. Which rows a statement reaches is settled by the differentiation handles it
is written in, so a coupling writes into a phase's rows by naming that phase's
variables and the system never routes anything.

Parametric: the ``couplings`` block of a spec has one entry per coupling the
mixture declares, each read against that coupling's own parameters class, and
:meth:`~ovpsolver.phase_field_system.system.PhaseFieldSystem.make_couplings`
constructs each from its block, as ``make_phase_fields`` does a phase from its
roster entry.

The phases are handed over at construction because a coupling is *about* them: a
cross interaction writes in their handles, and a drag reads their velocities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...fem.elements import ElementOwner
from ...parametric import Parametric
from ...parametric.parameters import ParametersT
from ..dissipative import Dissipative

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...diffuse_domain import DiffuseDomain
    from ...fem.elements import ElementSpec
    from ...solver.parameters import SolverParameters
    from ..phase_field import PhaseField
    from ..transport import Transported


class Coupling(Dissipative, ElementOwner, Parametric[ParametersT]):
    """One cross-phase interaction, declared the way a phase field is."""

    def __init__(
        self,
        solver_parameters: "SolverParameters",
        parameters: ParametersT,
        diffuse_domain: "DiffuseDomain",
        phase_fields: "Sequence[PhaseField]",
        k_B_T: float,
    ) -> None:
        super().__init__(solver_parameters, parameters, diffuse_domain)
        #: Every phase of the mixture, in roster order, which is the order every
        #: matrix a coupling contracts against is indexed in.
        self.phase_fields = tuple(phase_fields)
        #: The scale every declared energy is measured in, which belongs to the
        #: mixture and which a coupling has no other way to reach.
        self.k_B_T = k_B_T

    def transported(self) -> list["Transported"]:
        """The coupling's own fields, as the variables that carry them.

        Each is a :class:`~ovpsolver.phase_field_system.transport.Transported`,
        as a phase's internal states are, so its row, its step bound and its
        transport step are the machinery every other state goes through. A field
        that obeys no law a flux list can state overrides the row, as a relative
        strain moment between two deforming networks does. Nothing, for a
        coupling that owns no fields.
        """

        return []

    def irreversible_timestep(self, dt: float) -> None:
        """Advance whatever this coupling carries outside the OVP step.

        Nothing by default. A coupling with chemistry in it says so.
        """

    def declare_elements(self) -> list["ElementSpec"]:
        """Fields of the coupling's own; none by default."""

        return []

    def set_initial_conditions(self) -> None:
        """Put this coupling's own fields at ``t = 0``.

        Nothing for a coupling that owns none, the ordinary case. Called after
        the phases are set, so a coupling whose initial value depends on theirs
        can read them.
        """
