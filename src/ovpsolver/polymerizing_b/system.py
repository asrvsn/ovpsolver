"""Polymerizing Model B(+): the concrete mixture.

Two declarations and a chemical sub-step. The pressure, the saturation
constraint, the initial condition solve and the row assembly are the abstract
layer's; each phase supplies its own transport, friction, energy and
crosslinking; and the couplings supply the mixing energy and everything the
networks do to the rest of the mixture.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..fem.saveable import ParametricSaveable
from ..model_b.phase_field import CHPhaseField
from ..phase_field_system.couplings import Coupling
from ..phase_field_system.phase_field import PhaseField
from ..phase_field_system.system import PhaseFieldSystem
from .couplings import Drag, GelEntanglements, PolymerizingFloryHugginsCoupling
from .parameters import PolymerizingBParameters
from .phase_field import PolymerizingPhaseField


class PolymerizingB(PhaseFieldSystem, ParametricSaveable[PolymerizingBParameters]):
    """A saturated mixture of sol phases and crosslinking gel networks."""

    ## Construction

    def make_phase_fields(self) -> Sequence[PhaseField]:
        roster = self.parameters.phase_fields
        shared = (self.diffuse_domain, self.k_B_T)
        return (
            *(
                CHPhaseField(self.solver_parameters, phase, *shared)
                for phase in roster.sol
            ),
            *(
                PolymerizingPhaseField(self.solver_parameters, phase, *shared)
                for phase in roster.polymerizing
            ),
        )

    def make_couplings(self) -> Sequence[Coupling]:
        couplings = self.parameters.couplings
        shared = (self.diffuse_domain, self.phase_fields, self.k_B_T)
        return (
            PolymerizingFloryHugginsCoupling(
                self.solver_parameters, couplings.flory_huggins, *shared
            ),
            GelEntanglements(
                self.solver_parameters, couplings.gel_entanglements, *shared
            ),
            Drag(self.solver_parameters, couplings.drag, *shared),
        )

    def name(self) -> str:
        return "polymerizing_b"

    ## The irreversible half-step

    def irreversible_timestep(self, dt: float) -> None:
        """Advance the crosslinking chemistry one explicit Lie-split macro-step.

        Sub-cycled at the smallest of the gels' own bounds
        (:meth:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseField.gelation_max_dt`),
        and here rather than in any gel or coupling because every state has to
        advance by the same sub-step. Locking is born at a rate that multiplies
        one network's birth rate by the other's modulus, so it is evaluated at
        an instant every network is at: every coupling steps before any gel
        mutates the moduli it just read.
        """

        gels = tuple(
            field
            for field in self.phase_fields
            if isinstance(field, PolymerizingPhaseField)
        )
        if not gels:
            return

        chi_eps_finite_volume = self.diffuse_domain.chi_eps_finite_volume()
        remaining = dt
        while remaining > 0.0:
            substep = min(remaining, *(gel.gelation_max_dt() for gel in gels))
            for coupling in self.couplings:
                coupling.irreversible_timestep(substep)
            for gel in gels:
                gel.irreversible_timestep(
                    substep, chi_eps_finite_volume=chi_eps_finite_volume
                )
            remaining -= substep
