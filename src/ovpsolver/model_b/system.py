"""Multi-component Model B: phases that demix by flowing, and nothing more.

The pressure, the saturation constraint, the initial condition solve and the row
assembly are :class:`~ovpsolver.phase_field_system.system.PhaseFieldSystem`'s;
every phase is a :class:`~ovpsolver.model_b.phase_field.CHPhaseField`; and the
mixing energy is a coupling the parameters class declares. What is left is to
build the phases and the coupling.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..fem.saveable import ParametricSaveable
from ..phase_field_system.couplings import Coupling
from ..phase_field_system.phase_field import PhaseField
from ..phase_field_system.system import PhaseFieldSystem
from .couplings import CHFloryHugginsCoupling
from .parameters import ModelBParameters
from .phase_field import CHPhaseField


class ModelB(PhaseFieldSystem, ParametricSaveable[ModelBParameters]):
    """A saturated mixture of non-polymerizing phase fields."""

    ## Construction

    def make_phase_fields(self) -> Sequence[PhaseField]:
        return tuple(
            CHPhaseField(
                self.solver_parameters, phase, self.diffuse_domain, self.k_B_T
            )
            for phase in self.parameters.phase_fields
        )

    def make_couplings(self) -> Sequence[Coupling]:
        return (
            CHFloryHugginsCoupling(
                self.solver_parameters,
                self.parameters.couplings.flory_huggins,
                self.diffuse_domain,
                self.phase_fields,
                self.k_B_T,
            ),
        )

    def name(self) -> str:
        return "model_b"
