"""Parameters of the polymerizing incompressible mixture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..model_b.phase_field import CHPhaseFieldParameters
from ..parametric import Parameters, ParametersList
from ..phase_field_system.parameters import (
    CouplingsParameters,
    PhaseFieldSystemParameters,
)
from ..phase_field_system.visualize.layers import phase_layer
from .couplings import (
    DragParameters,
    GelEntanglementsParameters,
    PolymerizingFloryHugginsParameters,
)
from .phase_field import PolymerizingPhaseFieldParameters

if TYPE_CHECKING:
    from ..phase_field_system.phase_field import PhaseFieldParameters
    from ..phase_field_system.visualize.layers import Layer


class PolymerizingRosterParameters(Parameters):
    """The mixture's phases, sorted into the two kinds it has.

    A block and not a flat list, because which list a phase is in is the whole
    of what says whether it crosslinks: there is no field on a phase to read it
    off.

    Parameters
    ----------
    sol : phases that only ever flow, in mixture order. Read against Model B's
        :class:`~ovpsolver.model_b.phase_field.CHPhaseFieldParameters`, and
        named for what such a phase is beside a network, which is also the word
        the gel-side coefficients are named against.
    polymerizing : phases that crosslink into a gel network.
    """

    sol: tuple[CHPhaseFieldParameters, ...] = ParametersList(
        CHPhaseFieldParameters()
    )
    polymerizing: tuple[PolymerizingPhaseFieldParameters, ...] = ParametersList(
        PolymerizingPhaseFieldParameters()
    )


class PolymerizingBCouplingsParameters(CouplingsParameters):
    """What couples the phases of a polymerizing mixture.

    Parameters
    ----------
    flory_huggins : the four mixing coefficients ``chi_00`` to ``chi_11`` per
        pair of phases, on the state space the reaction coordinate extends.
    gel_entanglements : how readily any two networks lock together.
    drag : what the networks cost everything moving past them.
    """

    flory_huggins: PolymerizingFloryHugginsParameters = (
        PolymerizingFloryHugginsParameters()
    )
    gel_entanglements: GelEntanglementsParameters = GelEntanglementsParameters()
    drag: DragParameters = DragParameters()


class PolymerizingBParameters(PhaseFieldSystemParameters):
    """Polymerizing Model B(+): sol phases, gel phases, and what couples them.

    The base's roster, narrowed from a list of one kind of phase to a block
    holding two. Sol first, then polymerizing, which is the order the mixture
    builds its phases in; :attr:`phase_field_parameters` joins them, keeping the
    distinction out of everything downstream.

    Parameters
    ----------
    phase_fields : the two typed rosters, under ``sol`` and ``polymerizing``.
    couplings : the mixing energy, and everything the networks do to the rest of
        the mixture.
    """

    phase_fields: PolymerizingRosterParameters = PolymerizingRosterParameters()
    couplings: PolymerizingBCouplingsParameters = PolymerizingBCouplingsParameters()

    @property
    def phase_field_parameters(self) -> tuple["PhaseFieldParameters", ...]:
        return (*self.phase_fields.sol, *self.phase_fields.polymerizing)

    def phase_layers(self, *, group_polymerizing: bool = False) -> tuple["Layer", ...]:
        """Sol phases in one colour each, gels in two unless asked to group.

        A polymerizing phase is one conserved species with two states, and which
        a picture should show depends on the question: how much protein is here
        is one layer, how much of it has crosslinked is two. The densities are
        drawn rather than the fraction, because they sum to the phase and so
        composite into exactly the grouped picture.
        """

        layers = [phase_layer(parameters) for parameters in self.phase_fields.sol]
        for parameters in self.phase_fields.polymerizing:
            if group_polymerizing:
                layers.append(phase_layer(parameters))
                continue
            layers.append(
                phase_layer(parameters, "phi_sol", color=parameters.color_sol)
            )
            layers.append(
                phase_layer(parameters, "phi_gel", color=parameters.color_gel)
            )
        return tuple(layers)
