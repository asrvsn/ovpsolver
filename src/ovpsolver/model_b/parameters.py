"""Parameters of the plain multi-component mixture."""

from __future__ import annotations

from ..parametric import ParametersList
from ..phase_field_system.parameters import (
    CouplingsParameters,
    PhaseFieldSystemParameters,
)
from .couplings import CHFloryHugginsParameters
from .phase_field import CHPhaseFieldParameters


class ModelBCouplingsParameters(CouplingsParameters):
    """What couples the phases of a Model B mixture: their mixing, and nothing else.

    Parameters
    ----------
    flory_huggins : the mixing coefficient for each unordered pair of phases.
    """

    flory_huggins: CHFloryHugginsParameters = CHFloryHugginsParameters()


class ModelBParameters(PhaseFieldSystemParameters):
    """Model B: a saturated mixture of phases that only ever flow.

    The base class with its roster narrowed and its one coupling named: the
    shortest statement of a working mixture, and the one to copy when writing
    another. ``phase_fields`` is a flat list because this mixture draws no
    distinction between kinds of phase; one that does narrows it to a block of
    typed lists instead
    (:class:`~ovpsolver.polymerizing_b.parameters.PolymerizingRosterParameters`).
    """

    phase_fields: tuple[CHPhaseFieldParameters, ...] = ParametersList(
        CHPhaseFieldParameters(), minimum=1
    )
    couplings: ModelBCouplingsParameters = ModelBCouplingsParameters()
