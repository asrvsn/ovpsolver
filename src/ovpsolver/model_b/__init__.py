"""Multi-component Model B, the mixture with no chemistry in it.

    python -m ovpsolver.model_b run spec.yaml

Cahn-Hilliard phases that demix by flowing, coupled only by their Flory-Huggins
mixing. It is also the minimal mixture: a parameters class naming its roster and
its one coupling, and a system class building its phases.
"""

from .couplings import (
    CHFloryHugginsCoupling,
    CHFloryHugginsPairParameters,
    CHFloryHugginsParameters,
)
from .parameters import ModelBCouplingsParameters, ModelBParameters
from .system import ModelB

__all__ = [
    "ModelB",
    "ModelBCouplingsParameters",
    "ModelBParameters",
    "CHFloryHugginsCoupling",
    "CHFloryHugginsPairParameters",
    "CHFloryHugginsParameters",
]
