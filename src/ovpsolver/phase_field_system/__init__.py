"""The incompressible mixture the phase fields live in, and how a run starts.

Where :mod:`ovpsolver.phase_field_system.phase_field` stops at one phase, this
stops at the mixture's Rayleighian and the step that follows from it: the mixture
states ``R`` and differentiates it once against every rate
(:meth:`PhaseFieldSystem.rate_residual`), and each lagging variable then solves
its own transport row
(:func:`~ovpsolver.phase_field_system.transport.transport_step`).

A mixture class is the only thing that has to be named to know what a spec
means, so it is also where a run begins:

    PolymerizingB.run("pips.yaml")

or, for one this package ships, without any Python at all:

    python -m ovpsolver.polymerizing_b run pips.yaml

Nothing in the abstract layer knows about polymerization, sol/gel splits, or
Flory-Huggins. A concrete mixture declares its phases and couplings -- a roster
and a :class:`CouplingsParameters` subclass -- and builds them in
:meth:`PhaseFieldSystem.make_phase_fields` and
:meth:`PhaseFieldSystem.make_couplings`. Everything that is not a phase is a
coupling: the mixing energy, a drag between species, an entanglement that owns
fields of its own. :mod:`ovpsolver.model_b` is the shortest one that works and
:mod:`ovpsolver.polymerizing_b` is the one this codebase was written for.
"""

from .couplings import (
    Coupling,
    FloryHugginsCoupling,
    FloryHugginsPairParameters,
    FloryHugginsParameters,
)
from .parameters import (
    CouplingsParameters,
    PhaseFieldSystemParameters,
    SpecParameters,
)
from .run import load, read
from .system import PhaseFieldSystem

__all__ = [
    "Coupling",
    "CouplingsParameters",
    "FloryHugginsCoupling",
    "FloryHugginsPairParameters",
    "FloryHugginsParameters",
    "PhaseFieldSystem",
    "PhaseFieldSystemParameters",
    "SpecParameters",
    "load",
    "read",
]
