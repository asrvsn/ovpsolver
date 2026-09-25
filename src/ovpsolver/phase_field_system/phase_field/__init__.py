"""One phase of a stateful incompressible mixture, and what every phase has.

:class:`PhaseField` owns one conserved volume fraction. It declares what carries
that fraction, and whatever internal state rides along, as
:class:`~ovpsolver.phase_field_system.transport.Transported` variables whose flux
is a list of :class:`~ovpsolver.phase_field_system.transport.Flux` pieces; its
free energy as :class:`~ovpsolver.phase_field_system.energy.EnergyDensity` terms;
its friction and membrane resistance as a dissipation potential; and its pinned
normal flux at ``Sigma`` as a constraint potential. Every row of the step is
derived from those declarations, and nothing writes a residual by hand except a
variable that overrides ``transport_residual`` to state a law the flux list
cannot.

Only the base is here. Each kind of phase lives with the mixture that has it:
:class:`~ovpsolver.model_b.phase_field.CHPhaseField` is the Cahn-Hilliard species
moved by one Darcy velocity, and
:class:`~ovpsolver.polymerizing_b.phase_field.PolymerizingPhaseField` is the one
that crosslinks into a network. Neither is a special case of the other, and both
derive from :class:`PhaseField` directly.

The pressure that couples the phases, and the couplings that make them interact,
belong to the mixture (:mod:`ovpsolver.phase_field_system.system`).
"""

from .field import PhaseField
from .parameters import PhaseFieldParameters

__all__ = ["PhaseField", "PhaseFieldParameters"]
