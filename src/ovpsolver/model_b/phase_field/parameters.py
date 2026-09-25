from __future__ import annotations

from ...parametric import Nonnegative, Number, Positive
from ...phase_field_system.phase_field.parameters import PhaseFieldParameters


class CHPhaseFieldParameters(PhaseFieldParameters):
    """A phase field moved by a single Darcy velocity.

    The multi-species Cahn-Hilliard species: its free energy, what its one
    velocity costs in the bulk and against the inclusion surfaces, and the fluxes
    imposed on that velocity at the two kinds of boundary. None of its entries
    carries the ``_sol`` a polymerizing phase spells them with, and there is no
    ``color_sol``: this phase is all sol, so there is nothing for the infix to
    distinguish.

    Parameters
    ----------
    diffusivity : the ``D`` of this phase's Darcy mobility, so the drag
        its velocity pays is measured over ``D``.
    bulk_viscous_screening_length : the ``l_eta`` of the Brinkman term
        (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.bulk_viscosity`),
        which sets the bulk viscosity as ``eta = l_eta^2 / diffusivity``. Zero
        turns it off.
    monomer_polymerization_degree : the chain length ``N`` of this phase's ideal
        term ``(phi / N) ln phi``. One for a monomer.
    bulk_affinity : the composition-independent part of this phase's chemical
        potential, in units of ``k_B_T``.
    surface_drag : the coefficient of the tangential drag this phase's velocity
        pays against the inclusion surfaces, on the measure ``dGamma_eps``.
    surface_viscous_screening_length : the ``l_s`` of the surface viscosity
        (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.surface_viscosity`,
        which says how to size it), entering as ``eta_s = l_s^2``: a length in
        the units of the bulk one, so the two are directly comparable. Zero turns
        it off.
    boundary_flux_density : the normal flux ``Q`` imposed on this phase's
        velocity at the outer boundary, per unit area. The phases' must satisfy
        ``sum_i nu_i Q_i = 0``.
    surface_flux_density : the same, through the diffuse inclusion surfaces.
    """

    diffusivity: float = Nonnegative(1.0)
    bulk_viscous_screening_length: float = Nonnegative(0.0)
    monomer_polymerization_degree: float = Positive(1.0)
    bulk_affinity: float = Number(0.0)
    surface_drag: float = Nonnegative(0.0)
    surface_viscous_screening_length: float = Nonnegative(0.0)
    boundary_flux_density: float = Number(0.0)
    surface_flux_density: float = Number(0.0)
