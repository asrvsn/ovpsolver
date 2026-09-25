from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...parametric import (
    Count,
    Integer,
    Nonnegative,
    Number,
    Parameters,
    Positive,
    Text,
)
from ...phase_field_system.phase_field.parameters import PhaseFieldParameters
from ...phase_field_system.visualize.layers import MASK, PLAIN


class ZPartitionParameters(Parameters):
    """Finite-volume partition of the gelation coordinate ``z`` on ``[0, 1]``.

    Cells narrow geometrically towards ``z = 1``, each ``ratio`` times the width
    of the next, to shrink the ``O(dz)`` error of the trace moment read at that
    end. A ratio of one gives equal cells, which is the default: a spec that
    says nothing about grading gets none.
    """

    num_cells: int = Count()
    ratio: float = Number(1.0, minimum=1.0, coefficient=False)

    def derive(self) -> None:
        if self.ratio == 1.0:
            edges = np.linspace(0.0, 1.0, self.num_cells + 1)
        else:
            widths = self.ratio ** (self.num_cells - np.arange(1, self.num_cells + 1))
            edges = np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
            edges[-1] = 1.0
            # A steep enough ratio underflows the widest-to-narrowest spread.
            if np.any(np.diff(edges) <= 0.0):
                raise ValueError(
                    f"{self.where}: ratio {self.ratio} leaves cells of zero width "
                    f"among {self.num_cells}"
                )
        self.edges = np.ascontiguousarray(edges)

    @property
    def widths(self) -> npt.NDArray[np.float64]:
        return np.diff(self.edges)

    @property
    def midpoints(self) -> npt.NDArray[np.float64]:
        """The cell centres ``z_b``."""

        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def midpoint(self, b: "int | npt.ArrayLike") -> "float | npt.NDArray[np.float64]":
        return self.midpoints[b]

    def source_weight(
        self, functionality: int, b: "int | npt.ArrayLike"
    ) -> "float | npt.NDArray[np.float64]":
        """``s_b``, the cell average of ``z - z**(functionality - 1)`` over bin
        ``b``."""

        indices = np.arange(self.num_cells)[b]
        left = self.edges[indices]
        right = self.edges[indices + 1]
        width = right - left
        return self.midpoint(b) - (right**functionality - left**functionality) / (
            functionality * width
        )


class PolymerizingPhaseFieldParameters(PhaseFieldParameters):
    """A phase field whose monomers crosslink into a gel network.

    What the free monomers cost to move and inject, what the network costs, and
    the chemistry that decides how much of the phase is which. The sol-side
    entries repeat those of
    :class:`~ovpsolver.model_b.phase_field.CHPhaseFieldParameters`, because a
    crosslinking species has free monomers too, and the drags and fluxes carry a
    ``_sol`` or ``_gel`` to say which of the two velocities they act on.
    ``surface_flux_density`` and ``boundary_flux_density``, which the mixture
    reads off every phase alike, are the sol's: the network is not injected.

    ``z_partition`` is here rather than with the solver because the gelation
    coordinate is this phase's own state space: two crosslinking species in one
    mixture need not resolve their distributions alike.
    """

    monomer_polymerization_degree: float = Positive(1.0)
    #: The two coefficients of ``omega_i(alpha_i) = d^0 + d^1 alpha_i``, in
    #: units of ``k_B_T``. Setting ``bulk_affinity_1`` alone to zero recovers a
    #: composition-only affinity.
    #:
    #: They do not collapse into their sum. Differentiating gives
    #: ``d/dphi_i [omega_i phi_i] = d^0 + d^1``, so the *phase* row sees only the
    #: sum -- but ``omega_i phi_i = d^0 phi_i + d^1 r_i`` in the reacted-site
    #: fraction ``r_i``, and its ``c_x`` row sees ``-d^1 nu / f`` on its own.
    #: Declaring one number would silently drop that row, and somewhere hard to
    #: see: the potential it contributes is a constant, which has no facet jump,
    #: so what is lost is not a bulk force but the term's contribution at
    #: ``Sigma``, absorbed by ``Lambda_s`` to the accuracy that multiplier's P1
    #: space allows. Contrast ``surface_affinity``, which is one number because
    #: the equilibrium condition on ``c_x`` at the surface forces its ``d^1`` to
    #: vanish.
    bulk_affinity_0: float = Number(0.0)
    bulk_affinity_1: float = Number(0.0)
    #: The ``D`` of the sol's Darcy mobility, and the phase's only diffusivity:
    #: the network has no Darcy term. The gel's drags -- surface drag, Brinkman
    #: screening, and the floor that keeps its row solvable where there is no
    #: network yet -- still divide by a diffusivity to become coefficients, and
    #: they divide by this one, so both velocities' terms are measured on the
    #: scale of the sol's Darcy term. Of the whole sol and not of a monomer in
    #: it: the Darcy measure it weights is all of ``phi_sol``, so no chain
    #: length enters it.
    sol_diffusivity: float = Nonnegative(1.0)
    #: The ``l_eta`` of each velocity's Brinkman term, ``eta = l_eta^2 / D``
    #: against the single diffusivity above.
    sol_bulk_viscous_screening_length: float = Nonnegative(0.0)
    gel_bulk_viscous_screening_length: float = Nonnegative(0.0)
    #: The ``l_s`` of each velocity's surface viscosity
    #: (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.surface_viscosity`),
    #: ``eta_s = l_s^2``, in the units of its bulk screening length.
    sol_surface_viscous_screening_length: float = Nonnegative(0.0)
    gel_surface_viscous_screening_length: float = Nonnegative(0.0)
    surface_sol_drag: float = Nonnegative(0.0)
    boundary_sol_flux_density: float = Number(0.0)
    surface_sol_flux_density: float = Number(0.0)
    color_sol: str = Text("blue")

    monomer_functionality: int = Integer(3, minimum=3)
    #: Strictly positive, which is what makes this class the statement that the
    #: phase crosslinks: a species that does not react is a
    #: :class:`~ovpsolver.model_b.phase_field.CHPhaseFieldParameters`, not one of
    #: these with a gel velocity, a gelation coordinate and an entanglement state
    #: that nothing would ever feed. Asking which phases are networks is
    #: therefore asking the type.
    crosslinking_rate: float = Positive(1.0, coefficient=False)
    #: The CFL number ``CFL_u``, bounding the explicit sub-step in the gelation
    #: coordinate. Beside the partition it bounds, for the reason
    #: ``z_partition`` is on the phase.
    gelation_burgers_cfl: float = Number(
        0.5, minimum=0.0, maximum=1.0, exclusive_minimum=True, coefficient=False
    )
    #: The modulus each independent cycle contributes, in units of ``k_B_T``,
    #: scaling the shear-modulus birth rate. One is the phantom-network value.
    cycle_shear_modulus: float = Positive(1.0)
    surface_gel_drag: float = Nonnegative(1.0)
    boundary_gel_drag: float = Nonnegative(0.0)
    color_gel: str = Text("red")
    z_partition: ZPartitionParameters = ZPartitionParameters()

    #: The diagnostics this phase publishes that are not amounts of material.
    #: The conversion is a fraction of the phase, so it means nothing where there
    #: is no phase; the rest are the gel's elastic state, which is already
    #: resolved in diffuse-weighted form and whose transport row omits the
    #: ``chi`` every other state carries.
    plot_chi = {
        "reaction_extent": MASK,
        "gel_shear_modulus": PLAIN,
        "gel_strain_moment": PLAIN,
        "gel_hydrostatic_stress": PLAIN,
        "gel_deviatoric_stress": PLAIN,
    }

    @property
    def surface_flux_density(self) -> float:
        return self.surface_sol_flux_density

    @property
    def boundary_flux_density(self) -> float:
        return self.boundary_sol_flux_density
