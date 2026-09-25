"""A phase field carried by one Darcy velocity.

The whole phase moves together, on one velocity and one flux piece: the
multi-species Cahn-Hilliard species and the simplest concrete
:class:`~ovpsolver.phase_field_system.phase_field.PhaseField`. It states its
velocity, its flux, its convex-split free energy and its dissipation, and builds
no row: the base class derives every row from those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ufl

from ...fem.elements import VelocityElement
from ...fem.saveable import ParametricSaveable
from ...phase_field_system.energy import EnergyDensity
from ...phase_field_system.phase_field.field import PhaseField
from ...phase_field_system.transport import Flux
from ...solver.regularization import log_reg
from .parameters import CHPhaseFieldParameters

if TYPE_CHECKING:
    from ufl.core.expr import Expr


class CHPhaseField(PhaseField, ParametricSaveable[CHPhaseFieldParameters]):
    """Non-polymerizing phase field: all of it rides one velocity."""

    _v_test: "Expr"

    ## Elements

    def declare_velocity_elements(self) -> list[VelocityElement]:
        """One rate, named without a qualifier because there is nothing to
        distinguish it from."""

        return [
            VelocityElement(
                "v",
                "_v_test",
                degree=2,
                shape=(self.solver_parameters.dolfinx_mesh.geometry.dim,),
            )
        ]

    ## Transport

    def flux(self) -> list[Flux]:
        """The whole phase on the one velocity, one monomer volume arriving with
        each monomer injected at either boundary."""

        per_monomer = self.parameters.monomer_volume
        return [
            Flux(
                live_velocity=self.rates.v,
                lagged_velocity=self.prev.v,
                lagged_density=self.prev.phi,
                surface_flux=per_monomer * self.surface_flux(),
                boundary_flux=(
                    per_monomer * self.parameters.boundary_flux_density
                ),
            )
        ]

    ## Free energy

    def mixing_entropy(self, phi: "Expr") -> "Expr":
        """``k_B_T (phi / N) log phi``, the ideal term of the mixing free
        energy."""

        return (
            self.k_B_T
            * phi
            / self.parameters.monomer_polymerization_degree
            * log_reg(phi, self.solver_parameters.log_reg_delta)
        )

    def bulk_affinity(self, phi: "Expr") -> "Expr":
        """``k_B_T omega phi``, the affinity term of the mixing free energy at
        constant ``omega``."""

        return self.k_B_T * self.parameters.bulk_affinity * phi

    def energy_density(self) -> list[EnergyDensity]:
        phi_next = self.variable("phi", time_level="next")
        return [
            EnergyDensity(
                convex=self.mixing_entropy(phi_next) + self.bulk_affinity(phi_next)
            ),
            *super().energy_density(),
        ]

    ## Dissipation

    def dissipation(self) -> ufl.Form:
        density = self.prev.phi
        diffusivity = self.parameters.diffusivity
        eta = self.parameters.bulk_viscous_screening_length**2 / diffusivity
        return (
            super().dissipation()
            + self.darcy_dissipation(density, self.rates.v, diffusivity)
            + self.diffuse_domain.drag(
                self.parameters.surface_drag * density, self.rates.v
            )
            + self.bulk_viscosity(density, self.rates.v, eta)
            + self.surface_viscosity(
                density,
                self.rates.v,
                self.parameters.surface_viscous_screening_length,
            )
        )
