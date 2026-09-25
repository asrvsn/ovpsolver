"""Hook-and-loop entanglement between interpenetrating gel networks.

Two networks that have grown through each other are held together by the
entanglements they trapped. What that costs is a locking modulus and a relative
strain moment that belong to neither network: they are born from both networks'
birth rates at once, carried by the first network's velocity, and push back on
both. So this is a coupling that owns fields -- those two and a relative
volumetric moment, per ordered pair of networks that lock -- and it is energetic
and mechanical only; the friction a network exerts is :mod:`.drag`.

It stores energy and declares none. The moment the energy is stored in is lagged,
transported by the step but never solved for at ``k+1``, so there is no live value
to differentiate a declared energy against; the timestepping makes the increment
a stress power, as it does a single network's elastic energy, and that power is
the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import ufl

from ...fem.elements import Solve, StateElement
from ...fem.elements.dg0 import (
    DISCONTINUOUS_LAGRANGE,
    add_scaled_kronecker_outer,
    cell_scalars,
)
from ...parametric import Nonnegative, Pairs, Parameters, pairs
from ...phase_field_system.couplings import Coupling
from ...phase_field_system.transport import Transported
from ..phase_field import PolymerizingPhaseField
from . import roster

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ufl.core.expr import Expr

    from ...phase_field_system.energy import EnergyDensity
    from ...phase_field_system.phase_field import PhaseFieldParameters


class GelEntanglementsParameters(Parameters):
    """How readily two networks entangle, stated pairwise over the gels.

    Parameters
    ----------
    hook_loop_capture_volume : the volume over which two networks entangle, keyed
        by pairs of distinct *gels*. A pair that is absent does not lock.
    """

    hook_loop_capture_volume: dict = Pairs(Nonnegative())

    def populate(self, phases: "Sequence[PhaseFieldParameters]") -> None:
        """Fill the capture matrix, indexed over gels, from the pairs.

        A pair that is absent does not lock, which is the ordinary case, so
        nothing here requires coverage the way the drag does.
        """

        #: Roster positions of the networks; see :func:`.roster.gel_positions`.
        self.gel_indices = roster.gel_positions(phases)
        gels = [phases[position].name for position in self.gel_indices]
        index = {name: position for position, name in enumerate(gels)}
        matrix = np.zeros((len(gels), len(gels)), dtype=float)
        for pair, value in self.hook_loop_capture_volume.items():
            first, second = pairs.pair_positions(
                f"{self.where}.hook_loop_capture_volume", pair, index
            )
            if first == second:
                raise ValueError(
                    f"{self.where}.hook_loop_capture_volume"
                    f"[{pairs.show_pair(pair)}]: a gel cannot hook onto itself"
                )
            matrix[first, second] = value
            matrix[second, first] = value
        self.hook_loop_capture_matrix = matrix

    @property
    def locked_pairs(self) -> tuple[tuple[int, int], ...]:
        """Ordered gel pairs with a nonzero capture volume.

        Ordered, not unordered: ``tilde m_ij`` lives in gel ``i``'s frame and
        ``tilde m_ji`` in gel ``j``'s, so a locked pair is two distinct sets of
        fields transported by different velocity gradients.
        """

        capture = self.hook_loop_capture_matrix
        size = capture.shape[0]
        return tuple(
            (i, j)
            for i in range(size)
            for j in range(size)
            if not np.isclose(capture[i, j], 0.0)
        )


@dataclass(frozen=True)
class LockingTransported(Transported):
    """A locking field that the pair's relative motion feeds as it is carried.

    The carrying is the flux list; the feeding is no flux, so it is stated as
    :attr:`gain` and added to the row over the step. Explicit in the lagged
    locking and linear in the live velocities, so the row keeps the semi-implicit
    form of the rate step and the transport step solves it cell by cell like any
    other.
    """

    #: What the pair's relative motion adds per unit time, in the variable's
    #: shape.
    gain: "Expr | None" = None

    def transport_residual(self) -> ufl.Form:
        """The conservation row, less the gain over one step."""

        dt = self.solver_parameters.dt_live
        dx = self.solver_parameters.get_mesh_dx()
        return super().transport_residual() - dt * ufl.inner(self.gain, self.test) * dx


class GelEntanglements(Coupling[GelEntanglementsParameters]):
    """The locking fields of every ordered pair of networks that entangle."""

    def name(self) -> str:
        return "gel_entanglements"

    @property
    def gels(self) -> tuple[PolymerizingPhaseField, ...]:
        """The networks, in the order the capture matrix indexes them."""

        return tuple(
            self.phase_fields[position]
            for position in self.parameters.gel_indices
        )

    ## The locking fields

    def declare_elements(self) -> list[StateElement]:
        """Three lagging DG0 fields per locked pair: a modulus and two moments.

        Cell-constant because they are carried by the upwind DG0 transport, and
        lagging because the stress they exert is evaluated semi-implicitly: the
        power is linear in the live velocities and the moment it multiplies is
        the one from the last step. None, for a mixture with no locked pair.
        """

        dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        specs = []
        for pair in self.parameters.locked_pairs:
            modulus, volumetric, strain = self._field_names(pair)
            specs.append(
                StateElement(
                    modulus,
                    f"_{modulus}_test",
                    family=DISCONTINUOUS_LAGRANGE,
                    degree=0,
                    solve=Solve.ENERGY_LAGGING,
                )
            )
            specs.append(
                StateElement(
                    volumetric,
                    f"_{volumetric}_test",
                    family=DISCONTINUOUS_LAGRANGE,
                    degree=0,
                    solve=Solve.ENERGY_LAGGING,
                )
            )
            specs.append(
                StateElement(
                    strain,
                    f"_{strain}_test",
                    family=DISCONTINUOUS_LAGRANGE,
                    degree=0,
                    shape=(dim * dim, dim * dim),
                    symmetry=True,
                    solve=Solve.ENERGY_LAGGING,
                )
            )
        return specs

    def _field_names(self, pair: tuple[int, int]) -> tuple[str, str, str]:
        """This pair's three field names, qualified by the two networks, since
        every pair's fields live on this one coupling."""

        first, second = (self.gels[position].name() for position in pair)
        suffix = f"{first}_{second}"
        return (
            f"locking_modulus_{suffix}",
            f"relative_volumetric_moment_{suffix}",
            f"relative_strain_moment_{suffix}",
        )

    def set_initial_conditions(self) -> None:
        for pair in self.parameters.locked_pairs:
            for name in self._field_names(pair):
                for fields in (self.prev, self.next):
                    getattr(fields, name).x.array[:] = 0.0

    ## Energy release

    def energy_rate(self, terms: list["EnergyDensity"]) -> ufl.Form:
        """The locking stress power, pair by pair.

        Half to each network. Linear in the two live gel velocities, so
        differentiating the Rayleighian puts it into both velocity rows without
        this naming them.
        """

        dx = self.solver_parameters.get_mesh_dx()
        total = self.nothing()
        for pair in self.parameters.locked_pairs:
            first, second = (self.gels[position] for position in pair)
            total += (
                0.5
                * ufl.inner(self._first_stress(pair), ufl.grad(first.rates.v_g))
                * dx
                + 0.5
                * ufl.inner(self._second_stress(pair), ufl.grad(second.rates.v_g))
                * dx
            )
        return total

    def _first_stress(self, pair: tuple[int, int]) -> "Expr":
        """``tilde tau^fst``, the locking stress on the first network, from the
        lagged fields."""

        dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        modulus_name, _, strain_name = self._field_names(pair)
        moment = getattr(self.prev, strain_name)
        modulus = getattr(self.prev, modulus_name)
        identity = ufl.Identity(dim)
        return ufl.as_matrix(
            [
                [
                    0.5
                    * sum(
                        self._m4(moment, a, b, c, c, dim)
                        + self._m4(moment, b, a, c, c, dim)
                        for c in range(dim)
                    )
                    - modulus * identity[a, b]
                    for b in range(dim)
                ]
                for a in range(dim)
            ]
        )

    def _second_stress(self, pair: tuple[int, int]) -> "Expr":
        """``tilde tau^snd``, the locking stress on the second network, from the
        lagged fields."""

        dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        modulus_name, _, strain_name = self._field_names(pair)
        moment = getattr(self.prev, strain_name)
        modulus = getattr(self.prev, modulus_name)
        identity = ufl.Identity(dim)
        return ufl.as_matrix(
            [
                [
                    modulus * identity[c, d]
                    - 0.5
                    * sum(
                        self._m4(moment, a, a, c, d, dim)
                        + self._m4(moment, a, a, d, c, dim)
                        for a in range(dim)
                    )
                    for d in range(dim)
                ]
                for c in range(dim)
            ]
        )

    @staticmethod
    def _m4(moment: "Expr", a: int, b: int, c: int, d: int, dim: int) -> "Expr":
        """``tilde m^{abcd}``, stored as the symmetric ``(d^2, d^2)`` matrix."""

        return moment[a * dim + c, b * dim + d]

    ## Transport of the locking fields

    def transported(self) -> list[Transported]:
        """All three fields of every locked pair, carried by the first network's
        gel velocity and not densities of the free fluid."""

        variables: list[Transported] = []
        for pair in self.parameters.locked_pairs:
            variables += self._pair_transported(pair)
        return variables

    def _pair_transported(self, pair: tuple[int, int]) -> list[Transported]:
        """One pair's three fields: the modulus only carried, the volumetric
        moment fed by the divergence of the relative velocity, and the strain
        moment stretched by both networks' velocity gradients."""

        dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        first, second = (self.gels[position] for position in pair)
        modulus_name, volumetric_name, strain_name = self._field_names(pair)
        modulus = getattr(self.prev, modulus_name)
        strain = getattr(self.prev, strain_name)
        first_gradient = ufl.grad(first.rates.v_g)
        second_gradient = ufl.grad(second.rates.v_g)

        def stretching(a, b, c, d):
            return sum(
                self._m4(strain, a, e, c, d, dim) * first_gradient[b, e]
                + self._m4(strain, e, b, c, d, dim) * first_gradient[a, e]
                - self._m4(strain, a, b, c, e, dim) * second_gradient[e, d]
                - self._m4(strain, a, b, e, d, dim) * second_gradient[e, c]
                for e in range(dim)
            )

        def carried(name):
            return [first.gel_flux_carrying(getattr(self.prev, name))]

        return [
            Transported(
                self.element(modulus_name),
                flux=carried(modulus_name),
                chi_weighted=False,
            ),
            LockingTransported(
                self.element(volumetric_name),
                flux=carried(volumetric_name),
                chi_weighted=False,
                gain=modulus * ufl.div(first.rates.v_g - second.rates.v_g),
            ),
            LockingTransported(
                self.element(strain_name),
                flux=carried(strain_name),
                chi_weighted=False,
                gain=ufl.as_matrix(
                    [
                        [stretching(a, b, c, d) for b in range(dim) for d in range(dim)]
                        for a in range(dim)
                        for c in range(dim)
                    ]
                ),
            ),
        ]

    ## The irreversible half-step

    def irreversible_timestep(self, dt: float) -> None:
        """Forward-Euler every locked pair's modulus and strain moment over one
        chemistry sub-step, at a rate that multiplies each network's birth rate
        by the other's modulus.

        The sub-step is the mixture's, which calls this before any gel's, since
        the rate reads both networks' moduli
        (:meth:`~ovpsolver.polymerizing_b.system.PolymerizingB.irreversible_timestep`).
        """

        for pair in self.parameters.locked_pairs:
            self._lock(pair, dt)

    def _lock(self, pair: tuple[int, int], dt: float) -> None:
        """Advance one pair's locking modulus and strain moment."""

        i, j = pair
        first, second = self.gels[i], self.gels[j]
        modulus_name, _, strain_name = self._field_names(pair)
        rate = (
            self.parameters.hook_loop_capture_matrix[i, j] / float(self.k_B_T)
        ) * (
            first.shear_birth_rate()
            * cell_scalars(second.next.gel_shear_modulus)
            + second.shear_birth_rate()
            * cell_scalars(first.next.gel_shear_modulus)
        )
        cell_scalars(getattr(self.next, modulus_name))[:] += dt * rate
        add_scaled_kronecker_outer(
            getattr(self.next, strain_name),
            dt * rate,
            self.solver_parameters.dolfinx_mesh.geometry.dim,
        )
