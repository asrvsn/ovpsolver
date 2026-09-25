"""The friction a gel network exerts on everything moving past it.

A dissipation and nothing else: no fields, no energy, no chemistry. What a
network costs a velocity crossing it is quadratic in their relative velocity and
weighted by the product of the two densities in contact, so it vanishes wherever
either is absent, and differentiating it puts a friction into both velocity rows
without this naming either.

Its own coupling, apart from :mod:`.gel_entanglements`, because the two are
different kinds of statement about the same networks: entanglement is energetic
and mechanical, drag a new dissipation. A mixture could have either without the
other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import ufl

from ...parametric import Nonnegative, Pairs, Parameters, pairs
from ...phase_field_system.couplings import Coupling
from ..phase_field import PolymerizingPhaseField
from . import roster

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...phase_field_system.phase_field import PhaseField, PhaseFieldParameters
    from ...phase_field_system.transport import Flux


class DragParameters(Parameters):
    """Friction between a network and what moves past it, stated pairwise.

    Parameters
    ----------
    sol_gel : friction between a gel network and the sol it moves through,
        written ``(sol, gel)``: the first phase is what moves and the second the
        network it moves through. Required for every gel against every phase,
        including a gel against its own sol.
    gel_gel : friction between two distinct gel networks.
    """

    sol_gel: dict = Pairs(Nonnegative())
    gel_gel: dict = Pairs(Nonnegative())

    def populate(self, phases: "Sequence[PhaseFieldParameters]") -> None:
        """Fill both drag matrices from the pairs, checking coverage as it goes."""

        names = [parameters.name for parameters in phases]
        index = {name: position for position, name in enumerate(names)}
        #: Roster positions of the networks; see :func:`.roster.gel_positions`.
        self.gel_indices = roster.gel_positions(phases)
        gels = frozenset(self.gel_indices)
        self.check_sol_gel_order(index, gels)

        self.gel_sol_matrix = self.drag_matrix(
            "sol_gel",
            self.sol_gel,
            index,
            names,
            required={
                frozenset((gel, other))
                for gel in gels
                for other in range(len(phases))
            },
            # Only the gel's own row: the coefficient is a friction that gel
            # feels, and a sol has no row of its own to put it in.
            rows=lambda first, second: [
                (a, b) for a, b in ((first, second), (second, first)) if a in gels
            ],
        )
        self.gel_gel_matrix = self.drag_matrix(
            "gel_gel",
            self.gel_gel,
            index,
            names,
            required=pairs.distinct_pairs(sorted(gels)),
            rows=lambda first, second: [(first, second), (second, first)],
        )

    def check_sol_gel_order(self, index: dict[str, int], gels: frozenset[int]) -> None:
        """Refuse a ``sol_gel`` pair whose second phase has no gel.

        Coverage alone would not catch it: an unordered ``(protein, water)``
        covers the same pair as ``(water, protein)`` and would be filed under the
        protein's row all the same, so the spec could say the opposite of what
        it meant and read back identically.
        """

        for pair in self.sol_gel:
            _, second = pairs.pair_positions(
                f"{self.where}.sol_gel", pair, index
            )
            if second not in gels:
                raise ValueError(
                    f"{self.where}.sol_gel[{pairs.show_pair(pair)}]: {pair[1]} "
                    f"does not crosslink, so it has no gel for {pair[0]} to drag "
                    f"against. The pair is (sol, gel); the network goes second"
                )

    def drag_matrix(
        self,
        directive: str,
        coefficients: dict,
        index: dict[str, int],
        names: "Sequence[str]",
        *,
        required: set[frozenset[int]],
        rows: "Callable[[int, int], list[tuple[int, int]]]",
    ) -> npt.NDArray[np.float64]:
        """One drag coefficient's dense matrix, from its pairs.

        ``required`` is every pair the coefficient has to be stated for and
        ``rows`` says which entries a stated pair fills, since the two drags
        differ in both: a gel-sol friction is felt only in the gel's row, while a
        gel-gel one is felt in both.
        """

        matrix = np.zeros((len(names), len(names)), dtype=float)
        seen: set[frozenset[int]] = set()
        for pair, value in coefficients.items():
            first, second = pairs.pair_positions(
                f"{self.where}.{directive}", pair, index
            )
            key = frozenset((first, second))
            if key not in required:
                raise ValueError(
                    f"{self.where}.{directive}[{pairs.show_pair(pair)}] is not a pair "
                    f"this coefficient applies to"
                )
            seen.add(key)
            for row, column in rows(first, second):
                matrix[row, column] = value
        missing = required - seen
        if missing:
            raise ValueError(
                f"{self.where}.{directive} is missing "
                + ", ".join(
                    sorted(pairs.show_positions(pair, names) for pair in missing)
                )
            )
        return matrix


class Drag(Coupling[DragParameters]):
    """What every gel network in the mixture costs the velocities moving past it."""

    def name(self) -> str:
        return "drag"

    ## Dissipation

    def dissipation(self) -> ufl.Form:
        """What a gel network costs everything trying to move past it.

        Gel-sol is counted once per (gel, phase) ordered pair and gel-gel twice,
        hence the halved coefficient there.
        """

        dx = self.solver_parameters.get_mesh_dx()
        parameters = self.parameters
        rayleighian = ufl.as_ufl(0.0)

        for i in parameters.gel_indices:
            network = self.phase_fields[i].gel_flux()
            for j, other in enumerate(self.phase_fields):
                other_sol, other_gel = self.sol_gel_fluxes(other)
                against = [(other_sol, parameters.gel_sol_matrix[i, j])]
                if other_gel is not None:
                    against.append(
                        (other_gel, 0.5 * parameters.gel_gel_matrix[i, j])
                    )
                for flux, drag in against:
                    slip = network.live_velocity - flux.live_velocity
                    rayleighian += (
                        0.5
                        * float(drag)
                        * network.lagged_density
                        * flux.lagged_density
                        * ufl.dot(slip, slip)
                    )

        return self.diffuse_domain.chi_eps * rayleighian * dx

    def sol_gel_fluxes(self, field: "PhaseField") -> "tuple[Flux, Flux | None]":
        """A phase's content split into what is sol and what is gel.

        A distinction this mixture draws and not one every phase field has: a
        species that does not crosslink is all sol on its single velocity. In
        fluxes, so each side of a drag term comes with the velocity that carries
        it.
        """

        if isinstance(field, PolymerizingPhaseField):
            return field.sol_flux(), field.gel_flux()
        (sol,) = field.flux()
        return sol, None
