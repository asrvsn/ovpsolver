"""Flory-Huggins mixing as a quadratic form: what is true whatever the state space.

The free energy of mixing that no phase owns. Whatever a mixture's phases carry,
the pair interactions between them are one symmetric matrix ``H`` acting on a
state vector ``y``, with energy density ``(k_B_T/2) y^T H y``. Building ``H``
from the declared pairs, splitting it convex-concave, and expanding the quadratic
into terms are the same three steps in every mixture.

What differs is ``y``. A mixture that does not react has one slot per phase and
``H`` is the matrix of ``chi`` itself
(:mod:`ovpsolver.model_b.couplings.flory_huggins`). A mixture that crosslinks has
two slots per phase, because a coefficient may couple to the reacted-site
fraction rather than to the composition, and ``H`` is built from four
coefficients per pair (:mod:`ovpsolver.polymerizing_b.couplings.flory_huggins`).
Neither state space is a special case of the other, so both derive from here.

The split is over the whole matrix at once, never pair by pair: it needs a fixed
``S`` with ``S - H >= 0`` over the entire subspace, and a sum of per-pair splits
is a different and weaker statement. That is why one coupling holds every pair.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import ufl

from ...parametric import PairsParameters, Parameters, pairs
from ...parametric.parameters import ParametersT
from ...solver.instrument import LOGGER_NAME
from ..energy import EnergyDensity
from .base import Coupling

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ufl.core.expr import Expr

    from ..phase_field import PhaseFieldParameters

logger = logging.getLogger(f"{LOGGER_NAME}.ovp")

#: Below this the spectrum counts as positive semidefinite and needs no split.
CONVEXITY_TOLERANCE = 1.0e-12


class FloryHugginsPairParameters(Parameters):
    """The interaction coefficients between two phases.

    Empty here. What a pair carries is a statement about the state space the
    interaction lives on, so the concrete mixtures declare it.
    """


class FloryHugginsParameters(PairsParameters):
    """Mixing coefficients, one entry per unordered pair of phases.

    Every pair has to appear. An omitted one would read as "these two do not
    interact", which is a statement worth having to make: it is the difference
    between a mixture that stays mixed on purpose and one whose demixing was left
    out of the spec.
    """

    item = FloryHugginsPairParameters

    def populate(self, phases: "Sequence[PhaseFieldParameters]") -> None:
        """Turn the declared pairs into one dense matrix per coefficient.

        Called by the mixture once it has read its roster, because every check
        here is about the roster: that a pair names phases that exist, that every
        pair is present, and that no coefficient is stated which cannot apply.
        Failing here rather than at assembly means the message can name the pair.
        """

        names = [parameters.name for parameters in phases]
        index = {name: position for position, name in enumerate(names)}
        size = len(phases)

        self.matrices = {
            key: np.zeros((size, size), dtype=float) for key in self.coefficients()
        }
        required = pairs.distinct_pairs(range(size))
        seen: set[frozenset[int]] = set()
        for pair, block in self.pairs.items():
            first, second = pairs.pair_positions(self.where, pair, index)
            if first == second:
                raise ValueError(
                    f"{self.where}[{pairs.show_pair(pair)}]: a phase does not mix "
                    f"with itself, and what it costs to be surrounded by its own "
                    f"kind is already in its ideal term"
                )
            seen.add(frozenset((first, second)))
            expected = self.applicable(phases, first, second)
            if block.given != expected:
                raise ValueError(
                    f"{self.where}[{pairs.show_pair(pair)}] expects "
                    f"{', '.join(sorted(expected))}; got "
                    f"{', '.join(sorted(block.given)) or 'nothing'}"
                )
            flipped = index[pair[0]] > index[pair[1]]
            for key, value in self.values_of(block, flipped=flipped).items():
                self.matrices[key][first, second] = value
                self.matrices[key][second, first] = value

        missing = required - seen
        if missing:
            shown = sorted(pairs.show_positions(pair, names) for pair in missing)
            raise ValueError(f"{self.where} is missing {', '.join(shown)}")

    ## What a pair may say

    def coefficients(self) -> tuple[str, ...]:
        """Every coefficient name a pair may carry, in declaration order."""

        return tuple(self.item.declarations())

    def applicable(
        self, phases: "Sequence[PhaseFieldParameters]", first: int, second: int
    ) -> frozenset[str]:
        """Which coefficients *this* pair may carry.

        All of them, unless a mixture knows a reason one cannot apply to a
        particular pair -- a coefficient that multiplies a quantity one of the
        two phases does not have. Stating such a one is an error rather than a
        no-op, because it almost always means the pair was written backwards.
        """

        return frozenset(self.coefficients())

    def values_of(
        self, block: FloryHugginsPairParameters, *, flipped: bool
    ) -> dict[str, float]:
        """One pair's coefficients, oriented so the first index is the lower one.

        ``flipped`` says the document wrote the pair the other way round. Nothing
        to do by default, where every coefficient is symmetric in the pair; a
        mixture with coefficients that are not says so.
        """

        return {key: float(getattr(block, key)) for key in self.coefficients()}


class FloryHugginsCoupling(Coupling[ParametersT]):
    """``(k_B_T/2) y^T H y`` over whatever state the mixture's pairs couple.

    Declared as an energy rather than as a table of potentials: a potential is
    ``dE/dy_r``, recovered by differentiation, so nothing has to know in advance
    which phase a coefficient came from or which row it lands in.

    Owns no fields: a statement about phases that already exist, it declares no
    elements and has nothing to transport. A concrete mixture supplies
    :meth:`hessian` and :meth:`slots`, which between them are the state space.
    """

    ## Overrides

    def name(self) -> str:
        return "flory_huggins"

    def energy_density(self) -> list[EnergyDensity]:
        convex, concave = self.convex_split(self.hessian())
        return [
            EnergyDensity(
                convex=self.quadratic(convex, self.slots("next")),
                concave=self.quadratic(concave, self.slots("prev")),
            )
        ]

    ## What the state space is

    @abstractmethod
    def hessian(self) -> np.ndarray:
        """``H``: the second derivative of the mixing energy in ``y``.

        Constant in ``y`` as a requirement: a plain array of numbers, with no
        coefficient and no time level in it. The split needs a *fixed*
        symmetric ``S`` with ``S - H >= 0``, and :meth:`convex_split` produces one
        once, from the spectrum of what this returns, before the mesh exists. A
        matrix that drifted with the state would leave the split taken against a
        majorant that no longer majorizes, and the unconditional energy descent
        with it.

        The requirement is on the *representation*, not on the physics. An
        interaction that varies as a mixture reacts is admissible provided the
        mixture finds a state space on which its Hessian is constant and declares
        it in :meth:`slots`. The crosslinking mixture does exactly that: it
        expands the entropy on the moments of its degree-of-polymerization
        distribution, which are affine in ``{c_x, u_b}``, and the Hessian of its
        pair interactions is constant on the resulting "FH subspace" -- which is
        why a coefficient that couples to the reacted-site fraction buys a second
        slot per phase rather than a state-dependent ``chi``. A mixture that cannot
        find such a space cannot be convex-split ahead of time and does not
        belong on this class.
        """

    @abstractmethod
    def slots(self, time_level: str) -> "list[Expr | None]":
        """``y``: the state ``H`` acts on, at one time level.

        Differentiation handles rather than raw functions, so that the term
        reaches the same rows the phases' own energies do; that is how a coupling
        writes into a phase's row without naming it. A slot may be ``None`` where
        a phase has no such state, and then contributes nothing.

        Whatever space this names is the one :meth:`hessian` must be constant on.
        """

    ## Shared machinery

    def convex_split(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split ``H`` into a convex part and the concave remainder.

        A positive semidefinite ``H`` is already convex and needs no split. Any
        negative eigenvalue -- the whole point of Flory-Huggins, demixing being a
        concave direction -- is handled by a spectral shift: ``rho I``
        above the spectrum as the convex part and ``rho I - H`` as the concave
        one, which makes the step unconditionally energy-decreasing at the cost
        of some implicit damping.
        """

        eigenvalues = np.linalg.eigvalsh(matrix)
        smallest = float(eigenvalues[0])
        spectrum = np.array2string(eigenvalues, precision=6)

        if smallest >= -CONVEXITY_TOLERANCE:
            logger.info(
                "fh_convex_split active=false min_eigenvalue=%.6e eigenvalues=%s",
                smallest,
                spectrum,
            )
            return matrix, np.zeros_like(matrix)

        largest = float(eigenvalues[-1])
        padding = max(1.0e-12, 1.0e-8 * max(1.0, abs(largest)))
        rho = max(largest, 0.0) + padding
        convex = rho * np.eye(matrix.shape[0])
        logger.info(
            "fh_convex_split active=true min_eigenvalue=%.6e max_eigenvalue=%.6e "
            "rho=%.6e eigenvalues=%s",
            smallest,
            largest,
            rho,
            spectrum,
        )
        return convex, convex - matrix

    def quadratic(self, matrix: np.ndarray, slots: "Iterable[Expr | None]") -> "Expr":
        """``(k_B_T/2) y^T A y``, expanded scalar by scalar.

        Written out rather than as ``dot(y, A y)`` on a ``ufl.as_vector`` because
        the potentials are recovered with :func:`ufl.diff` against a
        ``ufl.variable``, which has no handler for ``Dot``: the quadratic has to
        already be a sum of products. Zero coefficients are dropped rather than
        added, and a ``None`` slot contributes nothing.
        """

        slots = list(slots)
        total = ufl.as_ufl(0.0)
        for row, left in enumerate(slots):
            if left is None:
                continue
            for column, right in enumerate(slots):
                coefficient = float(matrix[row, column])
                if right is None or coefficient == 0.0:
                    continue
                total += coefficient * left * right
        return 0.5 * self.k_B_T * total
