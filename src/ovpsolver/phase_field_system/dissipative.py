"""Contributing a term to the mixture's Rayleighian.

``R = dE/dt + Psi + C`` is assembled from three kinds of
contributor -- a phase field, a cross-phase coupling, and the mixture itself --
which share nothing but this interface. Every method defaults to an empty form,
so a contributor writes only the terms it has, and subclassing is the
declaration: a class that is not a :class:`Dissipative` cannot enter ``R`` by
growing a method with the right name.

Nothing here is a row. The declarations are summed over every contributor and
differentiated once, because the pressure enforcing saturation and the
interpenetration drag couple every phase to every other, and differentiating per
contributor would drop exactly those cross terms. A term written in another
object's velocity thereby enters that object's row without naming it.

:meth:`energy_density` returns densities, which carry no measure; the other
three return integrated forms.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import ufl

if TYPE_CHECKING:
    from ..solver.parameters import SolverParameters
    from .energy import EnergyDensity


class Dissipative(ABC):
    """One contributor to ``R = dE/dt + Psi + C``."""

    #: The mesh and step every declaration is written against, set by whatever
    #: else the contributor inherits. Needed here only so :meth:`nothing` can
    #: name a measure.
    solver_parameters: "SolverParameters"

    ## Declarations, summed into R and differentiated once

    def energy_density(self) -> list["EnergyDensity"]:
        """Free-energy densities this contributor owns, already convex-split.

        Varying one against a variable gives that variable's chemical potential,
        the convex part taken at ``k+1`` and the concave part at ``k``. That
        split is what makes the step's energy decrease unconditional, and it is
        declared rather than derived, since nothing can check that an expression
        is convex.

        A contributor that stores no energy declares none.
        """

        return []

    def energy_rate(self, terms: list["EnergyDensity"]) -> ufl.Form:
        """``dE/dt``: the rate this contributor releases energy, in the live rates.

        Derived from ``terms`` wherever the energy is stored in a variable the rate
        solve carries: each such variable's transport paired against its
        potential. Overridden where it is not: a strain moment or a locking
        modulus is lagged, so there is no ``k+1`` value to vary against and the
        semi-implicit stress power is itself the declaration.

        ``terms`` is the *mixture's* whole energy, because a variable's potential
        is the derivative of all of it: a Flory-Huggins coupling is declared by
        the system in the phases' own differentiation handles and reaches their
        rows only this way.
        """

        return self.nothing()

    def dissipation(self) -> ufl.Form:
        """``Psi``: the friction this contributor pays, quadratic in the live rates.

        Half the quadratic form, so varying it against a rate's test function gives
        that rate's friction row. Being the natural norm of the rate problem, it is
        also the scale the excess Rayleighian is measured against
        (:mod:`ovpsolver.solver.onsager`).
        """

        return self.nothing()

    def constraints(self) -> ufl.Form:
        """``C``: multiplier potentials for what this contributor forbids.

        Hard constraints only. A constraint relaxed to a finite permeability is a
        genuine dissipation and is declared as one, even when written through its
        multiplier as a Legendre transform: wherever that multiplier is
        stationary, the transform is worth the dissipation.
        """

        return self.nothing()

    def rayleighian(self) -> ufl.Form:
        """``R = dE/dt + Psi + C`` over this contributor's own declarations.

        For the mixture, whose :meth:`energy_density` is the whole energy, this
        is the Rayleighian the rate problem differentiates.
        """

        return (
            self.energy_rate(self.energy_density())
            + self.dissipation()
            + self.constraints()
        )

    ## Rows, stated rather than differentiated

    def aux_residual(self) -> ufl.Form:
        """Rows defining this contributor's auxiliary unknowns, already tested.

        An auxiliary unknown holds a potential that has no pointwise value. On a
        cell-constant mixture that is every derivative of the state: a DG0 field
        has no interior gradient or Laplacian, so each reaches a row as a sum over
        neighbouring cell pairs and needs a variable to live in before a flux can
        be paired against it. Each row is an identity between an unknown and a
        facet sum rather than a stationarity, so the system adds them to the rate
        solve verbatim.
        """

        return self.nothing()

    ## Helpers

    def nothing(self) -> ufl.Form:
        """A zero form on the bulk measure.

        A form rather than ``0``, since these are summed with real forms and UFL
        will not add a form to an integer.
        """

        return ufl.as_ufl(0.0) * self.solver_parameters.get_mesh_dx()
