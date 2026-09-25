"""Declared free-energy terms of a phase field, and where their potentials go.

A phase field declares its energy as a list of :class:`EnergyDensity`, each already
split into the convex part evaluated at ``k+1`` and the concave part evaluated at
``k``. Nothing here assembles: a term is a scalar UFL expression in
``ufl.variable``-wrapped fields, and everything the rows need is obtained by
differentiating it against the variable whose row is being built.

A term may also be declared by its tangent, a potential paired with its field,
when the potential is what the discrete space can hold and the energy is not.
The gradient penalty is the case: a cell-constant phase has no gradient, so the
Laplacian in its chemical potential lives in an auxiliary unknown defined by its
own row, and ``aux_potential phi`` hands it to the rows by the same
differentiation as every other term
(:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.energy_density`).

The energy rate of moving a field is ``-int mu div J``, and every state of the
mixture is cell-constant, so that integral has one discretization: the cell-wise
divergence theorem against an upwind facet flux
(:func:`ovpsolver.fem.elements.dg0.dg0_upwind_ibp`), which reads the potential as
its jump across a facet. :func:`require_facet_potential` refuses a coefficient
the jump would say nothing about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ufl
from ufl.algorithms import strip_variables
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.constantvalue import Zero
from ufl.variable import Variable

from ..fem.elements import ElementDomain

if TYPE_CHECKING:
    from dolfinx.fem import Function
    from ufl.core.expr import Expr


@dataclass(frozen=True)
class EnergyDensity:
    """One convex-split contribution to the free energy, per unit volume.

    A *density* and never an integral. ``domain`` names the measure the term
    belongs under and the consumer applies it -- ``chi_eps dx`` in the bulk,
    ``dgamma_eps dx`` on the diffuse surface -- so a declaration that multiplied
    by ``dx`` itself would be integrated twice, silently. That is checked rather
    than left to convention, because the failure is invisible in the assembled
    form.

    The split is the declaration: ``convex`` is evaluated at ``k+1`` and
    ``concave`` at ``k`` exactly as written, and the total is ``convex -
    concave``. The unconditional energy decrease of the step rests on *both*
    expressions being convex, the subtracted one included, which is what
    :meth:`~ovpsolver.phase_field_system.couplings.FloryHugginsCoupling.convex_split`
    returns.

    Both are written in the ``ufl.variable`` wrappers handed out by
    :meth:`~ovpsolver.fem.elements.ElementOwner.variable`, since the potentials
    are recovered from them by :func:`ufl.diff`. The wrappers are memoized per
    owner, name and time level, so terms declared by different objects
    differentiate consistently.

    Parameters
    ----------
    convex : the part evaluated at the new time level.
    concave : the part subtracted, evaluated at the old one; defaults to zero.
    domain : where the energy density lives, selecting the measure it is
        integrated against. A wetting affinity is declared on
        ``DIFFUSE_SURFACE``, and its derivative is what appears in the
        chemical-potential row as the natural boundary condition on ``phi``.
    """

    convex: "Expr | float" = 0.0
    concave: "Expr | float" = 0.0
    domain: ElementDomain = ElementDomain.BULK

    def __post_init__(self) -> None:
        for name in ("convex", "concave"):
            if isinstance(getattr(self, name), ufl.Form):
                raise TypeError(
                    f"{name} is a Form, so it already carries a measure. An "
                    "EnergyDensity is a density: the measure belongs to the "
                    f"domain it declares, and {name} * dx would be integrated "
                    "again wherever this term is used. Drop the measure"
                )

    def potential(self, next_variable: "Expr", prev_variable: "Expr") -> "Expr":
        """The chemical potential this term contributes to a variable's row.

        ``next_variable`` and ``prev_variable`` are the two time levels of the
        *same* variable, so the result is the convex-split derivative
        ``dE_convex/dx|_{k+1} - dE_concave/dx|_k``. A term not depending on the
        variable differentiates to zero and drops out.

        The derivative is applied rather than left symbolic, because whether the
        facet rule can read a potential depends on the fields it survives
        differentiation with (:func:`require_facet_potential`), not on the fields
        the term mentions.
        """

        derivative = ufl.diff(ufl.as_ufl(self.convex), next_variable) - ufl.diff(
            ufl.as_ufl(self.concave), prev_variable
        )
        return strip_variables(apply_derivatives(derivative))

    def bregman_divergence(self, lag_to_live: "dict[Function, Function]") -> "Expr":
        """What lagging the concave part costs this term, as an energy density.

        The step replaces the convex ``concave`` by its tangent at ``k``, and a
        convex function lies above its tangent, so the increment the scheme
        consumes majorizes the true one by exactly the Bregman divergence

            D(x_{k+1}, x_k) = E_e(x_{k+1}) - E_e(x_k) - E_e'(x_k).(x_{k+1} - x_k),

        which is nonnegative and vanishes with the increment.

        The numerical dissipation the time discretization adds on top of the
        flow: a verdict on ``dt``, where the excess Rayleighian of the Newton
        problem goes to zero at convergence and is a verdict on the solve.
        Reporting both keeps one from being read as the other.

        ``lag_to_live`` is :meth:`~ovpsolver.fem.elements.ElementOwner.lag_to_live`,
        supplied because the concave part is declared purely in lagged handles
        and holds no reference to the level it is compared with. A density, so
        the caller applies the measure ``domain`` names.
        """

        concave = ufl.as_ufl(self.concave)
        if isinstance(concave, Zero):
            return ufl.as_ufl(0.0)

        tangent = ufl.as_ufl(0.0)
        # In creation order: extract_type returns a set, whose order follows the
        # per-process string hash seed, and would give the form a different
        # signature (and a fresh JIT compile) on every run.
        variables = extract_type(concave, Variable)
        for variable in sorted(variables, key=lambda handle: handle.label().count()):
            lagged = variable.expression()
            live = lag_to_live.get(lagged)
            if live is None:
                raise ValueError(
                    f"the concave part of an energy term is written in {lagged}, "
                    "which is not a declared variable of the mixture, so there "
                    "is no k+1 value to measure the lag against"
                )
            tangent = tangent + ufl.inner(ufl.diff(concave, variable), live - lagged)

        at_prev = strip_variables(concave)
        at_next = ufl.replace(at_prev, lag_to_live)
        return at_next - at_prev - strip_variables(apply_derivatives(tangent))


def require_facet_potential(potential: "Expr") -> None:
    """Refuse a potential the facet rule cannot integrate.

    Every coefficient has to be cell-constant. Constants pass by depending on no
    field at all: their jump is zero, so they contribute only the boundary term,
    which is what a term affine in the state -- a bulk affinity -- should
    contribute.

    A *continuous* coefficient is the failure worth naming, because it would not
    fail on its own: its jump is identically zero, so the rule would silently
    drop the whole term and the potential would release no energy. A
    quadrature-family coefficient fails the other way, with no value on a facet
    at all.

    Raises
    ------
    ValueError
        If any coefficient is not cell-constant, naming which.
    """

    offenders = [
        f"{coefficient} ({coefficient.ufl_element()})"
        for coefficient in ufl.algorithms.extract_coefficients(potential)
        if coefficient.ufl_element().embedded_superdegree != 0
    ]
    if offenders:
        raise ValueError(
            f"potential depends on {', '.join(offenders)}, which is not "
            "cell-constant. Every state of the mixture is DG0 and the energy "
            "rate is the cell-wise divergence theorem against an upwind facet "
            "flux, whose only reading of a potential is its jump across a "
            "facet: a continuous coefficient has none and a quadrature one has "
            "no facet value at all. Declare the term in the states themselves"
        )
