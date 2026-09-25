"""Additive flux pieces, and the conservation law a list of them states.

Every transported quantity in the mixture has a flux of the form

    J = sum_alpha v_alpha rho_alpha,

a sum of velocities against the densities they carry, with no relation assumed
among the densities. A :class:`Flux` is one term of that sum. The upwind
selector needs exactly this decomposition: it picks a direction per velocity
and takes the matching density from the upwind cell, so what matters is which
density rides which velocity, not what they add up to.

Pieces, and not fractions of one total. The sol/gel split of ``c_x`` is a
difference of state variables and a fraction of the total only by division, and
every consumer of a flux either differentiates it against a velocity or takes its
upwind trace, both of which want the pieces. Additive pieces also make the
partition exact: a phase's sol piece is ``nu m_1`` and its gel piece the
remainder, so the two sum to the phase identically and the saturation constraint
sees the same total whatever the internal state does. A fraction times ``phi``
would leave a residue of the division in the sum.

Each piece carries two views of its velocity, which are different objects rather
than time levels of one:

``live_velocity``
    The rate unknown. Supplies the flux magnitude everywhere -- the transport
    row, the energy release rate, the volumetric flux the pressure constrains.
``lagged_velocity``
    The last solved value of the same rate. Used where only the *sign* of the
    velocity is wanted -- the upwind selector -- so that the rate residual stays
    affine in the rate and the energy estimate survives.

The density is lagged too, and named so because nothing can check it: explicit
in the density against an implicit accumulation is what makes transport
positivity-preserving under a step bound.

There is no view for the test function. Everything a flux enters is a term of
the Rayleighian, linear in the velocity, so a row is what differentiation makes
of the term.

Each piece also carries what it injects. The model injects only at the diffuse
inclusion surfaces and at ``Sigma`` -- there are no bulk sources on ``Omega`` --
so a list of pieces states the whole conservation law

    F(chi rho) + div(chi sum_alpha v_alpha rho_alpha) = sum_alpha s_alpha,

and everything built for a transported variable follows from the list alone. The
impermeability constraints read the injections back, as the amount a phase may
cross the diffuse surfaces by and the value its normal flux through ``Sigma`` is
pinned to, so a velocity that injects nothing states non-crossing without being
asked.

The two injections are named for where they act, and differ in units.
``surface_flux`` is per unit *bulk volume*: the inclusion surfaces exist only as
diffuse bands, so the spec's per-area number arrives smeared by the
diffuse-domain surface measure, and is integrated over ``Omega``.
``boundary_flux`` is per unit *boundary area*: ``Sigma`` is a real surface of the
mesh, so the spec's number is used as it is, and is integrated over ``Sigma``.
The spec's own numbers carry a ``_density`` suffix, marking a flux per unit area
(:class:`~ovpsolver.phase_field_system.phase_field.PhaseFieldParameters`).
Converting a smeared flux back to one per unit area is a division by the surface
density, which taken where it is used produces mesh artefacts, so nothing here
derives either from the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ufl.core.expr import Expr


@dataclass(frozen=True)
class Flux:
    """One velocity's term of a transported quantity's flux.

    Parameters
    ----------
    live_velocity : the rate unknown ``v_alpha`` (e.g. ``rates.v``).
    lagged_velocity : the last solved ``v_alpha`` (e.g. ``prev.v``), declared by
        ``StaticElement(..., store_lagged=True)``.
    lagged_density : the quantity ``rho_alpha`` this velocity carries, at the
        previous time level: a volume fraction for a phase, and a modulus, a
        crosslink moment or a strain for the internal states.
    surface_flux : what this piece injects at the inclusion surfaces, per unit
        bulk volume and time: the spec's per-area ``nu q``, smeared by the
        diffuse-domain surface measure. A source in the variable's own row that
        pairs against no velocity; it sits on the piece because the influx
        arrives with one of the velocities.

        Smeared per inclusion, ``nu q sum_a |grad phi_a|``, and not off the
        aggregate density: where two diffuse bands overlap the aggregate is the
        norm of a sum and would count the crossing region once where the phase
        crosses two surfaces there.
    boundary_flux : what this piece injects through the outer boundary
        ``Sigma``, per unit boundary area and time: the spec's number as it is,
        since ``Sigma`` is a real surface of the mesh.

        Doubles as the target of that velocity's ``Sigma`` constraint, which is
        what lets the transport row substitute the constrained value instead of
        integrating the flux through ``Sigma``. Which multiplier pins it is not
        stated here: a phase field pins each velocity on the one piece that
        answers it, so a piece carrying an internal state along an
        already-constrained velocity imposes nothing.
    """

    live_velocity: "Expr"
    lagged_velocity: "Expr"
    lagged_density: "Expr | float" = 1.0
    surface_flux: "Expr | float" = 0.0
    boundary_flux: "Expr | float" = 0.0
