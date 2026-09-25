"""How much phase crosses the inclusion surfaces, against how much may.

A diagnostic of one term:
:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.surface_crossing_penalty`,
whose crossing per unit volume is

    h = phi (v . grad(phi_incl)) + nu q_eps

-- the achieved crossing less the injection the spec permits, so ``h = 0`` is the
flux condition holding and a piece that injects nothing states non-crossing. Every
number below is built from that one ``h``, assembled the way the term is: at
``flux_bc_quadrature``, on the live P2 velocity against ``grad(phi_incl)``, over
the term's whole support. All three matter. A cell-averaged velocity read at
centroids destroys exactly the variation the cell integral is taken over; a unit
normal in place of ``grad(phi_incl)`` rescales by ``dGamma_eps``, which varies by an
order of magnitude across a band cell; and a band outside the level set omits the
half of the support inside the inclusions, where the term is assembled on plain
``dx`` and where the trouble tends to live.

What is measured: the cell integral, and only that
--------------------------------------------------
One number per frame per flux piece,

    crossing = sum_T |int_T h dx|.

For this scheme that *is* the crossing. The phases are DG0, so only a cell integral
can move material, and the condition is held by a cell-constant multiplier, which
pins ``int_T h`` one cell at a time. A pointwise ``int |h|`` also counts flux that
reverses inside a cell: real motion at the quadrature points, but motion no
transport row can read and no multiplier is asked to hold, so it measures the
quadrature rule rather than the solve.

Every fraction is of the flux *arriving* at the surface, ``int phi |v| dGamma_eps
dx``, so a percentage is the share of what reached the surface that went through
instead of turning along it. The domain-wide ``int_Omega phi |v|`` would mostly
report how much of the domain is band.

**That fraction has a null, and it is not zero.** Since ``grad(phi_incl) = n_eps
dGamma_eps``, a crossing resolved across a band cell integrates to the
surface-weighted mean of ``|cos(v, n_eps)|``, which for a direction field with no
preference at all is ``2/pi`` in 2D and ``1/2`` in 3D (:func:`isotropic_crossing`).
So the scale runs from zero, the surfaces turning the flow entirely along
themselves, to about ``0.64``, the surfaces doing nothing to it; a run at the null
has an inert membrane, not a moderate leak. The report prints the null beside the
column, because the number is unreadable without it.

**A column far below the null is not by itself a membrane that holds.** A cell
integral cancels a crossing that reverses within the cell, so a velocity carrying
structure at the grid scale reads near zero here while every quadrature point of
the band is crossing. What separates the two is not a finer measurement of this
term -- the reversing part is exactly what the scheme does not constrain -- but
whether the velocity has grid-scale structure at all, which is
:mod:`~ovpsolver.phase_field_system.analyze.mesh_correlation`'s question.

The term is written against each piece of a phase's flux separately -- a
polymerizing phase's sol and gel ride different velocities and carry different
densities -- so this reads the pieces
(:data:`~ovpsolver.phase_field_system.analyze.base.PIECES`), not one velocity per
phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import ufl
from dolfinx import fem

from ...diffuse_domain.sdf import phase as phase_of
from .base import (
    Diagnostic,
    Unanswerable,
    declare_frames,
    flux_pieces,
    frames_of,
)
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Field, Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--leak-tol``: how much of the phase arriving at a surface may
#: cross it, cell by cell, as a fraction of the flux arriving. The membrane has
#: finite permeability by construction, so this is a statement about how finite.
LEAK_TOLERANCE = 0.05


def isotropic_crossing(dim: int) -> float:
    """``<|cos theta|>`` for directions with no preference, in ``dim`` dimensions.

    What the crossing column reads for a velocity the surfaces do not turn at
    all and that is resolved across a band cell, and so the top of its useful
    range rather than one. Zero is a membrane that holds, or a crossing that
    reverses within the cell; this is a membrane that is not there.
    """

    if dim == 2:
        return 2.0 / np.pi
    if dim == 3:
        return 0.5
    raise ValueError(f"no isotropic crossing fraction in {dim} dimensions")


class SurfaceLeakage(Diagnostic):
    """How much phase crosses the inclusion surfaces, against how much may."""

    name = "analyze.surface_leakage"
    summary = "how much phase crosses the inclusion surfaces"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--leak-tol",
            type=float,
            default=LEAK_TOLERANCE,
            help="cell-integrated crossing allowed, as a fraction of the flux "
            f"arriving (default: {LEAK_TOLERANCE})",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        frames: "list[int] | None" = None,
        stride: int = 1,
        leak_tol: float = LEAK_TOLERANCE,
        **options: Any,
    ) -> None:
        domain = parameters.solver.diffuse_domain
        if not domain.has_inclusions(run.mesh):
            raise Unanswerable(
                "this run declares no inclusions, so there is no diffuse "
                "surface for a flux to leak across"
            )
        geometry = self.geometry(run, domain)
        report.field(
            "crossing measured",
            "sum_T |int_T h|, the cell integral the multiplier pins",
        )
        report.field(
            "quadrature", f"degree {domain.flux_bc_quadrature}, the term's own rule"
        )
        report.field(
            "permeability", f"varsigma = {float(domain.surface_permeability):g}"
        )
        null = isotropic_crossing(run.mesh.geometry.dim)
        report.field(
            "unconstrained flow crosses",
            f"{null:.1%} of what arrives; the crossing column runs from 0, where "
            f"the surfaces turn the flow along themselves, to this -- and falls "
            f"below it also where the crossing reverses within a cell",
        )
        report.note()

        walk = frames_of(run, frames, stride=stride)
        answered = False
        for phase in parameters.phase_field_parameters:
            pieces = flux_pieces(run, phase.name)
            if pieces is None:
                continue
            answered = True
            for index, (velocity, carried) in enumerate(pieces):
                self.one_piece(
                    report,
                    run,
                    geometry,
                    phase,
                    velocity,
                    carried,
                    injects=index == 0,
                    walk=walk,
                    leak_tol=leak_tol,
                )
        if not answered:
            raise Unanswerable(
                "no phase saved a velocity beside the density it carries; this "
                "reads the crossing each flux piece makes, so a spec has to save "
                "both halves of at least one piece"
            )

    ## Public

    def one_piece(
        self,
        report: "Report",
        run: "Run",
        geometry: dict,
        phase: Any,
        velocity: str,
        carried: str,
        *,
        injects: bool,
        walk: tuple[int, ...],
        leak_tol: float,
    ) -> None:
        label = f"{phase.name}.{velocity}"
        imposed = (
            float(getattr(phase, "surface_flux_density", 0.0))
            * float(getattr(phase, "monomer_volume", 1.0))
            if injects
            else 0.0
        )
        report.note(
            f"{label} carries {phase.name}.{carried}; the spec permits "
            f"nu q = {imposed:+.4g} per unit area"
            + (
                " -- nothing may cross, so the whole crossing is the defect"
                if imposed == 0.0
                else ""
            )
        )

        flow = run.field(f"{phase.name}.{velocity}")
        density = run.field(f"{phase.name}.{carried}")
        forms = self.forms(run, geometry, flow, density, imposed)

        table = report.table(
            Table(
                f"{label} frame",
                "time",
                "crossing",
                "of arriving",
                "net",
                formats={
                    "crossing": ".4e",
                    "of arriving": ".3%",
                    "net": "+.3%",
                },
            )
        )
        worst_leak, leak_frame = 0.0, walk[0]
        for frame in walk:
            flow.load_into(forms["flow"], frame)
            density.load_into(forms["density"], frame)
            per_cell = np.asarray(fem.assemble_vector(forms["crossing"]).array)
            arriving = float(
                np.asarray(fem.assemble_vector(forms["arriving"]).array).sum()
            )
            crossing = float(np.abs(per_cell).sum())

            def share(value: float) -> float:
                return value / arriving if arriving > 0.0 else float("nan")

            table.add(
                frame,
                float(run.times[frame]),
                crossing,
                share(crossing),
                share(float(per_cell.sum())),
            )
            if arriving > 0.0 and share(crossing) > worst_leak:
                worst_leak, leak_frame = share(crossing), frame

        report.check(
            worst_leak <= leak_tol,
            f"{label}: the surfaces hold what arrives at them",
            f"{worst_leak:.3%} of the arriving flux crosses, cell by cell, at "
            f"frame {leak_frame}, against {leak_tol:.2%} allowed and "
            f"{isotropic_crossing(run.mesh.geometry.dim):.1%} for a flow the "
            f"surfaces do not turn at all",
        )

    def forms(
        self,
        run: "Run",
        geometry: dict,
        flow: "Field",
        density: "Field",
        imposed: float,
    ) -> dict:
        """``h`` per cell, and the flux arriving, on the term's own rule.

        Built once per piece against functions refilled in place each frame, so
        the kernels are compiled once rather than once a frame.
        """

        flow_at = flow.function(0)
        density_at = density.function(0)
        test = ufl.TestFunction(run.cell_space())
        dx = geometry["dx"]

        crossing = density_at * ufl.dot(flow_at, geometry["grad_incl"])
        if imposed != 0.0:
            crossing = crossing + imposed * geometry["dgamma"]
        arriving = (
            density_at * ufl.sqrt(ufl.dot(flow_at, flow_at)) * geometry["dgamma"]
        )
        return {
            "flow": flow_at,
            "density": density_at,
            "crossing": fem.form(crossing * test * dx),
            "arriving": fem.form(arriving * test * dx),
        }

    def geometry(self, run: "Run", domain: Any) -> dict:
        """``grad(phi_incl)``, the surface density, and the term's own measure.

        Expressions rather than the run's tabulated copies: a reader has no
        ``DiffuseDomain`` to ask, and the geometry is analytic anyway.
        ``dgamma_eps`` is summed per inclusion rather than taken as the norm of the
        summed gradient, which counts two overlapping bands once.
        """

        eps = float(domain.domain_eps)
        mesh = run.mesh
        distances = domain.signed_distances_for(mesh)
        smeared = [phase_of(one, eps) for one in distances]
        return {
            "grad_incl": ufl.grad(sum(smeared)),
            "dgamma": sum(
                ufl.sqrt(ufl.dot(ufl.grad(one), ufl.grad(one))) for one in smeared
            ),
            "dx": ufl.Measure(
                "dx",
                domain=mesh,
                metadata={"quadrature_degree": int(domain.flux_bc_quadrature)},
            ),
        }
