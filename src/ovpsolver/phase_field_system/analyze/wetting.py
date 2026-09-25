"""Is the contact angle at the inclusion surfaces still the one the spec asked for?

``surface_affinity`` is a wetting energy per unit area, linear in the phase, and the
only thing in the spec that sets how the mixture meets an inclusion. It buys a
natural boundary condition rather than a constraint: with the diffuse surface
potential

    mu^Gamma = dg/dphi + kappa grad(phi).n_eps,

asking for an affinity ``a`` is asking for ``kappa grad(phi).n_eps = -k_B_T a`` at
the surface, and the contact angle follows from that slope.

Measured by an estimator of its own
-----------------------------------
The potential is written

    aux_potential = -kappa laplace(phi) + C / chi_eps,
    C = kappa grad(phi).grad(phi_incl) + (dg/dphi) dGamma_eps,

and ``C = 0`` is the condition holding. The row reads ``grad(phi).n_eps`` as a facet
sum or as the fitted slope of
:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.contact_slope`; this reads it from
the least-squares reconstruction of
:func:`~ovpsolver.fem.elements.dg0.dg0_least_squares_gradient_of`, independent of
both, which makes it a check on the row rather than a restatement of it. A two-point
difference along centroid joins near the surface normal would be cheaper and wrong:
it discards every facet that fails the alignment filter, and what survives is
selected by mesh orientation. The least-squares gradient is exact for a linear phase
on any mesh, and every cell in the band has one.

``grad(phi_incl)`` rather than a normal, because that is what the row contains and
because ``grad(phi_incl) = n_eps dGamma_eps`` carries the surface measure with it:
``C`` is a defect per unit area times the area there is, integrable against ``dx``
and bounded when divided by ``chi_eps``. The slope reported is ``C / dGamma_eps``
less the affinity, read where the surface density is near its peak so that no floor
enters.

The slope is read on the fluid side only. A wetting layer *peaks* at the surface,
so the normal derivative genuinely reverses sign across it, and a band straddling
the level set averages two real and opposite slopes into a near-cancellation that
is a fact about the band and not about the angle. The sharp-interface condition is
the limit from the fluid, so that is the side taken.

The absolute level is reported and not checked. The equilibrium the initial
condition solves for carries a far-field anchor
(:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.off_surface_weight_ufl`) pulling the
phase toward its declared volume fraction, which vanishes only exactly at the
surface-density peak and is back to four fifths of full strength half an ``eps``
out, competing with the wetting term across the band where the condition lives.
Runs come out near half the slope the spec asked for, which is the anchor and the
``O(eps)`` of the diffuse-domain approximation rather than a defect of the scheme;
a tolerance on it would be a tolerance on the regularization.

What *is* checked is what an angle set by the spec has no business doing: moving,
and acquiring a preferred direction. The angle is a material property, so the slope
should sit wherever the discretization puts it and stay; the drift is read against
the frame it happened at, because late in a run lateral motion in the layer can
outrun the condition. The angular structure is measured on ``C`` itself, by
:mod:`~ovpsolver.phase_field_system.analyze.mesh_mode`.

Beside them, because they move together, is how much phase is actually held: ``phi
at surface`` is the surface-weighted mean, and ``adsorption`` the excess over the far
field integrated outward, a length -- the thickness a layer of pure phase would need
to hold the same excess. A layer whose composition slides toward neutral while the
defect grows a symmetry is one story and not two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import ufl

from ...diffuse_domain.sdf import evaluate, phase as phase_of
from ...fem.elements.dg0 import (
    dg0_least_squares_gradient_of,
    mesh_cell_volumes,
)
from .base import Diagnostic, Unanswerable, declare_frames, frames_of
from .mesh_mode import MODE, Shells, axis_offset, lattice_axis
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from ...fem.save import Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Default for ``--band``: how far out of the zero level set a cell may sit and still
#: be measuring the surface, in multiples of ``eps``. Outward only, for the reason in
#: the module docstring. Half an ``eps`` out the diffuse surface density has fallen to
#: a fifth of its peak, so this is where the condition is imposed and little else is.
BAND = 0.5

#: Default for ``--reach``: how far out of the surface the adsorption integral runs,
#: in multiples of ``eps``. Far enough to contain the layer, near enough that it is
#: not measuring the bulk.
REACH = 4.0

#: Default for ``--drift-tol``: how far the surface slope may move from its initial
#: value, relative. While the imposed angle holds, the slope moves by three to five
#: per cent; once it stops holding, it goes on to move by twenty-two and then
#: forty-nine. A quarter passes the first and catches the second.
DRIFT_TOLERANCE = 0.25

#: Default for ``--mode-tol``: how large a six-fold modulation of the defect is
#: tolerable, as a fraction of the defect's own magnitude in the innermost shell.
#: Above the three tenths of one per cent a run holds while it is holding anything,
#: and well under the eighty-nine it reaches once it is not.
MODE_TOLERANCE = 0.2

#: Shell edges the angular structure of the defect is read in, in multiples of
#: ``eps``: the band the condition lives in and the two beyond it, kept apart because
#: the innermost can sit half a period away from those outside it, and averaging them
#: would report their difference.
SHELLS = (0.0, 1.0, 2.0, 4.0)


class Wetting(Diagnostic):
    """The imposed contact angle at the inclusions, and the layer it holds."""

    name = "analyze.wetting"
    summary = "check the inclusions impose the contact angle the spec asked for"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--drift-tol",
            type=float,
            dest="drift_tol",
            default=DRIFT_TOLERANCE,
            help="how far the surface slope may move from its initial value",
        )
        parser.add_argument(
            "--mode-tol",
            type=float,
            dest="mode_tol",
            default=MODE_TOLERANCE,
            help=f"how large an m={MODE} modulation of the defect is tolerable",
        )
        parser.add_argument(
            "--band",
            type=float,
            default=BAND,
            help="how far out of the surface to read the slope, in multiples of eps",
        )
        parser.add_argument(
            "--reach",
            type=float,
            default=REACH,
            help="how far out the adsorption integral runs, in multiples of eps",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        drift_tol: float = DRIFT_TOLERANCE,
        mode_tol: float = MODE_TOLERANCE,
        band: float = BAND,
        reach: float = REACH,
        frames: "list[int] | None" = None,
        stride: int = 1,
        **options: Any,
    ) -> None:
        domain = parameters.solver.diffuse_domain
        if not domain.has_inclusions(run.mesh):
            raise Unanswerable(
                "this run declares no inclusions, so there is no surface for a "
                "phase to wet and no contact angle to measure"
            )

        wetting = [
            phase
            for phase in parameters.phase_field_parameters
            if float(phase.surface_affinity) != 0.0
        ]
        if not wetting:
            raise Unanswerable(
                "every phase has 'surface_affinity' zero, so the spec asked "
                "for a neutral angle and there is nothing imposed to check"
            )
        missing = [phase.name for phase in wetting if not run.has(f"{phase.name}.phi")]
        if missing:
            raise Unanswerable(
                f"{', '.join(missing)} set an affinity but did not save 'phi', which "
                f"is the field the condition is about"
            )

        shells = Shells(run, domain, SHELLS)
        geometry = self._geometry(run, domain, shells, band)
        layer = self._layer(run, shells, reach)
        _, axis = lattice_axis(run.mesh, MODE)

        report.field("inclusions", f"{shells.inclusions} from {domain.geometry_path.name}")
        report.field("interface width", f"{2.0 * shells.eps:.4g}")
        report.field(
            "slope read on",
            f"{int(geometry['band'].sum())} cells, 0 to {band:g} eps outside the "
            f"level set, weighted by dGamma_eps",
        )
        report.field(
            "estimator",
            "C = kappa grad(phi).grad(phi_incl) + (dg/dphi) dGamma_eps, on a "
            "least-squares gradient independent of the row's",
        )
        report.field("mesh", f"m={MODE} axis of the facet normals at {axis:.2f} degrees")
        report.note()

        for phase in wetting:
            self._one_phase(
                report,
                run,
                parameters,
                phase,
                shells,
                geometry,
                layer,
                axis=axis,
                frames=frames,
                stride=stride,
                drift_tol=drift_tol,
                mode_tol=mode_tol,
            )
        report.note(
            "the absolute level is reported and not checked: the equilibrium solved "
            "for carries a far-field anchor that competes with the wetting term "
            "across this band, so about half the asked-for slope is the "
            "regularisation and not a defect. Drift and angular structure are the "
            "columns that carry a verdict."
        )

    ## Private helpers

    def _one_phase(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        phase: Any,
        shells: Shells,
        geometry: dict,
        layer: dict,
        *,
        axis: float,
        frames: "list[int] | None",
        stride: int,
        drift_tol: float,
        mode_tol: float,
    ) -> None:
        kappa = float(phase.kappa)
        affinity = float(phase.surface_affinity)
        target = -float(parameters.k_B_T) * affinity

        report.note(
            f"{phase.name}: affinity {affinity:+g}, kappa {kappa:g}, so the spec asks "
            f"for kappa dphi/dn = {target:+.4g} "
            f"({'wets' if affinity < 0 else 'is repelled by'} the inclusions)"
        )
        table = report.table(
            Table(
                f"{phase.name} frame",
                "time",
                "kappa dphi/dn",
                "of target",
                "drift",
                "phi at surface",
                "adsorption",
                f"m={MODE}",
                "vs nbrs",
                "off axis",
                formats={
                    f"m={MODE}": ".2%",
                    "vs nbrs": ".1f",
                    "off axis": ".1f",
                    "drift": ".2f",
                },
            )
        )

        times = np.asarray(run.times, dtype=float)
        cells = run.cell_space()
        handle = run.field(f"{phase.name}.phi")
        solver = run.solver_parameters
        weight = geometry["weight"]
        first: float | None = None
        drift = mode = 0.0
        drift_frame = mode_frame = 0

        for frame in frames_of(run, frames, stride=stride):
            values = handle.sample(frame, cells)
            gradient = np.asarray(
                dg0_least_squares_gradient_of(solver, values).x.array, dtype=float
            ).reshape(-1, run.mesh.geometry.dim)
            korteweg = kappa * np.sum(gradient * geometry["grad_incl"], axis=1)
            defect = korteweg + float(parameters.k_B_T) * affinity * geometry["dgamma"]
            slope = (korteweg / geometry["dgamma_positive"])[geometry["band"]]

            mean = float(np.average(slope, weights=weight))
            if first is None:
                first = mean
            moved = abs(mean - first) / abs(first) if first else float("nan")
            modulation, selectivity, offset = shells.innermost_structure(
                defect / abs(target)
            )
            if moved > drift:
                drift, drift_frame = moved, frame
            if modulation > mode:
                mode, mode_frame = modulation, frame

            array = np.asarray(values.x.array, dtype=float)
            table.add(
                frame,
                float(times[frame]),
                mean,
                mean / target,
                moved,
                float(np.average(array[geometry["band"]], weights=weight)),
                self._adsorption(array, layer),
                modulation,
                selectivity,
                axis_offset(offset, axis, MODE),
            )

        report.check(
            drift <= drift_tol,
            f"{phase.name}: the imposed angle holds through the run",
            f"the surface slope moved {drift:.0%} from its initial value by frame "
            f"{drift_frame}, against {drift_tol:.0%}; a contact angle set by the spec "
            f"is a material property and should not depend on the time",
        )
        report.check(
            mode <= mode_tol,
            f"{phase.name}: the imposed angle has no preferred direction",
            f"an m={MODE} modulation of {mode:.0%} by frame {mode_frame}, against "
            f"{mode_tol:.0%}; the inclusion is a circle, so a six-fold angle is the "
            f"mesh's symmetry and not the geometry's",
        )

    def _adsorption(self, values: np.ndarray, layer: dict) -> float:
        """Excess phase held outside the inclusions, as a thickness.

        The excess over the far field, integrated over the cells just outside the
        surface and divided by the surface's own length: the thickness a layer of
        pure phase would need to hold the same excess, to be read against the
        interface width. The far field is this frame's own mean well away from every
        inclusion, so once the bulk demixes this is an excess over the mixture's mean
        rather than over a plateau -- still the right comparison for whether the
        layer is holding, and why the column sits beside the slope.
        """

        far = float(np.average(values[layer["far"]], weights=layer["far_volume"]))
        excess = (values[layer["cell"]] - far) * layer["volume"]
        return float(excess.sum() / layer["perimeter"])

    def _geometry(self, run: "Run", domain: Any, shells: Shells, band: float) -> dict:
        """``grad phi_incl`` and ``dGamma_eps`` per cell, and the band to weight by.

        The expressions and not the run's tabulated copies, since the tabulated ones
        live on a quadrature element and a centroid is not one of its points. At a
        centroid the expression is what the copy would hold, evaluated exactly --
        interpolating onto the cell-constant space *is* evaluation there.
        """

        mesh = run.mesh
        dim = mesh.geometry.dim
        gradient = ufl.grad(
            sum(
                phase_of(distance, shells.eps)
                for distance in domain.signed_distances_for(mesh)
            )
        )
        dgamma = evaluate(ufl.sqrt(ufl.dot(gradient, gradient)), run.cell_space())
        inside = (shells.gap >= 0.0) & (shells.gap <= band)
        if not inside.any():
            raise Unanswerable(
                "no cell sits within '--band' multiples of eps outside a surface; "
                "the mesh is coarse against 'domain_eps'"
            )
        return {
            "grad_incl": evaluate(gradient, run.cell_space((dim,))).reshape(-1, dim),
            "dgamma": dgamma,
            "dgamma_positive": np.where(dgamma > 0.0, dgamma, 1.0),
            "band": inside,
            "weight": dgamma[inside]
            * mesh_cell_volumes(mesh, quadrature_degree=run.quadrature_degree)[inside],
        }

    def _layer(self, run: "Run", shells: Shells, reach: float) -> dict:
        """The cells the adsorption integral runs over, and the ones it calls far.

        Outside the level set only, so the integral is over fluid rather than over the
        inside of an inclusion, where the phase is regularised and means nothing.
        """

        volume = mesh_cell_volumes(
            run.mesh, quadrature_degree=run.quadrature_degree
        )
        inside = (shells.gap >= 0.0) & (shells.gap <= reach)
        outside = shells.gap > 2.0 * reach
        if not inside.any() or not outside.any():
            raise Unanswerable(
                "the inclusions leave no band to measure adsorption in, or no far "
                "field to measure it against; the domain is small against them"
            )
        return {
            "cell": inside,
            "volume": volume[inside],
            "far": outside,
            "far_volume": volume[outside],
            "perimeter": run.parameters.solver.diffuse_domain.diffuse_perimeter(run.mesh),
        }
