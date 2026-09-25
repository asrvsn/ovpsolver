"""Fixed diffuse domain: the background measure the mixture lives on.

The mixture occupies the complement of the inclusions, smeared over
``2 domain_eps``. The inclusion surfaces are prescribed and no rate variable moves
them, so unlike a :class:`~ovpsolver.phase_field_system.phase_field.PhaseField`
this carries no ``prev``/``next`` pair and takes no part in the solve: it is a set
of coefficients built once and read by every residual.

``chi_eps``
    Quadrature-element indicator of the free (mixture-bearing) region, floored at
    ``indicator_floor`` so that a row it weights stays nonsingular inside the
    inclusions. Cell terms read it; facet terms read
    :meth:`DiffuseDomain.chi_eps_analytic_ufl`, since a quadrature element has no
    value on a facet; offline readers evaluate
    :meth:`~ovpsolver.diffuse_domain.DiffuseDomainParameters.chi_eps_on`. All three
    are the one expression, exact at their own points. No form divides by it: the
    surface kernels that read as quotients against it are cancelled by hand
    (:meth:`DiffuseDomain.grad_phi_incl_over_chi_eps_analytic_ufl`), because a
    floor makes such a quotient finite, not right.
``grad_phi_incl``
    ``grad phi_incl = n_eps dGamma_eps`` on the bulk rule. A term assembled on
    another rule asks :meth:`DiffuseDomain.grad_phi_incl_degree` for its own.
``dgamma_eps``
    Diffuse surface density ``|grad phi_incl|``, with ``dgamma_eps_max`` its
    global maximum as a Constant.
``q_eps``
    Per-species diffuse surface sources, keyed by species name.

Every surface term is a defect per unit area against the area there is, and can
be written with ``n_eps`` and ``dGamma_eps`` or with ``grad_phi_incl`` against
``dGamma_eps``. The two agree in exact arithmetic and not in the discrete one: a
cancellation taken at two different sets of points leaves a residue with the
mesh's own symmetry in it. So the surface weights are tabulated once, on one
rule, and a form combining them reads arrays indexed by the same quadrature
points, with nothing re-evaluated.
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import ufl
from basix.ufl import element, quadrature_element
from dolfinx import fem

from ..fem.elements.dg0 import (
    FacetStencilSlope,
    assemble_cell_averages,
    cell_average_form,
    fit_facet_stencil_slope,
)
from ..fem.reduce import function_max
from ..fem.save.element import describe, interpolation_expression
from ..fem.saveable import ParametricSaveable, Saveable
from . import sdf
from .parameters import DiffuseDomainParameters

if TYPE_CHECKING:  # pragma: no cover
    from ufl.core.expr import Expr

    from ..solver.parameters import SolverParameters

logger = logging.getLogger(__name__)

#: Guard on the length that normalizes ``grad phi_incl`` into ``n_eps`` in
#: :meth:`DiffuseDomain.normal_rate_surface_gradient`, as a fraction of the
#: analytic peak ``3 / (2 eps)`` of ``dGamma_eps``.
#:
#: Not a spec parameter, since it cannot change an answer: below it the normal
#: shrinks to zero instead of becoming ``0/0``, and everything reading the normal
#: is weighted by a density smaller than the guard there. It is needed in the far
#: field, where the density underflows, and on the medial axis between two
#: inclusions, where the aggregate gradient cancels by symmetry and the normal is
#: genuinely undefined; zero keeps the surface terms continuous through it.
_NORMAL_GUARD = 1.0e-14


def quadrature_space(
    mesh, degree: int, shape: tuple[int, ...] = ()
) -> fem.FunctionSpace:
    """The quadrature space of ``degree`` on ``mesh``, scalar unless ``shape`` says.

    A coefficient on one of these can only be read by a form whose measure uses
    the same rule, so ``degree`` here and the ``quadrature_degree`` of whatever
    measure reads the coefficient have to be one number.
    """

    return fem.functionspace(
        mesh, quadrature_element(mesh.basix_cell(), value_shape=shape, degree=degree)
    )


class DiffuseDomain(ParametricSaveable[DiffuseDomainParameters]):
    """Materialized static diffuse-domain weights for one mesh."""

    def __init__(
        self,
        solver_parameters: SolverParameters,
        parameters: DiffuseDomainParameters,
    ) -> None:
        self.solver_parameters = solver_parameters
        self.parameters = parameters
        mesh = solver_parameters.dolfinx_mesh
        degree = parameters.indicator_quadrature_degree

        # Built on this mesh's own coordinate, so a geometry of the wrong
        # dimension fails in UFL saying so.
        self.signed_distances = sdf.signed_distances(
            parameters.signed_distance_definitions(),
            mesh,
            float(parameters.domain_eps),
        )
        #: Whether the geometry declares no inclusions, so that the indicator is
        #: one and its gradient zero everywhere: what a spec asks for when the
        #: dynamics are to be seeded by noise or by the outer boundary alone. The
        #: weights are then the literals ``1`` and ``0``, which the forms fold
        #: away, and nothing is tabulated, interpolated or saved.
        self.uniform = not self.signed_distances

        # Built on first request; see grad_phi_incl_degree, contact_slope,
        # normal_rate_surface_gradient and chi_eps_finite_volume.
        self._grad_phi_incl_by_degree: dict[int, fem.Function] = {}
        self._contact_slope: FacetStencilSlope | None = None
        self._surface_gradient_geometry: tuple[fem.Function, ...] | None = None
        self._finite_volume_values: np.ndarray | None = None

        if self.uniform:
            logger.info(
                "geometry declares no inclusions: chi is 1 and grad(chi) is 0 "
                "everywhere, so domain_eps=%s and indicator_floor=%s have no "
                "effect and no surface weight is tabulated",
                float(self.parameters.domain_eps),
                float(self.parameters.indicator_floor),
            )
            self.chi_eps = ufl.as_ufl(1.0)
            self.dgamma_eps = ufl.as_ufl(0.0)
            self.grad_phi_incl = ufl.as_vector([0.0] * mesh.geometry.dim)
            self.q_eps: dict[str, fem.Function] = {}
            # One rather than the maximum, which is zero here: it only divides.
            self.dgamma_eps_max = fem.Constant(mesh, np.float64(1.0))
            return

        self.chi_eps = fem.Function(
            quadrature_space(mesh, degree), name="diffuse_domain_chi_eps"
        )
        self._interpolate_expression(self.chi_eps, self.chi_eps_analytic_ufl())
        self.grad_phi_incl = self.grad_phi_incl_degree(degree)
        self.dgamma_eps = fem.Function(
            quadrature_space(mesh, degree), name="diffuse_domain_dgamma_eps"
        )
        self._interpolate_expression(self.dgamma_eps, self.dgamma_eps_analytic_ufl())
        self.dgamma_eps_max = fem.Constant(
            mesh, np.float64(max(function_max(self.dgamma_eps), 1.0e-14))
        )
        self.q_eps = {}
        for species_name in parameters.active_surface_flux_names:
            function = fem.Function(
                quadrature_space(mesh, degree),
                name=f"diffuse_domain_q_{species_name}_eps",
            )
            self._interpolate_expression(
                function,
                self._q_eps_analytic_ufl(
                    parameters.surface_flux_densities[species_name]
                ),
            )
            self.q_eps[species_name] = function

    ## Overrides

    def name(self) -> str:
        """What the saved fields are qualified by: the spec block's name.

        Fixed, since there is exactly one diffuse domain.
        """

        return "diffuse_domain"

    def declare_saveable(self) -> dict[str, Saveable]:
        """The materialized weights, each in the space it was built in.

        Every entry is static, since the inclusions do not move, and one-to-one,
        with nothing resampled. The quadrature fields are saved although a reader
        cannot draw one directly, because ``chi_eps`` at its quadrature points is
        what the solve weighted by; a reader wanting a drawable indicator evaluates
        :meth:`~ovpsolver.diffuse_domain.DiffuseDomainParameters.chi_eps_on`, which
        is exact. A uniform domain saves nothing: its weights are literals.
        """

        if self.uniform:
            return {}
        offered = {}
        # ``chi_eps`` goes out as ``indicator``, the word a spec and a series use.
        for name, function in (
            ("indicator", self.chi_eps),
            ("grad_phi_incl", self.grad_phi_incl),
            ("dgamma_eps", self.dgamma_eps),
        ):
            offered[name] = Saveable(
                describe(function.function_space, name=name),
                # Bound now, not to the loop variable.
                lambda held=function: held,
                static=True,
            )
        return offered

    ## Public

    def grad_phi_incl_degree(self, degree: int) -> fem.Function:
        """``grad phi_incl`` tabulated on the quadrature rule of ``degree``.

        A quadrature coefficient can only be read by a form whose measure uses its
        rule, so a term assembled on a rule other than the bulk one needs its own
        tabulation. Each degree is tabulated once and kept; the bulk rule's entry
        is :attr:`grad_phi_incl` itself.

        Tabulated rather than written into the form as
        :meth:`grad_phi_incl_analytic_ufl`: both are exact at the points, but the
        expression re-evaluates every inclusion's signed distance and its gradient
        at every quadrature point of every assembly, for a geometry that never
        moves.
        """

        gradient = self._grad_phi_incl_by_degree.get(degree)
        if gradient is None:
            mesh = self.solver_parameters.dolfinx_mesh
            gradient = fem.Function(
                quadrature_space(mesh, degree, (mesh.geometry.dim,)),
                name=f"diffuse_domain_grad_phi_incl_degree_{degree}",
            )
            self._interpolate_expression(gradient, self.grad_phi_incl_analytic_ufl())
            self._grad_phi_incl_by_degree[degree] = gradient
        return gradient

    def chi_eps_finite_volume(self) -> np.ndarray:
        """DG0 cell averages of :attr:`chi_eps` on the bulk rule; read-only.

        Assembled on first request and kept, since the geometry is static.
        """

        if self._finite_volume_values is None:
            volumes = self.solver_parameters.dg0_cell_volumes
            if self.uniform:
                values = np.ones_like(volumes)
            else:
                mesh = self.solver_parameters.dolfinx_mesh
                cells = fem.functionspace(
                    mesh, element("Discontinuous Lagrange", mesh.basix_cell(), 0)
                )
                values = assemble_cell_averages(
                    cell_average_form(
                        self.chi_eps,
                        cells,
                        quadrature_degree=self.parameters.indicator_quadrature_degree,
                    ),
                    volumes,
                )
            values.flags.writeable = False
            self._finite_volume_values = values
        return self._finite_volume_values

    def contact_slope(self) -> FacetStencilSlope:
        """The Korteweg half of the contact-angle defect, as fitted facet coefficients.

        That half is the cell functional ``int_T grad(phi).a dx`` with the kernel
        ``a = grad(phi_incl) / chi_eps``, fitted once over each cell's facet
        neighbours (:func:`~ovpsolver.fem.elements.dg0.fit_facet_stencil_slope`),
        exactly for any quadratic profile of the distance to the surfaces. Built
        on first request and kept, since the mesh and the inclusions never move.

        The distance is ``min_a r_a``, the signed distance to the union of the
        inclusions: one value per cell however many surfaces are near, with a kink
        only on a medial axis. Its orientation needs no care, since the targets
        are integrals of ``grad(r).a`` and carry it into the coefficients. They
        are assembled on the bulk rule, the rule of the defect's affinity half, so
        the two halves meet on the same quadrature.
        """

        if self.uniform:
            raise ValueError(
                "there is no contact-angle slope to fit: this geometry declares "
                "no inclusions, so grad(phi_incl) is zero everywhere"
            )
        if self._contact_slope is None:
            self._contact_slope = fit_facet_stencil_slope(
                reduce(ufl.min_value, self.signed_distances),
                self.grad_phi_incl_over_chi_eps_analytic_ufl(),
                self.solver_parameters.cell_centroids(),
                self.solver_parameters.get_mesh_dx(),
            )
        return self._contact_slope

    def contact_angle(self, phi: Expr, test: Expr) -> ufl.Form:
        r"""``int (grad(phi).grad(phi_incl) / chi_eps) w dx``, ``phi`` cell-constant.

        The Korteweg half of the wetting defect, without the ``kappa`` that
        scales it, which belongs to the phase
        (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.contact_angle_defect`).
        Here because the run and the initial wetting equilibrium
        (:meth:`~ovpsolver.phase_field_system.PhaseFieldSystem.wetting_equilibrium`)
        both call it and so cannot disagree about the discretisation, which would
        leave the difference as a force on the zeroth step. The phase is an
        argument so that the equilibrium solve can write it in the operator the
        run steps.

        A cell-constant phase has its whole gradient on the facets, so both
        writings are facet sums; ``contact_angle_gradient_fitting`` chooses.

        True: the cell functional ``int_T grad(phi).a dx``,
        ``a = grad(phi_incl) / chi_eps``, estimated from the facet neighbours with
        the coefficients :meth:`contact_slope` fits -- exact for any quadratic
        profile of the signed distance, blind to variation along the surface, and
        linear in the live phase with no stencil wider than the Laplacian's. The
        integral is fitted and not a centroid slope, because the kernel changes by
        an order of magnitude across a band cell and a centroid slope would drop
        their covariance, a first-order term. What that buys over a gradient
        reconstruction is the order of the orientation error on a wetting layer as
        steep as an interface, ``(h / ell)^2`` instead of ``h / ell``; the part of
        the phase that does not follow the distance still enters at first order.

        False: the jumps the phase already has. Per cell the divergence theorem
        gives

            int_K grad(phi).a dx = -int_K phi div(a) + int_dK phi (a.n),

        and the cell's own constant in the first and the facet average in the
        second leave the facet sum

            int (grad(phi).a) w dx = -sum_e |e| jump(phi) (a.n_e) avg(w).

        Nothing is reconstructed, but ``avg(w)`` spreads each facet's defect over
        the pair sharing it, and ``a`` is read per facet, where it can differ by an
        order of magnitude across one band cell. ``avg`` of the continuous ``a`` is
        exact, and needed because UFL will not read an unrestricted expression on
        an interior facet; it is taken of the vector before contracting with the
        single ``n+``, since averaging ``a+.n+`` and ``a-.n-`` would give a jump,
        which vanishes identically.
        """

        if self.uniform:
            return ufl.as_ufl(0.0) * self.solver_parameters.get_mesh_dx()
        if self.parameters.contact_angle_gradient_fitting:
            return self.contact_slope().pairing(
                phi,
                test,
                self.solver_parameters.cell_centroids(),
                self.solver_parameters.get_mesh_dS(),
            )

        normal = ufl.FacetNormal(self.solver_parameters.dolfinx_mesh)
        return (
            -ufl.jump(phi)
            * ufl.dot(
                ufl.avg(self.grad_phi_incl_over_chi_eps_analytic_ufl()),
                normal("+"),
            )
            * ufl.avg(test)
            * self.solver_parameters.get_mesh_dS()
        )

    def drag(self, density: Expr, velocity: Expr) -> ufl.Form:
        """Tangential drag ``(1/2) int density |P_eps v|^2 dGamma_eps`` on the surfaces.

        ``P_eps = I - n_eps n_eps`` projects out the normal, so only the component
        along the surface is dragged. Written through
        ``n_eps dGamma_eps = grad_phi_incl`` as
        ``|v|^2 dGamma_eps - (v.grad_phi_incl)^2 / dGamma_eps``, whose two weights
        are tabulated on the same points, so no normal is formed.
        """

        if self.uniform:
            return ufl.as_ufl(0.0) * self.solver_parameters.get_mesh_dx()
        dx = self.solver_parameters.get_mesh_dx()
        v_dot_grad = ufl.dot(velocity, self.grad_phi_incl)
        # Zero where there is no surface, the quotient's limit, which dividing
        # would give as 0/0 once the profile's derivative is exactly zero.
        normal_part = ufl.conditional(
            ufl.gt(self.dgamma_eps, 0.0), (v_dot_grad**2) / self.dgamma_eps, 0.0
        )
        return (
            0.5
            * density
            * (ufl.dot(velocity, velocity) * self.dgamma_eps - normal_part)
            * dx
        )

    def normal_rate_surface_gradient(self, velocity: Expr) -> Expr:
        """``sqrt(dGamma_eps) P_eps grad(v.n_eps)``, with no division left to the form.

        The along-surface gradient of a velocity's normal component, in the
        diffuse-surface operators ``n_eps`` and ``P_eps``, weighted so that its
        square is per unit area of diffuse surface. Expanded as

            grad(v.n_eps) = grad(v)^T n_eps + grad(n_eps)^T v,

        so the velocity enters only through ``v`` and ``grad(v)``, and the rest is
        geometry -- ``P_eps``, ``sqrt(dGamma_eps) n_eps`` and
        ``sqrt(dGamma_eps) P_eps grad(n_eps)^T`` -- tabulated on the bulk rule on
        first request, so a spec with no surface viscosity tabulates nothing. The
        one quotient, the normalization of ``grad(phi_incl)``, is taken there.

        The curvature term is not a correction to drop: for a slip with
        ``v.n_eps = 0`` on every level set, ``grad(v)^T n_eps`` alone equals
        ``-grad(n_eps)^T v``, of size ``|v| / R``, and a surface viscosity without
        it would resist tangential flow around every curved surface.

        The normalization is guarded as ``sqrt(|g|^2 + guard^2)``
        (:data:`_NORMAL_GUARD`) and not through a norm, because ``grad(n_eps)`` is
        taken symbolically and the gradient of ``|g|`` is ``0/0`` wherever the
        density underflows to zero.
        """

        # Zero, and not the expression below with no surface in it: sqrt(sqrt(0))
        # is finite and its derivative is not.
        if self.uniform:
            return ufl.as_vector(
                [0.0] * self.solver_parameters.dolfinx_mesh.geometry.dim
            )
        if self._surface_gradient_geometry is None:
            mesh = self.solver_parameters.dolfinx_mesh
            degree = self.parameters.indicator_quadrature_degree
            dim = mesh.geometry.dim
            gradient = self.grad_phi_incl_analytic_ufl()
            squared = ufl.dot(gradient, gradient)
            guard = (
                _NORMAL_GUARD * 3.0 / (2.0 * self.parameters.domain_eps)
            )
            normal = gradient / ufl.sqrt(squared + guard**2)
            projector = ufl.Identity(dim) - ufl.outer(normal, normal)
            root_density = ufl.sqrt(ufl.sqrt(squared))
            tabulated = []
            for name, shape, expression in (
                ("projector", (dim, dim), projector),
                ("root_normal", (dim,), root_density * normal),
                (
                    "root_curvature",
                    (dim, dim),
                    root_density * ufl.dot(projector, ufl.grad(normal).T),
                ),
            ):
                function = fem.Function(
                    quadrature_space(mesh, degree, shape),
                    name=f"diffuse_domain_surface_{name}",
                )
                self._interpolate_expression(function, expression)
                tabulated.append(function)
            self._surface_gradient_geometry = tuple(tabulated)

        projector, root_normal, root_curvature = self._surface_gradient_geometry
        return ufl.dot(projector, ufl.dot(root_normal, ufl.grad(velocity))) + ufl.dot(
            root_curvature, velocity
        )

    def off_surface_weight_ufl(self) -> Expr:
        """``max(1 - dGamma_eps / max(dGamma_eps), 0)``, a broad off-surface weight.

        Zero only at the peak of the surface density and one in the far field, so
        a far-field anchor (that of the initial wetting equilibrium) acts
        everywhere but on the surface.
        """

        if self.uniform:
            return ufl.as_ufl(1.0)
        return ufl.max_value(1.0 - self.dgamma_eps / self.dgamma_eps_max, 0.0)

    def q_eps_ufl(self, species_name: str) -> Expr:
        """The materialized diffuse surface source of one species; zero if none."""

        flux = self.parameters.surface_flux_densities.get(species_name)
        if flux is None or float(flux) == 0.0:
            return ufl.as_ufl(0.0)

        function = self.q_eps.get(species_name)
        if function is None:
            raise RuntimeError(
                f"nonzero surface flux for species {species_name!r} has no "
                "materialized field"
            )
        return function

    def chi_eps_analytic_ufl(self) -> Expr:
        """``chi_eps = 1 - phi_incl``, floored, as an expression.

        The diffuse indicator with ``indicator_floor`` added; what
        :attr:`chi_eps` tabulates. A facet term takes this, since a quadrature
        element has no value on a facet, and not a nodal interpolant: across a
        profile this steep that is off by about nine per cent with the mesh's
        symmetry in it, which through every facet flux and the Korteweg stiffness
        would be a force. Continuous, so ``avg`` of it is itself and its jump is
        zero, which keeps a numerical flux one number shared by two cells and
        saturation exact.
        """

        # No floor without inclusions: there is no row to keep solvable, and it
        # would only scale every form by one plus it.
        if self.uniform:
            return ufl.as_ufl(1.0)
        return (
            1.0
            - self.inclusion_phase_ufl()
            + self.parameters.indicator_floor
        )

    def grad_phi_incl_analytic_ufl(self) -> Expr:
        """``grad phi_incl = sum_a grad phi_a``.

        Differentiated by UFL from the user's signed distances, which is what lets
        an arbitrary shape cost nothing extra. What :attr:`grad_phi_incl` and every
        other tabulation of the gradient interpolate.
        """

        gradient = ufl.as_vector([0.0] * self.solver_parameters.dolfinx_mesh.geometry.dim)
        for phase in self._inclusion_phases_ufl():
            gradient += ufl.grad(phase)
        return gradient

    def dgamma_eps_analytic_ufl(self) -> Expr:
        """``dGamma_eps = |grad phi_incl|`` as an expression."""

        # Not sqrt(0), whose value is right and whose derivative is not.
        if self.uniform:
            return ufl.as_ufl(0.0)
        gradient = self.grad_phi_incl_analytic_ufl()
        return ufl.sqrt(ufl.dot(gradient, gradient))

    def grad_phi_incl_over_chi_eps_analytic_ufl(self) -> Expr:
        """``grad(phi_incl) / chi_eps``, with the quotient cancelled by hand.

        The kernel of both halves of the contact-angle defect, so the two cancel
        wherever the wetting condition holds, at any depth. As a quotient it is a
        ratio of two quantities vanishing at the same exponential rate into an
        inclusion: bounded, but not computable that way. For one inclusion, with
        ``t = tanh(3 r / eps)``, ``phi_a = (1 - t)/2``, ``1 - phi_a = (1 + t)/2``
        and ``grad(phi_a) = -(3/2eps)(1 - t^2) grad(r)``, so

            grad(phi_a) / (1 - phi_a) = -(3/eps)(1 - t) grad(r) = -(6/eps) phi_a grad(r),

        which is what this sums: nothing divides, and the kernel saturates smoothly
        at ``6/eps`` deep inside. The quotient itself is a true ``0/0`` past about
        ``6.3 eps`` of depth, where ``tanh`` returns exactly one, and a floor under
        its denominator is wrong well before that: both sides decay like
        ``exp(-6|r|/eps)``, so once ``chi_eps`` passes under the floor (about
        ``2.3 eps`` in, at the default) the answer is ``6/eps`` scaled by
        ``chi_eps / floor``, falling exponentially with depth.

        Summed per inclusion rather than formed off the aggregate ``phi_incl``,
        since only the single-inclusion denominator hides in its numerator:
        ``1 - sum_a phi_a`` is not ``prod_a (1 - phi_a)``. The two agree wherever
        the bands are disjoint; where two overlap this counts the crossing once per
        surface, as :meth:`_q_eps_analytic_ufl` does.
        """

        width = self.parameters.domain_eps
        kernel = ufl.as_vector([0.0] * self.solver_parameters.dolfinx_mesh.geometry.dim)
        for distance in self.signed_distances:
            kernel -= (
                (6.0 / width) * sdf.phase(distance, width) * ufl.grad(distance)
            )
        return kernel

    def dgamma_eps_over_chi_eps_analytic_ufl(self) -> Expr:
        """``dGamma_eps / chi_eps``, as the norm of the cancelled vector kernel.

        Taken off :meth:`grad_phi_incl_over_chi_eps_analytic_ufl` rather than
        written out as ``(6/eps) sum_a phi_a |grad(r_a)|``, so that
        ``n_eps dGamma_eps = grad(phi_incl)`` holds between the two halves of the
        defect pointwise and by construction: they are set against each other cell
        by cell, and an identity that holds only in exact arithmetic leaves a
        residue with the mesh's symmetry in it.
        """

        if self.uniform:
            return ufl.as_ufl(0.0)
        kernel = self.grad_phi_incl_over_chi_eps_analytic_ufl()
        return ufl.sqrt(ufl.dot(kernel, kernel))

    def inclusion_phase_ufl(self) -> Expr:
        """``phi_incl = sum_a phi_a``: the smeared indicator of the inclusions.

        ``1 - chi_eps`` up to the floor, and the weight for a term acting on what
        the inclusions occupy rather than on the mixture, such as the inclusion
        drag. Summed per surface, so a point inside two overlapping bands is
        inside twice.
        """

        total = ufl.as_ufl(0.0)
        for phase in self._inclusion_phases_ufl():
            total += phase
        return total

    ## Private helpers

    @staticmethod
    def _interpolate_expression(function: fem.Function, expression: Expr) -> None:
        """Interpolate ``expression`` into ``function``: exact at its own points."""

        function.interpolate(
            interpolation_expression(expression, function.function_space)
        )

    def _q_eps_analytic_ufl(self, flux: float) -> Expr:
        """One phase's injection through every surface, at a common strength.

        Per inclusion rather than off the aggregate ``dgamma_eps``, which where two
        diffuse bands overlap is the norm of a sum and would count the crossing
        region once instead of twice.
        """

        source = ufl.as_ufl(0.0)
        for phase in self._inclusion_phases_ufl():
            gradient = ufl.grad(phase)
            source += float(flux) * ufl.sqrt(ufl.dot(gradient, gradient))
        return source

    def _inclusion_phases_ufl(self) -> tuple[Expr, ...]:
        """Each inclusion's diffuse phase, smeared off its signed distance."""

        return tuple(
            sdf.phase(distance, self.parameters.domain_eps)
            for distance in self.signed_distances
        )
