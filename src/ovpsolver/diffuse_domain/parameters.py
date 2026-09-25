"""Geometry of the fixed inclusions, and the numbers that make it usable."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem

from ..fem.saveable import SaveableParameters
from ..mesh.generate import MESHERS
from ..parametric import (
    Boolean,
    Count,
    Nonnegative,
    Positive,
    Text,
)
from . import sdf

if TYPE_CHECKING:  # pragma: no cover
    from ufl.core.expr import Expr

    from ..mesh.shapes import OuterDomain


class DiffuseDomainParameters(SaveableParameters):
    """Geometry of the fixed inclusions excluded from the mixture domain.

    The mixture occupies the complement of the inclusions, smeared over
    ``domain_eps`` into the diffuse indicator ``chi_eps``. The inclusion surfaces
    are a fixed background measure, not a thermodynamic phase: nothing in the
    Onsager problem evolves them.

    What the inclusions are comes from ``geometry``, a Python file naming one
    signed distance per inclusion and the outer domain they sit in (see
    :mod:`ovpsolver.diffuse_domain.sdf`), so shape, size, position, count and
    dimension are the file's business and not this block's.

    The block sits under the solver because it *builds the mesh*. The diffuse
    surface terms live in a layer of width ``2 eps``, and how many cells cross it
    decides whether the answer is the physics or an artefact locked to the
    lattice: at sixteen cells across the width the normal-velocity leak through an
    inclusion has no preferred direction, at eight it carries an ``m = 6``
    modulation seven times its neighbouring modes. A mesh handed over as a file
    could not be checked against that width, so :mod:`ovpsolver.mesh.generate`
    builds it from the width, the geometry and the sizes below.

    Parameters
    ----------
    geometry : path to the file defining the inclusions and the outer domain,
        relative to the spec that names it. Copied beside the run's output, so a
        finished run carries the geometry it was posed on.
    domain_eps : the smearing parameter ``eps`` of the diffuse phase
        ``(1 - tanh(3 r / eps)) / 2``. The profile it produces is *twice* this
        wide, and ``2 domain_eps`` is the interface width the two mesh entries
        below are counted against.
    mesh_h : the cell size over the whole mesh, which sets the cost of the run.
        Uniform inside the inclusions too, since the diffuse method needs cells
        there as much as anywhere; only the band is refined below it, by
        ``min_elements_across_interface_width``. Measured against the mesh
        afterwards: ``SolverParameters.mesh_lengthscale`` reports what was built.
    min_elements_across_interface_width : how many cells the interface has to
        hold across its full ``2 eps``, so the band is refined to
        ``2 eps / min_elements_across_interface_width`` wherever that is finer than
        ``mesh_h``. Five is Aland, Lowengrub & Voigt (2010) sec. 3.2 and about where
        the six-fold leak starts to bite; sixteen is where it measures as no
        preference at all. A minimum and not a target: a ``mesh_h`` that already
        supplies this many gets a uniform mesh. Over-resolving is not free either:
        cells inside the band resolve a regularization rather than the mixture,
        and shorten the transport step through the CFL bound.
    mesh_algorithm : which gmsh algorithm builds the mesh. ``delaunay`` unless the
        object is to provoke the lattice artefact; see
        :data:`ovpsolver.mesh.generate.MESHERS`.
    inclusion_color : the colour a figure draws the inclusions in, as
        ``1 - chi_eps``, the region the mixture is excluded from.
    indicator_quadrature_degree : the rule ``chi_eps`` is tabulated on, and so the
        rule every bulk integral against it uses.
    indicator_floor : floor under ``chi_eps`` inside the inclusions, so that the
        pressure's mobility ``chi_eps sum_i phi_i / eta_b`` stays positive there
        and any other row this weights stays nonsingular. It does not make a
        quotient against ``chi_eps`` safe: the surface kernels that would form
        one cancel it analytically, since flooring a ratio of two quantities
        that vanish together returns a definite wrong answer.
    surface_permeability : the membrane permeability ``varsigma`` of the
        crossing dissipation. A crossing ``h`` per unit volume costs
        ``(1/2) int h^2 / (varsigma phi)``, so leakage scales with it and
        ``varsigma -> 0`` is the impermeable limit. Strictly positive, since it
        divides; a surface meant to be freely crossed is a large value here.
        Not independent of ``eps``: ``h`` carries ``dGamma_eps``, so the
        dissipation scales as ``1 / (varsigma eps)``, and a spec that refines the
        interface and means to keep its membrane scales this with ``eps`` --
        equivalently, holds ``varsigma dGamma_max`` fixed.
    flux_bc_quadrature : the quadrature degree the flux condition is assembled on.
        The condition is the crossing dissipation written as its Legendre
        transform against a cell-constant multiplier, so a cell imposes one
        condition on its integrated crossing whatever the rule; the rule decides
        only how faithfully that integral is taken.
    inclusion_drag : the coefficient ``n_s`` of the drag
        ``(n_s / 2) int phi_incl phi |v|^2`` every velocity pays inside the
        inclusions
        (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.inclusion_drag`).
        The diffuse-domain equations leave the velocity inside the inclusions free,
        and this pins it at the inclusions' own, zero; without it the interior is a
        second copy of the mixture with only the mesh to seed it. Bounded below by
        what quiets that copy against the Darcy drag, and above by the surface drag
        it reaches through the band, ``surface_drag / eps``. Zero turns it off.
    contact_angle_gradient_fitting : how the Korteweg half of the wetting defect
        (:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.contact_angle`) reads the
        slope of a cell-constant phase. True uses the slope fitted once per cell
        from the signed distance
        (:meth:`~ovpsolver.diffuse_domain.DiffuseDomain.contact_slope`), whose
        orientation error on a steep wetting layer is ``(h / ell)^2`` rather than
        ``h / ell``. False writes the term as the facet sum of the phase's jumps,
        which reconstructs nothing but reads the steep geometry per facet. Here
        and not under a phase because the run and the initial wetting equilibrium
        have to discretise the row the same way.
    save : the geometry fields to write out, by name, as for a phase's ``save``.
        Empty by default: the inclusions do not move, so a reader reconstructs the
        mask from the spec.
    """

    geometry: str = Text()
    domain_eps: float = Positive()
    mesh_h: float = Positive(coefficient=False)
    min_elements_across_interface_width: int = Count(16)
    mesh_algorithm: str = Text("delaunay", choices=tuple(sorted(MESHERS)))
    inclusion_color: str = Text("green")
    indicator_quadrature_degree: int = Count(6)
    indicator_floor: float = Nonnegative(1.0e-6)
    surface_permeability: float = Positive(3.7e-5)
    flux_bc_quadrature: int = Count(6)
    inclusion_drag: float = Nonnegative(0.0)
    contact_angle_gradient_fitting: bool = Boolean(False)

    # Per-instance caches, filled on first use. ``None`` here and not ``{}``,
    # which would be one dict shared by every instance.
    _definitions = None
    _distances = None
    _outer_domain = None

    ## Overrides

    def derive(self) -> None:
        #: Filled in by the mixture, which knows the phases; a mapping because
        #: materializing ``q_eps`` needs every phase's injection at once.
        self.surface_flux_densities: dict[str, float] = {}
        #: The mixture's ``rng_seed``, for a geometry that draws its arrangement,
        #: so one seed reproduces the whole spec. Defaulted so a geometry can be
        #: read without a mixture around it.
        self.rng_seed: int = 0
        #: Resolved against the spec's own directory by :meth:`resolve_against`.
        self.geometry_path = Path(self.geometry).expanduser()

    ## Public

    @property
    def active_surface_flux_names(self) -> tuple[str, ...]:
        """Phases with a nonzero diffuse surface source to materialize."""

        return tuple(
            name
            for name, flux in self.surface_flux_densities.items()
            if float(flux) != 0.0
        )

    def resolve_against(self, source: Path) -> None:
        """Locate the geometry file relative to the spec that named it."""

        given = Path(self.geometry).expanduser()
        self.geometry_path = (
            given if given.is_absolute() else Path(source).parent / given
        )
        if not self.geometry_path.is_file():
            raise ValueError(
                f"{self.where}.geometry names no file: {self.geometry_path}"
            )

    def geometry_assets(self) -> tuple[Path, ...]:
        """The files the geometry file reads, resolved beside it."""

        return sdf.load_assets(self.geometry_path)

    def signed_distance_definitions(
        self,
    ) -> Callable[[ufl.SpatialCoordinate, float], Sequence[Expr]]:
        """The loaded ``signed_distance_functions``, read once, seed bound."""

        if self._definitions is None:
            self._definitions = sdf.load(self.geometry_path, self.rng_seed)
        return self._definitions

    def signed_distances_for(self, mesh) -> tuple[Expr, ...]:
        """The inclusions' signed distances on ``mesh``, built once per mesh.

        Building them runs the user's ``signed_distance_functions``, which may be
        expensive -- a relaxed point configuration, a refitted STEP file -- and
        every reader of the geometry asks. Keyed by mesh identity, since the file,
        the seed and ``domain_eps`` are fixed for the life of these parameters;
        the mesh is held in the value, so its id cannot be reused by another mesh
        while the entry lives. At most two meshes ask: the coarse one the grading
        samples, and the run's.
        """

        if self._distances is None:
            self._distances = {}
        if id(mesh) not in self._distances:
            self._distances[id(mesh)] = (
                mesh,
                sdf.signed_distances(
                    self.signed_distance_definitions(), mesh, float(self.domain_eps)
                ),
            )
        return self._distances[id(mesh)][1]

    def outer_domain(self) -> OuterDomain:
        """The shape the mesh is built on, read once from the geometry file."""

        if self._outer_domain is None:
            self._outer_domain = sdf.load_outer_domain(
                self.geometry_path, self.rng_seed
            )
        return self._outer_domain

    ## Public: exact evaluation on a reader's mesh
    #
    # Every number enters as a ``float`` and not as the Constant the solve reads:
    # a Constant belongs to the mesh it was built on, a reader's mesh is read back
    # from the run's file, and an expression spanning two meshes will not
    # interpolate.

    def chi_eps_ufl(self, mesh) -> Expr:
        """The solve's indicator, floor included, as an expression on ``mesh``.

        For a reader that integrates it, such as one rebuilding a facet form,
        where only the expression has a value.
        """

        distances = self.signed_distances_for(mesh)
        # No floor without inclusions, as in the solve.
        if not distances:
            return ufl.as_ufl(1.0)
        inclusion = sum(sdf.phase(one, float(self.domain_eps)) for one in distances)
        return 1.0 - inclusion + float(self.indicator_floor)

    def chi_eps_on(self, space: fem.FunctionSpace) -> npt.NDArray[np.float64]:
        """:meth:`chi_eps_ufl` at ``space``'s own points, exactly.

        For a reader reproducing something the solve computed, which wants the
        weight the rows actually carried. A picture wants
        :meth:`bulk_indicator_on`.
        """

        return sdf.evaluate(self.chi_eps_ufl(space.mesh), space)

    def bulk_indicator_on(self, space: fem.FunctionSpace) -> npt.NDArray[np.float64]:
        """``1 - phi_incl`` at ``space``'s points: the indicator without the floor.

        What a picture weights a phase by, so that a phase vanishes where there is
        no domain for it. :meth:`chi_eps_on` adds :attr:`indicator_floor`, which at
        the sizes a floor is set to (``0.1``, say) would show a tenth of every
        phase inside every inclusion: a regularization keeping a row solvable, not
        physics.
        """

        if not self.has_inclusions(space.mesh):
            return self.chi_eps_on(space)
        return self.chi_eps_on(space) - float(self.indicator_floor)

    def has_inclusions(self, mesh) -> bool:
        """Whether the geometry puts anything inside the outer shape.

        An empty geometry is a domain like any other: the indicator is one and its
        gradient zero. Answered by building the distances on ``mesh``, once,
        through :meth:`signed_distances_for`.
        """

        return bool(self.signed_distances_for(mesh))

    def signed_distances_on(self, space: fem.FunctionSpace) -> npt.NDArray[np.float64]:
        """Each inclusion's signed distance at ``space``'s points, one row each."""

        return np.stack(
            [
                sdf.evaluate(distance, space)
                for distance in self.signed_distances_for(space.mesh)
            ]
        )

    def surface_normals_on(self, space: fem.FunctionSpace) -> npt.NDArray[np.float64]:
        """Each inclusion's surface normal at ``space``'s points, pointing *into* it.

        ``grad r_a`` points out of an inclusion, so the inward normal is its
        negative: the per-inclusion ``n_eps``, and the direction
        ``grad phi_incl`` carries. Normalized rather than assumed unit, since a
        user's expression is only a signed distance if they made it one.
        """

        dim = space.mesh.geometry.dim
        rows = []
        for distance in self.signed_distances_for(space.mesh):
            gradient = sdf.evaluate(-ufl.grad(distance), space).reshape(-1, dim)
            length = np.linalg.norm(gradient, axis=1)
            rows.append(gradient / np.where(length > 0.0, length, 1.0)[:, None])
        return np.stack(rows)

    def diffuse_perimeter(self, mesh) -> float:
        """``int dGamma_eps`` over ``mesh``: the total diffuse surface measure.

        The denominator for a per-area quantity, since the smeared surface is what
        the run integrates against; for a circle it agrees with ``2 pi R`` to the
        accuracy of the smearing.
        """

        gradient = sum(
            ufl.grad(sdf.phase(distance, float(self.domain_eps)))
            for distance in self.signed_distances_for(mesh)
        )
        measure = ufl.Measure(
            "dx",
            domain=mesh,
            metadata={
                "quadrature_degree": self.indicator_quadrature_degree
            },
        )
        return float(
            fem.assemble_scalar(
                fem.form(ufl.sqrt(ufl.dot(gradient, gradient)) * measure)
            )
        )
