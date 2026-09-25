"""The ``solver`` block of a spec, and the context declarations are written against.

Two things in one file, because a spec states them in one block.
:class:`SolverParameters` is the block itself, and everything the declarations
are written against hangs off it -- the mesh, the quadrature rule, the live
``dt`` constant. The other classes are its sub-blocks, and they are what the
driver decides from.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh

from ..diffuse_domain.parameters import DiffuseDomainParameters
from ..fem.elements import dg0
from ..fem.save.saver import MESH_FILE
from ..mesh import generate
from ..mesh.utils import max_cell_diameter
from ..parametric import (
    Boolean,
    Count,
    Fraction,
    Index,
    Nonnegative,
    Number,
    Options,
    Parameters,
    Positive,
    Text,
    bind_form_mesh,
)

if TYPE_CHECKING:
    from dolfinx.mesh import EntityMap
    from mpi4py import MPI
    from ufl.core.expr import Expr


class TimesteppingParameters(Parameters):
    """How the driver proposes, checks and retries a step.

    There is no a priori step estimate. An ``h_min / v_max`` bound would divide
    by a velocity that is an unknown of the very solve it is meant to bound, and
    positivity is governed by algebraic conditions on the assembled rows, not by
    a Courant number on a mesh spacing. So a step is proposed, taken, checked
    against those, and rolled back and retaken smaller if it fails.

    Parameters
    ----------
    dt : the macro step, and the output grid: one saved frame per one of these.
        Sub-cycled internally at whatever the transport bound allows, so it is a
        reporting interval rather than a step size.
    warm_start_dt : the first sub-step. Every later one is proposed from the
        bound the previous step reported; before any step there is no solved
        velocity and so no bound, and this is the guess that gets the first one
        solved. It is checked like any other.
    n_steps : how many macro steps to run.
    cfl_positive_transport : ``CFL_DG0``, the fraction of its own outflow bound
        a step may take. One number for the whole mixture, because every
        variable it declares is cell-constant and bounded the same way.
        Sufficient at ``1``, and below that only to buy margin. The other CFL
        constant, ``CFL_u``, bounds transport in the gelation coordinate, a
        state space a polymerizing phase carries rather than anything the mesh
        has, so that phase declares it.
    retry_cfl_timestep_fraction : how far a step rejected by its own transport
        bound is pulled back, as a fraction of that bound -- which already
        carries ``cfl_positive_transport``. The bound was assembled from the
        velocities of the step that violated it, so it only estimates the one
        the retry will face; undershooting keeps a retry from failing for the
        same reason. Where the bound cannot be aimed at, it scales the attempt
        instead
        (:meth:`~ovpsolver.solver.solver.PhaseFieldSolver.retry_step_size`). It
        does not set how close an accepted step runs to its bound: the next step
        proposes the bound itself.
    retry_diverged_timestep_fraction : how far a step is pulled back when the
        rate solve failed to converge, as a fraction of the attempt, there being
        no solved velocity and so no bound. Smaller by default: the convex split
        makes a step unconditionally descending, not unconditionally solvable,
        so a diverged Newton says the step was outside the basin rather than
        slightly too long. An over-small step costs little, since the one that
        succeeds proposes the transport bound again.
    max_retries : how many times one sub-step may be retaken before the run is
        abandoned.
    min_remainder_fraction : the shortest trailing sub-step worth taking, as a
        fraction of the macro step. Sub-cycling almost never lands exactly on the
        reporting interval, and the sliver left would cost a full nonlinear
        solve; below this the macro step ends early, and its reported length
        says so.
    positivity_floor_throw : hard lower bound on everything that declares itself
        non-negative, checked after every accepted sub-step: entry by entry, or
        on the eigenvalues of a tensor, whose positivity is the semidefinite
        cone. Below it the run stops
        (:meth:`~ovpsolver.solver.solver.PhaseFieldSolver._check_nonnegative`).
        Slightly negative because a cell sitting on an active constraint is at
        zero only up to roundoff. Null disables the check.
    """

    dt: float = Number(
        1.0e-3, minimum=0.0, exclusive_minimum=True, coefficient=False
    )
    warm_start_dt: float = Number(
        1.0e-6, minimum=0.0, exclusive_minimum=True, coefficient=False
    )
    n_steps: int = Index(1)
    cfl_positive_transport: float = Number(
        0.7, minimum=0.0, maximum=1.0, exclusive_minimum=True, coefficient=False
    )
    retry_cfl_timestep_fraction: float = Fraction(0.8, coefficient=False)
    retry_diverged_timestep_fraction: float = Fraction(0.5, coefficient=False)
    max_retries: int = Index(12)
    min_remainder_fraction: float = Number(
        0.1, minimum=0.0, maximum=1.0, exclusive_maximum=True, coefficient=False
    )
    positivity_floor_throw: float = Number(
        -1.0e-10, maximum=0.0, optional=True, coefficient=False
    )


class OVPStepParameters(Parameters):
    """The reversible half of the split, and what the Newton solve is told.

    What the nonlinear solve is asked to accept, and how its linearization is
    solved. ``petsc_options`` either factors that linearization whole, or asks
    for ``pc_type: fieldsplit``, in which case the solver supplies the one thing
    a spec cannot -- which unknowns are the live states and which are static
    (:meth:`~ovpsolver.solver.solver.PhaseFieldSolver.fieldsplit_index_sets`) --
    and every ``fieldsplit_state_*`` and ``fieldsplit_static_*`` option says how
    that half is solved.
    """

    skip: bool = Boolean(False)
    snes_max_it: int = Count(50)
    snes_rtol: float = Nonnegative(1.0e-8, coefficient=False)
    snes_atol: float = Nonnegative(1.0e-10, coefficient=False)
    #: Log a line per Newton iteration: the source-of-truth convergence trace
    #: (:meth:`~ovpsolver.solver.instrument.Instrument.newton`).
    snes_monitor: bool = Boolean(False)
    #: Stop the rate solve when the excess Rayleighian falls below this fraction
    #: of ``Psi*``, rather than when ``|F|`` falls below ``snes_rtol`` of its
    #: initial value (:mod:`ovpsolver.solver.onsager`).
    #:
    #: Nothing like ``snes_rtol`` numerically. The ratio tested is the relative
    #: suboptimality of the iterate, one at rest and zero at force balance, and
    #: the square of the relative rate error in the dissipation metric: ``1e-6``
    #: buys rates good to about a part in a thousand, ``1e-8`` a part in ten
    #: thousand, ``1e-10`` a part in a hundred thousand. ``1e-8`` is the sensible
    #: starting point.
    #:
    #: Absent by default: the measure is reported on every run whether or not
    #: this is set, and is worth reading a few times before it is handed the
    #: decision. PETSc's own residual and step tests still apply when it is set.
    excess_rayleighian_rtol: float = Number(
        None,
        minimum=0.0,
        exclusive_minimum=True,
        optional=True,
        coefficient=False,
    )
    # The SNES tolerances and iteration cap are declared in their own right and
    # injected by :meth:`derive`, so a spec states each of them once.
    petsc_options: dict = Options(
        {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )

    def derive(self) -> None:
        self.petsc_options.update(
            snes_rtol=self.snes_rtol,
            snes_atol=self.snes_atol,
            snes_max_it=self.snes_max_it,
        )


class IrreversibleStepParameters(Parameters):
    """The other half of the split: the chemistry.

    Skipping it runs the mixture as a pure Onsager flow, which is what makes the
    transport rows testable against a composition that only moves.
    """

    skip: bool = Boolean(False)


class SaveParameters(Parameters):
    """Where a run's output goes, and how often.

    ``path`` is resolved against the document that asked for it rather than the
    working directory, so a spec copied into its own output folder with
    ``path: .`` still points at that folder.

    Parameters
    ----------
    path : the output directory, which holds the saved fields, the archived
        spec, and the log.
    logfile : the run's own log, inside ``path``. Without one the trace goes to
        the console only, which loses it.
    every_k : save one frame every this many macro steps. The first step is
        always saved, since a run's initial condition is what everything after
        it is read against.
    resumable_every : overwrite the run's restart point every this many macro
        steps, independently of ``every_k`` and of the save lists: what it
        writes is whatever the mixture declared it needs to be put back as it
        was. One snapshot rewritten in place rather than a frame per step,
        because the generating function alone is sixty-four values per cell, so
        its history costs gigabytes and its latest value megabytes. There is no
        way to turn it off.
    dtype : the precision the frames are stored in. Half the disk for a field
        nobody will difference at the tenth digit.
    debug : write the per-step statistics as well as the per-step summary. On,
        because a reduction per unknown is nothing against a nonlinear solve on
        the same unknowns, and the one run you wish you had it for is the one
        that already finished.
    """

    path: str = Text()
    logfile: str = Text(optional=True)
    every_k: int = Count(1)
    resumable_every: int = Count(1)
    dtype: str = Text("float64", choices=("float32", "float64"))
    debug: bool = Boolean(True)

    def derive(self) -> None:
        #: Reset by :meth:`resolve_against` once the loader says where the
        #: document was; until then a bare relative path is all there is.
        self.directory = Path(self.path).expanduser()

    def resolve_against(self, source: Path) -> None:
        directory = Path(self.path).expanduser()
        self.directory = (
            directory if directory.is_absolute() else source.parent / directory
        )

    @property
    def log_path(self) -> Path | None:
        return None if self.logfile is None else self.directory / self.logfile


class SolverParameters(Parameters):
    """The mesh the mixture lives on, and how the driver steps it.

    Everything declared here is read from the spec; everything computed from it
    is the finite-element context those numbers imply -- the mesh, the measures
    written against it, the live timestep constant. They stay one object because
    every declaration in the model is written against that context, and
    splitting them would mean passing both everywhere one is wanted.

    Parameters
    ----------
    diffuse_domain : the geometry, and with it the mesh. Here rather than beside
        the mixture because ``domain_eps`` and the cell sizes are what the mesh
        is built from.
    timestepping : how long a step is, and what makes it too long.
    ovp_step, irreversible_step : the two halves of the Lie split. Either can be
        skipped, and both still see the same state rotation, so skipping one is
        a controlled experiment rather than a shortcut.
    save : where the output goes, and how often.
    log_reg_delta : ``delta``, the composition below which ``ln phi`` is
        continued linearly wherever this package takes a logarithm of one
        (:func:`~ovpsolver.solver.regularization.log_reg`, which says how it is
        chosen). One number for the whole run, not one per phase: it is a
        statement about where a Newton step may put a composition, the solver's
        business whatever species it belongs to.
    drag_measure_floor : ``(chi phi)_*``, the floor on every Darcy measure
        ``chi_eps phi``, taken as a smooth maximum
        (:func:`~ovpsolver.solver.regularization.smooth_max`), so the coefficient
        is ``sqrt((chi_eps phi)^2 + floor^2) / D``. Equivalently a cap
        ``D / floor`` on the mobility, which is what keeps a velocity row
        solvable where its carrier vanishes; the same floor lifts the membrane
        mobility of the crossing multiplier. One number for the whole run, since
        what it bounds is the condition number of a row, not a property of a
        species.
    """

    diffuse_domain: DiffuseDomainParameters = DiffuseDomainParameters()
    log_reg_delta: float = Positive(1.0e-4)
    drag_measure_floor: float = Fraction(5.0e-3)
    timestepping: TimesteppingParameters = TimesteppingParameters()
    ovp_step: OVPStepParameters = OVPStepParameters()
    irreversible_step: IrreversibleStepParameters = IrreversibleStepParameters()
    save: SaveParameters = SaveParameters()

    def derive(self) -> None:
        #: The rule ``chi_eps`` is tabulated on, and so the rule every bulk
        #: measure here uses. Owned by the diffuse domain, which is what
        #: ``chi_eps`` belongs to; the mixture sets it before building anything
        #: (:meth:`integrate_against`).
        self.indicator_quadrature_degree = 6

    ## The mesh

    @property
    def mesh_path(self) -> Path:
        """Where this run's mesh is archived: inside its own output, as a byte
        copy of the file it was read from (:attr:`mesh_source_path`)."""

        return self.save.directory / MESH_FILE

    @cached_property
    def mesh_source_path(self) -> Path:
        """The file this run's mesh is read from, generating one if there is none.

        Always a file, because writing a dolfinx mesh out and reading it back does
        not return the mesh written: dolfinx reorders cells and nodes for
        locality, and every dof vector a run saves is indexed by the numbering it
        assembled on. So a fresh mesh is generated, written, and read back before
        anything runs on it, and the archive is a byte copy of that same file.

        Not written straight into the output directory: the mesh is built while
        the parameters are, before the saver has decided whether that directory
        may be written to at all. The saver copies it in.
        """

        if self.mesh_path.is_file():
            return self.mesh_path

        geometry = self.diffuse_domain
        built = generate.graded_mesh(
            outer=geometry.outer_domain(),
            # Through the cache: the grading asks once to decide whether there
            # is anything to grade towards and again to sample the size field,
            # and a user's geometry need not be built twice for one mesh.
            distances_of=geometry.signed_distances_for,
            domain_eps=float(geometry.domain_eps),
            mesh_h=float(geometry.mesh_h),
            min_elements_across_interface_width=int(
                geometry.min_elements_across_interface_width
            ),
            mesh_algorithm=geometry.mesh_algorithm,
        )
        # Held on the instance, so the directory outlives the generator's
        # reference and is cleaned up with the parameters.
        self._mesh_scratch = TemporaryDirectory(prefix="ovpsolver_mesh_")
        source = Path(self._mesh_scratch.name) / "mesh.xdmf"
        generate.write(built, source)
        return source

    @cached_property
    def dolfinx_mesh(self) -> dmesh.Mesh:
        """The mesh the run is posed on, read from :attr:`mesh_source_path`
        whichever way it arose.

        A finished run's mesh is read and never regenerated, since gmsh is not
        promised to reproduce a mesh across versions or platforms. Lazy, because
        reading a spec is not running it: plotting and diagnostics read a spec to
        find out what a run was made of, and take their arrays from the output.
        """

        mesh = generate.read(self.mesh_source_path)
        # Every numeric parameter becomes a coefficient on this mesh from here
        # on: at the mesh and not at construction, because the mesh is built
        # from some of these same numbers, which have to be numbers until then.
        bind_form_mesh(mesh)
        return mesh

    @cached_property
    def mesh_lengthscale(self) -> float:
        """The mesh's own coarsest cell, measured rather than declared.

        ``mesh_h`` is only a request, and on a graded mesh a request about part of
        it: the coarse cells are the bulk ones and the fine ones are in the band,
        so no single declared number describes the cells the residuals are
        assembled on.
        """

        return float(max_cell_diameter(self.dolfinx_mesh))

    @property
    def mesh_comm(self) -> "MPI.Comm":
        """The communicator the mesh was built on, and so the run's."""

        return self.dolfinx_mesh.comm

    @cached_property
    def _outer_boundary(self) -> "tuple[dmesh.Mesh, EntityMap] | None":
        """The ``Sigma`` submesh and its facet-to-parent entity map, or ``None``
        for a mesh with no exterior facets."""

        mesh = self.dolfinx_mesh
        facet_dim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
        facets = np.asarray(dmesh.exterior_facet_indices(mesh.topology), dtype=np.int32)
        if facets.size == 0:
            return None
        boundary, facet_to_parent, _, _ = dmesh.create_submesh(mesh, facet_dim, facets)
        return boundary, facet_to_parent

    def get_outer_boundary_mesh(self) -> dmesh.Mesh:
        """Codimension-one mesh of the true outer boundary ``Sigma``."""

        if self._outer_boundary is None:
            raise ValueError("mesh has no exterior facets for a surface element")
        return self._outer_boundary[0]

    def get_outer_boundary_entity_maps(self) -> tuple["EntityMap", ...]:
        """Entity maps needed for forms coupling ``Sigma`` fields to bulk forms."""

        if self._outer_boundary is None:
            raise ValueError("mesh has no exterior facets for a surface form")
        return (self._outer_boundary[1],)

    @cached_property
    def _cell_centroids(self) -> fem.Function:
        return dg0.cell_centroids(self.dolfinx_mesh)

    def cell_centroids(self) -> fem.Function:
        """Cell midpoints as a DG0 vector field, built once for the run.

        Cached beside the mesh rather than rebuilt per form: every two-point
        facet rule in the mixture is scaled by the distance between neighbouring
        centroids, and there is one mesh.
        """

        return self._cell_centroids

    ## Forms

    def integrate_against(self, quadrature_degree: int) -> None:
        """Adopt the quadrature rule the diffuse domain is tabulated on.

        Called once, by the mixture, before any element is declared: the measure
        a form is written against has to be the one ``chi_eps`` has values at,
        and it is the domain that decides that.
        """

        self.indicator_quadrature_degree = quadrature_degree
        self.dg0_cell_volumes = dg0.mesh_cell_volumes(
            self.dolfinx_mesh,
            quadrature_degree=quadrature_degree,
        )

    @property
    def petsc_options(self) -> dict[str, Any]:
        return self.ovp_step.petsc_options

    def get_mesh_dx(self) -> ufl.Measure:
        """Bulk measure, at the quadrature rule ``chi_eps`` is tabulated on."""

        return ufl.Measure(
            "dx",
            domain=self.dolfinx_mesh,
            metadata={
                "quadrature_degree": self.indicator_quadrature_degree
            },
        )

    def get_mesh_dS(self) -> ufl.Measure:
        """Interior-facet measure, at FFCx's default degree: the quadrature-element
        diffuse-domain coefficients are bulk-only, and facet forms use polynomial
        coefficients."""

        return ufl.Measure(
            "dS",
            domain=self.dolfinx_mesh,
        )

    def get_mesh_ds(self) -> ufl.Measure:
        """Outer-boundary measure, at FFCx's default degree, as :meth:`get_mesh_dS`."""

        return ufl.Measure(
            "ds",
            domain=self.dolfinx_mesh,
        )

    def boundary_drag(self, density: "Expr", velocity: "Expr") -> ufl.Form:
        """Tangential drag on the true outer mesh boundary."""

        ds = self.get_mesh_ds()
        normal = ufl.FacetNormal(self.dolfinx_mesh)
        normal_velocity = ufl.dot(velocity, normal)
        tangential_norm_sq = (
            ufl.dot(velocity, velocity) - normal_velocity**2
        )
        return 0.5 * density * tangential_norm_sq * ds

    ## The timestep

    @cached_property
    def dt_live(self) -> fem.Constant:
        """The sub-step the rows are currently being assembled for.

        A ``fem.Constant`` and not a literal, for the reason every declared
        number is one: the forms read its value without it being part of their
        signature, so the step can change without recompiling anything. Its
        value is whatever :meth:`set_timestep` last wrote -- the sub-step being
        attempted, reset before every solve including every shortened retry --
        and not the ``dt`` the spec asked for.
        """

        return fem.Constant(
            self.dolfinx_mesh, default_scalar_type(self.timestepping.dt)
        )

    def set_timestep(self, dt: float) -> None:
        """Point :attr:`dt_live` at the sub-step about to be solved."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt_live.value = dt
