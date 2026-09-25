"""How much of a solved velocity is structure the mesh put there.

The transport rule reads a velocity through one number per facet. In
:func:`~ovpsolver.fem.elements.dg0.upwind_numerical_flux` the velocity enters only
as ``dot(carrier_velocity("+"), normal("+"))`` against a cell-constant density, so
``rho`` factors out of the facet integral and the row's entire dependence on ``v``
at facet ``F`` is, exactly,

    u_F = int_F v.n_F dS.

The tangential component and the interior of a cell do nothing. So the data here is
the vector ``{u_F}`` over interior facets: the velocity as the scheme sees it.

Why the measure is at patch scale and not per cell
--------------------------------------------------
The obvious measure fits one velocity vector to a triangle's three ``u_F`` and calls
the residual mesh structure. It cannot work. The map ``v -> (v.n_F)`` is
``R^2 -> R^3`` of rank two, so a triangle has exactly one residual direction; under
facet-length weights it is ``{r : sum_F |e_F| r_F n_F = 0}``, and since
``sum_F |e_F| n_F = 0`` by the divergence theorem that direction is ``r_F = const``.
Hence

    residual  ∝  sum_F |e_F| u_F  =  oint_{∂K} v.n  =  int_K div v.

**The unique per-triangle mode the facet data cannot explain is the divergence.**
Any single-cell criterion is therefore a compressibility measure, and under a nearly
hard saturation constraint it reads zero everywhere however locked the velocity is.

The patch-affine residual
-------------------------
So the fit is over a patch: cell ``K`` with its facet neighbours, and the nine
distinct facets they carry. An affine velocity ``a + G(x - x_K)`` is fitted to the
facet data, six unknowns against nine rows, and the weighted residual fraction

    c_K = ||W^(1/2) r|| / ||W^(1/2) u||   in [0, 1]

is reported. Zero means the patch's facet-normal velocities are exactly what some
affine field would produce, which is as smooth as transport can tell. The fit is
blind to uniform translation, shear *and* divergence, all three being affine, so it
does not relapse into the single-cell problem. For a field resolved on length ``L``
the residual is the quadratic-and-higher part and ``c_K = O((h/L)^2)``; for a mode
at the cell scale it is ``O(1)``.

Reported raw. One is attainable for arbitrary facet data but not for a continuous
P2 field, so the report prints where white noise in the P2 space on this mesh
actually scores (:meth:`FacetPatches.noise_ceiling`). Rescaling by that would make
the number depend on patch shape and stop cells being comparable with each other.

What the summaries weight by
----------------------------
``c`` is a cell field, and three measures of the mixture say where to read it
(:meth:`MeshCorrelation.weights`). ``phi_incl`` is used directly and the floored
``chi_eps`` is not: ``1 - chi_eps = phi_incl - floor`` is ``-floor`` throughout the
bulk, so a floored weight is negative almost everywhere. The free fluid and the
inclusion interiors partition unity, so the domain mean is recoverable from the two;
that is checked rather than assumed, since it catches exactly the class of
weight-construction bug the floor sign belongs to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx.mesh import entities_to_geometry

from ...fem.elements.dg0 import cell_centroids, mesh_cell_volumes
from .base import Diagnostic, Unanswerable, declare_frames, frames_of
from .report import Table

if TYPE_CHECKING:
    from argparse import ArgumentParser

    import numpy.typing as npt

    from ...fem.save import Field, Run
    from ..parameters import PhaseFieldSystemParameters
    from .report import Report

#: Unknowns in the affine fit, ``a`` and ``G`` together, per spatial dimension.
#: A patch needs at least one row beyond these to have any residual space at all.
AFFINE_UNKNOWNS = {2: 6, 3: 12}

#: How locked the run may be, read where the mixture lives, before the check
#: fails. Not a physical constant: it is the point past which the velocity's
#: grid-scale content stops being the tail of a resolved field.
LOCKING_TOLERANCE = 0.35

#: Draws used to measure where white noise in the P2 space scores. Enough that
#: the median is stable to a few parts in a thousand, and cheap since each draw
#: is one batched solve already built for the frames.
CEILING_DRAWS = 4


class MeshCorrelation(Diagnostic):
    """What share of each solved velocity is structure at the grid scale."""

    name = "analyze.mesh_correlation"
    summary = "how much of each velocity is mesh-scale structure"

    ## Overrides

    def declare(self, parser: "ArgumentParser") -> None:
        super().declare(parser)
        declare_frames(parser)
        parser.add_argument(
            "--velocities",
            nargs="+",
            metavar="FIELD",
            help="saved velocities to read, qualified as phase.name; by default "
            "every saved field that is a vector of the mesh's own dimension",
        )
        parser.add_argument(
            "--locking",
            type=float,
            default=LOCKING_TOLERANCE,
            help=f"how locked the free fluid may be (default: {LOCKING_TOLERANCE})",
        )

    def measure(
        self,
        report: "Report",
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        *,
        frames: "list[int] | None" = None,
        stride: int = 1,
        velocities: "list[str] | None" = None,
        locking: float = LOCKING_TOLERANCE,
        **options: Any,
    ) -> None:
        names = velocities or saved_velocities(run)
        if not names:
            raise Unanswerable(
                "no saved velocity: this reads the facet fluxes a velocity "
                "carries, so a spec has to put one of its v fields in a save list"
            )
        missing = [name for name in names if not run.has(name)]
        if missing:
            raise Unanswerable(f"{', '.join(missing)} was not saved")

        patches = FacetPatches(run)
        report.field("mesh", f"{patches.cells} cells, {patches.facets} interior facets")
        report.field(
            "patches",
            f"{patches.measurable.sum()} of {patches.cells} have a residual space "
            f"({patches.cells - int(patches.measurable.sum())} touch the boundary)",
        )
        ceiling = patches.noise_ceiling()
        report.field(
            "P2 noise scores", f"c = {ceiling:.3f} (the practical top of the range)"
        )
        report.note()

        weights = self.weights(run, parameters, patches)
        walk = frames_of(run, frames, stride=stride)
        for name in names:
            self.one_velocity(report, run, patches, weights, name, walk, locking)

    ## Public

    def one_velocity(
        self,
        report: "Report",
        run: "Run",
        patches: "FacetPatches",
        weights: "dict[str, npt.NDArray[np.float64]]",
        name: str,
        walk: tuple[int, ...],
        locking: float,
    ) -> None:
        table = report.table(
            Table(
                f"{name} frame",
                "time",
                "free fluid",
                "inclusions",
                "surfaces",
                "domain",
                "churn",
                formats={
                    "free fluid": ".4f",
                    "inclusions": ".4f",
                    "surfaces": ".4f",
                    "domain": ".4f",
                    "churn": ".3g",
                },
            )
        )
        field = run.field(name)
        worst, worst_frame = 0.0, walk[0]
        for frame in walk:
            facet_flux = patches.facet_flux(field, frame)
            correlation = patches.residual_fraction(facet_flux)
            # One mask for every summary, so the four means are averages of the
            # same numbers over the same cells and the partition identity below
            # is exact rather than approximate.
            usable = patches.measurable & np.isfinite(correlation)
            summaries = {
                where: weighted_mean(correlation, weight, usable)
                for where, weight in weights.items()
            }
            table.add(
                frame,
                float(run.times[frame]),
                summaries["free fluid"],
                summaries["inclusions"],
                summaries["surfaces"],
                summaries["domain"],
                patches.churn(facet_flux),
            )
            if summaries["free fluid"] > worst:
                worst, worst_frame = summaries["free fluid"], frame

        # The free fluid and the inclusions partition unity, so they reconstruct
        # the domain mean exactly and any departure is a weight built wrong -- a
        # floored chi_eps where phi_incl belongs -- rather than a property of the
        # velocity. Over the cells the means were taken on, since the weights
        # partition the domain there too, but only there.
        free = float(weights["free fluid"][usable].sum())
        share = free / float(weights["domain"][usable].sum())
        rebuilt = (
            share * summaries["free fluid"] + (1.0 - share) * summaries["inclusions"]
        )
        report.check(
            abs(rebuilt - summaries["domain"]) <= 1.0e-9 * max(1.0, abs(rebuilt)),
            f"{name}: the two bulk weights partition the domain",
            f"reconstructed {rebuilt:.12g} against {summaries['domain']:.12g}",
        )
        report.check(
            worst <= locking,
            f"{name}: the free fluid is not locked to the mesh",
            f"c = {worst:.4f} at frame {worst_frame}, against {locking:g} allowed "
            f"and {patches.noise_ceiling():.3f} for noise; a velocity this "
            f"structured at the grid scale is carrying the mesh rather than the "
            f"physics through the transport rows",
        )

    def weights(
        self,
        run: "Run",
        parameters: "PhaseFieldSystemParameters",
        patches: "FacetPatches",
    ) -> "dict[str, npt.NDArray[np.float64]]":
        """One cell weight per measure of the mixture, all on the same cells.

        ``phi_incl`` and its complement are read from the geometry rather than
        from a saved indicator, so a run that did not save one is still
        answerable and the weights cannot disagree with the spec that produced
        them.
        """

        domain = parameters.solver.diffuse_domain
        space = run.cell_space()
        eps = float(domain.domain_eps)
        free = np.asarray(domain.bulk_indicator_on(space), dtype=float)

        # ``dgamma_eps`` summed per inclusion, not the norm of the summed gradient,
        # which counts two overlapping bands once. By the eikonal property
        # |grad r| = 1, so |grad phi_a| = (6/eps) phi_a (1 - phi_a) with no
        # gradient left in it.
        distances = np.asarray(domain.signed_distances_on(space), dtype=float)
        smeared = 0.5 * (1.0 - np.tanh(3.0 * distances / eps))
        surface = (6.0 / eps) * (smeared * (1.0 - smeared)).sum(axis=0)

        return {
            "free fluid": patches.volumes * free,
            "inclusions": patches.volumes * (1.0 - free),
            "surfaces": patches.volumes * surface,
            "domain": patches.volumes,
        }


def saved_velocities(run: "Run") -> "list[str]":
    """Every saved field that is a velocity, by the element it was stored in.

    A velocity is a continuous vector of the mesh's own dimension, which is what
    the rate solve holds one in and what nothing else saved here is. Read off the
    run rather than rebuilt from the mixture, so a spec that saved only one of a
    phase's two velocities is answered for the one it saved.
    """

    dim = run.mesh.geometry.dim
    return [
        name
        for name in run.names
        if tuple(run.field(name).shape) == (dim,)
        and not run.field(name).is_cellwise
        and not run.field(name).is_quadrature
    ]


def weighted_mean(
    values: "npt.NDArray[np.float64]",
    weight: "npt.NDArray[np.float64]",
    usable: "npt.NDArray[np.bool_]",
) -> float:
    """``values`` averaged over the cells the fit could reach.

    Cells whose patch has no residual space are dropped rather than counted as
    zero, which would pull the mean towards "smooth" exactly where the measure
    had nothing to say.
    """

    total = float(weight[usable].sum())
    if total <= 0.0:
        return float("nan")
    return float(np.dot(values[usable], weight[usable]) / total)


class FacetPatches:
    """The mesh as the transport rule sees it: one number per interior facet.

    Built once per run, since the geometry does not move. Everything a frame
    needs -- which facets a patch owns, their lengths, their normals, and which
    P2 nodes to read a velocity off -- is topology and is settled here.
    """

    def __init__(self, run: "Run") -> None:
        # Local: scipy.spatial costs a third of a second, and every CLI
        # invocation imports this module to register the diagnostic.
        from scipy.spatial import cKDTree

        mesh = run.mesh
        dim = mesh.topology.dim
        if dim != 2:
            raise Unanswerable(
                f"this reads a triangle's facet patch and the mesh is {dim}D; a "
                f"tet patch carries 16 facets against 12 unknowns, which is too "
                f"close to rank-deficient to score without widening the patch"
            )
        self.dim = dim
        self.unknowns = AFFINE_UNKNOWNS[dim]

        mesh.topology.create_connectivity(dim - 1, dim)
        mesh.topology.create_connectivity(dim, dim - 1)
        mesh.topology.create_connectivity(dim - 1, 0)
        facet_cells = mesh.topology.connectivity(dim - 1, dim)
        counts = np.diff(facet_cells.offsets)
        interior = np.flatnonzero(counts == 2)
        self.facets = int(interior.size)
        self.cells = int(mesh.topology.index_map(dim).size_local)

        # Vertex coordinates through the geometry map: topology vertex numbering
        # and geometry node numbering are not guaranteed to agree.
        ends = entities_to_geometry(mesh, dim - 1, interior.astype(np.int32))
        points = np.asarray(mesh.geometry.x, dtype=float)[:, :dim]
        first, second = points[ends[:, 0]], points[ends[:, 1]]
        edge = second - first
        self.length = np.linalg.norm(edge, axis=1)
        # Rotate the edge by -90 degrees for a normal whose sign is a property of
        # the facet rather than of whichever cell is asking, so that u_F means the
        # same thing in both patches that contain it.
        self.normal = np.column_stack((edge[:, 1], -edge[:, 0])) / self.length[:, None]
        self.midpoint = 0.5 * (first + second)

        # A P2 field's nodes on a facet are its two ends and its midpoint, so
        # Simpson over those three is exact for the quadratic and int_F v.n dS
        # is read straight off the dofs with no quadrature and no evaluation.
        space = run.field(next(iter(saved_velocities(run)))).space
        nodes = np.asarray(space.tabulate_dof_coordinates(), dtype=float)[:, :dim]
        tree = cKDTree(nodes)
        self.simpson = np.empty((self.facets, 3), dtype=np.int64)
        for column, target in enumerate((first, self.midpoint, second)):
            distance, index = tree.query(target)
            worst = float(np.max(distance))
            if worst > 1.0e-10 * float(np.max(self.length)):
                raise Unanswerable(
                    f"a velocity node is {worst:.3e} from a facet node, so this "
                    f"space does not carry its dofs at the vertices and edge "
                    f"midpoints and Simpson would not be exact on it"
                )
            self.simpson[:, column] = index
        self.block = space.dofmap.bs
        self.nodes = len(nodes)

        self.volumes = np.asarray(
            mesh_cell_volumes(mesh, quadrature_degree=run.quadrature_degree),
            dtype=float,
        )
        self.centroid = np.asarray(
            cell_centroids(mesh).x.array, dtype=float
        ).reshape(-1, mesh.geometry.dim)[: self.cells, :dim]
        self._build_patches(mesh, dim, interior)
        self._ceiling: float | None = None

    ## Public

    def facet_flux(self, field: "Field", frame: int) -> "npt.NDArray[np.float64]":
        """``u_F = int_F v.n dS`` for every interior facet, exactly.

        Simpson over the facet's three P2 nodes, which is exact for a quadratic
        and is the whole of what the transport rule reads of ``v``.
        """

        return self.flux_of(
            np.asarray(field.values[frame], dtype=float).reshape(-1, self.block)
        )

    def flux_of(self, values: "npt.NDArray[np.float64]") -> "npt.NDArray[np.float64]":
        """``u_F`` from nodal values, which is the only way a facet flux is read.

        Shared with :meth:`noise_ceiling`, so the ceiling is scored on a field
        the velocity space can actually hold rather than on arbitrary facet data
        no continuous field produces.
        """

        ends = values[self.simpson[:, 0]] + values[self.simpson[:, 2]]
        middle = values[self.simpson[:, 1]]
        mean = (ends + 4.0 * middle) / 6.0
        return self.length * np.einsum("fi,fi->f", mean[:, : self.dim], self.normal)

    def residual_fraction(
        self, facet_flux: "npt.NDArray[np.float64]"
    ) -> "npt.NDArray[np.float64]":
        """``c_K`` per cell: the share of the patch's facet data no affine field
        explains, ``nan`` where the patch has no residual space."""

        index = np.where(self.valid, self.patch, 0)
        weight = np.where(self.valid, self.length[index], 0.0)
        normal = self.normal[index]
        offset = self.midpoint[index] - self.centroid[:, None, :]
        # Rows of (a + G d).n against the facet-mean normal velocity u_F/|e_F|.
        # The coefficient of G_ij is n_i d_j, flattened row-major after a.
        stretch = (normal[..., :, None] * offset[..., None, :]).reshape(
            self.cells, -1, self.dim * self.dim
        )
        design = np.concatenate([normal, stretch], axis=-1)
        data = np.where(self.valid, facet_flux[index] / np.where(weight > 0, weight, 1.0), 0.0)

        root = np.sqrt(weight)[..., None]
        scaled, target = design * root, data * root[..., 0]
        fitted = np.einsum(
            "cij,cj->ci", scaled, np.einsum("cij,cj->ci", np.linalg.pinv(scaled), target)
        )
        residual = np.linalg.norm(target - fitted, axis=1)
        total = np.linalg.norm(target, axis=1)
        out = np.divide(residual, total, out=np.zeros(self.cells), where=total > 0.0)
        return np.where(self.measurable, out, np.nan)

    def churn(self, facet_flux: "npt.NDArray[np.float64]") -> float:
        """Gross facet flux against net, over the whole mesh.

        Unbounded by construction, and deliberately not squeezed into a range: it
        is the multiplier on the transport work that locking buys, and what eats
        the step. A velocity whose facet fluxes cancel within a patch moves
        nothing while still paying the CFL bound for every facet it crosses.
        """

        gross = float(np.abs(facet_flux).sum())
        if gross == 0.0:
            # Identically zero, as the initial condition is: no churn rather
            # than infinite churn.
            return float("nan")
        net = abs(float(facet_flux.sum()))
        return gross / net if net > 0.0 else float("inf")

    def noise_ceiling(self) -> float:
        """Where white noise in the velocity's own space scores.

        The top of the range in practice: ``c_K = 1`` needs facet data no
        continuous field can produce. Measured on the mesh rather than assumed,
        because it depends on the patch shapes this mesh happens to have.
        """

        if self._ceiling is None:
            generator = np.random.default_rng(0)
            scores = [
                float(
                    np.nanmedian(
                        self.residual_fraction(
                            self.flux_of(
                                generator.normal(size=(self.nodes, self.block))
                            )
                        )
                    )
                )
                for _ in range(CEILING_DRAWS)
            ]
            self._ceiling = float(np.median(scores))
        return self._ceiling

    ## Private helpers

    def _build_patches(self, mesh: Any, dim: int, interior: np.ndarray) -> None:
        """Which interior facets each cell's patch owns, and the rows they make.

        The patch is the cell plus its facet neighbours, and the facet set is
        theirs together with duplicates removed -- nine for an interior triangle,
        fewer against the boundary. Padded to a fixed width so the fit is one
        batched solve rather than a loop, with the padding carrying zero weight.
        """

        cell_facets = mesh.topology.connectivity(dim, dim - 1)
        facet_cells = mesh.topology.connectivity(dim - 1, dim)
        # Interior facets renumbered densely, so a patch indexes into the arrays
        # built above; an exterior facet maps to -1 and is dropped from a patch.
        dense = np.full(mesh.topology.index_map(dim - 1).size_local, -1, dtype=np.int64)
        dense[interior] = np.arange(interior.size)

        width = 3 * (self.dim + 1)
        self.patch = np.full((self.cells, width), -1, dtype=np.int64)
        for cell in range(self.cells):
            own = cell_facets.links(cell)
            neighbours = [
                other
                for facet in own
                for other in facet_cells.links(facet)
                if other != cell
            ]
            owned = {
                int(dense[facet])
                for source in (cell, *neighbours)
                for facet in cell_facets.links(source)
                if dense[facet] >= 0
            }
            chosen = sorted(owned)[:width]
            self.patch[cell, : len(chosen)] = chosen

        self.valid = self.patch >= 0
        self.measurable = self.valid.sum(axis=1) > self.unknowns
