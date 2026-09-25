"""Direct nodal access and facet rules for DG0 (finite-volume) fields.

For piecewise-constant ``Discontinuous Lagrange`` degree-0 spaces the nodal
degrees of freedom coincide with the per-cell values, so the chemical timestep
can read and write them straight off the backing dof array without any
interpolation. The views returned here alias that array: in-place mutation
updates the underlying :class:`dolfinx.fem.Function`.

Since every field of the mixture is cell-constant, this is also where the facet
rules live that a cell-constant field needs and a nodal one does not. A
divergence becomes a sum of upwind facet fluxes (:func:`dg0_upwind_ibp`), a
gradient energy becomes a sum over neighbouring cell pairs
(:func:`two_point_stiffness`), and a gradient itself becomes a least-squares fit
to the slopes across a cell's facets (:func:`dg0_least_squares_gradient`) -- or,
along a direction the geometry knows, a slope fitted once to be exact on that
geometry (:class:`FacetStencilSlope`). Every row that moves a phase takes its
facet flux from :func:`upwind_numerical_flux`, so the phase is transported the
same way wherever it appears.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import ufl
from basix.ufl import element as basix_element
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.mesh import Mesh, compute_midpoints
from ufl.core.expr import Expr

from ..types import Function, FunctionSpace

if TYPE_CHECKING:
    from ...solver.parameters import SolverParameters

#: The family name a cell-constant element is declared with. The family only; the
#: degree is a separate field of the declaration, and it is degree zero that makes
#: an element of this family cell-constant. Here rather than restated at every
#: declaration, since every state of the mixture is one.
DISCONTINUOUS_LAGRANGE = "Discontinuous Lagrange"


def cell_centroids(mesh: Mesh) -> Function:
    """Cell midpoints, as a DG0 vector field.

    The two-point rules below are ``dS`` integrals of a jump, scaled by the
    distance between the two cells' centroids. As a DG0 field rather than an
    array this composes with UFL, where ``jump`` of it is the vector joining the
    two cells of a facet.
    """

    space = fem.functionspace(
        mesh,
        basix_element(
            DISCONTINUOUS_LAGRANGE, mesh.basix_cell(), 0, shape=(mesh.geometry.dim,)
        ),
    )
    topology = mesh.topology
    tdim = topology.dim
    index_map = topology.index_map(tdim)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    points = compute_midpoints(mesh, tdim, cells)
    function = fem.Function(space, name="cell_centroids")
    function.x.array[:] = points[:, : mesh.geometry.dim].reshape(-1)
    function.x.scatter_forward()
    return function


def inverse_centroid_distance(centroids: Function) -> Expr:
    """``1 / |x_+ - x_-|`` on an interior facet, for a two-point rule.

    Paired with ``dS`` this is the transmissibility ``|e| / d_e`` of a two-point
    flux, the facet area arriving from the measure. Consistent only where the
    centroid join is parallel to the facet normal: on a skewed facet the
    transmissibility is wrong by the cosine of the angle between them, so a
    two-point rule is a statement about the mesh as well as the equation.
    """

    join = ufl.jump(centroids)
    return 1.0 / ufl.sqrt(ufl.dot(join, join))


def two_point_stiffness(
    coefficient: Expr,
    trial: Expr,
    test: Expr,
    centroids: Function,
    dS: ufl.Measure,
) -> "ufl.Form":
    r"""The cell-pair form standing in for ``\int coefficient grad(u).grad(w)``.

        sum_e coefficient_e (|e| / d_e) jump(u) jump(w)

    A cell-constant field has no interior gradient, so a gradient energy reaches
    the rows as a sum over neighbouring pairs instead: the derivative of
    ``(1/2) sum_e coefficient_e (|e| / d_e) jump(u)^2``, declared here and only
    here so the row and the energy cannot drift apart.

    ``coefficient`` is averaged across the facet and so has to be restrictable
    there: the analytic expression for a diffuse weight, never the quadrature
    field, which has no value on a facet at all. A continuous expression makes
    ``avg`` a formality rather than an approximation.
    """

    return (
        ufl.avg(coefficient)
        * ufl.jump(trial)
        * ufl.jump(test)
        * inverse_centroid_distance(centroids)
        * dS
    )


def dg0_least_squares_gradient(
    solver_parameters: "SolverParameters",
    scalar: Expr,
    gradient: Expr,
    test: Expr,
) -> "ufl.Form":
    r"""Row fitting ``gradient`` to the slopes a cell sees across its facets.

    A cell-constant field has no interior gradient, so one has to be reconstructed
    from the neighbours. This is the least-squares fit: per cell, minimize

        (1/2) sum_e |e| (g . nhat_e - jump(u)_e / d_e)^2

    over the interior facets ``e`` of that cell, where ``nhat_e`` is the unit
    vector along the centroid join and ``d_e`` its length. The row below is its
    derivative, so the fit and the residual cannot drift apart.

    Chosen because it is *exact* for a linear field on any mesh, whatever the
    weights: for ``u = a.x`` the measured slope ``jump(u)/d`` is ``a.nhat`` on
    every facet, so ``g = a`` annihilates every residual term at once. A two-point
    estimate along a single join, or a Green-Gauss sum of facet averages, is exact
    only where the mesh cooperates -- the join parallel to the normal, or the facet
    midpoint on the join -- and elsewhere makes an error that depends on
    orientation. On a mesh with any lattice coherence that is a preferred
    direction, and a preferred direction in an operator is a force. On a triangle
    the three joins sit near 120 degrees apart, so the normal matrix is close to
    ``(3/2)|e| I`` and well conditioned.

    The exterior term states ``g . n = 0`` on ``Sigma``, weighted as one more
    neighbour, for consistency rather than as a regularization: with no wetting
    energy on the outer wall that is the natural boundary condition of the
    gradient energy, which :func:`two_point_stiffness` already states by carrying
    no ``ds`` term. It also keeps the row invertible for a corner cell with two
    boundary facets, whose one interior join alone gives a rank-one normal matrix.
    """

    centroids = solver_parameters.cell_centroids()
    inverse_distance = inverse_centroid_distance(centroids)
    direction = ufl.jump(centroids) * inverse_distance
    slope = ufl.jump(scalar) * inverse_distance

    interior = ufl.as_ufl(0.0)
    for side in ("+", "-"):
        interior += (ufl.dot(gradient(side), direction) - slope) * ufl.dot(
            test(side), direction
        )

    normal = ufl.FacetNormal(solver_parameters.dolfinx_mesh)
    return interior * solver_parameters.get_mesh_dS() + ufl.dot(
        gradient, normal
    ) * ufl.dot(test, normal) * solver_parameters.get_mesh_ds()


def dg0_least_squares_gradient_of(
    solver_parameters: "SolverParameters", scalar: Function
) -> Function:
    """The reconstruction of :func:`dg0_least_squares_gradient`, solved.

    For a reader: the solve has no gradient unknown, and reads a cell-constant
    field's slopes through its own facet rules. A diagnostic reconstructing the
    gradient this way measures it independently of those rules, and exactly for
    a linear field on any mesh.

    Cheap: each facet term reads one side's gradient against the same side's
    test function, so the system is block-diagonal, one small symmetric positive
    definite block per cell, and takes well under a second on a seventy-thousand
    cell mesh.
    """

    mesh = solver_parameters.dolfinx_mesh
    space = fem.functionspace(
        mesh,
        basix_element(
            DISCONTINUOUS_LAGRANGE, mesh.basix_cell(), 0, shape=(mesh.geometry.dim,)
        ),
    )
    form = dg0_least_squares_gradient(
        solver_parameters, scalar, ufl.TrialFunction(space), ufl.TestFunction(space)
    )
    gradient = fem.Function(space, name="grad_phi")
    fem_petsc.LinearProblem(
        ufl.lhs(form),
        ufl.rhs(form),
        u=gradient,
        petsc_options_prefix="dg0_least_squares_gradient_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
    return gradient


@dataclass(frozen=True)
class FacetStencilSlope:
    r"""A cell functional of a slope, estimated over the facet neighbours.

    ``L_T[u] = int_T grad(u).a dx`` has no value for a cell-constant ``u``, so it is
    estimated from the cell's facet neighbours ``k``,

        L_T[u] ~ sum_k beta_Tk (u_k - u_T),

    with the coefficients fitted once per cell to reproduce ``L_T`` exactly on a
    chosen set of functions (:func:`fit_facet_stencil_slope`). Whatever the set, the
    fitted ``beta`` lies in the row space of the conditions, so it factors as

        beta_Tk = tangential_T . (x_k - x_T) + radial_T (r_k - r_T)
                  + quadratic_T (r_k - r_T)^2,

    three DG0 coefficients contracted against what an interior-facet integral
    already has -- the centroid join and the jump of a DG0 distance. Nothing is
    stored per facet, and the operator reaches a row as one interior-facet form on
    the stencil a two-point stiffness uses: linear in the live field, no unknown,
    no wider sparsity.
    """

    tangential: Function
    radial: Function
    quadratic: Function
    distance: Function

    def pairing(
        self, scalar: Expr, test: Expr, centroids: Function, dS: ufl.Measure
    ) -> "ufl.Form":
        """``sum_T test_T L_T[scalar]``, as one interior-facet form.

        Each facet carries both of its cells' contributions, each read from its own
        side. Everything in the integrand is constant along the facet, so dividing
        by the facet area turns the facet integral into the plain sum over
        neighbour pairs. An exterior facet contributes nothing, which is what the
        fit assumed by giving a missing neighbour a zero column.
        """

        total = ufl.as_ufl(0.0)
        for own, other in (("+", "-"), ("-", "+")):
            rise = self.distance(other) - self.distance(own)
            coefficient = (
                ufl.dot(self.tangential(own), centroids(other) - centroids(own))
                + self.radial(own) * rise
                + self.quadratic(own) * rise**2
            )
            total += test(own) * coefficient * (scalar(other) - scalar(own))
        # Single-valued on the facet, but UFL lowers it through a cell's facet
        # Jacobian before it applies default restrictions, so a side has to be named.
        return total / ufl.FacetArea(centroids.function_space.mesh)("+") * dS


def fit_facet_stencil_slope(
    distance: Expr,
    kernel: Expr,
    centroids: Function,
    dx: ufl.Measure,
) -> FacetStencilSlope:
    r"""Fit ``L_T[u] = int_T grad(u).a dx`` over each cell's facet neighbours.

    ``a`` is the ``kernel`` and ``r`` the ``distance``, both analytic expressions.
    Per cell the coefficients reproduce ``L_T`` exactly on

        u = r - r_T,    u = (r - r_T)^2,    u = t.(x - x_T) for every t orthogonal to n_T,

    with ``n_T`` the direction of ``a`` at the centroid -- square on a simplex, three
    conditions on a triangle's three neighbours and four on a tetrahedron's four. The
    targets are the cell integrals ``int_T grad(r).a``, ``int_T 2 (r - r_T) grad(r).a``
    and ``P_T int_T a``, with ``P_T = I - n_T n_T``, assembled once on ``dx``.

    The integral and not a slope read at the centroid, because that is what the row
    contains: a centroid slope times the cell's integral of the kernel drops the
    covariance of the two across the cell, and with a kernel that changes by an
    order of magnitude across a band cell that is a term of order ``U''``, set by
    each cell's shape, which undoes most of what the quadratic condition is for.

    Why these functions: for ``u = U(r)``, ``L_T[u] = int_T U'(r) grad(r).a`` with
    ``U'(r)`` linear in ``r - r_T`` to first order, so the first two conditions make
    the estimate exact through second order for any profile of the distance. What is
    left is ``U^(3)`` against the difference between the stencil's third moment and
    the cell's, which still depends on how the stencil is oriented -- the imprint is
    of higher order, not removed. Curvature enters only through the actual values
    ``r_k`` and the targets, so nothing assumes a shape. The tangential condition
    makes the estimate blind to variation along the surface and keeps the system
    regular: a third-order condition in ``r`` in its place would be singular
    wherever two neighbours sit symmetrically about the normal. Fitting about
    ``r_T`` keeps the conditioning independent of where the cell sits.

    A neighbour a cell lacks, across the outer boundary, is a zero column, and the
    minimum-norm solution is taken. The tangential condition is written as ``d``
    rows of the projector, rank ``d - 1``, which the pseudo-inverse resolves without
    a tangent frame. Rows and targets are scaled by the longest join. Values are read
    and written through the dofmaps, so the fit sees exactly what a form restricted
    to either side of a facet sees. Ghost cells are fitted from the connectivity this
    process holds, which is the whole stencil in serial only.
    """

    vectors = centroids.function_space
    mesh = vectors.mesh
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    scalars = fem.functionspace(
        mesh, basix_element(DISCONTINUOUS_LAGRANGE, mesh.basix_cell(), 0)
    )

    midpoints = scalars.element.interpolation_points
    level = fem.Function(scalars, name="stencil_slope_distance")
    level.interpolate(fem.Expression(distance, midpoints))
    level.x.scatter_forward()
    along = fem.Function(vectors, name="stencil_slope_kernel")
    along.interpolate(fem.Expression(kernel, midpoints))

    scalar_test = ufl.TestFunction(scalars)
    rate = ufl.dot(ufl.grad(distance), kernel)
    radial_target = np.array(
        fem.assemble_vector(fem.form(rate * scalar_test * dx)).array
    )
    quadratic_target = np.array(
        fem.assemble_vector(
            fem.form(2.0 * (distance - level) * rate * scalar_test * dx)
        ).array
    )
    tangential_target = np.array(
        fem.assemble_vector(
            fem.form(ufl.dot(kernel, ufl.TestFunction(vectors)) * dx)
        ).array
    ).reshape(-1, gdim)

    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    index_map = mesh.topology.index_map(tdim)
    count = index_map.size_local + index_map.num_ghosts
    cells = np.arange(count)
    cell_facets = mesh.topology.connectivity(tdim, tdim - 1).array.reshape(count, -1)
    facet_cells = mesh.topology.connectivity(tdim - 1, tdim)
    starts = facet_cells.offsets[:-1]
    first = facet_cells.array[starts]
    shared = np.diff(facet_cells.offsets) == 2
    second = np.where(
        shared,
        facet_cells.array[np.minimum(starts + 1, facet_cells.array.size - 1)],
        -1,
    )
    neighbours = np.where(
        first[cell_facets] == cells[:, None], second[cell_facets], first[cell_facets]
    )
    present = neighbours >= 0
    others = np.where(present, neighbours, cells[:, None])

    # Everything below is indexed by cell, read through the dofmaps.
    cell_dofs = np.asarray(scalars.dofmap.list)[:count, 0]
    block_dofs = np.asarray(vectors.dofmap.list)[:count, 0]
    points = centroids.x.array.reshape(-1, gdim)[block_dofs]
    values = level.x.array[cell_dofs]
    raw = along.x.array.reshape(-1, gdim)[block_dofs]
    length = np.linalg.norm(raw, axis=1)
    normal = raw / np.where(length > 0.0, length, 1.0)[:, None]
    projector = np.eye(gdim)[None, :, :] - normal[:, :, None] * normal[:, None, :]

    join = np.where(present[..., None], points[others] - points[:, None, :], 0.0)
    rise = np.where(present, values[others] - values[:, None], 0.0)

    scale = np.linalg.norm(join, axis=2).max(axis=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    rows = np.concatenate(
        [
            np.einsum("cgh,ckh->cgk", projector, join) / scale[:, None, None],
            (rise / scale[:, None])[:, None, :],
            (rise**2 / scale[:, None] ** 2)[:, None, :],
        ],
        axis=1,
    )
    target = np.concatenate(
        [
            np.einsum("cgh,ch->cg", projector, tangential_target[block_dofs])
            / scale[:, None],
            (radial_target[cell_dofs] / scale)[:, None],
            (quadratic_target[cell_dofs] / scale**2)[:, None],
        ],
        axis=1,
    )
    solved = np.einsum(
        "cij,cj->ci",
        np.linalg.pinv(rows @ rows.transpose(0, 2, 1), hermitian=True),
        target,
    )

    tangential = fem.Function(vectors, name="stencil_slope_tangential")
    tangential.x.array.reshape(-1, gdim)[block_dofs] = np.einsum(
        "cgh,ch->cg", projector, solved[:, :gdim] / scale[:, None]
    )
    radial = fem.Function(scalars, name="stencil_slope_radial")
    radial.x.array[cell_dofs] = solved[:, gdim] / scale
    quadratic = fem.Function(scalars, name="stencil_slope_quadratic")
    quadratic.x.array[cell_dofs] = solved[:, gdim + 1] / scale**2
    for function in (tangential, radial, quadratic):
        function.x.scatter_forward()
    return FacetStencilSlope(tangential, radial, quadratic, level)


def mesh_cell_volumes(mesh: Mesh, *, quadrature_degree: int) -> np.ndarray:
    """The volume of every cell of ``mesh``, on its own scalar DG0 space.

    A cell-constant field's dofs *are* its cell values, so the volumes come back
    as a plain array indexed the same way every other cell-wise quantity is.
    """

    space = fem.functionspace(
        mesh, basix_element(DISCONTINUOUS_LAGRANGE, mesh.basix_cell(), 0)
    )
    return cell_volumes(space, quadrature_degree=quadrature_degree)


def cell_volumes(
    space: FunctionSpace,
    *,
    quadrature_degree: int,
) -> np.ndarray:
    """Assemble fixed cell volumes for a scalar DG0 space."""

    test = ufl.TestFunction(space)
    dx = ufl.Measure(
        "dx",
        domain=space.mesh,
        metadata={"quadrature_degree": quadrature_degree},
    )
    return np.array(fem.assemble_vector(fem.form(test * dx)).array)


def cell_average_form(
    coefficient: Expr,
    space: FunctionSpace,
    *,
    quadrature_degree: int,
) -> fem.Form:
    """The compiled numerator ``int_K coefficient dx`` of DG0 cell averages.

    Compiled once, to be reassembled by :func:`assemble_cell_averages` whenever
    only coefficient values or constants change.
    """

    test = ufl.TestFunction(space)
    dx = ufl.Measure(
        "dx",
        domain=space.mesh,
        metadata={"quadrature_degree": quadrature_degree},
    )
    return fem.form(coefficient * test * dx)


def assemble_cell_averages(form: fem.Form, volumes: np.ndarray) -> np.ndarray:
    """Assemble DG0 cell averages from a cached numerator form and volumes."""

    numerator = fem.assemble_vector(form).array
    averages = np.zeros_like(volumes)
    np.divide(numerator, volumes, out=averages, where=volumes > 0.0)
    return averages


def cell_values(function: Function) -> np.ndarray:
    """Return a mutable ``(num_cells, block_size)`` view of a blocked DG0 field."""

    block_size = function.function_space.dofmap.bs
    return function.x.array.reshape(-1, block_size)


def cell_scalars(function: Function) -> np.ndarray:
    """Return a mutable ``(num_cells,)`` view of a scalar DG0 field."""

    return function.x.array


def add_scaled_identity(function: Function, scale: np.ndarray) -> None:
    """Add ``scale[cell] * I`` to every cell block of a square DG0 tensor field.

    Supports both the dense ``(n, n)`` packing and the basix ``symmetry=True``
    (column-major upper-triangular) packing by touching only the diagonal slots
    of each cell's flattened block.
    """

    rows, columns = function.ufl_shape
    if rows != columns:
        raise ValueError("add_scaled_identity requires a square tensor field")

    values = cell_values(function)
    block_size = values.shape[1]
    if block_size == rows * columns:
        diagonal = np.arange(rows) * (columns + 1)
    elif block_size == rows * (rows + 1) // 2:
        diagonal = np.array([j * (j + 1) // 2 + j for j in range(rows)])
    else:
        raise ValueError("unrecognised DG0 tensor packing")

    values[:, diagonal] += np.asarray(scale)[:, None]


def add_scaled_kronecker_outer(
    function: Function, scale: np.ndarray, dim: int
) -> None:
    """Add ``scale[cell] * vec(I) vec(I)^T`` to a symmetric ``(d^2, d^2)`` DG0 field.

    In the ``(a, c), (b, d)`` matricization that is the tensor
    ``delta^{ac} delta^{bd}``: one at every pair of the flattened diagonal indices
    ``a * d + a``. Basix packs a symmetric block's entry ``(i, j)``, ``i <= j``, at
    ``j (j + 1) / 2 + i``, each unordered pair once.
    """

    values = cell_values(function)  # (num_cells, d^2 (d^2 + 1) / 2)
    diag = [a * dim + a for a in range(dim)]  # the support of vec(I)
    scale = np.asarray(scale)
    for a in range(dim):
        for b in range(a, dim):  # each unordered pair once
            hi, lo = diag[b], diag[a]  # diag is increasing, so hi >= lo
            values[:, hi * (hi + 1) // 2 + lo] += scale


def upwind_numerical_flux(
    flux_pieces: Iterable[tuple[Expr, Expr, Expr]],
    normal: Expr,
) -> Expr:
    r"""The single-valued upwind flux across an interior facet, in ``+`` normal.

    ``J_up . n^+``: each piece contributes its density taken from whichever side
    the *upwind* velocity flows out of, times the normal component of its
    *carrier* velocity. One number per facet, which is what makes the scheme
    conservative: the value one cell loses is the value the other gains.

    Shared with the step bound rather than rebuilt there. The bound's claim is
    about what the row will do to a cell, so it has to measure the flux the row
    applies; a second copy of this expression could drift from this one, and the
    claim would quietly stop being true.

    Returns a zero of no particular shape for empty pieces, so a caller pairing it
    with a shaped weight should add its own zero of the right shape.
    """

    total = None
    for upwind_velocity, carrier_velocity, rho in flux_pieces:
        outflow = ufl.conditional(
            ufl.ge(ufl.dot(upwind_velocity("+"), normal("+")), 0.0),
            1.0,
            0.0,
        )
        rho_upwind = outflow * rho("+") + (1.0 - outflow) * rho("-")
        term = rho_upwind * ufl.dot(carrier_velocity("+"), normal("+"))
        total = term if total is None else total + term
    return ufl.as_ufl(0.0) if total is None else total


def positive_part(expression: Expr) -> Expr:
    """``max(expression, 0)`` component by component, whatever the shape.

    ``ufl.max_value`` is scalar-only, and a step bound on a vector-valued state --
    the binned generating function -- needs the positive part of each bin
    separately, since the bins are bounded one by one.
    """

    expression = ufl.as_ufl(expression)
    shape = expression.ufl_shape
    if shape == ():
        return ufl.max_value(expression, 0.0)
    if len(shape) == 1:
        return ufl.as_vector(
            [ufl.max_value(expression[i], 0.0) for i in range(shape[0])]
        )
    if len(shape) == 2:
        return ufl.as_matrix(
            [
                [ufl.max_value(expression[i, j], 0.0) for j in range(shape[1])]
                for i in range(shape[0])
            ]
        )
    raise ValueError(f"no component-wise positive part for shape {shape}")


def dg0_outflow(
    flux_pieces: Iterable[tuple[Expr, Expr, Expr]],
    test: Expr,
    normal: Expr,
    dS: ufl.Measure,
) -> "ufl.Form":
    """What each cell stands to lose across its interior facets, per unit time.

        O_K = sum_{F in K} max(Phi_F, 0),   Phi_F the upwind flux out of K,

    assembled against a DG0 ``test`` so each cell's row is its own ``O_K``. The
    flux is :func:`upwind_numerical_flux` and not a stand-in for it, because a
    bound on what the transport row does to a cell has to measure the flux that
    row applies. Oriented out of the ``+`` cell, so what ``-`` loses is the
    positive part of its negative.

    Interior facets only. A source or a boundary influx that drains a cell does so
    on a different measure, and the caller that has one adds it.
    """

    flux = upwind_numerical_flux(flux_pieces, normal)
    return ufl.inner(positive_part(flux), test("+")) * dS + ufl.inner(
        positive_part(-flux), test("-")
    ) * dS


def dg0_emptying_times(held: np.ndarray, lost: np.ndarray) -> np.ndarray:
    """``held / lost`` per dof: how long each cell sustains its outflow before emptying.

    The ratio whose minimum over the cells bounds the step. ``inf`` where nothing
    leaves, since a cell losing nothing sets no bound; and a content already below
    zero counts as empty rather than as a negative time, which would read as a
    bound already satisfied.
    """

    times = np.full(np.shape(held), np.inf)
    active = lost > 0.0
    times[active] = np.maximum(held[active], 0.0) / lost[active]
    return times


def dg0_upwind_ibp(
    solver_parameters: "SolverParameters",
    mu: Expr,
    flux_pieces: Iterable[tuple[Expr, Expr, Expr]],
    *,
    boundary_flux_pieces: Iterable[tuple[Expr, Expr, Expr]] | None = None,
) -> "ufl.Form":
    r"""Weak DG0 evaluation of ``\int_\Omega mu (div J) dx`` by upwind fluxes.

    The cell-by-cell divergence theorem, with the upwind facet flux. For a
    cell-constant weight ``mu`` and an advective flux ``J = sum_a v_a rho_a``,
    ``grad(mu)`` has no interior meaning, so the volume integral is assembled as
    an interior-facet jump of ``mu`` against the single-valued upwind numerical
    flux, plus an imposed boundary flux on Sigma:

        \int_\Omega mu (div J) dx
          = \sum_{F in interior} \int_F jump(mu) (J_up . n) dS
          + \int_Sigma mu (J . n) ds .

    Parameters
    ----------
    solver_parameters : shared mesh and quadrature parameters.
    mu : scalar or tensor DG0 weight: a chemical-potential expression for the OVP
        residual, the advected field's DG0 test function for the transport update.
    flux_pieces : iterable of ``(upwind_velocity, carrier_velocity, rho)``:
        * upwind_velocity -- sets the upwind direction; pass a LAGGED/known
          velocity so the form stays affine in the rate.
        * carrier_velocity -- supplies the magnitude ``v . n``; pass the
          velocity TEST function for the OVP residual, or the solved/live
          velocity for the transport update.
        * rho -- advected scalar or tensor DG0 coefficient. Fold only
          facet-restrictable continuous weights into rho, e.g. the analytic
          indicator expression ``chi_eps_analytic_ufl()``; a quadrature *element*
          has no facet value and cannot be used on ``dS``/``ds``, though the
          expression it tabulates can.
    boundary_flux_pieces : optional boundary-only flux pieces. When supplied,
        adds ``\int_Sigma mu * sum_a rho_a (carrier_velocity_a . n) ds``.

    Returns
    -------
    ufl.Form
        Form for ``\int mu (div J) dx``. Apply the contextual sign: ``-`` for
        the OVP energy-rate residual, ``+dt`` for the transport increment.
    """

    normal = ufl.FacetNormal(solver_parameters.dolfinx_mesh)
    interior_facets = solver_parameters.get_mesh_dS()
    numerical_flux = 0.0 * mu + upwind_numerical_flux(flux_pieces, normal)
    boundary_flux = 0.0 * mu
    if boundary_flux_pieces is not None:
        for _upwind_velocity, carrier_velocity, rho in boundary_flux_pieces:
            boundary_flux += rho * ufl.dot(carrier_velocity, normal)
    if ufl.as_ufl(mu).ufl_domains():
        form = ufl.inner(ufl.jump(mu), numerical_flux) * interior_facets
    else:
        form = 0.0 * solver_parameters.get_mesh_dx()
    if boundary_flux_pieces is not None:
        outer_boundary = solver_parameters.get_mesh_ds()
        form += ufl.inner(mu, boundary_flux) * outer_boundary
    return form