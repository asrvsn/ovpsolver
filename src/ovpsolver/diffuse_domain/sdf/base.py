"""The geometry a run is posed on, as signed distances the user writes.

The diffuse-domain method needs one thing of the excluded region: a signed
distance ``r_a`` per inclusion, negative inside it and zero on its surface.
:func:`phase` smears it into a phase, differentiating that gives a surface
measure, and the indicator is one minus the sum of the phases. So that is the
whole interface, and a spec names a Python file that supplies it::

    import ufl
    from ovpsolver.diffuse_domain import sphere
    from ovpsolver.mesh import OuterDomain, disk

    def outer_domain(rng_seed: int) -> OuterDomain:
        \"\"\"The domain the mesh is built on, and whose boundary is Sigma.\"\"\"
        return disk(radius=1.0)

    def signed_distance_functions(
        rng_seed: int, x: ufl.SpatialCoordinate, eps: float
    ) -> list[ufl.core.expr.Expr]:
        \"\"\"One signed distance per inclusion, in the spatial coordinate.\"\"\"
        return [sphere((0.3, 0.1), 0.15)(x)]

Both take the spec's ``system.rng_seed``, so an arrangement drawn from a rule
(:mod:`ovpsolver.mesh.point_configurations`) is reproducible from the spec. Nothing
in the package assumes a shape: inclusions may differ from one another, and the only
dimension is the one ``x`` has.

The outer domain is deliberately a small closed set (:mod:`ovpsolver.mesh.shapes`).
An inclusion is only ever smeared and sampled, so any expression will do, but
``Sigma`` carries boundary conditions and the mesh has to end exactly on it, which
needs an entity a mesher can put nodes on. It is declared beside the inclusions
because the mesh is built from both: changing either is a different geometry, not a
different setting.

Why UFL and not numpy
---------------------
The facet terms have no other option. Every upwind flux, the two-point Korteweg
stiffness and the saturation row need the indicator on a facet, where only an
expression has a value; a discrete copy would have to be interpolated, and the
interpolant of a profile this steep is off by about nine per cent and carries the
mesh's own symmetry. Offline nothing is lost: :func:`evaluate` gets an expression
exactly at a space's own points, so one definition serves the solve, the
diagnostics and the pictures.

UFL also differentiates the signed distance, so a file that states ``r_a`` has
stated ``grad r_a`` too. For a true distance ``|grad r_a| = 1``, and the surface
density of one inclusion is then ``(6/eps) phi_a (1 - phi_a)``, a form offline
readers may use.
"""

from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import ufl
from dolfinx import fem

from ...fem.save.element import interpolation_expression
from ...mesh.shapes import OuterDomain

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from ufl.core.expr import Expr

#: What a geometry file defines: one signed distance per inclusion, the outer
#: domain the mesh is built on, and optionally the files the geometry reads.
ENTRY_POINT = "signed_distance_functions"
OUTER_DOMAIN_ENTRY_POINT = "outer_domain"
ASSETS_ENTRY_POINT = "assets"


def sphere(
    center: Sequence[float],
    radius: float,
) -> Callable[[ufl.SpatialCoordinate], Expr]:
    """The signed distance to a sphere.

    A convenience, not a special case: it returns the same kind of expression a
    hand-written shape would. ``center`` fixes the dimension.
    """

    coordinates = tuple(float(value) for value in center)
    if not coordinates:
        raise ValueError("a sphere's center needs at least one coordinate")
    if float(radius) <= 0.0:
        raise ValueError(f"a sphere's radius must be positive, not {radius}")

    def signed_distance(x):
        offset = x - ufl.as_vector(coordinates)
        return ufl.sqrt(ufl.dot(offset, offset)) - float(radius)

    return signed_distance


def _import(path: str | Path) -> tuple[Path, Any]:
    """A geometry file's path, and the file imported as a module.

    Imported under a private module name rather than added to the import path, so
    that a run's geometry cannot shadow an installed module and two runs in one
    session cannot shadow each other. Executed afresh on every call.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"no geometry definitions here: {source}")

    specification = importlib.util.spec_from_file_location(
        f"_ovpsolver_geometry_{abs(hash(str(source.resolve())))}", source
    )
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot import geometry definitions: {source}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return source, module


def _entry(source: Path, module: Any, name: str, expected: str) -> Callable:
    """The callable ``name`` a geometry file defines, or an error saying what it has to be."""

    entry = getattr(module, name, None)
    if entry is None:
        raise AttributeError(f"{source} defines no {name!r}; {expected}")
    if not callable(entry):
        raise TypeError(f"{source}: {name} has to be callable")
    return entry


def load(
    path: str | Path, rng_seed: int
) -> Callable[[ufl.SpatialCoordinate, float], Sequence[Expr]]:
    """The ``signed_distance_functions`` a geometry file defines, seed bound.

    Bound so that everything downstream calls one thing of ``(x, eps)``, and the
    seed reaches the geometry without being carried through every form.
    """

    source, module = _import(path)
    entry = _entry(
        source,
        module,
        ENTRY_POINT,
        "it has to be a function of the run's seed, the spatial coordinate and "
        "the smearing parameter returning one signed distance per inclusion",
    )
    return partial(entry, int(rng_seed))


def load_outer_domain(path: str | Path, rng_seed: int) -> OuterDomain:
    """The outer domain a geometry file declares, as a shape gmsh can build."""

    source, module = _import(path)
    entry = _entry(
        source,
        module,
        OUTER_DOMAIN_ENTRY_POINT,
        "it has to be a function of the run's seed returning one of the shapes "
        "in ovpsolver.mesh, such as disk(radius=1.0)",
    )
    produced = entry(int(rng_seed))
    if not isinstance(produced, OuterDomain):
        raise TypeError(
            f"{source}: {OUTER_DOMAIN_ENTRY_POINT} returned "
            f"{type(produced).__name__}, which is not an "
            f"ovpsolver.mesh.OuterDomain. Return one of the shapes in "
            f"ovpsolver.mesh, such as disk(radius=1.0), or a subclass of "
            f"OuterDomain that can build itself in gmsh"
        )
    return produced


def load_assets(path: str | Path) -> tuple[Path, ...]:
    """The files a geometry file reads, so that a run can archive them beside it.

    Optional, and empty for a geometry that is only expressions. A geometry built
    from a CAD export needs its source kept: the indicator is recomputed from the
    geometry rather than saved per frame, so a run that lost what its geometry
    reads could not say where its own inclusions were.

    Paths are resolved against the geometry file, which they sit beside now and
    once archived. Each must exist, so a missing one is refused while the run is
    set up rather than when its output is first plotted.
    """

    source, module = _import(path)
    entry = getattr(module, ASSETS_ENTRY_POINT, None)
    if entry is None:
        return ()
    if not callable(entry):
        raise TypeError(f"{source}: {ASSETS_ENTRY_POINT} has to be callable")

    resolved = []
    for asset in entry():
        asset = Path(asset).expanduser()
        if not asset.is_absolute():
            asset = source.parent / asset
        if not asset.is_file():
            raise FileNotFoundError(
                f"{source}: {ASSETS_ENTRY_POINT} names {asset}, which is not a "
                f"file. A geometry has to be able to read what it declares"
            )
        resolved.append(asset)
    return tuple(resolved)


def signed_distances(
    definitions: Callable[[ufl.SpatialCoordinate, float], Sequence[Expr]],
    mesh,
    eps: float,
) -> tuple[Expr, ...]:
    """Call a loaded definition on ``mesh``'s spatial coordinate, checking the result.

    A file returning the wrong sort of thing is refused here, in the file's own
    vocabulary, rather than failing later inside a form.

    ``eps`` is the run's smearing parameter, handed to the geometry so that one
    that approximates a shape (a CAD outline refitted as arcs, say) chooses its
    accuracy against the ``2 eps`` the method resolves, and never against a copy of
    the number that could go stale.

    An empty sequence is a geometry: the outer shape with nothing inside it (see
    :attr:`~ovpsolver.diffuse_domain.DiffuseDomain.uniform`).
    """

    produced = definitions(ufl.SpatialCoordinate(mesh), eps)
    if isinstance(produced, (str, bytes)) or not isinstance(produced, Sequence):
        raise TypeError(
            f"{ENTRY_POINT} has to return a sequence of signed distances, "
            f"not {type(produced).__name__}"
        )
    distances = tuple(ufl.as_ufl(one) for one in produced)
    for index, distance in enumerate(distances):
        if distance.ufl_shape != ():
            raise ValueError(
                f"{ENTRY_POINT} returned a signed distance of shape "
                f"{distance.ufl_shape} at index {index}; each one has to be a scalar"
            )
    return distances


#: Where :func:`phase` clamps its ``tanh`` argument: past ``tanh``'s saturation
#: in double precision, short of where UFL's derivative of it overflows.
PROFILE_ARGUMENT_BOUND = 300.0


def phase(signed_distance: Expr, eps: float | Expr) -> Expr:
    """One inclusion's diffuse phase ``phi_a = (1 - tanh(3 r_a / eps)) / 2``.

    ``eps`` is the smearing parameter, not the width of the profile, which is
    ``2 eps``.

    The argument is clamped to :data:`PROFILE_ARGUMENT_BOUND`, which changes no
    value: ``tanh`` is exactly one in double precision past about 19. It
    changes the derivative, which UFL writes as ``(2 cosh x / (cosh 2x + 1))^2``
    and which overflows to zero past 355 and to NaN past 710 -- 237 ``eps``
    from every inclusion, not far on a large domain with a thin interface.
    Clamped, it is an exact zero there instead.
    """

    bound = PROFILE_ARGUMENT_BOUND
    argument = ufl.max_value(ufl.min_value(3.0 * signed_distance / eps, bound), -bound)
    return 0.5 * (1.0 - ufl.tanh(argument))


def evaluate(expression: Expr, space: fem.FunctionSpace) -> npt.NDArray[np.float64]:
    """An expression's values at ``space``'s own interpolation points, exactly.

    Interpolation here is evaluation, not approximation: for the cell-constant and
    vertex spaces a reader draws and masks with, the points are the cell centroids
    and the vertices.
    """

    function = fem.Function(space)
    function.interpolate(interpolation_expression(expression, space))
    return np.asarray(function.x.array, dtype=float).copy()


__all__ = [
    "ASSETS_ENTRY_POINT",
    "ENTRY_POINT",
    "OUTER_DOMAIN_ENTRY_POINT",
    "evaluate",
    "load",
    "load_assets",
    "load_outer_domain",
    "phase",
    "signed_distances",
    "sphere",
]
