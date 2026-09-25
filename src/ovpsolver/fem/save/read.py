"""Reading a saved run back, as the functions the solver had.

Here rather than with the plotting because it is the other half of
:mod:`ovpsolver.fem.save.saver`: the two agree on a format, and a format whose
reader and writer live apart drifts.

What comes back is a :class:`dolfinx.fem.Function` on a mesh read out of the same
directory, on a space rebuilt from the mixture's own
:class:`~ovpsolver.fem.elements.ElementSpec`. Everything an analysis might want --
a value at a point, an integral, a gradient, the same field on a different space
-- is then a dolfinx call rather than something reimplemented here, and code
written to run inside the solve runs unchanged against a run that finished last
year.

The one operation worth having here is :meth:`Field.sample`, which puts a field
onto a space of the caller's choosing: by interpolation where the source can be
evaluated at points, and by L2 projection where it cannot. That is the one place
in the pipeline where a linear system is solved to make a picture, and it is
here, in the reader, where the caller asked for it.

A run is opened with the parameters it was run from wherever the caller has
them, which at an entry point it always does. They are not re-recorded in the
directory: the spec is archived there and parses against the same classes, so
asking it is asking the document that set the number rather than a copy of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.io import XDMFFile
from mpi4py import MPI

from ..elements import ElementSpec
from ..elements.dg0 import DISCONTINUOUS_LAGRANGE
from .element import describe, interpolation_expression, space_of
from .saver import MESH_FILE, META_FILE, STEPS_FILE, TIMES_FILE

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh

    from ...solver.parameters import SolverParameters

#: What a projection integrates at when the run's own degree is not to hand.
#: Only reached by a bare :class:`Run` opened without its parameters, which is
#: an ad-hoc read rather than an entry point.
DEFAULT_QUADRATURE_DEGREE = 6


@dataclass(frozen=True, slots=True)
class Field:
    """One saved field: its frames, and the space they are dofs of."""

    name: str
    run: "Run"
    values: np.ndarray
    element: ElementSpec
    mode: str
    #: Whether the one row this holds is the answer at every frame. Declared by
    #: whoever saved it, because a one-row array cannot say so for itself.
    static: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        """The value shape: empty for a scalar, ``(2,)`` for a planar vector."""

        return self.element.shape

    @property
    def is_quadrature(self) -> bool:
        return self.element.is_quadrature

    @property
    def is_cellwise(self) -> bool:
        """Whether one value is one cell's value, which is drawn flat."""

        return self.element.is_cellwise

    @property
    def n_frames(self) -> int:
        """How many rows this holds, which for a static field is one.

        The count of distinct values rather than of the frames they answer for,
        so a caller looping over it makes one pass over a field that has one
        value. Asking for any frame still works (see :meth:`function`), which
        keeps the distinction from reaching code that does not care.
        """

        return int(self.values.shape[0])

    @property
    def space(self) -> "fem.FunctionSpace":
        """The space these dofs belong to, rebuilt on the run's mesh.

        Held by the run rather than the field, so that two fields on the same
        element get the same object back and their dofs are comparable index for
        index.
        """

        return self.run.space(self.element)

    def function(self, frame: int) -> "fem.Function":
        """One frame, as the function the solver held at that step."""

        function = fem.Function(self.space, name=self.name)
        self.load_into(function, frame)
        return function

    def load_into(self, function: "fem.Function", frame: int) -> None:
        """Put one frame's dofs into ``function``, a function on :attr:`space`.

        In place, for a function a compiled form was built against: one rebuilt
        per frame would leave the form reading the first. Any frame of a static
        field is its one row, so a caller reading the geometry alongside a phase
        asks both for the same frame index and neither has to know which is which.
        """

        function.x.array[: self.values.shape[1]] = self.values[0 if self.static else frame]
        function.x.scatter_forward()

    def cell_values(self, frame: int) -> np.ndarray:
        """One frame on the run's cell space, shaped ``(cells, *shape)``.

        Every phase and state already lives there, so for those these are the
        saved dofs; a continuous field, such as a P2 velocity, is evaluated at
        the centroids.
        """

        sampled = self.sample(frame, self.run.cell_space(self.shape))
        return np.asarray(sampled.x.array, dtype=float).reshape(-1, *self.shape)

    def sample(
        self, frame: int, space: "fem.FunctionSpace | None" = None
    ) -> "fem.Function":
        """One frame, put onto ``space``, or left on the field's own space.

        The default is the field the solver held, unresampled. Anything else has
        to be asked for, because resampling is lossy in ways that do not announce
        themselves -- and with the scheme cell-constant, a cell-constant field
        sampled onto P1 is the common case rather than a corner.

        Interpolated where that is well defined and projected where it is not,
        which is two cases. A quadrature source has values only where the
        integration rule put them. A discontinuous source going into a continuous
        target has two values at every shared vertex and no reason to prefer
        either: interpolation would take whichever cell it visited last, which
        draws the mesh's cell ordering as streaks across the field.

        Projection is not a clamp: the L2 fit of a steep field onto P1 overshoots
        at the step, so a quadrature indicator can come back outside the range it
        had, and a residual that belongs to one cell is spread over the vertices
        around it. A caller who needs a bounded field clips it and knows it did.
        """

        source = self.function(frame)
        if space is None or space is source.function_space:
            return source
        target = fem.Function(space, name=self.name)
        into = space.ufl_element()
        if self.is_quadrature or (
            self.element.discontinuous and not into.discontinuous
        ):
            project(source, target, self.run.quadrature_degree)
        elif self.element.symmetry and not into.is_symmetric:
            # A symmetric element stores the upper triangle and ``interpolate``
            # copies dof for dof, so a 2x2's three numbers would land in the first
            # three of the target's four, as ``[[m00, m01], [m11, 0]]``. An
            # expression asks UFL for the value, and UFL applies the component map.
            target.interpolate(interpolation_expression(source, space))
        else:
            target.interpolate(source)
        target.x.scatter_forward()
        return target


class Run:
    """A saved output directory, opened for reading.

    Frames are memory-mapped rather than read, because a long run's fields are
    larger than memory and an analysis usually wants a few frames of many fields
    rather than all frames of one. The mesh is read once, on first use, and
    shared by every space built from it -- which is what makes two fields
    comparable: they are dofs on the same cells.
    """

    def __init__(self, path: str | Path, parameters: "Any | None" = None) -> None:
        self.path = Path(path).expanduser().resolve()
        meta = self.path / META_FILE
        if not meta.is_file():
            raise FileNotFoundError(f"no saved run here: {meta}")
        self.metadata = json.loads(meta.read_text(encoding="utf-8"))
        #: The mixture's parameters, when the caller had them: everything about
        #: the run that the metadata does not record is read from here.
        self.parameters = parameters
        self.n_frames = int(self.metadata["frames_written"])
        self._spaces: dict[tuple, "fem.FunctionSpace"] = {}

    @classmethod
    def of(cls, parameters: Any) -> "Run":
        """The run a parsed spec describes the output of."""

        return cls(parameters.solver.save.directory, parameters)

    @property
    def solver_parameters(self) -> "SolverParameters | None":
        """The run's solver parameters, posed on the mesh *this* reader holds.

        Priming that cache is the point of the property. A diagnostic assembling
        one of the solver's own forms -- the two-point stiffness, the gradient
        reconstruction -- indexes the result by cell against arrays read out of
        the frames, and the parameters would otherwise read the mesh file again
        and build the form on a second mesh object. The two readings agree, but
        nothing would say so.
        """

        if self.parameters is None:
            return None
        solver = self.parameters.solver
        solver.__dict__.setdefault("dolfinx_mesh", self.mesh)
        return solver

    @property
    def quadrature_degree(self) -> int:
        """The rule a projection out of a quadrature field integrates on.

        The run's own, so that a quadrature function is integrated at exactly
        the points it has values at rather than resampled onto a different rule
        on the way to being fitted.
        """

        if self.parameters is None:
            return DEFAULT_QUADRATURE_DEGREE
        return int(self.parameters.solver.diffuse_domain.indicator_quadrature_degree)

    ## Spaces

    def space(self, element: ElementSpec) -> "fem.FunctionSpace":
        """The space an element spec names, built once and shared.

        Keyed on the element's identity rather than the whole spec, and filed
        under both the identity asked for and the one basix reports for what it
        built, because those differ: ``Lagrange`` with no variant and ``P``,
        ``gll_warped`` name one element, a saved field says the second and a
        caller reaching for the P1 view says the first. Keyed on one alone, one
        element would get two spaces, and every comparison between them would go
        through an interpolation that is arithmetically the identity and
        numerically an expense.
        """

        if element.identity in self._spaces:
            return self._spaces[element.identity]
        built = space_of(self.mesh, element)
        canonical = describe(built, name=element.name).identity
        self._spaces[element.identity] = self._spaces.setdefault(canonical, built)
        return self._spaces[element.identity]

    def vertex_space(self, shape: tuple[int, ...] = ()) -> "fem.FunctionSpace":
        """The common P1 space of a given value shape.

        Where a run's continuous fields meet. A cell-constant field can be brought
        here, and a picture sometimes wants it -- a smooth composite, a contour --
        but it costs a projection and the steps it smooths are the physics, so
        :meth:`cell_space` is the space to reach for by default. A field already
        saved as P1 of this shape resolves to this very object.
        """

        return self.space(
            ElementSpec(
                name="vertex", family="Lagrange", degree=1, shape=tuple(shape)
            )
        )

    def cell_space(self, shape: tuple[int, ...] = ()) -> "fem.FunctionSpace":
        """The common DG0 space of a given value shape.

        Where nearly everything a run holds already lives -- every phase, every
        chemical state, the pressure and every quantity derived from them -- so
        they are dofs of one space on one mesh and their arrays line up entry for
        entry. The space to compare and composite in.
        """

        return self.space(
            ElementSpec(
                name="cells",
                family=DISCONTINUOUS_LAGRANGE,
                degree=0,
                shape=tuple(shape),
                discontinuous=True,
            )
        )

    @cached_property
    def mesh(self) -> "Mesh":
        """The run's mesh, as dolfinx wrote it.

        Read back rather than rebuilt from the spec, so that a saved dof vector
        indexes it exactly: the ordering of a mesh generated twice from the same
        description is not guaranteed, but the ordering of one written and read
        is.
        """

        with XDMFFile(MPI.COMM_SELF, self._mesh_file(), "r") as file:
            return file.read_mesh()

    def _mesh_file(self) -> Path:
        """Which XDMF in here is the mesh.

        The archive the run wrote, under its fixed name. Failing that, the only
        XDMF in the directory, so a directory whose mesh goes by another name
        still reads, and one holding several candidates is refused rather than
        guessed at.
        """

        archived = self.path / MESH_FILE
        if archived.is_file():
            return archived
        candidates = sorted(self.path.glob("*.xdmf"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one mesh in {self.path}, found {len(candidates)}"
            )
        return candidates[0]

    ## Frames

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.metadata["fields"]))

    @property
    def times(self) -> np.ndarray:
        return self._load(TIMES_FILE)[: self.n_frames]

    @property
    def steps(self) -> np.ndarray:
        return self._load(STEPS_FILE)[: self.n_frames]

    def has(self, name: str) -> bool:
        """Whether this run can hand back ``name``, which is not quite whether
        it meant to.

        The manifest is written when the run starts and lists what it intends to
        write, so an entry in it is a promise rather than a fact: the file can be
        absent afterwards -- a run killed before its first frame, a directory
        moved by something that balked at a very large file. The disk is the
        authority, so a reader fails here rather than deep inside a diagnostic.
        """

        entry = self.metadata["fields"].get(name)
        return entry is not None and (self.path / entry["filename"]).exists()

    def field(self, name: str) -> Field:
        try:
            entry = self.metadata["fields"][name]
        except KeyError:
            raise KeyError(
                f"{name!r} was not saved by this run; it has {', '.join(self.names)}"
            ) from None
        if not (self.path / entry["filename"]).exists():
            raise FileNotFoundError(
                f"{name!r} is named in {self.path / META_FILE} but "
                f"{entry['filename']} is not in {self.path}"
            )
        static = bool(entry.get("static", False))
        values = self._load(entry["filename"])
        return Field(
            name=name,
            run=self,
            # Truncating to the frames written is what separates a short run's
            # values from the zeros it preallocated. A static field wrote its one
            # row at frame zero and there is nothing to trim.
            values=values if static else values[: self.n_frames],
            element=ElementSpec.from_dict(entry["element"], name=name),
            mode=entry["mode"],
            static=static,
        )

    def fields(self) -> dict[str, Field]:
        return {name: self.field(name) for name in self.names}

    def _load(self, filename: str) -> np.ndarray:
        return np.load(self.path / filename, mmap_mode="r")


def project(
    source: "fem.Function", target: "fem.Function", quadrature_degree: int
) -> "fem.Function":
    """Put ``source`` onto ``target``'s space in the L2 sense.

    For the elements that cannot be interpolated from. The mass matrix is the
    target space's, and the right-hand side integrates the source against the
    target's test functions at the degree the run itself used, so that a
    quadrature function is integrated on exactly the points it has values at.
    """

    space = target.function_space
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure(
        "dx",
        domain=space.mesh,
        metadata={"quadrature_degree": quadrature_degree},
    )
    inner = ufl.inner if source.ufl_shape else (lambda a, b: a * b)
    problem = fem_petsc.LinearProblem(
        inner(trial, test) * dx,
        inner(source, test) * dx,
        u=target,
        petsc_options_prefix="ovpsolver_read_project_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    problem.solve()
    return target


__all__ = ["DEFAULT_QUADRATURE_DEGREE", "Field", "Run", "project"]
