"""Declaring finite elements: what an element is, and which solve determines it.

An owner (:class:`~ovpsolver.fem.elements.ElementOwner`) declares the finite
elements it owns as a flat list of :class:`ElementSpec`. The solver builds the
function spaces and functions once, binds them onto the owner's
``rates``/``prev``/``next`` bundles by object reference, and assembles the test
spaces of the two solves:

* the nonlinear *mechanical* (Onsager rate / stationarity) solve, whose unknowns
  are the quasistatic rates plus the subset of the ``k+1`` state that enters the
  convex energy or the elliptic coupling, and
* independent linear *state* solves for the remaining ``k+1`` fields.

``StaticElement`` is a quasistatic rate or multiplier: one function, always a
mechanical unknown. ``StateElement`` is a time-evolved field: a
``prev``/``next`` pair, assigned to whichever solve determines its ``next``
value by the :class:`Solve` on its spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from basix import LagrangeVariant
from basix.ufl import element, quadrature_element

from .dg0 import DISCONTINUOUS_LAGRANGE

if TYPE_CHECKING:
    from dolfinx.fem import Function
    from ufl.core.expr import Expr

    from .owner import ElementOwner


class Solve(Enum):
    """Whether the free energy sees a field's ``k+1`` value, and so which solve
    determines it.

    The two questions have one answer. A field the energy is evaluated at ``k+1``
    on must be solved for simultaneously with the rates, or the Rayleighian would
    be stationary for an energy nothing delivers; a field the energy only sees
    lagged can be swept up afterwards, cell by cell and with no matrix.

    Declaring a lagging field live is silent and expensive, out of proportion to
    the count for a field with components: a blocked element's sparsity in the
    assembled rate matrix couples every component of a cell to every other and to
    every velocity dof its flux reaches -- ``n^2`` entries a cell for ``n``
    components, whatever the form actually couples. So the energy should read
    scalar summaries carried alongside such a field, as ``u_bar`` and ``u_trace``
    summarize the generating function, and leave the field itself lagging.
    """

    NONE = auto()
    """Coefficient or diagnostic only; no solve determines it."""

    ENERGY_LIVE = auto()
    """The energy is evaluated at ``k+1``; unknown of the nonlinear rate solve."""

    ENERGY_LAGGING = auto()
    """The energy only sees it at ``k``; solved cell by cell after the rates."""


class ElementDomain(Enum):
    """Where on the mesh something lives: an unknown, or an energy density.

    Shared with :class:`~ovpsolver.phase_field_system.energy.EnergyDensity`
    because the measure an energy is integrated against is the one the unknowns
    supported there are integrated against.
    """

    BULK = auto()
    """The mesh interior; weighted by ``chi_eps`` where the free fluid is."""

    SURFACE = auto()
    """The outer boundary ``Sigma``."""

    DIFFUSE_SURFACE = auto()
    """The diffuse inclusion surfaces; weighted by ``dgamma_eps``."""


#: The family name basix reports for a quadrature element, and the one it takes.
#: Compared case-insensitively, because a declaration writes it the way it reads
#: and basix reports it lower case.
QUADRATURE = "quadrature"

#: The attributes that say what the element *is*, as opposed to what it is for.
#: Two specs naming the same element agree on all of these and may disagree on
#: everything else: a saved diagnostic and the state it was computed from can be
#: dofs of one space while belonging to different owners.
IDENTITY = (
    "family",
    "degree",
    "shape",
    "symmetry",
    "discontinuous",
    "variant",
    "quadrature_scheme",
    "domain",
)


@dataclass(frozen=True)
class ElementSpec:
    """A single finite element, and everything needed to reach its functions.

    ``name`` is the attribute bound on the owner's field bundles; ``test_attr``
    the attribute on the owner itself that receives the (shared mixed) test
    function. ``family``/``degree``/``shape``/``symmetry``/``discontinuous``
    /``variant`` are forwarded to :func:`basix.ufl.element`, and ``domain``
    selects the bulk mesh, the outer-boundary submesh or the diffuse inclusion
    surfaces.

    ``owner`` is stamped on by
    :meth:`~ovpsolver.fem.elements.ElementOwner.element_specs` rather than
    passed by the declaration, which is written inside the owner. With it, a spec
    is enough to reach the functions, the test function and the mesh. Without it,
    a spec still says exactly which element it is, which is what a saved run is
    described by (:meth:`of`, :meth:`to_dict`, :mod:`ovpsolver.fem.save.element`):
    one vocabulary for the solve and the archive rather than two kept in step.
    """

    name: str
    test_attr: str = ""
    family: str = "Lagrange"
    degree: int = 1
    shape: tuple[int, ...] = ()
    symmetry: bool = False
    #: How basix reports discontinuity: the family ``P`` and this flag. A
    #: declaration may say it with the family name (``"Discontinuous Lagrange"``)
    #: instead, which wins either way, and leave this ``False`` -- so
    #: :attr:`is_cellwise` and :meth:`describe` are only reliable on a spec read
    #: off a built space (:meth:`of`, :meth:`from_dict`).
    discontinuous: bool = False
    #: ``None`` means basix's default, which is what a declaration wants and what
    #: :meth:`of` reads back as the name of whatever that turned out to be.
    variant: str | None = None
    quadrature_scheme: str | None = None
    #: Whether the variable cannot go negative: entry by entry for a scalar or a
    #: vector, and as a semidefinite matrix for a square tensor. A property of
    #: what the variable is rather than of a transport row, so it is declared
    #: here and the solver's floor check reads it off every element, transported
    #: or not. ``False`` by default because the claim is not free: it puts the
    #: variable under the step bound and under the check that stops the run.
    positive: bool = False
    solve: Solve = Solve.ENERGY_LIVE
    domain: ElementDomain = ElementDomain.BULK
    # Out of comparison and repr: two specs are the same declaration if they say
    # the same thing, and an owner's repr is enormous.
    owner: "ElementOwner | None" = field(default=None, compare=False, repr=False)

    @property
    def label(self) -> str:
        """``<owner>.<name>``, which is what a log or an error should show."""

        return f"{self.owner.name()}.{self.name}" if self.owner else self.name

    def function(self, time_level: str = "next") -> "Function":
        """The function this spec's values live in at one time level.

        ``rates`` for a static element, which has only the one.
        """

        if isinstance(self, StaticElement):
            return getattr(self.owner.rates, self.name)
        return getattr(self.owner.fields_at(time_level), self.name)

    @property
    def test(self) -> "Expr":
        """The test function the solver bound for this element's row."""

        return getattr(self.owner, self.test_attr)

    ## What element this is

    @property
    def is_quadrature(self) -> bool:
        return self.family.lower() == QUADRATURE

    @property
    def is_cellwise(self) -> bool:
        """One value per cell: constant, and not shared with the neighbours."""

        return self.degree == 0 and self.discontinuous

    @property
    def identity(self) -> tuple:
        """What element this is, hashable, for looking a built space up by.

        Two specs with the same identity name the same element, so one built
        space serves both and their dofs are comparable entry by entry.
        """

        return tuple(getattr(self, attribute) for attribute in IDENTITY)

    def make_element(self, cell):
        """Build the basix UFL element for this spec on the given basix cell."""

        if self.is_quadrature:
            return quadrature_element(
                cell,
                value_shape=self.shape,
                scheme=self.quadrature_scheme,
                degree=self.degree,
            )

        kwargs: dict = {}
        if self.shape:
            kwargs["shape"] = self.shape
        if self.symmetry:
            kwargs["symmetry"] = True
        if self.discontinuous:
            kwargs["discontinuous"] = True
        if self.variant is not None:
            kwargs["lagrange_variant"] = LagrangeVariant[self.variant]
        return element(self.family, cell, self.degree, **kwargs)

    @classmethod
    def of(cls, space, **attributes) -> "ElementSpec":
        """The spec for the element ``space`` was built from.

        The inverse of :meth:`make_element`, so that a space nobody declared --
        the one a saved diagnostic gets materialized into -- can be described in
        the same terms as one that was.
        """

        built = space.ufl_element()
        variant = getattr(built, "lagrange_variant", None)
        return cls(
            family=str(built.family_name),
            degree=int(built.degree),
            shape=tuple(int(dim) for dim in built.reference_value_shape),
            symmetry=bool(built.is_symmetric),
            discontinuous=bool(built.discontinuous),
            variant=None if variant is None else variant.name,
            # Not reported by basix, and the only scheme anything here builds.
            quadrature_scheme=(
                "default" if str(built.family_name).lower() == QUADRATURE else None
            ),
            **{"name": "", "solve": Solve.NONE, **attributes},
        )

    ## As plain data

    def to_dict(self) -> dict:
        """What element this is, as JSON, for a saved run to be read back by.

        The identity only: a name, a test attribute, an owner and a solve are how
        the run reached this element, and a reader has the run's own metadata for
        those.
        """

        return {
            "family": self.family,
            "degree": self.degree,
            "shape": list(self.shape),
            "symmetry": self.symmetry,
            "discontinuous": self.discontinuous,
            "variant": self.variant,
            "quadrature_scheme": self.quadrature_scheme,
            "domain": self.domain.name,
        }

    @classmethod
    def from_dict(cls, description: dict, **attributes) -> "ElementSpec":
        """Rebuild what :meth:`to_dict` wrote."""

        return cls(
            family=description["family"],
            degree=int(description["degree"]),
            shape=tuple(int(dim) for dim in description["shape"]),
            symmetry=bool(description["symmetry"]),
            discontinuous=bool(description["discontinuous"]),
            variant=description["variant"],
            quadrature_scheme=description["quadrature_scheme"],
            domain=ElementDomain[description["domain"]],
            **{"name": "", "solve": Solve.NONE, **attributes},
        )

    def describe(self) -> str:
        """This element in the notation a person writes it in, for a log."""

        if self.is_quadrature:
            stem = f"Q{self.degree}"
        else:
            stem = f"{'DG' if self.discontinuous else 'P'}{self.degree}"
        if not self.shape:
            return stem
        return f"{stem}{tuple(self.shape)}{'sym' if self.symmetry else ''}"


@dataclass(frozen=True)
class StaticElement(ElementSpec):
    """Quasistatic rate / Lagrange-multiplier unknown.

    One function, stored on the owner's ``rates`` bundle; always a mechanical
    (nonlinear-solve) unknown. Set ``store_lagged`` to also keep the last solved
    rate as a coefficient on the owner's ``prev`` bundle, which is what an upwind
    selector reads.

    Never given a Dirichlet condition. Every constraint the mixture imposes --
    saturation, the imposed normal flux at ``Sigma``, the pressure gauge -- is
    weak, carried by a declared multiplier or regularization, so strong pinning
    would be a second, silent statement of something the Rayleighian already says.
    """

    store_lagged: bool = False


@dataclass(frozen=True)
class VelocityElement(StaticElement):
    """A rate that carries a phase, and the multipliers constraining it.

    Every velocity moving a phase is constrained the same two ways: it may not
    carry the phase across a diffuse inclusion surface except by the injection
    the flux declares, and its normal flux through ``Sigma`` is pinned to an
    imposed value. So a velocity names its multipliers here and the phase field
    builds them, rather than each concrete field declaring them and being able to
    forget one.

    Named on the velocity rather than derived from its flux because a flux is an
    expression in bound functions, which do not exist while declarations are
    being collected. That the fluxes and the velocities agree is checked once
    they do; see
    :meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.constraints`.

    Lagged by default, unlike a bare rate: a velocity is what an upwind selector
    reads, and a selector at the live velocity would cost the rate residual its
    affineness in the rate.
    """

    store_lagged: bool = True
    #: Pins the normal flux through ``Sigma``. Always present: a velocity that
    #: imposes nothing is stating non-crossing, which is still a constraint.
    boundary_multiplier: str = "Lambda"
    #: Relaxes what crosses the diffuse inclusion surfaces, cell by cell: the
    #: multiplier ``lambda`` of the crossing constraint. The trailing underscore
    #: dodges the Python keyword and is not part of the name.
    crossing_multiplier: str = "lambda_"

    def multipliers(self, *, has_inclusions: bool = True) -> list[StaticElement]:
        """The multiplier elements this velocity's constraints are paired with.

        One per constraint, each in the space it is supported on: the flux through
        ``Sigma`` on the codimension-one submesh, always, and the crossing of the
        diffuse surfaces over the bulk cells. The crossing multiplier is dropped
        where there are no inclusions: its residual ``phi v.grad(phi_incl) + nu
        q_eps`` is then identically zero, and an unknown per cell with an empty
        row would make the rate solve singular.
        """

        specs = []
        # P1 on the boundary submesh, which pins less than it looks. The row is
        # ``int_Sigma Lambda (phi v.n + S) = 0`` for every P1 ``Lambda``, so what
        # vanishes is the P1 moments of the boundary flux -- not the flux, and not
        # its integral over any one facet, whose indicator is cell-constant on
        # Sigma and so not P1. Constants are P1, so the net flux through Sigma,
        # which conserves the total, is pinned exactly. The P1 moments of the
        # boundary flux vanish to roundoff while its per-facet integrals do not.
        #
        # A row reading this boundary flux in another space therefore reads a
        # residual nobody pinned; the cell-constant pressure reads no flux through
        # ``Sigma`` for that reason (see
        # :meth:`~ovpsolver.phase_field_system.PhaseFieldSystem.compression_penalty`).
        # A cell-constant multiplier here closes the gap (the per-facet integrals
        # then vanish to roundoff too) but doubles the rate solve on a mixture
        # with a sol and a gel velocity, for exactness in a quantity nothing
        # downstream reads.
        specs.append(
            StaticElement(
                self.boundary_multiplier,
                test_attribute(self.boundary_multiplier),
                degree=1,
                domain=ElementDomain.SURFACE,
            )
        )
        # Cell-constant, and on the bulk mesh because the band it acts in is bulk
        # cells rather than a mesh entity. One value per cell pins the crossing
        # integrated over that cell, which is exactly what a cell-constant
        # transport row conserves. Off the band -- where ``grad(phi_incl)`` has
        # decayed without vanishing, and inside an absent phase -- the mobility it
        # is relaxed by keeps it determined, not its space; see
        # :meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.surface_crossing_penalty`.
        if has_inclusions:
            specs.append(
                StaticElement(
                    self.crossing_multiplier,
                    test_attribute(self.crossing_multiplier),
                    family=DISCONTINUOUS_LAGRANGE,
                    degree=0,
                    discontinuous=True,
                    domain=ElementDomain.BULK,
                )
            )
        return specs


def test_attribute(name: str) -> str:
    """The owner attribute a generated element's test function is bound to.

    One rule, so that a declaration nobody wrote by hand still has a predictable
    place to be read from. The trailing underscore of a name dodging a Python
    keyword is not part of the name.
    """

    return f"_{name.rstrip('_')}_test"


@dataclass(frozen=True)
class StateElement(ElementSpec):
    """Time-evolved field: a ``prev``/``next`` function pair.

    ``solve`` says whether the energy sees the ``next`` value, and so which solve
    owns it. Lagging by default: a state has to earn its place in the nonlinear
    solve by appearing in the energy.
    """

    solve: Solve = Solve.ENERGY_LAGGING
