"""The spec's blocks for a mixture: ``system``, its ``couplings``, and the root
that joins ``system`` to ``solver``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..fem.saveable import SaveableParameters
from ..parametric import (
    Index,
    Nonnegative,
    Parameters,
    ParametersList,
    Positive,
)
from ..solver.parameters import SolverParameters
from .phase_field import PhaseFieldParameters
from .visualize import layers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..diffuse_domain.parameters import DiffuseDomainParameters
    from .visualize.layers import Layer


#: What the mixture reads off every entry of its roster, whatever kind of phase
#: that entry describes. Checked by name rather than by requiring a common base,
#: so that a phase somebody else wrote satisfies it without inheriting from
#: anything in particular.
PHASE_INTERFACE = (
    "name",
    "initial_volume_fraction",
    "monomer_volume",
    "surface_flux_density",
    "save",
)


class CouplingsParameters(Parameters):
    """Every cross-phase interaction the mixture has, one declaration each.

    The ``couplings`` block of a spec. Each declaration here is a
    :class:`~ovpsolver.parametric.Parameters` class that a
    :class:`~ovpsolver.phase_field_system.couplings.Coupling` is ``Parametric``
    in, and declaring one is half of adding a coupling to a mixture::

        class ModelBCouplings(CouplingsParameters):
            flory_huggins = CHFloryHugginsParameters()

    The other half is the mixture's
    :meth:`~ovpsolver.phase_field_system.system.PhaseFieldSystem.make_couplings`,
    which constructs each one from its block, as ``make_phase_fields`` does a
    roster.

    Empty here, because a mixture with no cross interactions is a real mixture --
    a single phase filling the domain has nothing to couple to.
    """

    def populate(self, phases: "Sequence[PhaseFieldParameters]") -> None:
        """Let each coupling resolve itself against the roster.

        A coupling indexed by pairs cannot check itself: whether
        ``(water, protein)`` names two phases of this mixture is a question only
        the roster answers. So the mixture calls this once it has read its
        phases, while the coefficients are in hand and a message can name the
        pair. A coupling with nothing to resolve is skipped.
        """

        for field_name in self.declarations():
            resolve = getattr(getattr(self, field_name), "populate", None)
            if resolve is not None:
                resolve(phases)


class PhaseFieldSystemParameters(SaveableParameters):
    """The mixture: its phases, and what no phase owns.

    The ``system`` block of a spec: the material and nothing else. What steps it
    is the sibling ``solver`` block, and the two are joined by
    :class:`SpecParameters`, the root of the document. A ``system`` says what is
    being simulated and is the same whatever scheme resolves it; a ``solver``
    says how, and is the half that gets tuned. Keeping them apart also lets this
    block be read against whichever mixture class is being run while the solver
    beside it is the same class every time. The solver is still reachable as
    :attr:`solver`, because nearly every declaration is written against the mesh
    and its measures.

    Almost everything about a phase belongs to the phase. The thermal scale and
    the randomness of the initial condition are here because no phase owns them.

    ``phase_fields`` is the mixture's roster, a list of one kind of phase. A
    concrete system that distinguishes kinds narrows it to a block of typed lists
    and joins them in :meth:`phase_field_parameters`, which keeps the distinction
    out of everything downstream: the saturation constraint, the flux balance and
    the initial condition solve are the same whether a phase polymerizes or not.

    Parameters
    ----------
    couplings : the cross-phase interactions, one sub-block each. The line
        between a coupling and a phase is drawn by whose variables the term is
        written in, not by how many phases it mentions.
    k_B_T : Boltzmann constant times temperature; nondimensionalised to 1 by
        default, and the scale every declared energy is written in.
    dilatational_viscosity : the viscosity ``eta_b`` of compressible motion.
        Saturation holds up to a compression rate ``Phi p / eta_b``, so large
        is nearly incompressible; finite, because the same mobility keeps the
        pressure determined where there is no free fluid. A statement about how
        compressible the mixture is (it enters
        :meth:`~ovpsolver.phase_field_system.system.PhaseFieldSystem.compression_penalty`),
        so it belongs to the material rather than to the solver.
    initial_phase_noise_sigma : the standard deviation of the noise added to
        every phase once the initial condition is solved, which is how a run on
        the unstable side of the spinodal seeds its decomposition rather than
        waiting for the discretization to supply one. Zero leaves the solved
        state alone.
    rng_seed : seeds every random draw the mixture makes -- the initial noise,
        and in a subclass possibly more. With the material rather than the
        solver, because reproducing a run means reproducing its initial
        condition.
    save : the mixture-level fields to write out, by name: any the system
        offers in its ``declare_saveable``.
    """

    phase_fields: tuple[PhaseFieldParameters, ...] = ParametersList(
        PhaseFieldParameters()
    )
    couplings: CouplingsParameters = CouplingsParameters()
    k_B_T: float = Positive(1.0)
    dilatational_viscosity: float = Positive(1.0e6)
    initial_phase_noise_sigma: float = Nonnegative(0.0, coefficient=False)
    rng_seed: int = Index(0)

    @property
    def solver(self) -> SolverParameters:
        """The sibling ``solver`` block, through the spec that holds both."""

        root = self.parent
        if root is None:
            raise AttributeError(
                f"{self.where} has no solver: a system block reaches one through "
                f"the {SpecParameters.__name__} it was read as part of, and this "
                f"one was built on its own"
            )
        return root.solver

    def derive(self) -> None:
        #: The one generator the mixture draws from, so that a spec plus a seed
        #: is the whole of what a run needs to be reproduced.
        self.rng = np.random.default_rng(self.rng_seed)

        # Checks, but here rather than in validate, because everything below
        # reads the roster and indexes it by name: a phase missing part of the
        # interface would surface as an AttributeError from whichever check asked
        # first, and a duplicate name would quietly cost a phase its entry in the
        # flux table. Nothing here reads the solver, which the spec has not yet
        # adopted this block into; the checks spanning both blocks are the spec's.
        self.check_phases_are_usable()
        self.check_names_are_unique()
        # After both: a coupling is indexed by phase name, and resolving one
        # against a roster with a duplicate would file a coefficient under the
        # wrong phase.
        self.couplings.populate(self.phase_field_parameters)

    def validate(self) -> None:
        self.check_initial_volume_fractions()
        self.check_boundary_flux_balance()

    @property
    def phase_field_parameters(self) -> tuple[PhaseFieldParameters, ...]:
        """One entry per phase, in the order the system will build them."""

        return tuple(self.phase_fields)

    ## What a picture of this mixture is made of

    def phase_layers(self, *, group_polymerizing: bool = False) -> tuple["Layer", ...]:
        """The coloured layers a rendered frame composites, in draw order.

        Here rather than in the plotting code because the answer depends on what
        kinds of phase the mixture has, which is what a subclass adds. The
        general answer is one layer per phase, its volume fraction in the colour
        the phase was given.

        Parameters
        ----------
        group_polymerizing : whether a phase that resolves into components
            should be drawn as one layer rather than as its parts. Ignored here,
            where no phase has parts.
        """

        return tuple(
            layers.phase_layer(parameters)
            for parameters in self.phase_field_parameters
        )

    def inclusion_layer(self) -> "Layer | None":
        """The layer that draws the fixed inclusions, or ``None`` if there are none.

        The indicator is analytic and the spec that fixes it travels with the
        run, so unlike a solved field this is drawn whether or not anything was
        saved. Asked of the mesh, because a geometry file may return an empty
        list and only building its distances says so; the mesh is the run's own,
        which every caller here has already read.
        """

        geometry = self.solver.diffuse_domain
        if not geometry.has_inclusions(self.solver.dolfinx_mesh):
            return None
        return layers.inclusion_layer(geometry)

    @property
    def phase_field_names(self) -> tuple[str, ...]:
        return tuple(parameters.name for parameters in self.phase_field_parameters)

    @property
    def monomer_volumes(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                float(parameters.monomer_volume)
                for parameters in self.phase_field_parameters
            ],
            dtype=float,
        )

    def publish_surface_flux_densities(self, domain: "DiffuseDomainParameters") -> None:
        """Give the domain the injection each phase declared for itself.

        The phases own these numbers, since what is injected is a property of the
        material rather than of the geometry it is injected through; the domain
        holds them because it materializes ``q_eps`` and needs them all at once.
        Handed the domain by the spec, the one place both blocks are in hand.
        """

        domain.surface_flux_densities = {
            parameters.name: float(parameters.surface_flux_density)
            for parameters in self.phase_field_parameters
        }

    def check_phases_are_usable(self) -> None:
        """The roster is non-empty, and every entry answers what will be asked.

        A phase field somebody else wrote is read against their own parameters
        class, so nothing else checks :data:`PHASE_INTERFACE`. One message naming
        the phase and what is missing, rather than an ``AttributeError`` from
        whichever check happened to ask first.
        """

        roster = self.phase_field_parameters
        if not roster:
            raise ValueError(f"{self.where} needs at least one phase")
        for position, parameters in enumerate(roster):
            missing = [
                attribute
                for attribute in PHASE_INTERFACE
                if not hasattr(parameters, attribute)
            ]
            if missing:
                raise ValueError(
                    f"{self.where}: phase {position} is a "
                    f"{type(parameters).__name__}, which does not declare "
                    f"{', '.join(missing)}; every phase in a mixture has to, "
                    f"whatever else it is made of"
                )

    def check_names_are_unique(self) -> None:
        """The name is the key the diffuse domain files a phase's source under."""

        names = self.phase_field_names
        if len(set(names)) != len(names):
            raise ValueError(f"{self.where}: phase names must be unique, got {names}")

    def check_initial_volume_fractions(self) -> None:
        """The initial state has to lie on the simplex the pressure enforces."""

        total = float(
            sum(
                float(parameters.initial_volume_fraction)
                for parameters in self.phase_field_parameters
            )
        )
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"{self.where}: initial_volume_fraction values must sum to 1, "
                f"since the mixture is saturated; got {total!r}"
            )

    def check_surface_flux_balance(self, domain: "DiffuseDomainParameters") -> None:
        """``sum_i nu_i q_ia = 0``: what enters at a surface displaces something.

        The compatibility condition of the saturation constraint, checked on the
        numbers while they are in hand; :meth:`check_boundary_flux_balance` is
        its counterpart at ``Sigma``.
        """

        fluxes = np.array(
            [domain.surface_flux_densities[name] for name in self.phase_field_names],
            dtype=float,
        )
        balance = float(fluxes @ self.monomer_volumes)
        if not np.isclose(balance, 0.0):
            raise ValueError(
                f"{self.where}: surface fluxes must satisfy sum_i nu_i q_i = 0 "
                f"through the surfaces; got {balance!r}"
            )


    def check_boundary_flux_balance(self) -> None:
        """``sum_i nu_i Q_i = 0`` on ``Sigma``, which bulk saturation needs of
        the outer boundary, checked on the numbers while they are in hand."""

        fluxes = np.array(
            [
                float(parameters.boundary_flux_density)
                for parameters in self.phase_field_parameters
            ],
            dtype=float,
        )
        balance = float(fluxes @ self.monomer_volumes)
        if not np.isclose(balance, 0.0):
            raise ValueError(
                f"{self.where}: boundary fluxes must satisfy sum_i nu_i Q_i = 0; "
                f"got {balance!r}"
            )


class SpecParameters(Parameters):
    """A whole document: the material, and the machine that steps it.

    The root of every spec. A run is two statements -- ``system:`` says what is
    being simulated and ``solver:`` says how -- and which mixture class reads the
    first is not something the file says: the word on the command line
    (``python -m <mixture> run``) picks the class, so the same two words head
    every document.

    The two blocks are joined here and nowhere else. A phase declares what it is
    injected with at the inclusion surfaces and the geometry materializes that
    injection, so the numbers cross from one block to the other here, the only
    place both are in hand: each block is settled before the spec adopts it, and
    so cannot reach its sibling while it is being read.

    Parameters
    ----------
    solver : the mesh the mixture lives on, how the driver steps it, and the
        regularizations that keep its rows conditioned. Under it the inclusion
        geometry, because the geometry is what the mesh is built from.
    system : the mixture -- its phases, what they are made of, and what couples
        them.
    """

    solver: SolverParameters = SolverParameters()
    system: PhaseFieldSystemParameters = PhaseFieldSystemParameters()

    def derive(self) -> None:
        self.system.publish_surface_flux_densities(self.solver.diffuse_domain)
        # One seed for the whole spec: a geometry that draws its arrangement
        # varies with the same number the phase noise does.
        self.solver.diffuse_domain.rng_seed = int(self.system.rng_seed)

    def validate(self) -> None:
        self.system.check_surface_flux_balance(self.solver.diffuse_domain)

    def resolve_against(self, source: "Path") -> None:
        """Locate everything the document names by path, relative to the document.

        So that a spec and the files it points at travel as one, and a run
        archived beside its own copies reads back without editing.
        """

        self.solver.save.resolve_against(source)
        self.solver.diffuse_domain.resolve_against(source)
