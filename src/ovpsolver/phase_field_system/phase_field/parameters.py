from __future__ import annotations

from ...fem.saveable import SaveableParameters
from ...parametric import (
    Nonnegative,
    Number,
    Positive,
    Text,
)
from ..visualize.layers import CHI_MODES, MASK, OCCUPY


class PhaseFieldParameters(SaveableParameters):
    """The parameters every phase field has, whatever it is made of.

    Only the quantities :class:`~ovpsolver.phase_field_system.phase_field.PhaseField`
    itself reads: the gradient penalty that defines its chemical potential, the
    monomer volume that converts a number flux into a volume fraction flux, the
    affinity for the inclusion surfaces that enters that potential as a natural
    boundary condition, and how much of the phase there is to begin with.

    Everything else material -- which velocity carries what, the drags between
    them, the chemistry, and what each velocity costs in the bulk (its
    diffusivity and screening length) -- belongs to the concrete field that
    declares the velocities and internal states: a Cahn-Hilliard species declares
    one velocity, and a crosslinking one a sol and a gel pair.

    The two regularizations a velocity row needs -- the floor under its drag
    measure and the continuation of ``ln phi`` -- are under ``solver``, because
    they bound the conditioning of a row rather than describe a species, and a
    run that states them once cannot give two phases incompatible ones.

    Parameters
    ----------
    surface_affinity : the wetting energy per unit area of diffuse inclusion
        surface, linear in the phase and in units of ``k_B_T``. Positive wets, and
        a mixture whose affinities differ is one whose phases compete for the
        surfaces; what a value means as a contact angle is
        ``analyze.wetting``'s business.
    save : the fields to write out for this phase, by name: any the phase offers
        in its ``declare_saveable``, which is every element it declares and the
        diagnostics it derives.
    """

    name: str = Text()
    kappa: float = Positive(1.0e-4)
    monomer_volume: float = Positive(1.0)
    initial_volume_fraction: float = Nonnegative(1.0)
    surface_affinity: float = Number(0.0)
    color: str = Text("blue")

    @property
    def surface_flux_density(self) -> float:
        """What this phase is injected with at the inclusion surfaces, per area.

        A property rather than a declaration because the phase that has one
        names it for the velocity that carries it -- a sol phase has one
        velocity and a polymerizing phase has two -- while the mixture only needs
        the number, to check that what enters displaces something.

        A ``_density``: the spec's own number per unit area, where the same words
        without the suffix mean it already smeared into a flux per unit volume
        (:meth:`~ovpsolver.phase_field_system.phase_field.PhaseField.surface_flux`).
        """

        return 0.0

    @property
    def boundary_flux_density(self) -> float:
        """What this phase is injected with at the outer boundary, per area.

        A property for the same reason :attr:`surface_flux_density` is.
        ``Sigma`` is a real surface of the mesh, so nothing smears this: a flux
        piece's ``boundary_flux`` is this number as it is, scaled by what each
        monomer carries, still per unit boundary area.
        """

        return 0.0

    ## What a picture of this phase means

    #: How each of this phase's saved diagnostics relates to the inclusions, for
    #: drawing it. Everything unlisted is an amount of material, which is what
    #: the default :data:`~ovpsolver.phase_field_system.visualize.layers.OCCUPY`
    #: says: it exists only outside the inclusions, so a picture multiplies it by
    #: the indicator. :data:`~ovpsolver.phase_field_system.visualize.layers.MASK`
    #: is for a ratio, which carries no volume: weighting would dim the material
    #: rather than hide the inclusions, and the meaningless values inside them
    #: would still set the ends of the colourbar.
    #: :data:`~ovpsolver.phase_field_system.visualize.layers.PLAIN` is for a
    #: quantity that already carries the indicator.
    #:
    #: Declared rather than inferred, since whether a number is an amount or a
    #: ratio is a fact about what the phase means by it. ``drag_floor_indicator``
    #: is a fraction of the Darcy measure, and weighting it would zero it exactly
    #: inside the inclusions, where it is one and where it exists to be read.
    plot_chi: dict[str, str] = {"drag_floor_indicator": MASK}

    def plot_chi_mode(self, field: str) -> str:
        """How one of this phase's saved fields relates to the inclusions.

        Collected up the class hierarchy, base first, so a subclass declaring
        modes for its own diagnostics keeps the ones declared for the fields it
        inherits; stating the whole mapping afresh would silently drop them to
        ``OCCUPY``, which is wrong for every ratio.
        """

        declared: dict[str, str] = {}
        for owner in reversed(type(self).__mro__):
            declared.update(vars(owner).get("plot_chi", {}))
        mode = declared.get(field, OCCUPY)
        if mode not in CHI_MODES:
            raise ValueError(
                f"{type(self).__name__}.plot_chi says {field} is {mode!r}; the "
                f"modes are {', '.join(CHI_MODES)}"
            )
        return mode
