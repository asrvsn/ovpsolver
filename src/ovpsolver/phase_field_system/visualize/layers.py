"""What a figure is made of, decided by the mixture rather than by the plotter.

A rendered panel is a stack of layers, and only the mixture knows which layers it
has: a sol phase is one, a polymerizing phase one or two depending on whether the
gel is shown apart from the sol, and an inclusion is a layer drawn by its
complement. So the answer comes from
:meth:`~ovpsolver.phase_field_system.parameters.PhaseFieldSystemParameters.phase_layers`,
which a subclass with a new kind of phase overrides.

This module is the vocabulary that method speaks in: a saved field's name, a
colour, and how to read the field. No arrays, dolfinx or matplotlib, so that a
parameters class can build a layer without importing the rendering stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...diffuse_domain import DiffuseDomainParameters
    from ..phase_field import PhaseFieldParameters

#: How a layer's values relate to the inclusion indicator ``chi``.
#:
#: ``occupy``
#:     a volume fraction of the material outside the inclusions: multiplied by
#:     ``chi``, so it vanishes inside them.
#: ``complement``
#:     the inclusions themselves, drawn as ``1 - chi``.
#: ``plain``
#:     drawn as saved.
#: ``mask``
#:     a ratio, which has no volume and so would be dimmed rather than hidden by
#:     the ``chi`` product. Flattened to its minimum inside the inclusions, so
#:     the colour range is set by the material alone.
OCCUPY = "occupy"
COMPLEMENT = "complement"
PLAIN = "plain"
MASK = "mask"

#: The whole vocabulary, so a phase declaring how its diagnostics are drawn can
#: be told at once that it named a mode nothing implements.
CHI_MODES = (OCCUPY, COMPLEMENT, PLAIN, MASK)

#: The field of the layer drawing the inclusion geometry. The geometry is
#: analytic and its spec travels with the run, so a figure evaluates the
#: indicator exactly rather than reading back a discrete copy that would have to
#: be interpolated or projected. Angle brackets, so it cannot collide with a
#: qualified field name.
INDICATOR = "<indicator>"


@dataclass(frozen=True, slots=True)
class Layer:
    """One saved field, drawn as one row or as one colour in a composite.

    ``color`` is a name rather than an RGB triple: resolving it needs
    matplotlib, which a parameters object does not import, so the renderer does.

    With a colour a layer is a phase: one hue, with the value as opacity, so
    several compose into one picture. Without one it is a diagnostic: the value
    is the hue, through ``colormap`` or else the style's sequential or diverging
    map, and it gets a row and a colourbar of its own.
    """

    name: str
    field: str
    color: str | None = None
    colormap: str | None = None
    chi: str = OCCUPY

    @property
    def is_phase(self) -> bool:
        return self.color is not None


def phase_layer(
    parameters: PhaseFieldParameters, field: str = "phi", color: str | None = None
) -> Layer:
    """A phase's saved ``field``, by default its volume fraction, in its colour."""

    return Layer(
        name=parameters.name if field == "phi" else f"{parameters.name}.{field}",
        field=f"{parameters.name}.{field}",
        color=parameters.color if color is None else color,
    )


def inclusion_layer(parameters: DiffuseDomainParameters) -> Layer:
    """The layer drawing the fixed inclusions, as the complement of ``chi``.

    Whether there are any is the caller's question, since answering it needs a
    mesh; see
    :meth:`~ovpsolver.phase_field_system.parameters.PhaseFieldSystemParameters.inclusion_layer`.
    """

    return Layer(
        name="inclusion",
        field=INDICATOR,
        color=parameters.inclusion_color,
        chi=COMPLEMENT,
    )
