"""The driver of the Lie-split OVP timestep, and the numerics it owns.

:class:`~ovpsolver.solver.solver.PhaseFieldSolver` advances a
:class:`~ovpsolver.phase_field_system.PhaseFieldSystem`. Around it sit the
driver's choices rather than the material's: :mod:`.parameters` for the spec's
``solver`` block, :mod:`.regularization` for the floors that keep degenerate
coefficients away from zero, :mod:`.onsager` for the convergence measure,
:mod:`.instrument` and :mod:`.progress` for the log and the bar, and :mod:`.utils`
for PETSc diagnostics.

Exported lazily. The driver imports the phase-field layer, which imports this
package's leaf modules; eager re-exports would make each of those imports import
the driver first.
"""

from importlib import import_module

_EXPORTS = {
    "PhaseFieldSolver": "solver",
    "SolveDiverged": "utils",
    "TimesteppingParameters": "parameters",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__package__}.{_EXPORTS[name]}"), name)
