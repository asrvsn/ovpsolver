"""Finite-element solver for stateful phase-field systems governed by an Onsager
variational principle.

The C++ kernels are imported with the package; every other export is imported
from its submodule when first asked for (:func:`__getattr__`).
"""

import shlex
import sys
import sysconfig
from importlib import import_module

from .ovpsolver_ext import (
    burgers_flux,
    burgers_rhs,
    burgers_rhs_muscl_vanleer,
    burgers_step,
    burgers_upwind_values_muscl_vanleer,
    max_cfl_timestep,
)

#: The sysconfig entries holding linker flags.
_LINKER_FLAG_KEYS = ("LDSHARED", "LDCXXSHARED", "BLDSHARED", "LDFLAGS", "PY_LDFLAGS")


def _dedupe_linker_flags(mapping: dict | None) -> None:
    """Drop repeated ``-Wl,-rpath,...`` and ``-L...`` tokens from the linker flags.

    A value is rewritten only if something was dropped, so an untouched one keeps
    its original quoting.
    """

    for key in _LINKER_FLAG_KEYS:
        value = (mapping or {}).get(key)
        if not isinstance(value, str):
            continue
        tokens = shlex.split(value)
        seen: set[str] = set()
        kept: list[str] = []
        for token in tokens:
            if token.startswith(("-Wl,-rpath,", "-L")):
                if token in seen:
                    continue
                seen.add(token)
            kept.append(token)
        if len(kept) < len(tokens):
            mapping[key] = shlex.join(kept)


def _dedupe_sysconfig_linker_flags() -> None:
    """Strip the duplicate rpath and ``-L`` flags conda's macOS Python bakes in.

    Each duplicate makes the linker warn during FFCx/CFFI form compilation. The
    flags are read through two independent caches, stdlib ``sysconfig`` and the
    ``distutils`` vendored by setuptools, which is what CFFI compiles through;
    both are filled from the shared ``_sysconfigdata`` module. So that module is
    patched, which covers a cache filled later, and so is each cache already
    filled.
    """

    try:
        data_module = __import__(sysconfig._get_sysconfigdata_name())
        _dedupe_linker_flags(getattr(data_module, "build_time_vars", None))
    except Exception:
        pass
    _dedupe_linker_flags(sysconfig.get_config_vars())
    for module_name in ("setuptools._distutils.sysconfig", "distutils.sysconfig"):
        module = sys.modules.get(module_name)
        if module is not None:
            _dedupe_linker_flags(getattr(module, "_config_vars", None))


_dedupe_sysconfig_linker_flags()

#: Export name to the submodule that defines it, imported on first access.
_LAZY_EXPORTS = {
    "DiffuseDomain": "diffuse_domain",
    "DiffuseDomainParameters": "diffuse_domain",
    "sphere": "diffuse_domain",
    "ElementOwner": "fem.elements",
    "ElementSpec": "fem.elements",
    "Solve": "fem.elements",
    "StateElement": "fem.elements",
    "StaticElement": "fem.elements",
    "PhaseField": "phase_field_system.phase_field",
    "PhaseFieldParameters": "phase_field_system.phase_field",
    "PolymerizingPhaseField": "polymerizing_b.phase_field",
    "PolymerizingPhaseFieldParameters": "polymerizing_b.phase_field",
    "CHPhaseField": "model_b.phase_field",
    "CHPhaseFieldParameters": "model_b.phase_field",
    "PhaseFieldSystem": "phase_field_system",
    "PhaseFieldSystemParameters": "phase_field_system",
    "ModelB": "model_b",
    "ModelBParameters": "model_b",
    "PolymerizingB": "polymerizing_b",
    "PolymerizingBParameters": "polymerizing_b",
    "PhaseFieldSolver": "solver",
    "TimesteppingParameters": "solver",
    "Flux": "phase_field_system.transport",
    "Transported": "phase_field_system.transport",
    "SolverParameters": "solver.parameters",
    "ZPartitionParameters": "polymerizing_b.phase_field",
}

__all__ = [
    "burgers_flux",
    "burgers_rhs",
    "burgers_rhs_muscl_vanleer",
    "burgers_step",
    "burgers_upwind_values_muscl_vanleer",
    "max_cfl_timestep",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str) -> object:
    """Import the submodule owning ``name`` only when it is asked for.

    Every one of them loads dolfinx, which importing the package, or reaching
    only the C++ kernels, should not cost.
    """

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module_name}", __package__), name)
