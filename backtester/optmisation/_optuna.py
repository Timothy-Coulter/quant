"""Utilities for working with the optional Optuna dependency."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, cast

_optuna_module: ModuleType | None
try:  # pragma: no cover - optional dependency wiring
    import optuna as _imported_optuna
except ImportError:  # pragma: no cover - optional dependency wiring
    _optuna_module = None
else:  # pragma: no cover - optional dependency wiring
    _optuna_module = cast(ModuleType, _imported_optuna)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def require_optuna() -> ModuleType:
    """Return the Optuna module or raise a helpful error if it's missing."""
    if _optuna_module is None:
        raise ModuleNotFoundError(
            "optuna is required for optimization features. "
            "Install it with `pip install optuna` to enable this functionality."
        )
    return _optuna_module


def get_optuna_module() -> ModuleType | None:
    """Return the Optuna module when available."""
    return _optuna_module
