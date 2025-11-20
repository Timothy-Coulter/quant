"""Smoke tests for the documented public API surface."""

import importlib

import pytest

PUBLIC_MODULES = [
    "backtester",
    "backtester.core",
    "backtester.strategy",
    "backtester.strategy.portfolio",
    "backtester.strategy.signal",
    "backtester.strategy.orchestration",
    "backtester.data",
    "backtester.execution",
    "backtester.indicators",
    "backtester.model",
    "backtester.optmisation",
    "backtester.portfolio",
    "backtester.risk_management",
    "backtester.utils",
]


@pytest.mark.parametrize("module_path", PUBLIC_MODULES)
def test_declared_symbols_are_importable(module_path: str) -> None:
    """Every name listed in __all__ must be present on the module."""
    module = importlib.import_module(module_path)
    public_names = getattr(module, "__all__", None)
    assert public_names, f"{module_path} is missing an __all__ declaration"

    missing = [name for name in public_names if not hasattr(module, name)]
    assert not missing, f"{module_path} is missing exports: {missing}"
