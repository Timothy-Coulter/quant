"""Quantitative Backtesting Framework.

This package provides a comprehensive framework for quantitative portfolio backtesting
with advanced risk management, strategy implementation, and performance analysis.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = ["BacktestEngine", "BacktesterConfig", "DualPoolPortfolio"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestEngine": ("backtester.core.backtest_engine", "BacktestEngine"),
    "BacktesterConfig": ("backtester.core.config", "BacktesterConfig"),
    "DualPoolPortfolio": ("backtester.portfolio.dual_pool_portfolio", "DualPoolPortfolio"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy exports to keep import-time dependencies minimal."""
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose dynamically-resolved attributes via dir()."""
    return sorted(list(globals().keys()) + __all__)
