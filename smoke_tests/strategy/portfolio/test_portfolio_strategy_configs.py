"""Smoke tests for PortfolioStrategyConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.strategy.portfolio.portfolio_strategy_config import PortfolioStrategyConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_portfolio_strategy_payloads() -> list[tuple[Path, dict[str, Any]]]:
    strategy_dir = COMPONENT_CONFIGS / "strategy" / "portfolio"
    return [
        (path, strip_config_class(load_yaml(path))) for path in sorted(strategy_dir.glob("*.yaml"))
    ]


def test_portfolio_strategy_configs_load() -> None:
    """Ensure PortfolioStrategyConfig payloads instantiate successfully."""
    for path, payload in _load_portfolio_strategy_payloads():
        model = PortfolioStrategyConfig(**payload)
        assert isinstance(model, PortfolioStrategyConfig), f"Failed for {path}"
