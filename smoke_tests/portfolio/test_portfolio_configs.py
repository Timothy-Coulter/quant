"""Smoke tests for PortfolioConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.core.config import PortfolioConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_portfolio_payloads() -> list[tuple[Path, dict[str, Any]]]:
    portfolio_dir = COMPONENT_CONFIGS / "portfolio"
    return [
        (path, strip_config_class(load_yaml(path))) for path in sorted(portfolio_dir.glob("*.yaml"))
    ]


def test_portfolio_configs_load() -> None:
    """Ensure PortfolioConfig payloads instantiate successfully."""
    for path, payload in _load_portfolio_payloads():
        model = PortfolioConfig(**payload)
        assert isinstance(model, PortfolioConfig), f"Failed for {path}"
