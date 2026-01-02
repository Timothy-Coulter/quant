"""Smoke tests for SignalStrategyConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.strategy.signal.signal_strategy_config import SignalStrategyConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_strategy_payloads() -> list[tuple[Path, dict[str, Any]]]:
    strategy_dir = COMPONENT_CONFIGS / "strategy" / "signal"
    return [
        (path, strip_config_class(load_yaml(path))) for path in sorted(strategy_dir.glob("*.yaml"))
    ]


def test_signal_strategy_configs_load() -> None:
    """Ensure SignalStrategyConfig payloads instantiate successfully."""
    for path, payload in _load_strategy_payloads():
        model = SignalStrategyConfig(**payload)
        assert isinstance(model, SignalStrategyConfig), f"Failed for {path}"
