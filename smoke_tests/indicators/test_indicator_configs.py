"""Smoke tests for IndicatorConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.indicators.indicator_configs import IndicatorConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_indicator_payloads() -> list[tuple[Path, dict[str, Any]]]:
    indicators_dir = COMPONENT_CONFIGS / "indicators"
    return [
        (path, strip_config_class(load_yaml(path)))
        for path in sorted(indicators_dir.glob("*.yaml"))
    ]


def test_indicator_configs_load() -> None:
    """Ensure IndicatorConfig payloads instantiate successfully."""
    for path, payload in _load_indicator_payloads():
        model = IndicatorConfig(**payload)
        assert isinstance(model, IndicatorConfig), f"Failed for {path}"
