"""Smoke tests for OrchestrationConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.strategy.orchestration.orchestration_strategy_config import OrchestrationConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_orchestration_payloads() -> list[tuple[Path, dict[str, Any]]]:
    orchestration_dir = COMPONENT_CONFIGS / "strategy" / "orchestration"
    return [
        (path, strip_config_class(load_yaml(path)))
        for path in sorted(orchestration_dir.glob("*.yaml"))
    ]


def test_orchestration_configs_load() -> None:
    """Ensure OrchestrationConfig payloads instantiate successfully."""
    for path, payload in _load_orchestration_payloads():
        model = OrchestrationConfig(**payload)
        assert isinstance(model, OrchestrationConfig), f"Failed for {path}"
