"""Smoke tests for BacktesterConfig YAML snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.core.config import BacktesterConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_core_payloads() -> list[tuple[Path, dict[str, Any]]]:
    core_dir = COMPONENT_CONFIGS / "core"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(core_dir.glob("*.yaml"))]


def test_core_configs_load() -> None:
    """Ensure BacktesterConfig payloads instantiate successfully."""
    for path, payload in _load_core_payloads():
        model = BacktesterConfig(**payload)
        assert isinstance(model, BacktesterConfig), f"Failed for {path}"
