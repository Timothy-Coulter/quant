"""Smoke tests for execution component YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.core.config import ExecutionConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_execution_payloads() -> list[tuple[Path, dict[str, Any]]]:
    exec_dir = COMPONENT_CONFIGS / "execution"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(exec_dir.glob("*.yaml"))]


def test_execution_configs_load() -> None:
    """Ensure ExecutionConfig payloads instantiate successfully."""
    for path, payload in _load_execution_payloads():
        model = ExecutionConfig(**payload)
        assert isinstance(model, ExecutionConfig), f"Failed for {path}"
