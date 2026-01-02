"""Smoke tests for DataRetrievalConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.core.config import DataRetrievalConfig
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_data_payloads() -> list[tuple[Path, dict[str, Any]]]:
    data_dir = COMPONENT_CONFIGS / "data"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(data_dir.glob("*.yaml"))]


def test_data_configs_load() -> None:
    """Ensure DataRetrievalConfig payloads instantiate successfully."""
    for path, payload in _load_data_payloads():
        model = DataRetrievalConfig(**payload)
        assert isinstance(model, DataRetrievalConfig), f"Failed for {path}"
