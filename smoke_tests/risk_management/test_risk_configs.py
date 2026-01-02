"""Smoke tests for ComprehensiveRiskConfig YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.risk_management.component_configs.comprehensive_risk_config import (
    ComprehensiveRiskConfig,
)
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_risk_payloads() -> list[tuple[Path, dict[str, Any]]]:
    risk_dir = COMPONENT_CONFIGS / "risk_management"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(risk_dir.glob("*.yaml"))]


def test_risk_configs_load() -> None:
    """Ensure ComprehensiveRiskConfig payloads instantiate successfully."""
    for path, payload in _load_risk_payloads():
        model = ComprehensiveRiskConfig(**payload)
        assert isinstance(model, ComprehensiveRiskConfig), f"Failed for {path}"
