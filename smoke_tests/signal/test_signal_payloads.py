"""Smoke tests for simple signal payload fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class

REQUIRED_KEYS = {"signal_type", "action", "confidence", "metadata", "timestamp"}


def _load_signal_payloads() -> list[tuple[Path, dict[str, Any]]]:
    signal_dir = COMPONENT_CONFIGS / "signal"
    return [
        (path, strip_config_class(load_yaml(path))) for path in sorted(signal_dir.glob("*.yaml"))
    ]


def test_signal_payloads_have_expected_shape() -> None:
    """Validate basic contract for signal payload fixture YAMLs."""
    for path, payload in _load_signal_payloads():
        missing = REQUIRED_KEYS - set(payload)
        assert not missing, f"Missing keys {missing} in {path}"
        assert isinstance(
            payload["confidence"], (int, float)
        ), f"confidence must be numeric in {path}"
        assert isinstance(payload["metadata"], dict), f"metadata must be a mapping in {path}"
