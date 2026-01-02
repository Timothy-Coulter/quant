"""Smoke tests for utility support YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_utils_payloads() -> list[tuple[Path, dict[str, Any]]]:
    utils_dir = COMPONENT_CONFIGS / "utils"
    return [
        (path, strip_config_class(load_yaml(path))) for path in sorted(utils_dir.glob("*.yaml"))
    ]


def test_utils_configs_have_expected_shape() -> None:
    """Validate utility config payloads are non-empty and well-formed."""
    for path, payload in _load_utils_payloads():
        assert payload, f"Utility config empty: {path}"
        assert "__config_class__" not in payload or isinstance(payload.get("__config_class__"), str)
