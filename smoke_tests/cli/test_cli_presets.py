"""Smoke tests for CLI preset YAMLs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class

REQUIRED_KEYS = {"description", "args"}


def _load_cli_payloads() -> list[tuple[Path, dict[str, Any]]]:
    cli_dir = COMPONENT_CONFIGS / "cli"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(cli_dir.glob("*.yaml"))]


def _is_iterable_of_cli_values(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Iterable):
        return False
    allowed_types = (str, int, float, bool, date)
    return all(isinstance(item, allowed_types) for item in value)


def test_cli_presets_have_expected_shape() -> None:
    """Verify CLI preset YAMLs expose description and args in the correct shape."""
    for path, payload in _load_cli_payloads():
        missing = REQUIRED_KEYS - set(payload)
        assert not missing, f"Missing keys {missing} in {path}"
        assert _is_iterable_of_cli_values(
            payload["args"]
        ), f"args must be a sequence of CLI-friendly values in {path}"
