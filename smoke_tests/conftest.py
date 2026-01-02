"""Shared helpers for loading component config YAMLs in smoke tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPONENT_CONFIGS = PROJECT_ROOT / "component_configs"

# Ensure project root is importable when running pytest on this subdirectory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_findatapy_stub() -> None:
    """Provide a lightweight stub for findatapy imports used in config modules."""
    market_module = cast(Any, types.ModuleType("findatapy.market"))
    for name in ("Market", "MarketDataGenerator", "MarketDataRequest"):
        setattr(market_module, name, type(name, (), {}))
    timeseries_module = cast(Any, types.ModuleType("findatapy.timeseries"))
    timeseries_module.DataQuality = type("DataQuality", (), {})
    util_module = cast(Any, types.ModuleType("findatapy.util"))
    util_module.LoggerManager = type("LoggerManager", (), {})
    findatapy_root = cast(Any, types.ModuleType("findatapy"))
    findatapy_root.__path__ = []  # mark as package-like
    findatapy_root.market = market_module
    findatapy_root.timeseries = timeseries_module
    findatapy_root.util = util_module
    sys.modules["findatapy"] = findatapy_root
    sys.modules["findatapy.market"] = market_module
    sys.modules["findatapy.timeseries"] = timeseries_module
    sys.modules["findatapy.util"] = util_module


_install_findatapy_stub()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and assert it contains a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise AssertionError(f"YAML root must be a mapping: {path}")
    return dict(data)


def strip_config_class(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the __config_class__ marker to keep model constructors clean."""
    return {k: v for k, v in payload.items() if k != "__config_class__"}
