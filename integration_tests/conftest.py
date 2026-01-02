"""Shared fixtures and stubs for integration tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure repo root importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_findatapy_stubs() -> None:
    """Provide minimal stubs for findatapy to satisfy imports."""
    market_module = cast(Any, types.ModuleType("findatapy.market"))
    for name in ("Market", "MarketDataGenerator", "MarketDataRequest"):
        setattr(market_module, name, type(name, (), {}))
    timeseries_module = cast(Any, types.ModuleType("findatapy.timeseries"))
    timeseries_module.DataQuality = type("DataQuality", (), {})
    util_module = cast(Any, types.ModuleType("findatapy.util"))
    util_module.LoggerManager = type("LoggerManager", (), {})
    findatapy_root = cast(Any, types.ModuleType("findatapy"))
    findatapy_root.__path__ = []
    findatapy_root.market = market_module
    findatapy_root.timeseries = timeseries_module
    findatapy_root.util = util_module
    sys.modules["findatapy"] = findatapy_root
    sys.modules["findatapy.market"] = market_module
    sys.modules["findatapy.timeseries"] = timeseries_module
    sys.modules["findatapy.util"] = util_module


_install_findatapy_stubs()


@pytest.fixture(scope="session")
def sample_price_data() -> pd.DataFrame:
    """Deterministic OHLCV frame with mild upward trend."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base = np.linspace(100.0, 104.0, num=len(dates))
    noise = np.linspace(0.0, 1.0, num=len(dates)) * 0.1
    close = base + noise
    data = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full_like(close, 10_000.0),
        },
        index=dates,
    )
    data.index.name = "timestamp"
    return data


@pytest.fixture()
def patch_data_retrieval(monkeypatch: pytest.MonkeyPatch, sample_price_data: pd.DataFrame) -> None:
    """Patch DataRetrieval to avoid external calls and return sample data."""

    class FakeDataRetrieval:
        def __init__(self, config: Any) -> None:
            self.config = config

        def get_data(self, config_override: Any | None = None) -> pd.DataFrame:
            return sample_price_data.copy()

        @classmethod
        def default_config(cls) -> Any:
            from backtester.core.config import DataRetrievalConfig

            return DataRetrievalConfig()

    monkeypatch.setattr("backtester.data.data_retrieval.DataRetrieval", FakeDataRetrieval)
    monkeypatch.setattr("backtester.core.backtest_engine.DataRetrieval", FakeDataRetrieval)


@pytest.fixture()
def event_bus() -> EventBus:
    """Provide a shared event bus for orchestration tests."""
    return EventBus()
