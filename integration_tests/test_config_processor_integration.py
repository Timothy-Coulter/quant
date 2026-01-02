"""Integration test ensuring ConfigProcessor wiring into BacktestEngine."""

from __future__ import annotations

from pathlib import Path

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig, get_config
from backtester.core.config_processor import ConfigProcessor


def test_core_config_builds_engine_without_side_effects(patch_data_retrieval: None) -> None:
    """Load a core YAML, build BacktesterConfig, and initialise the engine."""
    core_yaml = Path("component_configs/core/momentum_daily.yaml")
    processor = ConfigProcessor(base=get_config())
    resolved = processor.apply(source=core_yaml)
    assert isinstance(resolved, BacktesterConfig)
    assert resolved.data is not None
    assert resolved.data.tickers == ["SPY", "QQQ"]

    engine = BacktestEngine(config=resolved)
    assert engine.config.strategy is not None
    assert engine.data_handler is not None
    assert engine.config.data is not None
    assert engine.config.data.tickers == ["SPY", "QQQ"]
