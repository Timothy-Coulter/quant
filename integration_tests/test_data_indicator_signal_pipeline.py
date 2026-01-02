"""Integration test for data → indicator → signal pipeline."""

from __future__ import annotations

import pandas as pd

from backtester.core.event_bus import EventBus
from backtester.strategy.signal.momentum_strategy import MomentumStrategy
from backtester.strategy.signal.signal_strategy_config import MomentumStrategyConfig


def test_momentum_strategy_generates_signals(
    sample_price_data: pd.DataFrame, event_bus: EventBus
) -> None:
    """MomentumStrategy should consume OHLCV data and emit scored signals."""
    config = MomentumStrategyConfig()
    strategy = MomentumStrategy(config=config, event_bus=event_bus)

    signals = strategy.generate_signals(sample_price_data, symbol="SPY")

    assert signals, "Expected at least one signal"
    first = signals[0]
    assert first["signal_type"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= float(first.get("confidence", 0.0)) <= 1.0
    assert "metadata" in first
