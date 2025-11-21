"""Workflow-focused tests for BacktestEngine."""

from __future__ import annotations

import types
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.core.event_bus import EventFilter
from backtester.strategy.signal.momentum_strategy import MomentumStrategy


def _sample_market_data(rows: int = 6) -> pd.DataFrame:
    """Create deterministic OHLCV data for workflow tests."""
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [100 + i for i in range(rows)],
            "High": [101 + i for i in range(rows)],
            "Low": [99 + i for i in range(rows)],
            "Close": [100.5 + i for i in range(rows)],
            "Volume": [1_000_000 for _ in range(rows)],
        },
        index=index,
    )
    return frame


@pytest.mark.parametrize("asset_count", [1, 2])
def test_engine_emits_ordered_events(asset_count: int) -> None:
    """Ensure the canonical market→signal→order→portfolio flow fires in sequence."""
    config = BacktesterConfig()
    engine = BacktestEngine(config=config)
    cast(Any, engine)._initialize_portfolio_strategy = types.MethodType(
        lambda self, symbols, params: None,
        engine,
    )
    market_data = _sample_market_data()
    engine.current_data = market_data
    if engine.config.data is not None:
        tickers = [f"ASSET{i}" for i in range(asset_count)]
        engine.config.data.tickers = tickers or ["ASSET0"]
    cast(Any, engine.data_handler).get_data = MagicMock(return_value=market_data)

    emitted_events: list[str] = []
    engine.event_bus.subscribe(
        lambda event: emitted_events.append(event.event_type),
        EventFilter(event_types={"MARKET_DATA", "SIGNAL", "ORDER", "PORTFOLIO_UPDATE"}),
    )

    def signal_side_effect(
        self: MomentumStrategy, data: pd.DataFrame, symbol: str
    ) -> list[dict[str, Any]]:
        """Emit deterministic BUY/SELL signals to stress the workflow."""
        return [
            {
                "signal_type": "SELL" if idx % 2 else "BUY",
                "confidence": 0.9,
            }
            for idx in range(asset_count)
        ]

    with patch.object(
        MomentumStrategy,
        "generate_signals",
        autospec=True,
        side_effect=signal_side_effect,
    ):
        results = engine.run_backtest()

    expected_ticks = len(market_data) - 1
    assert "performance" in results

    segments: list[list[str]] = []
    current: list[str] = []
    for event_type in emitted_events:
        if event_type == "MARKET_DATA":
            if current:
                segments.append(current)
            current = ["MARKET_DATA"]
        else:
            current.append(event_type)
    if current:
        segments.append(current)

    assert len(segments) == expected_ticks
    for segment in segments:
        assert segment[0] == "MARKET_DATA"
        assert segment[-1] == "PORTFOLIO_UPDATE"
        signal_indices = [i for i, etype in enumerate(segment) if etype == "SIGNAL"]
        order_indices = [i for i, etype in enumerate(segment) if etype == "ORDER"]
        assert signal_indices, "Expected at least one SIGNAL event per tick."
        assert signal_indices[0] > 0
        assert order_indices, "Expected ORDER events between signals and portfolio updates."
        assert min(order_indices) > max(signal_indices)
        assert max(order_indices) < len(segment) - 1
