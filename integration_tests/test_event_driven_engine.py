"""Integration tests for the event-driven backtest engine pipeline."""

from __future__ import annotations

import types
from collections import defaultdict
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.core.event_bus import EventFilter
from backtester.strategy.orchestration import StrategyKind
from backtester.strategy.signal.base_signal_strategy import BaseSignalStrategy
from backtester.strategy.signal.momentum_strategy import MomentumStrategy
from integration_tests.stubs import capture_processed_signals, install_engine_stubs


def _sample_market_data(rows: int = 5) -> pd.DataFrame:
    """Build deterministic OHLCV data for the integration scenario."""
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "Open": [100 + i for i in range(rows)],
        "High": [102 + i for i in range(rows)],
        "Low": [98 + i for i in range(rows)],
        "Close": [101 + i for i in range(rows)],
        "Volume": [1000.0 for _ in range(rows)],
    }
    return pd.DataFrame(data, index=index)


def test_backtest_engine_event_driven_flow() -> None:
    """Ensure the engine propagates events through the orchestrator to portfolio stubs."""
    config = BacktesterConfig()
    engine = BacktestEngine(config=config)

    market_data = _sample_market_data(rows=6)
    engine.current_data = market_data

    tracked_events: dict[str, list[Any]] = defaultdict(list)
    engine.event_bus.subscribe(
        lambda event: tracked_events[event.event_type].append(event),
        EventFilter(
            event_types={"MARKET_DATA", "SIGNAL", "ORDER", "PORTFOLIO_UPDATE", "RISK_ALERT"}
        ),
    )

    periods = len(market_data) - 1
    stub_portfolio, stub_broker, _ = install_engine_stubs(
        engine,
        risk_alert_after=max(2, periods * 2 - 2),
    )
    processed_signals: list[list[dict[str, Any]]] = []
    capture_processed_signals(engine, processed_signals)

    patched_metrics = types.MethodType(
        lambda self: {"final_portfolio_value": stub_portfolio.total_value}, engine
    )
    object.__setattr__(engine, "_calculate_performance_metrics", patched_metrics)

    original_create_strategy = engine.create_strategy

    def dual_strategy_factory(
        self: BacktestEngine, strategy_params: dict[str, Any] | None = None
    ) -> BaseSignalStrategy:
        """Create the default strategy and register a secondary copy for multi-strategy testing."""
        primary = original_create_strategy(strategy_params)
        secondary_config = self._build_momentum_config(strategy_params or {}, "secondary_strategy")
        secondary_strategy = MomentumStrategy(secondary_config, self.event_bus)
        self.strategy_orchestrator.register_strategy(
            identifier="secondary_strategy",
            strategy=secondary_strategy,
            kind=StrategyKind.SIGNAL,
            priority=1,
        )
        return primary

    patched_factory = types.MethodType(dual_strategy_factory, engine)
    object.__setattr__(engine, "create_strategy", patched_factory)

    with (
        patch.object(
            MomentumStrategy,
            "generate_signals",
            lambda self, data, symbol: [
                {
                    "signal_type": (
                        "SELL" if "secondary" in getattr(self, "name", "").lower() else "BUY"
                    ),
                    "confidence": 0.9,
                    "quantity": 1.0,
                    "metadata": {
                        "symbol": symbol,
                        "timestamp": (
                            data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
                        ),
                    },
                }
            ],
        ),
        ExitStack() as stack,
    ):
        mock_before_tick = stack.enter_context(
            patch.object(MomentumStrategy, "before_tick", autospec=True, return_value=None)
        )
        mock_after_tick = stack.enter_context(
            patch.object(MomentumStrategy, "after_tick", autospec=True, return_value=None)
        )
        mock_before_run = stack.enter_context(
            patch.object(MomentumStrategy, "before_run", autospec=True, return_value=None)
        )
        mock_after_run = stack.enter_context(
            patch.object(MomentumStrategy, "after_run", autospec=True, return_value=None)
        )
        results = engine.run_backtest()

    expected_iterations = periods
    strategy_count = 2

    assert len(processed_signals) == expected_iterations
    assert all(batch for batch in processed_signals)
    assert len(tracked_events["MARKET_DATA"]) == expected_iterations
    assert len(tracked_events["SIGNAL"]) >= expected_iterations * strategy_count
    assert tracked_events["ORDER"]
    assert len(tracked_events["PORTFOLIO_UPDATE"]) == expected_iterations
    last_portfolio_event = tracked_events["PORTFOLIO_UPDATE"][-1]
    assert last_portfolio_event.total_value == pytest.approx(stub_portfolio.total_value)
    assert tracked_events["RISK_ALERT"]
    assert tracked_events["RISK_ALERT"][0].metadata.get("violations") == ["MAX_DRAWDOWN"]
    assert stub_broker.order_manager.cancel_all_orders.called

    assert results["performance"]["final_portfolio_value"] == pytest.approx(
        stub_portfolio.total_value
    )
    assert stub_portfolio.before_run_calls == 1
    assert stub_portfolio.after_run_calls == 1
    assert stub_portfolio.before_tick_calls == expected_iterations
    assert stub_portfolio.after_tick_calls == expected_iterations
    assert stub_broker.before_run_calls == 1
    assert stub_broker.after_run_calls == 1
    assert stub_broker.before_tick_calls == expected_iterations
    assert stub_broker.after_tick_calls == expected_iterations
    assert mock_before_tick.call_count == expected_iterations
    assert mock_after_tick.call_count == expected_iterations
    assert mock_before_run.call_count == 1
    assert mock_after_run.call_count == 1
