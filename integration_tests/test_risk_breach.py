"""Integration scenarios that intentionally violate risk constraints."""

from __future__ import annotations

import types
from unittest.mock import patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.core.event_bus import EventFilter
from backtester.strategy.signal.momentum_strategy import MomentumStrategy
from integration_tests.stubs import install_engine_stubs


def _sample_market_data(rows: int = 4) -> pd.DataFrame:
    index = pd.date_range("2024-02-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": [200 + i for i in range(rows)],
            "High": [201 + i for i in range(rows)],
            "Low": [199 + i for i in range(rows)],
            "Close": [200 + i for i in range(rows)],
            "Volume": [500.0 for _ in range(rows)],
        },
        index=index,
    )


def test_run_halts_orders_and_emits_alert_on_risk_breach() -> None:
    """High-risk assessments should cancel orders and emit alerts."""
    config = BacktesterConfig()
    engine = BacktestEngine(config=config)
    engine.current_data = _sample_market_data(rows=5)

    risk_alerts = []
    engine.event_bus.subscribe(
        lambda event: risk_alerts.append(event),
        EventFilter(event_types={"RISK_ALERT"}),
    )

    stub_portfolio, stub_broker, stub_risk = install_engine_stubs(
        engine,
        risk_alert_after=1,
    )
    stub_risk.allow_new_positions = False
    stub_risk.persist_after_trigger = True
    patched_metrics = types.MethodType(
        lambda self: {"final_portfolio_value": stub_portfolio.total_value}, engine
    )
    object.__setattr__(engine, "_calculate_performance_metrics", patched_metrics)

    with patch.object(
        MomentumStrategy,
        "generate_signals",
        lambda self, data, symbol: [
            {
                "signal_type": "BUY",
                "confidence": 0.9,
                "quantity": 1.0,
                "metadata": {"symbol": symbol, "timestamp": data.index[-1]},
            }
        ],
    ):
        results = engine.run_backtest()

    assert risk_alerts, "Engine should emit at least one risk alert event."
    assert stub_broker.order_manager.cancel_all_orders.called
    assert all(entry['event'] != 'TRADE' for entry in engine.trade_history)
    assert any(entry['event'] == 'RISK_ALERT' for entry in engine.trade_history)
    assert not stub_broker.submitted_orders
    assert results["performance"]["final_portfolio_value"] == pytest.approx(
        stub_portfolio.total_value
    )
