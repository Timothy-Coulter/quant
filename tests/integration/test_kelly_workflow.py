"""End-to-end test for the Kelly portfolio workflow."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig
from backtester.risk_management.risk_control_manager import RiskControlManager
from backtester.strategy.portfolio.kelly_criterion_strategy import KellyCriterionStrategy
from backtester.strategy.signal.momentum_strategy import MomentumStrategy


def _sample_market_data(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(rows)],
            "High": [101 + i for i in range(rows)],
            "Low": [99 + i for i in range(rows)],
            "Close": [100.5 + i * 0.5 for i in range(rows)],
            "Volume": [2000.0 for _ in range(rows)],
        },
        index=index,
    )


def test_kelly_strategy_propagates_weights_through_engine() -> None:
    """Kelly target weights should drive portfolio orders and populate base/alpha metrics."""
    config = BacktesterConfig()
    engine = BacktestEngine(config=config)
    engine.current_data = _sample_market_data(rows=30)

    engine._calculate_performance_metrics = types.MethodType(
        lambda self: {"final_portfolio_value": self.current_portfolio.total_value},
        engine,
    )

    def constant_weights(
        self: KellyCriterionStrategy, market_data: dict[str, Any]
    ) -> dict[str, float]:
        weights = {symbol: 0.6 for symbol in self.symbols}
        self.kelly_metrics = {
            symbol: {'kelly_weight': 0.6, 'win_probability': 0.6} for symbol in self.symbols
        }
        return weights

    with (
        patch.object(RiskControlManager, "check_portfolio_risk", autospec=True) as mock_check,
        patch.object(RiskControlManager, "add_risk_signal", autospec=True, return_value=None),
        patch.object(RiskControlManager, "can_open_position", autospec=True, return_value=True),
        patch.object(RiskControlManager, "record_order", autospec=True, return_value=None),
        patch.object(
            KellyCriterionStrategy,
            "calculate_target_weights",
            autospec=True,
            side_effect=constant_weights,
        ),
        patch.object(
            MomentumStrategy,
            "generate_signals",
            autospec=True,
            side_effect=lambda self, data, symbol: [
                {
                    "signal_type": "BUY",
                    "confidence": 0.9,
                    "quantity": 1.0,
                    "metadata": {"symbol": symbol, "timestamp": data.index[-1]},
                }
            ],
        ),
    ):
        mock_check.return_value = {"risk_level": "LOW", "violations": []}
        results = engine.run_backtest()

    assert engine.portfolio_strategy is not None
    assert engine.portfolio_strategy.target_weights.get('SPY') == pytest.approx(0.6)
    assert engine.portfolio_strategy.trade_history

    portfolio_values = results['portfolio_values']
    base_values = results['base_values']
    alpha_values = results['alpha_values']
    expected_iterations = len(engine.current_data) - 1
    assert len(portfolio_values) == expected_iterations
    assert len(base_values) == len(portfolio_values)
    assert len(alpha_values) == len(portfolio_values)
    assert base_values[0] * 2 == pytest.approx(portfolio_values[0])
    assert alpha_values[0] * 2 == pytest.approx(portfolio_values[0])
    assert engine.current_portfolio is not None
    assert results['performance']['final_portfolio_value'] == pytest.approx(
        engine.current_portfolio.total_value
    )
