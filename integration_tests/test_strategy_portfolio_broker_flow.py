"""Integration test for strategy → portfolio → broker loop via BacktestEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktesterConfig, get_config
from backtester.core.config_processor import ConfigProcessor


@pytest.fixture()
def engine_with_config(patch_data_retrieval: None) -> BacktestEngine:
    """Build a BacktestEngine from the momentum core YAML."""
    core_yaml = Path("component_configs/core/momentum_daily.yaml")
    processor = ConfigProcessor(base=get_config())
    resolved = processor.apply(source=core_yaml)
    assert isinstance(resolved, BacktesterConfig)
    return BacktestEngine(config=resolved)


def test_run_backtest_creates_components(
    engine_with_config: BacktestEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run backtest with stubbed simulation to ensure component wiring."""

    def _fake_run_simulation(self: BacktestEngine) -> dict[str, Any]:
        return {
            "portfolio_values": [100000.0, 100500.0],
            "base_values": [50000.0, 50250.0],
            "alpha_values": [50000.0, 50250.0],
        }

    def _fake_performance(self: BacktestEngine) -> dict[str, float]:
        return {"total_return": 0.01, "sharpe_ratio": 1.2}

    monkeypatch.setattr(BacktestEngine, "_run_simulation", _fake_run_simulation, raising=False)
    monkeypatch.setattr(
        BacktestEngine, "_calculate_performance_metrics", _fake_performance, raising=False
    )

    results = engine_with_config.run_backtest()

    assert engine_with_config.current_broker is not None
    assert engine_with_config.current_portfolio is not None
    assert engine_with_config.current_risk_manager is not None
    assert results["portfolio_values"] == [100000.0, 100500.0]
    assert results["performance"]["total_return"] == pytest.approx(0.01)
