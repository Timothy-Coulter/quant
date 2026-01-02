"""Integration test for orchestration across multiple strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtester.core.event_bus import EventBus
from backtester.core.events import MarketDataEvent, create_market_data_event
from backtester.strategy.orchestration import (
    BaseStrategyOrchestrator,
    OrchestrationConfig,
    OrchestratorType,
    StrategyKind,
)
from backtester.strategy.orchestration.orchestration_strategy_config import StrategyReference


@dataclass
class StaticStrategy:
    """Simple stub that returns preconfigured signals."""

    name: str
    signals: list[dict[str, Any]]

    def generate_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the configured static signals."""
        return list(self.signals)


def _market_event_from_frame(symbol: str, data: pd.DataFrame) -> MarketDataEvent:
    """Create a market event from a single-row frame."""
    row = data.iloc[0]
    payload = {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "timestamp": data.index[0].timestamp() if hasattr(data.index[0], "timestamp") else None,
        "data_type": "bar",
    }
    event = create_market_data_event(symbol, payload)
    event.metadata["data_frame"] = data
    return event


def test_orchestrator_blends_weighted_signals(
    sample_price_data: pd.DataFrame, event_bus: EventBus
) -> None:
    """Ensemble orchestrator should merge signals with weights applied."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.ENSEMBLE,
        strategies=[
            StrategyReference(identifier="mom", weight=0.6),
            StrategyReference(identifier="mean_rev", weight=0.4),
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=event_bus)

    mom = StaticStrategy("mom", [{"signal_type": "BUY", "confidence": 0.7}])
    mean_rev = StaticStrategy("mean_rev", [{"signal_type": "SELL", "confidence": 0.55}])
    orchestrator.register_strategy("mom", mom, kind=StrategyKind.SIGNAL, weight=0.6)
    orchestrator.register_strategy("mean_rev", mean_rev, kind=StrategyKind.SIGNAL, weight=0.4)

    data = sample_price_data.iloc[:1]
    event = _market_event_from_frame("SPY", data)
    result = orchestrator.on_market_data(event, data)

    assert result.signals, "Expected orchestrated signals"
    merged = result.signals[0]
    assert merged.payload.get("aggregation") is not None
    total_weight = merged.payload["aggregation"]["total_weight"]
    assert total_weight >= 0.6
    assert merged.signal_type in {"BUY", "SELL"}
