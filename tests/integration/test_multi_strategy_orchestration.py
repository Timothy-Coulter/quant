"""Integration tests for the multi-strategy orchestrator pipeline."""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd
import pytest

from backtester.core.event_bus import EventBus, EventFilter
from backtester.core.events import create_market_data_event
from backtester.strategy.orchestration import (
    BaseStrategyOrchestrator,
    ConflictResolutionStrategy,
    OrchestrationConfig,
    OrchestratorType,
    StrategyKind,
    StrategyReference,
)


class RecordingStrategy:
    """Simple strategy stub that emits pre-configured signals."""

    def __init__(
        self,
        name: str,
        signals: list[dict[str, Any]],
        execution_order: deque[str],
    ) -> None:
        """Store static signals and shared execution order tracker."""
        self.name = name
        self._signals = [dict(signal) for signal in signals]
        self.execution_order = execution_order

    def generate_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the pre-configured signal payload."""
        self.execution_order.append(self.name)
        return [dict(signal) for signal in self._signals]


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000.0],
        },
        index=[pd.Timestamp("2024-03-01 09:30")],
    )


def test_orchestrator_honours_dependencies_and_merges_signals() -> None:
    """Sequential orchestrator should respect ordering and produce a merged signal."""
    event_bus = EventBus()
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.SEQUENTIAL,
        conflict_resolution=ConflictResolutionStrategy.WEIGHTED_MERGE,
        strategies=[
            StrategyReference(identifier="alpha", weight=2.0, priority=0),
            StrategyReference(identifier="beta", weight=1.0, priority=1, depends_on=["alpha"]),
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=event_bus)

    execution_order: deque[str] = deque()
    alpha = RecordingStrategy(
        "alpha",
        [{"signal_type": "BUY", "confidence": 0.8}],
        execution_order,
    )
    beta = RecordingStrategy(
        "beta",
        [{"signal_type": "SELL", "confidence": 0.7}],
        execution_order,
    )
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL, weight=2.0)
    orchestrator.register_strategy("beta", beta, kind=StrategyKind.SIGNAL, weight=1.0)

    emitted_events = []
    event_bus.subscribe(
        lambda event: emitted_events.append(event),
        EventFilter(event_types={"SIGNAL"}),
    )

    frame = _market_frame()
    market_event = create_market_data_event(
        "SPY", {"open": 100, "high": 101, "low": 99, "close": 100.5}
    )
    market_event.metadata["data_frame"] = frame.copy()

    result = orchestrator.on_market_data(market_event, frame)

    assert list(execution_order) == ["alpha", "beta"]
    assert len(result.signals) == 1
    merged_signal = result.signals[0]
    assert merged_signal.payload["signal_type"] == "BUY"
    aggregation = merged_signal.payload.get("aggregation", {})
    assert aggregation.get("total_weight") == pytest.approx(2.0)
    assert result.metadata["strategy_count"] == 2
    assert len(emitted_events) == 1
