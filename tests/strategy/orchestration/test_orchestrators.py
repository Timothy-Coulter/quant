"""Unit tests for the orchestration strategies."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from backtester.core.event_bus import Event, EventBus, EventFilter
from backtester.core.events import MarketDataEvent, SignalEvent, create_market_data_event
from backtester.strategy.orchestration import (
    BaseStrategyOrchestrator,
    ConflictResolutionStrategy,
    CoordinationRule,
    CoordinationRuleType,
    OrchestrationConfig,
    OrchestratorType,
    StrategyKind,
    StrategyReference,
    StrategyRole,
)


class StaticStrategy:
    """Test helper returning predetermined signals."""

    def __init__(self, name: str, signals: list[dict[str, float | str]]) -> None:
        """Initialise the static strategy stub."""
        self.name = name
        self._signals = signals
        self.call_count = 0

    def generate_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, float | str]]:
        """Return the preconfigured list of signals."""
        self.call_count += 1
        return self._signals


def _build_market_event(symbol: str, data: pd.DataFrame) -> MarketDataEvent:
    """Create a market data event wrapping the supplied frame."""
    row = data.iloc[0]
    payload = {
        "open": float(row.get("Open", row.get("open"))),
        "high": float(row.get("High", row.get("high"))),
        "low": float(row.get("Low", row.get("low"))),
        "close": float(row.get("Close", row.get("close"))),
        "volume": float(row.get("Volume", row.get("volume", 0.0))),
        "timestamp": data.index[0].timestamp() if hasattr(data.index[0], "timestamp") else None,
        "data_type": "bar",
    }
    event = create_market_data_event(symbol, payload)
    event.metadata["data_frame"] = data.copy()
    return event


def _base_data() -> pd.DataFrame:
    """Return a minimal one-row OHLCV frame for tests."""
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000.0],
        },
        index=[pd.Timestamp("2024-01-01 09:30")],
    )


def test_sequential_orchestrator_respects_priority_and_dependencies() -> None:
    """Sequential orchestrator honours dependency ordering before execution."""
    event_bus = EventBus()
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.SEQUENTIAL,
        conflict_resolution=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
        strategies=[
            StrategyReference(identifier="alpha", priority=0),
            StrategyReference(identifier="beta", priority=1, depends_on=["alpha"]),
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=event_bus)

    alpha = StaticStrategy("alpha", [{"signal_type": "BUY", "confidence": 0.8}])
    beta = StaticStrategy("beta", [{"signal_type": "SELL", "confidence": 0.6}])
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL, priority=0)
    orchestrator.register_strategy("beta", beta, kind=StrategyKind.SIGNAL, priority=1)

    captured: list[str] = []

    def _capture(event: Event) -> None:
        signal_event = cast(SignalEvent, event)
        captured.append(signal_event.signal_type.value)

    event_bus.subscribe(_capture, EventFilter(event_types={"SIGNAL"}))

    data = _base_data()
    result = orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert alpha.call_count == 1
    assert beta.call_count == 1
    assert len(result.signals) == 1
    assert result.signals[0].signal_type == "BUY"
    assert captured == ["BUY"]


def test_parallel_orchestrator_combines_signals_weighted_merge() -> None:
    """Parallel orchestrator merges signals into a single weighted result."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.PARALLEL,
        conflict_resolution=ConflictResolutionStrategy.WEIGHTED_MERGE,
        strategies=[
            StrategyReference(identifier="alpha", priority=0, weight=1.0),
            StrategyReference(identifier="gamma", priority=1, weight=2.0),
        ],
    )
    event_bus = EventBus()
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=event_bus)

    alpha = StaticStrategy("alpha", [{"signal_type": "BUY", "confidence": 0.6}])
    gamma = StaticStrategy("gamma", [{"signal_type": "BUY", "confidence": 0.9}])
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL, weight=1.0)
    orchestrator.register_strategy("gamma", gamma, kind=StrategyKind.SIGNAL, weight=2.0)

    data = _base_data()
    result = orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert len(result.signals) == 1
    merged_signal = result.signals[0]
    assert merged_signal.signal_type == "BUY"
    aggregation = merged_signal.payload.get("aggregation")
    assert isinstance(aggregation, dict)
    assert aggregation.get("total_weight") == pytest.approx(3.0)


def test_master_slave_prevents_slave_when_master_hold() -> None:
    """Slave strategies remain idle when the master emits only hold signals."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.MASTER_SLAVE,
        strategies=[
            StrategyReference(identifier="master", role=StrategyRole.MASTER),
            StrategyReference(identifier="slave", role=StrategyRole.SLAVE),
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=EventBus())

    master = StaticStrategy("master", [{"signal_type": "HOLD", "confidence": 0.4}])
    slave = StaticStrategy("slave", [{"signal_type": "BUY", "confidence": 0.7}])
    orchestrator.register_strategy("master", master, kind=StrategyKind.SIGNAL)
    orchestrator.register_strategy("slave", slave, kind=StrategyKind.SIGNAL)

    data = _base_data()
    result = orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert master.call_count == 1
    assert slave.call_count == 0
    assert all(signal.strategy_id == "master" for signal in result.signals)


def test_master_slave_allows_slave_when_master_confident() -> None:
    """Slave strategies execute once the master broadcasts a confident signal."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.MASTER_SLAVE,
        conflict_resolution=ConflictResolutionStrategy.CONSENSUS,
        strategies=[
            StrategyReference(identifier="master", role=StrategyRole.MASTER),
            StrategyReference(identifier="slave", role=StrategyRole.SLAVE),
        ],
        metadata={"master_threshold": 0.5},
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=EventBus())

    master = StaticStrategy("master", [{"signal_type": "BUY", "confidence": 0.8}])
    slave = StaticStrategy("slave", [{"signal_type": "BUY", "confidence": 0.7}])
    orchestrator.register_strategy("master", master, kind=StrategyKind.SIGNAL)
    orchestrator.register_strategy("slave", slave, kind=StrategyKind.SIGNAL)

    data = _base_data()
    result = orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert master.call_count == 1
    assert slave.call_count == 1
    assert len(result.signals) == 1
    assert result.signals[0].signal_type == "BUY"


def test_conditional_trigger_executes_secondary() -> None:
    """Trigger rules allow secondaries to execute after threshold signals."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.CONDITIONAL,
        conflict_resolution=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
        strategies=[
            StrategyReference(identifier="alpha"),
            StrategyReference(identifier="beta"),
        ],
        coordination_rules=[
            CoordinationRule(
                rule_type=CoordinationRuleType.TRIGGER,
                primary="alpha",
                secondary=["beta"],
                parameters={"min_confidence": 0.6},
            )
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=EventBus())

    alpha = StaticStrategy("alpha", [{"signal_type": "BUY", "confidence": 0.7}])
    beta = StaticStrategy("beta", [{"signal_type": "SELL", "confidence": 0.65}])
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL)
    orchestrator.register_strategy("beta", beta, kind=StrategyKind.SIGNAL)

    data = _base_data()
    orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert alpha.call_count == 1
    assert beta.call_count == 1


def test_conditional_fallback_runs_secondary_when_primary_inactive() -> None:
    """Fallback rules activate secondary strategies when primaries are idle."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.CONDITIONAL,
        strategies=[
            StrategyReference(identifier="alpha"),
            StrategyReference(identifier="delta"),
        ],
        coordination_rules=[
            CoordinationRule(
                rule_type=CoordinationRuleType.FALLBACK,
                primary="alpha",
                secondary=["delta"],
            )
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=EventBus())

    alpha = StaticStrategy("alpha", [{"signal_type": "HOLD", "confidence": 0.2}])
    delta = StaticStrategy("delta", [{"signal_type": "BUY", "confidence": 0.5}])
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL)
    orchestrator.register_strategy("delta", delta, kind=StrategyKind.SIGNAL)

    data = _base_data()
    orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert alpha.call_count == 1
    assert delta.call_count == 1


def test_ensemble_orchestrator_prefers_heaviest_weight() -> None:
    """Ensemble orchestrator favours the signal with the highest aggregate weight."""
    config = OrchestrationConfig(
        orchestrator_type=OrchestratorType.ENSEMBLE,
        conflict_resolution=ConflictResolutionStrategy.WEIGHTED_MERGE,
        strategies=[
            StrategyReference(identifier="alpha", weight=1.0),
            StrategyReference(identifier="beta", weight=3.0),
        ],
    )
    orchestrator = BaseStrategyOrchestrator.create(config=config, event_bus=EventBus())

    alpha = StaticStrategy("alpha", [{"signal_type": "BUY", "confidence": 0.9}])
    beta = StaticStrategy("beta", [{"signal_type": "SELL", "confidence": 0.7}])
    orchestrator.register_strategy("alpha", alpha, kind=StrategyKind.SIGNAL, weight=1.0)
    orchestrator.register_strategy("beta", beta, kind=StrategyKind.SIGNAL, weight=3.0)

    data = _base_data()
    result = orchestrator.on_market_data(_build_market_event("SPY", data), data)

    assert len(result.signals) == 1
    merged = result.signals[0]
    assert merged.signal_type == "SELL"
    aggregation = merged.payload.get("aggregation")
    assert isinstance(aggregation, dict)
    assert aggregation.get("total_weight") == pytest.approx(3.0)
