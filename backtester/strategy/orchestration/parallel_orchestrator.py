"""Parallel orchestration strategy implementation."""

from __future__ import annotations

import pandas as pd

from backtester.core.events import MarketDataEvent

from .base_orchestration import BaseStrategyOrchestrator, StrategySignal


class ParallelStrategyOrchestrator(BaseStrategyOrchestrator):
    """Execute all registered strategies for each market data event."""

    def _coordinate_signals(
        self,
        event: MarketDataEvent,
        data: pd.DataFrame,
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []

        for registration in self.get_registered_strategies():
            if not registration.enabled:
                continue

            produced = self._invoke_strategy(registration, data, event.symbol)
            if produced:
                signals.extend(produced)

        return signals
