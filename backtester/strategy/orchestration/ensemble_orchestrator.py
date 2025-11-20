"""Ensemble orchestration strategy implementation."""

from __future__ import annotations

import pandas as pd

from backtester.core.events import MarketDataEvent

from .base_orchestration import BaseStrategyOrchestrator, StrategySignal


class EnsembleStrategyOrchestrator(BaseStrategyOrchestrator):
    """Combine strategy outputs into a single ensemble signal."""

    def _coordinate_signals(
        self,
        event: MarketDataEvent,
        data: pd.DataFrame,
    ) -> list[StrategySignal]:
        component_signals: list[StrategySignal] = []

        for registration in self.get_registered_strategies():
            if not registration.enabled:
                continue
            produced = self._invoke_strategy(registration, data, event.symbol)
            component_signals.extend(produced)

        merged = self._merge_weighted_signals(component_signals)
        return [merged] if merged is not None else []
