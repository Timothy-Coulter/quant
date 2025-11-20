"""Master-slave orchestration strategy implementation."""

from __future__ import annotations

import pandas as pd

from backtester.core.events import MarketDataEvent

from .base_orchestration import BaseStrategyOrchestrator, StrategyRegistration, StrategySignal
from .orchestration_strategy_config import StrategyRole


class MasterSlaveStrategyOrchestrator(BaseStrategyOrchestrator):
    """Primary (master) strategy gates the execution of dependent slave strategies."""

    MIN_DEFAULT_CONFIDENCE = 0.1

    def _coordinate_signals(
        self,
        event: MarketDataEvent,
        data: pd.DataFrame,
    ) -> list[StrategySignal]:
        master = self._select_master()
        if master is None:
            return list(self._collect_all(event, data))

        master_signals = self._invoke_strategy(master, data, event.symbol)
        if not master_signals:
            return master_signals

        actionable = self._select_actionable(master_signals)
        if actionable is None:
            return master_signals

        slave_signals: list[StrategySignal] = []
        desired_type = actionable.signal_type

        for registration in self.get_registered_strategies():
            if registration.identifier == master.identifier:
                continue
            if registration.role not in (StrategyRole.SLAVE, StrategyRole.AUXILIARY):
                continue
            if not registration.enabled:
                continue

            produced = self._invoke_strategy(registration, data, event.symbol)
            aligned = [
                signal
                for signal in produced
                if signal.signal_type == desired_type or desired_type == "HOLD"
            ]
            slave_signals.extend(aligned)

        return master_signals + slave_signals

    def _select_master(self) -> StrategyRegistration | None:
        masters = [
            reg for reg in self.get_registered_strategies() if reg.role == StrategyRole.MASTER
        ]
        if masters:
            return sorted(masters, key=lambda reg: reg.priority)[0]

        registered = self.get_registered_strategies()
        if not registered:
            return None
        return sorted(registered, key=lambda reg: reg.priority)[0]

    def _select_actionable(self, signals: list[StrategySignal]) -> StrategySignal | None:
        threshold = float(self.config.metadata.get("master_threshold", self.MIN_DEFAULT_CONFIDENCE))
        for signal in signals:
            if signal.signal_type == "HOLD":
                continue
            if signal.confidence >= threshold:
                return signal
        return None

    def _collect_all(self, event: MarketDataEvent, data: pd.DataFrame) -> list[StrategySignal]:
        aggregated: list[StrategySignal] = []
        for registration in self.get_registered_strategies():
            if not registration.enabled:
                continue
            aggregated.extend(self._invoke_strategy(registration, data, event.symbol))
        return aggregated
