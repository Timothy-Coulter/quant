"""Sequential orchestration strategy implementation."""

from __future__ import annotations

import pandas as pd

from backtester.core.events import MarketDataEvent

from .base_orchestration import BaseStrategyOrchestrator, StrategyRegistration, StrategySignal
from .orchestration_strategy_config import CoordinationRuleType


class SequentialStrategyOrchestrator(BaseStrategyOrchestrator):
    """Run strategies sequentially, honouring explicit dependencies and priorities."""

    def _coordinate_signals(
        self,
        event: MarketDataEvent,
        data: pd.DataFrame,
    ) -> list[StrategySignal]:
        ordered = sorted(
            self.get_registered_strategies(),
            key=lambda registration: registration.priority,
        )

        produced: dict[str, list[StrategySignal]] = {}
        aggregated: list[StrategySignal] = []

        for registration in ordered:
            if not registration.enabled:
                continue

            if not self._dependencies_satisfied(registration, produced):
                self.logger.debug(
                    "Skipping %s as dependencies are not yet satisfied.", registration.identifier
                )
                continue

            signals = self._invoke_strategy(registration, data, event.symbol)
            if signals:
                produced[registration.identifier] = signals
                aggregated.extend(signals)

            self._apply_sequence_rules(registration.identifier, produced)

        return aggregated

    def _dependencies_satisfied(
        self,
        registration: StrategyRegistration,
        produced: dict[str, list[StrategySignal]],
    ) -> bool:
        reference = self.config.get_strategy(registration.identifier)
        if reference is None or not reference.depends_on:
            return True

        for dependency in reference.depends_on:
            signals = produced.get(dependency)
            if not signals:
                return False
            if all(signal.signal_type == "HOLD" for signal in signals):
                return False
        return True

    def _apply_sequence_rules(
        self,
        identifier: str,
        produced: dict[str, list[StrategySignal]],
    ) -> None:
        if not self.config.coordination_rules:
            return

        for rule in self.config.coordination_rules:
            if rule.rule_type != CoordinationRuleType.SEQUENCE:
                continue
            if rule.primary != identifier:
                continue

            if identifier not in produced:
                continue

            for secondary in rule.secondary:
                # Mark dependencies as satisfied explicitly by inserting an empty list.
                produced.setdefault(secondary, [])
