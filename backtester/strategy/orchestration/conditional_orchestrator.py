"""Conditional orchestration strategy implementation."""

from __future__ import annotations

import pandas as pd

from backtester.core.events import MarketDataEvent

from .base_orchestration import BaseStrategyOrchestrator, StrategyRegistration, StrategySignal
from .orchestration_strategy_config import CoordinationRule, CoordinationRuleType


class ConditionalStrategyOrchestrator(BaseStrategyOrchestrator):
    """Execute strategies based on coordination rules and run-time signals."""

    def _coordinate_signals(
        self,
        event: MarketDataEvent,
        data: pd.DataFrame,
    ) -> list[StrategySignal]:
        executed: set[str] = set()
        aggregated: list[StrategySignal] = []

        for rule in self.config.coordination_rules:
            primary_registration = self._get_registration(rule.primary)
            if primary_registration is None or not primary_registration.enabled:
                continue

            primary_signals = self._invoke_strategy(primary_registration, data, event.symbol)
            aggregated.extend(primary_signals)
            executed.add(primary_registration.identifier)

            if self._should_execute_secondary(rule, primary_signals):
                for secondary_id in rule.secondary:
                    secondary_registration = self._get_registration(secondary_id)
                    if (
                        secondary_registration is None
                        or not secondary_registration.enabled
                        or secondary_registration.identifier in executed
                    ):
                        continue
                    secondary_signals = self._invoke_strategy(
                        secondary_registration, data, event.symbol
                    )
                    aggregated.extend(secondary_signals)
                    executed.add(secondary_registration.identifier)

        for registration in self.get_registered_strategies():
            if not registration.enabled:
                continue
            if registration.identifier in executed:
                continue
            aggregated.extend(self._invoke_strategy(registration, data, event.symbol))

        return aggregated

    def _get_registration(self, identifier: str) -> StrategyRegistration | None:
        for registration in self.get_registered_strategies():
            if registration.identifier == identifier:
                return registration
        return None

    def _should_execute_secondary(
        self,
        rule: CoordinationRule,
        signals: list[StrategySignal],
    ) -> bool:
        if not signals:
            return rule.rule_type == CoordinationRuleType.FALLBACK

        if rule.rule_type == CoordinationRuleType.FALLBACK:
            actionable = any(signal.signal_type != "HOLD" for signal in signals)
            return not actionable

        monitored_types = set(
            str(value).upper() for value in rule.parameters.get("signal_types", ["BUY", "SELL"])
        )
        min_confidence = float(rule.parameters.get("min_confidence", 0.5))

        for signal in signals:
            if signal.signal_type not in monitored_types:
                continue
            if signal.confidence >= min_confidence:
                return True
        return False
