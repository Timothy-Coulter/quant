"""Factory helpers for assembling strategy configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backtester.core.config import PortfolioConfig, StrategyConfig
from backtester.strategy.portfolio.portfolio_strategy_config import (
    AllocationMethod,
    PortfolioStrategyConfig,
    PortfolioStrategyType,
)
from backtester.strategy.signal.signal_strategy_config import MomentumStrategyConfig


def build_momentum_strategy_config(
    base_strategy: StrategyConfig | None,
    *,
    symbols: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
) -> MomentumStrategyConfig:
    """Create a MomentumStrategyConfig from core strategy config plus overrides."""
    normalized_symbols = _normalize_symbols(symbols)
    strategy_name = (
        base_strategy.strategy_name
        if base_strategy and base_strategy.strategy_name
        else "momentum_strategy"
    )

    payload: dict[str, Any] = {
        "name": strategy_name,
        "strategy_name": strategy_name,
        "symbols": normalized_symbols,
    }

    if base_strategy is not None:
        strategy_data = base_strategy.model_dump(mode="python")
        for field in MomentumStrategyConfig.model_fields:
            if field in strategy_data and strategy_data[field] is not None:
                payload[field] = strategy_data[field]

    if overrides:
        for key, value in overrides.items():
            if key in MomentumStrategyConfig.model_fields and value is not None:
                payload[key] = value

    return MomentumStrategyConfig(**payload)


def build_portfolio_strategy_config(
    portfolio_config: PortfolioConfig | None,
    *,
    symbols: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
) -> PortfolioStrategyConfig:
    """Create a PortfolioStrategyConfig for Kelly-style allocation."""
    normalized_symbols = _normalize_symbols(symbols)
    base_name = (
        portfolio_config.portfolio_strategy_name
        if portfolio_config and portfolio_config.portfolio_strategy_name
        else "kelly_criterion"
    )

    payload: dict[str, Any] = {
        "strategy_name": base_name,
        "strategy_type": PortfolioStrategyType.KELLY_CRITERION,
        "symbols": normalized_symbols,
        "allocation_method": AllocationMethod.KELLY_CRITERION,
    }

    if portfolio_config is not None:
        config_data = portfolio_config.model_dump(mode="python")
        for field in PortfolioStrategyConfig.model_fields:
            if field in config_data and config_data[field] is not None:
                payload[field] = config_data[field]

    if overrides:
        for key, value in overrides.items():
            if key in PortfolioStrategyConfig.model_fields and value is not None:
                payload[key] = value

    return PortfolioStrategyConfig(**payload)


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for symbol in symbols:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            continue
        normalized.append(normalized_symbol)
    return normalized or ["SPY"]
