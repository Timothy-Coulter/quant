"""Shared Protocol definitions used to decouple core components.

The backtesting engine coordinates strategies, brokers, portfolios, and
risk managers.  These Protocols capture the minimal contracts that the
engine relies upon so components can be swapped more easily in tests or
future refactors.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyProtocol(Protocol):
    """Minimal interface required from signal strategies."""

    name: str

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        """Produce zero or more signal dictionaries for ``symbol``."""

    def reset(self) -> None:
        """Reset internal strategy state."""


@runtime_checkable
class PortfolioProtocol(Protocol):
    """Portfolio operations consumed by the engine."""

    initial_capital: float
    portfolio_values: list[float]

    def process_tick(
        self,
        *,
        timestamp: Any,
        market_data: dict[str, pd.DataFrame] | None = None,
        current_price: float | None = None,
        day_high: float | None = None,
        day_low: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update internal state for the latest market data snapshot."""

    def apply_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update portfolio holdings based on an executed order fill."""

    def reset(self) -> None:
        """Return the portfolio to its initial state."""


@runtime_checkable
class BrokerProtocol(Protocol):
    """Methods that the engine expects every broker implementation to expose."""

    cash_balance: float
    positions: dict[str, float]

    def set_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Provide the broker with historical data for ``symbol``."""

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create and execute an order returning the underlying order object."""

    def reset(self) -> None:
        """Return broker state to the initial conditions."""


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Risk manager helper hooks used by callers across the system."""

    def can_open_position(
        self,
        symbol: str,
        notional: float,
        portfolio_value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Return ``True`` if the requested exposure is within configured limits."""

    def record_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Capture metadata about a submitted order for auditability."""

    def record_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update tracked exposures after an order fill."""

    def check_portfolio_risk(
        self, portfolio_value: float, positions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate current portfolio risk and return diagnostic information."""

    def add_risk_signal(self, signal: dict[str, Any]) -> None:
        """Publish or log a risk related signal."""
