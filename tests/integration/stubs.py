"""Test helpers for wiring deterministic engine components."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from backtester.core.events import create_portfolio_update_event
from backtester.execution.order import OrderStatus


@dataclass
class StubPosition:
    """Simplified position representation used by the integration stubs."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float


@dataclass
class StubOrder:
    """Minimal order object returned by the stub broker."""

    order_id: str
    symbol: str
    side: str
    filled_quantity: float
    filled_price: float
    status: OrderStatus = OrderStatus.FILLED
    is_active: bool = False


class StubPortfolio:
    """Deterministic portfolio implementation that mirrors the GeneralPortfolio API."""

    def __init__(
        self,
        *,
        event_bus: Any,
        initial_capital: float = 100.0,
        commission_rate: float = 0.0,
        portfolio_id: str = "stub_portfolio",
    ) -> None:
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.portfolio_id = portfolio_id
        self.max_positions = 10

        self.cash = initial_capital
        self.positions: dict[str, StubPosition] = {}
        self.portfolio_values: list[float] = [initial_capital]
        self.trade_log: list[dict[str, Any]] = []
        self.cumulative_tax = 0.0

        self.before_run_calls = 0
        self.after_run_calls = 0
        self.before_tick_calls = 0
        self.after_tick_calls = 0

    @property
    def total_value(self) -> float:
        """Current portfolio value."""
        position_value = sum(p.quantity * p.current_price for p in self.positions.values())
        return self.cash + position_value

    def reset(self) -> None:
        """Reset internal state between backtests."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.portfolio_values = [self.initial_capital]
        self.trade_log.clear()
        self.cumulative_tax = 0.0

    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.before_run_calls += 1

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.after_run_calls += 1

    def before_tick(self, context: dict[str, Any]) -> None:
        self.before_tick_calls += 1

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        self.after_tick_calls += 1

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
        """Apply executed order information to the stub state."""
        metadata = metadata or {}
        position = self.positions.get(symbol)
        if side.upper() == "BUY":
            if position is None:
                position = StubPosition(
                    symbol=symbol, quantity=0.0, avg_price=price, current_price=price
                )
                self.positions[symbol] = position
            total_qty = position.quantity + quantity
            if total_qty <= 0:
                return
            position.avg_price = (
                (position.avg_price * position.quantity) + (price * quantity)
            ) / total_qty
            position.quantity = total_qty
            position.current_price = price
        else:
            if position is None:
                return
            position.quantity -= quantity
            position.current_price = price
            if position.quantity <= 0:
                del self.positions[symbol]

        self.trade_log.append(
            {
                "symbol": symbol,
                "side": side.upper(),
                "quantity": quantity,
                "price": price,
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )

    def process_tick(
        self,
        *,
        timestamp: Any,
        market_data: dict[str, pd.DataFrame] | None = None,
        current_price: float | None = None,
        day_high: float | None = None,
        day_low: float | None = None,
    ) -> dict[str, Any]:
        """Update valuations and broadcast a portfolio event."""
        position_updates: list[dict[str, Any]] = []
        if market_data:
            for symbol, frame in market_data.items():
                close = self._extract_close(frame)
                if close is None:
                    continue
                if symbol in self.positions:
                    self.positions[symbol].current_price = close
                    position_updates.append(
                        {
                            "symbol": symbol,
                            "market_price": close,
                            "quantity": self.positions[symbol].quantity,
                        }
                    )

        total_value = self.total_value
        self.portfolio_values.append(total_value)
        event = create_portfolio_update_event(
            portfolio_id=self.portfolio_id,
            total_value=total_value,
            cash_balance=self.cash,
            positions_value=total_value - self.cash,
            metadata={"position_updates": position_updates},
        )
        self.event_bus.publish(event, immediate=True)
        return {
            "timestamp": timestamp,
            "total_value": total_value,
            "position_updates": position_updates,
            "cash": self.cash,
            "position_count": len(self.positions),
            "day_high": day_high,
            "day_low": day_low,
            "financing_cost": 0.0,
        }

    @staticmethod
    def _extract_close(frame: pd.DataFrame) -> float | None:
        if frame is None or frame.empty:
            return None
        lower = {c.lower(): c for c in frame.columns}
        close_col = lower.get("close")
        if close_col:
            return float(frame[close_col].iloc[-1])
        return float(frame.iloc[-1].values[0])


class StubBroker:
    """Broker stub that fills orders immediately at the requested price."""

    def __init__(self) -> None:
        self.positions: dict[str, float] = {}
        self.current_prices: dict[str, float] = {}
        self.market_data: dict[str, pd.DataFrame] = {}
        self.order_manager: MagicMock = MagicMock()
        self.before_run_calls = 0
        self.after_run_calls = 0
        self.before_tick_calls = 0
        self.after_tick_calls = 0
        self._order_seq = 0
        self.submitted_orders: list[StubOrder] = []

    def reset(self) -> None:
        self.positions.clear()
        self.current_prices.clear()
        self.market_data.clear()
        self.submitted_orders.clear()
        self.order_manager.reset_mock()

    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.before_run_calls += 1

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.after_run_calls += 1

    def before_tick(self, context: dict[str, Any]) -> None:
        self.before_tick_calls += 1

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        self.after_tick_calls += 1

    def set_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        self.market_data[symbol] = data.copy()
        if "Close" in data.columns:
            self.current_prices[symbol] = float(data["Close"].iloc[-1])

    def get_current_price(self, symbol: str) -> float:
        return self.current_prices.get(symbol, 0.0)

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StubOrder:
        """Fill the order immediately and update broker positions."""
        metadata = metadata or {}
        self._order_seq += 1
        fill_price = price if price is not None else self.current_prices.get(symbol, 100.0)
        order = StubOrder(
            order_id=f"stub_order_{self._order_seq}",
            symbol=symbol,
            side=side.upper(),
            filled_quantity=quantity,
            filled_price=fill_price,
        )
        delta = quantity if side.upper() == "BUY" else -quantity
        self.positions[symbol] = self.positions.get(symbol, 0.0) + delta
        self.current_prices[symbol] = fill_price
        self.submitted_orders.append(order)
        return order


class StubRiskManager:
    """Risk manager stub that can be configured to emit alerts after N evaluations."""

    def __init__(
        self,
        *,
        alert_after: int = 10,
        alert_level: str = "HIGH",
        persist_after_trigger: bool = False,
    ) -> None:
        self.alert_after: int | None = alert_after
        self.alert_level = alert_level
        self.persist_after_trigger = persist_after_trigger
        self.call_count = 0
        self.risk_signals: list[dict[str, Any]] = []
        self.recorded_orders: list[dict[str, Any]] = []
        self.open_position_requests: list[dict[str, Any]] = []
        self.allow_new_positions = True

    def check_portfolio_risk(
        self,
        portfolio_value: float,
        positions: dict[str, Any],
    ) -> dict[str, Any]:
        self.call_count += 1
        trigger = self.alert_after is not None and self.call_count >= self.alert_after
        violations: list[str]
        if trigger:
            risk_level = self.alert_level
            violations = ["MAX_DRAWDOWN"]
            if not self.persist_after_trigger:
                self.alert_after = None
        else:
            risk_level = "LOW"
            violations = []

        return {
            "risk_level": risk_level,
            "violations": violations,
            "positions": positions,
            "portfolio_value": portfolio_value,
        }

    def add_risk_signal(self, signal: dict[str, Any]) -> None:
        self.risk_signals.append(signal)

    def can_open_position(
        self,
        symbol: str,
        notional: float,
        portfolio_value: float,
        metadata: dict[str, Any],
    ) -> bool:
        request = {
            "symbol": symbol,
            "notional": notional,
            "portfolio_value": portfolio_value,
            "metadata": metadata,
        }
        self.open_position_requests.append(request)
        return self.allow_new_positions

    def record_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.recorded_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "metadata": metadata or {},
            }
        )


def install_engine_stubs(
    engine: Any,
    *,
    risk_alert_after: int = 10,
    disable_portfolio_strategy: bool = True,
) -> tuple[StubPortfolio, StubBroker, StubRiskManager]:
    """Replace engine factories with deterministic stubs for integration tests."""
    stub_portfolio = StubPortfolio(event_bus=engine.event_bus)
    stub_broker = StubBroker()
    stub_risk = StubRiskManager(alert_after=risk_alert_after)

    def fake_create_portfolio(
        self: Any, portfolio_params: dict[str, Any] | None = None
    ) -> StubPortfolio:
        self.current_portfolio = stub_portfolio
        return stub_portfolio

    def fake_create_broker(self: Any) -> StubBroker:
        self.current_broker = stub_broker
        return stub_broker

    def fake_create_risk_manager(self: Any) -> StubRiskManager:
        self.current_risk_manager = stub_risk
        return stub_risk

    engine.create_portfolio = types.MethodType(fake_create_portfolio, engine)
    engine.create_broker = types.MethodType(fake_create_broker, engine)
    engine.create_risk_manager = types.MethodType(fake_create_risk_manager, engine)

    if disable_portfolio_strategy:

        def _skip_portfolio_strategy(
            self: Any, symbols: list[str], portfolio_params: dict[str, Any]
        ) -> None:
            self.portfolio_strategy = None

        engine._initialize_portfolio_strategy = types.MethodType(_skip_portfolio_strategy, engine)

    return stub_portfolio, stub_broker, stub_risk


def capture_processed_signals(engine: Any, storage: list[list[dict[str, Any]]]) -> None:
    """Wrap the engine's signal processing method so each invocation is recorded."""
    original = engine._process_signals_and_update_portfolio

    def recorder(
        self: Any,
        signals: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        storage.append([dict(signal) for signal in signals])
        return original(signals, *args, **kwargs)

    engine._process_signals_and_update_portfolio = types.MethodType(recorder, engine)
