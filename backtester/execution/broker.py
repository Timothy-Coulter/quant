"""Simulated Broker for Order Execution.

This module provides a simulated broker that handles order execution, market data,
commission calculations, and trade reporting.
"""

import logging
import random
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from backtester.core.config import ExecutionConfig, SimulatedBrokerConfig
from backtester.core.config_processor import (
    ConfigProcessor,
    ConfigValidationError,
)
from backtester.core.event_bus import EventBus, EventPriority
from backtester.core.events import OrderEvent as BusOrderEvent
from backtester.core.events import OrderSide as EventOrderSide
from backtester.core.events import OrderStatus as EventOrderStatus
from backtester.core.events import OrderType as EventOrderType
from backtester.core.interfaces import RiskManagerProtocol
from backtester.execution.order import OrderData, OrderManager, OrderSide, OrderType


class SimulatedBroker:
    """Simulated broker for order execution and trade reporting."""

    _VALID_SLIPPAGE_MODELS = {'normal', 'fixed', 'none'}

    def __init__(
        self,
        config: (
            SimulatedBrokerConfig | ExecutionConfig | Mapping[str, Any] | str | bytes | Path | None
        ) = None,
        *,
        config_overrides: Mapping[str, Any] | None = None,
        config_processor: ConfigProcessor | None = None,
        commission_rate: float | None = None,
        min_commission: float | None = None,
        spread: float | None = None,
        slippage_model: str | None = None,
        slippage_std: float | None = None,
        latency_ms: float | None = None,
        logger: logging.Logger | None = None,
        event_bus: EventBus | None = None,
        risk_manager: RiskManagerProtocol | None = None,
        initial_cash: float | None = None,
    ) -> None:
        """Initialize the simulated broker.

        Args:
            config: Execution configuration model, mapping, or YAML path.
            config_overrides: Mapping of overrides merged on top of ``config``.
            config_processor: Optional shared ConfigProcessor instance.
            commission_rate: Commission rate override for backward compatibility.
            min_commission: Minimum commission override for backward compatibility.
            spread: Bid-ask spread override for backward compatibility.
            slippage_model: Slippage model override for backward compatibility.
            slippage_std: Slippage std override for backward compatibility.
            latency_ms: Latency override for backward compatibility.
            logger: Optional logger instance
            event_bus: Optional event bus for publishing order execution events
            risk_manager: Optional risk manager used for additional risk checks
            initial_cash: Initial cash balance assigned to the broker
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.event_bus = event_bus
        self.risk_manager = risk_manager

        self._config_processor = config_processor or ConfigProcessor()
        merged_overrides = self._collect_config_overrides(
            config_overrides=config_overrides,
            commission_rate=commission_rate,
            min_commission=min_commission,
            spread=spread,
            slippage_model=slippage_model,
            slippage_std=slippage_std,
            latency_ms=latency_ms,
        )
        resolved_config = self._resolve_config(config=config, overrides=merged_overrides)

        self._config = resolved_config

        self.commission_rate = resolved_config.commission_rate
        self.min_commission = resolved_config.min_commission
        self.spread = resolved_config.spread
        self.slippage_model = resolved_config.slippage_model
        self.slippage_distribution = resolved_config.slippage_distribution
        self.slippage_std = resolved_config.slippage_std
        self.latency_ms = resolved_config.latency_ms
        self.latency_jitter_ms = resolved_config.latency_jitter_ms
        self.max_orders_per_minute = resolved_config.max_orders_per_minute
        self.order_cooldown_seconds = resolved_config.order_cooldown_seconds

        # State
        self.order_manager = OrderManager(logger, config=resolved_config)
        self.market_data: dict[str, pd.DataFrame] = {}
        self.current_prices: dict[str, float] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.cash_balance: float = initial_cash or 0.0
        self.positions: dict[str, float] = {}
        self.portfolio_value: float = self.cash_balance
        self._order_timestamps: deque[float] = deque()
        self._last_order_timestamp: float | None = None

        self.logger.info("Simulated broker initialized")

    @classmethod
    def default_config(cls) -> SimulatedBrokerConfig:
        """Return the default execution configuration for the broker."""
        return SimulatedBrokerConfig()

    @staticmethod
    def _collect_config_overrides(
        *,
        config_overrides: Mapping[str, Any] | None,
        commission_rate: float | None,
        min_commission: float | None,
        spread: float | None,
        slippage_model: str | None,
        slippage_std: float | None,
        latency_ms: float | None,
    ) -> dict[str, Any]:
        """Merge legacy kwargs with explicit overrides into a single mapping."""
        overrides: dict[str, Any] = dict(config_overrides or {})
        legacy = {
            'commission_rate': commission_rate,
            'min_commission': min_commission,
            'spread': spread,
            'slippage_model': slippage_model,
            'slippage_std': slippage_std,
            'latency_ms': latency_ms,
        }
        for key, value in legacy.items():
            if value is not None:
                overrides[key] = value
        return overrides

    def _resolve_config(
        self,
        *,
        config: (
            SimulatedBrokerConfig | ExecutionConfig | Mapping[str, Any] | str | bytes | Path | None
        ),
        overrides: Mapping[str, Any] | None,
    ) -> SimulatedBrokerConfig:
        """Resolve the configuration precedence for the broker."""
        payload, distribution_explicit = self._collect_execution_payload(
            config=config,
            overrides=overrides,
        )

        model_value = payload.get('slippage_model')
        if model_value not in self._VALID_SLIPPAGE_MODELS:
            payload['slippage_model'] = 'none'
            model_value = 'none'

        if not distribution_explicit:
            payload['slippage_distribution'] = model_value

        try:
            return SimulatedBrokerConfig(**payload)
        except ValidationError as exc:
            raise ConfigValidationError(
                "Unable to resolve execution configuration",
                component='execution',
                errors=exc.errors(),
            ) from exc

    def _collect_execution_payload(
        self,
        *,
        config: (
            SimulatedBrokerConfig | ExecutionConfig | Mapping[str, Any] | str | bytes | Path | None
        ),
        overrides: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        """Merge config sources and note whether distribution overrides were supplied."""
        payload = self.default_config().model_dump(mode="python")
        distribution_explicit = False

        if isinstance(config, (SimulatedBrokerConfig, ExecutionConfig)):
            if getattr(config, 'slippage_distribution', None) is not None:
                distribution_explicit = True
            payload = ConfigProcessor._deep_merge(payload, config.model_dump(mode="python"))
        elif config is not None:
            component_payload = self._config_processor.load_component_payload('execution', config)
            if component_payload is not None:
                if 'slippage_distribution' in component_payload:
                    distribution_explicit = True
                payload = ConfigProcessor._deep_merge(payload, component_payload)

        if overrides:
            overrides_dict = dict(overrides)
            payload = ConfigProcessor._deep_merge(payload, overrides_dict)
            if 'slippage_distribution' in overrides_dict:
                distribution_explicit = True

        return payload, distribution_explicit

    def _simulate_latency(self) -> None:
        """Apply simulated latency with optional jitter without blocking excessively."""
        total_ms = self.latency_ms + random.uniform(0, self.latency_jitter_ms)
        if total_ms <= 0:
            return
        time.sleep(min(total_ms / 1000.0, 0.05))

    def _respect_throttles(self, order: OrderData) -> bool:
        """Enforce cooldown and per-minute order throttles."""
        now = time.time()

        if self.order_cooldown_seconds > 0 and self._last_order_timestamp is not None:
            elapsed = now - self._last_order_timestamp
            if elapsed < self.order_cooldown_seconds:
                order.reject("Order cooldown in effect")
                self._publish_order_event(
                    order,
                    EventOrderStatus.REJECTED,
                    message="Order cooldown in effect",
                )
                return False

        if self.max_orders_per_minute > 0:
            while self._order_timestamps and now - self._order_timestamps[0] > 60:
                self._order_timestamps.popleft()
            if len(self._order_timestamps) >= self.max_orders_per_minute:
                order.reject("Order rate limit exceeded")
                self._publish_order_event(
                    order,
                    EventOrderStatus.REJECTED,
                    message="Order rate limit exceeded",
                )
                return False
            self._order_timestamps.append(now)

        self._last_order_timestamp = now
        return True

    # ------------------------------------------------------------------#
    # Lifecycle hooks
    # ------------------------------------------------------------------#
    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked before the simulation loop starts."""
        return None

    def before_tick(self, context: dict[str, Any]) -> None:
        """Hook invoked before processing each tick."""
        return None

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        """Hook invoked after completing each tick."""
        return None

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked after the simulation loop finishes."""
        return None

    def submit_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderData:
        """Create and immediately attempt to execute an order."""
        side_enum = OrderSide[side.upper()]
        order_type_enum = OrderType[order_type.upper()]
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side_enum,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
            metadata=metadata,
        )

        if self.risk_manager:
            self.risk_manager.record_order(
                symbol,
                side_enum.value,
                quantity,
                price or 0.0,
                metadata,
            )

        self.execute_order(order)
        return order

    def set_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Set market data for a symbol.

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
        """
        self.market_data[symbol] = data.copy()
        if 'Close' in data.columns:
            self.current_prices[symbol] = data['Close'].iloc[-1]
        self.logger.info(f"Set market data for {symbol}: {len(data)} records")

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        return self.current_prices.get(symbol, 0.0)

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Get bid and ask prices for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (bid, ask) prices
        """
        price = self.get_current_price(symbol)
        spread_pct = self.spread / 2
        bid = price * (1 - spread_pct)
        ask = price * (1 + spread_pct)
        return bid, ask

    def execute_order(self, order: OrderData, market_data: pd.Series | None = None) -> bool:
        """Execute an order based on current market conditions.

        Args:
            order: Order to execute
            market_data: Optional current market data

        Returns:
            True if order was executed, False otherwise
        """
        if not order.is_active:
            return False

        if not self._respect_throttles(order):
            return False

        self._simulate_latency()

        # Get current market data
        current_price = self.get_current_price(order.symbol)
        if current_price == 0.0:
            order.reject("No market data available")
            self._publish_order_event(order, EventOrderStatus.REJECTED, message="No market data")
            return False

        bid, ask = self.get_bid_ask(order.symbol)

        # Execute based on order type
        execution_price = self._determine_execution_price(order, current_price, bid, ask)
        if execution_price is None:
            return False

        # Calculate quantity to fill
        fill_quantity = min(
            order.remaining_quantity, self._calculate_max_quantity(order, execution_price)
        )
        if fill_quantity <= 0:
            order.reject("Insufficient funds or position limits")
            self._publish_order_event(
                order,
                EventOrderStatus.REJECTED,
                message="Insufficient funds or position limits",
            )
            return False

        if self.risk_manager and order.is_buy:
            requested_notional = fill_quantity * execution_price
            portfolio_value = (
                self.portfolio_value if self.portfolio_value > 0 else self.cash_balance
            )
            if not self.risk_manager.can_open_position(
                order.symbol,
                requested_notional,
                portfolio_value,
                order.metadata,
            ):
                order.reject("Risk manager blocked order")
                self._publish_order_event(
                    order,
                    EventOrderStatus.REJECTED,
                    message="Risk manager blocked order",
                )
                return False

        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)

        # Update order with fill
        order.update_fill(fill_quantity, execution_price, commission)

        # Record trade
        trade_record = {
            'timestamp': order.timestamp,
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': fill_quantity,
            'price': execution_price,
            'commission': commission,
            'notional': fill_quantity * execution_price,
            'order_type': order.order_type.value,
        }
        self.trade_history.append(trade_record)

        # Update broker positions and cash
        self._update_positions_and_cash(order, fill_quantity, execution_price, commission)

        self.logger.info(
            f"Executed {order}: {fill_quantity}@{execution_price:.4f}, commission: ${commission:.2f}"
        )
        self._publish_order_event(
            order,
            EventOrderStatus(order.status.value),
            fill_quantity=fill_quantity,
            fill_price=execution_price,
            commission=commission,
        )

        if self.risk_manager:
            self.risk_manager.record_fill(
                order.symbol,
                order.side.value,
                fill_quantity,
                execution_price,
                self.portfolio_value,
                order.metadata,
            )
        return True

    def _publish_order_event(
        self,
        order: OrderData,
        status: EventOrderStatus,
        *,
        fill_quantity: float | None = None,
        fill_price: float | None = None,
        commission: float | None = None,
        message: str | None = None,
    ) -> None:
        """Publish an order event to the event bus for downstream consumers."""
        if self.event_bus is None:
            return

        try:
            metadata = dict(order.metadata or {})
            if fill_quantity is not None:
                metadata['fill_quantity'] = fill_quantity
            if fill_price is not None:
                metadata['fill_price'] = fill_price
            if commission is not None:
                metadata['commission'] = commission
            if message:
                metadata['message'] = message

            event = BusOrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="simulated_broker",
                symbol=order.symbol,
                order_id=order.order_id or "",
                side=EventOrderSide(order.side.value),
                order_type=EventOrderType(order.order_type.value),
                quantity=order.quantity,
                priority=EventPriority.HIGH,
                metadata=metadata,
                price=fill_price,
                status=status,
                filled_quantity=order.filled_quantity,
                average_fill_price=order.filled_price,
                commission=order.commission,
            )
            self.event_bus.publish(event, immediate=True)
        except Exception as exc:  # pragma: no cover - diagnostics only
            self.logger.debug("Failed to publish order event: %s", exc)

    def _determine_execution_price(
        self, order: OrderData, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Determine execution price based on order type and market conditions.

        Args:
            order: Order to execute
            market_price: Current market price
            bid: Current bid price
            ask: Current ask price

        Returns:
            Execution price or None if not executable
        """
        # Handle market orders
        if order.order_type == OrderType.MARKET:
            return self._handle_market_order(order, bid, ask)

        # Handle limit orders
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order, market_price, bid, ask)

        # Handle stop orders
        elif order.order_type == OrderType.STOP:
            return self._handle_stop_order(order, market_price, bid, ask)

        # Handle stop-limit orders
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._handle_stop_limit_order(order, market_price, bid, ask)

    def _handle_market_order(self, order: OrderData, bid: float, ask: float) -> float:
        """Handle market order execution price determination."""
        if order.is_buy:
            return ask + self._calculate_slippage(order, ask)
        else:
            return bid + self._calculate_slippage(order, bid)

    def _handle_limit_order(
        self, order: OrderData, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle limit order execution price determination."""
        if order.is_buy and market_price <= (order.price or 0):
            assert order.price is not None
            return min(order.price, ask) + self._calculate_slippage(order, ask)
        elif order.is_sell and market_price >= (order.price or 0):
            assert order.price is not None
            return max(order.price, bid) + self._calculate_slippage(order, bid)
        else:
            return None  # Price not favorable

    def _handle_stop_order(
        self, order: OrderData, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle stop order execution price determination."""
        assert order.stop_price is not None
        if (order.is_buy and market_price >= order.stop_price) or (
            order.is_sell and market_price <= order.stop_price
        ):
            if order.is_buy:
                return ask + self._calculate_slippage(order, ask)
            else:
                return bid + self._calculate_slippage(order, bid)
        else:
            return None  # Stop not triggered

    def _handle_stop_limit_order(
        self, order: OrderData, market_price: float, bid: float, ask: float
    ) -> float | None:
        """Handle stop-limit order execution price determination."""
        assert order.stop_price is not None
        assert order.price is not None
        if (order.is_buy and market_price >= order.stop_price) or (
            order.is_sell and market_price <= order.stop_price
        ):
            if order.is_buy and market_price <= order.price:
                return min(order.price, ask) + self._calculate_slippage(order, ask)
            elif order.is_sell and market_price >= order.price:
                return max(order.price, bid) + self._calculate_slippage(order, bid)
            else:
                return None  # Price not favorable for limit
        else:
            return None  # Stop not triggered

    def _calculate_slippage(self, order: OrderData, reference_price: float) -> float:
        """Calculate slippage for an order.

        Args:
            order: Order being executed
            reference_price: Reference price for slippage calculation

        Returns:
            Slippage amount
        """
        distribution = (self.slippage_distribution or self.slippage_model or "none").lower()
        if distribution == "none":
            return 0.0
        if distribution == "fixed":
            return reference_price * self.slippage_std
        if distribution == "normal":
            slippage = np.random.normal(0, self.slippage_std)
            return reference_price * slippage
        if distribution == "lognormal":
            slippage = np.random.lognormal(mean=0.0, sigma=self.slippage_std) - 1.0
            return reference_price * slippage

        return 0.0

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade.

        Args:
            quantity: Trade quantity
            price: Trade price

        Returns:
            Commission amount
        """
        notional_value = quantity * price
        commission = notional_value * self.commission_rate
        return max(commission, self.min_commission)

    def _calculate_max_quantity(self, order: OrderData, price: float) -> float:
        """Calculate maximum executable quantity based on available capital and position limits.

        Args:
            order: Order to assess
            price: Execution price

        Returns:
            Maximum executable quantity
        """
        # In a real implementation, this would check available cash and position limits
        # For simulation, we'll use a simple approach
        available_cash = max(0, self.cash_balance)
        max_quantity_by_cash = available_cash / price if price > 0 else 0

        # Position limits could be implemented here
        max_position_size = 1000000  # Placeholder limit

        current_position = self.positions.get(order.symbol, 0)
        max_additional = max_position_size - current_position if order.is_buy else current_position

        return min(order.remaining_quantity, max_quantity_by_cash, max_additional)

    def _update_positions_and_cash(
        self, order: OrderData, fill_quantity: float, execution_price: float, commission: float
    ) -> None:
        """Update broker positions and cash balance.

        Args:
            order: Executed order
            fill_quantity: Quantity filled
            execution_price: Execution price
            commission: Commission amount
        """
        # Update positions
        if order.is_buy:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + fill_quantity
            cash_change = -(fill_quantity * execution_price + commission)
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - fill_quantity
            cash_change = fill_quantity * execution_price - commission

        self.cash_balance += cash_change

        # Update portfolio value
        self.portfolio_value = self.cash_balance
        for symbol, position in self.positions.items():
            if position != 0:
                current_price = self.get_current_price(symbol)
                self.portfolio_value += position * current_price

    def process_market_data_update(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 0,
    ) -> None:
        """Process market data update and execute pending orders.

        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Trading volume
        """
        # Update current price
        self.current_prices[symbol] = close_price

        # Execute pending orders for this symbol
        pending_orders = self.order_manager.get_active_orders(symbol)
        for order in pending_orders:
            self.execute_order(order)

        # Expire stale orders
        self._expire_stale_orders(symbol, timestamp)

    def _expire_stale_orders(self, symbol: str, timestamp: pd.Timestamp) -> None:
        """Expire orders that have been pending too long.

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
        """
        # In a real implementation, orders would expire based on time-in-force
        # For simulation, we'll skip this for now
        pass

    def get_account_summary(self) -> dict[str, Any]:
        """Get comprehensive account summary.

        Returns:
            Dictionary with account information
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if position != 0:
                current_price = self.get_current_price(symbol)
                # This is a simplified unrealized P&L calculation
                # In reality, you'd track the cost basis for each position
                unrealized_pnl += position * current_price * 0.01  # Placeholder

        return {
            'cash_balance': self.cash_balance,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'unrealized_pnl': unrealized_pnl,
            'total_commission': sum(trade['commission'] for trade in self.trade_history),
            'total_trades': len(self.trade_history),
            'order_summary': self.order_manager.get_order_summary(),
        }

    def get_trade_history(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get trade history.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of trade records
        """
        if symbol is None:
            return self.trade_history.copy()

        return [trade for trade in self.trade_history if trade['symbol'] == symbol]

    def reset(self) -> None:
        """Reset broker to initial state."""
        self.order_manager.reset()
        self.trade_history.clear()
        self.current_prices.clear()
        self.cash_balance = 0.0
        self.positions.clear()
        self.portfolio_value = 0.0
        self.logger.info("Broker reset to initial state")
