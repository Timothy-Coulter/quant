"""Order Management System.

This module provides a comprehensive order management system with support for
different order types, execution logic, and order lifecycle tracking.
"""

import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from backtester.core.config import SimulatedBrokerConfig


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderData:
    """Base order class."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None  # For limit orders
    stop_price: float | None = None  # For stop orders
    timestamp: pd.Timestamp | None = None
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float | None = None
    remaining_quantity: float = 0.0
    commission: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing for order setup."""
        if self.order_id is None:
            self.order_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]

    @property
    def notional_value(self) -> float:
        """Calculate notional value of the order."""
        return self.quantity * (self.filled_price or self.price or 0.0)

    def update_fill(self, fill_quantity: float, fill_price: float, commission: float = 0.0) -> None:
        """Update order with execution information.

        Args:
            fill_quantity: Quantity that was filled
            fill_price: Price at which fill occurred
            commission: Commission for this fill
        """
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(0.0, self.quantity - self.filled_quantity)
        self.filled_price = fill_price
        self.commission += commission

        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self, reason: str = "Cancelled by user") -> None:
        """Cancel the order.

        Args:
            reason: Reason for cancellation
        """
        if self.is_active:
            self.status = OrderStatus.CANCELLED
            assert self.metadata is not None
            self.metadata['cancel_reason'] = reason

    def reject(self, reason: str = "Rejected by broker") -> None:
        """Reject the order.

        Args:
            reason: Reason for rejection
        """
        self.status = OrderStatus.REJECTED
        assert self.metadata is not None
        self.metadata['reject_reason'] = reason

    def to_dict(self) -> dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'remaining_quantity': self.remaining_quantity,
            'commission': self.commission,
            'notional_value': self.notional_value,
            'metadata': self.metadata,
        }

    def __repr__(self) -> str:
        """Return string representation of the order."""
        price_str = self.price if self.price is not None else 'MKT'
        order_id = self.order_id[:8] if self.order_id else 'None'
        return (
            f"Order({order_id}, {self.symbol}, {self.side.value}, "
            f"{self.order_type.value}, {self.quantity}@{price_str})"
        )


class OrderManager:
    """Order management system."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        config: SimulatedBrokerConfig | None = None,
    ) -> None:
        """Initialize the order manager.

        Args:
            logger: Optional logger instance
            config: Optional broker configuration so metadata mirrors execution settings
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.orders: dict[str, OrderData] = {}
        self.order_history: list[OrderData] = []
        self.next_order_id: int = 1
        self._config = config

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderData:
        """Create a new order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            metadata: Additional order metadata

        Returns:
            Created order
        """
        # Validate order parameters
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")

        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"Price required for {order_type.value} orders")

        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError(f"Stop price required for {order_type.value} orders")

        # Create order
        metadata = dict(metadata or {})
        if self._config is not None:
            metadata.setdefault('commission_rate', self._config.commission_rate)
            metadata.setdefault('min_commission', self._config.min_commission)
        order = OrderData(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata,
        )

        # Store order
        assert order.order_id is not None
        self.orders[order.order_id] = order
        self.order_history.append(order)

        self.logger.info(f"Created order: {order}")
        return order

    def get_order(self, order_id: str) -> OrderData | None:
        """Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order if found, None otherwise
        """
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str, reason: str = "Cancelled") -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            True if order was cancelled, False otherwise
        """
        order = self.get_order(order_id)
        if order and order.is_active:
            order.cancel(reason)
            self.logger.info(f"Cancelled order {order_id}: {reason}")
            return True
        return False

    def cancel_all_orders(self, reason: str = "Cancelled all") -> int:
        """Cancel all active orders.

        Args:
            reason: Cancellation reason

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        for order_id in list(self.orders.keys()):
            if self.cancel_order(order_id, reason):
                cancelled_count += 1
        return cancelled_count

    def get_active_orders(self, symbol: str | None = None) -> list[OrderData]:
        """Get all active orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active orders
        """
        active_orders = [
            order
            for order in self.orders.values()
            if order.is_active and (symbol is None or order.symbol == symbol)
        ]
        return active_orders

    def get_order_history(self, symbol: str | None = None) -> list[OrderData]:
        """Get order history, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of historical orders
        """
        if symbol is None:
            return self.order_history.copy()

        return [order for order in self.order_history if order.symbol == symbol]

    def get_filled_orders(self, symbol: str | None = None) -> list[OrderData]:
        """Get all filled orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of filled orders
        """
        filled_orders = [
            order
            for order in self.order_history
            if order.status == OrderStatus.FILLED and (symbol is None or order.symbol == symbol)
        ]
        return filled_orders

    def update_order_status(
        self, order_id: str, filled_quantity: float, fill_price: float, commission: float = 0.0
    ) -> None:
        """Update order with fill information.

        Args:
            order_id: Order ID
            filled_quantity: Quantity filled
            fill_price: Fill price
            commission: Commission amount
        """
        order = self.get_order(order_id)
        if order:
            order.update_fill(filled_quantity, fill_price, commission)
            self.logger.debug(f"Updated order {order_id}: {filled_quantity}@{fill_price}")

    def get_order_summary(self) -> dict[str, Any]:
        """Get comprehensive order summary.

        Returns:
            Dictionary with order statistics
        """
        total_orders = len(self.order_history)
        active_orders = len(self.get_active_orders())
        filled_orders = len(self.get_filled_orders())
        cancelled_orders = len([o for o in self.order_history if o.status == OrderStatus.CANCELLED])
        rejected_orders = len([o for o in self.order_history if o.status == OrderStatus.REJECTED])

        total_volume = sum(o.notional_value for o in self.get_filled_orders())
        total_commission = sum(o.commission for o in self.order_history)

        return {
            'total_orders': total_orders,
            'active_orders': active_orders,
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'rejected_orders': rejected_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'total_volume': total_volume,
            'total_commission': total_commission,
            'active_orders_detail': [o.to_dict() for o in self.get_active_orders()],
            'recent_fills': [o.to_dict() for o in self.get_filled_orders()[-10:]],
        }

    def reset(self) -> None:
        """Reset order manager to initial state."""
        self.orders.clear()
        self.order_history.clear()
        self.next_order_id = 1
        self.logger.info("Order manager reset")


Order = OrderData

__all__ = ['Order', 'OrderData', 'OrderType', 'OrderSide', 'OrderStatus', 'OrderManager']
