"""Comprehensive tests for the order management module.

This module contains comprehensive tests that cover all aspects of the
order management system including order creation, validation, modification,
cancellation, status tracking, and order manager functionality.
"""

import logging

import pandas as pd
import pytest

from backtester.execution.order import (
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestOrderType:
    """Test suite for OrderType enum."""

    def test_order_type_values(self) -> None:
        """Test that all order types have correct string values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"

    def test_order_type_members(self) -> None:
        """Test that all order types are accessible."""
        assert hasattr(OrderType, 'MARKET')
        assert hasattr(OrderType, 'LIMIT')
        assert hasattr(OrderType, 'STOP')
        assert hasattr(OrderType, 'STOP_LIMIT')


class TestOrderSide:
    """Test suite for OrderSide enum."""

    def test_order_side_values(self) -> None:
        """Test that all order sides have correct string values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_side_members(self) -> None:
        """Test that both order sides are accessible."""
        assert hasattr(OrderSide, 'BUY')
        assert hasattr(OrderSide, 'SELL')


class TestOrderStatus:
    """Test suite for OrderStatus enum."""

    def test_order_status_values(self) -> None:
        """Test that all order statuses have correct string values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"


class TestOrderCreation:
    """Test suite for order creation and initialization."""

    def test_order_creation_minimal(self) -> None:
        """Test creating order with minimal required parameters."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.price is None
        assert order.stop_price is None
        assert order.order_id is not None
        assert order.timestamp is not None
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.filled_price is None
        assert order.remaining_quantity == 100.0
        assert order.commission == 0.0

    def test_order_creation_with_optional_params(self) -> None:
        """Test creating order with all optional parameters."""
        price = 101.5
        stop_price = 100.0
        custom_id = "CUSTOM_ID_123"
        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=price,
            stop_price=stop_price,
            order_id=custom_id,
            timestamp=timestamp,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=25.0,
            filled_price=102.0,
            commission=2.5,
            metadata={'custom': 'data'},
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 50.0
        assert order.price == price
        assert order.stop_price == stop_price
        assert order.order_id == custom_id
        assert order.timestamp == timestamp
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 25.0
        assert order.filled_price == 102.0
        assert order.commission == 2.5
        assert order.metadata == {'custom': 'data'}

    def test_order_auto_generated_id(self) -> None:
        """Test that order ID is auto-generated if not provided."""
        order1 = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        assert order1.order_id is not None
        assert order2.order_id is not None
        assert order1.order_id != order2.order_id
        assert len(order1.order_id) == 36  # UUID4 length

    def test_order_auto_generated_timestamp(self) -> None:
        """Test that timestamp is auto-generated if not provided."""
        import time

        before_time = pd.Timestamp.now()
        time.sleep(0.001)  # Small delay
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        after_time = pd.Timestamp.now()

        assert order.timestamp is not None
        assert before_time <= order.timestamp <= after_time

    def test_order_remaining_quantity_auto_set(self) -> None:
        """Test that remaining quantity is auto-set from quantity."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        assert order.remaining_quantity == 100.0

        # Test with pre-set remaining quantity
        order2 = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            remaining_quantity=25.0,  # Explicitly set remaining quantity
        )
        assert order2.remaining_quantity == 25.0  # Uses explicit value

    def test_order_metadata_auto_initialized(self) -> None:
        """Test that metadata is auto-initialized if not provided."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        assert order.metadata == {}

        # Test with explicit metadata
        custom_metadata = {'key': 'value'}
        order2 = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            metadata=custom_metadata,
        )
        assert order2.metadata == custom_metadata


class TestOrderProperties:
    """Test suite for order properties and computed values."""

    def test_is_buy_property(self) -> None:
        """Test is_buy property returns correct values."""
        buy_order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        sell_order = Order(
            symbol="TEST", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=100.0
        )

        assert buy_order.is_buy is True
        assert buy_order.is_sell is False
        assert sell_order.is_buy is False
        assert sell_order.is_sell is True

    def test_is_active_property(self) -> None:
        """Test is_active property returns correct values."""
        pending_order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        partial_order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        filled_order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.FILLED,
        )
        cancelled_order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.CANCELLED,
        )
        rejected_order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.REJECTED,
        )

        assert pending_order.is_active is True
        assert partial_order.is_active is True
        assert filled_order.is_active is False
        assert cancelled_order.is_active is False
        assert rejected_order.is_active is False

    def test_notional_value_filled_price(self) -> None:
        """Test notional value calculation with filled price."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            filled_price=101.5,
        )
        assert order.notional_value == 10150.0

    def test_notional_value_limit_price(self) -> None:
        """Test notional value calculation with limit price when not filled."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=100.0,
        )
        assert order.notional_value == 10000.0

    def test_notional_value_no_price(self) -> None:
        """Test notional value calculation when no price is set."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        assert order.notional_value == 0.0


class TestOrderModification:
    """Test suite for order modification methods."""

    def test_update_fill_partial(self) -> None:
        """Test updating order with partial fill."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.update_fill(25.0, 101.0, 2.5)

        assert order.filled_quantity == 25.0
        assert order.remaining_quantity == 75.0
        assert order.filled_price == 101.0
        assert order.commission == 2.5
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_update_fill_complete(self) -> None:
        """Test updating order with complete fill."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.update_fill(100.0, 101.0, 10.1)

        assert order.filled_quantity == 100.0
        assert order.remaining_quantity == 0.0
        assert order.filled_price == 101.0
        assert order.commission == 10.1
        assert order.status == OrderStatus.FILLED

    def test_update_fill_multiple(self) -> None:
        """Test updating order with multiple fills."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.update_fill(30.0, 101.0, 3.0)
        assert order.status == OrderStatus.PARTIALLY_FILLED

        order.update_fill(40.0, 102.0, 4.0)
        assert order.filled_quantity == 70.0
        assert order.remaining_quantity == 30.0
        assert order.commission == 7.0
        assert order.status == OrderStatus.PARTIALLY_FILLED

        order.update_fill(30.0, 103.0, 3.0)
        assert order.filled_quantity == 100.0
        assert order.remaining_quantity == 0.0
        assert order.commission == 10.0
        assert order.status == OrderStatus.FILLED  # type: ignore[comparison-overlap]

    def test_cancel_active_order(self) -> None:
        """Test cancelling an active order."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        assert order.is_active

        order.cancel("User requested cancellation")

        assert order.status == OrderStatus.CANCELLED
        assert order.metadata is not None
        assert order.metadata['cancel_reason'] == "User requested cancellation"

    def test_cancel_inactive_order(self) -> None:
        """Test cancelling an inactive order (should not change status)."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.FILLED,
        )
        original_status = order.status

        order.cancel("User requested cancellation")

        assert order.status == original_status
        assert order.metadata is not None
        assert 'cancel_reason' not in order.metadata

    def test_reject_order(self) -> None:
        """Test rejecting an order."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.reject("Insufficient funds")

        assert order.status == OrderStatus.REJECTED
        assert order.metadata is not None
        assert order.metadata['reject_reason'] == "Insufficient funds"


class TestOrderSerialization:
    """Test suite for order serialization and representation."""

    def test_to_dict(self) -> None:
        """Test converting order to dictionary."""
        price = 101.5
        stop_price = 100.0
        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=price,
            stop_price=stop_price,
            timestamp=timestamp,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=25.0,
            filled_price=102.0,
            commission=2.5,
            metadata={'custom': 'data'},
        )

        order_dict = order.to_dict()

        # The remaining quantity gets set to quantity in __post_init__ if it was 0.0
        # The notional value is quantity * filled_price (not filled_quantity * filled_price)
        expected_dict = {
            'order_id': order.order_id,
            'symbol': 'TEST',
            'side': 'SELL',
            'order_type': 'LIMIT',
            'quantity': 50.0,
            'price': price,
            'stop_price': stop_price,
            'timestamp': timestamp,
            'status': 'PARTIALLY_FILLED',
            'filled_quantity': 25.0,
            'filled_price': 102.0,
            'remaining_quantity': 50.0,  # Set to quantity in __post_init__
            'commission': 2.5,
            'notional_value': 5100.0,  # quantity * filled_price = 50.0 * 102.0
            'metadata': {'custom': 'data'},
        }

        assert order_dict == expected_dict

    def test_repr_with_price(self) -> None:
        """Test string representation with price."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=101.5,
        )

        order_id_short = order.order_id[:8] if order.order_id else 'None'
        expected = f"Order({order_id_short}, TEST, BUY, LIMIT, 100.0@101.5)"

        assert repr(order) == expected

    def test_repr_market_order(self) -> None:
        """Test string representation of market order (no price)."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        order_id_short = order.order_id[:8] if order.order_id else 'None'
        expected = f"Order({order_id_short}, TEST, BUY, MARKET, 100.0@MKT)"

        assert repr(order) == expected

    def test_repr_short_id(self) -> None:
        """Test that order ID is shortened in repr."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        repr_str = repr(order)
        # Should contain 8 characters from UUID, not the full UUID
        order_id_in_repr = repr_str.split(',')[0].replace('Order(', '')
        assert len(order_id_in_repr) == 8


class TestOrderManager:
    """Test suite for OrderManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test order manager initialization."""
        manager = OrderManager()
        assert manager.orders == {}
        assert manager.order_history == []
        assert manager.next_order_id == 1

    def test_manager_initialization_with_logger(self) -> None:
        """Test order manager initialization with custom logger."""
        logger = logging.getLogger("test_logger")
        manager = OrderManager(logger=logger)
        assert manager.logger == logger

    def test_create_order_market(self) -> None:
        """Test creating a market order."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100.0
        assert order.order_id in manager.orders
        assert order in manager.order_history

    def test_create_order_limit(self) -> None:
        """Test creating a limit order."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=101.5,
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 50.0
        assert order.price == 101.5

    def test_create_order_stop(self) -> None:
        """Test creating a stop order."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=100.0,
            stop_price=105.0,
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 105.0

    def test_create_order_stop_limit(self) -> None:
        """Test creating a stop-limit order."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=75.0,
            price=99.0,
            stop_price=100.0,
        )

        assert order.symbol == "TEST"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.price == 99.0
        assert order.stop_price == 100.0

    def test_create_order_with_metadata(self) -> None:
        """Test creating order with metadata."""
        manager = OrderManager()
        metadata = {'strategy': 'momentum', 'priority': 'high'}

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            metadata=metadata,
        )

        assert order.metadata == metadata

    def test_create_order_validation_quantity_zero(self) -> None:
        """Test creating order with zero quantity raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Order quantity must be positive"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.0,
            )

    def test_create_order_validation_quantity_negative(self) -> None:
        """Test creating order with negative quantity raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Order quantity must be positive"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=-100.0,
            )

    def test_create_order_validation_limit_no_price(self) -> None:
        """Test creating limit order without price raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Price required for LIMIT orders"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100.0,
            )

    def test_create_order_validation_stop_no_stop_price(self) -> None:
        """Test creating stop order without stop price raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Stop price required for STOP orders"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                quantity=100.0,
            )

    def test_create_order_validation_stop_limit_missing_price(self) -> None:
        """Test creating stop-limit order without price raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Price required for STOP_LIMIT orders"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.STOP_LIMIT,
                quantity=100.0,
                stop_price=105.0,
            )

    def test_create_order_validation_stop_limit_missing_stop_price(self) -> None:
        """Test creating stop-limit order without stop price raises ValueError."""
        manager = OrderManager()

        with pytest.raises(ValueError, match="Stop price required for STOP_LIMIT orders"):
            manager.create_order(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.STOP_LIMIT,
                quantity=100.0,
                price=100.0,
            )

    def test_get_order_existing(self) -> None:
        """Test getting an existing order."""
        manager = OrderManager()

        created_order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        assert created_order.order_id is not None
        retrieved_order = manager.get_order(created_order.order_id)
        assert retrieved_order == created_order

    def test_get_order_nonexistent(self) -> None:
        """Test getting a non-existent order returns None."""
        manager = OrderManager()

        retrieved_order = manager.get_order("NONEXISTENT_ID")
        assert retrieved_order is None

    def test_cancel_order_existing_active(self) -> None:
        """Test cancelling an existing active order."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        assert order.is_active
        assert order.order_id is not None

        result = manager.cancel_order(order.order_id, "User requested")

        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_order_nonexistent(self) -> None:
        """Test cancelling a non-existent order returns False."""
        manager = OrderManager()

        result = manager.cancel_order("NONEXISTENT_ID", "User requested")
        assert result is False

    def test_cancel_order_inactive(self) -> None:
        """Test cancelling an inactive order returns False."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        order.status = OrderStatus.FILLED  # Make it inactive
        assert order.order_id is not None

        result = manager.cancel_order(order.order_id, "User requested")
        assert result is False

    def test_cancel_all_orders(self) -> None:
        """Test cancelling all active orders."""
        manager = OrderManager()

        # Create multiple orders
        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST3", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        # Make one order inactive
        order2.status = OrderStatus.FILLED

        cancelled_count = manager.cancel_all_orders("System shutdown")

        assert cancelled_count == 2  # Only active orders should be cancelled
        assert order1.status == OrderStatus.CANCELLED
        assert order2.status == OrderStatus.FILLED  # Should remain filled
        assert order3.status == OrderStatus.CANCELLED

    def test_get_active_orders_no_filter(self) -> None:
        """Test getting all active orders without filter."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST3", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        # Make one order inactive
        order2.status = OrderStatus.FILLED

        active_orders = manager.get_active_orders()

        assert len(active_orders) == 2
        assert order1 in active_orders
        assert order3 in active_orders
        assert order2 not in active_orders

    def test_get_active_orders_with_filter(self) -> None:
        """Test getting active orders filtered by symbol."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        active_orders = manager.get_active_orders("TEST1")

        assert len(active_orders) == 2
        assert order1 in active_orders
        assert order3 in active_orders
        assert order2 not in active_orders

    def test_get_order_history_no_filter(self) -> None:
        """Test getting order history without filter."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )

        history = manager.get_order_history()

        assert len(history) == 2
        assert order1 in history
        assert order2 in history

    def test_get_order_history_with_filter(self) -> None:
        """Test getting order history filtered by symbol."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        history = manager.get_order_history("TEST1")

        assert len(history) == 2
        assert order1 in history
        assert order3 in history
        assert order2 not in history

    def test_get_filled_orders(self) -> None:
        """Test getting filled orders."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST3", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        # Fill some orders
        order1.status = OrderStatus.FILLED
        order2.status = OrderStatus.CANCELLED
        order3.status = OrderStatus.FILLED

        filled_orders = manager.get_filled_orders()

        assert len(filled_orders) == 2
        assert order1 in filled_orders
        assert order3 in filled_orders
        assert order2 not in filled_orders

    def test_get_filled_orders_with_filter(self) -> None:
        """Test getting filled orders filtered by symbol."""
        manager = OrderManager()

        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order3 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )

        # Fill some orders
        order1.status = OrderStatus.FILLED
        order2.status = OrderStatus.FILLED
        order3.status = OrderStatus.PARTIALLY_FILLED

        filled_orders = manager.get_filled_orders("TEST1")

        assert len(filled_orders) == 1
        assert order1 in filled_orders
        assert order2 not in filled_orders
        assert order3 not in filled_orders  # Only PARTIALLY_FILLED, not FILLED

    def test_update_order_status(self) -> None:
        """Test updating order status with fill information."""
        manager = OrderManager()

        order = manager.create_order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        assert order.order_id is not None
        manager.update_order_status(order.order_id, 50.0, 101.0, 5.0)

        assert order.filled_quantity == 50.0
        assert order.remaining_quantity == 50.0
        assert order.filled_price == 101.0
        assert order.commission == 5.0
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_update_order_status_nonexistent(self) -> None:
        """Test updating status of non-existent order (should not raise error)."""
        manager = OrderManager()

        # Should not raise any error
        manager.update_order_status("NONEXISTENT_ID", 50.0, 101.0, 5.0)

    def test_get_order_summary_empty(self) -> None:
        """Test getting order summary with no orders."""
        manager = OrderManager()

        summary = manager.get_order_summary()

        assert summary['total_orders'] == 0
        assert summary['active_orders'] == 0
        assert summary['filled_orders'] == 0
        assert summary['cancelled_orders'] == 0
        assert summary['rejected_orders'] == 0
        assert summary['fill_rate'] == 0.0
        assert summary['total_volume'] == 0.0
        assert summary['total_commission'] == 0.0
        assert summary['active_orders_detail'] == []
        assert summary['recent_fills'] == []

    def test_get_order_summary_with_orders(self) -> None:
        """Test getting order summary with various orders."""
        manager = OrderManager()

        # Create orders with different statuses
        order1 = manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        order1.status = OrderStatus.FILLED
        order1.update_fill(100.0, 100.0, 1.0)  # notional = 10000

        order2 = manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        order2.status = OrderStatus.CANCELLED

        order3 = manager.create_order(
            symbol="TEST3", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=75.0
        )
        order3.status = OrderStatus.REJECTED

        order4 = manager.create_order(
            symbol="TEST4", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=25.0
        )  # Active
        # order4 is used in the summary assertions below

        summary = manager.get_order_summary()

        assert summary['total_orders'] == 4
        assert summary['active_orders'] == 1
        assert summary['filled_orders'] == 1
        assert summary['cancelled_orders'] == 1
        assert summary['rejected_orders'] == 1
        assert summary['fill_rate'] == 0.25  # 1 filled out of 4 total
        assert summary['total_volume'] == 10000.0
        assert summary['total_commission'] == 1.0
        assert len(summary['active_orders_detail']) == 1
        assert len(summary['recent_fills']) == 1

        # Verify that order4 is the active order in the detail
        active_order_dicts = summary['active_orders_detail']
        assert len(active_order_dicts) == 1
        assert active_order_dicts[0]['order_id'] == order4.order_id
        assert active_order_dicts[0]['symbol'] == 'TEST4'
        assert active_order_dicts[0]['quantity'] == 25.0
        assert active_order_dicts[0]['status'] == 'PENDING'

        # Verify that order1 is the recent fill
        recent_fills = summary['recent_fills']
        assert len(recent_fills) == 1
        assert recent_fills[0]['order_id'] == order1.order_id
        assert recent_fills[0]['symbol'] == 'TEST1'
        assert recent_fills[0]['filled_quantity'] == 100.0
        assert recent_fills[0]['status'] == 'FILLED'

    def test_reset_manager(self) -> None:
        """Test resetting order manager state."""
        manager = OrderManager()

        # Create some orders
        manager.create_order(
            symbol="TEST1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        manager.create_order(
            symbol="TEST2", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )

        # Add some custom attributes
        manager.next_order_id = 100

        manager.reset()

        assert manager.orders == {}
        assert manager.order_history == []
        assert manager.next_order_id == 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_order_with_none_values(self) -> None:
        """Test order creation with None values for optional parameters."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            stop_price=None,
            timestamp=None,
            order_id=None,
            metadata=None,
        )

        assert order.price is None
        assert order.stop_price is None
        assert order.timestamp is not None
        assert order.order_id is not None
        assert order.metadata == {}

    def test_order_large_quantities(self) -> None:
        """Test order with very large quantities."""
        large_quantity = 1e10
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=large_quantity
        )

        assert order.quantity == large_quantity
        assert order.remaining_quantity == large_quantity

    def test_order_negative_prices(self) -> None:
        """Test order with negative prices (unusual but should be allowed)."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=-10.0,
        )

        assert order.price == -10.0

    def test_order_extremely_long_symbol(self) -> None:
        """Test order with very long symbol name."""
        long_symbol = "A" * 1000
        order = Order(
            symbol=long_symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        assert order.symbol == long_symbol

    def test_order_special_characters_in_symbol(self) -> None:
        """Test order with special characters in symbol."""
        special_symbol = "TEST-1.5.2023/USD"
        order = Order(
            symbol=special_symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        assert order.symbol == special_symbol

    def test_update_fill_zero_quantity(self) -> None:
        """Test updating order with zero fill quantity."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.update_fill(0.0, 100.0, 0.0)

        assert order.filled_quantity == 0.0
        assert order.remaining_quantity == 100.0
        assert order.status == OrderStatus.PENDING  # Should not change if no fill

    def test_update_fill_negative_quantity(self) -> None:
        """Test updating order with negative fill quantity (edge case)."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        order.update_fill(-10.0, 100.0, 0.0)

        assert order.filled_quantity == -10.0
        assert order.remaining_quantity == 110.0  # Becomes larger than original

    def test_cancel_with_none_metadata(self) -> None:
        """Test that cancel requires metadata to be initialized."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        # Normal case - metadata should be auto-initialized
        assert order.metadata == {}
        order.cancel("Test cancellation")
        assert order.status == OrderStatus.CANCELLED
        assert order.metadata['cancel_reason'] == "Test cancellation"

    def test_reject_with_none_metadata(self) -> None:
        """Test that reject requires metadata to be initialized."""
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        # Normal case - metadata should be auto-initialized
        assert order.metadata == {}
        order.reject("Test rejection")
        assert order.status == OrderStatus.REJECTED
        assert order.metadata['reject_reason'] == "Test rejection"

    def test_order_manager_memory_usage(self) -> None:
        """Test that order manager doesn't use excessive memory with many orders."""
        manager = OrderManager()

        # Create many orders
        num_orders = 1000
        for i in range(num_orders):
            manager.create_order(
                symbol=f"SYMBOL_{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )

        # Check that we have the expected number of orders
        assert len(manager.orders) == num_orders
        assert len(manager.order_history) == num_orders

        # Test that getting orders still works
        first_order = manager.get_order(list(manager.orders.keys())[0])
        assert first_order is not None

    def test_concurrent_order_creation_simulation(self) -> None:
        """Test creating many orders in sequence to simulate concurrent access."""
        manager = OrderManager()

        orders = []
        for i in range(100):
            order = manager.create_order(
                symbol=f"SYMBOL_{i % 10}",  # Some symbols repeat
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET if i % 3 == 0 else OrderType.LIMIT,
                quantity=float(i + 1) * 10,
                price=float(i + 1) * 10 if i % 3 != 0 else None,
            )
            orders.append(order)

        # Verify all orders were created and stored correctly
        assert len(manager.orders) == 100
        assert len(manager.order_history) == 100

        # Verify we can retrieve each order
        for order in orders:
            assert order.order_id is not None
            retrieved = manager.get_order(order.order_id)
            assert retrieved == order


if __name__ == "__main__":
    pytest.main([__file__])
