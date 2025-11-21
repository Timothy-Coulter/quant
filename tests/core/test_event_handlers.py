"""Tests for the event handlers module."""

import time
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from backtester.core.event_bus import Event
from backtester.core.event_handlers import (
    CompositeEventHandler,
    EventHandler,
    MarketDataHandler,
    OrderHandler,
    PortfolioHandler,
    RiskHandler,
    SignalHandler,
    StrategyHandler,
    create_event_handler,
)
from backtester.core.events import (
    MarketDataEvent,
    MarketDataType,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioUpdateEvent,
    RiskAlertEvent,
    RiskLevel,
    SignalEvent,
    SignalType,
    StrategyEvent,
    StrategyEventType,
)


class ConcreteEventHandler(EventHandler):
    """Concrete implementation of EventHandler for testing."""

    def handle_event(self, event: Event) -> None:
        """Handle the event."""
        self.event_count += 1


class TestEventHandler:
    """Test cases for the base EventHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ConcreteEventHandler("test_handler")

    def test_handler_initialization(self):
        """Test handler initialization."""
        assert self.handler.name == "test_handler"
        assert self.handler.enabled is True
        assert self.handler.event_count == 0
        assert self.handler.error_count == 0
        assert self.handler.logger is not None

    def test_handler_enable_disable(self):
        """Test handler enable/disable functionality."""
        # Test disable
        self.handler.disable()
        assert self.handler.enabled is False

        # Test enable
        self.handler.enable()
        assert self.handler.enabled is True

    def test_handler_can_handle(self):
        """Test default can_handle method."""
        event = Event("TEST", time.time(), "source")
        assert self.handler.can_handle(event) is True

    def test_handler_get_stats(self):
        """Test handler statistics."""
        stats = self.handler.get_stats()

        assert stats["name"] == "test_handler"
        assert stats["enabled"] is True
        assert stats["event_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0

    def test_handler_reset_stats(self):
        """Test handler statistics reset."""
        self.handler.event_count = 10
        self.handler.error_count = 2

        self.handler.enable()
        self.handler.disable()
        self.handler.enable()
        self.handler.disable()

        stats = self.handler.get_stats()
        assert stats["event_count"] == 10  # event count should remain unchanged
        assert stats["error_count"] == 2  # error count should remain unchanged


class TestMarketDataHandler:
    """Test cases for MarketDataHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MarketDataHandler("test_market_handler")

    def test_market_data_handler_initialization(self):
        """Test market data handler initialization."""
        assert self.handler.name == "test_market_handler"
        assert self.handler.processed_symbols == set()
        assert self.handler.last_prices == {}

    def test_handle_market_data_event(self):
        """Test handling market data events."""
        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=time.time(),
            source="market_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=152.0,
            volume=1000000,
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert "AAPL" in self.handler.processed_symbols
        assert self.handler.last_prices["AAPL"] == 152.0

    def test_handle_non_market_data_event(self):
        """Test handling non-market data events."""
        event = Event("OTHER_EVENT", time.time(), "source")

        with patch.object(self.handler.logger, 'warning') as mock_warning:
            self.handler.handle_event(event)
            mock_warning.assert_called_once()

    def test_handle_disabled_handler(self):
        """Test handling events with disabled handler."""
        self.handler.disable()
        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=time.time(),
            source="market_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=152.0,
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 0
        assert "AAPL" not in self.handler.processed_symbols

    def test_handle_event_error(self):
        """Test error handling in market data handler."""

        def failing_process(event):
            raise Exception("Handler error")

        cast(Any, self.handler)._process_market_data = failing_process

        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=time.time(),
            source="market_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=152.0,
        )

        with patch.object(self.handler.logger, 'error') as mock_error:
            self.handler.handle_event(event)

            assert self.handler.event_count == 1
            assert self.handler.error_count == 1
            mock_error.assert_called_once()

    def test_get_last_price(self):
        """Test getting last price for a symbol."""
        # Test non-existent symbol
        price = self.handler.get_last_price("NONEXISTENT")
        assert price is None

        # Test existing symbol
        self.handler.last_prices["AAPL"] = 152.0
        price = self.handler.get_last_price("AAPL")
        assert price == 152.0

    def test_get_processed_symbols(self):
        """Test getting processed symbols."""
        # Test empty list
        symbols = self.handler.get_processed_symbols()
        assert symbols == []

        # Test with symbols
        self.handler.processed_symbols.add("AAPL")
        self.handler.processed_symbols.add("GOOGL")
        symbols = self.handler.get_processed_symbols()
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert len(symbols) == 2


class TestSignalHandler:
    """Test cases for SignalHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SignalHandler("test_signal_handler")

    def test_signal_handler_initialization(self):
        """Test signal handler initialization."""
        assert self.handler.name == "test_signal_handler"
        assert self.handler.processed_signals == []
        assert self.handler.signal_stats == {}

    def test_handle_signal_event(self):
        """Test handling signal events."""
        event = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert len(self.handler.processed_signals) == 1
        assert self.handler.signal_stats["BUY"] == 1

        signal_data = self.handler.processed_signals[0]
        assert signal_data["symbol"] == "AAPL"
        assert signal_data["signal_type"] == "BUY"
        assert signal_data["strength"] == 0.8
        assert signal_data["confidence"] == 0.9

    def test_handle_multiple_signal_types(self):
        """Test handling multiple signal types."""
        buy_signal = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
        )

        sell_signal = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.SELL,
            strength=0.7,
        )

        self.handler.handle_event(buy_signal)
        self.handler.handle_event(sell_signal)

        assert self.handler.signal_stats["BUY"] == 1
        assert self.handler.signal_stats["SELL"] == 1

    def test_get_signal_stats(self):
        """Test getting signal statistics."""
        # Test empty stats
        stats = self.handler.get_signal_stats()
        assert stats == {}

        # Test with stats
        self.handler.signal_stats["BUY"] = 5
        self.handler.signal_stats["SELL"] = 3
        stats = self.handler.get_signal_stats()
        assert stats["BUY"] == 5
        assert stats["SELL"] == 3

    def test_get_processed_signals(self):
        """Test getting processed signals."""
        # Test empty list
        signals = self.handler.get_processed_signals()
        assert signals == []

        # Test with signals
        event = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.BUY,
        )
        self.handler.handle_event(event)

        signals = self.handler.get_processed_signals()
        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"


class TestOrderHandler:
    """Test cases for OrderHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = OrderHandler("test_order_handler")

    def test_order_handler_initialization(self):
        """Test order handler initialization."""
        assert self.handler.name == "test_order_handler"
        assert self.handler.active_orders == {}
        assert self.handler.completed_orders == []
        assert self.handler.order_stats == {}

    def test_handle_pending_order(self):
        """Test handling pending order."""
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.PENDING,
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert "order_123" in self.handler.active_orders
        assert len(self.handler.completed_orders) == 0
        assert self.handler.order_stats["PENDING"] == 1

    def test_handle_filled_order(self):
        """Test handling filled order."""
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.FILLED,
            filled_quantity=100.0,
            average_fill_price=152.0,
        )

        self.handler.handle_event(event)

        assert "order_123" not in self.handler.active_orders
        assert len(self.handler.completed_orders) == 1
        assert self.handler.order_stats["FILLED"] == 1

        completed_order = self.handler.completed_orders[0]
        assert completed_order.order_id == "order_123"
        assert completed_order.status == OrderStatus.FILLED
        assert completed_order.filled_quantity == 100.0
        assert completed_order.average_fill_price == 152.0

    def test_handle_multiple_orders(self):
        """Test handling multiple orders."""
        order1 = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_1",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.PENDING,
        )

        order2 = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_2",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            status=OrderStatus.FILLED,
        )

        self.handler.handle_event(order1)
        self.handler.handle_event(order2)

        assert len(self.handler.active_orders) == 1
        assert len(self.handler.completed_orders) == 1
        assert self.handler.order_stats["PENDING"] == 1
        assert self.handler.order_stats["FILLED"] == 1

    def test_get_active_orders(self):
        """Test getting active orders."""
        # Test empty orders
        orders = self.handler.get_active_orders()
        assert orders == {}

        # Test with orders
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.PENDING,
        )
        self.handler.handle_event(event)

        orders = self.handler.get_active_orders()
        assert "order_123" in orders
        assert orders["order_123"].symbol == "AAPL"

    def test_get_completed_orders(self):
        """Test getting completed orders."""
        # Test empty orders
        orders = self.handler.get_completed_orders()
        assert orders == []

        # Test with orders
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.FILLED,
        )
        self.handler.handle_event(event)

        orders = self.handler.get_completed_orders()
        assert len(orders) == 1
        assert orders[0].order_id == "order_123"

    def test_get_order_stats(self):
        """Test getting order statistics."""
        # Test empty stats
        stats = self.handler.get_order_stats()
        assert stats == {}

        # Test with stats
        self.handler.order_stats["PENDING"] = 2
        self.handler.order_stats["FILLED"] = 3
        self.handler.order_stats["CANCELLED"] = 1
        stats = self.handler.get_order_stats()
        assert stats["PENDING"] == 2
        assert stats["FILLED"] == 3
        assert stats["CANCELLED"] == 1


class TestPortfolioHandler:
    """Test cases for PortfolioHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PortfolioHandler("test_portfolio_handler")

    def test_portfolio_handler_initialization(self):
        """Test portfolio handler initialization."""
        assert self.handler.name == "test_portfolio_handler"
        assert self.handler.portfolio_history == []
        assert self.handler.portfolio_metrics == {}

    def test_handle_portfolio_update_event(self):
        """Test handling portfolio update events."""
        event = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
            metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.1},
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert len(self.handler.portfolio_history) == 1
        assert self.handler.portfolio_metrics["total_value"] == 50000.0
        assert self.handler.portfolio_metrics["sharpe_ratio"] == 1.5

    def test_handle_multiple_portfolio_updates(self):
        """Test handling multiple portfolio updates."""
        event1 = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
        )

        event2 = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=51000.0,
            cash_balance=9500.0,
            positions_value=41500.0,
        )

        self.handler.handle_event(event1)
        self.handler.handle_event(event2)

        assert len(self.handler.portfolio_history) == 2
        assert self.handler.portfolio_metrics["total_value"] == 51000.0
        assert self.handler.portfolio_metrics["cash_balance"] == 9500.0

    def test_get_portfolio_history(self):
        """Test getting portfolio history."""
        # Test empty history
        history = self.handler.get_portfolio_history()
        assert history == []

        # Test with history
        event = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
        )
        self.handler.handle_event(event)

        history = self.handler.get_portfolio_history()
        assert len(history) == 1
        assert history[0].portfolio_id == "portfolio_123"

    def test_get_current_metrics(self):
        """Test getting current portfolio metrics."""
        # Test empty metrics
        metrics = self.handler.get_current_metrics()
        assert metrics == {}

        # Test with metrics
        event = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
            metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.1},
        )
        self.handler.handle_event(event)

        metrics = self.handler.get_current_metrics()
        assert metrics["total_value"] == 50000.0
        assert metrics["cash_balance"] == 10000.0
        assert metrics["positions_value"] == 40000.0
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["max_drawdown"] == 0.1


class TestRiskHandler:
    """Test cases for RiskHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RiskHandler("test_risk_handler")

    def test_risk_handler_initialization(self):
        """Test risk handler initialization."""
        assert self.handler.name == "test_risk_handler"
        assert self.handler.active_alerts == []
        assert self.handler.alert_history == []
        assert self.handler.alert_stats == {}

    def test_handle_risk_alert_event(self):
        """Test handling risk alert events."""
        event = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_123",
            risk_level=RiskLevel.HIGH,
            message="Portfolio leverage exceeded",
            component="portfolio_manager",
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert len(self.handler.active_alerts) == 1
        assert len(self.handler.alert_history) == 1
        assert self.handler.alert_stats["HIGH"] == 1

        alert = self.handler.active_alerts[0]
        assert alert.alert_id == "alert_123"
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.message == "Portfolio leverage exceeded"

    def test_handle_multiple_risk_levels(self):
        """Test handling multiple risk levels."""
        high_alert = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_1",
            risk_level=RiskLevel.HIGH,
            message="High risk alert",
            component="test",
        )

        critical_alert = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_2",
            risk_level=RiskLevel.CRITICAL,
            message="Critical risk alert",
            component="test",
        )

        self.handler.handle_event(high_alert)
        self.handler.handle_event(critical_alert)

        assert self.handler.alert_stats["HIGH"] == 1
        assert self.handler.alert_stats["CRITICAL"] == 1
        assert len(self.handler.active_alerts) == 2

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Test empty alerts
        alerts = self.handler.get_active_alerts()
        assert alerts == []

        # Test with alerts
        event = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_123",
            risk_level=RiskLevel.HIGH,
            message="Test alert",
            component="test",
        )
        self.handler.handle_event(event)

        alerts = self.handler.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].alert_id == "alert_123"

    def test_get_alert_history(self):
        """Test getting alert history."""
        # Test empty history
        history = self.handler.get_alert_history()
        assert history == []

        # Test with history
        event = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_123",
            risk_level=RiskLevel.HIGH,
            message="Test alert",
            component="test",
        )
        self.handler.handle_event(event)

        history = self.handler.get_alert_history()
        assert len(history) == 1
        assert history[0].alert_id == "alert_123"

    def test_get_alert_stats(self):
        """Test getting alert statistics."""
        # Test empty stats
        stats = self.handler.get_alert_stats()
        assert stats == {}

        # Test with stats
        self.handler.alert_stats["LOW"] = 2
        self.handler.alert_stats["MEDIUM"] = 5
        self.handler.alert_stats["HIGH"] = 3
        self.handler.alert_stats["CRITICAL"] = 1
        stats = self.handler.get_alert_stats()
        assert stats["LOW"] == 2
        assert stats["MEDIUM"] == 5
        assert stats["HIGH"] == 3
        assert stats["CRITICAL"] == 1


class TestStrategyHandler:
    """Test cases for StrategyHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = StrategyHandler("test_strategy_handler")

    def test_strategy_handler_initialization(self):
        """Test strategy handler initialization."""
        assert self.handler.name == "test_strategy_handler"
        assert self.handler.strategy_events == []
        assert self.handler.strategy_stats == {}

    def test_handle_strategy_event(self):
        """Test handling strategy events."""
        event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.SIGNAL_GENERATED,
            strategy_name="MovingAverageStrategy",
            data={"symbol": "AAPL", "signal": "BUY"},
        )

        self.handler.handle_event(event)

        assert self.handler.event_count == 1
        assert len(self.handler.strategy_events) == 1
        assert self.handler.strategy_stats["SIGNAL_GENERATED"] == 1

        strategy_event = self.handler.strategy_events[0]
        assert strategy_event.strategy_id == "strategy_123"
        assert strategy_event.event_type == StrategyEventType.SIGNAL_GENERATED.value
        assert strategy_event.strategy_name == "MovingAverageStrategy"

    def test_handle_multiple_event_types(self):
        """Test handling multiple event types."""
        signal_event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.SIGNAL_GENERATED,
            strategy_name="MovingAverageStrategy",
        )

        error_event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.STRATEGY_ERROR,
            strategy_name="MovingAverageStrategy",
            error_details={"error": "Calculation error"},
        )

        self.handler.handle_event(signal_event)
        self.handler.handle_event(error_event)

        assert self.handler.strategy_stats["SIGNAL_GENERATED"] == 1
        assert self.handler.strategy_stats["STRATEGY_ERROR"] == 1
        assert len(self.handler.strategy_events) == 2

    def test_get_strategy_events(self):
        """Test getting strategy events."""
        # Test empty events
        events = self.handler.get_strategy_events()
        assert events == []

        # Test with events
        event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.SIGNAL_GENERATED,
            strategy_name="MovingAverageStrategy",
        )
        self.handler.handle_event(event)

        events = self.handler.get_strategy_events()
        assert len(events) == 1
        assert events[0].strategy_id == "strategy_123"

    def test_get_strategy_stats(self):
        """Test getting strategy statistics."""
        # Test empty stats
        stats = self.handler.get_strategy_stats()
        assert stats == {}

        # Test with stats
        self.handler.strategy_stats["SIGNAL_GENERATED"] = 10
        self.handler.strategy_stats["POSITION_OPENED"] = 5
        self.handler.strategy_stats["STRATEGY_ERROR"] = 2
        stats = self.handler.get_strategy_stats()
        assert stats["SIGNAL_GENERATED"] == 10
        assert stats["POSITION_OPENED"] == 5
        assert stats["STRATEGY_ERROR"] == 2


class TestCompositeEventHandler:
    """Test cases for CompositeEventHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.composite_handler = CompositeEventHandler("test_composite")
        self.sub_handler1 = Mock()
        self.sub_handler2 = Mock()

    def test_composite_handler_initialization(self):
        """Test composite handler initialization."""
        assert self.composite_handler.name == "test_composite"
        assert self.composite_handler.handlers == []

    def test_add_handler(self):
        """Test adding sub-handlers."""
        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.add_handler(self.sub_handler2)

        assert len(self.composite_handler.handlers) == 2
        assert self.sub_handler1 in self.composite_handler.handlers
        assert self.sub_handler2 in self.composite_handler.handlers

    def test_remove_handler(self):
        """Test removing sub-handlers."""
        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.add_handler(self.sub_handler2)

        # Remove first handler
        result = self.composite_handler.remove_handler(self.sub_handler1)
        assert result is True
        assert len(self.composite_handler.handlers) == 1
        assert self.sub_handler1 not in self.composite_handler.handlers
        assert self.sub_handler2 in self.composite_handler.handlers

        # Remove non-existent handler
        result = self.composite_handler.remove_handler(self.sub_handler1)
        assert result is False

    def test_handle_event_with_enabled_handlers(self):
        """Test handling events with enabled handlers."""
        self.sub_handler1.can_handle.return_value = True
        self.sub_handler2.can_handle.return_value = True
        self.sub_handler1.handle_event = Mock()
        self.sub_handler2.handle_event = Mock()

        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.add_handler(self.sub_handler2)

        event = Event("TEST", time.time(), "source")
        self.composite_handler.handle_event(event)

        self.sub_handler1.handle_event.assert_called_once_with(event)
        self.sub_handler2.handle_event.assert_called_once_with(event)
        assert self.composite_handler.event_count == 2

    def test_handle_event_with_disabled_handler(self):
        """Test handling events with disabled composite handler."""
        self.sub_handler1.can_handle.return_value = True
        self.sub_handler1.handle_event = Mock()

        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.disable()

        event = Event("TEST", time.time(), "source")
        self.composite_handler.handle_event(event)

        self.sub_handler1.handle_event.assert_not_called()
        assert self.composite_handler.event_count == 0

    def test_handle_event_with_filtered_handlers(self):
        """Test handling events with filtered handlers."""
        self.sub_handler1.can_handle.return_value = True
        self.sub_handler2.can_handle.return_value = False
        self.sub_handler1.handle_event = Mock()
        self.sub_handler2.handle_event = Mock()

        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.add_handler(self.sub_handler2)

        event = Event("TEST", time.time(), "source")
        self.composite_handler.handle_event(event)

        self.sub_handler1.handle_event.assert_called_once_with(event)
        self.sub_handler2.handle_event.assert_not_called()
        assert self.composite_handler.event_count == 1

    def test_handle_event_with_handler_error(self):
        """Test handling events with handler errors."""
        self.sub_handler1.can_handle.return_value = True
        self.sub_handler2.can_handle.return_value = True
        self.sub_handler1.handle_event = Mock(side_effect=Exception("Handler error"))
        self.sub_handler2.handle_event = Mock()

        self.composite_handler.add_handler(self.sub_handler1)
        self.composite_handler.add_handler(self.sub_handler2)

        event = Event("TEST", time.time(), "source")
        self.composite_handler.handle_event(event)

        self.sub_handler1.handle_event.assert_called_once_with(event)
        self.sub_handler2.handle_event.assert_called_once_with(event)
        assert self.composite_handler.event_count == 2
        assert self.composite_handler.error_count == 1


class TestCreateEventHandler:
    """Test cases for create_event_handler factory function."""

    def test_create_market_data_handler(self):
        """Test creating market data handler."""
        handler = create_event_handler("market_data", "test_handler")

        assert isinstance(handler, MarketDataHandler)
        assert handler.name == "test_handler"

    def test_create_signal_handler(self):
        """Test creating signal handler."""
        handler = create_event_handler("signal", "test_handler")

        assert isinstance(handler, SignalHandler)
        assert handler.name == "test_handler"

    def test_create_order_handler(self):
        """Test creating order handler."""
        handler = create_event_handler("order", "test_handler")

        assert isinstance(handler, OrderHandler)
        assert handler.name == "test_handler"

    def test_create_portfolio_handler(self):
        """Test creating portfolio handler."""
        handler = create_event_handler("portfolio", "test_handler")

        assert isinstance(handler, PortfolioHandler)
        assert handler.name == "test_handler"

    def test_create_risk_handler(self):
        """Test creating risk handler."""
        handler = create_event_handler("risk", "test_handler")

        assert isinstance(handler, RiskHandler)
        assert handler.name == "test_handler"

    def test_create_strategy_handler(self):
        """Test creating strategy handler."""
        handler = create_event_handler("strategy", "test_handler")

        assert isinstance(handler, StrategyHandler)
        assert handler.name == "test_handler"

    def test_create_composite_handler(self):
        """Test creating composite handler."""
        handler = create_event_handler("composite", "test_handler")

        assert isinstance(handler, CompositeEventHandler)
        assert handler.name == "test_handler"

    def test_create_invalid_handler_type(self):
        """Test creating invalid handler type."""
        with pytest.raises(ValueError, match="Unknown handler type"):
            create_event_handler("invalid_type", "test_handler")
