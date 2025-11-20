"""Tests for the event bus system."""

import time
from unittest.mock import patch

import pytest

from backtester.core.event_bus import Event, EventBus, EventFilter, EventPriority
from backtester.core.events import (
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
    create_market_data_event,
    create_order_event,
    create_signal_event,
)


class TestEvent:
    """Test cases for the Event base class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(event_type="TEST_EVENT", timestamp=time.time(), source="test_source")

        assert event.event_type == "TEST_EVENT"
        assert event.source == "test_source"
        assert event.priority == EventPriority.NORMAL
        assert isinstance(event.metadata, dict)
        assert event.event_id is not None

    def test_event_creation_with_priority(self):
        """Test event creation with custom priority."""
        event = Event(
            event_type="HIGH_PRIORITY",
            timestamp=time.time(),
            source="test_source",
            priority=EventPriority.HIGH,
        )

        assert event.priority == EventPriority.HIGH

    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = Event(
            event_type="TEST_EVENT",
            timestamp=1234567890.0,
            source="test_source",
            priority=EventPriority.HIGH,
            metadata={"key": "value"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "TEST_EVENT"
        assert event_dict["timestamp"] == 1234567890.0
        assert event_dict["source"] == "test_source"
        assert event_dict["priority"] == 3  # HIGH.value
        assert event_dict["metadata"] == {"key": "value"}
        assert event_dict["event_id"] is not None

    def test_event_validation_empty_event_type(self):
        """Test validation fails for empty event type."""
        with pytest.raises(ValueError, match="Event type cannot be empty"):
            Event(event_type="", timestamp=time.time(), source="test_source")

    def test_event_validation_empty_source(self):
        """Test validation fails for empty source."""
        with pytest.raises(ValueError, match="Event source cannot be empty"):
            Event(event_type="TEST", timestamp=time.time(), source="")


class TestEventFilter:
    """Test cases for the EventFilter class."""

    def test_event_filter_creation(self):
        """Test basic event filter creation."""
        event_filter = EventFilter()
        assert event_filter.event_types is None
        assert event_filter.sources is None
        assert event_filter.priority_min is None
        assert event_filter.priority_max is None
        assert event_filter.metadata_filters is None

    def test_event_filter_with_event_types(self):
        """Test event filter with event type restriction."""
        event_filter = EventFilter(event_types={"MARKET_DATA", "SIGNAL"})

        market_data_event = Event("MARKET_DATA", time.time(), "source")
        signal_event = Event("SIGNAL", time.time(), "source")
        order_event = Event("ORDER", time.time(), "source")

        assert event_filter.matches(market_data_event) is True
        assert event_filter.matches(signal_event) is True
        assert event_filter.matches(order_event) is False

    def test_event_filter_with_sources(self):
        """Test event filter with source restriction."""
        event_filter = EventFilter(sources={"market_feed", "strategy"})

        market_event = Event("EVENT", time.time(), "market_feed")
        strategy_event = Event("EVENT", time.time(), "strategy")
        other_event = Event("EVENT", time.time(), "other_source")

        assert event_filter.matches(market_event) is True
        assert event_filter.matches(strategy_event) is True
        assert event_filter.matches(other_event) is False

    def test_event_filter_with_priority_range(self):
        """Test event filter with priority range restriction."""
        event_filter = EventFilter(
            priority_min=EventPriority.NORMAL, priority_max=EventPriority.HIGH
        )

        low_event = Event("EVENT", time.time(), "source", EventPriority.LOW)
        normal_event = Event("EVENT", time.time(), "source", EventPriority.NORMAL)
        high_event = Event("EVENT", time.time(), "source", EventPriority.HIGH)
        critical_event = Event("EVENT", time.time(), "source", EventPriority.CRITICAL)

        assert event_filter.matches(low_event) is False
        assert event_filter.matches(normal_event) is True
        assert event_filter.matches(high_event) is True
        assert event_filter.matches(critical_event) is False

    def test_event_filter_with_metadata(self):
        """Test event filter with metadata filtering."""
        event_filter = EventFilter(metadata_filters={"symbol": "AAPL", "side": "BUY"})

        matching_event = Event(
            "EVENT", time.time(), "source", metadata={"symbol": "AAPL", "side": "BUY"}
        )
        non_matching_event = Event(
            "EVENT", time.time(), "source", metadata={"symbol": "GOOGL", "side": "BUY"}
        )
        partial_match_event = Event("EVENT", time.time(), "source", metadata={"symbol": "AAPL"})

        assert event_filter.matches(matching_event) is True
        assert event_filter.matches(non_matching_event) is False
        assert event_filter.matches(partial_match_event) is False

    def test_event_filter_combined(self):
        """Test event filter with multiple criteria."""
        event_filter = EventFilter(
            event_types={"SIGNAL"},
            sources={"strategy"},
            priority_min=EventPriority.NORMAL,
            metadata_filters={"symbol": "AAPL"},
        )

        matching_event = Event(
            "SIGNAL", time.time(), "strategy", EventPriority.HIGH, {"symbol": "AAPL"}
        )

        wrong_type_event = Event(
            "ORDER", time.time(), "strategy", EventPriority.HIGH, {"symbol": "AAPL"}
        )

        wrong_source_event = Event(
            "SIGNAL", time.time(), "market_feed", EventPriority.HIGH, {"symbol": "AAPL"}
        )

        wrong_priority_event = Event(
            "SIGNAL", time.time(), "strategy", EventPriority.LOW, {"symbol": "AAPL"}
        )

        wrong_metadata_event = Event(
            "SIGNAL", time.time(), "strategy", EventPriority.HIGH, {"symbol": "GOOGL"}
        )

        assert event_filter.matches(matching_event) is True
        assert event_filter.matches(wrong_type_event) is False
        assert event_filter.matches(wrong_source_event) is False
        assert event_filter.matches(wrong_priority_event) is False
        assert event_filter.matches(wrong_metadata_event) is False


class TestEventBus:
    """Test cases for the EventBus class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()
        self.handler_called = False
        self.received_event = None

    def test_event_bus_initialization(self):
        """Test event bus initialization."""
        assert self.event_bus.logger is not None
        assert self.event_bus._filtered_subscriptions == []
        assert self.event_bus._event_queue == []
        assert self.event_bus._processing is False
        assert self.event_bus._published_events == 0
        assert self.event_bus._processed_events == 0
        assert self.event_bus._dropped_events == 0

    def test_subscribe_without_filter(self):
        """Test subscribing without a filter."""

        def handler(event):
            self.handler_called = True
            self.received_event = event

        subscription_id = self.event_bus.subscribe(handler)

        assert subscription_id is not None
        assert len(self.event_bus._filtered_subscriptions) == 1
        assert self.event_bus._filtered_subscriptions[0].handler == handler

    def test_subscribe_with_filter(self):
        """Test subscribing with a filter."""

        def handler(event):
            self.handler_called = True
            self.received_event = event

        event_filter = EventFilter(event_types={"MARKET_DATA"})
        subscription_id = self.event_bus.subscribe(handler, event_filter)

        assert subscription_id is not None
        assert len(self.event_bus._filtered_subscriptions) == 1
        assert self.event_bus._filtered_subscriptions[0].event_filter == event_filter
        assert self.event_bus._filtered_subscriptions[0].handler == handler

    def test_subscribe_invalid_handler(self):
        """Test subscribing with invalid handler."""
        with pytest.raises(ValueError, match="Handler must be callable"):
            self.event_bus.subscribe("not_a_function")

    def test_unsubscribe(self):
        """Test unsubscribing."""

        def handler(event):
            pass

        subscription_id = self.event_bus.subscribe(handler)
        result = self.event_bus.unsubscribe(subscription_id)

        assert result is True
        assert len(self.event_bus._filtered_subscriptions) == 0

    def test_unsubscribe_requires_exact_identifier(self):
        """Ensure unsubscribe only removes the matching subscription ID."""
        calls: list[str] = []

        def handler_a(event):
            calls.append("a")

        def handler_b(event):
            calls.append("b")

        sub_a = self.event_bus.subscribe(handler_a)
        sub_b = self.event_bus.subscribe(handler_b)

        # Using a non-existent ID should leave subscriptions untouched
        assert self.event_bus.unsubscribe("sub_999") is False
        assert len(self.event_bus._filtered_subscriptions) == 2

        assert self.event_bus.unsubscribe(sub_a) is True
        assert len(self.event_bus._filtered_subscriptions) == 1
        assert self.event_bus._filtered_subscriptions[0].subscription_id == sub_b

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing non-existent subscription."""
        result = self.event_bus.unsubscribe("nonexistent_id")
        assert result is False

    def test_publish_event(self):
        """Test publishing an event."""

        def handler(event):
            self.handler_called = True
            self.received_event = event

        self.event_bus.subscribe(handler)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event)

        assert self.handler_called is True
        assert self.received_event == event
        assert self.event_bus._published_events == 1
        assert self.event_bus._processed_events == 1

    def test_publish_invalid_event(self):
        """Test publishing invalid event."""
        with pytest.raises(ValueError, match="Event must be an instance of Event class"):
            self.event_bus.publish("not_an_event")

    def test_publish_immediate(self):
        """Test publishing immediate event."""

        def handler(event):
            self.handler_called = True

        self.event_bus.subscribe(handler)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event, immediate=True)

        assert self.handler_called is True
        assert self.event_bus._event_queue == []

    def test_publish_queued(self):
        """Test publishing queued event."""

        def handler(event):
            self.handler_called = True

        self.event_bus.subscribe(handler)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event, immediate=False)

        assert self.handler_called is True
        assert self.event_bus._event_queue == []
        # Event should be processed immediately even when queued due to single event

    def test_publish_respects_subscription_order(self):
        """Validate that handlers are invoked in subscribe order."""
        fan_out_order: list[str] = []

        def handler_a(event):
            fan_out_order.append("a")

        def handler_b(event):
            fan_out_order.append("b")

        def handler_c(event):
            fan_out_order.append("c")

        self.event_bus.subscribe(handler_a)
        self.event_bus.subscribe(handler_b)
        self.event_bus.subscribe(handler_c)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event)

        assert fan_out_order == ["a", "b", "c"]

    def test_register_handler(self):
        """Test registering type-specific handler."""

        def handler(event):
            self.handler_called = True

        handler_id = self.event_bus.register_handler("TEST_EVENT", handler)

        assert handler_id is not None
        assert "TEST_EVENT" in self.event_bus._event_handlers
        assert handler in self.event_bus._event_handlers["TEST_EVENT"]

    def test_register_handler_invalid(self):
        """Test registering invalid handler."""
        with pytest.raises(ValueError, match="Handler must be callable"):
            self.event_bus.register_handler("TEST_EVENT", "not_a_function")

    def test_deregister_handler(self):
        """Test deregistering handler."""

        def handler(event):
            pass

        handler_id = self.event_bus.register_handler("TEST_EVENT", handler)
        result = self.event_bus.deregister_handler("TEST_EVENT", handler_id)

        assert result is True
        assert len(self.event_bus._event_handlers["TEST_EVENT"]) == 0

    def test_deregister_nonexistent_handler(self):
        """Test deregistering non-existent handler."""
        result = self.event_bus.deregister_handler("TEST_EVENT", "nonexistent_id")
        assert result is False

    def test_metadata_filter_list_membership(self):
        """Ensure metadata filters treat lists as sets for membership tests."""
        flt = EventFilter(metadata_filters={'symbols': {'AAPL', 'MSFT'}})
        event = Event("MARKET_DATA", time.time(), "src", metadata={'symbols': ['MSFT']})
        assert flt.matches(event) is True

        event.metadata['symbols'] = ['TSLA']
        assert flt.matches(event) is False

    def test_metadata_filter_wildcard(self):
        """The '*' metadata filter should match all payloads."""
        flt = EventFilter(metadata_filters={'symbols': '*'})
        event = Event("MARKET_DATA", time.time(), "src", metadata={'symbols': ['ANY']})
        assert flt.matches(event) is True

    def test_handler_error_handling(self):
        """Test error handling in event handlers."""

        def failing_handler(event):
            raise Exception("Handler error")

        def working_handler(event):
            self.handler_called = True

        self.event_bus.subscribe(failing_handler)
        self.event_bus.subscribe(working_handler)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event)

        # Working handler should still be called
        assert self.handler_called is True
        # Error should be logged but not propagate
        assert self.event_bus._dropped_events == 0

    def test_get_metrics_reports_queue_and_counts(self):
        """get_metrics should expose queue depth and processed counts."""
        event = Event(event_type="TEST_EVENT", timestamp=time.time(), source="unittest")

        with patch.object(self.event_bus, "_process_queued_events") as mock_process:
            mock_process.side_effect = lambda: None
            self.event_bus.publish(event)

        metrics = self.event_bus.get_metrics()
        assert metrics['published_events'] == 1
        assert metrics['queue_depth'] == 1
        assert metrics['processed_events'] == 0

        self.event_bus._process_queued_events()
        metrics = self.event_bus.get_metrics()
        assert metrics['queue_depth'] == 0
        assert metrics['processed_events'] >= 1

    def test_clear_queue(self):
        """Test clearing event queue."""
        # Add some events to queue
        event1 = Event("TEST_EVENT", time.time(), "source1")
        event2 = Event("TEST_EVENT", time.time(), "source2")

        self.event_bus._event_queue = [event1, event2]

        cleared_count = self.event_bus.clear_queue()

        assert cleared_count == 2
        assert len(self.event_bus._event_queue) == 0

    def test_get_metrics(self):
        """Test getting event bus metrics."""

        def handler(event):
            pass

        self.event_bus.subscribe(handler)

        event = Event("TEST_EVENT", time.time(), "test_source")
        self.event_bus.publish(event)

        metrics = self.event_bus.get_metrics()

        assert metrics["published_events"] == 1
        assert metrics["processed_events"] == 1
        assert metrics["dropped_events"] == 0
        assert metrics["queued_events"] == 0
        assert metrics["active_subscriptions"] == 1
        assert metrics["registered_handlers"] == 0

    def test_reset_metrics(self):
        """Test resetting event bus metrics."""

        def handler(event):
            pass

        self.event_bus.subscribe(handler)
        self.event_bus.publish(Event("TEST_EVENT", time.time(), "test_source"))

        self.event_bus.reset_metrics()

        metrics = self.event_bus.get_metrics()
        assert metrics["published_events"] == 0
        assert metrics["processed_events"] == 0
        assert metrics["dropped_events"] == 0

    def test_get_status(self):
        """Test getting event bus status."""
        status = self.event_bus.get_status()

        assert status["initialized"] is True
        assert status["processing"] is False
        assert status["queue_size"] == 0
        assert status["subscriptions_count"] == 0
        assert status["handlers_count"] == 0
        assert "metrics" in status


# class TestEventBusAsync:
#     """Test cases for async event bus functionality."""

#     @pytest.mark.asyncio
#     async def test_publish_async(self):
#         """Test async event publishing."""
#         event_bus = EventBus()
#         handler_called = False
#         received_event = None

#         def handler(event):
#             nonlocal handler_called, received_event
#             handler_called = True
#             received_event = event

#         event_bus.subscribe(handler)

#         event = Event("ASYNC_TEST", time.time(), "async_source")
#         await event_bus.publish_async(event)

#         # Give some time for async processing
#         await asyncio.sleep(0.01)

#         assert handler_called is True
#         assert received_event == event

#     @pytest.mark.asyncio
#     async def test_multiple_async_publish(self):
#         """Test multiple async event publishing."""
#         event_bus = EventBus()
#         received_events = []

#         def handler(event):
#             received_events.append(event)

#         event_bus.subscribe(handler)

#         # Publish multiple events
#         for i in range(5):
#             event = Event(f"ASYNC_TEST_{i}", time.time(), "async_source")
#             await event_bus.publish_async(event)

#         # Give some time for async processing
#         await asyncio.sleep(0.01)

#         assert len(received_events) == 5


class TestEventBusIntegration:
    """Integration tests for event bus with specific event types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.event_bus = EventBus()

    def test_market_data_event_flow(self):
        """Test market data event processing flow."""
        received_events = []

        def market_data_handler(event):
            if isinstance(event, MarketDataEvent):
                received_events.append(event)

        self.event_bus.subscribe(market_data_handler)

        # Create and publish market data event
        market_data = create_market_data_event(
            symbol="AAPL",
            data={"open": 150.0, "high": 155.0, "low": 149.0, "close": 152.0, "volume": 1000000},
        )

        self.event_bus.publish(market_data)

        assert len(received_events) == 1
        assert received_events[0].symbol == "AAPL"
        assert received_events[0].close_price == 152.0

    def test_signal_event_flow(self):
        """Test signal event processing flow."""
        received_signals = []

        def signal_handler(event):
            if isinstance(event, SignalEvent):
                received_signals.append(event)

        self.event_bus.subscribe(signal_handler)

        # Create and publish signal event
        signal_event = create_signal_event(
            symbol="AAPL", signal_type="BUY", strength=0.8, confidence=0.9
        )

        self.event_bus.publish(signal_event)

        assert len(received_signals) == 1
        assert received_signals[0].symbol == "AAPL"
        assert received_signals[0].signal_type.value == "BUY"
        assert received_signals[0].strength == 0.8
        assert received_signals[0].confidence == 0.9

    def test_order_event_flow(self):
        """Test order event processing flow."""
        received_orders = []

        def order_handler(event):
            if isinstance(event, OrderEvent):
                received_orders.append(event)

        self.event_bus.subscribe(order_handler)

        # Create and publish order event
        order_event = create_order_event(
            symbol="AAPL", side="BUY", order_type="MARKET", quantity=100.0
        )

        self.event_bus.publish(order_event)

        assert len(received_orders) == 1
        assert received_orders[0].symbol == "AAPL"
        assert received_orders[0].side.value == "BUY"
        assert received_orders[0].order_type.value == "MARKET"
        assert received_orders[0].quantity == 100.0

    def test_filtered_subscription(self):
        """Test filtered subscription functionality."""
        market_events = []
        signal_events = []

        def market_handler(event):
            if isinstance(event, MarketDataEvent):
                market_events.append(event)

        def signal_handler(event):
            if isinstance(event, SignalEvent):
                signal_events.append(event)

        # Subscribe with filters
        market_filter = EventFilter(event_types={"MARKET_DATA"})
        signal_filter = EventFilter(event_types={"SIGNAL"})

        self.event_bus.subscribe(market_handler, market_filter)
        self.event_bus.subscribe(signal_handler, signal_filter)

        # Publish both types of events
        market_event = create_market_data_event("AAPL", {"close": 150.0})
        signal_event = create_signal_event("AAPL", "BUY")

        self.event_bus.publish(market_event)
        self.event_bus.publish(signal_event)

        assert len(market_events) == 1
        assert len(signal_events) == 1
        assert market_events[0].symbol == "AAPL"
        assert signal_events[0].symbol == "AAPL"
