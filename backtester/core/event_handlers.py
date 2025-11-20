"""Event handlers for the event-driven backtesting system.

This module provides abstract base classes and concrete implementations for
handling various types of events in the backtesting framework.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from .event_bus import Event
from .events import (
    MarketDataEvent,
    OrderEvent,
    OrderStatus,
    PortfolioUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
    StrategyEvent,
)


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    def __init__(self, name: str, logger: logging.Logger | None = None):
        """Initialize the event handler.

        Args:
            name: Handler name for identification
            logger: Logger instance for handler operations
        """
        self.name = name
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enabled = True
        self.event_count = 0
        self.error_count = 0

        self.logger.info(f"Initialized handler: {self.name}")

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle an incoming event.

        Args:
            event: Event to handle
        """
        pass

    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the given event.

        Args:
            event: Event to check

        Returns:
            True if handler can process the event, False otherwise
        """
        return True

    def enable(self) -> None:
        """Enable this handler."""
        self.enabled = True
        self.logger.info(f"Enabled handler: {self.name}")

    def disable(self) -> None:
        """Disable this handler."""
        self.enabled = False
        self.logger.info(f"Disabled handler: {self.name}")

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dictionary of handler statistics
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'event_count': self.event_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.event_count, 1),
        }


class MarketDataHandler(EventHandler):
    """Handler for market data events."""

    def __init__(self, name: str = "market_data_handler", logger: logging.Logger | None = None):
        """Initialize the market data handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.processed_symbols: set[str] = set()
        self.last_prices: dict[str, float] = {}

    def handle_event(self, event: Event) -> None:
        """Handle market data events.

        Args:
            event: Market data event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, MarketDataEvent):
            self.logger.warning(f"Received non-market data event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process market data
            self._process_market_data(event)

            # Update last known prices
            self.last_prices[event.symbol] = event.close_price
            self.processed_symbols.add(event.symbol)

            self.logger.debug(f"Processed market data for {event.symbol}: {event.close_price}")

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing market data for {event.symbol}: {e}")

    def _process_market_data(self, event: MarketDataEvent) -> None:
        """Process market data (to be overridden by subclasses)."""
        # Base implementation - can be extended by subclasses
        pass

    def get_last_price(self, symbol: str) -> float | None:
        """Get the last processed price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Last price or None if not available
        """
        return self.last_prices.get(symbol)

    def get_processed_symbols(self) -> list[str]:
        """Get list of processed symbols.

        Returns:
            List of processed symbol names
        """
        return list(self.processed_symbols)


class SignalHandler(EventHandler):
    """Handler for trading signal events."""

    def __init__(self, name: str = "signal_handler", logger: logging.Logger | None = None):
        """Initialize the signal handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.processed_signals: list[dict[str, Any]] = []
        self.signal_stats: dict[str, int] = {}

    def handle_event(self, event: Event) -> None:
        """Handle signal events.

        Args:
            event: Signal event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, SignalEvent):
            self.logger.warning(f"Received non-signal event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process signal
            signal_data = self._process_signal(event)
            self.processed_signals.append(signal_data)

            # Update signal statistics
            signal_type = event.signal_type.value
            self.signal_stats[signal_type] = self.signal_stats.get(signal_type, 0) + 1

            self.logger.info(f"Processed signal: {event.signal_type.value} for {event.symbol}")

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing signal for {event.symbol}: {e}")

    def _process_signal(self, event: SignalEvent) -> dict[str, Any]:
        """Process trading signal (to be overridden by subclasses)."""
        # Base implementation - store signal data
        return {
            'symbol': event.symbol,
            'signal_type': event.signal_type.value,
            'strength': event.strength,
            'confidence': event.confidence,
            'timestamp': event.timestamp,
            'metadata': event.metadata,
        }

    def get_signal_stats(self) -> dict[str, int]:
        """Get signal processing statistics.

        Returns:
            Dictionary of signal type counts
        """
        return self.signal_stats.copy()

    def get_processed_signals(self) -> list[dict[str, Any]]:
        """Get list of processed signals.

        Returns:
            List of processed signal data
        """
        return self.processed_signals.copy()


class OrderHandler(EventHandler):
    """Handler for order events."""

    def __init__(self, name: str = "order_handler", logger: logging.Logger | None = None):
        """Initialize the order handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.active_orders: dict[str, OrderEvent] = {}
        self.completed_orders: list[OrderEvent] = []
        self.order_stats: dict[str, int] = {}

    def handle_event(self, event: Event) -> None:
        """Handle order events.

        Args:
            event: Order event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, OrderEvent):
            self.logger.warning(f"Received non-order event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process order
            self._process_order(event)

            # Update order statistics
            order_status = event.status.value
            self.order_stats[order_status] = self.order_stats.get(order_status, 0) + 1

            self.logger.info(f"Processed order: {event.order_id} - {event.status.value}")

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing order {event.order_id}: {e}")

    def _process_order(self, event: OrderEvent) -> None:
        """Process order (to be overridden by subclasses)."""
        # Update order status
        if event.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            # Move to completed orders
            if event.order_id in self.active_orders:
                del self.active_orders[event.order_id]
            self.completed_orders.append(event)
        else:
            # Keep in active orders
            self.active_orders[event.order_id] = event

    def get_active_orders(self) -> dict[str, OrderEvent]:
        """Get active orders.

        Returns:
            Dictionary of active orders
        """
        return self.active_orders.copy()

    def get_completed_orders(self) -> list[OrderEvent]:
        """Get completed orders.

        Returns:
            List of completed orders
        """
        return self.completed_orders.copy()

    def get_order_stats(self) -> dict[str, int]:
        """Get order processing statistics.

        Returns:
            Dictionary of order status counts
        """
        return self.order_stats.copy()


class PortfolioHandler(EventHandler):
    """Handler for portfolio update events."""

    def __init__(self, name: str = "portfolio_handler", logger: logging.Logger | None = None):
        """Initialize the portfolio handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.portfolio_history: list[PortfolioUpdateEvent] = []
        self.portfolio_metrics: dict[str, Any] = {}

    def handle_event(self, event: Event) -> None:
        """Handle portfolio update events.

        Args:
            event: Portfolio update event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, PortfolioUpdateEvent):
            self.logger.warning(f"Received non-portfolio event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process portfolio update
            self._process_portfolio_update(event)

            self.logger.debug(f"Processed portfolio update for {event.portfolio_id}")

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing portfolio update for {event.portfolio_id}: {e}")

    def _process_portfolio_update(self, event: PortfolioUpdateEvent) -> None:
        """Process portfolio update (to be overridden by subclasses)."""
        # Add to history
        self.portfolio_history.append(event)

        # Update metrics
        self.portfolio_metrics = {
            'total_value': event.total_value,
            'cash_balance': event.cash_balance,
            'positions_value': event.positions_value,
            'last_update': event.timestamp,
        }

        # Add custom metrics if available
        if event.metrics:
            self.portfolio_metrics.update(event.metrics)

    def get_portfolio_history(self) -> list[PortfolioUpdateEvent]:
        """Get portfolio update history.

        Returns:
            List of portfolio update events
        """
        return self.portfolio_history.copy()

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current portfolio metrics.

        Returns:
            Dictionary of current portfolio metrics
        """
        return self.portfolio_metrics.copy()


class RiskHandler(EventHandler):
    """Handler for risk alert events."""

    def __init__(self, name: str = "risk_handler", logger: logging.Logger | None = None):
        """Initialize the risk handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.active_alerts: list[RiskAlertEvent] = []
        self.alert_history: list[RiskAlertEvent] = []
        self.alert_stats: dict[str, int] = {}

    def handle_event(self, event: Event) -> None:
        """Handle risk alert events.

        Args:
            event: Risk alert event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, RiskAlertEvent):
            self.logger.warning(f"Received non-risk event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process risk alert
            self._process_risk_alert(event)

            # Update alert statistics
            risk_level = event.risk_level.value
            self.alert_stats[risk_level] = self.alert_stats.get(risk_level, 0) + 1

            self.logger.warning(f"Processed risk alert: {event.risk_level.value} - {event.message}")

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing risk alert {event.alert_id}: {e}")

    def _process_risk_alert(self, event: RiskAlertEvent) -> None:
        """Process risk alert (to be overridden by subclasses)."""
        # Add to active alerts
        self.active_alerts.append(event)

        # Add to history
        self.alert_history.append(event)

        # Clean up old alerts (keep last 100)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

    def get_active_alerts(self) -> list[RiskAlertEvent]:
        """Get active risk alerts.

        Returns:
            List of active risk alerts
        """
        return self.active_alerts.copy()

    def get_alert_history(self) -> list[RiskAlertEvent]:
        """Get risk alert history.

        Returns:
            List of risk alert events
        """
        return self.alert_history.copy()

    def get_alert_stats(self) -> dict[str, int]:
        """Get risk alert statistics.

        Returns:
            Dictionary of risk level counts
        """
        return self.alert_stats.copy()


class StrategyHandler(EventHandler):
    """Handler for strategy events."""

    def __init__(self, name: str = "strategy_handler", logger: logging.Logger | None = None):
        """Initialize the strategy handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.strategy_events: list[StrategyEvent] = []
        self.strategy_stats: dict[str, int] = {}

    def handle_event(self, event: Event) -> None:
        """Handle strategy events.

        Args:
            event: Strategy event to process
        """
        if not self.enabled:
            return

        if not isinstance(event, StrategyEvent):
            self.logger.warning(f"Received non-strategy event: {event.event_type}")
            return

        try:
            self.event_count += 1

            # Process strategy event
            self._process_strategy_event(event)

            # Update strategy statistics
            event_type = event.event_type_enum.value
            self.strategy_stats[event_type] = self.strategy_stats.get(event_type, 0) + 1

            self.logger.info(
                f"Processed strategy event: {event.event_type_enum.value} for {event.strategy_name}"
            )

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing strategy event for {event.strategy_name}: {e}")

    def _process_strategy_event(self, event: StrategyEvent) -> None:
        """Process strategy event (to be overridden by subclasses)."""
        # Add to strategy events
        self.strategy_events.append(event)

        # Clean up old events (keep last 1000)
        if len(self.strategy_events) > 1000:
            self.strategy_events = self.strategy_events[-1000:]

    def get_strategy_events(self) -> list[StrategyEvent]:
        """Get strategy events.

        Returns:
            List of strategy events
        """
        return self.strategy_events.copy()

    def get_strategy_stats(self) -> dict[str, int]:
        """Get strategy event statistics.

        Returns:
            Dictionary of event type counts
        """
        return self.strategy_stats.copy()


# Composite handler for handling multiple event types
class CompositeEventHandler(EventHandler):
    """Handler that can handle multiple event types using sub-handlers."""

    def __init__(self, name: str = "composite_handler", logger: logging.Logger | None = None):
        """Initialize the composite event handler.

        Args:
            name: Handler name
            logger: Logger instance
        """
        super().__init__(name, logger)
        self.handlers: list[EventHandler] = []

    def add_handler(self, handler: EventHandler) -> None:
        """Add a sub-handler.

        Args:
            handler: Handler to add
        """
        self.handlers.append(handler)
        self.logger.info(f"Added handler: {handler.name}")

    def remove_handler(self, handler: EventHandler) -> bool:
        """Remove a sub-handler.

        Args:
            handler: Handler to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.info(f"Removed handler: {handler.name}")
            return True
        return False

    def handle_event(self, event: Event) -> None:
        """Handle event by delegating to appropriate sub-handlers.

        Args:
            event: Event to process
        """
        if not self.enabled:
            return

        for handler in self.handlers:
            if handler.can_handle(event):
                try:
                    handler.handle_event(event)
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Error in sub-handler {handler.name}: {e}")
                finally:
                    self.event_count += 1

    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the given event.

        Args:
            event: Event to check

        Returns:
            True if handler can process the event, False otherwise
        """
        # Composite handler can handle any event that at least one sub-handler can handle
        return any(handler.can_handle(event) for handler in self.handlers)


# Factory function for creating event handlers
def create_event_handler(
    handler_type: str, name: str, logger: logging.Logger | None = None
) -> (
    MarketDataHandler
    | SignalHandler
    | OrderHandler
    | PortfolioHandler
    | RiskHandler
    | StrategyHandler
    | CompositeEventHandler
):
    """Create an event handler of the specified type.

    Args:
        handler_type: Type of handler to create
        name: Handler name
        logger: Logger instance

    Returns:
        Created event handler

    Raises:
        ValueError: If handler type is not recognized
    """
    handler_classes = {
        'market_data': MarketDataHandler,
        'signal': SignalHandler,
        'order': OrderHandler,
        'portfolio': PortfolioHandler,
        'risk': RiskHandler,
        'strategy': StrategyHandler,
        'composite': CompositeEventHandler,
    }

    if handler_type not in handler_classes:
        raise ValueError(f"Unknown handler type: {handler_type}")

    handler_class = handler_classes[handler_type]

    # Skip abstract class for factory function
    if hasattr(handler_class, '__abstractmethods__') and handler_class.__abstractmethods__:
        raise ValueError(f"Cannot create abstract handler type: {handler_type}")

    # Create and return the concrete handler
    if handler_type == 'market_data':
        return MarketDataHandler(name, logger)
    elif handler_type == 'signal':
        return SignalHandler(name, logger)
    elif handler_type == 'order':
        return OrderHandler(name, logger)
    elif handler_type == 'portfolio':
        return PortfolioHandler(name, logger)
    elif handler_type == 'risk':
        return RiskHandler(name, logger)
    elif handler_type == 'strategy':
        return StrategyHandler(name, logger)
    elif handler_type == 'composite':
        return CompositeEventHandler(name, logger)
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")
