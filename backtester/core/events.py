"""Core event types for the event-driven backtesting system.

This module defines all the standard event types used throughout the backtesting
framework, including market data, signals, orders, portfolio updates, risk alerts,
and strategy events.
"""

import time
from enum import Enum
from typing import Any

from .event_bus import Event as Event


class MarketDataType(Enum):
    """Types of market data events."""

    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    TICK = "tick"
    AGGREGATED = "aggregated"


class SignalType(Enum):
    """Types of trading signals."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    INCREASE_POSITION = "INCREASE_POSITION"
    DECREASE_POSITION = "DECREASE_POSITION"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


class OrderType(Enum):
    """Types of orders."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order sides."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderStatus(Enum):
    """Order execution statuses."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RiskLevel(Enum):
    """Risk alert levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StrategyEventType(Enum):
    """Types of strategy events."""

    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    REBALANCE_REQUESTED = "REBALANCE_REQUESTED"
    STRATEGY_UPDATE = "STRATEGY_UPDATE"
    STRATEGY_ERROR = "STRATEGY_ERROR"
    STRATEGY_COMPLETE = "STRATEGY_COMPLETE"


class MarketDataEvent(Event):
    """Event containing market data information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        symbol: str,
        data_type: MarketDataType,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        volume: float | None = None,
        timestamp_data: float | None = None,
        bid_price: float | None = None,
        ask_price: float | None = None,
        trade_count: int | None = None,
    ):
        """Initialize market data event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.symbol = symbol
        self.data_type = data_type
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.timestamp_data = timestamp_data
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.trade_count = trade_count

        # Validate market data
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.close_price <= 0:
            raise ValueError("Close price must be positive")

        # Set timestamp data if not provided
        if self.timestamp_data is None:
            self.timestamp_data = self.timestamp


class SignalEvent(Event):
    """Event containing trading signal information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        symbol: str,
        signal_type: SignalType,
        priority: int = 3,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        strength: float = 1.0,
        confidence: float = 1.0,
        suggested_quantity: float | None = None,
        price: float | None = None,
        reason: str | None = None,
    ):
        """Initialize signal event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence
        self.suggested_quantity = suggested_quantity
        self.price = price
        self.reason = reason

        # Validate signal data
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not isinstance(self.signal_type, SignalType):
            raise ValueError("Signal type must be a SignalType enum")
        if not 0 <= self.strength <= 1:
            raise ValueError("Signal strength must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Signal confidence must be between 0 and 1")


class OrderEvent(Event):
    """Event containing order information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        symbol: str,
        order_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        priority: int = 3,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        price: float | None = None,
        status: OrderStatus | str = OrderStatus.PENDING,
        filled_quantity: float = 0.0,
        average_fill_price: float | None = None,
        commission: float = 0.0,
        timestamp_exchange: float | None = None,
        error_message: str | None = None,
    ):
        """Initialize order event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.symbol = symbol
        self.order_id = order_id
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.status: OrderStatus = self._coerce_order_status(status)
        self.filled_quantity = filled_quantity
        self.average_fill_price = average_fill_price
        self.commission = commission
        self.timestamp_exchange = timestamp_exchange
        self.error_message = error_message

        # Validate order data
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.order_id:
            raise ValueError("Order ID cannot be empty")
        if not isinstance(self.side, OrderSide):
            raise ValueError("Order side must be an OrderSide enum")
        if not isinstance(self.order_type, OrderType):
            raise ValueError("Order type must be an OrderType enum")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive if specified")
        if self.filled_quantity < 0:
            raise ValueError("Filled quantity cannot be negative")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")

    @staticmethod
    def _coerce_order_status(status: OrderStatus | str) -> OrderStatus:
        """Convert raw status input into an OrderStatus enum."""
        if isinstance(status, OrderStatus):
            return status
        try:
            return OrderStatus(status.upper())
        except (AttributeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("Order status must be an OrderStatus enum") from exc


class PortfolioUpdateEvent(Event):
    """Event containing portfolio state information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        portfolio_id: str,
        total_value: float,
        cash_balance: float,
        positions_value: float,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        positions: dict[str, dict[str, Any]] | None = None,
        returns: dict[str, float] | None = None,
        metrics: dict[str, Any] | None = None,
        timestamp_portfolio: float | None = None,
    ):
        """Initialize portfolio update event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.portfolio_id = portfolio_id
        self.total_value = total_value
        self.cash_balance = cash_balance
        self.positions_value = positions_value
        self.positions = positions or {}
        self.returns = returns or {}
        self.metrics = metrics or {}
        self.timestamp_portfolio = timestamp_portfolio

        # Validate portfolio data
        if not self.portfolio_id:
            raise ValueError("Portfolio ID cannot be empty")
        if self.total_value < 0:
            raise ValueError("Total value cannot be negative")
        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")
        if self.positions_value < 0:
            raise ValueError("Positions value cannot be negative")


class RiskAlertEvent(Event):
    """Event containing risk alert information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        alert_id: str,
        risk_level: RiskLevel,
        message: str,
        component: str,
        priority: int = 4,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        violations: list[str] | None = None,
        current_value: float | None = None,
        threshold_value: float | None = None,
        suggested_action: str | None = None,
    ):
        """Initialize risk alert event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.alert_id = alert_id
        self.risk_level = risk_level
        self.message = message
        self.component = component
        self.violations = violations or []
        self.current_value = current_value
        self.threshold_value = threshold_value
        self.suggested_action = suggested_action

        # Validate risk alert data
        if not self.alert_id:
            raise ValueError("Alert ID cannot be empty")
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError("Risk level must be a RiskLevel enum")
        if not self.message:
            raise ValueError("Alert message cannot be empty")
        if not self.component:
            raise ValueError("Component name cannot be empty")


class StrategyEvent(Event):
    """Event containing strategy-specific information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        strategy_id: str,
        event_type_enum: StrategyEventType,
        strategy_name: str,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        data: dict[str, Any] | None = None,
        performance_metrics: dict[str, Any] | None = None,
        error_details: dict[str, Any] | None = None,
    ):
        """Initialize strategy event."""
        # Convert string to enum if needed
        if isinstance(event_type_enum, str):
            event_type_enum = StrategyEventType(event_type_enum.upper())

        # Use the enum value as the event_type for consistency
        super().__init__(event_type_enum.value, timestamp, source, priority, metadata, event_id)
        self.strategy_id = strategy_id
        self.event_type_enum = event_type_enum
        self.strategy_name = strategy_name
        self.data = data or {}
        self.performance_metrics = performance_metrics
        self.error_details = error_details

        # Validate strategy event data
        if not self.strategy_id:
            raise ValueError("Strategy ID cannot be empty")
        if not isinstance(self.event_type_enum, StrategyEventType):
            raise ValueError("Event type must be a StrategyEventType enum")
        if not self.strategy_name:
            raise ValueError("Strategy name cannot be empty")


class BacktestEvent(Event):
    """Event containing backtest lifecycle information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        backtest_id: str,
        event_type_name: str,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        progress: float = 0.0,
        total_periods: int | None = None,
        completed_periods: int = 0,
        error_message: str | None = None,
    ):
        """Initialize backtest event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.backtest_id = backtest_id
        self.event_type = event_type_name
        self.progress = progress
        self.total_periods = total_periods
        self.completed_periods = completed_periods
        self.error_message = error_message

        # Validate backtest event data
        if not self.backtest_id:
            raise ValueError("Backtest ID cannot be empty")
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        if not 0 <= self.progress <= 1:
            raise ValueError("Progress must be between 0 and 1")
        if self.completed_periods < 0:
            raise ValueError("Completed periods cannot be negative")


class DataUpdateEvent(Event):
    """Event containing data update information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        data_source: str,
        symbols: list[str],
        update_type: str,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        data_points: int = 0,
        error_message: str | None = None,
    ):
        """Initialize data update event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.data_source = data_source
        self.symbols = symbols
        self.update_type = update_type
        self.data_points = data_points
        self.error_message = error_message

        # Validate data update event data
        if not self.data_source:
            raise ValueError("Data source cannot be empty")
        if not self.symbols:
            raise ValueError("Symbols list cannot be empty")
        if self.data_points < 0:
            raise ValueError("Data points cannot be negative")


class PerformanceEvent(Event):
    """Event containing performance update information."""

    def __init__(
        self,
        event_type: str,
        timestamp: float,
        source: str,
        portfolio_id: str,
        period: str,
        priority: int = 2,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        benchmark_metrics: dict[str, Any] | None = None,
        comparison_metrics: dict[str, Any] | None = None,
    ):
        """Initialize performance event."""
        super().__init__(event_type, timestamp, source, priority, metadata, event_id)
        self.portfolio_id = portfolio_id
        self.period = period
        self.metrics = metrics or {}
        self.benchmark_metrics = benchmark_metrics
        self.comparison_metrics = comparison_metrics

        # Validate performance event data
        if not self.portfolio_id:
            raise ValueError("Portfolio ID cannot be empty")
        if not self.period:
            raise ValueError("Period cannot be empty")


# Factory functions for creating common events
def create_market_data_event(
    symbol: str,
    data: dict[str, Any],
    source: str = "market_data_feed",
    priority: int = 2,
) -> MarketDataEvent:
    """Create a market data event from raw data dictionary."""
    event_time = time.time()
    payload = dict(data)

    data_type_raw = payload.get("data_type", MarketDataType.BAR.value)
    if isinstance(data_type_raw, MarketDataType):
        data_type = data_type_raw
    else:
        try:
            data_type = MarketDataType(str(data_type_raw).lower())
        except ValueError:
            data_type = MarketDataType.BAR

    open_price = float(payload.get("open", payload.get("Open", 0.0)))
    high_price = float(payload.get("high", payload.get("High", open_price)))
    low_price = float(payload.get("low", payload.get("Low", open_price)))
    close_price = float(payload.get("close", payload.get("Close", open_price)))
    volume_value = payload.get("volume", payload.get("Volume"))
    volume = float(volume_value) if volume_value is not None else None
    timestamp_data = payload.get("timestamp")

    symbols_value = payload.get("symbols") or [symbol]
    if isinstance(symbols_value, str):
        symbols: list[str] = [symbols_value]
    else:
        symbols = list(symbols_value)
    if symbol not in symbols:
        symbols.append(symbol)

    metadata = {
        "symbol": symbol,
        "symbols": symbols,
        "bar": {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": timestamp_data or event_time,
        },
        "provenance": {
            "source": source,
            "ingested_at": event_time,
            "data_type": data_type.value,
        },
        "raw": payload,
    }

    if "data_frame" in payload:
        metadata["data_frame"] = payload["data_frame"]

    if "symbol" not in payload:
        payload["symbol"] = symbol
    payload.setdefault("symbols", symbols)

    return MarketDataEvent(
        event_type="MARKET_DATA",
        timestamp=event_time,
        source=source,
        symbol=symbol,
        data_type=data_type,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        timestamp_data=timestamp_data,
        bid_price=payload.get("bid"),
        ask_price=payload.get("ask"),
        trade_count=payload.get("trade_count"),
        metadata=metadata,
        priority=priority,
    )


def create_signal_event(
    symbol: str,
    signal_type: str | SignalType,
    strength: float = 1.0,
    confidence: float = 1.0,
    source: str = "strategy",
    priority: int = 3,
    metadata: dict[str, Any] | None = None,
) -> SignalEvent:
    """Create a signal event."""
    if isinstance(signal_type, str):
        signal_type = SignalType(signal_type.upper())

    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("symbol", symbol)
    metadata_payload.setdefault("symbols", [symbol])
    metadata_payload.setdefault("signal_type", signal_type.value)
    metadata_payload.setdefault("source_strategy", source)

    return SignalEvent(
        event_type="SIGNAL",
        timestamp=time.time(),
        source=source,
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
        confidence=confidence,
        priority=priority,
        metadata=metadata_payload,
    )


def create_order_event(
    symbol: str,
    side: str | OrderSide,
    order_type: str | OrderType,
    quantity: float,
    source: str = "strategy",
    priority: int = 3,
    metadata: dict[str, Any] | None = None,
) -> OrderEvent:
    """Create an order event."""
    if isinstance(side, str):
        side = OrderSide(side.upper())
    if isinstance(order_type, str):
        order_type = OrderType(order_type.upper())

    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("symbol", symbol)
    metadata_payload.setdefault("side", side.value)
    metadata_payload.setdefault("order_type", order_type.value)
    metadata_payload.setdefault("source_strategy", source)

    return OrderEvent(
        event_type="ORDER",
        timestamp=time.time(),
        source=source,
        symbol=symbol,
        order_id=f"order_{int(time.time() * 1000)}",
        side=side,
        order_type=order_type,
        quantity=quantity,
        priority=priority,
        metadata=metadata_payload,
    )


def create_portfolio_update_event(
    portfolio_id: str,
    total_value: float,
    cash_balance: float,
    positions_value: float,
    source: str = "portfolio_manager",
    priority: int = 2,
    metadata: dict[str, Any] | None = None,
) -> PortfolioUpdateEvent:
    """Create a portfolio update event."""
    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault("portfolio_id", portfolio_id)
    metadata_payload.setdefault("total_value", total_value)
    metadata_payload.setdefault("cash_balance", cash_balance)
    metadata_payload.setdefault("positions_value", positions_value)

    return PortfolioUpdateEvent(
        event_type="PORTFOLIO_UPDATE",
        timestamp=time.time(),
        source=source,
        portfolio_id=portfolio_id,
        total_value=total_value,
        cash_balance=cash_balance,
        positions_value=positions_value,
        priority=priority,
        metadata=metadata_payload,
    )


def create_risk_alert_event(
    alert_id: str,
    risk_level: str | RiskLevel,
    message: str,
    component: str,
    source: str = "risk_manager",
    priority: int = 4,
) -> RiskAlertEvent:
    """Create a risk alert event."""
    if isinstance(risk_level, str):
        risk_level = RiskLevel(risk_level.upper())

    event_metadata = {
        "component": component,
        "alert_id": alert_id,
    }

    return RiskAlertEvent(
        event_type="RISK_ALERT",
        timestamp=time.time(),
        source=source,
        alert_id=alert_id,
        risk_level=risk_level,
        message=message,
        component=component,
        priority=priority,
        metadata=event_metadata,
    )


def create_strategy_event(
    strategy_id: str,
    strategy_name: str,
    event_type: str | StrategyEventType,
    source: str = "strategy",
    priority: int = 2,
) -> StrategyEvent:
    """Create a strategy event."""
    if isinstance(event_type, str):
        event_type = StrategyEventType(event_type.upper())

    return StrategyEvent(
        event_type="STRATEGY_EVENT",
        timestamp=time.time(),
        source=source,
        strategy_id=strategy_id,
        event_type_enum=event_type,
        strategy_name=strategy_name,
        priority=priority,
    )
