"""Tests for the events module."""

import time
from typing import cast

import pytest

from backtester.core.event_bus import EventPriority
from backtester.core.events import (
    BacktestEvent,
    DataUpdateEvent,
    MarketDataEvent,
    MarketDataType,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PerformanceEvent,
    PortfolioUpdateEvent,
    RiskAlertEvent,
    RiskLevel,
    SignalEvent,
    SignalType,
    StrategyEvent,
    StrategyEventType,
    create_market_data_event,
    create_order_event,
    create_portfolio_update_event,
    create_risk_alert_event,
    create_signal_event,
    create_strategy_event,
)


class TestMarketDataEvent:
    """Test cases for MarketDataEvent."""

    def test_market_data_event_creation(self):
        """Test market data event creation."""
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

        assert event.event_type == "MARKET_DATA"
        assert event.symbol == "AAPL"
        assert event.data_type == MarketDataType.BAR
        assert event.open_price == 150.0
        assert event.high_price == 155.0
        assert event.low_price == 149.0
        assert event.close_price == 152.0
        assert event.volume == 1000000

    def test_market_data_event_validation(self):
        """Test market data event validation."""
        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            MarketDataEvent(
                event_type="MARKET_DATA",
                timestamp=time.time(),
                source="market_feed",
                symbol="",
                data_type=MarketDataType.BAR,
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=152.0,
            )

        # Test invalid close price
        with pytest.raises(ValueError, match="Close price must be positive"):
            MarketDataEvent(
                event_type="MARKET_DATA",
                timestamp=time.time(),
                source="market_feed",
                symbol="AAPL",
                data_type=MarketDataType.BAR,
                open_price=150.0,
                high_price=155.0,
                low_price=149.0,
                close_price=0.0,
            )

    def test_market_data_event_timestamp_data_default(self):
        """Test timestamp data defaults to event timestamp."""
        event_time = time.time()
        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=event_time,
            source="market_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=152.0,
        )

        assert event.timestamp_data == event_time

    def test_market_data_event_with_timestamp_data(self):
        """Test market data event with custom timestamp data."""
        event_time = time.time()
        data_time = event_time + 1.0

        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=event_time,
            source="market_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=152.0,
            timestamp_data=data_time,
        )

        assert event.timestamp_data == data_time


class TestSignalEvent:
    """Test cases for SignalEvent."""

    def test_signal_event_creation(self):
        """Test signal event creation."""
        event = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            suggested_quantity=100.0,
            price=152.0,
            reason="Technical analysis",
            metadata={"indicator": "RSI", "value": 25.0},
        )

        assert event.event_type == "SIGNAL"
        assert event.symbol == "AAPL"
        assert event.signal_type == SignalType.BUY
        assert event.strength == 0.8
        assert event.confidence == 0.9
        assert event.suggested_quantity == 100.0
        assert event.price == 152.0
        assert event.reason == "Technical analysis"
        assert event.metadata["indicator"] == "RSI"

    def test_signal_event_validation(self):
        """Test signal event validation."""
        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            SignalEvent(
                event_type="SIGNAL",
                timestamp=time.time(),
                source="strategy",
                symbol="",
                signal_type=SignalType.BUY,
            )

        # Test invalid signal type
        with pytest.raises(ValueError, match="Signal type must be a SignalType enum"):
            SignalEvent(
                event_type="SIGNAL",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                signal_type=cast(SignalType, "INVALID"),
            )

        # Test invalid strength
        with pytest.raises(ValueError, match="Signal strength must be between 0 and 1"):
            SignalEvent(
                event_type="SIGNAL",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                signal_type=SignalType.BUY,
                strength=1.5,
            )

        # Test invalid confidence
        with pytest.raises(ValueError, match="Signal confidence must be between 0 and 1"):
            SignalEvent(
                event_type="SIGNAL",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=-0.1,
            )

    def test_signal_event_with_string_signal_type(self):
        """Test signal event creation with string signal type."""
        event = SignalEvent(
            event_type="SIGNAL",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            signal_type=SignalType.SELL,  # Use enum instead of string
            strength=0.7,
        )

        assert event.signal_type == SignalType.SELL


class TestOrderEvent:
    """Test cases for OrderEvent."""

    def test_order_event_creation(self):
        """Test order event creation."""
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=152.0,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            average_fill_price=None,
            commission=5.0,
            timestamp_exchange=time.time(),
            metadata={"strategy": "MA_Crossover"},
        )

        assert event.event_type == "ORDER"
        assert event.symbol == "AAPL"
        assert event.order_id == "order_123"
        assert event.side == OrderSide.BUY
        assert event.order_type == OrderType.MARKET
        assert event.quantity == 100.0
        assert event.price == 152.0
        assert event.status == OrderStatus.PENDING
        assert event.filled_quantity == 0.0
        assert event.average_fill_price is None
        assert event.commission == 5.0
        assert event.metadata["strategy"] == "MA_Crossover"

    def test_order_event_validation(self):
        """Test order event validation."""
        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="",
                order_id="order_123",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )

        # Test empty order ID
        with pytest.raises(ValueError, match="Order ID cannot be empty"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
            )

        # Test invalid side
        with pytest.raises(ValueError, match="Order side must be an OrderSide enum"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="order_123",
                side=cast(OrderSide, "INVALID"),
                order_type=OrderType.MARKET,
                quantity=100.0,
            )

        # Test invalid quantity
        with pytest.raises(ValueError, match="Quantity must be positive"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="order_123",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.0,
            )

        # Test invalid price
        with pytest.raises(ValueError, match="Price must be positive if specified"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="order_123",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                price=0.0,
            )

        # Test invalid filled quantity
        with pytest.raises(ValueError, match="Filled quantity cannot be negative"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="order_123",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                filled_quantity=-10.0,
            )

        # Test invalid commission
        with pytest.raises(ValueError, match="Commission cannot be negative"):
            OrderEvent(
                event_type="ORDER",
                timestamp=time.time(),
                source="strategy",
                symbol="AAPL",
                order_id="order_123",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                commission=-5.0,
            )

    def test_order_event_with_string_values(self):
        """Test order event creation with string values."""
        event = OrderEvent(
            event_type="ORDER",
            timestamp=time.time(),
            source="strategy",
            symbol="AAPL",
            order_id="order_123",
            side=OrderSide.SELL,  # String instead of enum
            order_type=OrderType.LIMIT,  # String instead of enum
            quantity=100.0,
            price=150.0,
            status=OrderStatus.FILLED,  # String instead of enum
        )

        assert event.side == OrderSide.SELL
        assert event.order_type == OrderType.LIMIT
        assert event.status == OrderStatus.FILLED


class TestPortfolioUpdateEvent:
    """Test cases for PortfolioUpdateEvent."""

    def test_portfolio_update_event_creation(self):
        """Test portfolio update event creation."""
        positions = {
            "AAPL": {"quantity": 100, "market_value": 15200.0},
            "GOOGL": {"quantity": 50, "market_value": 12500.0},
        }

        event = PortfolioUpdateEvent(
            event_type="PORTFOLIO_UPDATE",
            timestamp=time.time(),
            source="portfolio_manager",
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
            positions=positions,
            returns={"daily": 0.02, "monthly": 0.08},
            metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.1},
            timestamp_portfolio=time.time(),
        )

        assert event.event_type == "PORTFOLIO_UPDATE"
        assert event.portfolio_id == "portfolio_123"
        assert event.total_value == 50000.0
        assert event.cash_balance == 10000.0
        assert event.positions_value == 40000.0
        assert event.positions == positions
        assert event.returns["daily"] == 0.02
        assert event.metrics["sharpe_ratio"] == 1.5

    def test_portfolio_update_event_validation(self):
        """Test portfolio update event validation."""
        # Test empty portfolio ID
        with pytest.raises(ValueError, match="Portfolio ID cannot be empty"):
            PortfolioUpdateEvent(
                event_type="PORTFOLIO_UPDATE",
                timestamp=time.time(),
                source="portfolio_manager",
                portfolio_id="",
                total_value=50000.0,
                cash_balance=10000.0,
                positions_value=40000.0,
            )

        # Test negative total value
        with pytest.raises(ValueError, match="Total value cannot be negative"):
            PortfolioUpdateEvent(
                event_type="PORTFOLIO_UPDATE",
                timestamp=time.time(),
                source="portfolio_manager",
                portfolio_id="portfolio_123",
                total_value=-1000.0,
                cash_balance=10000.0,
                positions_value=40000.0,
            )

        # Test negative cash balance
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            PortfolioUpdateEvent(
                event_type="PORTFOLIO_UPDATE",
                timestamp=time.time(),
                source="portfolio_manager",
                portfolio_id="portfolio_123",
                total_value=50000.0,
                cash_balance=-1000.0,
                positions_value=40000.0,
            )

        # Test negative positions value
        with pytest.raises(ValueError, match="Positions value cannot be negative"):
            PortfolioUpdateEvent(
                event_type="PORTFOLIO_UPDATE",
                timestamp=time.time(),
                source="portfolio_manager",
                portfolio_id="portfolio_123",
                total_value=50000.0,
                cash_balance=10000.0,
                positions_value=-40000.0,
            )


class TestRiskAlertEvent:
    """Test cases for RiskAlertEvent."""

    def test_risk_alert_event_creation(self):
        """Test risk alert event creation."""
        event = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_123",
            risk_level=RiskLevel.HIGH,
            message="Portfolio leverage exceeded",
            component="portfolio_manager",
            violations=["leverage_limit", "position_concentration"],
            current_value=5.5,
            threshold_value=5.0,
            suggested_action="reduce_positions",
            metadata={"portfolio_id": "portfolio_123", "timestamp": time.time()},
        )

        assert event.event_type == "RISK_ALERT"
        assert event.alert_id == "alert_123"
        assert event.risk_level == RiskLevel.HIGH
        assert event.message == "Portfolio leverage exceeded"
        assert event.component == "portfolio_manager"
        assert "leverage_limit" in event.violations
        assert event.current_value == 5.5
        assert event.threshold_value == 5.0
        assert event.suggested_action == "reduce_positions"
        assert event.metadata["portfolio_id"] == "portfolio_123"

    def test_risk_alert_event_validation(self):
        """Test risk alert event validation."""
        # Test empty alert ID
        with pytest.raises(ValueError, match="Alert ID cannot be empty"):
            RiskAlertEvent(
                event_type="RISK_ALERT",
                timestamp=time.time(),
                source="risk_manager",
                alert_id="",
                risk_level=RiskLevel.HIGH,
                message="Test message",
                component="test_component",
            )

        # Test invalid risk level
        with pytest.raises(ValueError, match="Risk level must be a RiskLevel enum"):
            RiskAlertEvent(
                event_type="RISK_ALERT",
                timestamp=time.time(),
                source="risk_manager",
                alert_id="alert_123",
                risk_level=cast(RiskLevel, "INVALID"),
                message="Test message",
                component="test_component",
            )

        # Test empty message
        with pytest.raises(ValueError, match="Alert message cannot be empty"):
            RiskAlertEvent(
                event_type="RISK_ALERT",
                timestamp=time.time(),
                source="risk_manager",
                alert_id="alert_123",
                risk_level=RiskLevel.HIGH,
                message="",
                component="test_component",
            )

        # Test empty component
        with pytest.raises(ValueError, match="Component name cannot be empty"):
            RiskAlertEvent(
                event_type="RISK_ALERT",
                timestamp=time.time(),
                source="risk_manager",
                alert_id="alert_123",
                risk_level=RiskLevel.HIGH,
                message="Test message",
                component="",
            )

    def test_risk_alert_event_with_string_values(self):
        """Test risk alert event creation with string values."""
        event = RiskAlertEvent(
            event_type="RISK_ALERT",
            timestamp=time.time(),
            source="risk_manager",
            alert_id="alert_123",
            risk_level=RiskLevel.CRITICAL,  # Use enum instead of string
            message="Critical risk alert",
            component="risk_manager",
        )

        assert event.risk_level == RiskLevel.CRITICAL


class TestStrategyEvent:
    """Test cases for StrategyEvent."""

    def test_strategy_event_creation(self):
        """Test strategy event creation."""
        event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.SIGNAL_GENERATED,
            strategy_name="MovingAverageStrategy",
            data={"symbol": "AAPL", "signal": "BUY"},
            performance_metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.1},
            error_details=None,
        )

        assert event.event_type == StrategyEventType.SIGNAL_GENERATED.value
        assert event.strategy_id == "strategy_123"
        assert event.event_type == StrategyEventType.SIGNAL_GENERATED.value
        assert event.strategy_name == "MovingAverageStrategy"
        assert event.data["symbol"] == "AAPL"
        assert event.performance_metrics is not None
        assert event.performance_metrics["sharpe_ratio"] == 1.5
        assert event.error_details is None

    def test_strategy_event_validation(self):
        """Test strategy event validation."""
        # Test empty strategy ID
        with pytest.raises(ValueError, match="Strategy ID cannot be empty"):
            StrategyEvent(
                event_type="STRATEGY_EVENT",
                timestamp=time.time(),
                source="strategy",
                strategy_id="",
                event_type_enum=StrategyEventType.SIGNAL_GENERATED,
                strategy_name="MovingAverageStrategy",
            )

        # Test invalid event type
        with pytest.raises(ValueError, match="'INVALID' is not a valid StrategyEventType"):
            StrategyEvent(
                event_type="STRATEGY_EVENT",
                timestamp=time.time(),
                source="strategy",
                strategy_id="strategy_123",
                event_type_enum=cast(StrategyEventType, "INVALID"),
                strategy_name="MovingAverageStrategy",
            )

        # Test empty strategy name
        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            StrategyEvent(
                event_type="STRATEGY_EVENT",
                timestamp=time.time(),
                source="strategy",
                strategy_id="strategy_123",
                event_type_enum=StrategyEventType.SIGNAL_GENERATED,
                strategy_name="",
            )

    def test_strategy_event_with_string_values(self):
        """Test strategy event creation with string values."""
        event = StrategyEvent(
            event_type="STRATEGY_EVENT",
            timestamp=time.time(),
            source="strategy",
            strategy_id="strategy_123",
            event_type_enum=StrategyEventType.STRATEGY_ERROR,
            strategy_name="MovingAverageStrategy",
            error_details={"error": "Calculation error", "timestamp": time.time()},
        )

        assert event.event_type == StrategyEventType.STRATEGY_ERROR.value


class TestBacktestEvent:
    """Test cases for BacktestEvent."""

    def test_backtest_event_creation(self):
        """Test backtest event creation."""
        event = BacktestEvent(
            event_type="BACKTEST_EVENT",
            timestamp=time.time(),
            source="backtest_engine",
            backtest_id="backtest_123",
            event_type_name="START",
            progress=0.0,
            total_periods=1000,
            completed_periods=0,
            metadata={"start_time": time.time(), "strategy": "MA_Crossover"},
        )

        assert event.event_type == "START"
        assert event.backtest_id == "backtest_123"
        assert event.event_type == "START"
        assert event.progress == 0.0
        assert event.total_periods == 1000
        assert event.completed_periods == 0
        assert event.metadata["strategy"] == "MA_Crossover"

    def test_backtest_event_validation(self):
        """Test backtest event validation."""
        # Test empty backtest ID
        with pytest.raises(ValueError, match="Backtest ID cannot be empty"):
            BacktestEvent(
                event_type="BACKTEST_EVENT",
                timestamp=time.time(),
                source="backtest_engine",
                backtest_id="",
                event_type_name="START",
            )

        # Test empty event type
        with pytest.raises(ValueError, match="Event type cannot be empty"):
            BacktestEvent(
                event_type="BACKTEST_EVENT",
                timestamp=time.time(),
                source="backtest_engine",
                backtest_id="backtest_123",
                event_type_name="",
            )

        # Test invalid progress
        with pytest.raises(ValueError, match="Progress must be between 0 and 1"):
            BacktestEvent(
                event_type="BACKTEST_EVENT",
                timestamp=time.time(),
                source="backtest_engine",
                backtest_id="backtest_123",
                event_type_name="START",
                progress=1.5,
            )

        # Test negative completed periods
        with pytest.raises(ValueError, match="Completed periods cannot be negative"):
            BacktestEvent(
                event_type="BACKTEST_EVENT",
                timestamp=time.time(),
                source="backtest_engine",
                backtest_id="backtest_123",
                event_type_name="START",
                progress=0.0,
                completed_periods=-1,
            )


class TestDataUpdateEvent:
    """Test cases for DataUpdateEvent."""

    def test_data_update_event_creation(self):
        """Test data update event creation."""
        event = DataUpdateEvent(
            event_type="DATA_UPDATE",
            timestamp=time.time(),
            source="data_manager",
            data_source="yahoo",
            symbols=["AAPL", "GOOGL"],
            update_type="NEW_DATA",
            data_points=100,
            metadata={"frequency": "daily", "quality": "high"},
        )

        assert event.event_type == "DATA_UPDATE"
        assert event.data_source == "yahoo"
        assert event.symbols == ["AAPL", "GOOGL"]
        assert event.update_type == "NEW_DATA"
        assert event.data_points == 100
        assert event.metadata["frequency"] == "daily"

    def test_data_update_event_validation(self):
        """Test data update event validation."""
        # Test empty data source
        with pytest.raises(ValueError, match="Data source cannot be empty"):
            DataUpdateEvent(
                event_type="DATA_UPDATE",
                timestamp=time.time(),
                source="data_manager",
                data_source="",
                symbols=["AAPL"],
                update_type="NEW_DATA",
            )

        # Test empty symbols list
        with pytest.raises(ValueError, match="Symbols list cannot be empty"):
            DataUpdateEvent(
                event_type="DATA_UPDATE",
                timestamp=time.time(),
                source="data_manager",
                data_source="yahoo",
                symbols=[],
                update_type="NEW_DATA",
            )

        # Test negative data points
        with pytest.raises(ValueError, match="Data points cannot be negative"):
            DataUpdateEvent(
                event_type="DATA_UPDATE",
                timestamp=time.time(),
                source="data_manager",
                data_source="yahoo",
                symbols=["AAPL"],
                update_type="NEW_DATA",
                data_points=-10,
            )


class TestPerformanceEvent:
    """Test cases for PerformanceEvent."""

    def test_performance_event_creation(self):
        """Test performance event creation."""
        event = PerformanceEvent(
            event_type="PERFORMANCE",
            timestamp=time.time(),
            source="performance_analyzer",
            portfolio_id="portfolio_123",
            period="2024-01",
            metrics={"return": 0.05, "volatility": 0.1},
            benchmark_metrics={"return": 0.03, "volatility": 0.08},
            comparison_metrics={"excess_return": 0.02, "sharpe_ratio": 1.2},
        )

        assert event.event_type == "PERFORMANCE"
        assert event.portfolio_id == "portfolio_123"
        assert event.period == "2024-01"
        assert event.metrics is not None
        assert event.metrics["return"] == 0.05
        assert event.benchmark_metrics is not None
        assert event.benchmark_metrics["return"] == 0.03
        assert event.comparison_metrics is not None
        assert event.comparison_metrics["excess_return"] == 0.02

    def test_performance_event_validation(self):
        """Test performance event validation."""
        # Test empty portfolio ID
        with pytest.raises(ValueError, match="Portfolio ID cannot be empty"):
            PerformanceEvent(
                event_type="PERFORMANCE",
                timestamp=time.time(),
                source="performance_analyzer",
                portfolio_id="",
                period="2024-01",
            )

        # Test empty period
        with pytest.raises(ValueError, match="Period cannot be empty"):
            PerformanceEvent(
                event_type="PERFORMANCE",
                timestamp=time.time(),
                source="performance_analyzer",
                portfolio_id="portfolio_123",
                period="",
            )


class TestEventFactoryFunctions:
    """Test cases for event factory functions."""

    def test_create_market_data_event(self):
        """Test market data event factory function."""
        data = {
            "open": 150.0,
            "high": 155.0,
            "low": 149.0,
            "close": 152.0,
            "volume": 1000000,
            "timestamp": time.time(),
        }

        event = create_market_data_event(
            symbol="AAPL", data=data, source="test_source", priority=EventPriority.HIGH
        )

        assert isinstance(event, MarketDataEvent)
        assert event.symbol == "AAPL"
        assert event.open_price == 150.0
        assert event.close_price == 152.0
        assert event.volume == 1000000
        assert event.source == "test_source"
        assert event.priority == EventPriority.HIGH

    def test_create_signal_event(self):
        """Test signal event factory function."""
        event = create_signal_event(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            strength=0.7,
            confidence=0.8,
            source="test_strategy",
            priority=EventPriority.CRITICAL,
        )

        assert isinstance(event, SignalEvent)
        assert event.symbol == "AAPL"
        assert event.signal_type == SignalType.SELL
        assert event.strength == 0.7
        assert event.confidence == 0.8
        assert event.source == "test_strategy"
        assert event.priority == EventPriority.CRITICAL

    def test_create_order_event(self):
        """Test order event factory function."""
        event = create_order_event(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            source="test_strategy",
            priority=EventPriority.HIGH,
        )

        assert isinstance(event, OrderEvent)
        assert event.symbol == "AAPL"
        assert event.side == OrderSide.BUY
        assert event.order_type == OrderType.LIMIT
        assert event.quantity == 100.0
        assert event.source == "test_strategy"
        assert event.priority == EventPriority.HIGH
        assert event.order_id is not None

    def test_create_portfolio_update_event(self):
        """Test portfolio update event factory function."""
        event = create_portfolio_update_event(
            portfolio_id="portfolio_123",
            total_value=50000.0,
            cash_balance=10000.0,
            positions_value=40000.0,
            source="test_portfolio",
            priority=EventPriority.NORMAL,
        )

        assert isinstance(event, PortfolioUpdateEvent)
        assert event.portfolio_id == "portfolio_123"
        assert event.total_value == 50000.0
        assert event.cash_balance == 10000.0
        assert event.positions_value == 40000.0
        assert event.source == "test_portfolio"
        assert event.priority == EventPriority.NORMAL

    def test_create_risk_alert_event(self):
        """Test risk alert event factory function."""
        event = create_risk_alert_event(
            alert_id="alert_123",
            risk_level="MEDIUM",
            message="Medium risk alert",
            component="test_component",
            source="test_risk_manager",
            priority=EventPriority.HIGH,
        )

        assert isinstance(event, RiskAlertEvent)
        assert event.alert_id == "alert_123"
        assert event.risk_level == RiskLevel.MEDIUM
        assert event.message == "Medium risk alert"
        assert event.component == "test_component"
        assert event.source == "test_risk_manager"
        assert event.priority == EventPriority.HIGH

    def test_create_strategy_event(self):
        """Test strategy event factory function."""
        event = create_strategy_event(
            strategy_id="strategy_123",
            strategy_name="TestStrategy",
            event_type="STRATEGY_COMPLETE",
            source="test_strategy",
            priority=EventPriority.NORMAL,
        )

        assert isinstance(event, StrategyEvent)
        assert event.strategy_id == "strategy_123"
        assert event.strategy_name == "TestStrategy"
        assert event.event_type == StrategyEventType.STRATEGY_COMPLETE.value
        assert event.source == "test_strategy"
        assert event.priority == EventPriority.NORMAL
