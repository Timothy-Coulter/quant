"""Comprehensive tests for the SimulatedBroker module.

This module contains comprehensive tests that cover all aspects of the
SimulatedBroker functionality including order execution, market data handling,
commission calculation, position management, and configuration usage.
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from backtester.core.config import SimulatedBrokerConfig
from backtester.execution.broker import SimulatedBroker
from backtester.execution.order import Order, OrderSide, OrderStatus, OrderType


class TestSimulatedBrokerConfig:
    """Test suite for SimulatedBrokerConfig."""

    def test_default_config_creation(self) -> None:
        """Test creating config with default values."""
        config = SimulatedBrokerConfig()
        assert config.commission_rate == 0.001
        assert config.min_commission == 1.0
        assert config.spread == 0.0001
        assert config.slippage_model == "normal"
        assert config.slippage_std == 0.0005
        assert config.latency_ms == 0.0
        assert config.latency_jitter_ms == 0.0
        assert config.max_orders_per_minute == 0
        assert config.order_cooldown_seconds == 0.0
        assert config.slippage_distribution == "normal"

    def test_custom_config_creation(self) -> None:
        """Test creating config with custom values."""
        config = SimulatedBrokerConfig(
            commission_rate=0.002,
            min_commission=2.0,
            spread=0.0002,
            slippage_model="fixed",
            slippage_distribution="lognormal",
            slippage_std=0.001,
            latency_ms=5.0,
            latency_jitter_ms=1.5,
            max_orders_per_minute=25,
            order_cooldown_seconds=0.5,
        )
        assert config.commission_rate == 0.002
        assert config.min_commission == 2.0
        assert config.spread == 0.0002
        assert config.slippage_model == "fixed"
        assert config.slippage_distribution == "lognormal"
        assert config.slippage_std == 0.001
        assert config.latency_ms == 5.0
        assert config.latency_jitter_ms == 1.5
        assert config.max_orders_per_minute == 25
        assert config.order_cooldown_seconds == 0.5

    def test_slippage_model_validation_valid(self) -> None:
        """Test slippage model validation with valid values."""
        for model in ["normal", "fixed", "none"]:
            config = SimulatedBrokerConfig(slippage_model=model)
            assert config.slippage_model == model

    def test_slippage_model_validation_invalid(self) -> None:
        """Test slippage model validation with invalid values."""
        with pytest.raises(ValueError, match="slippage_model must be one of"):
            SimulatedBrokerConfig(slippage_model="invalid")

    def test_config_pydantic_validation(self) -> None:
        """Test that config properly validates types."""
        config = SimulatedBrokerConfig(
            commission_rate=0.001,  # Should be float
            min_commission=1.0,
        )
        assert isinstance(config.commission_rate, float)

    def test_config_serialization(self) -> None:
        """Test config can be serialized to dict."""
        config = SimulatedBrokerConfig(commission_rate=0.002, min_commission=2.0)
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["commission_rate"] == 0.002
        assert config_dict["min_commission"] == 2.0

    def test_default_config_factory(self) -> None:
        """SimulatedBroker.default_config should return a config instance."""
        config = SimulatedBroker.default_config()
        assert isinstance(config, SimulatedBrokerConfig)

    def test_init_with_yaml_config(self) -> None:
        """Broker should load configuration from YAML files."""
        config_path = (
            Path(__file__).resolve().parents[2] / "component_configs" / "execution" / "retail.yaml"
        )
        broker = SimulatedBroker(config=config_path)
        assert pytest.approx(broker.commission_rate, rel=1e-6) == 0.001
        assert broker.max_orders_per_minute == 120

    def test_config_overrides_are_applied(self) -> None:
        """Config overrides must win over base config values."""
        broker = SimulatedBroker(
            config={'commission_rate': 0.002},
            config_overrides={'latency_ms': 12.5},
        )
        assert broker.commission_rate == 0.002
        assert broker.latency_ms == 12.5


class TestSimulatedBrokerInitialization:
    """Test suite for broker initialization."""

    def test_init_with_config(self) -> None:
        """Test broker initialization with config object."""
        config = SimulatedBrokerConfig(
            commission_rate=0.002,
            min_commission=2.0,
            spread=0.0002,
            slippage_model="fixed",
            slippage_std=0.001,
            latency_ms=5.0,
        )
        broker = SimulatedBroker(config=config)
        assert broker.commission_rate == 0.002
        assert broker.min_commission == 2.0
        assert broker.spread == 0.0002
        assert broker.slippage_model == "fixed"
        assert broker.slippage_std == 0.001
        assert broker.latency_ms == 5.0

    def test_init_with_individual_params(self) -> None:
        """Test broker initialization with individual parameters (backward compatibility)."""
        broker = SimulatedBroker(
            commission_rate=0.003,
            min_commission=3.0,
            spread=0.0003,
            slippage_model="none",
            slippage_std=0.0001,
            latency_ms=10.0,
        )
        assert broker.commission_rate == 0.003
        assert broker.min_commission == 3.0
        assert broker.spread == 0.0003
        assert broker.slippage_model == "none"
        assert broker.slippage_std == 0.0001
        assert broker.latency_ms == 10.0

    def test_init_with_defaults(self) -> None:
        """Test broker initialization with default values."""
        broker = SimulatedBroker()
        assert broker.commission_rate == 0.001
        assert broker.min_commission == 1.0
        assert broker.spread == 0.0001
        assert broker.slippage_model == "normal"
        assert broker.slippage_std == 0.0005
        assert broker.latency_ms == 0.0

    def test_init_with_logger(self) -> None:
        """Test broker initialization with custom logger."""
        logger = logging.getLogger("test_logger")
        broker = SimulatedBroker(logger=logger)
        assert broker.logger == logger

    def test_initial_state(self) -> None:
        """Test broker initial state."""
        broker = SimulatedBroker()
        assert broker.cash_balance == 0.0
        assert broker.portfolio_value == 0.0
        assert broker.positions == {}
        assert broker.trade_history == []
        assert broker.current_prices == {}
        assert broker.market_data == {}


class TestSimulatedBrokerMarketData:
    """Test suite for market data handling."""

    def test_set_market_data_valid(self) -> None:
        """Test setting valid market data."""
        broker = SimulatedBroker()
        data = pd.DataFrame(
            {
                'Open': [100.0, 101.0, 102.0],
                'High': [102.0, 103.0, 104.0],
                'Low': [99.0, 100.0, 101.0],
                'Close': [101.0, 102.0, 103.0],
                'Volume': [1000, 1100, 1200],
            }
        )
        broker.set_market_data("TEST", data)
        assert "TEST" in broker.market_data
        assert "TEST" in broker.current_prices
        assert broker.current_prices["TEST"] == 103.0

    def test_set_market_data_no_close_column(self) -> None:
        """Test setting market data without Close column."""
        broker = SimulatedBroker()
        data = pd.DataFrame(
            {
                'Open': [100.0, 101.0],
                'High': [102.0, 103.0],
                'Low': [99.0, 100.0],
                'Volume': [1000, 1100],
            }
        )
        broker.set_market_data("TEST", data)
        assert "TEST" in broker.market_data
        # When there's no Close column, current_prices should not have the key
        assert "TEST" not in broker.current_prices

    def test_set_market_data_updates_existing(self) -> None:
        """Test updating existing market data."""
        broker = SimulatedBroker()
        data1 = pd.DataFrame({'Close': [100.0]})
        data2 = pd.DataFrame({'Close': [105.0]})

        broker.set_market_data("TEST", data1)
        assert broker.current_prices["TEST"] == 100.0

        broker.set_market_data("TEST", data2)
        assert broker.current_prices["TEST"] == 105.0

    def test_get_current_price_existing(self) -> None:
        """Test getting price for existing symbol."""
        broker = SimulatedBroker()
        broker.current_prices["TEST"] = 100.5
        assert broker.get_current_price("TEST") == 100.5

    def test_get_current_price_nonexistent(self) -> None:
        """Test getting price for non-existent symbol."""
        broker = SimulatedBroker()
        assert broker.get_current_price("NONEXISTENT") == 0.0

    def test_get_bid_ask(self) -> None:
        """Test getting bid and ask prices."""
        broker = SimulatedBroker(spread=0.002)  # 0.2% spread
        broker.current_prices["TEST"] = 100.0
        bid, ask = broker.get_bid_ask("TEST")
        # Bid should be 100.0 * (1 - 0.001) = 99.9
        # Ask should be 100.0 * (1 + 0.001) = 100.1
        assert bid == 99.9
        assert ask == 100.1


class TestOrderExecution:
    """Test suite for order execution functionality."""

    @pytest.fixture
    def broker(self) -> SimulatedBroker:
        """Create a broker with basic configuration."""
        return SimulatedBroker()

    @pytest.fixture
    def market_data(self) -> pd.DataFrame:
        """Create sample market data."""
        return pd.DataFrame(
            {
                'Open': [100.0, 101.0, 102.0],
                'High': [102.0, 103.0, 104.0],
                'Low': [99.0, 100.0, 101.0],
                'Close': [101.0, 102.0, 103.0],
                'Volume': [1000, 1100, 1200],
            }
        )

    def test_execute_market_order_buy_success(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test successful market buy order execution."""
        broker.set_market_data("TEST", market_data)

        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        # Set initial cash
        broker.cash_balance = 20000.0

        result = broker.execute_order(order)
        assert result is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100.0
        assert order.filled_price is not None
        assert order.commission > 0
        assert len(broker.trade_history) == 1

    def test_execute_market_order_sell_success(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test successful market sell order execution."""
        broker.set_market_data("TEST", market_data)

        # Set initial position and cash
        broker.positions["TEST"] = 100.0
        broker.cash_balance = 10000.0  # Need some cash for commission

        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50.0,
        )

        result = broker.execute_order(order)
        assert result is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 50.0
        assert order.commission > 0

    def test_execute_inactive_order(self, broker: SimulatedBroker) -> None:
        """Test execution of inactive order."""
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        order.status = OrderStatus.CANCELLED

        result = broker.execute_order(order)
        assert result is False

    def test_execute_order_no_market_data(self, broker: SimulatedBroker) -> None:
        """Test execution without market data."""
        order = Order(
            symbol="NONEXISTENT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        result = broker.execute_order(order)
        assert result is False
        assert order.status == OrderStatus.REJECTED

    def test_order_rate_limit_rejects_excess_orders(self, market_data: pd.DataFrame) -> None:
        """Rate limiting should reject the second order within the same minute."""
        broker = SimulatedBroker(config_overrides={'max_orders_per_minute': 1})
        broker.set_market_data("TEST", market_data)
        broker.cash_balance = 20000.0

        first = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        second = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0)

        assert broker.execute_order(first) is True
        assert broker.execute_order(second) is False
        assert second.status == OrderStatus.REJECTED

    def test_order_cooldown_blocks_back_to_back_orders(self, market_data: pd.DataFrame) -> None:
        """Cooldown enforcement should reject orders submitted too quickly."""
        broker = SimulatedBroker(config_overrides={'order_cooldown_seconds': 5.0})
        broker.set_market_data("TEST", market_data)
        broker.cash_balance = 20000.0

        first = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        second = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=5.0)

        assert broker.execute_order(first) is True
        assert broker.execute_order(second) is False
        assert second.status == OrderStatus.REJECTED

    def test_execute_limit_order_favorable(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test successful limit order execution with favorable price."""
        broker.set_market_data("TEST", market_data)
        broker.cash_balance = 20000.0

        # Current price is 103.0
        # Buy limit order at 104.0 should execute (market price 103.0 <= limit price 104.0)
        order1 = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=104.0,
        )
        result = broker.execute_order(order1)
        assert result is True

        # Buy limit order at 100.0 should NOT execute (market price 103.0 > limit price 100.0)
        order2 = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=100.0,
        )
        result = broker.execute_order(order2)
        assert result is False

        # Buy limit order at 103.0 should execute (market price 103.0 <= limit price 103.0)
        order3 = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=103.0,
        )
        result = broker.execute_order(order3)
        assert result is True

    def test_execute_limit_order_unfavorable(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test limit order that doesn't get filled due to unfavorable price."""
        broker.set_market_data("TEST", market_data)
        broker.cash_balance = 20000.0

        # Place buy limit order at 100.0 when market is 102.0
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=100.0,
        )

        result = broker.execute_order(order)
        assert result is False

    def test_execute_sell_limit_order(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test sell limit order execution."""
        broker.set_market_data("TEST", market_data)
        broker.positions["TEST"] = 100.0
        broker.cash_balance = 10000.0  # Need cash for commission

        # Current price is 103.0
        # Sell limit order at 102.0 should execute (market price 103.0 >= limit price 102.0)
        order1 = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=102.0,
        )
        result = broker.execute_order(order1)
        assert result is True

        # Sell limit order at 105.0 should NOT execute (market price 103.0 < limit price 105.0)
        order2 = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=105.0,
        )
        result = broker.execute_order(order2)
        assert result is False

    def test_commission_calculation(self, broker: SimulatedBroker) -> None:
        """Test commission calculation."""
        # Test minimum commission applies
        broker.commission_rate = 0.001
        broker.min_commission = 1.0

        # Small trade that should hit minimum
        commission = broker._calculate_commission(10.0, 10.0)  # $100 notional
        assert commission == 1.0  # Minimum commission

        # Large trade that should use percentage
        commission = broker._calculate_commission(1000.0, 10.0)  # $10,000 notional
        assert commission == 10.0  # 0.1% of $10,000

    def test_position_and_cash_update_buy(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test position and cash updates for buy order."""
        broker.set_market_data("TEST", market_data)
        broker.cash_balance = 20000.0

        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        broker.execute_order(order)

        assert broker.positions["TEST"] == 100.0
        assert broker.cash_balance < 20000.0  # Cash should decrease
        assert broker.portfolio_value > 0.0  # Should have position value

    def test_position_and_cash_update_sell(
        self, broker: SimulatedBroker, market_data: pd.DataFrame
    ) -> None:
        """Test position and cash updates for sell order."""
        broker.set_market_data("TEST", market_data)
        broker.positions["TEST"] = 100.0
        broker.current_prices["TEST"] = 102.0
        broker.cash_balance = 10000.0

        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50.0,
        )

        broker.execute_order(order)

        assert broker.positions["TEST"] == 50.0
        assert broker.cash_balance > 10000.0  # Cash should increase

    def test_submit_order_with_risk_manager(self, market_data: pd.DataFrame) -> None:
        """submit_order should call risk manager hooks when provided."""
        risk_manager = Mock()
        risk_manager.can_open_position.return_value = True
        broker = SimulatedBroker(risk_manager=risk_manager, initial_cash=10000.0)
        broker.set_market_data("TEST", market_data)

        order = broker.submit_order(symbol="TEST", side="BUY", quantity=10.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        risk_manager.record_order.assert_called_once()
        risk_manager.record_fill.assert_called_once()


class TestSlippageModels:
    """Test suite for different slippage models."""

    def test_slippage_none(self) -> None:
        """Test no slippage model."""
        broker = SimulatedBroker(slippage_model="none", slippage_std=0.001)
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        slippage = broker._calculate_slippage(order, 100.0)
        assert slippage == 0.0

    def test_slippage_fixed(self) -> None:
        """Test fixed slippage model."""
        broker = SimulatedBroker(slippage_model="fixed", slippage_std=0.001)
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        slippage = broker._calculate_slippage(order, 100.0)
        assert slippage == 0.1  # 100.0 * 0.001

    def test_slippage_normal(self) -> None:
        """Test normal (random) slippage model."""
        broker = SimulatedBroker(slippage_model="normal", slippage_std=0.001)
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        # With random slippage, result should be around 0 but can vary
        # Test with mocked random to ensure consistent results
        with patch('numpy.random.normal') as mock_normal:
            mock_normal.return_value = 0.0005
            slippage = broker._calculate_slippage(order, 100.0)
            assert slippage == 0.05  # 100.0 * 0.0005

    def test_slippage_invalid_model(self) -> None:
        """Test invalid slippage model defaults to no slippage."""
        broker = SimulatedBroker(slippage_model="invalid_model", slippage_std=0.001)
        order = Order(
            symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        slippage = broker._calculate_slippage(order, 100.0)
        assert slippage == 0.0  # Should default to no slippage

    def test_slippage_lognormal_distribution(self) -> None:
        """Lognormal slippage distribution should rely on numpy's lognormal sampler."""
        broker = SimulatedBroker(
            config_overrides={'slippage_distribution': 'lognormal'}, slippage_std=0.1
        )
        order = Order(symbol="TEST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)

        with patch('numpy.random.lognormal') as mock_lognormal:
            mock_lognormal.return_value = 1.01
            slippage = broker._calculate_slippage(order, 100.0)
            assert slippage == pytest.approx(1.0, rel=1e-9)


class TestAccountOperations:
    """Test suite for account and reporting operations."""

    def test_account_summary_empty(self) -> None:
        """Test account summary with no positions or trades."""
        broker = SimulatedBroker()
        summary = broker.get_account_summary()

        assert summary['cash_balance'] == 0.0
        assert summary['portfolio_value'] == 0.0
        assert summary['positions'] == {}
        assert summary['unrealized_pnl'] == 0.0
        assert summary['total_commission'] == 0.0
        assert summary['total_trades'] == 0
        assert 'order_summary' in summary

    def test_account_summary_with_positions(self) -> None:
        """Test account summary with positions."""
        broker = SimulatedBroker()
        broker.positions["TEST"] = 100.0
        broker.current_prices["TEST"] = 102.0
        broker.cash_balance = 5000.0

        # Update portfolio value manually to reflect the position
        broker.portfolio_value = broker.cash_balance
        for symbol, position in broker.positions.items():
            if position != 0:
                current_price = broker.get_current_price(symbol)
                broker.portfolio_value += position * current_price

        summary = broker.get_account_summary()

        assert summary['cash_balance'] == 5000.0
        assert summary['positions']["TEST"] == 100.0
        assert summary['portfolio_value'] > 0.0

    def test_trade_history_all(self) -> None:
        """Test retrieving all trade history."""
        broker = SimulatedBroker()

        # Add some mock trades
        broker.trade_history = [
            {'symbol': 'TEST1', 'side': 'BUY', 'quantity': 100},
            {'symbol': 'TEST2', 'side': 'SELL', 'quantity': 50},
        ]

        history = broker.get_trade_history()
        assert len(history) == 2
        assert history[0]['symbol'] == 'TEST1'
        assert history[1]['symbol'] == 'TEST2'

    def test_trade_history_filtered(self) -> None:
        """Test retrieving filtered trade history."""
        broker = SimulatedBroker()

        # Add some mock trades
        broker.trade_history = [
            {'symbol': 'TEST1', 'side': 'BUY', 'quantity': 100},
            {'symbol': 'TEST2', 'side': 'SELL', 'quantity': 50},
            {'symbol': 'TEST1', 'side': 'BUY', 'quantity': 75},
        ]

        history = broker.get_trade_history('TEST1')
        assert len(history) == 2
        assert all(trade['symbol'] == 'TEST1' for trade in history)

    def test_reset_broker(self) -> None:
        """Test broker reset functionality."""
        broker = SimulatedBroker()

        # Set up some state
        broker.positions["TEST"] = 100.0
        broker.cash_balance = 5000.0
        broker.trade_history.append({'test': 'trade'})
        broker.current_prices["TEST"] = 102.0

        broker.reset()

        assert broker.positions == {}
        assert broker.cash_balance == 0.0
        assert broker.trade_history == []
        assert broker.current_prices == {}
        assert broker.portfolio_value == 0.0


class TestMarketDataUpdate:
    """Test suite for market data update processing."""

    def test_process_market_data_update(self) -> None:
        """Test processing market data update."""
        broker = SimulatedBroker()

        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        broker.process_market_data_update(
            symbol="TEST",
            timestamp=timestamp,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000,
        )

        assert broker.current_prices["TEST"] == 101.0

    def test_process_market_data_update_with_pending_orders(self) -> None:
        """Test processing market data update with pending orders."""
        broker = SimulatedBroker()
        broker.cash_balance = 20000.0

        # Create and add a limit order that should be fillable
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=101.0,  # Should fill at current price
        )
        # order_id should never be None after order creation
        assert order.order_id is not None
        broker.order_manager.orders[order.order_id] = order

        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        broker.process_market_data_update(
            symbol="TEST",
            timestamp=timestamp,
            open_price=100.0,
            high_price=103.0,
            low_price=99.0,
            close_price=101.0,
            volume=1000,
        )

        assert order.status == OrderStatus.FILLED

    def test_expire_stale_orders(self) -> None:
        """Test expiring stale orders."""
        broker = SimulatedBroker()
        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        # This should not raise any errors
        broker._expire_stale_orders("TEST", timestamp)


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_execute_order_zero_quantity(self) -> None:
        """Test execution of zero quantity order."""
        broker = SimulatedBroker()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.0,
        )

        result = broker.execute_order(order)
        assert result is False

    def test_execute_order_negative_quantity(self) -> None:
        """Test execution of negative quantity order."""
        broker = SimulatedBroker()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=-100.0,
        )

        result = broker.execute_order(order)
        assert result is False

    def test_max_quantity_calculation_no_cash(self) -> None:
        """Test max quantity calculation with no cash."""
        broker = SimulatedBroker()
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        max_qty = broker._calculate_max_quantity(order, 100.0)
        assert max_qty == 0.0  # No cash available

    def test_max_quantity_calculation_with_cash(self) -> None:
        """Test max quantity calculation with available cash."""
        broker = SimulatedBroker()
        broker.cash_balance = 10000.0
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        max_qty = broker._calculate_max_quantity(order, 100.0)
        assert max_qty == 100.0  # Can buy 100 shares

    def test_determine_execution_price_market_order(self) -> None:
        """Test execution price determination for market orders."""
        # Use "none" slippage model for predictable results
        broker = SimulatedBroker(slippage_model="none")
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        price = broker._determine_execution_price(order, 100.0, 99.5, 100.5)
        assert price is not None
        assert price >= 100.5  # Should be at least the ask price of 100.5


if __name__ == "__main__":
    pytest.main([__file__])
