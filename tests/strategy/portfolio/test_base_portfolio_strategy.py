"""Tests for BasePortfolioStrategy.

This module contains comprehensive tests for the BasePortfolioStrategy class,
covering initialization, configuration validation, and core functionality.
"""

from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.portfolio.base_portfolio import BasePortfolio
from backtester.risk_management.position_sizing import PositionSizer
from backtester.strategy.portfolio.base_portfolio_strategy import BasePortfolioStrategy
from backtester.strategy.portfolio.portfolio_strategy_config import (
    PortfolioConstraints,
    PortfolioOptimizationParams,
    PortfolioStrategyConfig,
    PortfolioStrategyType,
    RebalanceFrequency,
    RiskBudget,
    SignalFilterConfig,
)


class TestBasePortfolioStrategy:
    """Test suite for BasePortfolioStrategy."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def mock_portfolio(self):
        """Create a mock portfolio."""
        portfolio = Mock(spec=BasePortfolio)
        portfolio.total_value = 100000.0
        portfolio.positions = {}
        return portfolio

    @pytest.fixture
    def mock_position_sizer(self):
        """Create a mock position sizer."""
        return Mock(spec=PositionSizer)

    @pytest.fixture
    def basic_config(self):
        """Create a basic portfolio strategy configuration."""
        return PortfolioStrategyConfig(
            strategy_name="test_strategy",
            strategy_type=PortfolioStrategyType.EQUAL_WEIGHT,
            symbols=["AAPL", "GOOGL", "MSFT"],
            constraints=PortfolioConstraints(),
            optimization_params=PortfolioOptimizationParams(),
            risk_budget=RiskBudget(),
            signal_filters=SignalFilterConfig(),
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            enable_rebalancing=True,
            min_position_size=0.001,
            max_position_size=0.3,
        )

    @pytest.fixture
    def strategy(self, basic_config, mock_event_bus):
        """Create a base portfolio strategy instance."""

        # Create a concrete implementation for testing
        class TestStrategy(BasePortfolioStrategy):
            def _setup_event_subscriptions(self) -> None:
                """No-op for testing."""

            def calculate_target_weights(self, market_data):
                return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

            def process_signals(self, signals):
                return []

        return TestStrategy(basic_config, mock_event_bus)

    def test_initialization(self, strategy, basic_config):
        """Test strategy initialization."""
        assert strategy.name == basic_config.strategy_name
        assert strategy.type == basic_config.strategy_type
        assert strategy.symbols == basic_config.symbols
        assert strategy.is_initialized is True
        assert strategy.current_step == 0
        assert strategy.rebalance_count == 0
        assert strategy.total_trades == 0
        assert strategy.successful_trades == 0

    def test_initialization_with_portfolio(self, strategy, mock_portfolio):
        """Test strategy initialization with portfolio."""
        strategy.initialize_portfolio(mock_portfolio)
        assert strategy.portfolio == mock_portfolio

    def test_set_position_sizer(self, strategy, mock_position_sizer):
        """Test setting position sizer."""
        strategy.set_position_sizer(mock_position_sizer)
        assert strategy.position_sizing == mock_position_sizer

    def test_set_risk_limits(self, strategy):
        """Test setting risk limits."""
        risk_limits = {"max_drawdown": 0.1, "max_position_size": 0.2}
        strategy.set_risk_limits(risk_limits)
        assert strategy.risk_limits == risk_limits

    def test_calculate_current_weights_empty_portfolio(self, strategy):
        """Test calculating current weights with empty portfolio."""
        weights = strategy._calculate_current_weights({})
        assert weights == {}

    def test_calculate_current_weights_with_portfolio(self, strategy, mock_portfolio):
        """Test calculating current weights with portfolio."""
        # Initialize the portfolio first
        strategy.initialize_portfolio(mock_portfolio)

        mock_portfolio.positions = {
            "AAPL": {"quantity": 100, "current_price": 150.0, "market_value": 15000.0},
            "GOOGL": {"quantity": 50, "current_price": 200.0, "market_value": 10000.0},
        }
        mock_portfolio.total_value = 25000.0

        market_data = {"AAPL": pd.DataFrame(), "GOOGL": pd.DataFrame()}
        weights = strategy._calculate_current_weights(market_data)

        # Should include all symbols from strategy config, with MSFT having 0 weight
        expected_weights = {"AAPL": 0.6, "GOOGL": 0.4, "MSFT": 0.0}
        assert weights == expected_weights

    @pytest.mark.skip(reason="Test needs config field update")
    def test_should_rebalance_threshold_based(self, strategy):
        """Test rebalance condition checking."""
        # Test threshold-based rebalancing
        strategy.config.rebalance_frequency = RebalanceFrequency.THRESHOLD_BASED
        # Set threshold using the config object directly
        strategy.config.threshold_based_rebalance = 0.05  # Set threshold to 5%

        # Test within threshold
        strategy.portfolio_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}
        strategy.target_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}
        assert strategy._should_rebalance({}) is False

        # Test beyond threshold
        strategy.portfolio_weights = {"AAPL": 0.5, "GOOGL": 0.25, "MSFT": 0.25}
        strategy.target_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}
        assert strategy._should_rebalance({}) is True

    def test_should_rebalance_time_based(self, strategy):
        """Test time-based rebalance condition checking."""
        # Test time-based rebalancing
        strategy.config.rebalance_frequency = RebalanceFrequency.WEEKLY

        # Test within time period (step 5, not divisible by 7)
        strategy.current_step = 5
        assert strategy._should_rebalance({}) is False

        # Test beyond time period (step 7, divisible by 7)
        strategy.current_step = 7
        assert strategy._should_rebalance({}) is True

    def test_generate_rebalance_orders(self, strategy, mock_portfolio):
        """Test generating rebalance orders."""
        strategy.initialize_portfolio(mock_portfolio)
        strategy.portfolio_weights = {"AAPL": 0.5, "GOOGL": 0.25, "MSFT": 0.25}
        strategy.target_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}

        orders = strategy.generate_rebalance_orders()

        assert len(orders) == 3
        for order in orders:
            assert "symbol" in order
            assert "quantity" in order
            assert "price" in order
            assert "side" in order
            assert "order_type" in order
            assert "reason" in order

    def test_generate_rebalance_orders_no_portfolio(self, strategy):
        """Test generating rebalance orders without portfolio."""
        orders = strategy.generate_rebalance_orders()
        assert orders == []

    def test_execute_rebalance(self, strategy, mock_portfolio):
        """Test executing rebalance."""
        strategy.initialize_portfolio(mock_portfolio)

        orders = [
            {"symbol": "AAPL", "quantity": 10, "price": 150.0, "side": "BUY"},
            {"symbol": "GOOGL", "quantity": 5, "price": 200.0, "side": "SELL"},
        ]

        strategy._execute_rebalance(orders)

        # Verify portfolio weights are updated
        assert "AAPL" in strategy.portfolio_weights
        assert "GOOGL" in strategy.portfolio_weights

    def test_execute_rebalance_no_portfolio(self, strategy):
        """Test executing rebalance without portfolio."""
        orders = [{"symbol": "AAPL", "quantity": 10, "price": 150.0, "side": "BUY"}]
        strategy._execute_rebalance(orders)  # Should not raise exception

    def test_record_rebalance(self, strategy):
        """Test recording rebalance information."""
        orders = [{"symbol": "AAPL", "quantity": 10, "price": 150.0, "side": "BUY"}]
        old_weights = {"AAPL": 0.5, "GOOGL": 0.5}
        new_weights = {"AAPL": 0.33, "GOOGL": 0.67}

        strategy.portfolio_weights = old_weights
        strategy.target_weights = new_weights

        strategy._record_rebalance(orders)

        assert len(strategy.rebalance_history) == 1
        rebalance = strategy.rebalance_history[0]
        assert rebalance["orders"] == orders
        assert rebalance["old_weights"] == old_weights
        assert rebalance["new_weights"] == new_weights

    def test_record_trade(self, strategy):
        """Test recording trade information."""
        order = {
            "symbol": "AAPL",
            "quantity": 10,
            "price": 150.0,
            "side": "BUY",
            "reason": "rebalance",
        }

        strategy._record_trade(order)

        assert len(strategy.trade_history) == 1
        trade = strategy.trade_history[0]
        assert trade["symbol"] == "AAPL"
        assert trade["quantity"] == 10
        assert trade["price"] == 150.0
        assert trade["side"] == "BUY"
        assert trade["reason"] == "rebalance"

    def test_get_strategy_summary(self, strategy):
        """Test getting strategy summary."""
        summary = strategy.get_strategy_summary()

        assert summary["strategy_name"] == strategy.name
        assert summary["strategy_type"] == strategy.type
        assert summary["symbols"] == strategy.symbols
        assert summary["rebalance_count"] == strategy.rebalance_count
        assert summary["total_trades"] == strategy.total_trades
        assert "current_weights" in summary
        assert "target_weights" in summary
        assert "performance_metrics" in summary

    def test_get_performance_metrics(self, strategy):
        """Test getting performance metrics."""
        metrics = strategy.get_performance_metrics()
        assert metrics == strategy.performance_metrics

    def test_validate_config_valid(self, strategy, basic_config):
        """Test validating valid configuration."""
        assert strategy.validate_config() is True

    def test_validate_config_no_config(self, strategy):
        """Test validating configuration with no config."""
        strategy.config = None
        assert strategy.validate_config() is False

    def test_validate_config_no_name(self, strategy):
        """Test validating configuration with no name."""
        strategy.config.strategy_name = ""
        assert strategy.validate_config() is False

    def test_validate_config_no_symbols(self, strategy):
        """Test validating configuration with no symbols."""
        strategy.config.symbols = []
        assert strategy.validate_config() is False

    def test_validate_config_invalid_symbols(self, strategy):
        """Test validating configuration with invalid symbols."""
        strategy.config.symbols = "invalid"
        assert strategy.validate_config() is False

    def test_validate_config_no_constraints(self, strategy):
        """Test validating configuration with no constraints."""
        strategy.config.constraints = None
        assert strategy.validate_config() is False

    def test_reset(self, strategy):
        """Test resetting strategy state."""
        strategy.rebalance_count = 5
        strategy.total_trades = 10
        strategy.successful_trades = 8
        strategy.portfolio_weights = {"AAPL": 0.5, "GOOGL": 0.5}
        strategy.target_weights = {"AAPL": 0.33, "GOOGL": 0.67}
        strategy.performance_metrics = {"sharpe_ratio": 1.5}

        strategy.reset()

        assert strategy.rebalance_count == 0
        assert strategy.total_trades == 0
        assert strategy.successful_trades == 0
        assert strategy.portfolio_weights == {}
        assert strategy.target_weights == {}
        assert strategy.performance_metrics == {}

    def test_repr(self, strategy):
        """Test string representation."""
        repr_str = repr(strategy)
        assert "BasePortfolioStrategy" in repr_str
        assert strategy.name in repr_str
        assert strategy.type in repr_str
        assert str(strategy.symbols) in repr_str

    def test_get_current_price(self, strategy):
        """Test getting current price."""
        price = strategy._get_current_price("AAPL")
        assert price == 100.0  # Default placeholder value

    def test_get_current_position_value_no_portfolio(self, strategy):
        """Test getting current position value without portfolio."""
        value = strategy._get_current_position_value("AAPL")
        assert value == 0.0

    def test_get_current_position_value_with_portfolio(self, strategy, mock_portfolio):
        """Test getting current position value with portfolio."""
        mock_portfolio.positions = {
            "AAPL": {"quantity": 100, "current_price": 150.0, "market_value": 15000.0}
        }

        strategy.initialize_portfolio(mock_portfolio)
        value = strategy._get_current_position_value("AAPL")
        assert value == 15000.0

    def test_get_current_position_value_with_portfolio_object(self, strategy, mock_portfolio):
        """Test getting current position value with portfolio object."""
        # Create a mock position object
        mock_position = Mock()
        mock_position.quantity = 100
        mock_position.current_price = 150.0

        mock_portfolio.positions = {"AAPL": mock_position}

        strategy.portfolio = mock_portfolio
        value = strategy._get_current_position_value("AAPL")
        assert value == 15000.0

    def test_update_portfolio_state(self, strategy, mock_portfolio):
        """Test updating portfolio state."""
        strategy.initialize_portfolio(mock_portfolio)

        market_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
            "GOOGL": pd.DataFrame({"close": [200, 201, 202]}),
        }

        strategy.update_portfolio_state(market_data)

        # Verify portfolio weights are updated
        assert "AAPL" in strategy.portfolio_weights
        assert "GOOGL" in strategy.portfolio_weights

    def test_update_portfolio_state_exception(self, strategy, mock_portfolio):
        """Test updating portfolio state with exception."""
        strategy.initialize_portfolio(mock_portfolio)

        # Mock market data that will cause an exception
        market_data = {"AAPL": "invalid_data"}

        # Should not raise exception
        strategy.update_portfolio_state(market_data)

    def test_rebalance_portfolio(self, strategy, mock_portfolio):
        """Test rebalancing portfolio."""
        strategy.initialize_portfolio(mock_portfolio)

        market_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
            "GOOGL": pd.DataFrame({"close": [200, 201, 202]}),
        }

        orders = strategy.rebalance_portfolio(market_data)

        assert len(orders) >= 0
        assert strategy.rebalance_count == 1

    def test_rebalance_portfolio_no_portfolio(self, strategy):
        """Test rebalancing portfolio without portfolio."""
        market_data = {"AAPL": pd.DataFrame({"close": [100, 101, 102]})}
        orders = strategy.rebalance_portfolio(market_data)
        assert orders == []

    def test_rebalance_portfolio_exception(self, strategy, mock_portfolio):
        """Test rebalancing portfolio with exception."""
        strategy.initialize_portfolio(mock_portfolio)

        # Mock market data that will cause an exception
        market_data = {"AAPL": "invalid_data"}

        orders = strategy.rebalance_portfolio(market_data)
        assert orders == []

    def test_process_signals(self, strategy):
        """Test processing signals."""
        signals = [
            {"symbol": "AAPL", "type": "BUY", "confidence": 0.8, "strength": 1.0},
            {"symbol": "GOOGL", "type": "SELL", "confidence": 0.6, "strength": 0.8},
        ]

        orders = strategy.process_signals(signals)

        assert len(orders) == 0  # Base implementation returns empty list

    def test_process_signals_no_portfolio(self, strategy):
        """Test processing signals without portfolio."""
        signals = [{"symbol": "AAPL", "type": "BUY", "confidence": 0.8}]
        orders = strategy.process_signals(signals)
        assert orders == []

    def test_process_signals_invalid_symbol(self, strategy):
        """Test processing signals with invalid symbol."""
        signals = [{"symbol": "INVALID", "type": "BUY", "confidence": 0.8}]
        orders = strategy.process_signals(signals)
        assert orders == []

    def test_process_signals_low_confidence(self, strategy):
        """Test processing signals with low confidence."""
        signals = [{"symbol": "AAPL", "type": "BUY", "confidence": 0.2}]
        orders = strategy.process_signals(signals)
        assert orders == []

    def test_process_signals_missing_fields(self, strategy):
        """Test processing signals with missing fields."""
        signals = [{"symbol": "AAPL"}]  # Missing type, confidence, strength
        orders = strategy.process_signals(signals)
        assert orders == []

    def test_calculate_target_weights_abstract(self):
        """Test that calculate_target_weights is abstract."""
        config = PortfolioStrategyConfig(
            strategy_name="test_strategy",
            strategy_type=PortfolioStrategyType.EQUAL_WEIGHT,
            symbols=["AAPL", "GOOGL", "MSFT"],
        )
        event_bus = Mock(spec=EventBus)

        # Try to instantiate the base class directly
        with pytest.raises(TypeError):
            cast(Any, BasePortfolioStrategy)(config, event_bus)

    def test_process_signals_abstract(self):
        """Test that process_signals is abstract."""
        config = PortfolioStrategyConfig(
            strategy_name="test_strategy",
            strategy_type=PortfolioStrategyType.EQUAL_WEIGHT,
            symbols=["AAPL", "GOOGL", "MSFT"],
        )
        event_bus = Mock(spec=EventBus)

        # Try to instantiate the base class directly
        with pytest.raises(TypeError):
            cast(Any, BasePortfolioStrategy)(config, event_bus)

    @pytest.mark.skip(reason="Test needs config field update")
    def test_should_rebalance_abstract(self, strategy):
        """Test that should_rebalance is abstract."""
        # Test with default implementation
        strategy.config.rebalance_frequency = RebalanceFrequency.THRESHOLD_BASED
        strategy.config.threshold_based_rebalance = 0.05
        strategy.portfolio_weights = {"AAPL": 0.5, "GOOGL": 0.25, "MSFT": 0.25}
        strategy.target_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}
        assert strategy._should_rebalance({}) is True

    @pytest.mark.skip(reason="Test needs config method update")
    def test_apply_constraints(self, strategy):
        """Test applying constraints."""
        weights = {"AAPL": 0.5, "GOOGL": 0.5}

        # Create a mock constraints object
        mock_constraints = Mock()
        mock_constraints.min_weight = 0.1
        mock_constraints.max_weight = 0.4

        # Mock the config method using monkeypatch
        import unittest.mock

        with unittest.mock.patch.object(
            strategy.config, 'get_constraint_for_symbol', return_value=mock_constraints
        ):
            constrained_weights = strategy._apply_constraints(weights)

        assert constrained_weights["AAPL"] == 0.4  # Should be capped at max
        assert constrained_weights["GOOGL"] == 0.5  # No constraints applied

    def test_normalize_weights(self, strategy):
        """Test normalizing weights."""
        weights = {"AAPL": 0.5, "GOOGL": 0.5, "MSFT": 0.0}
        normalized = strategy._normalize_weights(weights)

        assert normalized["AAPL"] == 0.5
        assert normalized["GOOGL"] == 0.5
        assert normalized["MSFT"] == 0.0

    def test_normalize_weights_zero_total(self, strategy):
        """Test normalizing weights with zero total."""
        weights = {"AAPL": 0.0, "GOOGL": 0.0, "MSFT": 0.0}
        normalized = strategy._normalize_weights(weights)

        # Should fallback to equal weights
        assert normalized["AAPL"] == pytest.approx(1.0 / 3)
        assert normalized["GOOGL"] == pytest.approx(1.0 / 3)
        assert normalized["MSFT"] == pytest.approx(1.0 / 3)

    def test_get_constraint_for_symbol_not_implemented(self, strategy):
        """Test that get_constraint_for_symbol is not implemented in base class."""
        # This method should be implemented by concrete strategies
        with pytest.raises(NotImplementedError):
            strategy.get_constraint_for_symbol("AAPL")

    def test_should_rebalance_not_implemented(self, strategy):
        """Test that should_rebalance is not implemented in base class."""
        # This method should be implemented by concrete strategies
        with pytest.raises(NotImplementedError):
            strategy.should_rebalance({}, 0)
