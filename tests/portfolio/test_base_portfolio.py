"""Test cases for BasePortfolio abstract class.

This module contains unit tests for the BasePortfolio abstract class,
testing the common functionality and interface.
"""

import logging
from datetime import datetime
from typing import Any

import pytest

from backtester.portfolio.base_portfolio import BasePortfolio


class ConcretePortfolio(BasePortfolio):
    """Concrete implementation of BasePortfolio for testing."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize concrete portfolio for testing.

        Args:
            **kwargs: Keyword arguments to pass to parent class
        """
        super().__init__(**kwargs)
        self.cash = 1000.0  # Add cash attribute for testing

    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Any,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        leverage: float = 1.0,
        pool_type: str = 'base',
        **kwargs: Any,
    ) -> bool:
        """Add a position to the portfolio.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Position price
            timestamp: Position timestamp
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            leverage: Leverage factor (1.0 = no leverage)
            pool_type: Pool type for dual pool portfolios
            **kwargs: Additional keyword arguments

        Returns:
            True indicating successful position addition
        """
        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: Any,
        quantity: float | None = None,
    ) -> bool:
        """Close a position in the portfolio.

        Args:
            symbol: Symbol to close
            price: Close price
            timestamp: Close timestamp
            quantity: Quantity to close (defaults to full position)

        Returns:
            True indicating successful position closure
        """
        return True

    def process_tick(
        self,
        timestamp: Any,
        market_data: dict[str, Any] | None = None,
        current_price: float | None = None,
        day_high: float | None = None,
        day_low: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a market tick.

        Args:
            timestamp: Current timestamp
            market_data: Optional market data dictionary
            current_price: Optional current price
            day_high: Optional day high price
            day_low: Optional day low price
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with portfolio information
        """
        return {'total_value': self.total_value}


class TestBasePortfolio:
    """Test cases for BasePortfolio class."""

    def test_init(self) -> None:
        """Test portfolio initialization."""
        portfolio = ConcretePortfolio(
            initial_capital=1000.0,
            commission_rate=0.001,
            interest_rate_daily=0.00025,
            spread_rate=0.0002,
            slippage_std=0.0005,
            funding_enabled=True,
            tax_rate=0.25,
        )

        assert portfolio.initial_capital == 1000.0
        assert portfolio.commission_rate == 0.001
        assert portfolio.interest_rate_daily == 0.00025
        assert portfolio.spread_rate == 0.0002
        assert portfolio.slippage_std == 0.0005
        assert portfolio.funding_enabled is True
        assert portfolio.tax_rate == 0.25
        assert portfolio.cash == 1000.0
        assert portfolio.portfolio_values == [1000.0]
        assert portfolio.trade_log == []
        assert portfolio.cumulative_tax == 0.0

    def test_init_with_logger(self) -> None:
        """Test portfolio initialization with custom logger."""
        logger = logging.getLogger('test')
        portfolio = ConcretePortfolio(logger=logger)

        assert portfolio.logger == logger

    def test_total_value_property(self) -> None:
        """Test total_value property."""
        portfolio = ConcretePortfolio()
        portfolio.portfolio_values = [1000.0, 1100.0, 1050.0]

        assert portfolio.total_value == 1050.0

    def test_get_performance_metrics(self) -> None:
        """Test performance metrics calculation."""
        portfolio = ConcretePortfolio(initial_capital=1000.0)
        portfolio.portfolio_values = [1000.0, 1100.0, 1200.0, 1100.0]
        portfolio.trade_log = [
            {'action': 'CLOSE', 'realized_pnl': 50.0},
            {'action': 'CLOSE', 'realized_pnl': -20.0},
        ]

        metrics = portfolio.get_performance_metrics()

        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'cumulative_tax' in metrics
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'win_rate' in metrics
        assert 'final_portfolio_value' in metrics
        assert 'portfolio_values' in metrics
        assert 'trade_log' in metrics

        # Test specific values
        assert abs(metrics['total_return'] - 10.0) < 0.01  # (1100 - 1000) / 1000 * 100
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['win_rate'] == 50.0

    def test_reset(self) -> None:
        """Test portfolio reset functionality."""
        portfolio = ConcretePortfolio()
        portfolio.portfolio_values = [1000.0, 1100.0]
        portfolio.trade_log = [{'action': 'TEST'}]
        portfolio.cumulative_tax = 50.0
        portfolio.current_year = 2023
        portfolio.yearly_gains = {'test': 100.0}

        portfolio.reset()

        assert portfolio.portfolio_values == [portfolio.initial_capital]
        assert portfolio.trade_log == []
        assert portfolio.cumulative_tax == 0.0
        assert portfolio.current_year is None
        assert portfolio.yearly_gains == {}  # type: ignore[unreachable]

    def test_handle_tax_calculation_new_year(self) -> None:
        """Test tax calculation when new year is detected."""
        portfolio = ConcretePortfolio()
        portfolio.current_year = 2022
        portfolio.yearly_gains = {'2022': 100.0}
        portfolio.cash = 1000.0

        # Mock datetime for testing
        mock_timestamp = datetime(2023, 1, 1)
        portfolio._handle_tax_calculation(mock_timestamp)

        assert portfolio.cumulative_tax == 45.0  # 100.0 * 0.45
        assert portfolio.cash == 955.0  # 1000.0 - 45.0
        assert portfolio.current_year == 2023
        assert portfolio.yearly_gains == {}

    def test_handle_tax_calculation_same_year(self) -> None:
        """Test tax calculation within same year."""
        portfolio = ConcretePortfolio()
        portfolio.current_year = 2023
        portfolio.yearly_gains = {'2023': 100.0}
        initial_cash = portfolio.cash

        mock_timestamp = datetime(2023, 6, 15)
        portfolio._handle_tax_calculation(mock_timestamp)

        # No tax should be applied for same year
        assert portfolio.cumulative_tax == 0.0
        assert portfolio.cash == initial_cash
        assert portfolio.current_year == 2023
        assert portfolio.yearly_gains == {'2023': 100.0}

    def test_handle_tax_calculation_initialization(self) -> None:
        """Test tax calculation initialization."""
        portfolio = ConcretePortfolio()
        assert portfolio.current_year is None

        mock_timestamp = datetime(2023, 6, 15)
        portfolio._handle_tax_calculation(mock_timestamp)

        assert portfolio.current_year == 2023
        assert portfolio.yearly_gains == {}

    def test_calculate_costs(self) -> None:
        """Test cost calculation for trades."""
        portfolio = ConcretePortfolio()
        portfolio.commission_rate = 0.001
        portfolio.spread_rate = 0.0002
        portfolio.slippage_std = 0.0005

        # Test with fixed random seed for consistent slippage
        import random

        random.seed(42)

        commission, spread_cost, slippage = portfolio._calculate_costs(1000.0, 100, 10.0)

        assert commission == 1.0  # 1000.0 * 0.001
        assert spread_cost == 0.2  # 1000.0 * 0.0002
        assert isinstance(slippage, float)

    def test_log_trade(self) -> None:
        """Test trade logging functionality."""
        portfolio = ConcretePortfolio()
        timestamp = datetime(2023, 1, 1)

        portfolio._log_trade(
            timestamp=timestamp,
            action='BUY',
            symbol='AAPL',
            quantity=100,
            price=150.0,
            extra_field='test_value',
        )

        assert len(portfolio.trade_log) == 1
        trade_record = portfolio.trade_log[0]

        assert trade_record['timestamp'] == timestamp
        assert trade_record['action'] == 'BUY'
        assert trade_record['symbol'] == 'AAPL'
        assert trade_record['quantity'] == 100
        assert trade_record['price'] == 150.0
        assert trade_record['extra_field'] == 'test_value'

    def test_get_summary(self) -> None:
        """Test portfolio summary generation."""
        portfolio = ConcretePortfolio(initial_capital=1000.0)
        portfolio.portfolio_values = [1000.0, 1100.0]

        summary = portfolio.get_summary()

        assert 'total_value' in summary
        assert 'total_return' in summary
        assert 'total_trades' in summary
        assert 'cumulative_tax' in summary

        assert summary['total_value'] == 1100.0
        assert abs(summary['total_return'] - 10.0) < 0.01  # (1100 - 1000) / 1000 * 100
        assert summary['total_trades'] == 0
        assert summary['cumulative_tax'] == 0.0

    def test_abstract_methods(self) -> None:
        """Test that abstract methods must be implemented."""
        # This should work fine since BasePortfolio is abstract
        # but we can't instantiate it directly without a concrete implementation

        # Test that we can create a concrete implementation
        portfolio = ConcretePortfolio()

        # Test that abstract methods are available
        assert hasattr(portfolio, 'add_position')
        assert hasattr(portfolio, 'close_position')
        assert hasattr(portfolio, 'process_tick')

        # Test that they work as expected
        assert portfolio.add_position('AAPL', 100, 150.0, datetime.now()) is True
        assert portfolio.close_position('AAPL', 160.0, datetime.now()) is True
        result = portfolio.process_tick(datetime.now())
        assert 'total_value' in result

    def test_abstract_class_enforcement(self) -> None:
        """Test that BasePortfolio cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePortfolio()  # type: ignore[abstract]
