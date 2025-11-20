"""Test cases for DualPoolPortfolio class.

This module contains unit tests for the DualPoolPortfolio class used for
dual-pool leverage strategy management.
"""

from datetime import datetime

import pytest

from backtester.portfolio.dual_pool_portfolio import DualPoolPortfolio


class TestDualPoolPortfolio:
    """Test cases for DualPoolPortfolio class."""

    def test_init_default(self) -> None:
        """Test portfolio initialization with default values."""
        portfolio = DualPoolPortfolio()

        assert portfolio.initial_capital == 100.0
        assert portfolio.leverage_base == 1.0
        assert portfolio.leverage_alpha == 3.0
        assert portfolio.base_to_alpha_split == 0.2
        assert portfolio.alpha_to_base_split == 0.2
        assert portfolio.stop_loss_base == 0.025
        assert portfolio.stop_loss_alpha == 0.025
        assert portfolio.take_profit_target == 0.10
        assert portfolio.max_total_leverage == 4.0

    def test_init_custom(self) -> None:
        """Test portfolio initialization with custom values."""
        portfolio = DualPoolPortfolio(
            initial_capital=10000.0,
            leverage_base=1.5,
            leverage_alpha=4.0,
            base_to_alpha_split=0.3,
            alpha_to_base_split=0.15,
            stop_loss_base=0.02,
            stop_loss_alpha=0.03,
            take_profit_target=0.15,
        )

        assert portfolio.initial_capital == 10000.0
        assert portfolio.leverage_base == 1.5
        assert portfolio.leverage_alpha == 4.0
        assert portfolio.base_to_alpha_split == 0.3
        assert portfolio.alpha_to_base_split == 0.15
        assert portfolio.stop_loss_base == 0.02
        assert portfolio.stop_loss_alpha == 0.03
        assert portfolio.take_profit_target == 0.15

    def test_pool_initialization(self) -> None:
        """Test that pools are properly initialized."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Check base pool
        assert portfolio.base_pool.pool_type == 'base'
        assert portfolio.base_pool.leverage == 1.0
        assert portfolio.base_pool.capital == 500.0  # Half of initial capital
        assert portfolio.base_pool.active is False

        # Check alpha pool
        assert portfolio.alpha_pool.pool_type == 'alpha'
        assert portfolio.alpha_pool.leverage == 3.0
        assert portfolio.alpha_pool.capital == 500.0  # Half of initial capital
        assert portfolio.alpha_pool.active is False

    def test_add_position_to_base_pool(self) -> None:
        """Test adding position to base pool."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        result = portfolio.add_position(
            symbol='AAPL', quantity=10, price=50.0, timestamp=datetime(2023, 1, 1), pool_type='base'
        )

        assert result is True
        assert portfolio.base_pool.positions is not None
        assert 'AAPL' in portfolio.base_pool.positions

    def test_add_position_to_alpha_pool(self) -> None:
        """Test adding position to alpha pool."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        result = portfolio.add_position(
            symbol='MSFT',
            quantity=5,
            price=100.0,
            timestamp=datetime(2023, 1, 1),
            pool_type='alpha',
        )

        assert result is True
        assert portfolio.alpha_pool.positions is not None
        assert 'MSFT' in portfolio.alpha_pool.positions

    def test_add_position_invalid_pool(self) -> None:
        """Test adding position to invalid pool type."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        with pytest.raises(ValueError):
            portfolio.add_position(
                symbol='AAPL',
                quantity=10,
                price=50.0,
                timestamp=datetime(2023, 1, 1),
                pool_type='invalid',
            )

    def test_close_position_from_base_pool(self) -> None:
        """Test closing position from base pool."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 50.0, datetime(2023, 1, 1), pool_type='base')

        result = portfolio.close_position(
            symbol='AAPL', price=60.0, timestamp=datetime(2023, 1, 2), pool_type='base'
        )

        assert result is True
        assert portfolio.base_pool.positions is not None
        assert 'AAPL' not in portfolio.base_pool.positions

    def test_close_position_from_alpha_pool(self) -> None:
        """Test closing position from alpha pool."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)
        portfolio.add_position('MSFT', 5, 100.0, datetime(2023, 1, 1), pool_type='alpha')

        result = portfolio.close_position(
            symbol='MSFT', price=120.0, timestamp=datetime(2023, 1, 2), pool_type='alpha'
        )

        assert result is True
        assert portfolio.alpha_pool.positions is not None
        assert 'MSFT' not in portfolio.alpha_pool.positions

    def test_process_tick(self) -> None:
        """Test market tick processing."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        result = portfolio.process_tick(
            timestamp=datetime(2023, 1, 1), current_price=100.0, day_high=105.0, day_low=95.0
        )

        assert 'timestamp' in result
        assert 'total_value' in result
        assert 'base_pool' in result
        assert 'alpha_pool' in result
        assert 'base_active' in result
        assert 'alpha_active' in result

        # Both pools should be activated
        assert result['base_active'] is True
        assert result['alpha_active'] is True

    def test_get_total_leverage(self) -> None:
        """Test total leverage calculation."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        # Default leverage should be 0.0 when no positions are active
        leverage = portfolio.get_total_leverage()
        assert leverage == 0.0

    def test_check_risk_limits(self) -> None:
        """Test risk limits checking."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0, max_total_leverage=4.0)

        # Should be within limits initially
        assert portfolio.check_risk_limits() is True

    def test_get_pool_performance(self) -> None:
        """Test pool performance metrics."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        base_perf = portfolio.get_pool_performance('base')
        alpha_perf = portfolio.get_pool_performance('alpha')

        assert 'total_return' in base_perf
        assert 'sharpe_ratio' in base_perf
        assert 'total_return' in alpha_perf
        assert 'sharpe_ratio' in alpha_perf

    def test_get_pool_value(self) -> None:
        """Test pool value retrieval."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        base_value = portfolio.get_pool_value('base')
        alpha_value = portfolio.get_pool_value('alpha')

        assert base_value == 500.0  # Half of initial capital
        assert alpha_value == 500.0  # Half of initial capital

    def test_get_pool_value_invalid_pool(self) -> None:
        """Test getting pool value for invalid pool type."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        with pytest.raises(ValueError):
            portfolio.get_pool_value('invalid')

    def test_total_value_property(self) -> None:
        """Test total_value property."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0, cash=100.0)

        total_value = portfolio.total_value
        expected_value = portfolio.base_pool.capital + portfolio.alpha_pool.capital + portfolio.cash
        assert total_value == expected_value

    def test_reset(self) -> None:
        """Test portfolio reset functionality."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 50.0, datetime(2023, 1, 1), pool_type='base')
        portfolio.trade_log.append({'test': 'data'})
        portfolio.cumulative_tax = 25.0

        portfolio.reset()

        # Check base pool reset
        assert portfolio.base_pool.capital == 500.0
        assert portfolio.base_pool.active is False
        assert portfolio.base_pool.positions == {}

        # Check alpha pool reset
        assert portfolio.alpha_pool.capital == 500.0
        assert portfolio.alpha_pool.active is False
        assert portfolio.alpha_pool.positions == {}

        # Check portfolio tracking reset
        assert portfolio.trade_log == []
        assert portfolio.cumulative_tax == 0.0
        assert portfolio.portfolio_values == [1000.0]

    def test_get_summary(self) -> None:
        """Test portfolio summary generation."""
        portfolio = DualPoolPortfolio(initial_capital=1000.0)

        summary = portfolio.get_summary()

        assert 'total_value' in summary
        assert 'base_pool' in summary
        assert 'alpha_pool' in summary
        assert 'total_leverage' in summary
        assert 'cash' in summary

        assert 'capital' in summary['base_pool']
        assert 'leverage' in summary['base_pool']
        assert 'active' in summary['base_pool']
        assert 'available_capital' in summary['base_pool']

        assert 'capital' in summary['alpha_pool']
        assert 'leverage' in summary['alpha_pool']
        assert 'active' in summary['alpha_pool']
        assert 'available_capital' in summary['alpha_pool']

    def test_process_tick_triggers_stop_loss_and_logs_trade(self) -> None:
        """process_tick should emit exit reasons and trade logs when limits breach."""
        portfolio = DualPoolPortfolio(
            initial_capital=1000.0,
            stop_loss_base=0.05,
            stop_loss_alpha=0.05,
            take_profit_target=0.10,
            commission_rate=0.0,
            spread_rate=0.0,
            slippage_std=0.0,
            funding_enabled=False,
        )

        portfolio.process_tick(
            timestamp=datetime(2023, 1, 1), current_price=100.0, day_high=100.0, day_low=100.0
        )

        result = portfolio.process_tick(
            timestamp=datetime(2023, 1, 2), current_price=100.0, day_high=101.0, day_low=90.0
        )

        assert result['base_exit'] == 'STOP_LOSS'
        assert result['alpha_exit'] == 'STOP_LOSS'
        assert len(portfolio.trade_log) == 1
        trade = portfolio.trade_log[0]
        assert trade['base_exit'] == 'STOP_LOSS'
        assert trade['alpha_exit'] == 'STOP_LOSS'

    def test_process_tick_take_profit_logs_trade(self) -> None:
        """A strong move in favour of the pools should trigger take-profit exits."""
        portfolio = DualPoolPortfolio(
            initial_capital=1000.0,
            take_profit_target=0.05,
            commission_rate=0.0,
            spread_rate=0.0,
            slippage_std=0.0,
            funding_enabled=False,
        )

        portfolio.process_tick(
            timestamp=datetime(2023, 1, 1), current_price=100.0, day_high=100.0, day_low=100.0
        )

        result = portfolio.process_tick(
            timestamp=datetime(2023, 1, 2), current_price=100.0, day_high=110.0, day_low=99.0
        )

        assert result['base_exit'] == 'TAKE_PROFIT'
        assert result['alpha_exit'] == 'TAKE_PROFIT'
        assert portfolio.trade_log
        trade = portfolio.trade_log[-1]
        assert trade['base_exit'] == 'TAKE_PROFIT'
        assert trade['alpha_exit'] == 'TAKE_PROFIT'
        assert trade['base_capital_change'] != 0
