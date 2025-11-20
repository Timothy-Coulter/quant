"""Test cases for GeneralPortfolio class.

This module contains comprehensive unit tests for the GeneralPortfolio class
used for multi-asset portfolio management.
"""

from datetime import datetime

import pandas as pd
import pytest

from backtester.portfolio.general_portfolio import GeneralPortfolio


class TestGeneralPortfolio:
    """Test cases for GeneralPortfolio class."""

    def test_init_default(self) -> None:
        """Test portfolio initialization with default values."""
        portfolio = GeneralPortfolio()

        assert portfolio.initial_capital == 100.0
        assert portfolio.commission_rate == 0.001
        assert portfolio.max_positions == 10
        assert portfolio.cash == 100.0
        assert portfolio.positions == {}
        assert portfolio.portfolio_values == [100.0]

    def test_init_custom(self) -> None:
        """Test portfolio initialization with custom values."""
        portfolio = GeneralPortfolio(
            initial_capital=10000.0, max_positions=20, commission_rate=0.002, tax_rate=0.25
        )

        assert portfolio.initial_capital == 10000.0
        assert portfolio.max_positions == 20
        assert portfolio.commission_rate == 0.002
        assert portfolio.tax_rate == 0.25
        assert portfolio.cash == 10000.0

    def test_total_value_property(self) -> None:
        """Test total_value property calculation."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)

        # Add a position
        from backtester.portfolio.position import Position

        portfolio.positions['AAPL'] = Position(
            symbol='AAPL', quantity=10, avg_price=100.0, current_price=110.0
        )

        total_value = portfolio.total_value
        assert total_value == 2100.0  # 1000 cash + 1100 position value

    def test_add_position_success(self) -> None:
        """Test successful position addition."""
        portfolio = GeneralPortfolio(initial_capital=10000.0)

        result = portfolio.add_position(
            symbol='AAPL', quantity=10, price=50.0, timestamp=datetime(2023, 1, 1)
        )

        assert result is True
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.symbol == 'AAPL'
        assert position.quantity == 10
        assert position.avg_price == 50.0
        assert portfolio.cash < 10000.0  # Cash should be reduced

    def test_add_position_max_positions_reached(self) -> None:
        """Test position addition when max positions reached."""
        portfolio = GeneralPortfolio(initial_capital=10000.0, max_positions=2)

        # Fill up to max positions
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))
        portfolio.add_position('MSFT', 5, 200.0, datetime(2023, 1, 2))

        # Try to add third position
        result = portfolio.add_position('GOOGL', 2, 300.0, datetime(2023, 1, 3))

        assert result is False
        assert 'GOOGL' not in portfolio.positions
        assert len(portfolio.positions) == 2

    def test_close_position_success(self) -> None:
        """Test successful position closing."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        result = portfolio.close_position(
            symbol='AAPL', price=120.0, timestamp=datetime(2023, 1, 2)
        )

        assert result is True
        assert 'AAPL' not in portfolio.positions
        assert portfolio.cash > 1000.0  # Should have profit

    def test_close_nonexistent_position(self) -> None:
        """Test closing a position that doesn't exist."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)

        result = portfolio.close_position(
            symbol='NONEXISTENT', price=100.0, timestamp=datetime(2023, 1, 1)
        )

        assert result is False

    def test_update_position(self) -> None:
        """Test position update functionality."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        result = portfolio.update_position(
            symbol='AAPL', quantity=5, price=110.0, timestamp=datetime(2023, 1, 2)
        )

        assert result is True
        position = portfolio.positions['AAPL']
        assert position.quantity == 15  # 10 + 5
        # Average price should be weighted: (10*100 + 5*110) / 15 = 103.33
        expected_avg = (10 * 100.0 + 5 * 110.0) / 15
        assert abs(position.avg_price - expected_avg) < 0.01

    def test_process_tick(self) -> None:
        """Test market tick processing."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        # Create market data
        market_data = {'AAPL': pd.DataFrame({'Close': [105.0], 'High': [106.0], 'Low': [104.0]})}

        result = portfolio.process_tick(timestamp=datetime(2023, 1, 2), market_data=market_data)

        assert 'timestamp' in result
        assert 'total_value' in result
        assert 'cash' in result
        assert 'position_count' in result
        assert result['position_count'] == 1

    def test_can_add_position(self) -> None:
        """Test can_add_position method."""
        portfolio = GeneralPortfolio(initial_capital=1000.0, max_positions=2)

        # Should be able to add
        assert portfolio.can_add_position('AAPL') is True

        # Add a position
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        # Should still be able to add different symbol
        assert portfolio.can_add_position('MSFT') is True

        # Add second position
        portfolio.add_position('MSFT', 5, 200.0, datetime(2023, 1, 2))

        # Should not be able to add more
        assert portfolio.can_add_position('GOOGL') is False

        # Should not be able to add existing symbol
        assert portfolio.can_add_position('AAPL') is False

    def test_get_position_value_current(self) -> None:
        """Test current position value retrieval."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        # Update current price
        portfolio.positions['AAPL'].current_price = 120.0

        value = portfolio.get_position_value_current('AAPL')
        assert value == 1200.0  # 10 * 120.0

        # Test non-existent position
        value = portfolio.get_position_value_current('NONEXISTENT')
        assert value == 0.0

    def test_get_position_allocation(self) -> None:
        """Test position allocation calculation."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))
        portfolio.positions['AAPL'].current_price = 150.0

        allocation = portfolio.get_position_allocation('AAPL')
        # Position value = 1500, total value = 1500 (position) + remaining cash
        expected_allocation = 1500.0 / (1500.0 + portfolio.cash)
        assert abs(allocation - expected_allocation) < 0.01

    def test_reset(self) -> None:
        """Test portfolio reset functionality."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))
        portfolio.cash = 500.0
        portfolio.trade_log.append({'test': 'data'})
        portfolio.cumulative_tax = 25.0

        portfolio.reset()

        assert portfolio.cash == 1000.0
        assert portfolio.positions == {}
        assert portfolio.trade_log == []
        assert portfolio.cumulative_tax == 0.0
        assert portfolio.portfolio_values == [1000.0]

    def test_get_performance_metrics(self) -> None:
        """Test performance metrics calculation."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.portfolio_values = [1000.0, 1100.0, 1200.0, 1150.0]
        portfolio.trade_log = [
            {'action': 'CLOSE', 'realized_pnl': 50.0},
            {'action': 'CLOSE', 'realized_pnl': -20.0},
        ]

        metrics = portfolio.get_performance_metrics()

        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'final_cash' in metrics
        assert 'current_positions' in metrics
        assert metrics['current_positions'] == 0

    def test_get_summary(self) -> None:
        """Test portfolio summary generation."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))

        summary = portfolio.get_summary()

        assert 'total_value' in summary
        assert 'cash' in summary
        assert 'positions' in summary
        assert 'current_positions' in summary
        assert 'max_positions' in summary
        assert summary['current_positions'] == 1
        assert summary['max_positions'] == 10

    def test_apply_fill_updates_positions(self) -> None:
        """Executed fills should create/update and close positions."""
        portfolio = GeneralPortfolio(initial_capital=1000.0)
        timestamp = datetime(2023, 1, 1)

        # Buy creates a new position
        portfolio.apply_fill(
            symbol='AAPL',
            side='BUY',
            quantity=5,
            price=100.0,
            timestamp=timestamp,
        )
        assert 'AAPL' in portfolio.positions
        assert portfolio.positions['AAPL'].quantity == 5

        # Additional buy updates the existing position
        portfolio.apply_fill(
            symbol='AAPL',
            side='BUY',
            quantity=5,
            price=110.0,
            timestamp=timestamp,
        )
        assert portfolio.positions['AAPL'].quantity == 10

        # Sells reduce and eventually remove the position
        portfolio.apply_fill(
            symbol='AAPL',
            side='SELL',
            quantity=5,
            price=120.0,
            timestamp=timestamp,
        )
        assert portfolio.positions['AAPL'].quantity == 5

        portfolio.apply_fill(
            symbol='AAPL',
            side='SELL',
            quantity=5,
            price=125.0,
            timestamp=timestamp,
        )
        assert 'AAPL' not in portfolio.positions

    def test_process_tick_applies_financing_costs(self) -> None:
        """process_tick should deduct financing costs for leveraged positions."""
        portfolio = GeneralPortfolio(
            initial_capital=1000.0,
            commission_rate=0.0,
            spread_rate=0.0,
            slippage_std=0.0,
            interest_rate_daily=0.01,
        )
        portfolio.add_position('AAPL', 10, 100.0, datetime(2023, 1, 1))
        cash_before = portfolio.cash

        result = portfolio.process_tick(
            timestamp=datetime(2023, 1, 2),
            market_data={'AAPL': pd.DataFrame([{'Close': 120.0, 'High': 121.0, 'Low': 119.0}])},
        )

        expected_cost = 10 * 120.0 * 0.01
        assert result['position_count'] == 1
        assert pytest.approx(result['financing_cost']) == pytest.approx(expected_cost)
        assert pytest.approx(portfolio.cash) == pytest.approx(cash_before - expected_cost)

    @pytest.mark.parametrize(
        "day_high,day_low,expected_reason",
        [
            (106.0, 100.0, 'TAKE_PROFIT'),
            (100.0, 94.0, 'STOP_LOSS'),
        ],
    )
    def test_process_tick_closes_positions_on_thresholds(
        self, day_high: float, day_low: float, expected_reason: str
    ) -> None:
        """Stop-loss and take-profit levels should close positions and log trades."""
        portfolio = GeneralPortfolio(
            initial_capital=1000.0,
            commission_rate=0.0,
            spread_rate=0.0,
            slippage_std=0.0,
        )
        portfolio.add_position(
            'AAPL',
            10,
            100.0,
            datetime(2023, 1, 1),
            stop_loss_price=95.0,
            take_profit_price=105.0,
        )

        result = portfolio.process_tick(
            timestamp=datetime(2023, 1, 2),
            market_data={
                'AAPL': pd.DataFrame([{'Close': day_high, 'High': day_high, 'Low': day_low}])
            },
        )

        assert result['position_updates'][0]['close_reason'] == expected_reason
        assert 'AAPL' not in portfolio.positions
        assert result['position_count'] == 0
        assert portfolio.trade_log[-1]['action'] == 'CLOSE'
        assert portfolio.trade_log[-1]['reason'] == expected_reason
