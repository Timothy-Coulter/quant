"""Test cases for Position dataclass.

This module contains unit tests for the Position dataclass used to track
individual positions within a portfolio.
"""

from datetime import datetime

from backtester.portfolio.position import Position


class TestPosition:
    """Test cases for Position class."""

    def test_position_creation(self) -> None:
        """Test basic position creation."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            avg_price=150.0,
            timestamp=datetime(2023, 1, 1),
            current_price=155.0,
            stop_loss_price=140.0,
            take_profit_price=170.0,
        )

        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.timestamp == datetime(2023, 1, 1)
        assert position.current_price == 155.0
        assert position.stop_loss_price == 140.0
        assert position.take_profit_price == 170.0
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0
        assert position.commission_paid == 0.0
        assert position.total_cost == 15000.0  # Auto-calculated: 100 * 150.0
        assert position.total_commission == 0.0
        assert position.entry_timestamp == datetime(2023, 1, 1)  # Auto-mapped

    def test_position_creation_without_timestamp(self) -> None:
        """Test position creation without timestamp."""
        position = Position(symbol='MSFT', quantity=50, avg_price=300.0)

        assert position.symbol == 'MSFT'
        assert position.quantity == 50
        assert position.avg_price == 300.0
        assert position.timestamp is None
        assert position.entry_timestamp is None
        assert position.total_cost == 15000.0  # 50 * 300.0

    def test_update_quantity(self) -> None:
        """Test position quantity update and average price recalculation."""
        position = Position(symbol='TSLA', quantity=100, avg_price=200.0)

        # Add 50 more shares at $220
        position.update_quantity(50, 220.0)

        assert position.quantity == 150
        expected_avg = (100 * 200.0 + 50 * 220.0) / 150  # 206.67
        assert abs(position.avg_price - expected_avg) < 0.01

    def test_get_current_value(self) -> None:
        """Test current value calculation."""
        position = Position(symbol='NVDA', quantity=100, avg_price=400.0, current_price=450.0)

        current_value = position.get_current_value(450.0)
        assert current_value == 45000.0  # 100 * 450.0

    def test_get_unrealized_pnl(self) -> None:
        """Test unrealized P&L calculation."""
        position = Position(symbol='META', quantity=50, avg_price=300.0, current_price=350.0)

        unrealized_pnl = position.get_unrealized_pnl(350.0)
        assert unrealized_pnl == 2500.0  # (350.0 - 300.0) * 50

    def test_close_position_full_close(self) -> None:
        """Test full position close."""
        position = Position(symbol='NFLX', quantity=100, avg_price=400.0)

        realized_pnl = position.close_position(450.0, 100, datetime(2023, 1, 2))

        assert realized_pnl == 5000.0  # (450.0 - 400.0) * 100
        assert position.quantity == 0
        assert position.realized_pnl == 5000.0

    def test_close_position_partial_close(self) -> None:
        """Test partial position close."""
        position = Position(symbol='DIS', quantity=200, avg_price=100.0)

        realized_pnl = position.close_position(110.0, 50, datetime(2023, 1, 2))

        assert realized_pnl == 500.0  # (110.0 - 100.0) * 50
        assert position.quantity == 150  # 200 - 50
        assert position.realized_pnl == 500.0

    def test_get_weight_with_portfolio_value(self) -> None:
        """Test position weight calculation with portfolio value."""
        position = Position(symbol='CRM', quantity=100, avg_price=200.0, current_price=220.0)

        weight = position.get_weight(100000.0, 220.0)
        assert weight == 0.22  # (100 * 220.0) / 100000.0

    def test_update_market_data_normal_conditions(self) -> None:
        """Test market data update under normal conditions."""
        position = Position(symbol='INTC', quantity=100, avg_price=50.0, current_price=52.0)

        result = position.update_market_data(55.0, 56.0, 54.0)

        assert position.current_price == 55.0
        assert position.unrealized_pnl == 500.0  # (55.0 - 50.0) * 100
        assert not result['should_close']
        assert result['close_reason'] is None

    def test_update_market_data_stop_loss_triggered(self) -> None:
        """Test market data update with stop loss trigger."""
        position = Position(
            symbol='AMD', quantity=100, avg_price=100.0, stop_loss_price=95.0, current_price=98.0
        )

        result = position.update_market_data(96.0, 97.0, 94.0)  # day_low=94.0 triggers stop loss

        assert position.current_price == 96.0
        assert result['should_close'] is True
        assert result['close_reason'] == 'STOP_LOSS'
        assert result['exit_price'] == 95.0
        assert position.unrealized_pnl == -500.0  # (95.0 - 100.0) * 100

    def test_update_market_data_take_profit_triggered(self) -> None:
        """Test market data update with take profit trigger."""
        position = Position(
            symbol='PYPL',
            quantity=100,
            avg_price=100.0,
            take_profit_price=110.0,
            current_price=105.0,
        )

        result = position.update_market_data(
            109.0, 111.0, 108.0
        )  # day_high=111.0 triggers take profit

        assert position.current_price == 109.0
        assert result['should_close'] is True
        assert result['close_reason'] == 'TAKE_PROFIT'
        assert result['exit_price'] == 110.0
        assert position.unrealized_pnl == 1000.0  # (110.0 - 100.0) * 100

    def test_repr(self) -> None:
        """Test string representation of position."""
        position = Position(
            symbol='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
        )

        repr_str = repr(position)
        assert 'AAPL' in repr_str
        assert '100' in repr_str
        assert '150.0' in repr_str
        assert '155.0' in repr_str
        assert '500.0' in repr_str
        assert '200.0' in repr_str
