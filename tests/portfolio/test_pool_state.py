"""Test cases for PoolState dataclass.

This module contains unit tests for the PoolState dataclass used to track
individual pools within a dual-pool portfolio management system.
"""

from backtester.portfolio.pool_state import PoolState


class TestPoolState:
    """Test cases for PoolState class."""

    def test_pool_creation(self) -> None:
        """Test basic pool creation."""
        pool = PoolState(
            pool_type='base',
            leverage=2.0,
            max_allocation=0.8,
            capital=1000.0,
            active=True,
            entry_price=100.0,
        )

        assert pool.pool_type == 'base'
        assert pool.leverage == 2.0
        assert pool.max_allocation == 0.8
        assert pool.capital == 1000.0
        assert pool.active is True
        assert pool.entry_price == 100.0
        assert pool.position_value == 0.0
        assert pool.unrealized_pnl == 0.0
        assert pool.realized_pnl == 0.0
        assert pool.available_capital == 0.0
        assert pool.used_capital == 0.0
        assert pool.positions == {}

    def test_allocate_capital(self) -> None:
        """Test capital allocation to pool."""
        pool = PoolState(
            pool_type='base',
            leverage=1.5,
            max_allocation=0.7,
            capital=1000.0,
            available_capital=500.0,
        )

        pool.allocate_capital(300.0)

        assert pool.available_capital == 800.0  # 500.0 + 300.0

    def test_use_capital(self) -> None:
        """Test capital usage from pool."""
        pool = PoolState(
            pool_type='base',
            leverage=2.0,
            max_allocation=0.6,
            available_capital=1000.0,
            used_capital=200.0,
        )

        pool.use_capital(300.0)

        assert pool.available_capital == 700.0  # 1000.0 - 300.0
        assert pool.used_capital == 500.0  # 200.0 + 300.0

    def test_get_current_leverage(self) -> None:
        """Test current leverage calculation."""
        pool = PoolState(
            pool_type='alpha',
            leverage=3.0,
            max_allocation=0.2,
            available_capital=400.0,
            used_capital=600.0,
        )

        current_leverage = pool.get_current_leverage()
        assert abs(current_leverage - 1.67) < 0.01  # (600.0 + 400.0) / 600.0

    def test_add_position(self) -> None:
        """Test adding a position to the pool."""
        pool = PoolState(pool_type='base', leverage=2.0, max_allocation=0.7)

        position_data = {'quantity': 100, 'price': 50.0, 'entry_price': 50.0}

        pool.add_position('AAPL', position_data)

        assert pool.positions is not None
        assert 'AAPL' in pool.positions
        assert pool.positions['AAPL'] == position_data
