"""Pool state management for dual-pool portfolio strategy.

This module defines the PoolState dataclass used to track individual pools
within a dual-pool portfolio management system.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PoolState:
    """State information for a single pool in a dual-pool portfolio.

    This class represents a pool with leverage, allocation limits, and health monitoring.
    """

    pool_type: str
    leverage: float
    max_allocation: float
    capital: float = 0.0
    active: bool = False
    entry_price: float = 0.0
    position_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    available_capital: float = 0.0
    used_capital: float = 0.0
    positions: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing for PoolState."""
        if self.positions is None:
            self.positions = {}

    def allocate_capital(self, amount: float) -> None:
        """Allocate capital to this pool.

        Args:
            amount: Amount of capital to allocate
        """
        self.available_capital += amount

    def use_capital(self, amount: float) -> None:
        """Use capital from this pool.

        Args:
            amount: Amount of capital to use
        """
        if self.available_capital >= amount:
            self.available_capital -= amount
            self.used_capital += amount

    def get_current_leverage(self) -> float:
        """Get current leverage ratio.

        Returns:
            Current leverage ratio
        """
        if self.used_capital == 0:
            return 1.0
        return (self.used_capital + self.available_capital) / self.used_capital

    def get_max_leverage(self) -> float:
        """Get maximum leverage for this pool.

        Returns:
            Maximum leverage for this pool
        """
        return self.leverage

    def check_health(self) -> dict[str, Any]:
        """Check pool health metrics.

        Returns:
            Dictionary with health metrics
        """
        current_leverage = self.get_current_leverage()
        utilization_rate = (
            self.used_capital / (self.used_capital + self.available_capital)
            if (self.used_capital + self.available_capital) > 0
            else 0
        )

        health_status = "healthy"
        if current_leverage > self.leverage * 0.9:
            health_status = "high_risk"
        elif current_leverage > self.leverage * 0.7:
            health_status = "moderate"

        return {
            'leverage_ratio': current_leverage,
            'utilization_rate': utilization_rate,
            'health_status': health_status,
        }

    def can_allocate(self, amount: float) -> bool:
        """Check if a capital amount can be allocated.

        Args:
            amount: Amount to allocate

        Returns:
            True if amount can be allocated
        """
        return self.available_capital >= amount

    def get_available_margin(self) -> float:
        """Get available margin for new positions.

        Returns:
            Available margin
        """
        return self.available_capital

    def get_utilized_margin(self) -> float:
        """Get currently utilized margin.

        Returns:
            Utilized margin
        """
        return self.used_capital

    def get_pool_value(self) -> float:
        """Get current total pool value.

        Returns:
            Total pool value (capital + unrealized P&L)
        """
        return self.capital + self.unrealized_pnl

    def add_position(self, symbol: str, position_data: Any) -> None:
        """Add a position to this pool.

        Args:
            symbol: Trading symbol
            position_data: Position data
        """
        if self.positions is None:
            self.positions = {}
        self.positions[symbol] = position_data

    def remove_position(self, symbol: str) -> None:
        """Remove a position from this pool.

        Args:
            symbol: Trading symbol to remove
        """
        if self.positions and symbol in self.positions:
            del self.positions[symbol]

    def get_position(self, symbol: str) -> Any:
        """Get a position from this pool.

        Args:
            symbol: Trading symbol

        Returns:
            Position data or None if not found
        """
        if self.positions and symbol in self.positions:
            return self.positions[symbol]
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if this pool has a position for the symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if position exists
        """
        return self.positions is not None and symbol in self.positions

    def __repr__(self) -> str:
        """String representation of the pool state."""
        return (
            f"PoolState(pool_type='{self.pool_type}', leverage={self.leverage:.1f}, "
            f"capital={self.capital:.2f}, active={self.active}, "
            f"available_capital={self.available_capital:.2f}, "
            f"used_capital={self.used_capital:.2f}, "
            f"unrealized_pnl={self.unrealized_pnl:.2f})"
        )
