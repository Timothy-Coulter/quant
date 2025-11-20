"""Position data class for portfolio management.

This module defines the Position dataclass used to track individual positions
within a portfolio, including all relevant metrics and calculations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Position:
    """State information for a single position in a portfolio.

    This class represents a trading position with all relevant information
    needed for portfolio management, including quantities, prices, P&L, and costs.
    """

    symbol: str
    quantity: float
    avg_price: float
    timestamp: Any = None
    current_price: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    total_cost: float | None = None
    total_commission: float = 0.0
    entry_timestamp: Any = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Auto-calculate total_cost if not provided
        if self.total_cost is None:
            self.total_cost = self.quantity * self.avg_price

        # Handle timestamp -> entry_timestamp mapping
        if self.entry_timestamp is None and self.timestamp is not None:
            self.entry_timestamp = self.timestamp

    def update_quantity(self, additional_quantity: float, price: float) -> None:
        """Update position quantity and recalculate average price.

        Args:
            additional_quantity: Additional quantity to add
            price: Price of the additional quantity
        """
        total_quantity = self.quantity + additional_quantity
        total_cost = (self.quantity * self.avg_price) + (additional_quantity * price)
        new_avg_price = total_cost / total_quantity

        self.quantity = total_quantity
        self.avg_price = new_avg_price

    def get_current_value(self, current_price: float) -> float:
        """Get current market value of position.

        Args:
            current_price: Current market price

        Returns:
            Current market value
        """
        return self.quantity * current_price

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        return (current_price - self.avg_price) * self.quantity

    def close_position(self, exit_price: float, quantity: float, timestamp: Any) -> float:
        """Close position and return realized P&L.

        Args:
            exit_price: Exit price
            quantity: Quantity to close
            timestamp: Exit timestamp

        Returns:
            Realized P&L
        """
        if quantity > self.quantity:
            quantity = self.quantity

        realized_pnl = (exit_price - self.avg_price) * quantity
        self.quantity -= quantity
        self.realized_pnl += realized_pnl

        return realized_pnl

    def get_weight(self, portfolio_value: float, current_price: float | None = None) -> float:
        """Get position weight in portfolio.

        Args:
            portfolio_value: Total portfolio value
            current_price: Current market price (uses current_price if None)

        Returns:
            Position weight as decimal (0.0 to 1.0)
        """
        if portfolio_value <= 0:
            return 0.0

        price = current_price if current_price is not None else self.current_price
        if price <= 0:
            # Use avg_price as fallback for weight calculation
            price = self.avg_price if self.avg_price > 0 else 0

        if price <= 0:
            return 0.0

        return (self.quantity * price) / portfolio_value

    def update_market_data(
        self, current_price: float, day_high: float, day_low: float
    ) -> dict[str, Any]:
        """Update position with current market data.

        Args:
            current_price: Current market price
            day_high: High price for the day
            day_low: Low price for the day

        Returns:
            Dictionary with position update information
        """
        old_price = self.current_price
        self.current_price = current_price

        # Calculate unrealized P&L
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity

        # Check stop loss
        should_close = False
        close_reason = None
        exit_price = current_price

        if self.stop_loss_price and day_low <= self.stop_loss_price:
            should_close = True
            close_reason = "STOP_LOSS"
            exit_price = self.stop_loss_price
            self.unrealized_pnl = (exit_price - self.avg_price) * self.quantity

        # Check take profit
        elif self.take_profit_price and day_high >= self.take_profit_price:
            should_close = True
            close_reason = "TAKE_PROFIT"
            exit_price = self.take_profit_price
            self.unrealized_pnl = (exit_price - self.avg_price) * self.quantity

        # Calculate daily P&L change
        if old_price != current_price:
            daily_pnl_change = (current_price - old_price) * self.quantity
        else:
            daily_pnl_change = 0.0

        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.avg_price,
            'current_price': current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl_change': daily_pnl_change,
            'should_close': should_close,
            'close_reason': close_reason,
            'exit_price': exit_price,
        }

    def __repr__(self) -> str:
        """String representation of the position."""
        return (
            f"Position(symbol='{self.symbol}', quantity={self.quantity:.2f}, "
            f"avg_price={self.avg_price:.4f}, current_price={self.current_price:.4f}, "
            f"unrealized_pnl={self.unrealized_pnl:.2f}, realized_pnl={self.realized_pnl:.2f})"
        )
