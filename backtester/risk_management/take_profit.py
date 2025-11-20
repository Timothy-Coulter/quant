"""Take Profit Management System.

This module provides comprehensive take profit functionality with support for
fixed, percentage, trailing, and dynamic take profit strategies.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from backtester.risk_management.component_configs.take_profit_config import (
    TakeProfitConfig,
    TakeProfitType,
)


class TakeProfit:
    """Take profit management with multiple types of mechanisms."""

    def __init__(
        self,
        config: TakeProfitConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the take profit mechanism.

        Args:
            config: TakeProfitConfig with take profit parameters
            logger: Optional logger instance
        """
        self.config: TakeProfitConfig = config or TakeProfitConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # State tracking
        self.activation_price: float | None = self.config.activation_price
        self.entry_price: float | None = None
        self.lowest_price: float = float('inf')  # For trailing take profit
        self.is_active: bool = True
        self.triggered: bool = False
        self.target_price: float | None = None

        # Test compatibility attributes
        self.take_profit_type = self.config.take_profit_type
        self.take_profit_value = self.config.take_profit_value
        self.triggered_price: float | None = None
        self.triggered_timestamp: datetime | None = None
        self.trigger_price: float | None = None
        self.trigger_time: datetime | None = None
        self.stop_loss_value: float | None = None
        self.trailing_take_profit_pct = self.config.trailing_take_profit_pct
        self.partial_take_profit_levels: list[float] = []
        self.scaling_factors: list[float] = []
        self.enforce_rr_ratio: bool = False

        # Attributes for trailing stop functionality
        self.highest_price: float = 0.0
        self.stop_price: float | None = None
        self.trailing_stop_pct: float = self.trailing_take_profit_pct

    def initialize_position(self, entry_price: float, timestamp: pd.Timestamp) -> None:
        """Initialize take profit for a new position.

        Args:
            entry_price: Entry price of the position
            timestamp: Entry timestamp
        """
        self.entry_price = entry_price
        self.lowest_price = entry_price
        self.is_active = True
        self.triggered = False
        self.trigger_price = None
        self.trigger_time = None

        # Set activation price if configured
        if self.config.activation_price is None:
            self.activation_price = entry_price

        # Calculate initial target price
        self.target_price = self._calculate_target_price(entry_price, entry_price)

        self.logger.debug(
            f"Take profit initialized: entry={entry_price:.4f}, target={self.target_price:.4f}"
        )

    def update(self, current_price: float, timestamp: pd.Timestamp) -> dict[str, Any]:
        """Update take profit with current market price.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            Dictionary with take profit status and any trigger information
        """
        if not self.is_active or self.entry_price is None:
            return {
                'triggered': False,
                'target_price': None,
                'action': 'NONE',
                'reason': 'Take profit not active',
            }

        result = {
            'triggered': False,
            'target_price': self.target_price,
            'action': 'NONE',
            'reason': 'No action required',
        }

        # Update lowest price for trailing take profit
        if (
            self.config.take_profit_type == TakeProfitType.TRAILING
            and current_price < self.lowest_price
        ):
            self.lowest_price = current_price

        # Recalculate target price
        old_target_price = self.target_price
        self.target_price = self._calculate_target_price(current_price, self.lowest_price)

        # Check if take profit is triggered
        # For trailing take profit, only trigger if current price is at or above target
        # and the target was calculated based on downward movement
        should_trigger = False

        if self.config.take_profit_type == TakeProfitType.TRAILING:
            # For trailing take profit, trigger only if price has moved down to target
            # and the target was set based on a lower price
            tolerance = 1e-8  # Small tolerance for floating point precision
            should_trigger = (
                current_price >= self.target_price - tolerance
                and current_price
                <= self.lowest_price * (1 + self.config.trail_distance) + tolerance
            )
        else:
            # For other types, trigger normally with tolerance
            tolerance = 1e-8  # Small tolerance for floating point precision
            should_trigger = current_price >= self.target_price - tolerance

        if should_trigger:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            result.update(
                {
                    'triggered': True,
                    'target_price': self.target_price,
                    'action': 'TAKE_PROFIT',
                    'reason': f'Take profit triggered at {current_price:.4f}',
                    'exit_price': current_price,
                    'pnl_pct': (
                        (current_price - self.entry_price) / self.entry_price
                        if self.entry_price
                        else 0
                    ),
                    'timestamp': timestamp,
                }
            )
            self.is_active = False

            self.logger.info(
                f"Take profit triggered: {result['reason']}, P&L: {result['pnl_pct']:.2%}"
            )

        # Check maximum gain limit
        elif self.config.max_gain_value is not None:
            gain_amount = current_price - self.entry_price
            if gain_amount >= self.config.max_gain_value:
                self.triggered = True
                self.trigger_price = current_price
                self.trigger_time = timestamp
                result.update(
                    {
                        'triggered': True,
                        'target_price': self.target_price,
                        'action': 'MAX_GAIN',
                        'reason': f'Maximum gain target reached: {gain_amount:.4f}',
                        'exit_price': current_price,
                        'pnl_pct': (
                            (current_price - self.entry_price) / self.entry_price
                            if self.entry_price
                            else 0
                        ),
                        'timestamp': timestamp,
                    }
                )
                self.is_active = False

                self.logger.info(f"Maximum gain target reached: {result['reason']}")

        # Log target price changes for trailing profit
        if (
            self.config.take_profit_type == TakeProfitType.TRAILING
            and old_target_price != self.target_price
        ):
            self.logger.debug(
                f"Trailing profit updated: {old_target_price:.4f} -> {self.target_price:.4f}"
            )

        return result

    def _calculate_target_price(self, current_price: float, reference_price: float) -> float:
        """Calculate target price based on configuration.

        Args:
            current_price: Current market price
            reference_price: Reference price (lowest for trailing, entry for others)

        Returns:
            Calculated target price
        """
        if self.config.take_profit_type == TakeProfitType.FIXED:
            # For fixed take profit, calculate as percentage of reference price
            return reference_price * (1 + self.config.take_profit_value)

        elif self.config.take_profit_type == TakeProfitType.PERCENTAGE:
            return reference_price * (1 + self.config.take_profit_value)

        elif self.config.take_profit_type == TakeProfitType.PRICE:
            # For price take profit, treat the value as percentage
            return reference_price * (1 + self.config.take_profit_value)

        elif self.config.take_profit_type == TakeProfitType.TRAILING:
            trail_target = reference_price * (1 + self.config.trail_distance)

            # Ensure we only move the target down (never up)
            if self.target_price is not None:
                max_target = self.target_price - (reference_price * self.config.trail_step)
                return min(trail_target, max_target)
            else:
                return trail_target

        return reference_price * 1.05  # Default 5% profit target

    def get_status(self) -> dict[str, Any]:
        """Get current take profit status.

        Returns:
            Dictionary with take profit status information
        """
        return {
            'active': self.is_active,
            'triggered': self.triggered,
            'target_price': self.target_price,
            'entry_price': self.entry_price,
            'lowest_price': self.lowest_price if self.lowest_price != float('inf') else None,
            'config': {
                'type': (
                    self.config.take_profit_type.value
                    if hasattr(self.config.take_profit_type, 'value')
                    else str(self.config.take_profit_type)
                ),
                'value': self.config.take_profit_value,
                'trail_distance': self.config.trail_distance,
                'max_gain_value': self.config.max_gain_value,
            },
        }

    def calculate_target_price(self, entry_price: float, side: str = 'long') -> float:
        """Calculate target price based on entry price and side.

        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position

        Returns:
            Calculated target price
        """
        if side.lower() == 'short':
            # For short positions, target is below entry price
            if self.config.take_profit_type == TakeProfitType.FIXED:
                # For fixed take profit, calculate as percentage of entry price
                return entry_price * (1 - self.config.take_profit_value)
            elif self.config.take_profit_type == TakeProfitType.PERCENTAGE:
                return entry_price * (1 - self.config.take_profit_value)
            elif self.config.take_profit_type == TakeProfitType.PRICE:
                return self.config.take_profit_value

        # Long position logic (default)
        return self._calculate_target_price(entry_price, entry_price)

    def check_target(self, entry_price: float, current_price: float, side: str = 'long') -> bool:
        """Check if take profit should trigger.

        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            side: 'long' or 'short' position

        Returns:
            True if take profit should trigger
        """
        # Update entry price if not set
        if self.entry_price is None:
            self.entry_price = entry_price

        target_price = self.calculate_target_price(entry_price, side)

        if side.lower() == 'long':
            return current_price >= target_price
        else:  # short
            return current_price <= target_price

    def activate(self) -> None:
        """Activate the take profit."""
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate the take profit."""
        self.is_active = False

    def trigger(self) -> None:
        """Manually trigger the take profit."""
        self.triggered = True
        self.trigger_price = self.target_price
        self.trigger_time = datetime.now()
        self.is_active = False

    def reset(self) -> None:
        """Reset take profit to initial state."""
        self.activation_price = self.config.activation_price
        self.entry_price = None
        self.lowest_price = float('inf')
        self.is_active = True
        self.triggered = False
        self.trigger_price = None
        self.trigger_time = None
        self.target_price = None
        self.logger.debug("Take profit reset")

    def setup_trailing_target(self, entry_price: float, side: str = 'long') -> None:
        """Setup trailing take profit target.

        Args:
            entry_price: Entry price of the position
            side: 'long' or 'short' position
        """
        self.entry_price = entry_price
        self.lowest_price = entry_price
        if side.lower() == 'long':
            self.target_price = entry_price * (1 + self.trailing_take_profit_pct)
        else:
            self.target_price = entry_price * (1 - self.trailing_take_profit_pct)

    def setup_partial_take_profit(
        self, entry_price: float, quantity: int, side: str = 'long'
    ) -> list[float]:
        """Setup partial take profit levels.

        Args:
            entry_price: Entry price
            quantity: Position quantity
            side: 'long' or 'short'

        Returns:
            List of partial take profit levels
        """
        if not self.partial_take_profit_levels:
            self.partial_take_profit_levels = [0.5, 0.25, 0.25]  # Default: 50%, 25%, 25%

        levels = []
        for i, _fraction in enumerate(self.partial_take_profit_levels):
            if side.lower() == 'long':
                level_price = entry_price * (1 + (i + 1) * 0.02)  # 2%, 4%, 6% etc.
            else:
                level_price = entry_price * (1 - (i + 1) * 0.02)  # -2%, -4%, -6% etc.
            levels.append(level_price)

        return levels

    def calculate_scaled_target(
        self, entry_price: float, confidence_level: str, side: str = 'long'
    ) -> float:
        """Calculate scaled target based on confidence level.

        Args:
            entry_price: Entry price
            confidence_level: 'low', 'medium', 'high'
            side: 'long' or 'short'

        Returns:
            Scaled target price
        """
        scaling_factors = {'low': 1.02, 'medium': 1.05, 'high': 1.08}
        factor = scaling_factors.get(confidence_level.lower(), 1.05)

        if side.lower() == 'long':
            return entry_price * factor
        else:
            return entry_price * (2 - factor)

    def update_trailing_target(
        self, current_price: float, timestamp: pd.Timestamp
    ) -> dict[str, Any]:
        """Update trailing take profit target with current market data.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            Dictionary with update results
        """
        if not self.is_active or self.entry_price is None:
            return {'triggered': False, 'action': 'NONE'}

        # Update lowest price for trailing take profit
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        # Recalculate target price
        self.target_price = self.lowest_price * (1 + self.trailing_take_profit_pct)

        # Check if target is reached
        triggered = current_price >= self.target_price
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            self.is_active = False

        return {
            'triggered': triggered,
            'target_price': self.target_price,
            'lowest_price': self.lowest_price,
            'action': 'TAKE_PROFIT' if triggered else 'NONE',
        }

    def update_trailing_stop(self, current_price: float, timestamp: pd.Timestamp) -> dict[str, Any]:
        """Update trailing stop with current market data.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            Dictionary with update results
        """
        if not self.is_active or self.entry_price is None:
            return {'triggered': False, 'action': 'NONE'}

        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Recalculate stop price
        self.stop_price = self.highest_price * (1 - self.trailing_stop_pct)

        # Check if stop is triggered
        triggered = current_price <= self.stop_price
        if triggered:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_time = timestamp
            self.is_active = False

        return {
            'triggered': triggered,
            'stop_price': self.stop_price,
            'highest_price': self.highest_price,
            'action': 'STOP_LOSS' if triggered else 'NONE',
        }
