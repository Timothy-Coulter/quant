"""Position Sizing Management System.

This module provides comprehensive position sizing functionality with support for
fixed percentage, Kelly criterion, risk-based, volatility-adjusted, and correlation-adjusted sizing.
"""

import logging
from typing import Any

from backtester.risk_management.component_configs.position_sizing_config import (
    PositionSizingConfig,
    SizingMethod,
)


class PositionSizer:
    """Position sizing component for risk management."""

    def __init__(
        self,
        config: PositionSizingConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize position sizer.

        Args:
            config: PositionSizingConfig with position sizing parameters
            logger: Optional logger instance
        """
        self.config: PositionSizingConfig = config or PositionSizingConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def calculate_position_size(
        self,
        portfolio_value: float | None = None,
        volatility: float = 0.0,
        conviction: float = 1.0,
        account_value: float | None = None,
        entry_price: float = 100.0,
        symbol: str | None = None,
        side: str | None = None,
    ) -> float:
        """Calculate position size based on configured sizing method.

        Args:
            portfolio_value: Current portfolio value (defaults to account value)
            volatility: Current market volatility
            conviction: Signal conviction factor (0.0 to 2.0)
            account_value: Total account value (defaults to portfolio_value)
            entry_price: Entry price of the position
            symbol: Optional symbol identifier for logging/compatibility
            side: Optional trade direction (e.g. BUY/SELL)

        Returns:
            Position size as fraction of portfolio
        """
        # Currently symbol and side are informational only but are accepted to maintain
        # interface compatibility with strategy components that pass them through.
        _ = symbol, side

        effective_portfolio_value = (
            portfolio_value if portfolio_value is not None else account_value
        )

        # Handle zero or very small portfolio value
        if (
            effective_portfolio_value is None
            or effective_portfolio_value <= 0
            or (account_value is not None and account_value <= 0)
        ):
            return 0.0

        if account_value is None:
            account_value = effective_portfolio_value

        # Get base position size
        position_size = self._get_base_position_size()

        # Apply method-specific sizing
        position_size = self._apply_method_sizing(
            position_size, account_value, entry_price, volatility, conviction
        )

        # Apply volatility adjustment
        position_size = self._apply_volatility_adjustment(position_size, volatility)

        # Apply bounds
        position_size = self._apply_bounds(position_size)

        self.logger.debug(
            f"Position size calculated: {position_size:.3f} "
            f"(conviction: {conviction:.2f}, vol: {volatility:.3f})"
        )

        return position_size

    def _get_base_position_size(self) -> float:
        """Get base position size from config.

        Returns:
            Base position size
        """
        return self.config.max_position_size

    def _apply_method_sizing(
        self,
        position_size: float,
        account_value: float,
        entry_price: float,
        volatility: float,
        conviction: float,
    ) -> float:
        """Apply method-specific position sizing logic.

        Args:
            position_size: Current position size
            account_value: Account value
            entry_price: Entry price
            volatility: Market volatility
            conviction: Conviction factor

        Returns:
            Adjusted position size
        """
        if self.config.sizing_method == SizingMethod.RISK_BASED:
            return self._apply_risk_based_sizing(account_value, entry_price, volatility, conviction)
        elif self.config.sizing_method == SizingMethod.VOLATILITY_ADJUSTED:
            return self._calculate_volatility_adjusted_size(account_value, entry_price, volatility)
        elif self.config.sizing_method == SizingMethod.CORRELATION_ADJUSTED:
            return self._calculate_correlation_adjusted_size(account_value, entry_price)
        elif self.config.sizing_method == SizingMethod.KELLY:
            return self._calculate_kelly_size(account_value)
        else:
            # Fixed percentage method
            return position_size * conviction

    def _apply_risk_based_sizing(
        self,
        account_value: float,
        entry_price: float,
        volatility: float,
        conviction: float,
    ) -> float:
        """Apply risk-based sizing with conviction factors.

        Args:
            account_value: Account value
            entry_price: Entry price
            volatility: Market volatility
            conviction: Conviction factor

        Returns:
            Risk-based position size
        """
        position_size = self._calculate_risk_based_size(account_value, entry_price, volatility)

        # Apply conviction factors from config
        conviction_factor = self._get_conviction_factor(conviction)
        position_size *= conviction_factor

        return position_size

    def _get_conviction_factor(self, conviction: float) -> float:
        """Get conviction factor based on conviction level.

        Args:
            conviction: Conviction value

        Returns:
            Conviction factor
        """
        if conviction <= 0.6:
            return self.config.conviction_factors.get('low', 0.7)
        elif conviction <= 1.2:
            return self.config.conviction_factors.get('medium', 1.0)
        else:
            return self.config.conviction_factors.get('high', 1.3)

    def _apply_volatility_adjustment(self, position_size: float, volatility: float) -> float:
        """Apply volatility adjustment if enabled.

        Args:
            position_size: Current position size
            volatility: Market volatility

        Returns:
            Volatility-adjusted position size
        """
        if (
            self.config.volatility_adjustment
            and volatility > 0
            and self.config.sizing_method != SizingMethod.RISK_BASED
        ):
            volatility_factor = max(0.1, 1.0 - volatility * 5)
            position_size *= volatility_factor
        return position_size

    def _apply_bounds(self, position_size: float) -> float:
        """Apply position size bounds.

        Args:
            position_size: Current position size

        Returns:
            Bounded position size
        """
        if self.config.sizing_method != SizingMethod.RISK_BASED:
            return max(
                self.config.min_position_size, min(position_size, self.config.max_position_size)
            )
        return position_size

    def _calculate_risk_based_size(
        self, account_value: float, entry_price: float, volatility: float
    ) -> float:
        """Calculate risk-based position size.

        Args:
            account_value: Total account value
            entry_price: Entry price of the position
            volatility: Current market volatility

        Returns:
            Risk-based position size
        """
        # Handle invalid inputs
        if account_value <= 0 or entry_price <= 0:
            return 0.0

        # Base position size from risk per trade
        position_value = account_value * self.config.risk_per_trade

        # Adjust for volatility
        if volatility > 0:
            volatility_factor = max(0.1, 1.0 - volatility * 5)
            position_value *= volatility_factor

        # Convert to fraction of account
        position_size = position_value / account_value

        # Apply bounds for consistency with other methods
        return max(self.config.min_position_size, min(position_size, self.config.max_position_size))

    def _calculate_volatility_adjusted_size(
        self, account_value: float, entry_price: float, volatility: float
    ) -> float:
        """Calculate volatility-adjusted position size.

        Args:
            account_value: Total account value
            entry_price: Entry price of the position
            volatility: Current market volatility

        Returns:
            Volatility-adjusted position size
        """
        # Base position size
        base_size = self.config.max_position_size

        # Reduce position size for higher volatility
        volatility_adjustment = max(0.1, 1.0 - volatility * 5)

        return base_size * volatility_adjustment

    def _calculate_correlation_adjusted_size(
        self, account_value: float, entry_price: float, portfolio_correlation: float = 0.0
    ) -> float:
        """Calculate correlation-adjusted position size.

        Args:
            account_value: Total account value
            entry_price: Entry price of the position
            portfolio_correlation: Correlation with existing portfolio

        Returns:
            Correlation-adjusted position size
        """
        # Base position size
        base_size = self.config.max_position_size

        # Reduce position size for higher correlation
        correlation_adjustment = max(0.5, 1.0 - portfolio_correlation)

        return base_size * correlation_adjustment

    def _calculate_kelly_size(self, account_value: float) -> float:
        """Calculate Kelly criterion position size.

        Args:
            account_value: Total account value

        Returns:
            Kelly criterion position size
        """
        if (
            self.config.kelly_win_rate is not None
            and self.config.kelly_avg_win is not None
            and self.config.kelly_avg_loss is not None
            and self.config.kelly_avg_loss > 0
        ):
            win_rate = self.config.kelly_win_rate
            avg_win = self.config.kelly_avg_win
            avg_loss = self.config.kelly_avg_loss

            # Kelly formula: f* = (bp - q) / b
            # where b = odds ratio, p = win probability, q = loss probability
            b = avg_win / avg_loss  # odds ratio
            p = win_rate  # win probability
            q = 1 - win_rate  # loss probability

            kelly_fraction = (b * p - q) / b

            # Return as fraction of portfolio, not capped immediately
            # Cap at maximum position size
            return max(0.0, min(kelly_fraction, self.config.max_position_size))

        # Fallback to fixed percentage if Kelly parameters not available
        return self.config.risk_per_trade

    def calculate_position_size_fixed_risk(
        self, account_value: float, entry_price: float, stop_price: float
    ) -> float:
        """Calculate position size based on fixed risk per trade.

        Args:
            account_value: Account value
            entry_price: Entry price
            stop_price: Stop loss price

        Returns:
            Position size in value terms
        """
        risk_amount = account_value * self.config.risk_per_trade
        stop_distance = abs(entry_price - stop_price)

        if stop_distance == 0:
            return 0.0

        # Position size in shares = risk_amount / stop_distance
        # But we need to return dollar amount, not shares
        position_value = risk_amount / (stop_distance / entry_price)
        max_position_value = account_value * self.config.max_position_size

        return min(position_value, max_position_value)

    def calculate_position_size_percentage(
        self, account_value: float, entry_price: float, percentage: float
    ) -> float:
        """Calculate position size based on percentage of account.

        Args:
            account_value: Account value
            entry_price: Entry price
            percentage: Percentage of account to use

        Returns:
            Position size in shares
        """
        position_value = account_value * percentage
        return position_value / entry_price

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion fraction.

        Args:
            win_rate: Historical win rate
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Kelly fraction
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # odds ratio
        q = 1 - win_rate  # loss probability
        p = win_rate  # win probability

        # Kelly formula: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b

        # Return as fraction of portfolio, not capped immediately
        # Cap at maximum position size
        return max(0.0, min(kelly_fraction, self.config.max_position_size))

    def calculate_position_size_risk_based(self, **params: Any) -> float:
        """Comprehensive risk-based position sizing.

        Args:
            **params: Risk parameters including account_value, entry_price,
                     stop_price, volatility, correlation, conviction_level

        Returns:
            Risk-based position size
        """
        account_value = params.get('account_value', 10000.0)
        entry_price = params.get('entry_price', 100.0)
        stop_price = params.get('stop_price', entry_price * 0.95)
        volatility = params.get('volatility', 0.0)
        correlation = params.get('correlation', 0.0)
        conviction_level = params.get('conviction_level', 'medium')

        # Base position size from fixed risk
        position_size = self.calculate_position_size_fixed_risk(
            account_value, entry_price, stop_price
        )

        # Apply volatility adjustment
        if volatility > 0:
            volatility_factor = max(0.1, 1.0 - volatility * 5)
            position_size *= volatility_factor

        # Apply correlation adjustment
        if correlation > 0:
            correlation_factor = max(0.5, 1.0 - correlation)
            position_size *= correlation_factor

        # Apply conviction adjustment
        conviction_factors = self.config.conviction_factors
        conviction_factor = conviction_factors.get(conviction_level, 1.0)
        position_size *= conviction_factor

        # Convert to fraction of portfolio
        position_size = position_size / account_value

        # Apply bounds as fractions
        max_size = self.config.max_position_size
        min_size = self.config.min_position_size

        return float(max(min_size, min(position_size, max_size)))

    def enforce_constraints(
        self, position_size: float, account_value: float, entry_price: float
    ) -> float:
        """Enforce position sizing constraints.

        Args:
            position_size: Calculated position size
            account_value: Account value
            entry_price: Entry price

        Returns:
            Constrained position size
        """
        # Calculate actual position value
        position_value = position_size * entry_price

        # Check minimum size constraint
        min_position_value = account_value * self.config.min_position_size
        if position_value < min_position_value:
            position_size = min_position_value / entry_price

        # Check maximum size constraint
        max_position_value = account_value * self.config.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price

        return position_size

    def get_sizing_metrics(self) -> dict[str, Any]:
        """Get position sizing configuration metrics.

        Returns:
            Dictionary with sizing configuration
        """
        return {
            'max_position_size': self.config.max_position_size,
            'min_position_size': self.config.min_position_size,
            'risk_per_trade': self.config.risk_per_trade,
            'sizing_method': self.config.sizing_method,
            'volatility_adjustment': self.config.volatility_adjustment,
            'max_daily_trades': self.config.max_daily_trades,
            'max_sector_exposure': self.config.max_sector_exposure,
            'max_correlation': self.config.max_correlation,
            'conviction_factors': self.config.conviction_factors,
        }
