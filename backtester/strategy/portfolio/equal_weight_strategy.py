"""Equal weight portfolio strategy implementation.

This module implements the equal weight portfolio strategy that allocates
capital equally across all assets in the portfolio.
"""

import logging
from typing import Any

import pandas as pd

from .base_portfolio_strategy import BasePortfolioStrategy
from .portfolio_strategy_config import PortfolioStrategyConfig


class EqualWeightStrategy(BasePortfolioStrategy):
    """Equal weight portfolio strategy implementation.

    This strategy allocates capital equally across all assets in the portfolio.
    Each asset receives the same weight regardless of market conditions.
    """

    def __init__(self, config: PortfolioStrategyConfig, event_bus: Any) -> None:
        """Initialize the equal weight strategy.

        Args:
            config: Portfolio strategy configuration
            event_bus: Event bus for event-driven communication
        """
        super().__init__(config, event_bus)

        self.logger = logging.getLogger(__name__)
        self.name = config.strategy_name
        self.symbols = config.symbols

        # Strategy-specific parameters
        self.rebalance_frequency = config.rebalance_frequency
        self.enable_rebalancing = config.enable_rebalancing
        self.min_position_size = config.min_position_size
        self.max_position_size = config.max_position_size

        # Performance tracking
        self.rebalance_history: list[dict[str, Any]] = []
        self.weight_history: list[dict[str, Any]] = []

    def calculate_target_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate equal target portfolio weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to equal weights
        """
        try:
            # Equal weight allocation
            n_symbols = len(self.symbols)
            if n_symbols == 0:
                return {}

            equal_weight = 1.0 / n_symbols

            # Create weights dictionary
            target_weights = {symbol: equal_weight for symbol in self.symbols}

            # Apply constraints
            target_weights = self._apply_constraints(target_weights)

            # Normalize weights
            target_weights = self._normalize_weights(target_weights)

            # Store for tracking
            self.weight_history.append(
                {
                    'timestamp': pd.Timestamp.now(),
                    'weights': target_weights.copy(),
                    'method': 'equal_weight',
                }
            )

            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-100:]

            return target_weights

        except Exception as e:
            self.logger.error(f"Error calculating equal weights: {e}")
            # Fallback to empty weights
            return {}

    def process_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process trading signals and generate portfolio actions.

        Args:
            signals: List of trading signals

        Returns:
            List of portfolio actions (orders)
        """
        if not self.portfolio:
            self.logger.warning("Portfolio not initialized, cannot process signals")
            return []

        portfolio_actions = []

        for signal in signals:
            symbol = signal.get('symbol')
            if not symbol or symbol not in self.symbols:
                continue

            signal_type = signal.get('type', 'BUY')
            confidence = signal.get('confidence', 0.5)
            strength = signal.get('strength', 1.0)

            # Filter signals based on confidence
            if confidence < self.config.signal_filters.min_confidence:
                continue

            # Calculate target position size based on equal weight
            target_weight = self.target_weights.get(symbol, 0.0)
            if target_weight <= 0:
                continue

            # Get current portfolio value
            total_value = (
                self.portfolio.total_value if hasattr(self.portfolio, 'total_value') else 100000.0
            )

            # Calculate target position value
            target_position_value = total_value * target_weight

            # Get current position
            current_position_value = self._get_current_position_value(symbol)

            # Calculate trade value
            trade_value = target_position_value - current_position_value

            if abs(trade_value) < 1e-6:  # Skip if no significant trade
                continue

            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                continue

            # Calculate trade quantity
            trade_quantity = trade_value / current_price

            # Apply signal strength and confidence
            trade_quantity *= strength * confidence

            # Apply equal weight-specific adjustments
            trade_quantity *= target_weight  # Scale by target weight

            # Determine side
            side = 'BUY' if trade_quantity > 0 else 'SELL'
            quantity = abs(trade_quantity)

            # Apply position sizing if available
            if self.position_sizing:
                quantity = self.position_sizing.calculate_position_size(
                    account_value=total_value, entry_price=current_price, symbol=symbol, side=side
                )

            # Create order
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'price': current_price,
                'side': side,
                'order_type': 'MARKET',
                'reason': f'signal_{signal_type}',
                'confidence': confidence,
                'strength': strength,
                'target_weight': target_weight,
                'timestamp': signal.get('timestamp'),
            }

            portfolio_actions.append(order)

        return portfolio_actions

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply portfolio constraints to weights.

        Args:
            weights: Dictionary of target weights

        Returns:
            Dictionary of constrained weights
        """
        constrained_weights = {}

        for symbol, weight in weights.items():
            # Get symbol-specific constraints
            constraints = self.config.get_constraint_for_symbol(symbol)

            # Apply constraints
            constrained_weight = max(
                constraints['min_weight'], min(constraints['max_weight'], weight)
            )
            constrained_weights[symbol] = constrained_weight

        return constrained_weights

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Dictionary of weights

        Returns:
            Dictionary of normalized weights
        """
        total_weight = sum(weights.values())

        if total_weight <= 0:
            # Fallback to equal weights
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

        return {symbol: weight / total_weight for symbol, weight in weights.items()}

    def _get_current_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol.

        Args:
            symbol: Symbol to get position value for

        Returns:
            Current position value
        """
        if not self.portfolio:
            return 0.0

        if hasattr(self.portfolio, 'positions') and symbol in self.portfolio.positions:
            position = self.portfolio.positions[symbol]
            if isinstance(position, dict):
                return position.get('market_value', 0.0)
            else:
                return getattr(position, 'quantity', 0) * getattr(position, 'current_price', 0)

        return 0.0

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price
        """
        # This would typically fetch from market data
        # For now, return a placeholder value
        return 100.0

    def _get_strategy_specific_metrics(self) -> dict[str, Any]:
        """Get strategy-specific performance metrics.

        Returns:
            Dictionary of strategy-specific metrics
        """
        metrics = {
            'rebalance_frequency': self.rebalance_frequency,
            'enable_rebalancing': self.enable_rebalancing,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'rebalance_history': self.rebalance_history.copy(),
            'weight_history': self.weight_history.copy(),
        }

        # Calculate weight stability metrics
        if self.weight_history:
            latest_weights = self.weight_history[-1]['weights']
            for symbol, weight in latest_weights.items():
                metrics[f'{symbol}_current_weight'] = weight
                metrics[f'{symbol}_target_weight'] = weight

        # Calculate rebalance efficiency metrics
        if self.rebalance_history:
            metrics['total_rebalances'] = len(self.rebalance_history)
            metrics['avg_rebalance_size'] = self._calculate_avg_rebalance_size()

        return metrics

    def _calculate_avg_rebalance_size(self) -> float:
        """Calculate average rebalance size.

        Returns:
            Average rebalance size
        """
        if not self.rebalance_history:
            return 0.0

        total_size = 0.0
        count = 0

        for rebalance in self.rebalance_history:
            orders = rebalance.get('orders', [])
            for order in orders:
                quantity = order.get('quantity', 0)
                price = order.get('price', 0)
                total_size += quantity * price
                count += 1

        return total_size / max(1, count)

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get strategy summary.

        Returns:
            Dictionary with strategy summary
        """
        summary = {
            'strategy_name': self.name,
            'strategy_type': 'Equal Weight',
            'symbols': self.symbols,
            'rebalance_frequency': self.rebalance_frequency,
            'enable_rebalancing': self.enable_rebalancing,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'rebalance_count': self.rebalance_count,
            'total_trades': self.total_trades,
            'success_rate': self.successful_trades / max(1, self.total_trades),
            'current_weights': self.portfolio_weights.copy(),
            'target_weights': self.target_weights.copy(),
        }

        # Add performance metrics
        metrics = self._get_strategy_specific_metrics()
        summary.update(metrics)

        return summary

    def validate_config(self) -> bool:
        """Validate strategy configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate base configuration
            if not super().validate_config():
                return False

            # Validate symbols
            if not self.symbols:
                self.logger.error("No symbols specified for equal weight strategy")
                return False

            # Validate rebalance frequency
            valid_frequencies = [
                'daily',
                'weekly',
                'monthly',
                'quarterly',
                'semi_annually',
                'annually',
                'continuous',
                'threshold_based',
            ]
            if self.rebalance_frequency not in valid_frequencies:
                self.logger.error(f"Invalid rebalance frequency: {self.rebalance_frequency}")
                return False

            # Validate position sizes
            if self.min_position_size < 0 or self.min_position_size > 1:
                self.logger.error("Min position size must be between 0 and 1")
                return False

            if self.max_position_size < 0 or self.max_position_size > 1:
                self.logger.error("Max position size must be between 0 and 1")
                return False

            if self.min_position_size > self.max_position_size:
                self.logger.error("Min position size cannot be greater than max position size")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()

        # Reset strategy-specific state
        self.rebalance_history.clear()
        self.weight_history.clear()

        self.logger.info(f"Equal weight strategy {self.name} reset")

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"EqualWeightStrategy(name='{self.name}', symbols={self.symbols})"
