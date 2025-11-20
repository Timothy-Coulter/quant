"""Kelly criterion portfolio strategy implementation.

This module implements the Kelly criterion portfolio strategy that optimizes
position sizing based on the Kelly formula to maximize long-term growth.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .base_portfolio_strategy import BasePortfolioStrategy
from .portfolio_strategy_config import PortfolioStrategyConfig


class KellyCriterionStrategy(BasePortfolioStrategy):
    """Kelly criterion portfolio strategy implementation.

    This strategy uses the Kelly formula to optimize position sizing based on
    win probability, average win, and average loss to maximize long-term growth.
    """

    def __init__(self, config: PortfolioStrategyConfig, event_bus: Any) -> None:
        """Initialize the Kelly criterion strategy.

        Args:
            config: Portfolio strategy configuration
            event_bus: Event bus for event-driven communication
        """
        super().__init__(config, event_bus)

        self.logger = logging.getLogger(__name__)
        self.name = config.strategy_name
        self.symbols = config.symbols

        # Strategy-specific parameters
        self.lookback_period = config.optimization_params.lookback_period
        self.risk_free_rate = config.optimization_params.risk_free_rate
        risk_parameters = config.risk_parameters

        def _risk_value(field: str, default: float) -> float:
            source = risk_parameters
            if source is None:
                return default
            if isinstance(source, Mapping):
                value = source.get(field, default)
            else:
                value = getattr(source, field, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        self.kelly_fraction = _risk_value('kelly_fraction', 0.25)
        self.min_win_rate = _risk_value('min_win_rate', 0.01)
        self.enable_rebalancing = config.enable_rebalancing
        self.min_position_size = config.min_position_size
        self.max_position_size = config.max_position_size

        # Kelly calculation parameters
        self.trade_history: list[dict[str, Any]] = []
        self.win_loss_history: dict[str, list[dict[str, float]]] = {
            symbol: [] for symbol in self.symbols
        }
        self.kelly_metrics: dict[str, dict[str, float]] = {symbol: {} for symbol in self.symbols}

        # Performance tracking
        self.kelly_history: list[dict[str, Any]] = []
        self.position_sizing_history: list[dict[str, Any]] = []

    def _setup_event_subscriptions(self) -> None:
        """Kelly strategy currently operates without event subscriptions."""
        # Portfolio actions are triggered directly via the engine; no subscriptions required.
        return

    def calculate_target_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate Kelly criterion target portfolio weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to Kelly criterion weights
        """
        try:
            # Extract price data
            price_data = {}
            for symbol, data in market_data.items():
                if not data.empty and 'close' in data.columns:
                    price_data[symbol] = data['close'].tail(self.lookback_period)

            if not price_data:
                # Fallback to equal weights if no market data available
                return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

            # Calculate Kelly metrics for each symbol
            kelly_weights = {}
            for symbol in self.symbols:
                if symbol in price_data:
                    kelly_weight = self._calculate_kelly_weight(symbol, price_data[symbol])
                    kelly_weights[symbol] = kelly_weight
                else:
                    kelly_weights[symbol] = 0.0

            # Apply constraints
            kelly_weights = self._apply_constraints(kelly_weights)

            # Normalize weights
            kelly_weights = self._normalize_weights(kelly_weights)

            # Store for tracking
            self.kelly_history.append(
                {
                    'timestamp': pd.Timestamp.now(),
                    'kelly_weights': kelly_weights.copy(),
                    'kelly_metrics': self.kelly_metrics.copy(),
                }
            )

            if len(self.kelly_history) > 100:
                self.kelly_history = self.kelly_history[-100:]

            return kelly_weights

        except Exception as e:
            self.logger.error(f"Error calculating Kelly weights: {e}")
            # Fallback to equal weights
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def _calculate_kelly_weight(self, symbol: str, price_data: pd.Series) -> float:
        """Calculate Kelly criterion weight for a symbol.

        Args:
            symbol: Symbol to calculate weight for
            price_data: Price data for the symbol

        Returns:
            Kelly criterion weight
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()

            # Calculate win/loss statistics
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            # Skip if insufficient data
            if len(wins) < 5 or len(losses) < 5:
                return 0.0

            # Calculate win probability
            win_probability = len(wins) / len(returns)

            # Skip if win probability is too low
            if win_probability < self.min_win_rate:
                return 0.0

            # Calculate average win and loss
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            # Skip if average loss is zero
            if avg_loss == 0:
                return 0.0

            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(win_probability, avg_win, avg_loss)

            # Apply Kelly fraction scaling
            kelly_weight = kelly_fraction * self.kelly_fraction

            # Store metrics
            self.kelly_metrics[symbol] = {
                'win_probability': win_probability,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'kelly_fraction': kelly_fraction,
                'kelly_weight': kelly_weight,
            }

            return kelly_weight

        except Exception as e:
            self.logger.error(f"Error calculating Kelly weight for {symbol}: {e}")
            return 0.0

    def _calculate_kelly_fraction(
        self, win_probability: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly fraction.

        Args:
            win_probability: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Kelly fraction
        """
        try:
            # Kelly formula: f* = (bp - q) / b
            # where b = odds ratio, p = win probability, q = loss probability
            b = avg_win / avg_loss  # odds ratio
            p = win_probability  # win probability
            q = 1 - win_probability  # loss probability

            kelly_fraction = (b * p - q) / b

            # Ensure fraction is positive and reasonable
            kelly_fraction = max(0.0, min(kelly_fraction, 1.0))

            return kelly_fraction

        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0

    def process_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: C901
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

            # Calculate target position size based on Kelly weight
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

            # Apply Kelly criterion-specific adjustments
            kelly_metrics = self.kelly_metrics.get(symbol, {})
            if kelly_metrics:
                # Adjust based on Kelly metrics
                kelly_weight = kelly_metrics.get('kelly_weight', 0.0)
                win_probability = kelly_metrics.get('win_probability', 0.0)

                # Apply Kelly scaling
                trade_quantity *= kelly_weight

                # Apply confidence adjustment based on win probability
                confidence_adjustment = min(1.0, win_probability / 0.5)  # Scale by win probability
                trade_quantity *= confidence_adjustment

            # Determine side
            side = 'BUY' if trade_quantity > 0 else 'SELL'
            quantity = abs(trade_quantity)

            # Apply position sizing if available
            if self.position_sizing:
                quantity = self.position_sizing.calculate_position_size(
                    account_value=total_value, entry_price=current_price, symbol=symbol, side=side
                )

            if not self._enforce_risk_controls(
                symbol, side, quantity, current_price, total_value, signal_type
            ):
                continue

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
                'kelly_weight': kelly_metrics.get('kelly_weight', 0.0),
                'win_probability': kelly_metrics.get('win_probability', 0.0),
                'timestamp': signal.get('timestamp'),
            }

            portfolio_actions.append(order)

            # Record trade for Kelly calculation
            self._record_trade_for_kelly(symbol, order)

        return portfolio_actions

    def _record_trade_for_kelly(self, symbol: str, order: dict[str, Any]) -> None:
        """Record trade for Kelly calculation.

        Args:
            symbol: Symbol that was traded
            order: Order that was executed
        """
        try:
            # Calculate trade return (this would be updated when the trade is closed)
            trade_record = {
                'symbol': symbol,
                'quantity': order['quantity'],
                'price': order['price'],
                'side': order['side'],
                'timestamp': order['timestamp'],
                'return': 0.0,  # Will be updated when trade is closed
                'is_closed': False,
            }

            self.trade_history.append(trade_record)

            # Keep only recent trade history
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error recording trade for Kelly calculation: {e}")

    def _enforce_risk_controls(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        signal_type: str,
    ) -> bool:
        """Apply shared risk checks for each prospective order."""
        if not self.risk_manager:
            return True

        metadata = {'strategy': self.name, 'signal': signal_type}
        if side == 'BUY' and not self.risk_manager.can_open_position(
            symbol, quantity * price, portfolio_value, metadata
        ):
            return False

        self.risk_manager.record_order(symbol, side, quantity, price, metadata)
        return True

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
            'lookback_period': self.lookback_period,
            'risk_free_rate': self.risk_free_rate,
            'kelly_fraction': self.kelly_fraction,
            'min_win_rate': self.min_win_rate,
            'max_position_size': self.max_position_size,
            'enable_rebalancing': self.enable_rebalancing,
            'min_position_size': self.min_position_size,
            'kelly_history': self.kelly_history.copy(),
            'position_sizing_history': self.position_sizing_history.copy(),
        }

        # Calculate Kelly metrics for each symbol
        for symbol, kelly_metric in self.kelly_metrics.items():
            if kelly_metric:
                metrics[f'{symbol}_win_probability'] = kelly_metric.get('win_probability', 0.0)
                metrics[f'{symbol}_avg_win'] = kelly_metric.get('avg_win', 0.0)
                metrics[f'{symbol}_avg_loss'] = kelly_metric.get('avg_loss', 0.0)
                metrics[f'{symbol}_kelly_fraction'] = kelly_metric.get('kelly_fraction', 0.0)
                metrics[f'{symbol}_kelly_weight'] = kelly_metric.get('kelly_weight', 0.0)

        # Calculate Kelly efficiency metrics
        if self.kelly_history:
            latest_kelly = self.kelly_history[-1]
            latest_weights = latest_kelly.get('kelly_weights', {})
            latest_metrics = latest_kelly.get('kelly_metrics', {})

            # Calculate Kelly efficiency
            total_kelly_weight = sum(latest_weights.values())
            if total_kelly_weight > 0:
                kelly_efficiency = min(
                    weight / (total_kelly_weight / len(self.symbols))
                    for weight in latest_weights.values()
                )
                metrics['kelly_efficiency'] = kelly_efficiency

            # Calculate average Kelly metrics
            avg_win_probability = np.mean(
                [metric.get('win_probability', 0.0) for metric in latest_metrics.values()]
            )
            avg_kelly_fraction = np.mean(
                [metric.get('kelly_fraction', 0.0) for metric in latest_metrics.values()]
            )

            metrics['avg_win_probability'] = avg_win_probability
            metrics['avg_kelly_fraction'] = avg_kelly_fraction

        return metrics

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get strategy summary.

        Returns:
            Dictionary with strategy summary
        """
        summary = {
            'strategy_name': self.name,
            'strategy_type': 'Kelly Criterion',
            'symbols': self.symbols,
            'lookback_period': self.lookback_period,
            'risk_free_rate': self.risk_free_rate,
            'kelly_fraction': self.kelly_fraction,
            'min_win_rate': self.min_win_rate,
            'max_position_size': self.max_position_size,
            'enable_rebalancing': self.enable_rebalancing,
            'min_position_size': self.min_position_size,
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

            validation_rules = [
                (bool(self.symbols), "No symbols specified for Kelly criterion strategy"),
                (self.lookback_period > 0, "Lookback period must be positive"),
                (self.risk_free_rate >= 0, "Risk free rate must be non-negative"),
                (
                    0 < self.kelly_fraction <= 1,
                    "Kelly fraction must be between 0 and 1",
                ),
                (
                    0 <= self.min_win_rate <= 1,
                    "Min win rate must be between 0 and 1",
                ),
                (
                    0 <= self.min_position_size <= 1,
                    "Min position size must be between 0 and 1",
                ),
                (
                    0 <= self.max_position_size <= 1,
                    "Max position size must be between 0 and 1",
                ),
                (
                    self.min_position_size <= self.max_position_size,
                    "Min position size cannot be greater than max position size",
                ),
            ]

            for condition, message in validation_rules:
                if not condition:
                    self.logger.error(message)
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()

        # Reset strategy-specific state
        self.trade_history.clear()
        for symbol in self.symbols:
            self.win_loss_history[symbol].clear()
            self.kelly_metrics[symbol].clear()

        self.kelly_history.clear()
        self.position_sizing_history.clear()

        self.logger.info(f"Kelly criterion strategy {self.name} reset")

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"KellyCriterionStrategy(name='{self.name}', symbols={self.symbols})"
