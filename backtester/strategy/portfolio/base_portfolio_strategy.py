"""Base portfolio strategy class for implementing portfolio allocation strategies.

This module defines the abstract base class that all portfolio allocation strategies
must inherit from. It provides the common interface and functionality required
for managing portfolio allocation, rebalancing, and position management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from backtester.core.event_bus import EventBus
from backtester.core.interfaces import RiskManagerProtocol
from backtester.portfolio.base_portfolio import BasePortfolio
from backtester.risk_management.position_sizing import PositionSizer

from .portfolio_strategy_config import PortfolioStrategyConfig, RebalanceFrequency


class BasePortfolioStrategy(ABC):
    """Abstract base class for portfolio allocation strategies.

    This class provides the common interface and functionality that all portfolio
    allocation strategies must implement. It follows the modular component
    architecture with proper typing, logging, and validation integration.
    """

    def __init__(self, config: PortfolioStrategyConfig, event_bus: EventBus) -> None:
        """Initialize the portfolio strategy.

        Args:
            config: Portfolio strategy configuration parameters
            event_bus: Event bus for event-driven communication
        """
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Strategy state
        self.name = config.strategy_name
        self.type = config.strategy_type
        self.symbols = config.symbols
        self.is_initialized = False
        self.current_step = 0
        self.rebalance_count = 0
        self.total_trades = 0
        self.successful_trades = 0

        # Portfolio state
        self.portfolio: BasePortfolio | None = None
        self.portfolio_weights: dict[str, float] = {}
        self.target_weights: dict[str, float] = {}
        self.last_rebalance_time: Any | None = None

        # Risk management integration
        self.position_sizing: PositionSizer | None = None
        self.risk_limits: dict[str, Any] | None = None
        self.risk_manager: RiskManagerProtocol | None = None

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.rebalance_history: list[dict[str, Any]] = []

        # Subscribe to relevant events
        self._setup_event_subscriptions()

        self.is_initialized = True
        self.logger.info(f"Initialized portfolio strategy: {self.name} (type: {self.type})")

    @abstractmethod
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for the strategy."""
        # This would typically subscribe to SignalEvent, PortfolioUpdateEvent, etc.
        # For now, we'll leave this as a placeholder for future implementation
        pass

    def initialize_portfolio(self, portfolio: BasePortfolio) -> None:
        """Initialize the portfolio for this strategy.

        Args:
            portfolio: Portfolio instance to manage
        """
        self.portfolio = portfolio

        # Set initial portfolio weights
        initial_weights = self.calculate_target_weights({})
        self.portfolio_weights = initial_weights.copy()

        self.logger.info(f"Initialized portfolio for strategy {self.name}")

    def set_position_sizer(self, position_sizer: PositionSizer) -> None:
        """Set the position sizer for risk management integration.

        Args:
            position_sizer: Position sizing component
        """
        self.position_sizing = position_sizer
        self.logger.info(f"Set position sizer for strategy {self.name}")

    def set_risk_limits(self, risk_limits: dict[str, Any]) -> None:
        """Set risk limits for risk management integration.

        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits
        self.logger.info(f"Set risk limits for strategy {self.name}")

    def set_risk_manager(self, risk_manager: RiskManagerProtocol | None) -> None:
        """Connect a risk manager so strategies can delegate risk checks."""
        self.risk_manager = risk_manager
        if risk_manager is not None:
            self.logger.info("Risk manager attached to strategy %s", self.name)

    def update_portfolio_state(self, market_data: dict[str, pd.DataFrame]) -> None:
        """Update portfolio state based on market data.

        Args:
            market_data: Dictionary mapping symbols to market data
        """
        try:
            # Update current portfolio weights based on market data
            current_weights = self._calculate_current_weights(market_data)
            self.portfolio_weights = current_weights

            # Check if rebalancing is needed
            if self._should_rebalance(market_data):
                self.rebalance_portfolio(market_data)

        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")

    def _calculate_current_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate current portfolio weights based on market data.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to current weights
        """
        if not self.portfolio:
            return {}

        try:
            total_value = (
                self.portfolio.total_value if hasattr(self.portfolio, 'total_value') else 100000.0
            )
            current_weights = {}

            for symbol in self.symbols:
                position_value = self._get_position_value(symbol)
                current_weights[symbol] = position_value / total_value if total_value > 0 else 0.0

            # Normalize weights
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {k: v / total_weight for k, v in current_weights.items()}

            return current_weights

        except Exception as e:
            self.logger.error(f"Error calculating current weights: {e}")
            return {}

    def _get_position_value(self, symbol: str) -> float:
        """Get position value for a symbol.

        Args:
            symbol: Symbol to get position value for

        Returns:
            Position value
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

    def _should_rebalance(self, market_data: dict[str, pd.DataFrame]) -> bool:
        """Check if portfolio should be rebalanced.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            True if rebalancing is needed
        """
        try:
            # Check if rebalancing is enabled
            if not getattr(self.config, 'enable_rebalancing', True):
                return False

            # Check time-based rebalancing
            if getattr(self.config, 'rebalance_frequency', None) == RebalanceFrequency.WEEKLY:
                # Simple time-based logic: rebalance every 7 steps
                return self.current_step % 7 == 0

            # Check threshold-based rebalancing
            threshold_based_rebalance = getattr(self.config, 'threshold_based_rebalance', 0.05)
            if threshold_based_rebalance > 0:
                # Check if any weight has deviated significantly from target
                for symbol, current_weight in self.portfolio_weights.items():
                    target_weight = self.target_weights.get(symbol, 0.0)
                    if abs(current_weight - target_weight) > threshold_based_rebalance:
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking rebalancing condition: {e}")
            return False

    def rebalance_portfolio(self, market_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Rebalance portfolio based on target weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            List of portfolio actions (orders)
        """
        try:
            # Calculate target weights
            self.target_weights = self.calculate_target_weights(market_data)

            # Generate rebalance orders
            rebalance_orders = self.generate_rebalance_orders()

            # Execute rebalance
            if rebalance_orders and self.portfolio:
                self._execute_rebalance(rebalance_orders)

            # Record rebalance
            self._record_rebalance(rebalance_orders)

            self.rebalance_count += 1
            self.last_rebalance_time = self.current_step

            self.logger.info(f"Rebalanced portfolio: {self.rebalance_count} rebalances")

            return rebalance_orders

        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return []

    def generate_rebalance_orders(self) -> list[dict[str, Any]]:
        """Generate rebalance orders based on target weights.

        Returns:
            List of rebalance orders
        """
        if not self.portfolio:
            self.logger.warning("Portfolio not initialized, cannot generate rebalance orders")
            return []

        try:
            rebalance_orders = []
            total_value = (
                self.portfolio.total_value if hasattr(self.portfolio, 'total_value') else 100000.0
            )

            for symbol in self.symbols:
                current_weight = self.portfolio_weights.get(symbol, 0.0)
                target_weight = self.target_weights.get(symbol, 0.0)

                # Skip if no significant change
                if abs(current_weight - target_weight) < 1e-6:
                    continue

                # Calculate trade value
                trade_value = total_value * (target_weight - current_weight)

                # Get current price
                current_price = self._get_current_price(symbol)
                if current_price <= 0:
                    continue

                # Calculate trade quantity
                trade_quantity = trade_value / current_price

                # Determine side
                side = 'BUY' if trade_quantity > 0 else 'SELL'
                quantity = abs(trade_quantity)

                # Apply position sizing if available
                if self.position_sizing:
                    quantity = self.position_sizing.calculate_position_size(
                        account_value=total_value,
                        entry_price=current_price,
                        symbol=symbol,
                        side=side,
                    )

                # Create order
                order = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': current_price,
                    'side': side,
                    'order_type': 'MARKET',
                    'reason': 'rebalance',
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'timestamp': self.current_step,
                }

                rebalance_orders.append(order)

            return rebalance_orders

        except Exception as e:
            self.logger.error(f"Error generating rebalance orders: {e}")
            return []

    def _execute_rebalance(self, orders: list[dict[str, Any]]) -> None:
        """Execute rebalance orders.

        Args:
            orders: List of orders to execute
        """
        if not self.portfolio:
            return

        try:
            for order in orders:
                # This would typically call the portfolio's execute_order method
                # For now, we'll just log the order
                self.logger.debug(f"Executing rebalance order: {order}")

                # Update portfolio weights
                symbol = order['symbol']
                quantity = order['quantity']
                price = order['price']
                side = order['side']

                # Calculate position value change
                position_value_change = quantity * price

                # Update portfolio weights (simplified)
                total_value = (
                    self.portfolio.total_value
                    if hasattr(self.portfolio, 'total_value')
                    else 100000.0
                )
                weight_change = position_value_change / total_value if total_value > 0 else 0.0

                if side == 'BUY':
                    self.portfolio_weights[symbol] = (
                        self.portfolio_weights.get(symbol, 0.0) + weight_change
                    )
                else:
                    self.portfolio_weights[symbol] = (
                        self.portfolio_weights.get(symbol, 0.0) - weight_change
                    )

                # Normalize weights
                total_weight = sum(self.portfolio_weights.values())
                if total_weight > 0:
                    self.portfolio_weights = {
                        k: v / total_weight for k, v in self.portfolio_weights.items()
                    }

                # Record trade
                self._record_trade(order)

        except Exception as e:
            self.logger.error(f"Error executing rebalance: {e}")

    def _record_rebalance(self, orders: list[dict[str, Any]]) -> None:
        """Record rebalance information.

        Args:
            orders: List of orders executed in rebalance
        """
        rebalance_record = {
            'timestamp': self.current_step,
            'rebalance_number': self.rebalance_count,
            'orders': orders.copy(),
            'old_weights': self.portfolio_weights.copy(),
            'new_weights': self.target_weights.copy(),
        }

        self.rebalance_history.append(rebalance_record)

        # Keep only recent rebalance history
        if len(self.rebalance_history) > 100:
            self.rebalance_history = self.rebalance_history[-100:]

    def _record_trade(self, order: dict[str, Any]) -> None:
        """Record trade information.

        Args:
            order: Order that was executed
        """
        trade_record = {
            'timestamp': self.current_step,
            'symbol': order['symbol'],
            'quantity': order['quantity'],
            'price': order['price'],
            'side': order['side'],
            'reason': order.get('reason', 'unknown'),
            'rebalance_number': self.rebalance_count,
        }

        self.trade_history.append(trade_record)
        self.total_trades += 1

        # Keep only recent trade history
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

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

    def _get_current_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol.

        Args:
            symbol: Symbol to get position value for

        Returns:
            Current position value
        """
        return self._get_position_value(symbol)

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Dictionary of weights to normalize

        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights
            n = len(weights)
            return {k: 1.0 / n for k in weights}

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply portfolio constraints to weights.

        Args:
            weights: Dictionary of weights to constrain

        Returns:
            Constrained weights
        """
        constrained_weights = weights.copy()

        for symbol, weight in weights.items():
            # Get constraints for this symbol
            constraint = self.config.get_constraint_for_symbol(symbol)
            if constraint:
                if 'min_weight' in constraint and weight < constraint['min_weight']:
                    constrained_weights[symbol] = constraint['min_weight']
                elif 'max_weight' in constraint and weight > constraint['max_weight']:
                    constrained_weights[symbol] = constraint['max_weight']

        # Normalize after applying constraints
        return self._normalize_weights(constrained_weights)

    @abstractmethod
    def calculate_target_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate target portfolio weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to target weights
        """
        pass

    @abstractmethod
    def process_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process trading signals and generate portfolio actions.

        Args:
            signals: List of trading signals

        Returns:
            List of portfolio actions (orders)
        """
        pass

    def get_constraint_for_symbol(self, symbol: str) -> Any:
        """Get constraints for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Constraint object for the symbol

        Raises:
            NotImplementedError: If not implemented by concrete strategy
        """
        raise NotImplementedError("Concrete strategies must implement get_constraint_for_symbol")

    def should_rebalance(self, market_data: dict[str, pd.DataFrame], current_step: int) -> bool:
        """Determine if portfolio should be rebalanced.

        Args:
            market_data: Current market data
            current_step: Current time step

        Returns:
            True if rebalancing is needed

        Raises:
            NotImplementedError: If not implemented by concrete strategy
        """
        raise NotImplementedError("Concrete strategies must implement should_rebalance")

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get strategy summary.

        Returns:
            Dictionary with strategy summary
        """
        return {
            'strategy_name': self.name,
            'strategy_type': self.type,
            'symbols': self.symbols,
            'rebalance_count': self.rebalance_count,
            'total_trades': self.total_trades,
            'success_rate': self.successful_trades / max(1, self.total_trades),
            'current_weights': self.portfolio_weights.copy(),
            'target_weights': self.target_weights.copy(),
            'last_rebalance_time': self.last_rebalance_time,
            'performance_metrics': self.performance_metrics.copy(),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics.copy()

    def validate_config(self) -> bool:
        """Validate strategy configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate basic configuration
            if not self.config:
                self.logger.error("No configuration provided")
                return False

            if not self.config.strategy_name:
                self.logger.error("Strategy name is required")
                return False

            if not self.config.symbols:
                self.logger.error("No symbols specified")
                return False

            # Validate symbols
            if not isinstance(self.config.symbols, list) or len(self.config.symbols) == 0:
                self.logger.error("Symbols must be a non-empty list")
                return False

            # Validate constraints
            if self.config.constraints is None:
                self.logger.error("Constraints cannot be None")
                return False
            if not hasattr(self.config.constraints, '__dict__') and not isinstance(
                self.config.constraints, dict
            ):
                self.logger.error("Constraints must be a dictionary or object with attributes")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def reset(self) -> None:
        """Reset strategy state."""
        self.rebalance_count = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.portfolio_weights.clear()
        self.target_weights.clear()
        self.last_rebalance_time = None
        self.performance_metrics.clear()
        self.trade_history.clear()
        self.rebalance_history.clear()

        self.logger.info(f"Portfolio strategy {self.name} reset")

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return (
            f"BasePortfolioStrategy(name='{self.name}', type='{self.type}', symbols={self.symbols})"
        )
