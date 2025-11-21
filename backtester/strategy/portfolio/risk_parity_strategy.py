"""Risk parity portfolio strategy implementation.

This module implements the risk parity portfolio strategy that allocates
capital based on risk contribution rather than equal weights.
"""

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base_portfolio_strategy import BasePortfolioStrategy
from .portfolio_strategy_config import PortfolioStrategyConfig, RiskBudget, SignalFilterConfig


class RiskParityStrategy(BasePortfolioStrategy):
    """Risk parity portfolio strategy implementation.

    This strategy allocates capital based on risk contribution rather than
    equal weights. Each asset contributes equally to the overall portfolio risk.
    """

    def __init__(self, config: PortfolioStrategyConfig, event_bus: Any) -> None:
        """Initialize the risk parity strategy.

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
        self.target_risk_contribution = 1.0 / len(self.symbols) if self.symbols else 0.0
        budget = cast(RiskBudget, config.risk_budget)
        self.risk_budget_method = budget.risk_budget_method
        self.enable_rebalancing = config.enable_rebalancing
        self.min_position_size = config.min_position_size
        self.max_position_size = config.max_position_size

        # Risk calculation parameters
        self.volatility_history: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        self.covariance_history: list[npt.NDArray[np.float_]] = []
        self.risk_contribution_history: list[dict[str, float]] = []

        # Performance tracking
        self.risk_metrics: dict[str, Any] = {}
        self.optimization_history: list[dict[str, Any]] = []

    def calculate_target_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate risk parity target portfolio weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to risk parity weights
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

            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(price_data)

            # Calculate target weights based on risk contributions
            target_weights = self._calculate_risk_parity_weights(risk_contributions)

            # Apply constraints
            target_weights = self._apply_constraints(target_weights)

            # Normalize weights
            target_weights = self._normalize_weights(target_weights)

            # Store for tracking
            self.risk_contribution_history.append(risk_contributions.copy())
            if len(self.risk_contribution_history) > 100:
                self.risk_contribution_history = self.risk_contribution_history[-100:]

            # Store optimization history
            self.optimization_history.append(
                {
                    'timestamp': pd.Timestamp.now(),
                    'method': 'risk_parity',
                    'risk_contributions': risk_contributions.copy(),
                    'target_weights': target_weights.copy(),
                    'symbols': self.symbols.copy(),
                }
            )

            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]

            return target_weights

        except Exception as e:
            self.logger.error(f"Error calculating risk parity weights: {e}")
            # Fallback to equal weights
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def _calculate_risk_contributions(
        self, price_data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """Calculate risk contributions for each symbol.

        Args:
            price_data: Dictionary mapping symbols to price data

        Returns:
            Dictionary mapping symbols to risk contributions
        """
        try:
            # Calculate returns
            returns_data = pd.DataFrame()
            for symbol, prices in price_data.items():
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns

            # Calculate covariance matrix
            cov_matrix = returns_data.cov().values

            # Store covariance matrix history
            self.covariance_history.append(cov_matrix)
            if len(self.covariance_history) > 50:
                self.covariance_history = self.covariance_history[-50:]

            # Calculate volatilities
            volatilities = {}
            for symbol in returns_data.columns:
                volatility = returns_data[symbol].std() * np.sqrt(252)  # Annualized
                volatilities[symbol] = volatility

                # Store volatility history
                self.volatility_history[symbol].append(volatility)
                if len(self.volatility_history[symbol]) > 100:
                    self.volatility_history[symbol] = self.volatility_history[symbol][-100:]

            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contribution_weights(volatilities, cov_matrix)

            return risk_contributions

        except Exception as e:
            self.logger.error(f"Error calculating risk contributions: {e}")
            # Fallback to equal risk contributions
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def _calculate_risk_contribution_weights(
        self, volatilities: dict[str, float], cov_matrix: npt.NDArray[np.float_]
    ) -> dict[str, float]:
        """Calculate risk contribution weights.

        Args:
            volatilities: Dictionary of volatilities for each symbol
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of risk contribution weights
        """
        try:
            symbols = list(volatilities.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            # Calculate portfolio weights (initial guess)
            initial_weights = np.ones(n_assets) / n_assets

            # Calculate marginal risk contributions
            marginal_risk_contributions = cov_matrix @ initial_weights

            # Calculate risk contributions
            risk_contributions = initial_weights * marginal_risk_contributions

            # Normalize to get risk contribution weights
            total_risk_contribution = np.sum(risk_contributions)
            if total_risk_contribution > 0:
                risk_contribution_weights = risk_contributions / total_risk_contribution
            else:
                risk_contribution_weights = np.ones(n_assets) / n_assets

            # Convert to dictionary
            result = {symbols[i]: risk_contribution_weights[i] for i in range(n_assets)}

            return result

        except Exception as e:
            self.logger.error(f"Error calculating risk contribution weights: {e}")
            # Fallback to equal weights
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def _calculate_risk_parity_weights(
        self, risk_contributions: dict[str, float]
    ) -> dict[str, float]:
        """Calculate risk parity weights based on risk contributions.

        Args:
            risk_contributions: Dictionary of risk contributions

        Returns:
            Dictionary of risk parity weights
        """
        try:
            # Get risk budget
            risk_budget = self.config.get_risk_budget()

            # Calculate risk parity weights
            risk_parity_weights = {}
            total_risk_contribution = sum(risk_contributions.values())

            if total_risk_contribution > 0:
                for symbol, contribution in risk_contributions.items():
                    # Calculate weight based on risk budget
                    if self.risk_budget_method == "equal":
                        # Equal risk budget
                        target_contribution = self.target_risk_contribution
                    else:
                        # Custom risk budget
                        target_contribution = risk_budget.get(symbol, self.target_risk_contribution)

                    # Calculate weight to achieve target contribution
                    weight = target_contribution / contribution if contribution > 0 else 0.0
                    risk_parity_weights[symbol] = weight
            else:
                # Fallback to equal weights
                risk_parity_weights = {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

            return risk_parity_weights

        except Exception as e:
            self.logger.error(f"Error calculating risk parity weights: {e}")
            # Fallback to equal weights
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

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

        filters = cast(SignalFilterConfig, self.config.signal_filters)
        for signal in signals:
            symbol = signal.get('symbol')
            if not symbol or symbol not in self.symbols:
                continue

            signal_type = signal.get('type', 'BUY')
            confidence = signal.get('confidence', 0.5)
            strength = signal.get('strength', 1.0)

            # Filter signals based on confidence
            if confidence < filters.min_confidence:
                continue

            # Calculate target position size based on risk parity weight
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

            # Apply risk parity-specific adjustments
            volatility = self._get_current_volatility(symbol)
            if volatility > 0:
                # Adjust for volatility (higher volatility = smaller position)
                volatility_adjustment = min(1.0, 1.0 / (1.0 + volatility))
                trade_quantity *= volatility_adjustment

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
                'risk_contribution': self.target_weights.get(symbol, 0.0),
                'volatility': volatility,
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
                value = position.get('market_value', 0.0)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
            quantity = float(getattr(position, 'quantity', 0.0) or 0.0)
            price = float(getattr(position, 'current_price', 0.0) or 0.0)
            return quantity * price

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

    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol.

        Args:
            symbol: Symbol to get volatility for

        Returns:
            Current volatility
        """
        if symbol in self.volatility_history and self.volatility_history[symbol]:
            return self.volatility_history[symbol][-1]
        return 0.0

    def _get_strategy_specific_metrics(self) -> dict[str, Any]:
        """Get strategy-specific performance metrics.

        Returns:
            Dictionary of strategy-specific metrics
        """
        metrics = {
            'lookback_period': self.lookback_period,
            'risk_free_rate': self.risk_free_rate,
            'target_risk_contribution': self.target_risk_contribution,
            'risk_budget_method': self.risk_budget_method,
            'enable_rebalancing': self.enable_rebalancing,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'optimization_history': self.optimization_history.copy(),
            'risk_contribution_history': self.risk_contribution_history.copy(),
        }

        # Calculate risk metrics
        if self.risk_contribution_history:
            latest_contributions = self.risk_contribution_history[-1]
            for symbol, contribution in latest_contributions.items():
                metrics[f'{symbol}_risk_contribution'] = contribution

            # Calculate risk parity efficiency
            total_risk = sum(latest_contributions.values())
            if total_risk > 0:
                risk_parity_efficiency = min(
                    contribution / (total_risk / len(self.symbols))
                    for contribution in latest_contributions.values()
                )
                metrics['risk_parity_efficiency'] = risk_parity_efficiency

        # Calculate volatility metrics
        if self.volatility_history:
            for symbol, vol_history in self.volatility_history.items():
                if vol_history:
                    metrics[f'{symbol}_current_volatility'] = vol_history[-1]
                    metrics[f'{symbol}_avg_volatility'] = np.mean(vol_history)
                    metrics[f'{symbol}_volatility_std'] = np.std(vol_history)

        # Calculate optimization metrics
        if self.optimization_history:
            latest_optimization = self.optimization_history[-1]
            metrics['latest_method'] = latest_optimization.get('method', 'unknown')
            metrics['latest_risk_contributions'] = latest_optimization.get('risk_contributions', {})
            metrics['latest_target_weights'] = latest_optimization.get('target_weights', {})

        return metrics

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get strategy summary.

        Returns:
            Dictionary with strategy summary
        """
        summary = {
            'strategy_name': self.name,
            'strategy_type': 'Risk Parity',
            'symbols': self.symbols,
            'lookback_period': self.lookback_period,
            'risk_free_rate': self.risk_free_rate,
            'target_risk_contribution': self.target_risk_contribution,
            'risk_budget_method': self.risk_budget_method,
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

            checks = [
                (bool(self.symbols), "No symbols specified for risk parity strategy"),
                (self.lookback_period > 0, "Lookback period must be positive"),
                (self.risk_free_rate >= 0, "Risk free rate must be non-negative"),
                (
                    self.target_risk_contribution > 0,
                    "Target risk contribution must be positive",
                ),
                (
                    self.risk_budget_method
                    in {'equal', 'custom', 'inverse_volatility', 'market_cap'},
                    f"Invalid risk budget method: {self.risk_budget_method}",
                ),
                (0 <= self.min_position_size <= 1, "Min position size must be between 0 and 1"),
                (0 <= self.max_position_size <= 1, "Max position size must be between 0 and 1"),
                (
                    self.min_position_size <= self.max_position_size,
                    "Min position size cannot be greater than max position size",
                ),
            ]

            for condition, message in checks:
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
        for symbol in self.symbols:
            self.volatility_history[symbol].clear()

        self.covariance_history.clear()
        self.risk_contribution_history.clear()
        self.optimization_history.clear()
        self.risk_metrics.clear()

        self.logger.info(f"Risk parity strategy {self.name} reset")

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"RiskParityStrategy(name='{self.name}', symbols={self.symbols})"
