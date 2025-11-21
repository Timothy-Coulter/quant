"""Modern Portfolio Theory (MPT) strategy implementation.

This module implements the Modern Portfolio Theory strategy that optimizes
portfolio allocation based on mean-variance optimization to maximize
Sharpe ratio or minimize portfolio risk.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from .base_portfolio_strategy import BasePortfolioStrategy
from .portfolio_strategy_config import PortfolioStrategyConfig, SignalFilterConfig


class ModernPortfolioTheoryStrategy(BasePortfolioStrategy):
    """Modern Portfolio Theory (MPT) strategy implementation.

    This strategy uses mean-variance optimization to find the optimal portfolio
    allocation that maximizes the Sharpe ratio or minimizes portfolio risk.
    """

    def __init__(self, config: PortfolioStrategyConfig, event_bus: Any) -> None:
        """Initialize the MPT strategy.

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
        self.target_return = config.optimization_params.target_return
        self.target_risk = config.optimization_params.target_risk
        self.risk_aversion = config.optimization_params.risk_aversion
        self.optimization_method = config.optimization_params.optimization_method
        self.max_iterations = config.optimization_params.max_iterations
        self.convergence_tolerance = config.optimization_params.convergence_tolerance

        # MPT calculation parameters
        self.expected_returns: dict[str, list[float]] = {symbol: [] for symbol in self.symbols}
        self.covariance_matrices: list[npt.NDArray[np.float_]] = []
        self.portfolio_returns: list[float] = []
        self.portfolio_risks: list[float] = []
        self.sharpe_ratios: list[float] = []

        # Performance tracking
        self.optimization_history: list[dict[str, Any]] = []
        self.frontier_points: list[tuple[float, float]] = []  # (risk, return)

    def calculate_target_weights(self, market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calculate MPT optimal portfolio weights.

        Args:
            market_data: Dictionary mapping symbols to market data

        Returns:
            Dictionary mapping symbols to optimal weights
        """
        try:
            price_data = self._extract_price_data(market_data)
            if not price_data:
                return self._equal_weight_allocation()

            expected_returns, cov_matrix = self._calculate_mpt_parameters(price_data)
            self._update_tracking_metrics(expected_returns, cov_matrix)

            optimal_weights = self._compute_optimal_weights(expected_returns, cov_matrix)
            constrained_weights = self._apply_constraints(optimal_weights)
            normalized_weights = self._normalize_weights(constrained_weights)

            portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(
                normalized_weights, expected_returns, cov_matrix
            )
            sharpe_ratio = self._compute_sharpe_ratio(portfolio_return, portfolio_risk)
            self._record_optimization_snapshot(
                normalized_weights, expected_returns, portfolio_return, portfolio_risk, sharpe_ratio
            )

            return normalized_weights

        except Exception as e:
            self.logger.error(f"Error calculating MPT weights: {e}")
            return self._equal_weight_allocation()

    def _extract_price_data(self, market_data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """Extract closing price series from raw market data."""
        price_data: dict[str, pd.Series] = {}
        for symbol, data in market_data.items():
            if not data.empty and 'close' in data.columns:
                price_data[symbol] = data['close'].tail(self.lookback_period)
        return price_data

    def _update_tracking_metrics(
        self, expected_returns: dict[str, float], cov_matrix: npt.NDArray[np.float_]
    ) -> None:
        """Track rolling statistics for diagnostics."""
        for symbol, ret in expected_returns.items():
            self.expected_returns[symbol].append(ret)
            if len(self.expected_returns[symbol]) > 100:
                self.expected_returns[symbol] = self.expected_returns[symbol][-100:]

        self.covariance_matrices.append(cov_matrix)
        if len(self.covariance_matrices) > 50:
            self.covariance_matrices = self.covariance_matrices[-50:]

    def _compute_optimal_weights(
        self, expected_returns: dict[str, float], cov_matrix: npt.NDArray[np.float_]
    ) -> dict[str, float]:
        """Select and execute the configured optimizer."""
        optimizer_map: dict[str, Callable[[], dict[str, float]]] = {
            'mean_variance': lambda: self._optimize_mean_variance(expected_returns, cov_matrix),
            'maximum_sharpe': lambda: self._optimize_maximum_sharpe(expected_returns, cov_matrix),
            'minimum_variance': lambda: self._optimize_minimum_variance(cov_matrix),
            'target_return': lambda: self._optimize_target_return(
                expected_returns, cov_matrix, self.target_return
            ),
            'target_risk': lambda: self._optimize_target_risk(
                expected_returns, cov_matrix, self.target_risk
            ),
        }
        optimizer = optimizer_map.get(
            self.optimization_method,
            lambda: self._optimize_mean_variance(expected_returns, cov_matrix),
        )
        return optimizer()

    def _compute_sharpe_ratio(self, portfolio_return: float, portfolio_risk: float) -> float:
        """Compute Sharpe ratio while guarding against division by zero."""
        if portfolio_risk <= 0:
            return 0.0
        return (portfolio_return - self.risk_free_rate) / portfolio_risk

    def _record_optimization_snapshot(
        self,
        weights: dict[str, float],
        expected_returns: dict[str, float],
        portfolio_return: float,
        portfolio_risk: float,
        sharpe_ratio: float,
    ) -> None:
        """Persist the latest optimization artefacts for reporting."""
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_risks.append(portfolio_risk)
        self.sharpe_ratios.append(sharpe_ratio)

        if len(self.portfolio_returns) > 100:
            self.portfolio_returns = self.portfolio_returns[-100:]
            self.portfolio_risks = self.portfolio_risks[-100:]
            self.sharpe_ratios = self.sharpe_ratios[-100:]

        self.optimization_history.append(
            {
                'timestamp': pd.Timestamp.now(),
                'method': self.optimization_method,
                'optimal_weights': weights.copy(),
                'expected_returns': expected_returns.copy(),
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
            }
        )

        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]

    def _equal_weight_allocation(self) -> dict[str, float]:
        """Return equal weights across configured symbols."""
        if not self.symbols:
            return {}
        weight = 1.0 / len(self.symbols)
        return {symbol: weight for symbol in self.symbols}

    def _calculate_mpt_parameters(
        self, price_data: dict[str, pd.DataFrame]
    ) -> tuple[dict[str, float], npt.NDArray[np.float_]]:
        """Calculate expected returns and covariance matrix for MPT.

        Args:
            price_data: Dictionary mapping symbols to price data

        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        try:
            # Calculate returns
            returns_data = pd.DataFrame()
            for symbol, prices in price_data.items():
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns

            # Calculate expected returns (annualized)
            expected_returns = {}
            for symbol in returns_data.columns:
                mean_return = returns_data[symbol].mean()
                annualized_return = (1 + mean_return) ** 252 - 1  # Annualize
                expected_returns[symbol] = annualized_return

            # Calculate covariance matrix (annualized)
            cov_matrix = returns_data.cov().values * 252  # Annualize

            return expected_returns, cov_matrix

        except Exception as e:
            self.logger.error(f"Error calculating MPT parameters: {e}")
            # Fallback to simple estimates
            n_symbols = len(price_data)
            expected_returns = {symbol: 0.1 for symbol in price_data}
            cov_matrix = np.eye(n_symbols) * 0.1  # Identity matrix with 10% variance
            return expected_returns, cov_matrix

    def _optimize_mean_variance(
        self, expected_returns: dict[str, float], cov_matrix: npt.NDArray[np.float_]
    ) -> dict[str, float]:
        """Optimize portfolio using mean-variance optimization.

        Args:
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of optimal weights
        """
        try:
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)

            # Objective function: minimize portfolio variance
            def objective(weights: npt.NDArray[np.float_]) -> float:
                return float(weights.T @ cov_matrix @ weights)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            ]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance},
            )

            if result.success:
                optimal_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
                return optimal_weights
            else:
                self.logger.warning(f"Mean-variance optimization failed: {result.message}")
                return {symbol: 1.0 / n_assets for symbol in symbols}

        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {e}")
            return {symbol: 1.0 / len(expected_returns) for symbol in expected_returns}

    def _optimize_maximum_sharpe(
        self, expected_returns: dict[str, float], cov_matrix: npt.NDArray[np.float_]
    ) -> dict[str, float]:
        """Optimize portfolio to maximize Sharpe ratio.

        Args:
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of optimal weights
        """
        try:
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            mu = np.array([expected_returns[symbol] for symbol in symbols])

            # Objective function: minimize negative Sharpe ratio
            def objective(weights: npt.NDArray[np.float_]) -> float:
                portfolio_return = float(weights.T @ mu)
                portfolio_risk = float(np.sqrt(weights.T @ cov_matrix @ weights))
                return (
                    -(portfolio_return - self.risk_free_rate) / portfolio_risk
                    if portfolio_risk > 0
                    else -portfolio_return
                )

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            ]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance},
            )

            if result.success:
                optimal_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
                return optimal_weights
            else:
                self.logger.warning(f"Maximum Sharpe optimization failed: {result.message}")
                return {symbol: 1.0 / n_assets for symbol in symbols}

        except Exception as e:
            self.logger.error(f"Error in maximum Sharpe optimization: {e}")
            return {symbol: 1.0 / len(expected_returns) for symbol in expected_returns}

    def _optimize_minimum_variance(self, cov_matrix: npt.NDArray[np.float_]) -> dict[str, float]:
        """Optimize portfolio to minimize variance.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of optimal weights
        """
        try:
            n_assets = cov_matrix.shape[0]

            # Objective function: minimize portfolio variance
            def objective(weights: npt.NDArray[np.float_]) -> float:
                return float(weights.T @ cov_matrix @ weights)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            ]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance},
            )

            if result.success:
                # Map back to symbols (simplified)
                symbols = self.symbols[:n_assets]
                return {symbols[i]: result.x[i] for i in range(n_assets)}
            else:
                self.logger.warning(f"Minimum variance optimization failed: {result.message}")
                return {symbol: 1.0 / n_assets for symbol in self.symbols[:n_assets]}

        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {e}")
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def _optimize_target_return(
        self,
        expected_returns: dict[str, float],
        cov_matrix: npt.NDArray[np.float_],
        target_return: float | None,
    ) -> dict[str, float]:
        """Optimize portfolio to achieve target return with minimum risk.

        Args:
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix
            target_return: Target portfolio return

        Returns:
            Dictionary of optimal weights
        """
        try:
            desired_return = target_return if target_return is not None else self.target_return
            if desired_return is None:
                desired_return = float(np.mean(list(expected_returns.values())))

            symbols = list(expected_returns.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            mu = np.array([expected_returns[symbol] for symbol in symbols])

            # Objective function: minimize portfolio variance
            def objective(weights: npt.NDArray[np.float_]) -> float:
                return float(weights.T @ cov_matrix @ weights)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {
                    'type': 'eq',
                    'fun': lambda w, target=desired_return: w.T @ mu - target,
                },
                {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            ]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance},
            )

            if result.success:
                optimal_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
                return optimal_weights
            else:
                self.logger.warning(f"Target return optimization failed: {result.message}")
                return {symbol: 1.0 / n_assets for symbol in symbols}

        except Exception as e:
            self.logger.error(f"Error in target return optimization: {e}")
            return {symbol: 1.0 / len(expected_returns) for symbol in expected_returns}

    def _optimize_target_risk(
        self,
        expected_returns: dict[str, float],
        cov_matrix: npt.NDArray[np.float_],
        target_risk: float | None,
    ) -> dict[str, float]:
        """Optimize portfolio to achieve target risk with maximum return.

        Args:
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix
            target_risk: Target portfolio risk

        Returns:
            Dictionary of optimal weights
        """
        try:
            desired_risk = target_risk if target_risk is not None else self.target_risk
            if desired_risk is None:
                desired_risk = float(np.std(list(expected_returns.values())))

            symbols = list(expected_returns.keys())
            n_assets = len(symbols)

            # Convert to numpy arrays
            mu = np.array([expected_returns[symbol] for symbol in symbols])

            # Objective function: maximize portfolio return
            def objective(weights: npt.NDArray[np.float_]) -> float:
                return float(-weights.T @ mu)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {
                    'type': 'eq',
                    'fun': (lambda w, target=desired_risk: np.sqrt(w.T @ cov_matrix @ w) - target),
                },
                {'type': 'ineq', 'fun': lambda w: w},  # Weights >= 0
            ]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance},
            )

            if result.success:
                optimal_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
                return optimal_weights
            else:
                self.logger.warning(f"Target risk optimization failed: {result.message}")
                return {symbol: 1.0 / n_assets for symbol in symbols}

        except Exception as e:
            self.logger.error(f"Error in target risk optimization: {e}")
            return {symbol: 1.0 / len(expected_returns) for symbol in expected_returns}

    def _calculate_portfolio_metrics(
        self,
        weights: dict[str, float],
        expected_returns: dict[str, float],
        cov_matrix: npt.NDArray[np.float_],
    ) -> tuple[float, float]:
        """Calculate portfolio return and risk.

        Args:
            weights: Dictionary of portfolio weights
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix

        Returns:
            Tuple of (portfolio_return, portfolio_risk)
        """
        try:
            symbols = list(expected_returns.keys())

            # Convert to numpy arrays
            w = np.array([weights.get(symbol, 0.0) for symbol in symbols])
            mu = np.array([expected_returns[symbol] for symbol in symbols])

            # Calculate portfolio return
            portfolio_return = w.T @ mu

            # Calculate portfolio risk
            portfolio_risk = np.sqrt(w.T @ cov_matrix @ w)

            return portfolio_return, portfolio_risk

        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return 0.0, 0.0

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

            # Calculate target position size based on MPT weight
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

            # Apply MPT-specific adjustments
            expected_return = (
                self.expected_returns.get(symbol, [0.0])[-1]
                if self.expected_returns.get(symbol)
                else 0.0
            )
            trade_quantity *= (
                1.0 + expected_return * 0.1
            )  # Small adjustment based on expected return

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
                'expected_return': expected_return,
                'timestamp': signal.get('timestamp'),
            }

            portfolio_actions.append(order)

        return portfolio_actions

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

    def _get_strategy_specific_metrics(self) -> dict[str, Any]:
        """Get strategy-specific performance metrics.

        Returns:
            Dictionary of strategy-specific metrics
        """
        metrics = {
            'optimization_method': self.optimization_method,
            'risk_aversion': self.risk_aversion,
            'target_return': self.target_return,
            'target_risk': self.target_risk,
            'lookback_period': self.lookback_period,
            'optimization_history': self.optimization_history.copy(),
            'frontier_points': self.frontier_points.copy(),
        }

        # Calculate portfolio performance metrics
        if self.portfolio_returns and self.portfolio_risks:
            metrics['avg_portfolio_return'] = np.mean(self.portfolio_returns)
            metrics['avg_portfolio_risk'] = np.mean(self.portfolio_risks)
            metrics['avg_sharpe_ratio'] = np.mean(self.sharpe_ratios)
            metrics['max_portfolio_return'] = np.max(self.portfolio_returns)
            metrics['min_portfolio_risk'] = np.min(self.portfolio_risks)
            metrics['max_sharpe_ratio'] = np.max(self.sharpe_ratios)

        # Calculate expected return metrics
        if self.expected_returns:
            for symbol, returns in self.expected_returns.items():
                if returns:
                    metrics[f'{symbol}_expected_return'] = np.mean(returns)
                    metrics[f'{symbol}_expected_return_volatility'] = np.std(returns)

        # Calculate optimization efficiency metrics
        if self.optimization_history:
            latest_optimization = self.optimization_history[-1]
            metrics['latest_sharpe_ratio'] = latest_optimization.get('sharpe_ratio', 0.0)
            metrics['latest_portfolio_return'] = latest_optimization.get('portfolio_return', 0.0)
            metrics['latest_portfolio_risk'] = latest_optimization.get('portfolio_risk', 0.0)

        return metrics

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get strategy summary.

        Returns:
            Dictionary with strategy summary
        """
        summary = {
            'strategy_name': self.name,
            'strategy_type': 'Modern Portfolio Theory',
            'symbols': self.symbols,
            'optimization_method': self.optimization_method,
            'risk_aversion': self.risk_aversion,
            'target_return': self.target_return,
            'target_risk': self.target_risk,
            'lookback_period': self.lookback_period,
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

    def calculate_efficient_frontier(
        self,
        expected_returns: dict[str, float],
        cov_matrix: npt.NDArray[np.float_],
        num_points: int = 20,
    ) -> list[tuple[float, float]]:
        """Calculate efficient frontier points.

        Args:
            expected_returns: Dictionary of expected returns
            cov_matrix: Covariance matrix
            num_points: Number of points to calculate

        Returns:
            List of (risk, return) tuples representing the efficient frontier
        """
        try:
            symbols = list(expected_returns.keys())

            # Calculate minimum variance portfolio
            min_var_weights = self._optimize_minimum_variance(cov_matrix)
            min_var_return, _ = self._calculate_portfolio_metrics(
                min_var_weights, expected_returns, cov_matrix
            )

            # Calculate maximum return portfolio
            max_symbol = max(expected_returns, key=lambda sym: expected_returns[sym])
            max_return_weights = {
                symbol: 1.0 if symbol == max_symbol else 0.0 for symbol in symbols
            }
            max_return, _ = self._calculate_portfolio_metrics(
                max_return_weights, expected_returns, cov_matrix
            )

            # Generate frontier points
            frontier_points = []
            for i in range(num_points):
                target_return = min_var_return + (max_return - min_var_return) * i / (
                    num_points - 1
                )
                weights = self._optimize_target_return(expected_returns, cov_matrix, target_return)
                return_val, risk_val = self._calculate_portfolio_metrics(
                    weights, expected_returns, cov_matrix
                )
                frontier_points.append((risk_val, return_val))

            self.frontier_points = frontier_points
            return frontier_points

        except Exception as e:
            self.logger.error(f"Error calculating efficient frontier: {e}")
            return []

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
                self.logger.error("No symbols specified for MPT strategy")
                return False

            # Validate optimization method
            valid_methods = [
                'mean_variance',
                'maximum_sharpe',
                'minimum_variance',
                'target_return',
                'target_risk',
            ]
            if self.optimization_method not in valid_methods:
                self.logger.error(f"Invalid optimization method: {self.optimization_method}")
                return False

            # Validate risk aversion
            if self.risk_aversion <= 0:
                self.logger.error("Risk aversion must be positive")
                return False

            # Validate lookback period
            if self.lookback_period <= 0:
                self.logger.error("Lookback period must be positive")
                return False

            # Validate target return and risk
            if self.target_return is not None and self.target_return < 0:
                self.logger.error("Target return must be non-negative")
                return False

            if self.target_risk is not None and self.target_risk <= 0:
                self.logger.error("Target risk must be positive")
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
            self.expected_returns[symbol].clear()

        self.covariance_matrices.clear()
        self.portfolio_returns.clear()
        self.portfolio_risks.clear()
        self.sharpe_ratios.clear()
        self.optimization_history.clear()
        self.frontier_points.clear()

        self.logger.info(f"Modern Portfolio Theory strategy {self.name} reset")

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return f"ModernPortfolioTheoryStrategy(name='{self.name}', symbols={self.symbols})"
