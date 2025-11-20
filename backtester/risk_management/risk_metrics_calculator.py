"""Risk Metrics Calculator.

This module provides comprehensive risk metrics calculation functionality including
VaR, Expected Shortfall, drawdown, and other risk measures.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd


class RiskMetricsCalculator:
    """Risk metrics calculation component."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize risk metrics calculator.

        Args:
            logger: Optional logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def calculate_portfolio_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        lookback_period: int = 252,
    ) -> float:
        """Calculate portfolio Value at Risk (VaR) using historical simulation.

        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            lookback_period: Number of periods to look back (default: 252)

        Returns:
            VaR value as a negative number indicating potential loss
        """
        if len(returns) < lookback_period:
            recent_returns = returns.tail(len(returns)) if len(returns) > 0 else returns
        else:
            recent_returns = returns.tail(lookback_period)

        if len(recent_returns) == 0:
            return 0.0

        # Historical simulation VaR (negative value indicating loss)
        var = np.percentile(recent_returns, (1 - confidence_level) * 100)
        return float(var)

    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        lookback_period: int = 252,
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR) - average loss beyond VaR.

        Args:
            returns: Portfolio returns series
            confidence_level: Confidence level for calculation (default: 0.95)
            lookback_period: Number of periods to look back (default: 252)

        Returns:
            Expected Shortfall value
        """
        var = self.calculate_portfolio_var(returns, confidence_level, lookback_period)

        if len(returns) == 0:
            return 0.0

        recent_returns = (
            returns.tail(lookback_period) if len(returns) > lookback_period else returns
        )

        # Expected Shortfall: average of returns below VaR
        tail_returns = recent_returns[recent_returns <= var]
        if len(tail_returns) == 0:
            return var

        expected_shortfall = tail_returns.mean()
        return float(expected_shortfall)

    def calculate_max_drawdown(self, returns: pd.Series) -> dict[str, float]:
        """Calculate maximum drawdown and related metrics.

        Args:
            returns: Portfolio returns series

        Returns:
            Dictionary with drawdown metrics
        """
        if len(returns) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0,
                'current_drawdown_duration': 0,
            }

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative_returns.cummax()

        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Find max drawdown duration
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            max_dd_duration = 0
        else:
            # Find longest consecutive period in drawdown
            drawdown_periods = []
            current_period = 0
            for in_dd in in_drawdown:
                if in_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0

        # Current drawdown
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0

        # Current drawdown duration
        current_dd_duration = 0
        if current_drawdown < 0:
            for i in range(len(in_drawdown) - 1, -1, -1):
                if in_drawdown.iloc[i]:
                    current_dd_duration += 1
                else:
                    break

        return {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': int(max_dd_duration),
            'current_drawdown': float(current_drawdown),
            'current_drawdown_duration': int(current_dd_duration),
        }

    def calculate_volatility_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate volatility-related metrics.

        Args:
            returns: Portfolio returns series

        Returns:
            Dictionary with volatility metrics
        """
        if len(returns) == 0:
            return {
                'volatility': 0.0,
                'annualized_volatility': 0.0,
                'rolling_volatility': 0.0,
            }

        # Daily volatility
        volatility = returns.std()

        # Annualized volatility (assuming 252 trading days)
        annualized_volatility = volatility * np.sqrt(252)

        # Rolling volatility (last 30 days)
        if len(returns) >= 30:
            rolling_volatility = returns.tail(30).std() * np.sqrt(252)
        else:
            rolling_volatility = annualized_volatility

        return {
            'volatility': float(volatility),
            'annualized_volatility': float(annualized_volatility),
            'rolling_volatility': float(rolling_volatility),
        }

    def calculate_performance_metrics(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> dict[str, float]:
        """Calculate performance metrics.

        Args:
            returns: Portfolio returns series
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary with performance metrics
        """
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
            }

        # Total return
        total_return = (1 + returns).prod() - 1

        # Annualized return
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio (using only negative returns for downside)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # Calmar ratio (return / max drawdown)
        drawdown_metrics = self.calculate_max_drawdown(returns)
        max_drawdown = abs(drawdown_metrics['max_drawdown'])
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
        }

    def calculate_risk_attribution(self, returns: pd.Series) -> dict[str, Any]:
        """Calculate risk attribution metrics.

        Args:
            returns: Portfolio returns series

        Returns:
            Dictionary with risk attribution
        """
        if len(returns) == 0:
            return {
                'total_risk': 0.0,
                'systematic_risk': 0.0,
                'idiosyncratic_risk': 0.0,
            }

        # Simplified risk attribution
        total_risk = returns.std()

        # Assumed 70% systematic, 30% idiosyncratic
        systematic_risk = total_risk * 0.7
        idiosyncratic_risk = total_risk * 0.3

        return {
            'total_risk': float(total_risk),
            'systematic_risk': float(systematic_risk),
            'idiosyncratic_risk': float(idiosyncratic_risk),
        }

    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets.

        Args:
            returns_df: DataFrame with returns for multiple assets

        Returns:
            Correlation matrix DataFrame
        """
        if returns_df.empty:
            return pd.DataFrame()

        return returns_df.corr()

    def calculate_portfolio_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        confidence_level: float = 0.95,
        lookback_period: int = 252,
    ) -> dict[str, Any]:
        """Calculate comprehensive portfolio metrics.

        Args:
            returns: Portfolio returns series
            risk_free_rate: Risk-free rate
            confidence_level: Confidence level for VaR calculations
            lookback_period: Lookback period for calculations

        Returns:
            Dictionary with all portfolio metrics
        """
        # Calculate all metrics
        var = self.calculate_portfolio_var(returns, confidence_level, lookback_period)
        expected_shortfall = self.calculate_expected_shortfall(
            returns, confidence_level, lookback_period
        )
        drawdown_metrics = self.calculate_max_drawdown(returns)
        volatility_metrics = self.calculate_volatility_metrics(returns)
        performance_metrics = self.calculate_performance_metrics(returns, risk_free_rate)
        risk_attribution = self.calculate_risk_attribution(returns)

        # Combine all metrics
        portfolio_metrics = {
            # Risk metrics
            'var': var,
            'expected_shortfall': expected_shortfall,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_drawdown_duration': drawdown_metrics['max_drawdown_duration'],
            'current_drawdown': drawdown_metrics['current_drawdown'],
            'current_drawdown_duration': drawdown_metrics['current_drawdown_duration'],
            'volatility': volatility_metrics['volatility'],
            'annualized_volatility': volatility_metrics['annualized_volatility'],
            'rolling_volatility': volatility_metrics['rolling_volatility'],
            # Performance metrics
            **performance_metrics,
            # Risk attribution
            **risk_attribution,
            # Additional metrics
            'total_observations': len(returns),
            'confidence_level': confidence_level,
            'lookback_period': lookback_period,
        }

        return portfolio_metrics

    def calculate_stress_test_scenarios(
        self,
        portfolio_returns: pd.Series,
        market_scenarios: dict[str, pd.Series],
    ) -> dict[str, Any]:
        """Calculate stress test results for different market scenarios.

        Args:
            portfolio_returns: Historical portfolio returns
            market_scenarios: Dictionary of market scenario return series

        Returns:
            Dictionary with stress test results
        """
        results = {}

        for scenario_name, scenario_returns in market_scenarios.items():
            if len(scenario_returns) == 0 or len(portfolio_returns) == 0:
                results[scenario_name] = {
                    'var_impact': 0.0,
                    'max_drawdown_impact': 0.0,
                    'volatility_impact': 0.0,
                }
                continue

            # Calculate scenario metrics
            scenario_var = self.calculate_portfolio_var(scenario_returns)
            current_var = self.calculate_portfolio_var(portfolio_returns)

            scenario_dd = self.calculate_max_drawdown(scenario_returns)
            current_dd = self.calculate_max_drawdown(portfolio_returns)

            scenario_vol = self.calculate_volatility_metrics(scenario_returns)[
                'annualized_volatility'
            ]
            current_vol = self.calculate_volatility_metrics(portfolio_returns)[
                'annualized_volatility'
            ]

            # Calculate impacts
            results[scenario_name] = {
                'var_impact': scenario_var - current_var,
                'max_drawdown_impact': scenario_dd['max_drawdown'] - current_dd['max_drawdown'],
                'volatility_impact': scenario_vol - current_vol,
                'scenario_var': scenario_var,
                'scenario_max_drawdown': scenario_dd['max_drawdown'],
                'scenario_volatility': scenario_vol,
            }

        return results
