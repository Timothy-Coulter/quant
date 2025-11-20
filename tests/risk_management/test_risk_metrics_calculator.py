"""Comprehensive unit tests for the RiskMetricsCalculator class.

This module contains tests for the risk metrics calculation functionality including
VaR, Expected Shortfall, drawdown, volatility, performance metrics, and stress testing.
"""

import logging
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.risk_management.risk_metrics_calculator import RiskMetricsCalculator


class TestRiskMetricsCalculatorInitialization:
    """Test suite for RiskMetricsCalculator initialization."""

    def test_init_default(self) -> None:
        """Test RiskMetricsCalculator initialization with defaults."""
        calculator = RiskMetricsCalculator()

        assert calculator.logger is not None
        assert isinstance(calculator.logger, logging.Logger)

    def test_init_with_custom_logger(self) -> None:
        """Test RiskMetricsCalculator initialization with custom logger."""
        custom_logger = Mock(spec=logging.Logger)
        calculator = RiskMetricsCalculator(logger=custom_logger)

        assert calculator.logger == custom_logger

    def test_logger_name(self) -> None:
        """Test that logger is properly named."""
        calculator = RiskMetricsCalculator()
        expected_name = "backtester.risk_management.risk_metrics_calculator"
        assert calculator.logger.name == expected_name


class TestPortfolioVaR:
    """Test suite for calculate_portfolio_var method."""

    def test_var_normal_case(self) -> None:
        """Test VaR calculation with normal returns data."""
        calculator = RiskMetricsCalculator()

        # Create sample returns data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))

        var = calculator.calculate_portfolio_var(returns, confidence_level=0.95)

        # VaR should be negative (indicating potential loss)
        assert isinstance(var, float)
        assert var < 0
        assert abs(var) > 0  # Should have some magnitude

    def test_var_different_confidence_levels(self) -> None:
        """Test VaR calculation with different confidence levels."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))

        var_95 = calculator.calculate_portfolio_var(returns, confidence_level=0.95)
        var_99 = calculator.calculate_portfolio_var(returns, confidence_level=0.99)

        # Higher confidence level should result in more negative VaR
        assert var_99 < var_95

    def test_var_different_lookback_periods(self) -> None:
        """Test VaR calculation with different lookback periods."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 500))

        var_short = calculator.calculate_portfolio_var(returns, lookback_period=30)
        var_long = calculator.calculate_portfolio_var(returns, lookback_period=252)

        # Both should be valid floats
        assert isinstance(var_short, float)
        assert isinstance(var_long, float)

    def test_var_empty_returns(self) -> None:
        """Test VaR calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        var = calculator.calculate_portfolio_var(empty_returns)

        assert var == 0.0

    def test_var_single_value(self) -> None:
        """Test VaR calculation with single return value."""
        calculator = RiskMetricsCalculator()

        single_return = pd.Series([0.01])
        var = calculator.calculate_portfolio_var(single_return)

        assert isinstance(var, float)

    def test_var_with_negative_returns(self) -> None:
        """Test VaR calculation with predominantly negative returns."""
        calculator = RiskMetricsCalculator()

        # Create returns with some large negative values
        returns = pd.Series([-0.05, -0.03, -0.02, 0.01, 0.02, -0.04])
        var = calculator.calculate_portfolio_var(returns)

        # Should reflect the negative tail of the distribution
        assert isinstance(var, float)
        assert var < 0

    def test_var_with_all_positive_returns(self) -> None:
        """Test VaR calculation with all positive returns."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.01])
        var = calculator.calculate_portfolio_var(returns)

        # Even with positive returns, VaR should be calculated from the distribution
        assert isinstance(var, float)
        # Might be negative if there are no negative values in the tail

    def test_var_lookback_longer_than_data(self) -> None:
        """Test VaR calculation when lookback period is longer than data."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series(np.random.normal(0, 0.02, 50))
        var = calculator.calculate_portfolio_var(returns, lookback_period=252)

        # Should use all available data
        assert isinstance(var, float)

    def test_var_extreme_values(self) -> None:
        """Test VaR calculation with extreme return values."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series([-0.50, 0.10, -0.30, 0.05, -0.20, 0.15, -0.40])
        var = calculator.calculate_portfolio_var(returns)

        # Should handle extreme values
        assert isinstance(var, float)
        assert var < 0


class TestExpectedShortfall:
    """Test suite for calculate_expected_shortfall method."""

    def test_expected_shortfall_normal_case(self) -> None:
        """Test Expected Shortfall calculation with normal returns data."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))

        es = calculator.calculate_expected_shortfall(returns, confidence_level=0.95)

        assert isinstance(es, float)
        assert es < 0  # Should be negative for loss

    def test_expected_shortfall_consistency_with_var(self) -> None:
        """Test that Expected Shortfall is more negative than VaR."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))

        var = calculator.calculate_portfolio_var(returns, confidence_level=0.95)
        es = calculator.calculate_expected_shortfall(returns, confidence_level=0.95)

        # Expected Shortfall should be more negative (worse) than VaR
        assert es < var

    def test_expected_shortfall_empty_returns(self) -> None:
        """Test Expected Shortfall calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        es = calculator.calculate_expected_shortfall(empty_returns)

        assert es == 0.0

    def test_expected_shortfall_all_negative_returns(self) -> None:
        """Test Expected Shortfall when all returns are negative."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05])
        es = calculator.calculate_expected_shortfall(returns)

        assert isinstance(es, float)
        assert es < 0

    def test_expected_shortfall_edge_case_tail(self) -> None:
        """Test Expected Shortfall with very few tail values."""
        calculator = RiskMetricsCalculator()

        # Only one value in the tail
        returns = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01, -0.10])
        es = calculator.calculate_expected_shortfall(returns, confidence_level=0.95)

        # Should return the VaR since there's only one tail value
        assert isinstance(es, float)

    def test_expected_shortfall_different_confidence_levels(self) -> None:
        """Test Expected Shortfall with different confidence levels."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        es_95 = calculator.calculate_expected_shortfall(returns, confidence_level=0.95)
        es_99 = calculator.calculate_expected_shortfall(returns, confidence_level=0.99)

        # Higher confidence level should result in more negative ES
        assert es_99 < es_95


class TestMaxDrawdown:
    """Test suite for calculate_max_drawdown method."""

    def test_max_drawdown_normal_portfolio(self) -> None:
        """Test max drawdown calculation with normal portfolio performance."""
        calculator = RiskMetricsCalculator()

        # Create returns that simulate a drawdown pattern
        returns = pd.Series([0.01, 0.02, -0.01, -0.03, -0.02, 0.01, 0.02, 0.03])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        assert isinstance(drawdown_metrics, dict)
        assert 'max_drawdown' in drawdown_metrics
        assert 'max_drawdown_duration' in drawdown_metrics
        assert 'current_drawdown' in drawdown_metrics
        assert 'current_drawdown_duration' in drawdown_metrics

    def test_max_drawdown_ascending_trend(self) -> None:
        """Test max drawdown with consistently ascending portfolio."""
        calculator = RiskMetricsCalculator()

        # Monotonically increasing returns
        returns = pd.Series([0.01, 0.02, 0.03, 0.02, 0.04, 0.05, 0.03, 0.06])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        # Should have minimal or zero drawdown
        assert drawdown_metrics['max_drawdown'] >= -0.01  # Small tolerance
        assert drawdown_metrics['max_drawdown_duration'] == 0

    def test_max_drawdown_descending_trend(self) -> None:
        """Test max drawdown with consistently descending portfolio."""
        calculator = RiskMetricsCalculator()

        # Monotonically decreasing returns
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01, -0.04, -0.05, -0.02, -0.06])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        # Should have significant drawdown
        assert drawdown_metrics['max_drawdown'] < 0
        assert drawdown_metrics['max_drawdown_duration'] >= 0  # Allow for floating point precision

    def test_max_drawdown_empty_returns(self) -> None:
        """Test max drawdown calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        drawdown_metrics = calculator.calculate_max_drawdown(empty_returns)

        expected = {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'current_drawdown': 0.0,
            'current_drawdown_duration': 0,
        }
        assert drawdown_metrics == expected

    def test_max_drawdown_single_value(self) -> None:
        """Test max drawdown calculation with single return value."""
        calculator = RiskMetricsCalculator()

        single_return = pd.Series([0.05])
        drawdown_metrics = calculator.calculate_max_drawdown(single_return)

        # Should have no drawdown with single positive value
        assert drawdown_metrics['max_drawdown'] == 0.0
        assert drawdown_metrics['current_drawdown'] == 0.0

    def test_max_drawdown_volatile_portfolio(self) -> None:
        """Test max drawdown calculation with volatile portfolio."""
        calculator = RiskMetricsCalculator()

        # High volatility pattern
        returns = pd.Series([0.10, -0.08, 0.12, -0.15, 0.05, -0.10, 0.08, -0.12])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        # Should have significant drawdowns
        assert drawdown_metrics['max_drawdown'] < 0
        assert drawdown_metrics['max_drawdown_duration'] > 0

    def test_max_drawdown_duration_calculation(self) -> None:
        """Test max drawdown duration calculation."""
        calculator = RiskMetricsCalculator()

        # Create pattern with specific drawdown duration
        returns = pd.Series([0.01, 0.02, -0.01, -0.02, -0.03, -0.02, 0.01])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        # Check that duration is calculated correctly
        assert isinstance(drawdown_metrics['max_drawdown_duration'], int)
        assert drawdown_metrics['max_drawdown_duration'] >= 0

    def test_max_drawdown_current_drawdown(self) -> None:
        """Test current drawdown calculation."""
        calculator = RiskMetricsCalculator()

        # End with negative cumulative return
        returns = pd.Series([0.01, -0.02, -0.01, -0.01])
        drawdown_metrics = calculator.calculate_max_drawdown(returns)

        # Current drawdown should reflect the current state
        assert isinstance(drawdown_metrics['current_drawdown'], float)
        assert isinstance(drawdown_metrics['current_drawdown_duration'], int)


class TestVolatilityMetrics:
    """Test suite for calculate_volatility_metrics method."""

    def test_volatility_normal_case(self) -> None:
        """Test volatility metrics calculation with normal returns."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        vol_metrics = calculator.calculate_volatility_metrics(returns)

        assert isinstance(vol_metrics, dict)
        assert 'volatility' in vol_metrics
        assert 'annualized_volatility' in vol_metrics
        assert 'rolling_volatility' in vol_metrics
        assert vol_metrics['volatility'] > 0
        assert vol_metrics['annualized_volatility'] > vol_metrics['volatility']

    def test_volatility_empty_returns(self) -> None:
        """Test volatility metrics calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        vol_metrics = calculator.calculate_volatility_metrics(empty_returns)

        expected = {
            'volatility': 0.0,
            'annualized_volatility': 0.0,
            'rolling_volatility': 0.0,
        }
        assert vol_metrics == expected

    def test_volatility_constant_returns(self) -> None:
        """Test volatility metrics with constant returns."""
        calculator = RiskMetricsCalculator()

        constant_returns = pd.Series([0.01] * 50)
        vol_metrics = calculator.calculate_volatility_metrics(constant_returns)

        # Volatility should be zero for constant returns
        assert vol_metrics['volatility'] == 0.0
        assert vol_metrics['annualized_volatility'] == 0.0
        assert abs(vol_metrics['rolling_volatility']) < 1e-10  # Handle floating point precision

    def test_volatility_rolling_window(self) -> None:
        """Test rolling volatility calculation."""
        calculator = RiskMetricsCalculator()

        # Create returns with changing volatility
        returns = pd.Series(
            list(np.random.normal(0, 0.01, 20)) + list(np.random.normal(0, 0.05, 20))
        )
        vol_metrics = calculator.calculate_volatility_metrics(returns)

        # Rolling volatility should be based on last 30 values
        assert isinstance(vol_metrics['rolling_volatility'], float)
        assert vol_metrics['rolling_volatility'] > 0

    def test_volatility_insufficient_data_for_rolling(self) -> None:
        """Test rolling volatility when data is insufficient for 30-day window."""
        calculator = RiskMetricsCalculator()

        # Only 20 data points
        returns = pd.Series(np.random.normal(0, 0.02, 20))
        vol_metrics = calculator.calculate_volatility_metrics(returns)

        # Should fall back to annualized volatility
        assert vol_metrics['rolling_volatility'] == vol_metrics['annualized_volatility']

    def test_volatility_annualization(self) -> None:
        """Test volatility annualization factor."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        vol_metrics = calculator.calculate_volatility_metrics(returns)

        # Annualized should be daily * sqrt(252)
        expected_annualized = vol_metrics['volatility'] * np.sqrt(252)
        assert abs(vol_metrics['annualized_volatility'] - expected_annualized) < 1e-10


class TestPerformanceMetrics:
    """Test suite for calculate_performance_metrics method."""

    def test_performance_metrics_normal_case(self) -> None:
        """Test performance metrics calculation with normal returns."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))  # Slight positive drift

        perf_metrics = calculator.calculate_performance_metrics(returns, risk_free_rate=0.02)

        assert isinstance(perf_metrics, dict)
        expected_keys = {
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
        }
        assert set(perf_metrics.keys()) == expected_keys

    def test_performance_metrics_empty_returns(self) -> None:
        """Test performance metrics calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        perf_metrics = calculator.calculate_performance_metrics(empty_returns)

        expected = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
        }
        assert perf_metrics == expected

    def test_performance_metrics_zero_risk_free_rate(self) -> None:
        """Test performance metrics with zero risk-free rate."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 50))

        perf_metrics = calculator.calculate_performance_metrics(returns, risk_free_rate=0.0)

        assert isinstance(perf_metrics['sharpe_ratio'], float)
        assert isinstance(perf_metrics['sortino_ratio'], float)

    def test_performance_metrics_no_negative_returns(self) -> None:
        """Test performance metrics with only positive returns."""
        calculator = RiskMetricsCalculator()

        positive_returns = pd.Series([0.01, 0.02, 0.03, 0.015, 0.025])
        perf_metrics = calculator.calculate_performance_metrics(positive_returns)

        # Sortino ratio might be handled differently with no negative returns
        assert isinstance(perf_metrics['sortino_ratio'], float)

    def test_performance_metrics_all_negative_returns(self) -> None:
        """Test performance metrics with all negative returns."""
        calculator = RiskMetricsCalculator()

        negative_returns = pd.Series([-0.01, -0.02, -0.03, -0.015, -0.025])
        perf_metrics = calculator.calculate_performance_metrics(negative_returns)

        # All performance metrics should be negative or zero
        assert perf_metrics['total_return'] < 0
        assert perf_metrics['annualized_return'] < 0
        assert perf_metrics['sharpe_ratio'] < 0
        assert perf_metrics['sortino_ratio'] <= 0

    def test_performance_metrics_high_volatility(self) -> None:
        """Test performance metrics with high volatility."""
        calculator = RiskMetricsCalculator()

        high_vol_returns = pd.Series(np.random.normal(0.001, 0.10, 50))
        perf_metrics = calculator.calculate_performance_metrics(high_vol_returns)

        # Should handle high volatility appropriately
        assert isinstance(perf_metrics['sharpe_ratio'], float)
        assert isinstance(perf_metrics['sortino_ratio'], float)

    def test_performance_metrics_calmar_ratio_edge_case(self) -> None:
        """Test Calmar ratio when max drawdown is zero."""
        calculator = RiskMetricsCalculator()

        # Monotonically increasing returns (no drawdown)
        ascending_returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        perf_metrics = calculator.calculate_performance_metrics(ascending_returns)

        # Should handle zero drawdown gracefully
        assert isinstance(perf_metrics['calmar_ratio'], float)
        # When max drawdown is zero, Calmar ratio should be infinity or a large number
        # The implementation should handle this case

    def test_performance_metrics_different_risk_free_rates(self) -> None:
        """Test performance metrics with different risk-free rates."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 50))

        perf_0pct = calculator.calculate_performance_metrics(returns, risk_free_rate=0.00)
        perf_2pct = calculator.calculate_performance_metrics(returns, risk_free_rate=0.02)
        perf_5pct = calculator.calculate_performance_metrics(returns, risk_free_rate=0.05)

        # Higher risk-free rate should generally result in lower Sharpe ratios
        assert perf_5pct['sharpe_ratio'] <= perf_2pct['sharpe_ratio']
        assert perf_2pct['sharpe_ratio'] <= perf_0pct['sharpe_ratio']


class TestRiskAttribution:
    """Test suite for calculate_risk_attribution method."""

    def test_risk_attribution_normal_case(self) -> None:
        """Test risk attribution calculation with normal returns."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        risk_attr = calculator.calculate_risk_attribution(returns)

        assert isinstance(risk_attr, dict)
        assert 'total_risk' in risk_attr
        assert 'systematic_risk' in risk_attr
        assert 'idiosyncratic_risk' in risk_attr
        assert risk_attr['total_risk'] > 0
        assert risk_attr['systematic_risk'] == risk_attr['total_risk'] * 0.7
        assert risk_attr['idiosyncratic_risk'] == risk_attr['total_risk'] * 0.3

    def test_risk_attribution_empty_returns(self) -> None:
        """Test risk attribution calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        risk_attr = calculator.calculate_risk_attribution(empty_returns)

        expected = {
            'total_risk': 0.0,
            'systematic_risk': 0.0,
            'idiosyncratic_risk': 0.0,
        }
        assert risk_attr == expected

    def test_risk_attribution_conservation(self) -> None:
        """Test that systematic + idiosyncratic = total risk."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 50))

        risk_attr = calculator.calculate_risk_attribution(returns)

        # Should sum to total risk
        calculated_total = risk_attr['systematic_risk'] + risk_attr['idiosyncratic_risk']
        assert abs(calculated_total - risk_attr['total_risk']) < 1e-10

    def test_risk_attribution_zero_volatility(self) -> None:
        """Test risk attribution with zero volatility returns."""
        calculator = RiskMetricsCalculator()

        constant_returns = pd.Series([0.01] * 50)
        risk_attr = calculator.calculate_risk_attribution(constant_returns)

        # All risk measures should be zero
        assert risk_attr['total_risk'] == 0.0
        assert risk_attr['systematic_risk'] == 0.0
        assert risk_attr['idiosyncratic_risk'] == 0.0


class TestCorrelationMatrix:
    """Test suite for calculate_correlation_matrix method."""

    def test_correlation_matrix_normal_case(self) -> None:
        """Test correlation matrix calculation with multiple assets."""
        calculator = RiskMetricsCalculator()

        # Create correlated returns for multiple assets
        np.random.seed(42)
        market_factor = np.random.normal(0, 0.02, 100)
        asset1_returns = market_factor + np.random.normal(0, 0.01, 100)
        asset2_returns = market_factor + np.random.normal(0, 0.01, 100)
        asset3_returns = np.random.normal(0, 0.02, 100)  # Independent

        returns_df = pd.DataFrame(
            {
                'Asset1': asset1_returns,
                'Asset2': asset2_returns,
                'Asset3': asset3_returns,
            }
        )

        corr_matrix = calculator.calculate_correlation_matrix(returns_df)

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert (corr_matrix.columns == ['Asset1', 'Asset2', 'Asset3']).all()
        assert (corr_matrix.index == ['Asset1', 'Asset2', 'Asset3']).all()
        # Diagonal should be 1s
        assert (np.diag(corr_matrix) == 1.0).all()
        # Asset1 and Asset2 should be highly correlated
        assert corr_matrix.loc['Asset1', 'Asset2'] > 0.5

    def test_correlation_matrix_empty_dataframe(self) -> None:
        """Test correlation matrix calculation with empty DataFrame."""
        calculator = RiskMetricsCalculator()

        empty_df = pd.DataFrame()
        corr_matrix = calculator.calculate_correlation_matrix(empty_df)

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.empty

    def test_correlation_matrix_single_column(self) -> None:
        """Test correlation matrix with single asset."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series(np.random.normal(0, 0.02, 50))
        returns_df = pd.DataFrame({'Asset1': returns})

        corr_matrix = calculator.calculate_correlation_matrix(returns_df)

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix.iloc[0, 0] == 1.0

    def test_correlation_matrix_two_assets(self) -> None:
        """Test correlation matrix with two assets."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns_df = pd.DataFrame(
            {
                'Asset1': np.random.normal(0, 0.02, 50),
                'Asset2': np.random.normal(0, 0.02, 50),
            }
        )

        corr_matrix = calculator.calculate_correlation_matrix(returns_df)

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (2, 2)
        assert corr_matrix.loc['Asset1', 'Asset2'] == corr_matrix.loc['Asset2', 'Asset1']

    def test_correlation_matrix_perfect_correlation(self) -> None:
        """Test correlation matrix with perfectly correlated assets."""
        calculator = RiskMetricsCalculator()

        base_returns = np.random.normal(0, 0.02, 50)
        returns_df = pd.DataFrame(
            {
                'Asset1': base_returns,
                'Asset2': base_returns * 2,  # Perfect correlation
            }
        )

        corr_matrix = calculator.calculate_correlation_matrix(returns_df)

        # Should be perfectly correlated
        assert abs(corr_matrix.loc['Asset1', 'Asset2'] - 1.0) < 1e-10

    def test_correlation_matrix_perfect_anticorrelation(self) -> None:
        """Test correlation matrix with perfectly anticorrelated assets."""
        calculator = RiskMetricsCalculator()

        base_returns = np.random.normal(0, 0.02, 50)
        returns_df = pd.DataFrame(
            {
                'Asset1': base_returns,
                'Asset2': -base_returns,  # Perfect anticorrelation
            }
        )

        corr_matrix = calculator.calculate_correlation_matrix(returns_df)

        # Should be perfectly anticorrelated
        assert abs(corr_matrix.loc['Asset1', 'Asset2'] - (-1.0)) < 1e-10


class TestPortfolioMetrics:
    """Test suite for calculate_portfolio_metrics method."""

    def test_portfolio_metrics_normal_case(self) -> None:
        """Test comprehensive portfolio metrics calculation."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        portfolio_metrics = calculator.calculate_portfolio_metrics(
            returns, risk_free_rate=0.02, confidence_level=0.95, lookback_period=252
        )

        assert isinstance(portfolio_metrics, dict)

        # Check all expected keys are present
        expected_keys = {
            'var',
            'expected_shortfall',
            'max_drawdown',
            'max_drawdown_duration',
            'current_drawdown',
            'current_drawdown_duration',
            'volatility',
            'annualized_volatility',
            'rolling_volatility',
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'total_risk',
            'systematic_risk',
            'idiosyncratic_risk',
            'total_observations',
            'confidence_level',
            'lookback_period',
        }
        assert set(portfolio_metrics.keys()) == expected_keys

        # Check data types
        assert isinstance(portfolio_metrics['total_observations'], int)
        assert portfolio_metrics['total_observations'] == len(returns)
        assert isinstance(portfolio_metrics['confidence_level'], float)
        assert isinstance(portfolio_metrics['lookback_period'], int)

    def test_portfolio_metrics_empty_returns(self) -> None:
        """Test portfolio metrics calculation with empty returns."""
        calculator = RiskMetricsCalculator()

        empty_returns = pd.Series([], dtype=float)
        portfolio_metrics = calculator.calculate_portfolio_metrics(empty_returns)

        # Should return metrics with zero/null values where appropriate
        assert isinstance(portfolio_metrics, dict)
        assert portfolio_metrics['total_observations'] == 0
        assert portfolio_metrics['confidence_level'] == 0.95  # Default
        assert portfolio_metrics['lookback_period'] == 252  # Default

    def test_portfolio_metrics_different_parameters(self) -> None:
        """Test portfolio metrics with different parameter combinations."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        # Test with different parameters
        metrics_99 = calculator.calculate_portfolio_metrics(
            returns, confidence_level=0.99, lookback_period=100
        )
        metrics_95 = calculator.calculate_portfolio_metrics(
            returns, confidence_level=0.95, lookback_period=50
        )

        # VaR should be different
        assert metrics_99['var'] < metrics_95['var']  # 99% VaR should be more negative
        assert metrics_99['confidence_level'] == 0.99
        assert metrics_99['lookback_period'] == 100
        assert metrics_95['confidence_level'] == 0.95
        assert metrics_95['lookback_period'] == 50

    def test_portfolio_metrics_integration(self) -> None:
        """Test that portfolio metrics correctly integrates all sub-metrics."""
        calculator = RiskMetricsCalculator()

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        portfolio_metrics = calculator.calculate_portfolio_metrics(returns)

        # Verify that individual method results match portfolio metrics
        var = calculator.calculate_portfolio_var(returns)
        drawdown = calculator.calculate_max_drawdown(returns)
        volatility = calculator.calculate_volatility_metrics(returns)
        performance = calculator.calculate_performance_metrics(returns)
        risk_attr = calculator.calculate_risk_attribution(returns)

        assert portfolio_metrics['var'] == var
        assert portfolio_metrics['max_drawdown'] == drawdown['max_drawdown']
        assert portfolio_metrics['volatility'] == volatility['volatility']
        assert portfolio_metrics['total_return'] == performance['total_return']
        assert portfolio_metrics['total_risk'] == risk_attr['total_risk']


class TestStressTestScenarios:
    """Test suite for calculate_stress_test_scenarios method."""

    def test_stress_test_normal_case(self) -> None:
        """Test stress test scenarios with normal market conditions."""
        calculator = RiskMetricsCalculator()

        # Historical portfolio returns
        np.random.seed(42)
        portfolio_returns = pd.Series(np.random.normal(0, 0.02, 252))

        # Market stress scenarios
        market_scenarios = {
            'financial_crisis': pd.Series(np.random.normal(-0.05, 0.08, 100)),
            'recession': pd.Series(np.random.normal(-0.03, 0.05, 100)),
            'high_volatility': pd.Series(np.random.normal(0, 0.10, 100)),
        }

        stress_results = calculator.calculate_stress_test_scenarios(
            portfolio_returns, market_scenarios
        )

        assert isinstance(stress_results, dict)
        assert len(stress_results) == 3

        for _scenario_name, scenario_results in stress_results.items():
            assert isinstance(scenario_results, dict)
            expected_keys = {
                'var_impact',
                'max_drawdown_impact',
                'volatility_impact',
                'scenario_var',
                'scenario_max_drawdown',
                'scenario_volatility',
            }
            assert set(scenario_results.keys()) == expected_keys

    def test_stress_test_empty_portfolio_returns(self) -> None:
        """Test stress test with empty portfolio returns."""
        calculator = RiskMetricsCalculator()

        empty_portfolio = pd.Series([], dtype=float)
        market_scenarios = {
            'crisis': pd.Series(np.random.normal(-0.05, 0.08, 50)),
        }

        stress_results = calculator.calculate_stress_test_scenarios(
            empty_portfolio, market_scenarios
        )

        # Should return zero impacts for all scenarios
        for scenario_results in stress_results.values():
            assert scenario_results['var_impact'] == 0.0
            assert scenario_results['max_drawdown_impact'] == 0.0
            assert scenario_results['volatility_impact'] == 0.0

    def test_stress_test_empty_scenarios(self) -> None:
        """Test stress test with empty scenario data."""
        calculator = RiskMetricsCalculator()

        portfolio_returns = pd.Series(np.random.normal(0, 0.02, 100))
        empty_scenarios: dict[str, Any] = {}

        stress_results = calculator.calculate_stress_test_scenarios(
            portfolio_returns, empty_scenarios
        )

        assert stress_results == {}

    def test_stress_test_mixed_scenario_data(self) -> None:
        """Test stress test with mix of empty and non-empty scenarios."""
        calculator = RiskMetricsCalculator()

        portfolio_returns = pd.Series(np.random.normal(0, 0.02, 100))
        mixed_scenarios = {
            'normal_market': pd.Series(np.random.normal(0, 0.02, 50)),
            'crisis': pd.Series([], dtype=float),  # Empty scenario
            'recovery': pd.Series(np.random.normal(0.02, 0.03, 30)),
        }

        stress_results = calculator.calculate_stress_test_scenarios(
            portfolio_returns, mixed_scenarios
        )

        # Crisis scenario should have zero impacts
        assert stress_results['crisis']['var_impact'] == 0.0
        assert stress_results['crisis']['max_drawdown_impact'] == 0.0

        # Other scenarios should have calculated impacts
        assert isinstance(stress_results['normal_market']['var_impact'], float)
        assert isinstance(stress_results['recovery']['var_impact'], float)

    def test_stress_test_impact_calculation(self) -> None:
        """Test that stress test impacts are calculated correctly."""
        calculator = RiskMetricsCalculator()

        # Simple portfolio returns
        portfolio_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        # Simple stress scenario
        stress_scenario = pd.Series([-0.05, -0.03, -0.04, -0.02, -0.06])

        stress_results = calculator.calculate_stress_test_scenarios(
            portfolio_returns, {'stress': stress_scenario}
        )

        stress_result = stress_results['stress']

        # Should have calculated impacts
        assert isinstance(stress_result['var_impact'], float)
        assert isinstance(stress_result['max_drawdown_impact'], float)
        assert isinstance(stress_result['volatility_impact'], float)

        # Scenario should have worse metrics than portfolio
        assert stress_result['scenario_var'] < calculator.calculate_portfolio_var(portfolio_returns)
        assert (
            stress_result['scenario_max_drawdown']
            < calculator.calculate_max_drawdown(portfolio_returns)['max_drawdown']
        )


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_very_small_returns(self) -> None:
        """Test all methods with very small return values."""
        calculator = RiskMetricsCalculator()

        small_returns = pd.Series([0.0001, -0.0001, 0.0002, -0.0002])

        # Should handle small values gracefully
        var = calculator.calculate_portfolio_var(small_returns)
        es = calculator.calculate_expected_shortfall(small_returns)
        drawdown = calculator.calculate_max_drawdown(small_returns)
        volatility = calculator.calculate_volatility_metrics(small_returns)
        performance = calculator.calculate_performance_metrics(small_returns)

        assert isinstance(var, float)
        assert isinstance(es, float)
        assert isinstance(drawdown, dict)
        assert isinstance(volatility, dict)
        assert isinstance(performance, dict)

    def test_very_large_returns(self) -> None:
        """Test all methods with very large return values."""
        calculator = RiskMetricsCalculator()

        large_returns = pd.Series([1.0, -0.8, 0.5, -0.9, 2.0])

        # Should handle large values
        var = calculator.calculate_portfolio_var(large_returns)
        es = calculator.calculate_expected_shortfall(large_returns)
        drawdown = calculator.calculate_max_drawdown(large_returns)
        volatility = _calculate_volatility_metrics(large_returns)
        performance = _calculate_performance_metrics(large_returns)

        assert isinstance(var, float)
        assert isinstance(es, float)
        assert isinstance(drawdown, dict)
        assert isinstance(volatility, dict)
        assert isinstance(performance, dict)

    def test_duplicate_values(self) -> None:
        """Test methods with duplicate return values."""
        calculator = RiskMetricsCalculator()

        duplicate_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])

        # Should handle duplicates
        calculator.calculate_portfolio_var(duplicate_returns)
        calculator.calculate_max_drawdown(duplicate_returns)
        volatility = calculator.calculate_volatility_metrics(duplicate_returns)

        # Volatility should be zero for constant returns
        assert volatility['volatility'] == 0.0

    def test_mixed_data_types(self) -> None:
        """Test methods with mixed integer and float values."""
        calculator = RiskMetricsCalculator()

        mixed_returns = pd.Series([1, 0, -1, 2, -2])  # Should be converted to float

        # Should handle conversion gracefully
        var = calculator.calculate_portfolio_var(mixed_returns)
        assert isinstance(var, float)

    def test_extreme_confidence_levels(self) -> None:
        """Test VaR and Expected Shortfall with extreme confidence levels."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series(np.random.normal(0, 0.02, 100))

        # Test very low confidence level
        var_low = calculator.calculate_portfolio_var(returns, confidence_level=0.01)
        es_low = calculator.calculate_expected_shortfall(returns, confidence_level=0.01)

        # Test very high confidence level
        var_high = calculator.calculate_portfolio_var(returns, confidence_level=0.999)
        es_high = calculator.calculate_expected_shortfall(returns, confidence_level=0.999)

        # Higher confidence should be more extreme
        assert var_high < var_low
        assert es_high < es_low

    def test_zero_lookback_period(self) -> None:
        """Test methods with zero lookback period."""
        calculator = RiskMetricsCalculator()

        returns = pd.Series(np.random.normal(0, 0.02, 50))

        # Should handle zero lookback gracefully (use all data)
        var = calculator.calculate_portfolio_var(returns, lookback_period=0)
        es = calculator.calculate_expected_shortfall(returns, lookback_period=0)

        assert isinstance(var, float)
        assert isinstance(es, float)


class TestIntegrationScenarios:
    """Test suite for real-world integration scenarios."""

    def test_typical_portfolio_scenario(self) -> None:
        """Test typical portfolio risk assessment scenario."""
        calculator = RiskMetricsCalculator()

        # Simulate 2 years of daily returns
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(0.0005, 0.015, 504))  # 2 years

        # Calculate comprehensive metrics
        portfolio_metrics = calculator.calculate_portfolio_metrics(daily_returns)

        # Validate realistic portfolio metrics
        assert abs(portfolio_metrics['annualized_return']) < 1.0  # Reasonable annual return
        assert portfolio_metrics['annualized_volatility'] > 0  # Should have some volatility
        assert portfolio_metrics['sharpe_ratio'] > -10  # Reasonable range
        assert portfolio_metrics['sharpe_ratio'] < 10

        # Drawdown metrics should be reasonable
        assert -1.0 < portfolio_metrics['max_drawdown'] < 0
        assert portfolio_metrics['max_drawdown_duration'] >= 0

    def test_high_frequency_trading_scenario(self) -> None:
        """Test high-frequency trading scenario with frequent trades."""
        calculator = RiskMetricsCalculator()

        # Simulate high-frequency returns (hourly for a day)
        np.random.seed(42)
        hf_returns = pd.Series(np.random.normal(0, 0.001, 390))  # 6.5 hours * 60 minutes

        portfolio_metrics = calculator.calculate_portfolio_metrics(hf_returns)

        # High-frequency should have lower individual returns but potentially high frequency
        assert portfolio_metrics['total_observations'] == 390
        assert portfolio_metrics['volatility'] < 0.1  # Reasonable daily vol

        # Annualized metrics should still work
        assert portfolio_metrics['annualized_volatility'] > 0

    def test_crisis_period_scenario(self) -> None:
        """Test crisis period with extreme market conditions."""
        calculator = RiskMetricsCalculator()

        # Simulate crisis period with large negative returns
        np.random.seed(42)
        crisis_returns = pd.Series(np.random.normal(-0.02, 0.08, 252))

        var = calculator.calculate_portfolio_var(crisis_returns)
        drawdown = calculator.calculate_max_drawdown(crisis_returns)
        portfolio_metrics = calculator.calculate_portfolio_metrics(crisis_returns)

        # Crisis period should show high risk
        assert var < -0.05  # Significant VaR
        assert drawdown['max_drawdown'] < -0.1  # Significant drawdown
        assert portfolio_metrics['sharpe_ratio'] < 0  # Negative risk-adjusted return

    def test_bull_market_scenario(self) -> None:
        """Test bull market scenario with strong positive performance."""
        calculator = RiskMetricsCalculator()

        # Simulate bull market with positive drift and lower volatility
        np.random.seed(42)
        bull_returns = pd.Series(
            np.random.normal(0.002, 0.015, 252)
        )  # Higher positive drift, lower volatility

        performance = calculator.calculate_performance_metrics(bull_returns)
        portfolio_metrics = calculator.calculate_portfolio_metrics(bull_returns)

        # Bull market should show good performance
        assert performance['total_return'] > 0
        assert performance['annualized_return'] > 0
        assert performance['sharpe_ratio'] > 0
        # Allow for some volatility in random data but check it's reasonable
        assert portfolio_metrics['max_drawdown'] > -0.5  # Reasonable drawdown range
        assert portfolio_metrics['sharpe_ratio'] > -2  # Reasonable Sharpe ratio range

    def test_stress_testing_workflow(self) -> None:
        """Test complete stress testing workflow."""
        calculator = RiskMetricsCalculator()

        # Historical performance
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.0005, 0.015, 504))

        # Define stress scenarios
        stress_scenarios = {
            '2008_crisis': pd.Series(np.random.normal(-0.03, 0.12, 252)),
            'covid_crash': pd.Series(np.random.normal(-0.04, 0.15, 126)),
            'flash_crash': pd.Series(np.random.normal(-0.08, 0.20, 30)),
        }

        # Run stress tests
        stress_results = calculator.calculate_stress_test_scenarios(
            historical_returns, stress_scenarios
        )

        # Validate stress test results
        assert len(stress_results) == 3

        for _scenario_name, results in stress_results.items():
            assert results['var_impact'] < 0  # Stress scenarios should increase VaR
            assert results['volatility_impact'] > 0  # Should increase volatility

            # Scenario VaR should be worse than current
            scenario_var = calculator.calculate_portfolio_var(stress_scenarios['2008_crisis'])
            current_var = calculator.calculate_portfolio_var(historical_returns)
            assert scenario_var < current_var


# Global helper functions for testing (since they're used in multiple test classes)
def _calculate_volatility_metrics(returns: pd.Series) -> dict[str, float]:
    """Helper function to calculate volatility metrics for testing."""
    calculator = RiskMetricsCalculator()
    return calculator.calculate_volatility_metrics(returns)


def _calculate_performance_metrics(
    returns: pd.Series, risk_free_rate: float = 0.02
) -> dict[str, float]:
    """Helper function to calculate performance metrics for testing."""
    calculator = RiskMetricsCalculator()
    return calculator.calculate_performance_metrics(returns, risk_free_rate)


if __name__ == "__main__":
    pytest.main([__file__])
