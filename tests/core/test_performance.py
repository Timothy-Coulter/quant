"""Comprehensive tests for the performance calculation and analysis module.

This module contains tests for performance metrics calculation,
risk-adjusted returns, drawdown analysis, and performance reporting.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.core.performance import (
    DrawdownAnalyzer,
    PerformanceAnalyzer,
    PerformanceCalculator,
    PerformanceMetrics,
    ReturnAnalyzer,
    RiskMetrics,
    SharpeCalculator,
)


class TestPerformanceCalculator:
    """Test suite for the PerformanceCalculator class."""

    def test_initialization(self) -> None:
        """Test PerformanceCalculator initialization."""
        calculator = PerformanceCalculator()

        assert calculator.risk_free_rate == 0.02  # Default 2%
        assert calculator.benchmark_return == 0.0  # Default 0%

    def test_initialization_custom_params(self) -> None:
        """Test PerformanceCalculator with custom parameters."""
        calculator = PerformanceCalculator(risk_free_rate=0.015, benchmark_return=0.08)

        assert calculator.risk_free_rate == 0.015
        assert calculator.benchmark_return == 0.08

    def test_calculate_total_return(self) -> None:
        """Test total return calculation."""
        calculator = PerformanceCalculator()

        # Test with simple price series
        prices = pd.Series([100, 110, 105, 120, 115])
        result = calculator.calculate_total_return(prices)

        expected_return = (115 - 100) / 100  # 15%
        assert abs(result - expected_return) < 0.001

    def test_calculate_annualized_return(self) -> None:
        """Test annualized return calculation."""
        calculator = PerformanceCalculator()

        # Test with 2-year period
        initial_value = 100
        final_value = 121  # 21% total return over 2 years
        periods = 2

        result = calculator.calculate_annualized_return(initial_value, final_value, periods)
        expected = (final_value / initial_value) ** (1 / periods) - 1

        assert abs(result - expected) < 0.001

    def test_calculate_sharpe_ratio_basic(self) -> None:
        """Test basic Sharpe ratio calculation."""
        calculator = PerformanceCalculator()

        # Create sample returns
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])

        result = calculator.calculate_sharpe_ratio(returns)

        # Should be calculable (non-zero variance)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_sharpe_ratio_with_risk_free_rate(self) -> None:
        """Test Sharpe ratio with custom risk-free rate."""
        calculator = PerformanceCalculator(risk_free_rate=0.02)

        returns = pd.Series([0.05, -0.01, 0.03, 0.02, 0.04])

        result = calculator.calculate_sharpe_ratio(returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_sortino_ratio(self) -> None:
        """Test Sortino ratio calculation."""
        calculator = PerformanceCalculator()

        returns = pd.Series([0.02, -0.05, 0.03, -0.01, 0.04, 0.01])

        result = calculator.calculate_sortino_ratio(returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        calculator = PerformanceCalculator()

        # Create price series with known drawdown
        prices = pd.Series([100, 110, 90, 95, 105, 80, 100, 120])

        result = calculator.calculate_max_drawdown(prices)

        # Should identify the largest peak-to-trough decline
        assert isinstance(result, dict)
        assert 'max_drawdown' in result
        assert 'drawdown_series' in result
        assert isinstance(result['drawdown_series'], pd.Series)

    def test_calculate_volatility(self) -> None:
        """Test volatility calculation."""
        calculator = PerformanceCalculator()

        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])

        result = calculator.calculate_volatility(returns)

        assert result >= 0  # Volatility should be non-negative
        assert not np.isnan(result)

    def test_calculate_win_rate(self) -> None:
        """Test win rate calculation."""
        calculator = PerformanceCalculator()

        trades_pnl = pd.Series([100, -50, 200, -30, 150, -20])

        result = calculator.calculate_win_rate(trades_pnl)

        expected_wins = 3  # Positive PNl trades
        expected_rate = expected_wins / len(trades_pnl)

        assert result == expected_rate

    def test_calculate_profit_factor(self) -> None:
        """Test profit factor calculation."""
        calculator = PerformanceCalculator()

        trades_pnl = pd.Series([100, -50, 200, -30, 150, -20])

        result = calculator.calculate_profit_factor(trades_pnl)

        gross_profit = trades_pnl[trades_pnl > 0].sum()  # 450
        gross_loss = abs(trades_pnl[trades_pnl < 0].sum())  # 100
        expected = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        assert result == expected

    def test_calculate_calmar_ratio(self) -> None:
        """Test Calmar ratio calculation."""
        calculator = PerformanceCalculator()

        prices = pd.Series([100, 110, 90, 120, 100, 130])

        max_dd_result = calculator.calculate_max_drawdown(prices)
        annual_return = calculator.calculate_annualized_return(100, 130, 1.5)

        result = calculator.calculate_calmar_ratio(annual_return, max_dd_result['max_drawdown'])

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_beta(self) -> None:
        """Test beta calculation against benchmark."""
        calculator = PerformanceCalculator()

        asset_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        benchmark_returns = pd.Series([0.01, -0.005, 0.02, -0.015, 0.008])

        result = calculator.calculate_beta(asset_returns, benchmark_returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_alpha(self) -> None:
        """Test alpha calculation."""
        calculator = PerformanceCalculator()

        asset_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        benchmark_returns = pd.Series([0.01, -0.005, 0.02, -0.015, 0.008])
        risk_free_rate = 0.01

        result = calculator.calculate_alpha(asset_returns, benchmark_returns, risk_free_rate)

        assert isinstance(result, (int, float))

    def test_calculate_information_ratio(self) -> None:
        """Test information ratio calculation."""
        calculator = PerformanceCalculator()

        asset_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        benchmark_returns = pd.Series([0.01, -0.005, 0.02, -0.015, 0.008])

        result = calculator.calculate_information_ratio(asset_returns, benchmark_returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_var(self) -> None:
        """Test Value at Risk (VaR) calculation."""
        calculator = PerformanceCalculator()

        returns = pd.Series([0.02, -0.05, 0.03, -0.01, 0.04, -0.02, 0.01])

        result = calculator.calculate_var(returns, confidence_level=0.95)

        assert isinstance(result, (int, float))
        assert result < 0  # VaR should be negative (loss)

    def test_calculate_cvar(self) -> None:
        """Test Conditional Value at Risk (CVaR) calculation."""
        calculator = PerformanceCalculator()

        returns = pd.Series([0.02, -0.05, 0.03, -0.01, 0.04, -0.02, 0.01])

        result = calculator.calculate_cvar(returns, confidence_level=0.95)

        assert isinstance(result, (int, float))
        assert result <= 0  # CVaR should be negative or zero


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creation of PerformanceMetrics object."""
        metrics = PerformanceMetrics(
            total_return=0.15, annualized_return=0.12, sharpe_ratio=1.25, max_drawdown=-0.08
        )

        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.12
        assert metrics.sharpe_ratio == 1.25
        assert metrics.max_drawdown == -0.08

    def test_metrics_to_dict(self) -> None:
        """Test conversion of metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            sharpe_ratio=1.25,
            max_drawdown=-0.08,
            win_rate=0.65,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['total_return'] == 0.15
        assert result['sharpe_ratio'] == 1.25

    def test_metrics_from_dict(self) -> None:
        """Test creation of metrics from dictionary."""
        data = {
            'total_return': 0.15,
            'annualized_return': 0.12,
            'sharpe_ratio': 1.25,
            'max_drawdown': -0.08,
        }

        metrics = PerformanceMetrics.from_dict(data)

        assert metrics.total_return == 0.15
        assert metrics.annualized_return == 0.12


class TestRiskMetrics:
    """Test suite for RiskMetrics dataclass."""

    def test_risk_metrics_creation(self) -> None:
        """Test creation of RiskMetrics object."""
        risk_metrics = RiskMetrics(
            volatility=0.18, var_95=-0.05, cvar_95=-0.07, beta=1.1, correlation=0.85
        )

        assert risk_metrics.volatility == 0.18
        assert risk_metrics.var_95 == -0.05
        assert risk_metrics.cvar_95 == -0.07
        assert risk_metrics.beta == 1.1

    def test_risk_metrics_comparison(self) -> None:
        """Test comparison of risk metrics."""
        risk1 = RiskMetrics(volatility=0.15, var_95=-0.04, cvar_95=-0.06, beta=1.0, correlation=0.8)
        risk2 = RiskMetrics(volatility=0.20, var_95=-0.06, cvar_95=-0.08, beta=1.2, correlation=0.9)

        assert risk1.volatility < risk2.volatility
        assert risk1.var_95 > risk2.var_95  # Less negative is better


class TestDrawdownAnalyzer:
    """Test suite for DrawdownAnalyzer class."""

    def test_initialization(self) -> None:
        """Test DrawdownAnalyzer initialization."""
        analyzer = DrawdownAnalyzer()

        assert analyzer.min_drawdown_threshold == -0.20  # Default 20%

    def test_find_drawdown_periods(self) -> None:
        """Test identification of drawdown periods."""
        analyzer = DrawdownAnalyzer()

        prices = pd.Series([100, 110, 90, 95, 85, 100, 120])
        drawdowns = analyzer.calculate_drawdowns(prices)

        periods = analyzer.find_drawdown_periods(drawdowns)

        assert isinstance(periods, list)
        assert len(periods) > 0  # Should identify at least one drawdown

    def test_calculate_drawdown_duration(self) -> None:
        """Test calculation of drawdown duration."""
        analyzer = DrawdownAnalyzer()

        # Create drawdown series
        drawdowns = pd.Series([0, -0.05, -0.08, -0.03, 0, -0.02, 0])

        duration = analyzer.calculate_drawdown_duration(drawdowns)

        assert isinstance(duration, dict)
        assert 'avg_duration' in duration
        assert 'max_duration' in duration

    def test_calculate_recovery_factor(self) -> None:
        """Test recovery factor calculation."""
        analyzer = DrawdownAnalyzer()

        total_return = 0.15
        max_drawdown = -0.08

        result = analyzer.calculate_recovery_factor(total_return, max_drawdown)

        expected = abs(total_return / max_drawdown)
        assert abs(result - expected) < 0.001


class TestReturnAnalyzer:
    """Test suite for ReturnAnalyzer class."""

    def test_initialization(self) -> None:
        """Test ReturnAnalyzer initialization."""
        analyzer = ReturnAnalyzer()

        assert analyzer.compound_returns is True

    def test_calculate_cumulative_returns(self) -> None:
        """Test cumulative returns calculation."""
        analyzer = ReturnAnalyzer()

        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

        result = analyzer.calculate_cumulative_returns(returns)

        assert isinstance(result, pd.Series)
        assert len(result) == len(returns) + 1  # Includes starting value
        assert result.iloc[0] == 1.0  # Starting value should be 1.0

    def test_calculate_rolling_returns(self) -> None:
        """Test rolling returns calculation."""
        analyzer = ReturnAnalyzer()

        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.02])
        window = 3

        result = analyzer.calculate_rolling_returns(returns, window)

        assert isinstance(result, pd.Series)
        # Rolling returns will have NaN values for the first window-1 values, then calculated values
        assert len(result) == len(returns)
        assert (
            result.iloc[window - 1 :].notna().all()
        )  # Values from window onwards should be calculated

    def test_decompose_returns(self) -> None:
        """Test return decomposition."""
        analyzer = ReturnAnalyzer()

        prices = pd.Series([100, 102, 101, 105, 103, 108])

        decomposition = analyzer.decompose_returns(prices)

        assert isinstance(decomposition, dict)
        assert 'cumulative_return' in decomposition
        assert 'volatility' in decomposition
        assert 'skewness' in decomposition
        assert 'kurtosis' in decomposition


class TestSharpeCalculator:
    """Test suite for SharpeCalculator class."""

    def test_initialization(self) -> None:
        """Test SharpeCalculator initialization."""
        calculator = SharpeCalculator(risk_free_rate=0.02)

        assert calculator.risk_free_rate == 0.02

    def test_calculate_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        calculator = SharpeCalculator()

        excess_returns = pd.Series([0.03, -0.01, 0.05, -0.02, 0.04])

        result = calculator.calculate_sharpe_ratio(excess_returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_information_ratio(self) -> None:
        """Test information ratio calculation."""
        calculator = SharpeCalculator()

        active_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])

        result = calculator.calculate_information_ratio(active_returns)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_calculate_treynor_ratio(self) -> None:
        """Test Treynor ratio calculation."""
        calculator = SharpeCalculator()

        portfolio_returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        beta = 1.2

        result = calculator.calculate_treynor_ratio(portfolio_returns, beta)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)


@pytest.mark.parametrize(
    "returns,expected_sharpe",
    [
        ([0.02, -0.01, 0.03, -0.02, 0.01], "calculable"),
        ([0.01] * 10, "infinite"),  # Constant returns should give inf Sharpe due to zero std
        ([-0.05] * 5 + [0.05] * 5, "calculable"),
    ],
)
def test_sharpe_ratio_parametrized(returns: list[float], expected_sharpe: str) -> None:
    """Parametrized test for Sharpe ratio calculations."""
    calculator = SharpeCalculator()
    returns_series = pd.Series(returns)

    result = calculator.calculate_sharpe_ratio(returns_series)

    if expected_sharpe == "calculable":
        assert not np.isnan(result)
        assert np.isfinite(result)
    elif expected_sharpe == "infinite":
        # When std is zero, Sharpe ratio becomes inf, which is expected behavior
        assert np.isinf(result)
    else:
        # This case should not occur with current test data
        raise AssertionError(f"Unexpected expected_sharpe value: {expected_sharpe}")


class TestOperationalMetrics:
    """Tests for operational metrics instrumentation."""

    def test_record_operational_sample_tracks_extremes(self) -> None:
        """Operational samples should update aggregated snapshots."""
        analyzer = PerformanceAnalyzer()
        analyzer.record_operational_sample(
            latency_ms=5.0,
            queue_depth=2,
            events_processed=10,
            throughput_per_second=2000.0,
        )
        analyzer.record_operational_sample(
            latency_ms=8.0,
            queue_depth=5,
            events_processed=5,
            throughput_per_second=1500.0,
        )

        metrics = analyzer.get_operational_metrics()
        assert metrics.samples == 2
        assert metrics.max_latency_ms == pytest.approx(8.0)
        assert metrics.max_queue_depth == 5
        assert metrics.total_events_processed == 15
        assert metrics.avg_latency_ms > 0

    def test_comprehensive_analysis_includes_operational_metrics(self) -> None:
        """Operational metrics should be surfaced in comprehensive results."""
        analyzer = PerformanceAnalyzer()
        analyzer.record_operational_sample(
            latency_ms=1.0,
            queue_depth=1,
            events_processed=1,
            throughput_per_second=100.0,
        )

        prices = pd.Series([100, 101, 102, 103])
        results = analyzer.comprehensive_analysis(prices)

        assert 'operational_metrics' in results
        ops = results['operational_metrics']
        assert ops['samples'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
