"""Shared test fixtures and data for the backtester test suite.

This module provides reusable fixtures, mock data, and helper utilities
that are used across multiple test modules.
"""

import json
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


# Sample market data for testing
@pytest.fixture(scope="session")
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)  # For reproducible tests

    # Generate realistic OHLCV data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = initial_price * np.cumprod(1 + returns)

    # Create OHLCV data
    data = pd.DataFrame(
        {
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture(scope="session")
def multiple_symbol_data() -> dict[str, pd.DataFrame]:
    """Generate sample data for multiple symbols."""
    symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']
    symbol_data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol

        # Different base prices for different symbols
        base_prices = {'SPY': 300, 'AAPL': 150, 'GOOGL': 2500, 'MSFT': 200, 'TSLA': 600}
        initial_price = base_prices.get(symbol, 100)

        returns = np.random.normal(0.001, 0.025, len(dates))
        prices = initial_price * np.cumprod(1 + returns)

        data = pd.DataFrame(
            {
                'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'Close': prices,
                'Volume': np.random.randint(500000, 5000000, len(dates)),
            },
            index=dates,
        )

        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

        symbol_data[symbol] = data

    return symbol_data


@pytest.fixture
def portfolio_test_data() -> dict[str, Any]:
    """Generate test data for portfolio testing."""
    return {
        'initial_capital': 10000.0,
        'positions': {
            'SPY': {'quantity': 10, 'avg_price': 400.0},
            'AAPL': {'quantity': 5, 'avg_price': 150.0},
            'GOOGL': {'quantity': 2, 'avg_price': 2500.0},
        },
        'cash': 1000.0,
        'commission_rate': 0.001,
    }


@pytest.fixture
def strategy_test_config() -> dict[str, Any]:
    """Generate test configuration for strategies."""
    return {
        'moving_average': {
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'sma',
            'signal_threshold': 0.02,
        },
        'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
        'bollinger_bands': {'period': 20, 'std_dev': 2.0},
    }


@pytest.fixture
def risk_control_test_config() -> dict[str, Any]:
    """Generate test configuration for risk controls."""
    return {
        'stop_loss': {
            'stop_loss_type': 'PERCENTAGE',
            'stop_loss_value': 0.025,
            'trailing_stop_pct': 0.05,
        },
        'take_profit': {
            'take_profit_type': 'PERCENTAGE',
            'take_profit_value': 0.10,
            'trailing_take_profit_pct': 0.03,
        },
        'max_drawdown_limit': 0.15,
        'position_size_limit': 0.10,
    }


@pytest.fixture
def performance_test_metrics() -> dict[str, Any]:
    """Generate sample performance metrics for testing."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")

    # Generate portfolio value series
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))
    portfolio_values = 10000 * np.cumprod(1 + returns)

    # Generate benchmark returns
    benchmark_returns = np.random.normal(0.0008, 0.012, len(dates))
    benchmark_values = 10000 * np.cumprod(1 + benchmark_returns)

    return {
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark_values,
        'returns': returns,
        'benchmark_returns': benchmark_returns,
        'trades': generate_test_trades_data(),
        'positions': generate_test_positions_data(),
    }


def generate_test_trades_data() -> pd.DataFrame:
    """Generate test trades data."""
    return pd.DataFrame(
        {
            'timestamp': pd.date_range(start="2020-01-01", periods=50, freq="W"),
            'symbol': ['SPY'] * 25 + ['AAPL'] * 15 + ['GOOGL'] * 10,
            'action': ['buy', 'sell'] * 25,
            'quantity': np.random.randint(10, 100, 50),
            'price': np.random.uniform(100, 500, 50),
            'pnl': np.random.uniform(-500, 800, 50),
            'commission': np.random.uniform(1, 10, 50),
        }
    )


def generate_test_positions_data() -> pd.DataFrame:
    """Generate test positions data."""
    symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']
    return pd.DataFrame(
        {
            'symbol': symbols,
            'quantity': [10, 5, 2, 8, 3],
            'avg_price': [400.0, 150.0, 2500.0, 200.0, 600.0],
            'current_price': [405.0, 152.0, 2525.0, 198.0, 615.0],
            'market_value': [4050.0, 760.0, 5050.0, 1584.0, 1845.0],
            'unrealized_pnl': [50.0, 10.0, 50.0, -16.0, 45.0],
        }
    )


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_data_provider() -> Mock:
    """Create a mock data provider."""
    provider = Mock()
    provider.get_data = Mock(
        return_value=pd.DataFrame(
            {
                'Close': [100, 101, 102, 103, 104],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )
    )
    provider.validate_data = Mock(return_value=True)
    provider.get_metadata = Mock(return_value={'symbol': 'SPY', 'currency': 'USD'})
    return provider


@pytest.fixture
def mock_broker() -> Mock:
    """Create a mock broker."""
    broker = Mock()
    broker.submit_order = Mock(
        return_value={
            'order_id': 'test_order_123',
            'status': 'filled',
            'fill_price': 400.0,
            'filled_quantity': 10,
            'commission': 4.0,
        }
    )
    broker.cancel_order = Mock(return_value={'status': 'cancelled'})
    broker.get_account_info = Mock(
        return_value={'cash': 9000.0, 'buying_power': 18000.0, 'positions': {}}
    )
    broker.get_order_status = Mock(return_value={'status': 'filled'})
    return broker


@pytest.fixture
def mock_strategy() -> Mock:
    """Create a mock strategy."""
    strategy = Mock()
    strategy.generate_signals = Mock(
        return_value=pd.Series([0, 1, 0, -1, 1], index=pd.date_range('2020-01-01', periods=5))
    )
    strategy.validate_parameters = Mock(return_value=True)
    strategy.optimize_parameters = Mock(
        return_value={
            'best_params': {'fast_period': 10, 'slow_period': 20},
            'best_score': 1.25,
            'optimization_details': {},
        }
    )
    strategy.backtest = Mock(
        return_value={
            'signals': pd.Series([0, 1, 0, -1, 1]),
            'trades': pd.DataFrame(),
            'performance': {'total_return': 0.15, 'sharpe_ratio': 1.25},
        }
    )
    return strategy


@pytest.fixture
def mock_portfolio() -> Mock:
    """Create a mock portfolio."""
    portfolio = Mock()
    portfolio.add_position = Mock(return_value=True)
    portfolio.close_position = Mock(return_value={'pnl': 100.0, 'commission': 2.0})
    portfolio.get_total_value = Mock(return_value=10500.0)
    portfolio.get_allocation = Mock(return_value={'SPY': 0.40, 'AAPL': 0.25, 'CASH': 0.35})
    portfolio.rebalance = Mock(return_value={'trades': [], 'new_allocation': {}})
    portfolio.can_add_position = Mock(return_value=True)
    return portfolio


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        'backtest': {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'slippage': 0.0005,
            'leverage': 1.0,
        },
        'data': {
            'source': 'yahoo',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'interval': '1d',
        },
        'strategy': {
            'name': 'moving_average',
            'parameters': {'fast_period': 10, 'slow_period': 20, 'ma_type': 'sma'},
        },
        'risk_management': {
            'stop_loss': 0.025,
            'take_profit': 0.10,
            'max_positions': 10,
            'position_size_limit': 0.20,
        },
        'performance': {'benchmark': 'SPY', 'risk_free_rate': 0.02, 'target_return': 0.12},
    }


@pytest.fixture
def benchmark_data() -> pd.Series:
    """Generate benchmark data for comparison testing."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    np.random.seed(123)  # Different seed for benchmark

    # Generate S&P 500-like returns
    returns = np.random.normal(0.0008, 0.012, len(dates))
    benchmark_values = 100 * np.cumprod(1 + returns)

    return pd.Series(benchmark_values, index=dates)


@pytest.fixture
def economic_indicators() -> pd.DataFrame:
    """Generate sample economic indicators."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="M")

    return pd.DataFrame(
        {
            'GDP_growth': np.random.normal(0.02, 0.01, len(dates)),
            'inflation_rate': np.random.normal(0.025, 0.005, len(dates)),
            'unemployment_rate': np.random.normal(0.04, 0.01, len(dates)),
            'fed_funds_rate': np.random.normal(0.02, 0.015, len(dates)),
            'VIX': np.random.normal(20, 5, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sector_data() -> pd.DataFrame:
    """Generate sample sector performance data."""
    return pd.DataFrame(
        {
            'Technology': [100, 105, 98, 110, 115],
            'Healthcare': [100, 102, 103, 101, 106],
            'Financial': [100, 97, 95, 100, 103],
            'Energy': [100, 90, 85, 95, 100],
            'Consumer': [100, 103, 105, 107, 109],
        },
        index=pd.date_range('2020-01-01', periods=5, freq='Q'),
    )


@pytest.fixture
def news_sentiment_data() -> pd.DataFrame:
    """Generate sample news sentiment data."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")

    return pd.DataFrame(
        {
            'sentiment_score': np.random.normal(0, 0.5, len(dates)),
            'news_volume': np.random.randint(10, 100, len(dates)),
            'sentiment_volatility': np.random.uniform(0.1, 0.8, len(dates)),
        },
        index=dates,
    )


class TestDataGenerator:
    """Helper class for generating test data."""

    @staticmethod
    def create_trending_data(
        length: int = 100, start_price: float = 100, trend: float = 0.001, volatility: float = 0.02
    ) -> pd.Series:
        """Create trending price data."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=length, freq="D")

        returns = np.random.normal(trend, volatility, length)
        prices = start_price * np.cumprod(1 + returns)

        return pd.Series(prices, index=dates)

    @staticmethod
    def create_volatile_data(
        length: int = 100, start_price: float = 100, volatility: float = 0.05
    ) -> pd.Series:
        """Create volatile price data."""
        np.random.seed(123)
        dates = pd.date_range(start="2020-01-01", periods=length, freq="D")

        returns = np.random.normal(0, volatility, length)
        prices = start_price * np.cumprod(1 + returns)

        return pd.Series(prices, index=dates)

    @staticmethod
    def create_sideways_data(
        length: int = 100, price_range: tuple[float, float] = (90, 110)
    ) -> pd.Series:
        """Create sideways/consolidating price data."""
        np.random.seed(456)
        dates = pd.date_range(start="2020-01-01", periods=length, freq="D")

        min_price, max_price = price_range
        prices: list[float] = []
        current_price = (min_price + max_price) / 2

        for _ in range(length):
            change = np.random.uniform(-2, 2)
            current_price += change
            current_price = max(min_price, min(max_price, current_price))
            prices.append(current_price)

        return pd.Series(prices, index=dates)

    @staticmethod
    def create_synthetic_ohlcv(length: int = 100, start_price: float = 100) -> pd.DataFrame:
        """Create synthetic OHLCV data."""
        np.random.seed(789)
        dates = pd.date_range(start="2020-01-01", periods=length, freq="D")

        returns = np.random.normal(0.001, 0.02, length)
        close_prices = start_price * np.cumprod(1 + returns)

        data = pd.DataFrame(index=dates)

        # Generate OHLC from closes
        for i, close in enumerate(close_prices):
            daily_range = abs(np.random.normal(0, close * 0.01))

            if i == 0:
                open_price = close
            else:
                open_price = close_prices[i - 1] + np.random.normal(0, close * 0.005)

            high = max(open_price, close) + daily_range * np.random.uniform(0.3, 1.0)
            low = min(open_price, close) - daily_range * np.random.uniform(0.3, 1.0)
            volume = np.random.randint(500000, 2000000)

            data.loc[dates[i]] = {
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
            }

        return data


class ConfigFactory:
    """Factory for creating test configurations."""

    @staticmethod
    def create_backtest_config(**overrides: Any) -> dict[str, Any]:
        """Create a backtest configuration."""
        config: dict[str, Any] = {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'slippage': 0.0005,
            'leverage': 1.0,
            'max_positions': 10,
            'position_size_limit': 0.20,
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_strategy_config(
        strategy_name: str = 'moving_average', **overrides: Any
    ) -> dict[str, Any]:
        """Create a strategy configuration."""
        base_configs: dict[str, dict[str, Any]] = {
            'moving_average': {
                'fast_period': 10,
                'slow_period': 20,
                'ma_type': 'sma',
                'signal_threshold': 0.02,
            },
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'bollinger_bands': {'period': 20, 'std_dev': 2.0},
        }

        config: dict[str, Any] = base_configs.get(strategy_name, {})
        config.update(overrides)
        return config

    @staticmethod
    def create_risk_config(**overrides: Any) -> dict[str, Any]:
        """Create a risk management configuration."""
        config = {
            'stop_loss': 0.025,
            'take_profit': 0.10,
            'max_drawdown_limit': 0.15,
            'position_size_limit': 0.10,
            'max_leverage': 2.0,
        }
        config.update(overrides)
        return config


class MockMarketData:
    """Class for creating mock market data scenarios."""

    @staticmethod
    def bull_market() -> pd.Series:
        """Create bull market conditions."""
        dates = pd.date_range(start="2020-03-01", end="2021-12-31", freq="D")
        np.random.seed(42)

        returns = np.random.normal(0.0015, 0.015, len(dates))  # Positive bias
        prices = 100 * np.cumprod(1 + np.array(returns))

        return pd.Series(prices, index=dates)

    @staticmethod
    def bear_market() -> pd.Series:
        """Create bear market conditions."""
        dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        np.random.seed(123)

        returns = np.random.normal(-0.001, 0.02, len(dates))  # Negative bias
        prices = 400 * np.cumprod(1 + returns)

        return pd.Series(prices, index=dates)

    @staticmethod
    def crisis_data() -> pd.Series:
        """Create financial crisis-like data."""
        dates = pd.date_range(start="2008-09-01", end="2009-03-31", freq="D")
        np.random.seed(456)

        # Create a sharp drop followed by gradual recovery
        returns: list[float] = []
        for i in range(len(dates)):
            if i < len(dates) // 3:
                # Crash phase
                returns.append(float(np.random.normal(-0.03, 0.04)))
            else:
                # Recovery phase
                returns.append(float(np.random.normal(0.01, 0.025)))

        prices = 100 * np.cumprod(1 + np.array(returns))

        return pd.Series(prices, index=dates)


# Utility functions for test data comparison
def assert_dataframe_equal(
    df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False, check_exact: bool = True
) -> None:
    """Assert that two DataFrames are equal."""
    assert isinstance(df1, pd.DataFrame), "First argument must be DataFrame"
    assert isinstance(df2, pd.DataFrame), "Second argument must be DataFrame"
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    assert list(df1.columns) == list(
        df2.columns
    ), f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}"

    if check_dtype:
        assert df1.dtypes.equals(df2.dtypes), "Data type mismatch"

    if check_exact:
        assert df1.equals(df2), "DataFrame values are not equal"
    else:
        pd.testing.assert_frame_equal(df1, df2, rtol=1e-10, atol=1e-10)


def assert_series_equal(s1: pd.Series, s2: pd.Series, check_dtype: bool = False) -> None:
    """Assert that two Series are equal."""
    assert isinstance(s1, pd.Series), "First argument must be Series"
    assert isinstance(s2, pd.Series), "Second argument must be Series"
    assert s1.shape == s2.shape, f"Shape mismatch: {s1.shape} vs {s2.shape}"

    if check_dtype:
        assert s1.dtype == s2.dtype, "Data type mismatch"

    pd.testing.assert_series_equal(s1, s2, rtol=1e-10, atol=1e-10)


def create_test_config_file(config_dict: dict[str, Any], file_path: str) -> None:
    """Create a test configuration file."""
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_test_config_file(file_path: str) -> dict[str, Any]:
    """Load a test configuration file."""
    with open(file_path) as f:
        result = json.load(f)
        assert isinstance(result, dict)
        return result


# Pytest markers for test categorization
pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.integration, pytest.mark.performance]
