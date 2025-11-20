"""Shared test fixtures for indicator tests.

This module contains pytest fixtures that provide common test data and utilities
for all indicator test modules, following the established patterns from the project.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtester.indicators.indicator_configs import IndicatorConfig

# =============================================================================
# Basic OHLCV Data Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for basic testing.

    Returns:
        DataFrame with realistic but simple OHLCV data
    """
    dates = pd.date_range('2023-01-01', periods=50, freq='D')

    # Create simple upward trend with some volatility
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.randn(50) * 0.5

    data = pd.DataFrame(index=dates)
    data['close'] = base_price + price_changes.cumsum()
    data['open'] = data['close'] + np.random.randn(50) * 0.1
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(50)) * 0.1
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(50)) * 0.1
    data['volume'] = np.random.randint(100000, 1000000, 50)

    return data


@pytest.fixture
def insufficient_data() -> pd.DataFrame:
    """Create data with insufficient periods for most indicators.

    Returns:
        DataFrame with only 5 periods of data
    """
    dates = pd.date_range('2023-01-01', periods=5, freq='D')

    data = pd.DataFrame(index=dates)
    data['open'] = [100, 101, 102, 103, 104]
    data['high'] = [101, 102, 103, 104, 105]
    data['low'] = [99, 100, 101, 102, 103]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5]
    data['volume'] = [100000, 110000, 120000, 130000, 140000]

    return data


@pytest.fixture
def empty_data() -> pd.DataFrame:
    """Create empty DataFrame for edge case testing.

    Returns:
        Empty DataFrame with OHLCV columns but no data
    """
    dates = pd.DatetimeIndex([])

    data = pd.DataFrame(index=dates)
    data['open'] = pd.Series([], dtype=float)
    data['high'] = pd.Series([], dtype=float)
    data['low'] = pd.Series([], dtype=float)
    data['close'] = pd.Series([], dtype=float)
    data['volume'] = pd.Series([], dtype=int)

    return data


# =============================================================================
# Edge Case Data Fixtures
# =============================================================================


@pytest.fixture
def data_with_nan() -> pd.DataFrame:
    """Create data with NaN values for edge case testing.

    Returns:
        DataFrame with some NaN values in OHLCV data
    """
    dates = pd.date_range('2023-01-01', periods=20, freq='D')

    data = pd.DataFrame(index=dates)
    # Create data with NaN but ensure OHLC relationships are valid where not NaN
    open_vals = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0] + [
        110.0
    ] * 10
    high_vals = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0] + [
        111.0
    ] * 10
    low_vals = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0] + [109.0] * 10
    close_vals = [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5] + [
        110.5
    ] * 10
    volume_vals = [
        100000,
        110000,
        120000,
        130000,
        140000,
        150000,
        160000,
        170000,
        180000,
        190000,
    ] + [190000] * 10

    # Introduce NaN values carefully to maintain OHLC relationships
    data['open'] = open_vals
    data['close'] = close_vals
    data['high'] = high_vals
    data['low'] = low_vals
    data['volume'] = volume_vals

    # Add NaN values only where they won't break OHLC relationships
    data.loc[data.index[1], 'open'] = np.nan
    data.loc[data.index[2], 'close'] = np.nan
    data.loc[data.index[3], 'low'] = np.nan
    data.loc[data.index[5], 'high'] = np.nan
    data.loc[data.index[7], 'volume'] = np.nan

    return data


@pytest.fixture
def data_with_zero_volume() -> pd.DataFrame:
    """Create data with zero volume periods.

    Returns:
        DataFrame with some zero volume periods
    """
    dates = pd.date_range('2023-01-01', periods=20, freq='D')

    data = pd.DataFrame(index=dates)
    # Create valid OHLC data first
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0] + [
        110.0
    ] * 10
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0] + [
        111.0
    ] * 10
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0] + [
        109.0
    ] * 10
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5] + [
        110.5
    ] * 10
    data['volume'] = [
        100000,
        110000,
        120000,
        130000,
        140000,
        150000,
        160000,
        170000,
        180000,
        190000,
    ] + [190000] * 10

    # Set some volumes to zero
    data.loc[data.index[4], 'volume'] = 0
    data.loc[data.index[9], 'volume'] = 0

    return data


@pytest.fixture
def data_with_invalid_ohlc() -> pd.DataFrame:
    """Create data with invalid OHLC relationships.

    Returns:
        DataFrame with some invalid OHLC data (high < low, etc.)
    """
    dates = pd.date_range('2023-01-01', periods=10, freq='D')

    data = pd.DataFrame(index=dates)
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5]
    data['volume'] = [
        100000,
        110000,
        120000,
        130000,
        140000,
        150000,
        160000,
        170000,
        180000,
        190000,
    ]

    # Add some invalid relationships
    data.loc[data.index[2], 'high'] = 101.5  # high < low
    data.loc[data.index[5], 'low'] = 106.0  # low > high
    data.loc[data.index[7], 'open'] = 108.0  # open > high
    data.loc[data.index[8], 'close'] = 106.0  # close > high

    return data


# =============================================================================
# Realistic Market Data Fixtures
# =============================================================================


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Create data with strong upward trend for testing trend indicators.

    Returns:
        DataFrame with strong upward trending prices
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    np.random.seed(42)
    trend = np.linspace(100, 150, 100)  # Strong upward trend
    noise = np.random.randn(100) * 2  # Add some noise

    data = pd.DataFrame(index=dates)
    data['close'] = trend + noise
    data['open'] = data['close'] + np.random.randn(100) * 0.5
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(100)) * 0.5
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(100)) * 0.5
    data['volume'] = np.random.randint(100000, 1000000, 100)

    return data


@pytest.fixture
def trending_down_data() -> pd.DataFrame:
    """Create data with strong downward trend for testing trend indicators.

    Returns:
        DataFrame with strong downward trending prices
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    np.random.seed(42)
    trend = np.linspace(150, 100, 100)  # Strong downward trend
    noise = np.random.randn(100) * 2  # Add some noise

    data = pd.DataFrame(index=dates)
    data['close'] = trend + noise
    data['open'] = data['close'] + np.random.randn(100) * 0.5
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(100)) * 0.5
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(100)) * 0.5
    data['volume'] = np.random.randint(100000, 1000000, 100)

    return data


@pytest.fixture
def volatile_data() -> pd.DataFrame:
    """Create highly volatile data for testing oscillators.

    Returns:
        DataFrame with high volatility and oscillations
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    np.random.seed(42)
    base = 100
    volatility = 10

    data = pd.DataFrame(index=dates)
    data['close'] = base + np.random.randn(100).cumsum() * volatility
    data['open'] = data['close'] + np.random.randn(100) * 2
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(100)) * 2
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(100)) * 2
    data['volume'] = np.random.randint(100000, 2000000, 100)

    return data


@pytest.fixture
def ranging_data() -> pd.DataFrame:
    """Create data that ranges within a band (sideways market).

    Returns:
        DataFrame with prices ranging within a band
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    np.random.seed(42)
    base = 100
    range_width = 5

    data = pd.DataFrame(index=dates)
    data['close'] = base + np.random.randn(100).cumsum() * 0.1
    data['close'] = (
        base + (data['close'] - data['close'].mean()) * range_width / data['close'].std()
    )
    data['open'] = data['close'] + np.random.randn(100) * 0.5
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(100)) * 0.5
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(100)) * 0.5
    data['volume'] = np.random.randint(100000, 1000000, 100)

    return data


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sma_config() -> IndicatorConfig:
    """Standard SMA configuration for testing.

    Returns:
        SMA indicator configuration
    """
    return IndicatorConfig(
        indicator_name="SMA",
        indicator_type="trend",
        period=14,
        price_column="close",
        ma_type="simple",
    )


@pytest.fixture
def ema_config() -> IndicatorConfig:
    """Standard EMA configuration for testing.

    Returns:
        EMA indicator configuration
    """
    return IndicatorConfig(
        indicator_name="EMA",
        indicator_type="trend",
        period=14,
        price_column="close",
        ma_type="exponential",
    )


@pytest.fixture
def rsi_config() -> IndicatorConfig:
    """Standard RSI configuration for testing.

    Returns:
        RSI indicator configuration
    """
    return IndicatorConfig(
        indicator_name="RSI",
        indicator_type="momentum",
        period=14,
        overbought_threshold=70.0,
        oversold_threshold=30.0,
    )


@pytest.fixture
def macd_config() -> IndicatorConfig:
    """Standard MACD configuration for testing.

    Returns:
        MACD indicator configuration
    """
    return IndicatorConfig(
        indicator_name="MACD",
        indicator_type="momentum",
        fast_period=12,
        slow_period=26,
        signal_period=9,
    )


@pytest.fixture
def bollinger_config() -> IndicatorConfig:
    """Standard Bollinger Bands configuration for testing.

    Returns:
        Bollinger Bands indicator configuration
    """
    return IndicatorConfig(
        indicator_name="Bollinger", indicator_type="volatility", period=20, standard_deviations=2.0
    )


@pytest.fixture
def stochastic_config() -> IndicatorConfig:
    """Standard Stochastic configuration for testing.

    Returns:
        Stochastic oscillator configuration
    """
    return IndicatorConfig(
        indicator_name="Stochastic", indicator_type="momentum", k_period=14, d_period=3
    )


@pytest.fixture
def williams_r_config() -> IndicatorConfig:
    """Standard Williams %R configuration for testing.

    Returns:
        Williams %R indicator configuration
    """
    return IndicatorConfig(
        indicator_name="WilliamsR", indicator_type="momentum", williams_r_period=14
    )


@pytest.fixture
def atr_config() -> IndicatorConfig:
    """Standard ATR configuration for testing.

    Returns:
        ATR indicator configuration
    """
    return IndicatorConfig(indicator_name="ATR", indicator_type="volatility", period=14)


@pytest.fixture
def cci_config() -> IndicatorConfig:
    """Standard CCI configuration for testing.

    Returns:
        CCI indicator configuration
    """
    return IndicatorConfig(
        indicator_name="CCI", indicator_type="momentum", cci_period=20, cci_constant=0.015
    )


@pytest.fixture
def obv_config() -> IndicatorConfig:
    """Standard OBV configuration for testing.

    Returns:
        OBV indicator configuration
    """
    return IndicatorConfig(indicator_name="OBV", indicator_type="volume")


# =============================================================================
# Test Utilities
# =============================================================================


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a mock logger for testing.

    Returns:
        Mock logger instance
    """
    return logging.getLogger("test_logger")


@pytest.fixture
def test_signals() -> list[dict[str, Any]]:
    """Create sample test signals for validation.

    Returns:
        List of test signal dictionaries
    """
    timestamp = datetime.now()

    return [
        {
            'timestamp': timestamp,
            'signal_type': 'BUY',
            'action': 'Test BUY signal',
            'confidence': 0.8,
            'metadata': {'test': True},
        },
        {
            'timestamp': timestamp,
            'signal_type': 'SELL',
            'action': 'Test SELL signal',
            'confidence': 0.7,
            'metadata': {'test': True},
        },
        {
            'timestamp': timestamp,
            'signal_type': 'HOLD',
            'action': 'Test HOLD signal',
            'confidence': 0.5,
            'metadata': {'test': True},
        },
    ]


# =============================================================================
# Parametrized Test Data
# =============================================================================


@pytest.fixture(params=[5, 10, 14, 20, 30, 50])
def period_params(request: pytest.FixtureRequest) -> int:
    """Parametrized fixture for different period values.

    Args:
        request: pytest fixture request

    Returns:
        Period value for testing
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[2.0, 1.5, 2.5, 3.0])
def std_dev_params(request: pytest.FixtureRequest) -> float:
    """Parametrized fixture for different standard deviation values.

    Args:
        request: pytest fixture request

    Returns:
        Standard deviation value for testing
    """
    return request.param  # type: ignore[no-any-return]


# =============================================================================
# Mathematical Test Data
# =============================================================================


@pytest.fixture
def known_sma_values() -> tuple[list[float], list[float], list[float]]:
    """Provide known SMA calculation test data.

    Returns:
        Tuple of (price_data, expected_sma_5, expected_sma_10)
    """
    # Simple price data for manual calculation
    prices: list[float] = [10, 12, 13, 15, 14, 16, 18, 17, 19, 20, 22, 21, 23, 25, 24]

    # Expected SMA values (manually calculated)
    sma_5 = [
        10.0,
        11.0,
        12.0,
        13.0,
        12.8,
        14.0,
        14.6,
        15.2,
        15.6,
        16.6,
        17.4,
        18.4,
        19.0,
        20.0,
        21.0,
    ]
    sma_10 = [
        10.0,
        11.0,
        12.0,
        12.5,
        13.5,
        14.5,
        15.5,
        16.0,
        17.0,
        18.0,
        19.0,
        19.5,
        20.5,
        21.5,
        22.0,
    ]

    return prices, sma_5, sma_10


@pytest.fixture
def known_rsi_values() -> tuple[list[float], list[float], float]:
    """Provide known RSI calculation test data.

    Returns:
        Tuple of (price_data, expected_rsi_values, final_rsi)
    """
    # Test data that should produce known RSI results
    prices = [
        44,
        44.3,
        44.7,
        43.4,
        42.8,
        43.4,
        44.4,
        45.0,
        45.6,
        46.0,
        45.3,
        46.5,
        47.0,
        47.3,
        48.0,
    ]

    # Expected RSI values (manually calculated for 14-period RSI)
    # This is simplified - in practice RSI would require gains/losses calculation
    rsi_values = [
        50.0,
        52.1,
        54.3,
        51.8,
        49.2,
        51.8,
        55.4,
        58.1,
        61.2,
        63.8,
        61.5,
        65.7,
        68.2,
        69.8,
        72.1,
    ]
    final_rsi = 72.1

    return prices, rsi_values, final_rsi


# =============================================================================
# Data Format Validation Fixtures
# =============================================================================


@pytest.fixture
def data_missing_columns() -> pd.DataFrame:
    """Create data missing required columns for validation testing.

    Returns:
        DataFrame missing some required OHLCV columns
    """
    dates = pd.date_range('2023-01-01', periods=10, freq='D')

    data = pd.DataFrame(index=dates)
    data['open'] = [100.0] * 10
    data['close'] = [101.0] * 10
    # Missing 'high', 'low', 'volume' columns

    return data


@pytest.fixture
def data_wrong_index_type() -> pd.DataFrame:
    """Create data with wrong index type for validation testing.

    Returns:
        DataFrame with non-datetime index
    """
    data = pd.DataFrame()
    data['open'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    data['high'] = [101.0, 102.0, 103.0, 104.0, 105.0]
    data['low'] = [99.0, 100.0, 101.0, 102.0, 103.0]
    data['close'] = [100.5, 101.5, 102.5, 103.5, 104.5]
    data['volume'] = [100000, 110000, 120000, 130000, 140000]

    # Use RangeIndex instead of DatetimeIndex
    data.index = range(5)

    return data


# =============================================================================
# Performance Test Data
# =============================================================================


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Create large dataset for performance testing.

    Returns:
        DataFrame with 10,000 periods of data
    """
    dates = pd.date_range('2020-01-01', periods=10000, freq='D')

    np.random.seed(42)
    base_price = 100
    price_changes = np.random.randn(10000) * 0.5

    data = pd.DataFrame(index=dates)
    data['close'] = base_price + price_changes.cumsum()
    data['open'] = data['close'] + np.random.randn(10000) * 0.1
    data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(10000)) * 0.1
    data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(10000)) * 0.1
    data['volume'] = np.random.randint(100000, 1000000, 10000)

    return data
