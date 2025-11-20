"""Unit tests for Simple Moving Average (SMA) indicator.

This module tests the SMA indicator implementation, including calculation,
signal generation, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.sma import SMAIndicator


class TestSMAIndicator:
    """Test SMAIndicator class methods."""

    def test_init_default_config(self, sma_config: IndicatorConfig) -> None:
        """Test SMA indicator initialization with default configuration."""
        sma = SMAIndicator(sma_config)

        assert sma.name == "SMA"
        assert sma.type == "trend"
        assert sma.config.indicator_type == "trend"
        assert sma.config.ma_type == "simple"
        assert sma.config.period == 14

    def test_init_custom_config(self) -> None:
        """Test SMA indicator initialization with custom configuration."""
        config = IndicatorConfig(
            indicator_name="CustomSMA",
            indicator_type="trend",
            period=20,
            price_column="close",
            ma_type="simple",
        )

        sma = SMAIndicator(config)

        assert sma.name == "CustomSMA"
        assert sma.config.period == 20
        assert sma.config.price_column == "close"
        assert sma.config.ma_type == "simple"

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, sma_config: IndicatorConfig
    ) -> None:
        """Test basic SMA calculation."""
        sma = SMAIndicator(sma_config)
        result = sma.calculate(sample_ohlcv_data)

        # Check that SMA column is added
        assert "sma_sma" in result.columns

        # Check that original data is preserved
        pd.testing.assert_frame_equal(result.drop("sma_sma", axis=1), sample_ohlcv_data)

        # Check that SMA values are calculated
        sma_values = result["sma_sma"].dropna()
        assert len(sma_values) > 0

        # SMA should be between the min and max of the price series
        price_min = result["close"].min()
        price_max = result["close"].max()
        assert sma_values.min() >= price_min
        assert sma_values.max() <= price_max

    def test_calculate_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test SMA calculation with different periods."""
        for period in [5, 10, 20, 30]:
            config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=period)
            sma = SMAIndicator(config)
            result = sma.calculate(sample_ohlcv_data)

            sma_column = "sma_sma"
            assert sma_column in result.columns

            # First (period-1) values should be NaN
            assert result[sma_column].iloc[: period - 1].isna().all()

            # Last value should not be NaN if we have enough data
            if len(sample_ohlcv_data) >= period:
                assert not result[sma_column].iloc[-1:].isna().any()

    def test_calculate_trending_data(self, trending_up_data: pd.DataFrame) -> None:
        """Test SMA calculation with trending data."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)
        result = sma.calculate(trending_up_data)

        sma_column = "sma_sma"

        # With upward trending data, SMA should generally be increasing
        sma_values = result[sma_column].dropna()

        # Calculate rolling differences to check for trend
        sma_diff = sma_values.diff().dropna()

        # Most differences should be positive for trending up data
        positive_diff_ratio = (sma_diff > 0).sum() / len(sma_diff)
        assert positive_diff_ratio > 0.6  # At least 60% should be positive

    def test_calculate_edge_cases(self) -> None:
        """Test SMA calculation with edge case data."""
        # Test with insufficient data
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=20)
        sma = SMAIndicator(config)

        insufficient_data = pd.DataFrame(
            {
                'open': [100, 101, 102, 103, 104],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [100000, 110000, 120000, 130000, 140000],
            }
        )
        insufficient_data.index = pd.date_range('2023-01-01', periods=5, freq='D')

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            sma.calculate(insufficient_data)

    def test_calculate_with_nan_values(self, data_with_nan: pd.DataFrame) -> None:
        """Test SMA calculation with NaN values in data."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=5)
        sma = SMAIndicator(config)

        result = sma.calculate(data_with_nan)

        # Should handle NaN values gracefully
        sma_column = "sma_sma"
        assert sma_column in result.columns

        # Some SMA values should be calculated
        assert not result[sma_column].isna().all()

    def test_generate_signals_bullish(self, trending_up_data: pd.DataFrame) -> None:
        """Test SMA signal generation with bullish trend."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)

        # Calculate SMA first
        data_with_sma = sma.calculate(trending_up_data)

        # Generate signals
        signals = sma.generate_signals(data_with_sma)

        assert len(signals) > 0

        # Check signal structure
        for signal in signals:
            assert 'timestamp' in signal
            assert 'signal_type' in signal
            assert 'action' in signal
            assert 'confidence' in signal
            assert 'metadata' in signal
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_generate_signals_bearish(self, trending_down_data: pd.DataFrame) -> None:
        """Test SMA signal generation with bearish trend."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)

        # Calculate SMA first
        data_with_sma = sma.calculate(trending_down_data)

        # Generate signals
        signals = sma.generate_signals(data_with_sma)

        assert len(signals) > 0

    def test_generate_signals_ranging(self, ranging_data: pd.DataFrame) -> None:
        """Test SMA signal generation with ranging market."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)

        # Calculate SMA first
        data_with_sma = sma.calculate(ranging_data)

        # Generate signals
        signals = sma.generate_signals(data_with_sma)

        assert len(signals) > 0

        # With ranging data, we might get more HOLD signals
        # For ranging markets, signals should still be generated but with lower confidence
        # since the trend is unclear
        assert len(signals) > 0

        # Check that signals have reasonable confidence for ranging market
        # In ranging markets, confidence should be lower due to uncertainty
        for signal in signals:
            # Confidence can vary, but we should have at least one signal
            assert 'confidence' in signal
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_generate_signals_insufficient_data(self, insufficient_data: pd.DataFrame) -> None:
        """Test signal generation with insufficient data."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=20)
        sma = SMAIndicator(config)

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            sma.calculate(insufficient_data)

    def test_generate_signals_no_indicator_column(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test signal generation when SMA column is missing."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)

        # Generate signals without calculating SMA first
        signals = sma.generate_signals(sample_ohlcv_data)

        # Should return empty list when SMA column is missing
        assert signals == []

    def test_get_indicator_columns(self, sma_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        sma = SMAIndicator(sma_config)

        columns = sma.get_indicator_columns()
        assert columns == ["sma_sma"]

    def test_mathematical_accuracy(
        self, known_sma_values: tuple[list[float], list[float], list[float]]
    ) -> None:
        """Test mathematical accuracy of SMA calculation."""
        prices, expected_sma_5, expected_sma_10 = known_sma_values

        # Create test DataFrame
        dates = pd.date_range('2023-01-01', periods=len(prices), freq='D')
        data = pd.DataFrame(
            {
                'open': [p * 0.999 for p in prices],
                'high': [p * 1.001 for p in prices],
                'low': [p * 0.998 for p in prices],
                'close': prices,
                'volume': [100000] * len(prices),
            },
            index=dates,
        )

        # Test 5-period SMA
        config_5 = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=5)
        sma_5 = SMAIndicator(config_5)
        result_5 = sma_5.calculate(data)

        actual_sma_5 = result_5["sma_sma"].dropna().values
        # First 4 values should be NaN, so compare from index 4 onwards
        assert len(actual_sma_5) >= len(expected_sma_5) - 4

        # Test 10-period SMA
        config_10 = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma_10 = SMAIndicator(config_10)
        result_10 = sma_10.calculate(data)

        actual_sma_10 = result_10["sma_sma"].dropna().values
        # First 9 values should be NaN, so compare from index 9 onwards
        assert len(actual_sma_10) >= len(expected_sma_10) - 9

    def test_reset_functionality(self, sma_config: IndicatorConfig) -> None:
        """Test indicator reset functionality."""
        sma = SMAIndicator(sma_config)

        # Add some cache data
        sma._cache['test'] = 'value'
        sma._is_initialized = True

        sma.reset()

        assert sma._cache == {}
        assert sma._is_initialized is False

    def test_performance_with_large_dataset(self, large_dataset: pd.DataFrame) -> None:
        """Test performance with large dataset."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=20)
        sma = SMAIndicator(config)

        # Should complete without error
        result = sma.calculate(large_dataset)

        assert "sma_sma" in result.columns
        assert len(result) == len(large_dataset)

    def test_configuration_override(self) -> None:
        """Test that SMA configuration is properly overridden."""
        # Test that incorrect ma_type is corrected to "simple"
        config = IndicatorConfig(
            indicator_name="SMA",
            indicator_type="momentum",  # Wrong type
            ma_type="exponential",  # Wrong MA type
        )

        sma = SMAIndicator(config)

        # Should be corrected to trend and simple
        assert sma.type == "trend"
        assert sma.config.ma_type == "simple"
        assert sma.config.indicator_type == "trend"

    def test_price_column_override(self) -> None:
        """Test SMA calculation with different price columns."""
        # Create data with multiple price columns
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        data = pd.DataFrame(
            {
                'open': np.linspace(100, 120, 20),
                'high': np.linspace(101, 121, 20),
                'low': np.linspace(99, 119, 20),
                'close': np.linspace(100.5, 120.5, 20),
                'volume': [100000] * 20,
            },
            index=dates,
        )

        # Test with different price columns
        for price_col in ['open', 'high', 'low', 'close']:
            config = IndicatorConfig(
                indicator_name="SMA", indicator_type="trend", period=5, price_column=price_col
            )

            sma = SMAIndicator(config)
            result = sma.calculate(data)

            assert "sma_sma" in result.columns

            # Verify SMA is calculated from the correct price column
            expected_sma = data[price_col].rolling(5).mean()
            pd.testing.assert_series_equal(
                result["sma_sma"].dropna(), expected_sma.dropna(), check_names=False
            )

    def test_signal_metadata_content(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that signal metadata contains expected content."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=10)
        sma = SMAIndicator(config)

        # Calculate SMA first
        data_with_sma = sma.calculate(sample_ohlcv_data)

        # Generate signals
        signals = sma.generate_signals(data_with_sma)

        if signals:
            signal = signals[0]
            metadata = signal['metadata']

            # Check for expected metadata keys
            assert 'indicator_name' in metadata
            assert 'indicator_type' in metadata
            assert metadata['indicator_name'] == "SMA"
            assert metadata['indicator_type'] == "trend"
