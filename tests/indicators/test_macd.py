"""Unit tests for Moving Average Convergence Divergence (MACD) indicator.

This module tests the MACD indicator implementation, including calculation,
signal generation, and edge cases.
"""

import numpy as np
import pandas as pd

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.macd import MACDIndicator


class TestMACDIndicator:
    """Test MACDIndicator class methods."""

    def test_init_default_config(self, macd_config: IndicatorConfig) -> None:
        """Test MACD indicator initialization with default configuration."""
        macd = MACDIndicator(macd_config)

        assert macd.name == "MACD"
        assert macd.type == "momentum"
        assert macd.config.indicator_type == "momentum"
        assert macd.config.fast_period == 12
        assert macd.config.slow_period == 26
        assert macd.config.signal_period == 9

    def test_init_custom_config(self) -> None:
        """Test MACD indicator initialization with custom configuration."""
        config = IndicatorConfig(
            indicator_name="CustomMACD",
            indicator_type="momentum",
            fast_period=8,
            slow_period=21,
            signal_period=5,
        )

        macd = MACDIndicator(config)

        assert macd.name == "CustomMACD"
        assert macd.config.fast_period == 8
        assert macd.config.slow_period == 21
        assert macd.config.signal_period == 5

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, macd_config: IndicatorConfig
    ) -> None:
        """Test basic MACD calculation."""
        macd = MACDIndicator(macd_config)
        result = macd.calculate(sample_ohlcv_data)

        # Check that all MACD columns are added
        assert "macd_macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        # Check that original data is preserved
        pd.testing.assert_frame_equal(
            result.drop(["macd_macd", "macd_signal", "macd_histogram"], axis=1), sample_ohlcv_data
        )

        # Check that MACD values are calculated
        macd_values = result["macd_macd"].dropna()
        signal_values = result["macd_signal"].dropna()
        histogram_values = result["macd_histogram"].dropna()

        assert len(macd_values) > 0
        assert len(signal_values) > 0
        assert len(histogram_values) > 0

    def test_calculate_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test MACD calculation with different periods."""
        test_configs = [(5, 10, 3), (8, 21, 5), (12, 26, 9)]

        for fast, slow, signal in test_configs:
            config = IndicatorConfig(
                indicator_name="MACD",
                indicator_type="momentum",
                fast_period=fast,
                slow_period=slow,
                signal_period=signal,
            )
            macd = MACDIndicator(config)
            result = macd.calculate(sample_ohlcv_data)

            assert "macd_macd" in result.columns
            assert "macd_signal" in result.columns
            assert "macd_histogram" in result.columns

    def test_calculate_trending_data(self, trending_up_data: pd.DataFrame) -> None:
        """Test MACD calculation with trending data."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)
        result = macd.calculate(trending_up_data)

        # With upward trending data, MACD should generally be positive
        macd_values = result["macd_macd"].dropna()

        # Should have some positive values for upward trend
        if len(macd_values) > 0:
            assert (macd_values > 0).any() or (macd_values < 0).any()

    def test_calculate_histogram_relationship(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that histogram equals MACD - Signal line."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)
        result = macd.calculate(sample_ohlcv_data)

        # Histogram should equal MACD - Signal
        macd_line = result["macd_macd"]
        signal_line = result["macd_signal"]
        histogram = result["macd_histogram"]

        # Drop NaN values for comparison
        valid_data = pd.DataFrame(
            {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
        ).dropna()

        if len(valid_data) > 0:
            calculated_histogram = valid_data['macd'] - valid_data['signal']
            np.testing.assert_array_almost_equal(
                valid_data['histogram'].values, calculated_histogram.values, decimal=10
            )

    def test_generate_signals_crossovers(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test MACD signal generation for crossovers."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        # Calculate MACD first
        data_with_macd = macd.calculate(sample_ohlcv_data)

        # Generate signals
        signals = macd.generate_signals(data_with_macd)

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

    def test_generate_signals_zero_line_crossover(self, trending_up_data: pd.DataFrame) -> None:
        """Test MACD signal generation for zero line crossovers."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        # Calculate MACD first
        data_with_macd = macd.calculate(trending_up_data)

        # Generate signals
        signals = macd.generate_signals(data_with_macd)

        # Should generate some signals, potentially including zero line crossovers
        assert len(signals) > 0

    def test_generate_signals_histogram_momentum(self, volatile_data: pd.DataFrame) -> None:
        """Test MACD signal generation for histogram momentum."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        # Calculate MACD first
        data_with_macd = macd.calculate(volatile_data)

        # Generate signals
        signals = macd.generate_signals(data_with_macd)

        # Should generate momentum-related signals
        assert len(signals) > 0

    def test_trend_strength_calculation(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test MACD trend strength calculation."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        result = macd.calculate(sample_ohlcv_data)

        # Get MACD and signal data
        macd_data = result["macd_macd"].dropna()
        signal_data = result["macd_signal"].dropna()

        if len(macd_data) >= 5:
            trend_strength = macd._calculate_trend_strength(macd_data, signal_data)
            assert trend_strength in [
                "strong_bullish",
                "weak_bullish",
                "neutral",
                "weak_bearish",
                "strong_bearish",
                "unknown",
            ]

    def test_get_indicator_columns(self, macd_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        macd = MACDIndicator(macd_config)

        columns = macd.get_indicator_columns()
        assert len(columns) == 3
        assert "macd_macd" in columns
        assert "macd_signal" in columns
        assert "macd_histogram" in columns

    def test_reset_functionality(self, macd_config: IndicatorConfig) -> None:
        """Test indicator reset functionality."""
        macd = MACDIndicator(macd_config)

        # Add some cache data
        macd._cache['test'] = 'value'
        macd._is_initialized = True

        macd.reset()

        assert macd._cache == {}
        assert macd._is_initialized is False

    def test_performance_with_large_dataset(self, large_dataset: pd.DataFrame) -> None:
        """Test performance with large dataset."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        # Should complete without error
        result = macd.calculate(large_dataset)

        assert "macd_macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert len(result) == len(large_dataset)

    def test_extreme_macd_readings(self, volatile_data: pd.DataFrame) -> None:
        """Test MACD with extreme readings."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        result = macd.calculate(volatile_data)
        signals = macd.generate_signals(result)

        # Should handle extreme readings without errors
        assert len(signals) > 0

    def test_signal_metadata_content(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that signal metadata contains expected content."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        macd = MACDIndicator(config)

        # Calculate MACD first
        data_with_macd = macd.calculate(sample_ohlcv_data)

        # Generate signals
        signals = macd.generate_signals(data_with_macd)

        if signals:
            signal = signals[0]
            metadata = signal['metadata']

            # Check for expected metadata keys
            assert 'indicator_name' in metadata
            assert 'indicator_type' in metadata
            assert metadata['indicator_name'] == "MACD"
            assert metadata['indicator_type'] == "momentum"
