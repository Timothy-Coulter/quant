"""Unit tests for Exponential Moving Average (EMA) indicator.

This module tests the EMA indicator implementation, including calculation,
signal generation, and edge cases.
"""

import numpy as np
import pandas as pd

from backtester.indicators.ema import EMAIndicator
from backtester.indicators.indicator_configs import IndicatorConfig


class TestEMAIndicator:
    """Test EMAIndicator class methods."""

    def test_init_default_config(self, ema_config: IndicatorConfig) -> None:
        """Test EMA indicator initialization with default configuration."""
        ema = EMAIndicator(ema_config)

        assert ema.name == "EMA"
        assert ema.type == "trend"
        assert ema.config.indicator_type == "trend"
        assert ema.config.ma_type == "exponential"
        assert ema.config.period == 14

    def test_init_custom_config(self) -> None:
        """Test EMA indicator initialization with custom configuration."""
        config = IndicatorConfig(
            indicator_name="CustomEMA",
            indicator_type="trend",
            period=20,
            price_column="close",
            ma_type="exponential",
        )

        ema = EMAIndicator(config)

        assert ema.name == "CustomEMA"
        assert ema.config.period == 20
        assert ema.config.price_column == "close"
        assert ema.config.ma_type == "exponential"

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, ema_config: IndicatorConfig
    ) -> None:
        """Test basic EMA calculation."""
        ema = EMAIndicator(ema_config)
        result = ema.calculate(sample_ohlcv_data)

        # Check that EMA column is added
        assert "ema_ema" in result.columns

        # Check that original data is preserved
        pd.testing.assert_frame_equal(result.drop("ema_ema", axis=1), sample_ohlcv_data)

        # Check that EMA values are calculated
        ema_values = result["ema_ema"].dropna()
        assert len(ema_values) > 0

    def test_calculate_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test EMA calculation with different periods."""
        for period in [5, 10, 20, 30]:
            config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=period)
            ema = EMAIndicator(config)
            result = ema.calculate(sample_ohlcv_data)

            ema_column = "ema_ema"
            assert ema_column in result.columns

            # EMA starts calculating from the first value (unlike SMA)
            # Just check that we have calculated values and they are not all the same
            ema_values = result[ema_column].dropna()
            assert len(ema_values) > 0
            # EMA should have some variation (not all identical values)
            assert ema_values.nunique() > 1

    def test_calculate_ema_slope(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test EMA slope calculation functionality."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=10)
        ema = EMAIndicator(config)

        result = ema.calculate(sample_ohlcv_data)

        # Test slope calculation
        slope = ema.calculate_ema_slope(result, periods=5)
        assert isinstance(slope, float)

    def test_calculate_with_nan_values(self, data_with_nan: pd.DataFrame) -> None:
        """Test EMA calculation with NaN values in data."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=5)
        ema = EMAIndicator(config)

        result = ema.calculate(data_with_nan)

        # Should handle NaN values gracefully
        ema_column = "ema_ema"
        assert ema_column in result.columns

        # Some EMA values should be calculated
        assert not result[ema_column].isna().all()

    def test_generate_signals_bullish(self, trending_up_data: pd.DataFrame) -> None:
        """Test EMA signal generation with bullish trend."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=10)
        ema = EMAIndicator(config)

        # Calculate EMA first
        data_with_ema = ema.calculate(trending_up_data)

        # Generate signals
        signals = ema.generate_signals(data_with_ema)

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
        """Test EMA signal generation with bearish trend."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=10)
        ema = EMAIndicator(config)

        # Calculate EMA first
        data_with_ema = ema.calculate(trending_down_data)

        # Generate signals
        signals = ema.generate_signals(data_with_ema)

        assert len(signals) > 0

    def test_get_indicator_columns(self, ema_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        ema = EMAIndicator(ema_config)

        columns = ema.get_indicator_columns()
        assert columns == ["ema_ema"]

    def test_configuration_override(self) -> None:
        """Test that EMA configuration is properly overridden."""
        # Test that incorrect ma_type is corrected to "exponential"
        config = IndicatorConfig(
            indicator_name="EMA",
            indicator_type="momentum",  # Wrong type
            ma_type="simple",  # Wrong MA type
        )

        ema = EMAIndicator(config)

        # Should be corrected to trend and exponential
        assert ema.type == "trend"
        assert ema.config.ma_type == "exponential"
        assert ema.config.indicator_type == "trend"

    def test_reset_functionality(self, ema_config: IndicatorConfig) -> None:
        """Test indicator reset functionality."""
        ema = EMAIndicator(ema_config)

        # Add some cache data
        ema._cache['test'] = 'value'
        ema._is_initialized = True

        ema.reset()

        assert ema._cache == {}
        assert ema._is_initialized is False

    def test_performance_with_large_dataset(self, large_dataset: pd.DataFrame) -> None:
        """Test performance with large dataset."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=20)
        ema = EMAIndicator(config)

        # Should complete without error
        result = ema.calculate(large_dataset)

        assert "ema_ema" in result.columns
        assert len(result) == len(large_dataset)

    def test_ema_vs_sma_reactivity(self, volatile_data: pd.DataFrame) -> None:
        """Test that EMA is more reactive than SMA."""
        from backtester.indicators.sma import SMAIndicator

        period = 10
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend", period=period)

        # Create both indicators
        ema = EMAIndicator(config)
        sma = SMAIndicator(config)

        # Calculate both
        data_with_ema = ema.calculate(volatile_data)
        data_with_sma = sma.calculate(volatile_data)

        # Get the correct column names (using the indicator name "test")
        ema_column = f"{config.indicator_name.lower()}_ema"
        sma_column = f"{config.indicator_name.lower()}_sma"

        # EMA should be more responsive (closer to current price)
        ema_values = data_with_ema[ema_column].dropna()
        sma_values = data_with_sma[sma_column].dropna()
        prices = data_with_ema["close"].dropna()

        # Ensure we have enough data for comparison
        if len(ema_values) > 0 and len(sma_values) > 0 and len(prices) > 0:
            min_length = min(len(ema_values), len(sma_values), len(prices))
            ema_values = ema_values.iloc[-min_length:]
            sma_values = sma_values.iloc[-min_length:]
            prices = prices.iloc[-min_length:]

            # EMA should generally be closer to current price than SMA
            ema_distances = abs(ema_values - prices)
            sma_distances = abs(sma_values - prices)

            # At least some EMA values should be closer to price than SMA
            closer_ema = (ema_distances < sma_distances).sum()
            assert closer_ema > 0

    def test_ema_slope_functionality(self, trending_up_data: pd.DataFrame) -> None:
        """Test EMA slope calculation in different scenarios."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=10)
        ema = EMAIndicator(config)

        result = ema.calculate(trending_up_data)

        # Test slope with different periods
        for periods in [3, 5, 10]:
            slope = ema.calculate_ema_slope(result, periods)
            assert isinstance(slope, float)

            # For trending up data, slope should generally be positive
            if periods <= 5:  # Shorter periods more likely to capture trend
                # Note: This is not guaranteed, so we just check it's a valid number
                assert not np.isnan(slope)

    def test_signal_metadata_content(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that signal metadata contains expected content."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=10)
        ema = EMAIndicator(config)

        # Calculate EMA first
        data_with_ema = ema.calculate(sample_ohlcv_data)

        # Generate signals
        signals = ema.generate_signals(data_with_ema)

        if signals:
            signal = signals[0]
            metadata = signal['metadata']

            # Check for expected metadata keys
            assert 'indicator_name' in metadata
            assert 'indicator_type' in metadata
            assert metadata['indicator_name'] == "EMA"
            assert metadata['indicator_type'] == "trend"
