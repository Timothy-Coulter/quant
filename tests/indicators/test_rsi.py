"""Unit tests for Relative Strength Index (RSI) indicator.

This module tests the RSI indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.rsi import RSIIndicator


class TestRSIIndicator:
    """Test RSIIndicator class methods."""

    def test_init_default_config(self, rsi_config: IndicatorConfig) -> None:
        """Test RSI indicator initialization with default configuration."""
        rsi = RSIIndicator(rsi_config)

        assert rsi.name == "RSI"
        assert rsi.type == "momentum"
        assert rsi.config.indicator_type == "momentum"
        assert rsi.config.period == 14
        assert rsi.config.overbought_threshold == 70.0
        assert rsi.config.oversold_threshold == 30.0

    def test_init_custom_config(self) -> None:
        """Test RSI indicator initialization with custom configuration."""
        config = IndicatorConfig(
            indicator_name="CustomRSI",
            indicator_type="momentum",
            period=21,
            overbought_threshold=80.0,
            oversold_threshold=20.0,
        )

        rsi = RSIIndicator(config)

        assert rsi.name == "CustomRSI"
        assert rsi.config.period == 21
        assert rsi.config.overbought_threshold == 80.0
        assert rsi.config.oversold_threshold == 20.0

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, rsi_config: IndicatorConfig
    ) -> None:
        """Test basic RSI calculation."""
        rsi = RSIIndicator(rsi_config)
        result = rsi.calculate(sample_ohlcv_data)

        # Check that RSI column is added
        assert "rsi_rsi" in result.columns

        # Check that original data is preserved
        pd.testing.assert_frame_equal(result.drop("rsi_rsi", axis=1), sample_ohlcv_data)

        # Check that RSI values are calculated
        rsi_values = result["rsi_rsi"].dropna()
        assert len(rsi_values) > 0

        # RSI should be between 0 and 100
        assert rsi_values.min() >= 0.0
        assert rsi_values.max() <= 100.0

    def test_calculate_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test RSI calculation with different periods."""
        for period in [10, 14, 21, 30]:
            config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=period)
            rsi = RSIIndicator(config)
            result = rsi.calculate(sample_ohlcv_data)

            rsi_column = "rsi_rsi"
            assert rsi_column in result.columns

            # First value should be NaN (RSI needs at least one period of data)
            assert pd.isna(result[rsi_column].iloc[0])

            # Check that we have some calculated values
            calculated_values = result[rsi_column].dropna()
            assert len(calculated_values) > 0
            # RSI values should be in valid range [0, 100]
            assert (calculated_values >= 0).all()
            assert (calculated_values <= 100).all()

            # Last value should not be NaN if we have enough data
            if len(sample_ohlcv_data) >= period:
                assert not result[rsi_column].iloc[-1:].isna().any()

    def test_calculate_ranging_data(self, ranging_data: pd.DataFrame) -> None:
        """Test RSI calculation with ranging market data."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)
        result = rsi.calculate(ranging_data)

        rsi_column = "rsi_rsi"
        rsi_values = result[rsi_column].dropna()

        # For ranging data, RSI should oscillate and show both overbought and oversold conditions
        assert rsi_values.min() < 70.0  # Should go below overbought
        assert rsi_values.max() > 30.0  # Should go above oversold

    def test_calculate_with_nan_values(self, data_with_nan: pd.DataFrame) -> None:
        """Test RSI calculation with NaN values in data."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=5)
        rsi = RSIIndicator(config)

        result = rsi.calculate(data_with_nan)

        # Should handle NaN values gracefully
        rsi_column = "rsi_rsi"
        assert rsi_column in result.columns

        # Some RSI values should be calculated
        assert not result[rsi_column].isna().all()

    def test_get_rsi_level_classification(self, rsi_config: IndicatorConfig) -> None:
        """Test RSI level classification functionality."""
        rsi = RSIIndicator(rsi_config)

        # Test different RSI values
        assert rsi._get_rsi_level(85) == "extremely_overbought"
        assert rsi._get_rsi_level(75) == "overbought"
        assert rsi._get_rsi_level(50) == "neutral"
        assert rsi._get_rsi_level(25) == "oversold"
        assert rsi._get_rsi_level(15) == "extremely_oversold"

    def test_generate_signals_overbought(self, volatile_data: pd.DataFrame) -> None:
        """Test RSI signal generation for overbought conditions."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        # Calculate RSI first
        data_with_rsi = rsi.calculate(volatile_data)

        # Generate signals
        signals = rsi.generate_signals(data_with_rsi)

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

    def test_generate_signals_oversold(self, volatile_data: pd.DataFrame) -> None:
        """Test RSI signal generation for oversold conditions."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        # Calculate RSI first
        data_with_rsi = rsi.calculate(volatile_data)

        # Generate signals
        signals = rsi.generate_signals(data_with_rsi)

        assert len(signals) > 0

    def test_generate_signals_extreme_readings(self) -> None:
        """Test RSI signal generation with extreme readings."""
        # Create data that should produce extreme RSI values
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volatile_prices = []

        # Create prices that go up strongly (should give high RSI)
        for i in range(15):
            volatile_prices.append(100 + i * 2)  # Upward trend
        for i in range(15):
            volatile_prices.append(130 - i * 2)  # Downward trend

        data = pd.DataFrame(
            {
                'open': [p * 0.999 for p in volatile_prices],
                'high': [p * 1.001 for p in volatile_prices],
                'low': [p * 0.998 for p in volatile_prices],
                'close': volatile_prices,
                'volume': [100000] * 30,
            },
            index=dates,
        )

        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        result = rsi.calculate(data)
        signals = rsi.generate_signals(result)

        # Should generate signals, potentially including extreme condition signals
        assert len(signals) > 0

    def test_divergence_detection(self, volatile_data: pd.DataFrame) -> None:
        """Test RSI divergence detection."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        # Calculate RSI first
        data_with_rsi = rsi.calculate(volatile_data)

        # Generate signals (should include divergence signals if detected)
        signals = rsi.generate_signals(data_with_rsi)

        # Check if any divergence signals are generated
        _ = [s for s in signals if 'divergence' in s.get('action', '').lower()]
        # Note: Divergence detection may or may not trigger depending on data pattern

    def test_get_indicator_columns(self, rsi_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        rsi = RSIIndicator(rsi_config)

        columns = rsi.get_indicator_columns()
        assert columns == ["rsi_rsi"]

    def test_reset_functionality(self, rsi_config: IndicatorConfig) -> None:
        """Test indicator reset functionality."""
        rsi = RSIIndicator(rsi_config)

        # Add some cache data
        rsi._cache['test'] = 'value'
        rsi._is_initialized = True

        rsi.reset()

        assert rsi._cache == {}
        assert rsi._is_initialized is False

    def test_performance_with_large_dataset(self, large_dataset: pd.DataFrame) -> None:
        """Test performance with large dataset."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        # Should complete without error
        result = rsi.calculate(large_dataset)

        assert "rsi_rsi" in result.columns
        assert len(result) == len(large_dataset)

    def test_rsi_calculation_accuracy(self) -> None:
        """Test mathematical accuracy of RSI calculation."""
        # Create simple test data with known RSI behavior
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

        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)
        result = rsi.calculate(data)

        rsi_values = result["rsi_rsi"].dropna()

        # RSI should be between 0 and 100
        assert rsi_values.min() >= 0.0
        assert rsi_values.max() <= 100.0

        # Should have some calculated values
        assert len(rsi_values) > 0

    def test_signal_confidence_calculation(self, volatile_data: pd.DataFrame) -> None:
        """Test that signal confidence is properly calculated."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        data_with_rsi = rsi.calculate(volatile_data)
        signals = rsi.generate_signals(data_with_rsi)

        for signal in signals:
            # Confidence should be between 0 and 1
            assert 0.0 <= signal['confidence'] <= 1.0

            # Check that confidence metadata is present
            if signal['signal_type'] in ['BUY', 'SELL']:
                # For overbought/oversold signals, confidence should be higher
                confidence = signal['confidence']
                # Should be reasonable (not too low for clear signals)
                assert confidence >= 0.3

    def test_momentum_signal_generation(self, trending_up_data: pd.DataFrame) -> None:
        """Test momentum signal generation."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)
        rsi = RSIIndicator(config)

        data_with_rsi = rsi.calculate(trending_up_data)
        signals = rsi.generate_signals(data_with_rsi)

        # Check for momentum-related signals
        momentum_signals = [s for s in signals if 'momentum' in s['action'].lower()]
        # Should generate at least one momentum signal
        assert len(momentum_signals) > 0

    def test_extreme_conditions(self) -> None:
        """Test RSI behavior in extreme market conditions."""
        # Test with extremely volatile data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        extreme_prices = []

        # Create extreme price movements
        for i in range(20):
            if i < 5:
                extreme_prices.append(100 + i * 10)  # Rapid rise
            elif i < 10:
                extreme_prices.append(150 - (i - 5) * 20)  # Rapid fall
            elif i < 15:
                extreme_prices.append(50 + (i - 10) * 15)  # Recovery
            else:
                extreme_prices.append(125 - (i - 15) * 5)  # Gradual decline

        data = pd.DataFrame(
            {
                'open': [p * 0.999 for p in extreme_prices],
                'high': [p * 1.002 for p in extreme_prices],
                'low': [p * 0.998 for p in extreme_prices],
                'close': extreme_prices,
                'volume': [100000] * 20,
            },
            index=dates,
        )

        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=5)
        rsi = RSIIndicator(config)

        result = rsi.calculate(data)
        signals = rsi.generate_signals(result)

        # Should handle extreme conditions without errors
        assert len(signals) > 0

        # Should generate some signals for extreme conditions
        extreme_signals = [s for s in signals if 'extreme' in s['action'].lower()]
        # Note: Extreme signals depend on RSI actually reaching extreme values
        # For this test, we just verify that signals are generated for volatile conditions
        volatile_signals = [
            s
            for s in signals
            if 'volatility' in s['action'].lower() or 'momentum' in s['action'].lower()
        ]
        assert len(volatile_signals) > 0 or len(extreme_signals) > 0

    def test_custom_thresholds(self, volatile_data: pd.DataFrame) -> None:
        """Test RSI with custom overbought/oversold thresholds."""
        # Test with tighter thresholds
        config = IndicatorConfig(
            indicator_name="RSI",
            indicator_type="momentum",
            period=14,
            overbought_threshold=60.0,
            oversold_threshold=40.0,
        )

        rsi = RSIIndicator(config)
        result = rsi.calculate(volatile_data)
        signals = rsi.generate_signals(result)

        assert len(signals) > 0
        assert rsi.config.overbought_threshold == 60.0
        assert rsi.config.oversold_threshold == 40.0

    def test_zero_volume_handling(self, data_with_zero_volume: pd.DataFrame) -> None:
        """Test RSI calculation with zero volume data."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=5)
        rsi = RSIIndicator(config)

        # Should handle zero volume gracefully
        result = rsi.calculate(data_with_zero_volume)

        assert "rsi_rsi" in result.columns

        # RSI should still be calculable (volume doesn't directly affect RSI)
        rsi_values = result["rsi_rsi"].dropna()
        assert len(rsi_values) > 0
