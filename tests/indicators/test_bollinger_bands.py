"""Unit tests for Bollinger Bands indicator.

This module tests the Bollinger Bands indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.bollinger_bands import BollingerBandsIndicator
from backtester.indicators.indicator_configs import IndicatorConfig


class TestBollingerBandsIndicator:
    """Test BollingerBandsIndicator class methods."""

    def test_init_default_config(self, bollinger_config: IndicatorConfig) -> None:
        """Test Bollinger Bands indicator initialization with default configuration."""
        bb = BollingerBandsIndicator(bollinger_config)

        assert bb.name == "Bollinger"
        assert bb.type == "volatility"
        assert bb.config.indicator_type == "volatility"
        assert bb.config.period == 20
        assert bb.config.standard_deviations == 2.0

    def test_init_custom_config(self) -> None:
        """Test Bollinger Bands indicator initialization with custom configuration."""
        config = IndicatorConfig(
            indicator_name="CustomBB",
            indicator_type="volatility",
            period=30,
            standard_deviations=2.5,
        )

        bb = BollingerBandsIndicator(config)

        assert bb.name == "CustomBB"
        assert bb.config.period == 30
        assert bb.config.standard_deviations == 2.5

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, bollinger_config: IndicatorConfig
    ) -> None:
        """Test basic Bollinger Bands calculation."""
        bb = BollingerBandsIndicator(bollinger_config)
        result = bb.calculate(sample_ohlcv_data)

        # Check that all Bollinger Band columns are added
        assert "bollinger_upper" in result.columns
        assert "bollinger_middle" in result.columns
        assert "bollinger_lower" in result.columns

        # Check that original data is preserved
        pd.testing.assert_frame_equal(
            result.drop(["bollinger_upper", "bollinger_middle", "bollinger_lower"], axis=1),
            sample_ohlcv_data,
        )

    def test_calculate_different_periods(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Bollinger Bands calculation with different periods."""
        for period in [10, 20, 30]:
            config = IndicatorConfig(
                indicator_name="Bollinger", indicator_type="volatility", period=period
            )
            bb = BollingerBandsIndicator(config)
            result = bb.calculate(sample_ohlcv_data)

            assert "bollinger_upper" in result.columns
            assert "bollinger_middle" in result.columns
            assert "bollinger_lower" in result.columns

    def test_calculate_band_relationships(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test that Bollinger Bands have correct mathematical relationships."""
        config = IndicatorConfig(indicator_name="Bollinger", indicator_type="volatility", period=20)
        bb = BollingerBandsIndicator(config)
        result = bb.calculate(sample_ohlcv_data)

        # Upper band should be above middle band
        upper = result["bollinger_upper"].dropna()
        middle = result["bollinger_middle"].dropna()
        lower = result["bollinger_lower"].dropna()

        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_generate_signals_price_vs_bands(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Bollinger Bands signal generation for price vs bands."""
        config = IndicatorConfig(indicator_name="Bollinger", indicator_type="volatility")
        bb = BollingerBandsIndicator(config)

        # Calculate Bollinger Bands first
        data_with_bb = bb.calculate(sample_ohlcv_data)

        # Generate signals
        signals = bb.generate_signals(data_with_bb)

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

    def test_get_indicator_columns(self, bollinger_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        bb = BollingerBandsIndicator(bollinger_config)

        columns = bb.get_indicator_columns()
        assert len(columns) == 3
        assert "bollinger_upper" in columns
        assert "bollinger_middle" in columns
        assert "bollinger_lower" in columns

    def test_reset_functionality(self, bollinger_config: IndicatorConfig) -> None:
        """Test indicator reset functionality."""
        bb = BollingerBandsIndicator(bollinger_config)

        # Add some cache data
        bb._cache['test'] = 'value'
        bb._is_initialized = True

        bb.reset()

        assert bb._cache == {}
        assert bb._is_initialized is False
