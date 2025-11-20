"""Unit tests for Average True Range (ATR) indicator.

This module tests the ATR indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.atr import ATRIndicator
from backtester.indicators.indicator_configs import IndicatorConfig


class TestATRIndicator:
    """Test ATRIndicator class methods."""

    def test_init_default_config(self, atr_config: IndicatorConfig) -> None:
        """Test ATR indicator initialization with default configuration."""
        atr = ATRIndicator(atr_config)

        assert atr.name == "ATR"
        assert atr.type == "volatility"
        assert atr.config.indicator_type == "volatility"
        assert atr.config.period == 14

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, atr_config: IndicatorConfig
    ) -> None:
        """Test basic ATR calculation."""
        atr = ATRIndicator(atr_config)
        result = atr.calculate(sample_ohlcv_data)

        # Check that ATR column is added
        assert "atr_atr" in result.columns

        # ATR should be positive
        atr_values = result["atr_atr"].dropna()
        assert (atr_values >= 0).all()

    def test_calculate_true_range(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test True Range calculation."""
        config = IndicatorConfig(indicator_name="ATR", indicator_type="volatility")
        atr = ATRIndicator(config)

        true_range = atr.calculate_true_range(sample_ohlcv_data)

        # True Range should be positive
        assert (true_range.dropna() >= 0).all()

    def test_generate_signals_volatility(self, volatile_data: pd.DataFrame) -> None:
        """Test ATR signal generation for volatility changes."""
        config = IndicatorConfig(indicator_name="ATR", indicator_type="volatility")
        atr = ATRIndicator(config)

        # Calculate ATR first
        data_with_atr = atr.calculate(volatile_data)

        # Generate signals
        signals = atr.generate_signals(data_with_atr)

        assert len(signals) > 0

        for signal in signals:
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_get_indicator_columns(self, atr_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        atr = ATRIndicator(atr_config)

        columns = atr.get_indicator_columns()
        assert "atr_atr" in columns

    def test_get_volatility_percentile(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test volatility percentile calculation."""
        config = IndicatorConfig(indicator_name="ATR", indicator_type="volatility")
        atr = ATRIndicator(config)

        result = atr.calculate(sample_ohlcv_data)
        percentile = atr.get_volatility_percentile(result)

        # Should be between 0 and 100
        assert 0.0 <= percentile <= 100.0
