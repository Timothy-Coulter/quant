"""Unit tests for Williams %R (Williams Percent Range) indicator.

This module tests the Williams %R indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.williams_r import WilliamsROscillator


class TestWilliamsROscillator:
    """Test WilliamsROscillator class methods."""

    def test_init_default_config(self, williams_r_config: IndicatorConfig) -> None:
        """Test Williams %R initialization with default configuration."""
        wr = WilliamsROscillator(williams_r_config)

        assert wr.name == "WilliamsR"
        assert wr.type == "momentum"
        assert wr.config.indicator_type == "momentum"
        assert wr.config.williams_r_period == 14

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, williams_r_config: IndicatorConfig
    ) -> None:
        """Test basic Williams %R calculation."""
        wr = WilliamsROscillator(williams_r_config)
        result = wr.calculate(sample_ohlcv_data)

        # Check that Williams %R column is added
        assert "williamsr_wr" in result.columns

        # Williams %R should be between -100 and 0
        wr_values = result["williamsr_wr"].dropna()
        assert wr_values.min() >= -100.0
        assert wr_values.max() <= 0.0

    def test_generate_signals_overbought_oversold(self, volatile_data: pd.DataFrame) -> None:
        """Test Williams %R signal generation for overbought/oversold conditions."""
        config = IndicatorConfig(indicator_name="WilliamsR", indicator_type="momentum")
        wr = WilliamsROscillator(config)

        # Calculate Williams %R first
        data_with_wr = wr.calculate(volatile_data)

        # Generate signals
        signals = wr.generate_signals(data_with_wr)

        assert len(signals) > 0

        for signal in signals:
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_get_indicator_columns(self, williams_r_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        wr = WilliamsROscillator(williams_r_config)

        columns = wr.get_indicator_columns()
        assert "williamsr_wr" in columns
