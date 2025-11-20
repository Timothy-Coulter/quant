"""Unit tests for Commodity Channel Index (CCI) indicator.

This module tests the CCI indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.cci import CCIIndicator
from backtester.indicators.indicator_configs import IndicatorConfig


class TestCCIIndicator:
    """Test CCIIndicator class methods."""

    def test_init_default_config(self, cci_config: IndicatorConfig) -> None:
        """Test CCI indicator initialization with default configuration."""
        cci = CCIIndicator(cci_config)

        assert cci.name == "CCI"
        assert cci.type == "momentum"
        assert cci.config.indicator_type == "momentum"
        assert cci.config.cci_period == 20
        assert cci.config.cci_constant == 0.015

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, cci_config: IndicatorConfig
    ) -> None:
        """Test basic CCI calculation."""
        cci = CCIIndicator(cci_config)
        result = cci.calculate(sample_ohlcv_data)

        # Check that CCI column is added
        assert "cci_cci" in result.columns

    def test_calculate_typical_price(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test Typical Price calculation."""
        config = IndicatorConfig(indicator_name="CCI", indicator_type="momentum")
        cci = CCIIndicator(config)

        typical_price = cci.calculate_typical_price(sample_ohlcv_data)

        # Typical price should be between low and high
        assert (typical_price >= sample_ohlcv_data['low']).all()
        assert (typical_price <= sample_ohlcv_data['high']).all()

    def test_generate_signals_extreme_readings(self, volatile_data: pd.DataFrame) -> None:
        """Test CCI signal generation for extreme readings."""
        config = IndicatorConfig(indicator_name="CCI", indicator_type="momentum")
        cci = CCIIndicator(config)

        # Calculate CCI first
        data_with_cci = cci.calculate(volatile_data)

        # Generate signals
        signals = cci.generate_signals(data_with_cci)

        assert len(signals) > 0

        for signal in signals:
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_get_indicator_columns(self, cci_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        cci = CCIIndicator(cci_config)

        columns = cci.get_indicator_columns()
        assert "cci_cci" in columns

    def test_get_cci_levels(self, cci_config: IndicatorConfig) -> None:
        """Test getting standard CCI interpretation levels."""
        cci = CCIIndicator(cci_config)

        levels = cci.get_cci_levels()

        assert "extreme_overbought" in levels
        assert "extreme_oversold" in levels
        assert levels["extreme_overbought"] == 200
        assert levels["extreme_oversold"] == -200
