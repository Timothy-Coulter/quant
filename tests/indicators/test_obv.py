"""Unit tests for On-Balance Volume (OBV) indicator.

This module tests the OBV indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.obv import OBVIndicator


class TestOBVIndicator:
    """Test OBVIndicator class methods."""

    def test_init_default_config(self, obv_config: IndicatorConfig) -> None:
        """Test OBV indicator initialization with default configuration."""
        obv = OBVIndicator(obv_config)

        assert obv.name == "OBV"
        assert obv.type == "volume"
        assert obv.config.indicator_type == "volume"

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, obv_config: IndicatorConfig
    ) -> None:
        """Test basic OBV calculation."""
        obv = OBVIndicator(obv_config)
        result = obv.calculate(sample_ohlcv_data)

        # Check that OBV column is added
        assert "obv_obv" in result.columns

    def test_calculate_obv_change(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test OBV change calculation."""
        config = IndicatorConfig(indicator_name="OBV", indicator_type="volume")
        obv = OBVIndicator(config)

        result = obv.calculate(sample_ohlcv_data)
        change = obv.calculate_obv_change(result, periods=1)

        # Should return a numeric value
        assert isinstance(change, (int, float))

    def test_generate_signals_divergence(self, volatile_data: pd.DataFrame) -> None:
        """Test OBV signal generation for divergences."""
        config = IndicatorConfig(indicator_name="OBV", indicator_type="volume")
        obv = OBVIndicator(config)

        # Calculate OBV first
        data_with_obv = obv.calculate(volatile_data)

        # Generate signals
        signals = obv.generate_signals(data_with_obv)

        assert len(signals) > 0

        for signal in signals:
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_get_indicator_columns(self, obv_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        obv = OBVIndicator(obv_config)

        columns = obv.get_indicator_columns()
        assert "obv_obv" in columns

    def test_calculate_slope(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test slope calculation functionality."""
        config = IndicatorConfig(indicator_name="OBV", indicator_type="volume")
        obv = OBVIndicator(config)

        result = obv.calculate(sample_ohlcv_data)
        obv_data = result["obv_obv"].dropna()

        if len(obv_data) > 1:
            slope = obv._calculate_slope(obv_data.index, obv_data)
            assert isinstance(slope, float)
