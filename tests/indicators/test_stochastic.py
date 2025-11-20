"""Unit tests for Stochastic Oscillator indicator.

This module tests the Stochastic Oscillator indicator implementation, including calculation,
signal generation, and edge cases.
"""

import pandas as pd

from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.stochastic import StochasticOscillator


class TestStochasticOscillator:
    """Test StochasticOscillator class methods."""

    def test_init_default_config(self, stochastic_config: IndicatorConfig) -> None:
        """Test Stochastic Oscillator initialization with default configuration."""
        stoch = StochasticOscillator(stochastic_config)

        assert stoch.name == "Stochastic"
        assert stoch.type == "momentum"
        assert stoch.config.indicator_type == "momentum"
        assert stoch.config.k_period == 14
        assert stoch.config.d_period == 3

    def test_calculate_basic(
        self, sample_ohlcv_data: pd.DataFrame, stochastic_config: IndicatorConfig
    ) -> None:
        """Test basic Stochastic calculation."""
        stoch = StochasticOscillator(stochastic_config)
        result = stoch.calculate(sample_ohlcv_data)

        # Check that both %K and %D columns are added
        assert "stochastic_k" in result.columns
        assert "stochastic_d" in result.columns

    def test_generate_signals_overbought_oversold(self, volatile_data: pd.DataFrame) -> None:
        """Test Stochastic signal generation for overbought/oversold conditions."""
        config = IndicatorConfig(indicator_name="Stochastic", indicator_type="momentum")
        stoch = StochasticOscillator(config)

        # Calculate Stochastic first
        data_with_stoch = stoch.calculate(volatile_data)

        # Generate signals
        signals = stoch.generate_signals(data_with_stoch)

        assert len(signals) > 0

        for signal in signals:
            assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signal['confidence'] <= 1.0

    def test_get_indicator_columns(self, stochastic_config: IndicatorConfig) -> None:
        """Test getting indicator column names."""
        stoch = StochasticOscillator(stochastic_config)

        columns = stoch.get_indicator_columns()
        assert "stochastic_k" in columns
        assert "stochastic_d" in columns
