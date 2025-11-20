"""Unit tests for IndicatorFactory pattern.

This module tests the factory pattern for creating indicator instances,
including registration, creation, and validation functionality.
"""

import pytest

from backtester.indicators.atr import ATRIndicator
from backtester.indicators.bollinger_bands import BollingerBandsIndicator
from backtester.indicators.cci import CCIIndicator
from backtester.indicators.ema import EMAIndicator
from backtester.indicators.factory import BaseIndicator, IndicatorFactory
from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.indicators.macd import MACDIndicator
from backtester.indicators.obv import OBVIndicator
from backtester.indicators.rsi import RSIIndicator
from backtester.indicators.sma import SMAIndicator
from backtester.indicators.stochastic import StochasticOscillator
from backtester.indicators.williams_r import WilliamsROscillator


class TestIndicatorFactory:
    """Test IndicatorFactory class methods."""

    def test_register_and_create_sma(self) -> None:
        """Test registering and creating SMA indicator."""
        config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", period=14)

        # Test that SMA is available
        available = IndicatorFactory.get_available_indicators()
        assert "sma" in available

        # Test creating SMA indicator
        sma = IndicatorFactory.create("sma", config)
        assert isinstance(sma, SMAIndicator)
        assert sma.config == config

    def test_register_and_create_ema(self) -> None:
        """Test registering and creating EMA indicator."""
        config = IndicatorConfig(indicator_name="EMA", indicator_type="trend", period=14)

        # Test that EMA is available
        available = IndicatorFactory.get_available_indicators()
        assert "ema" in available

        # Test creating EMA indicator
        ema = IndicatorFactory.create("ema", config)
        assert isinstance(ema, EMAIndicator)
        assert ema.config == config

    def test_register_and_create_rsi(self) -> None:
        """Test registering and creating RSI indicator."""
        config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14)

        # Test that RSI is available
        available = IndicatorFactory.get_available_indicators()
        assert "rsi" in available

        # Test creating RSI indicator
        rsi = IndicatorFactory.create("rsi", config)
        assert isinstance(rsi, RSIIndicator)
        assert rsi.config == config

    def test_register_and_create_macd(self) -> None:
        """Test registering and creating MACD indicator."""
        config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")

        # Test that MACD is available
        available = IndicatorFactory.get_available_indicators()
        assert "macd" in available

        # Test creating MACD indicator
        macd = IndicatorFactory.create("macd", config)
        assert isinstance(macd, MACDIndicator)
        assert macd.config == config

    def test_register_and_create_bollinger_bands(self) -> None:
        """Test registering and creating Bollinger Bands indicator."""
        config = IndicatorConfig(indicator_name="Bollinger", indicator_type="volatility")

        # Test that Bollinger Bands is available
        available = IndicatorFactory.get_available_indicators()
        assert "bollinger_bands" in available

        # Test creating Bollinger Bands indicator
        bb = IndicatorFactory.create("bollinger_bands", config)
        assert isinstance(bb, BollingerBandsIndicator)
        assert bb.config == config

    def test_register_and_create_stochastic(self) -> None:
        """Test registering and creating Stochastic indicator."""
        config = IndicatorConfig(indicator_name="Stochastic", indicator_type="momentum")

        # Test that Stochastic is available
        available = IndicatorFactory.get_available_indicators()
        assert "stochastic" in available

        # Test creating Stochastic indicator
        stoch = IndicatorFactory.create("stochastic", config)
        assert isinstance(stoch, StochasticOscillator)
        assert stoch.config == config

    def test_register_and_create_williams_r(self) -> None:
        """Test registering and creating Williams %R indicator."""
        config = IndicatorConfig(indicator_name="WilliamsR", indicator_type="momentum")

        # Test that Williams %R is available
        available = IndicatorFactory.get_available_indicators()
        assert "williams_r" in available

        # Test creating Williams %R indicator
        wr = IndicatorFactory.create("williams_r", config)
        assert isinstance(wr, WilliamsROscillator)
        assert wr.config == config

    def test_register_and_create_atr(self) -> None:
        """Test registering and creating ATR indicator."""
        config = IndicatorConfig(indicator_name="ATR", indicator_type="volatility")

        # Test that ATR is available
        available = IndicatorFactory.get_available_indicators()
        assert "atr" in available

        # Test creating ATR indicator
        atr = IndicatorFactory.create("atr", config)
        assert isinstance(atr, ATRIndicator)
        assert atr.config == config

    def test_register_and_create_cci(self) -> None:
        """Test registering and creating CCI indicator."""
        config = IndicatorConfig(indicator_name="CCI", indicator_type="momentum")

        # Test that CCI is available
        available = IndicatorFactory.get_available_indicators()
        assert "cci" in available

        # Test creating CCI indicator
        cci = IndicatorFactory.create("cci", config)
        assert isinstance(cci, CCIIndicator)
        assert cci.config == config

    def test_register_and_create_obv(self) -> None:
        """Test registering and creating OBV indicator."""
        config = IndicatorConfig(indicator_name="OBV", indicator_type="volume")

        # Test that OBV is available
        available = IndicatorFactory.get_available_indicators()
        assert "obv" in available

        # Test creating OBV indicator
        obv = IndicatorFactory.create("obv", config)
        assert isinstance(obv, OBVIndicator)
        assert obv.config == config

    def test_create_unknown_indicator(self) -> None:
        """Test creating unknown indicator raises ValueError."""
        config = IndicatorConfig(indicator_name="Unknown", indicator_type="trend")

        with pytest.raises(ValueError, match="Unknown indicator: unknown_indicator"):
            IndicatorFactory.create("unknown_indicator", config)

    def test_create_with_default_config(self) -> None:
        """Factory should fall back to default configurations when none provided."""
        indicator = IndicatorFactory.create("rsi")
        assert isinstance(indicator, RSIIndicator)
        assert indicator.config.indicator_name == 'rsi'

    def test_default_config_lookup(self) -> None:
        """default_config helper should instantiate registered configs."""
        config = IndicatorFactory.default_config("sma")
        assert config.factory_name == 'sma'
        assert config.indicator_type == 'trend'

    def test_is_registered(self) -> None:
        """Test checking if indicator is registered."""
        # Test registered indicators
        assert IndicatorFactory.is_registered("sma") is True
        assert IndicatorFactory.is_registered("ema") is True
        assert IndicatorFactory.is_registered("rsi") is True
        assert IndicatorFactory.is_registered("macd") is True

        # Test unregistered indicator
        assert IndicatorFactory.is_registered("unknown_indicator") is False

    def test_get_available_indicators(self) -> None:
        """Test getting list of available indicators."""
        available = IndicatorFactory.get_available_indicators()

        # Should include all registered indicators
        expected_indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger_bands",
            "stochastic",
            "williams_r",
            "atr",
            "cci",
            "obv",
        ]

        for indicator in expected_indicators:
            assert indicator in available

    def test_create_indicator_with_custom_config(self) -> None:
        """Test creating indicator with custom configuration."""
        # Test SMA with custom period
        config = IndicatorConfig(indicator_name="SMA_20", indicator_type="trend", period=20)

        sma = IndicatorFactory.create("sma", config)
        assert sma.config.period == 20

        # Test RSI with custom thresholds
        rsi_config = IndicatorConfig(
            indicator_name="RSI_Alt",
            indicator_type="momentum",
            overbought_threshold=80.0,
            oversold_threshold=20.0,
        )

        rsi = IndicatorFactory.create("rsi", rsi_config)
        assert rsi.config.overbought_threshold == 80.0
        assert rsi.config.oversold_threshold == 20.0

    def test_factory_pattern_independence(self) -> None:
        """Test that different instances are independent."""
        config1 = IndicatorConfig(indicator_name="SMA1", indicator_type="trend", period=10)
        config2 = IndicatorConfig(indicator_name="SMA2", indicator_type="trend", period=20)

        sma1 = IndicatorFactory.create("sma", config1)
        sma2 = IndicatorFactory.create("sma", config2)

        # Should be different instances
        assert sma1 is not sma2

        # Should have different configurations
        assert sma1.config.period == 10
        assert sma2.config.period == 20

    def test_decorator_registration(self) -> None:
        """Test that the @IndicatorFactory.register decorator works."""
        # This test verifies that the decorator registration pattern is working
        # by checking that the registered classes are available through the factory

        # All indicators should be registered and available
        all_indicators = [
            ("sma", SMAIndicator),
            ("ema", EMAIndicator),
            ("rsi", RSIIndicator),
            ("macd", MACDIndicator),
            ("bollinger_bands", BollingerBandsIndicator),
            ("stochastic", StochasticOscillator),
            ("williams_r", WilliamsROscillator),
            ("atr", ATRIndicator),
            ("cci", CCIIndicator),
            ("obv", OBVIndicator),
        ]

        for name, expected_class in all_indicators:
            assert IndicatorFactory.is_registered(name)

            config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
            instance = IndicatorFactory.create(name, config)
            assert isinstance(instance, expected_class)

    def test_create_all_indicators_success(self) -> None:
        """Test that all registered indicators can be created successfully."""
        indicators_to_test = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger_bands",
            "stochastic",
            "williams_r",
            "atr",
            "cci",
            "obv",
        ]

        for indicator_name in indicators_to_test:
            # Create appropriate config based on indicator type
            if indicator_name in ["sma", "ema"]:
                config = IndicatorConfig(indicator_name="Test", indicator_type="trend", period=14)
            elif indicator_name == "macd":
                config = IndicatorConfig(indicator_name="Test", indicator_type="momentum")
            elif indicator_name == "bollinger_bands":
                config = IndicatorConfig(indicator_name="Test", indicator_type="volatility")
            elif indicator_name == "obv":
                config = IndicatorConfig(indicator_name="Test", indicator_type="volume")
            else:
                config = IndicatorConfig(indicator_name="Test", indicator_type="momentum")

            # Should not raise any exception
            indicator = IndicatorFactory.create(indicator_name, config)
            assert isinstance(indicator, BaseIndicator)
            assert indicator.name == "Test"
