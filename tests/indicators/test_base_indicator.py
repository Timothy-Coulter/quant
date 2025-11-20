"""Unit tests for BaseIndicator and indicator configuration classes.

This module tests the base indicator functionality including validation,
signal generation, and configuration management.
"""

import logging
from typing import Any

import pandas as pd
import pytest

from backtester.indicators.base_indicator import BaseIndicator
from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.signal.signal_types import SignalGenerator, SignalType


class TestIndicatorConfig:
    """Test IndicatorConfig class methods."""

    def test_init_default_config(self) -> None:
        """Test IndicatorConfig initialization with defaults."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")

        assert config.indicator_name == "Test"
        assert config.indicator_type == "trend"
        assert config.period == 14
        assert config.short_period == 5
        assert config.long_period == 20
        assert config.ma_type == "simple"
        assert config.price_column == "close"
        assert config.overbought_threshold == 70.0
        assert config.oversold_threshold == 30.0
        assert config.standard_deviations == 2.0
        assert config.cache_calculations is True
        assert config.calculate_realtime is False

    def test_init_custom_config(self) -> None:
        """Test IndicatorConfig initialization with custom values."""
        config = IndicatorConfig(
            indicator_name="Custom",
            indicator_type="momentum",
            period=20,
            overbought_threshold=80.0,
            oversold_threshold=20.0,
            ma_type="exponential",
        )

        assert config.indicator_name == "Custom"
        assert config.indicator_type == "momentum"
        assert config.period == 20
        assert config.overbought_threshold == 80.0
        assert config.oversold_threshold == 20.0
        assert config.ma_type == "exponential"

    def test_config_validation_indicator_type(self) -> None:
        """Test indicator type validation."""
        # Valid types should work
        valid_types = ['trend', 'momentum', 'volume', 'volatility', 'oscillator']
        for vtype in valid_types:
            config = IndicatorConfig(indicator_name="Test", indicator_type=vtype)
            assert config.indicator_type == vtype

        # Invalid type should raise ValueError
        with pytest.raises(ValueError, match="indicator_type must be one of"):
            IndicatorConfig(indicator_name="Test", indicator_type="invalid_type")

    def test_config_validation_ma_type(self) -> None:
        """Test moving average type validation."""
        # Valid types should work
        valid_types = ['simple', 'exponential', 'weighted']
        for ma_type in valid_types:
            config = IndicatorConfig(indicator_name="Test", indicator_type="trend", ma_type=ma_type)
            assert config.ma_type == ma_type

        # Invalid type should raise ValueError
        with pytest.raises(ValueError, match="ma_type must be one of"):
            IndicatorConfig(indicator_name="Test", indicator_type="trend", ma_type="invalid_ma")

    def test_config_validation_positive_periods(self) -> None:
        """Test that all period parameters are positive."""
        # Valid positive periods
        config = IndicatorConfig(
            indicator_name="Test", indicator_type="trend", period=10, short_period=5, long_period=20
        )
        assert config.period == 10
        assert config.short_period == 5
        assert config.long_period == 20

        # Zero or negative periods should raise ValueError
        with pytest.raises(ValueError, match="Period must be positive"):
            IndicatorConfig(indicator_name="Test", indicator_type="trend", period=0)

        with pytest.raises(ValueError, match="Period must be positive"):
            IndicatorConfig(indicator_name="Test", indicator_type="trend", period=-5)

    def test_config_validation_float_ranges(self) -> None:
        """Test float parameter range validation."""
        # Valid ranges
        config = IndicatorConfig(
            indicator_name="Test",
            indicator_type="oscillator",
            overbought_threshold=0.8,
            oversold_threshold=0.2,
            signal_sensitivity=0.7,
            confidence_threshold=0.6,
        )
        assert config.overbought_threshold == 0.8
        assert config.oversold_threshold == 0.2

    def test_get_indicator_columns(self) -> None:
        """Test column name generation for different indicator types."""
        # SMA
        sma_config = IndicatorConfig(indicator_name="SMA", indicator_type="trend", ma_type="simple")
        columns = sma_config.get_indicator_columns()
        assert columns == ["sma_sma"]

        # EMA
        ema_config = IndicatorConfig(
            indicator_name="EMA", indicator_type="trend", ma_type="exponential"
        )
        columns = ema_config.get_indicator_columns()
        assert columns == ["ema_ema"]

        # RSI
        rsi_config = IndicatorConfig(indicator_name="RSI", indicator_type="momentum")
        columns = rsi_config.get_indicator_columns()
        assert columns == ["rsi_rsi"]

        # MACD
        macd_config = IndicatorConfig(indicator_name="MACD", indicator_type="momentum")
        columns = macd_config.get_indicator_columns()
        assert columns == ["macd_macd", "macd_signal", "macd_histogram"]

        # Bollinger Bands
        bb_config = IndicatorConfig(indicator_name="Bollinger", indicator_type="volatility")
        columns = bb_config.get_indicator_columns()
        assert columns == ["bollinger_upper", "bollinger_middle", "bollinger_lower"]

        # Stochastic
        stoch_config = IndicatorConfig(indicator_name="Stochastic", indicator_type="momentum")
        columns = stoch_config.get_indicator_columns()
        assert columns == ["stochastic_k", "stochastic_d"]

    def test_validate_for_indicator(self) -> None:
        """Test indicator-specific configuration validation."""
        # Valid MACD configuration
        macd_config = IndicatorConfig(
            indicator_name="MACD",
            indicator_type="momentum",
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )
        macd_config.validate_for_indicator()  # Should not raise

        # Invalid MACD configuration (fast >= slow)
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            invalid_macd = IndicatorConfig(
                indicator_name="MACD", indicator_type="momentum", fast_period=26, slow_period=12
            )
            invalid_macd.validate_for_indicator()

        # Valid RSI configuration
        rsi_config = IndicatorConfig(
            indicator_name="RSI",
            indicator_type="momentum",
            overbought_threshold=70.0,
            oversold_threshold=30.0,
        )
        rsi_config.validate_for_indicator()  # Should not raise

        # Invalid RSI configuration (overbought <= oversold)
        with pytest.raises(
            ValueError, match="Overbought threshold must be greater than oversold threshold"
        ):
            invalid_rsi = IndicatorConfig(
                indicator_name="RSI",
                indicator_type="momentum",
                overbought_threshold=30.0,
                oversold_threshold=70.0,
            )
            invalid_rsi.validate_for_indicator()


class TestSignalTypes:
    """Test signal types and signal generation utilities."""

    def test_signal_type_enum(self) -> None:
        """Test SignalType enumeration."""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"

    def test_create_signal_valid(self) -> None:
        """Test creating valid signals."""
        signal = SignalGenerator.create_signal(
            signal_type=SignalType.BUY, action="Test action", confidence=0.8, timestamp="2023-01-01"
        )

        assert signal['signal_type'] == "BUY"
        assert signal['action'] == "Test action"
        assert signal['confidence'] == 0.8
        assert signal['timestamp'] == "2023-01-01"
        assert signal['metadata'] == {}

    def test_create_signal_with_metadata(self) -> None:
        """Test creating signal with metadata."""
        metadata = {'test': True, 'value': 42}
        signal = SignalGenerator.create_signal(
            signal_type=SignalType.SELL,
            action="Test action",
            confidence=0.6,
            timestamp="2023-01-01",
            metadata=metadata,
        )

        assert signal['metadata'] == metadata

    def test_create_signal_invalid_confidence(self) -> None:
        """Test creating signal with invalid confidence levels."""
        # Confidence too high
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SignalGenerator.create_signal(
                signal_type=SignalType.BUY,
                action="Test action",
                confidence=1.5,
                timestamp="2023-01-01",
            )

        # Confidence too low
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SignalGenerator.create_signal(
                signal_type=SignalType.BUY,
                action="Test action",
                confidence=-0.1,
                timestamp="2023-01-01",
            )

    def test_validate_signal_valid(self) -> None:
        """Test validating valid signal."""
        valid_signal = {
            'timestamp': '2023-01-01',
            'signal_type': 'BUY',
            'action': 'Test action',
            'confidence': 0.8,
            'metadata': {'test': True},
        }

        assert SignalGenerator.validate_signal(valid_signal) is True

    def test_validate_signal_missing_fields(self) -> None:
        """Test validating signal with missing required fields."""
        # Missing signal_type
        invalid_signal = {
            'timestamp': '2023-01-01',
            'action': 'Test action',
            'confidence': 0.8,
            'metadata': {'test': True},
        }
        with pytest.raises(ValueError, match="Missing required field: signal_type"):
            SignalGenerator.validate_signal(invalid_signal)

        # Missing confidence
        invalid_signal = {
            'timestamp': '2023-01-01',
            'signal_type': 'BUY',
            'action': 'Test action',
            'metadata': {'test': True},
        }
        with pytest.raises(ValueError, match="Missing required field: confidence"):
            SignalGenerator.validate_signal(invalid_signal)

    def test_validate_signal_invalid_values(self) -> None:
        """Test validating signal with invalid values."""
        # Invalid signal_type
        invalid_signal = {
            'timestamp': '2023-01-01',
            'signal_type': 'INVALID',
            'action': 'Test action',
            'confidence': 0.8,
            'metadata': {'test': True},
        }
        with pytest.raises(ValueError, match="Invalid signal_type"):
            SignalGenerator.validate_signal(invalid_signal)

        # Invalid confidence
        invalid_signal = {
            'timestamp': '2023-01-01',
            'signal_type': 'BUY',
            'action': 'Test action',
            'confidence': 2.0,
            'metadata': {'test': True},
        }
        with pytest.raises(ValueError, match="Confidence must be a number between 0.0 and 1.0"):
            SignalGenerator.validate_signal(invalid_signal)


class MockIndicator(BaseIndicator):
    """Mock indicator for testing base class functionality."""

    def __init__(self, config: IndicatorConfig, logger: logging.Logger | None = None) -> None:
        """Initialize mock indicator.

        Args:
            config: Indicator configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock indicator values.

        Args:
            data: Input market data

        Returns:
            DataFrame with mock indicator values
        """
        # Simple mock calculation
        result = data.copy()
        result[f"{self.name.lower()}_value"] = data['close'].rolling(5).mean()
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate mock trading signals.

        Args:
            data: DataFrame with market data and calculated indicator values

        Returns:
            List of signal dictionaries
        """
        # Simple mock signal generation
        return [
            self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Mock buy signal",
                confidence=0.5,
                timestamp=data.index[-1],
            )
        ]


class TestBaseIndicator:
    """Test BaseIndicator class methods."""

    def test_init_default_logger(self, mock_logger: logging.Logger) -> None:
        """Test BaseIndicator initialization with default logger."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        assert indicator.config == config
        assert indicator.name == "Test"
        assert indicator.type == "trend"
        assert indicator.logger is not None
        assert indicator._cache == {}
        assert indicator._is_initialized is False

    def test_init_custom_logger(self, mock_logger: logging.Logger) -> None:
        """Test BaseIndicator initialization with custom logger."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config, mock_logger)

        assert indicator.logger == mock_logger

    def test_validate_data_valid(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Test data validation with valid data."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        # Should not raise
        assert indicator.validate_data(sample_ohlcv_data) is True

    def test_validate_data_empty(self, empty_data: pd.DataFrame) -> None:
        """Test data validation with empty data."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        with pytest.raises(ValueError, match="Input data cannot be None or empty"):
            indicator.validate_data(empty_data)

    def test_validate_data_missing_columns(self, data_missing_columns: pd.DataFrame) -> None:
        """Test data validation with missing required columns."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        with pytest.raises(ValueError, match="Missing required columns"):
            indicator.validate_data(data_missing_columns)

    def test_validate_data_wrong_index_type(self, data_wrong_index_type: pd.DataFrame) -> None:
        """Test data validation with wrong index type."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        with pytest.raises(ValueError, match="Data must be datetime indexed"):
            indicator.validate_data(data_wrong_index_type)

    def test_validate_data_insufficient_periods(self, insufficient_data: pd.DataFrame) -> None:
        """Test data validation with insufficient periods."""
        config = IndicatorConfig(
            indicator_name="Test",
            indicator_type="trend",
            period=20,  # Need 20 periods but only have 5
        )
        indicator = MockIndicator(config)

        with pytest.raises(ValueError, match="Insufficient data: need at least 20 periods"):
            indicator.validate_data(insufficient_data)

    def test_validate_data_invalid_ohlc(self, data_with_invalid_ohlc: pd.DataFrame) -> None:
        """Test data validation with invalid OHLC relationships."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        # The test data has invalid OHLC relationships, so validation should fail
        # Check that it fails with any of the OHLC validation errors
        with pytest.raises(
            ValueError,
            match="High prices must be >= low prices|High prices must be >= open prices|Low prices must be <= open prices",
        ):
            indicator.validate_data(data_with_invalid_ohlc)

    def test_get_required_columns(self) -> None:
        """Test getting required data columns."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        columns = indicator.get_required_columns()
        assert columns == ['open', 'high', 'low', 'close', 'volume']

    def test_reset(self, mock_logger: logging.Logger) -> None:
        """Test indicator reset functionality."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config, mock_logger)

        # Add some cache data
        indicator._cache['test'] = 'value'
        indicator._is_initialized = True

        indicator.reset()

        assert indicator._cache == {}
        assert indicator._is_initialized is False

    def test_get_indicator_info(self) -> None:
        """Test getting indicator information."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend", period=14)
        indicator = MockIndicator(config)

        info = indicator.get_indicator_info()

        assert info['name'] == "Test"
        assert info['type'] == "trend"
        assert info['period'] == 14
        assert info['is_initialized'] is False
        assert 'config' in info

    def test_cache_operations(self) -> None:
        """Test cache get/set operations."""
        config = IndicatorConfig(
            indicator_name="Test", indicator_type="trend", cache_calculations=True
        )
        indicator = MockIndicator(config)

        # Test set and get
        indicator._set_cached_result("test_key", "test_value")
        result = indicator._get_cached_result("test_key")
        assert result == "test_value"

        # Test missing key
        result = indicator._get_cached_result("missing_key")
        assert result is None

        # Test with caching disabled
        config.cache_calculations = False
        indicator._set_cached_result("test_key2", "test_value2")
        result = indicator._get_cached_result("test_key2")
        assert result is None

    def test_create_standard_signal(self) -> None:
        """Test creating standardized signals."""
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend")
        indicator = MockIndicator(config)

        signal = indicator._create_standard_signal(
            signal_type=SignalType.BUY,
            action="Test action",
            confidence=0.8,
            timestamp="2023-01-01",
            metadata_key="metadata_value",
        )

        assert signal['signal_type'] == "BUY"
        assert signal['action'] == "Test action"
        assert signal['confidence'] == 0.8
        assert signal['timestamp'] == "2023-01-01"
        assert 'indicator_name' in signal['metadata']
        assert 'indicator_type' in signal['metadata']
        assert signal['metadata']['metadata_key'] == "metadata_value"

    def test_configuration_validation(self) -> None:
        """Test configuration validation during initialization."""
        # Valid configuration
        config = IndicatorConfig(indicator_name="Test", indicator_type="trend", period=14)
        MockIndicator(config)  # Should not raise

        # Invalid configuration - missing name (empty string)
        with pytest.raises((ValueError, Exception)):  # Can be ValueError or ValidationError
            invalid_config = IndicatorConfig(indicator_name="", indicator_type="trend")
            MockIndicator(invalid_config)

        # Invalid configuration - missing type (empty string)
        with pytest.raises((ValueError, Exception)):  # Can be ValueError or ValidationError
            invalid_config = IndicatorConfig(indicator_name="Test", indicator_type="")
            MockIndicator(invalid_config)

        # Invalid configuration - negative period
        with pytest.raises((ValueError, Exception)):  # Can be ValueError or ValidationError
            invalid_config = IndicatorConfig(
                indicator_name="Test", indicator_type="trend", period=-5
            )
            MockIndicator(invalid_config)
