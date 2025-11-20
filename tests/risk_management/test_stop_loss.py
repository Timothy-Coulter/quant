"""Comprehensive unit tests for the StopLoss class.

This module contains tests for the stop loss functionality including
all stop loss types, initialization, position management, trigger conditions,
and edge cases.
"""

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from backtester.risk_management.component_configs.stop_loss_config import (
    StopLossConfig,
    StopLossType,
)
from backtester.risk_management.stop_loss import StopLoss


class TestStopLossConfig:
    """Test suite for StopLossConfig validation and behavior."""

    def test_init_default_config(self) -> None:
        """Test StopLossConfig initialization with defaults."""
        config = StopLossConfig()

        assert config.stop_loss_type == StopLossType.PERCENTAGE
        assert config.stop_loss_value == 0.02
        assert config.trail_distance == 0.01
        assert config.trail_step == 0.005
        assert config.max_loss_value is None
        assert config.activation_price is None
        assert config.trailing_stop_pct == 0.05
        assert config.is_trailing is False

    def test_init_custom_config(self) -> None:
        """Test StopLossConfig initialization with custom values."""
        config = StopLossConfig(
            stop_loss_type=StopLossType.FIXED,
            stop_loss_value=0.5,  # Use percentage for fixed type (50%)
            trail_distance=0.02,
            trail_step=0.01,
            max_loss_value=100.0,
            activation_price=95.0,
            trailing_stop_pct=0.10,
        )

        assert config.stop_loss_type == StopLossType.FIXED
        assert config.stop_loss_value == 0.5
        assert config.trail_distance == 0.02
        assert config.trail_step == 0.01
        assert config.max_loss_value == 100.0
        assert config.activation_price == 95.0
        assert config.trailing_stop_pct == 0.10
        assert config.is_trailing is False

    def test_init_trailing_config(self) -> None:
        """Test StopLossConfig initialization with trailing stop loss."""
        config = StopLossConfig(
            stop_loss_type=StopLossType.TRAILING,
            trail_distance=0.03,
            trail_step=0.01,
        )

        assert config.stop_loss_type == StopLossType.TRAILING
        assert config.is_trailing is True

    def test_config_validation_stop_loss_value(self) -> None:
        """Test stop_loss_value validation constraints."""
        # Valid values
        config = StopLossConfig(stop_loss_value=0.05)
        assert config.stop_loss_value == 0.05

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            StopLossConfig(stop_loss_value=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            StopLossConfig(stop_loss_value=1.5)  # Too large (le=1.0)

    def test_config_validation_trail_distance(self) -> None:
        """Test trail_distance validation constraints."""
        # Valid values
        config = StopLossConfig(trail_distance=0.02)
        assert config.trail_distance == 0.02

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            StopLossConfig(trail_distance=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            StopLossConfig(trail_distance=0.6)  # Too large (le=0.5)

    def test_config_validation_trail_step(self) -> None:
        """Test trail_step validation constraints."""
        # Valid values
        config = StopLossConfig(trail_step=0.008)
        assert config.trail_step == 0.008

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            StopLossConfig(trail_step=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            StopLossConfig(trail_step=0.15)  # Too large (le=0.1)

    def test_config_validation_max_loss_value(self) -> None:
        """Test max_loss_value validation constraints."""
        # Valid value
        config = StopLossConfig(max_loss_value=100.0)
        assert config.max_loss_value == 100.0

        # Invalid value
        with pytest.raises(ValidationError):
            StopLossConfig(max_loss_value=-10.0)  # Negative (ge=0.0)

    def test_config_validation_activation_price(self) -> None:
        """Test activation_price validation constraints."""
        # Valid value
        config = StopLossConfig(activation_price=50.0)
        assert config.activation_price == 50.0

        # Invalid value
        with pytest.raises(ValidationError):
            StopLossConfig(activation_price=-10.0)  # Negative (ge=0.0)

    def test_config_validation_trailing_stop_pct(self) -> None:
        """Test trailing_stop_pct validation constraints."""
        # Valid values
        config = StopLossConfig(trailing_stop_pct=0.08)
        assert config.trailing_stop_pct == 0.08

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            StopLossConfig(trailing_stop_pct=0.005)  # Too small (ge=0.01)

        with pytest.raises(ValidationError):
            StopLossConfig(trailing_stop_pct=0.25)  # Too large (le=0.2)

    def test_is_trailing_property(self) -> None:
        """Test is_trailing property for different types."""
        config_fixed = StopLossConfig(stop_loss_type=StopLossType.FIXED)
        assert config_fixed.is_trailing is False

        config_percentage = StopLossConfig(stop_loss_type=StopLossType.PERCENTAGE)
        assert config_percentage.is_trailing is False

        config_trailing = StopLossConfig(stop_loss_type=StopLossType.TRAILING)
        assert config_trailing.is_trailing is True

        config_trailing_pct = StopLossConfig(stop_loss_type=StopLossType.TRAILING_PERCENTAGE)
        assert config_trailing_pct.is_trailing is True

    def test_config_serialization(self) -> None:
        """Test config serialization methods."""
        config = StopLossConfig(
            stop_loss_type=StopLossType.FIXED,
            stop_loss_value=0.5,  # 50% as percentage
        )

        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["stop_loss_type"] == "FIXED"
        assert config_dict["stop_loss_value"] == 0.5

        # Test JSON serialization
        config_json = config.model_dump_json()
        assert isinstance(config_json, str)
        assert '"stop_loss_type":"FIXED"' in config_json

    def test_all_stop_loss_types(self) -> None:
        """Test all stop loss types can be configured."""
        types = [
            StopLossType.FIXED,
            StopLossType.PERCENTAGE,
            StopLossType.PRICE,
            StopLossType.TRAILING,
            StopLossType.TRAILING_PERCENTAGE,
        ]

        for stop_type in types:
            config = StopLossConfig(stop_loss_type=stop_type)
            assert config.stop_loss_type == stop_type


class TestStopLoss:
    """Test suite for the StopLoss class."""

    @pytest.fixture
    def default_config(self) -> StopLossConfig:
        """Create default StopLossConfig for testing."""
        return StopLossConfig()

    @pytest.fixture
    def fixed_config(self) -> StopLossConfig:
        """Create fixed stop loss configuration for testing."""
        return StopLossConfig(
            stop_loss_type=StopLossType.FIXED,
            stop_loss_value=0.4,  # 40% fixed stop loss
        )

    @pytest.fixture
    def trailing_config(self) -> StopLossConfig:
        """Create trailing stop loss configuration for testing."""
        return StopLossConfig(
            stop_loss_type=StopLossType.TRAILING,
            trail_distance=0.01,  # 1% trail
            trail_step=0.005,  # 0.5% step
        )

    @pytest.fixture
    def percentage_config(self) -> StopLossConfig:
        """Create percentage stop loss configuration for testing."""
        return StopLossConfig(
            stop_loss_type=StopLossType.PERCENTAGE,
            stop_loss_value=0.05,  # 5% stop loss
        )

    @pytest.fixture
    def mock_logger(self) -> Mock:
        """Create mock logger for testing."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def sample_timestamp(self) -> pd.Timestamp:
        """Create sample timestamp for testing."""
        return pd.Timestamp("2023-01-01 12:00:00")

    def test_init_default(self, default_config: StopLossConfig) -> None:
        """Test StopLoss initialization with default config."""
        stop_loss = StopLoss()

        assert stop_loss.config == default_config
        assert stop_loss.logger is not None
        assert isinstance(stop_loss.logger, logging.Logger)
        assert stop_loss.activation_price == default_config.activation_price
        assert stop_loss.entry_price is None
        assert stop_loss.highest_price == 0.0
        assert stop_loss.is_active is True
        assert stop_loss.stop_triggered is False
        assert stop_loss.stop_price is None

    def test_init_custom_config(self, fixed_config: StopLossConfig, mock_logger: Mock) -> None:
        """Test StopLoss initialization with custom config and logger."""
        stop_loss = StopLoss(config=fixed_config, logger=mock_logger)

        assert stop_loss.config == fixed_config
        assert stop_loss.logger == mock_logger
        assert stop_loss.stop_loss_type == StopLossType.FIXED
        assert stop_loss.stop_loss_value == 0.4

    def test_initialize_position_long(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization for long position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        assert stop_loss.entry_price == entry_price
        assert stop_loss.highest_price == entry_price
        assert stop_loss.is_active is True
        assert stop_loss.stop_triggered is False
        assert stop_loss.triggered is False
        assert stop_loss.trigger_price is None
        assert stop_loss.trigger_time is None
        assert stop_loss.activation_price == entry_price  # Default when None
        assert stop_loss.stop_price is not None

    def test_initialize_position_with_activation_price(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization with custom activation price."""
        config = StopLossConfig(activation_price=95.0)
        stop_loss = StopLoss(config=config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        assert stop_loss.activation_price == 95.0
        assert stop_loss.entry_price == entry_price

    def test_initialize_position_already_initialized(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization when already initialized."""
        stop_loss = StopLoss(config=default_config)

        entry_price1 = 100.0
        stop_loss.initialize_position(entry_price1, sample_timestamp)

        entry_price2 = 110.0
        stop_loss.initialize_position(entry_price2, sample_timestamp)

        # Should update to new values
        assert stop_loss.entry_price == entry_price2
        assert stop_loss.highest_price == entry_price2
        assert stop_loss.trigger_price is None
        assert stop_loss.trigger_time is None

    def test_update_not_active(self, default_config: StopLossConfig) -> None:
        """Test update when stop loss is not active."""
        stop_loss = StopLoss(config=default_config)
        stop_loss.is_active = False

        result = stop_loss.update(90.0, pd.Timestamp.now())

        assert result['triggered'] is False
        assert result['stop_price'] is None
        assert result['action'] == 'NONE'
        assert result['reason'] == 'Stop loss not active'

    def test_update_no_entry_price(self, default_config: StopLossConfig) -> None:
        """Test update when no entry price is set."""
        stop_loss = StopLoss(config=default_config)
        # entry_price is None by default

        result = stop_loss.update(90.0, pd.Timestamp.now())

        assert result['triggered'] is False
        assert result['stop_price'] is None
        assert result['action'] == 'NONE'
        assert result['reason'] == 'Stop loss not active'

    def test_update_fixed_stop_loss_triggered(
        self, fixed_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with fixed stop loss that triggers."""
        # For FIXED type, stop_loss_value is treated as percentage
        config = StopLossConfig(stop_loss_type=StopLossType.FIXED, stop_loss_value=0.4)  # 40% stop
        stop_loss = StopLoss(config=config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops to 40% stop level (60.0)
        current_price = 60.0
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        expected_stop = entry_price * (1 - 0.4)  # 60.0
        assert abs(result['stop_price'] - expected_stop) < 0.001
        assert result['action'] == 'STOP_LOSS'
        assert stop_loss.is_active is False
        assert stop_loss.triggered is True
        assert stop_loss.trigger_price == current_price
        assert stop_loss.trigger_time == sample_timestamp
        assert abs(result['pnl_pct'] - (-0.4)) < 0.001  # 40% loss

    def test_update_fixed_stop_loss_not_triggered(
        self, fixed_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with fixed stop loss that doesn't trigger."""
        # For FIXED type, stop_loss_value is treated as percentage
        config = StopLossConfig(stop_loss_type=StopLossType.FIXED, stop_loss_value=0.4)  # 40% stop
        stop_loss = StopLoss(config=config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops but not to 40% stop level (60.0)
        current_price = 70.0
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is False
        expected_stop = entry_price * (1 - 0.4)  # 60.0
        assert abs(result['stop_price'] - expected_stop) < 0.001
        assert result['action'] == 'NONE'
        assert stop_loss.is_active is True
        assert stop_loss.triggered is False

    def test_update_percentage_stop_loss_triggered(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with percentage stop loss that triggers."""
        stop_loss = StopLoss(config=percentage_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops to 5% stop loss (95.0)
        current_price = 95.0
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        expected_stop = entry_price * (1 - 0.05)  # 95.0
        assert abs(result['stop_price'] - expected_stop) < 0.001
        assert result['action'] == 'STOP_LOSS'
        assert abs(result['pnl_pct'] - (-0.05)) < 0.001  # 5% loss

    def test_update_percentage_stop_loss_not_triggered(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with percentage stop loss that doesn't trigger."""
        stop_loss = StopLoss(config=percentage_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops but not to 5% stop loss
        current_price = 98.0
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is False
        expected_stop = entry_price * (1 - 0.05)  # 95.0
        assert abs(result['stop_price'] - expected_stop) < 0.001
        assert result['action'] == 'NONE'

    def test_update_trailing_stop_loss_tracking(
        self, trailing_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with trailing stop loss that tracks price."""
        stop_loss = StopLoss(config=trailing_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price moves up - should track
        result1 = stop_loss.update(110.0, sample_timestamp)
        assert result1['triggered'] is False
        assert stop_loss.highest_price == 110.0

        # Price moves up further - should track again
        result2 = stop_loss.update(120.0, sample_timestamp)
        assert result2['triggered'] is False
        assert stop_loss.highest_price == 120.0

        # Price moves down but not below trailing stop - should not trigger
        # With 1% trail distance, stop should be at 120 * 0.99 = 118.8
        # So 117.0 should trigger (below 118.8), change to 118.9 to not trigger
        result3 = stop_loss.update(118.9, sample_timestamp)  # Slightly above trail stop
        assert result3['triggered'] is False

    def test_update_trailing_stop_loss_triggered(
        self, trailing_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with trailing stop loss that eventually triggers."""
        stop_loss = StopLoss(config=trailing_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price moves up
        stop_loss.update(110.0, sample_timestamp)
        stop_loss.update(120.0, sample_timestamp)
        assert stop_loss.highest_price == 120.0

        # Price drops below trailing stop
        current_price = 117.0  # Should trigger based on trail_distance
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'STOP_LOSS'

    def test_update_max_loss_limit_triggered(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with maximum loss limit that triggers."""
        config = StopLossConfig(
            max_loss_value=3.0,  # $3 max loss
            stop_loss_value=0.10,  # Wide stop so max loss triggers first
        )
        stop_loss = StopLoss(config=config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops to trigger max loss limit first
        current_price = 96.0  # $4 loss > $3 max, but < $10 stop
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'MAX_LOSS'
        assert 'Maximum loss limit reached' in result['reason']
        assert stop_loss.is_active is False

    def test_update_max_loss_limit_not_triggered(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with maximum loss limit that doesn't trigger."""
        config = StopLossConfig(
            max_loss_value=10.0,  # $10 max loss
            stop_loss_value=0.08,  # 8% stop (wider than max loss)
        )
        stop_loss = StopLoss(config=config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price drops to $6 loss, which is < $10 max loss and < 8% stop
        current_price = 94.0  # $6 loss < $10 max, < 8% stop (8% would be 92.0)
        result = stop_loss.update(current_price, sample_timestamp)

        assert result['triggered'] is False
        assert result['action'] == 'NONE'
        assert stop_loss.is_active is True

    def test_calculate_stop_price_fixed(self, fixed_config: StopLossConfig) -> None:
        """Test _calculate_stop_price for fixed type."""
        # Use a custom config since the fixture has 0.5 which is large
        config = StopLossConfig(stop_loss_type=StopLossType.FIXED, stop_loss_value=0.2)  # 20%
        stop_loss = StopLoss(config=config)

        current_price = 100.0
        reference_price = 100.0
        stop_price = stop_loss._calculate_stop_price(current_price, reference_price)

        # For fixed type with percentage, should be reference_price * (1 - stop_loss_value)
        expected = reference_price * (1 - 0.2)  # 80.0
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_percentage(self, percentage_config: StopLossConfig) -> None:
        """Test _calculate_stop_price for percentage type."""
        stop_loss = StopLoss(config=percentage_config)

        current_price = 100.0
        reference_price = 100.0
        stop_price = stop_loss._calculate_stop_price(current_price, reference_price)

        expected = 100.0 * (1 - 0.05)  # 95.0
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_trailing(self, trailing_config: StopLossConfig) -> None:
        """Test _calculate_stop_price for trailing type."""
        stop_loss = StopLoss(config=trailing_config)

        current_price = 100.0
        reference_price = 100.0
        stop_price = stop_loss._calculate_stop_price(current_price, reference_price)

        # Should calculate based on trail distance
        expected = reference_price * (1 - 0.02)  # 98.0
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_trailing_with_existing_stop(
        self, trailing_config: StopLossConfig
    ) -> None:
        """Test _calculate_stop_price for trailing with existing stop price."""
        stop_loss = StopLoss(config=trailing_config)
        stop_loss.stop_price = 95.0  # Existing stop

        current_price = 110.0
        reference_price = 110.0
        stop_price = stop_loss._calculate_stop_price(current_price, reference_price)

        # Should move stop up, not down
        assert stop_price > 95.0

    def test_calculate_stop_price_default(self, default_config: StopLossConfig) -> None:
        """Test _calculate_stop_price with default fallback."""
        # Create config with unknown type for fallback
        config = StopLossConfig(stop_loss_type=StopLossType.PRICE)
        stop_loss = StopLoss(config=config)

        current_price = 100.0
        reference_price = 100.0
        stop_price = stop_loss._calculate_stop_price(current_price, reference_price)

        # Should fall back to 5% default
        expected = reference_price * 0.95
        assert abs(stop_price - expected) < 0.001

    def test_get_status_default(self, default_config: StopLossConfig) -> None:
        """Test get_status method with default state."""
        stop_loss = StopLoss(config=default_config)

        status = stop_loss.get_status()

        assert status['active'] is True
        assert status['triggered'] is False
        assert status['stop_price'] is None
        assert status['entry_price'] is None
        assert status['highest_price'] == 0.0
        assert status['stop_type'] == 'PERCENTAGE'  # Enum value
        assert status['stop_value'] == 0.02
        assert status['is_active'] is True
        assert status['distance_to_stop'] == 0.0
        assert 'config' in status

    def test_get_status_with_position(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test get_status with initialized position."""
        stop_loss = StopLoss(config=percentage_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Initialize the stop price first
        stop_loss._calculate_stop_price(entry_price, entry_price)

        status = stop_loss.get_status()

        assert status['entry_price'] == entry_price
        assert status['highest_price'] == entry_price
        assert status['stop_price'] is not None
        expected_distance = (entry_price - status['stop_price']) / entry_price
        assert abs(status['distance_to_stop'] - expected_distance) < 0.001

    def test_calculate_stop_price_long(self, default_config: StopLossConfig) -> None:
        """Test calculate_stop_price for long position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_price = stop_loss.calculate_stop_price(entry_price, 'long')

        # For percentage stop, should be entry * (1 - stop_loss_value)
        expected = entry_price * (1 - 0.02)
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_short(self, default_config: StopLossConfig) -> None:
        """Test calculate_stop_price for short position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_price = stop_loss.calculate_stop_price(entry_price, 'short')

        # For short position with percentage stop, should be above entry
        expected = entry_price * (1 + 0.02)
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_short_fixed(self, fixed_config: StopLossConfig) -> None:
        """Test calculate_stop_price for short position with fixed stop."""
        stop_loss = StopLoss(config=fixed_config)

        entry_price = 100.0
        stop_price = stop_loss.calculate_stop_price(entry_price, 'short')

        # For fixed stop with percentage, should return entry * (1 + stop_loss_value)
        expected = entry_price * (1 + 0.4)  # 140.0
        assert abs(stop_price - expected) < 0.001

    def test_calculate_stop_price_case_insensitive(self, default_config: StopLossConfig) -> None:
        """Test calculate_stop_price with different case sides."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        stop_price_lower = stop_loss.calculate_stop_price(entry_price, 'long')
        stop_price_upper = stop_loss.calculate_stop_price(entry_price, 'LONG')
        stop_price_mixed = stop_loss.calculate_stop_price(entry_price, 'LoNg')

        assert abs(stop_price_lower - stop_price_upper) < 0.001
        assert abs(stop_price_upper - stop_price_mixed) < 0.001

    def test_check_trigger_long_position(self, default_config: StopLossConfig) -> None:
        """Test check_trigger for long position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        # Should not trigger when above stop
        assert not stop_loss.check_trigger(entry_price, 110.0, 'long')

        # Should trigger when at or below stop
        stop_price = stop_loss.calculate_stop_price(entry_price, 'long')
        assert stop_loss.check_trigger(entry_price, stop_price, 'long')
        assert stop_loss.check_trigger(entry_price, 90.0, 'long')

    def test_check_trigger_short_position(self, default_config: StopLossConfig) -> None:
        """Test check_trigger for short position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        # Should not trigger when below stop (for short)
        assert not stop_loss.check_trigger(entry_price, 90.0, 'short')

        # Should trigger when at or above stop (for short)
        stop_price = stop_loss.calculate_stop_price(entry_price, 'short')
        assert stop_loss.check_trigger(entry_price, stop_price, 'short')
        assert stop_loss.check_trigger(entry_price, 110.0, 'short')

    def test_check_trigger_updates_entry_price(self, default_config: StopLossConfig) -> None:
        """Test check_trigger updates entry price when None."""
        stop_loss = StopLoss(config=default_config)

        assert stop_loss.entry_price is None

        entry_price = 100.0
        stop_loss.check_trigger(entry_price, 90.0, 'long')

        assert stop_loss.entry_price == entry_price

    def test_activate(self, default_config: StopLossConfig) -> None:
        """Test activate method."""
        stop_loss = StopLoss(config=default_config)
        stop_loss.is_active = False

        stop_loss.activate()

        assert stop_loss.is_active is True

    def test_deactivate(self, default_config: StopLossConfig) -> None:
        """Test deactivate method."""
        stop_loss = StopLoss(config=default_config)

        stop_loss.deactivate()

        assert stop_loss.is_active is False

    def test_trigger(self, default_config: StopLossConfig) -> None:
        """Test trigger method."""
        stop_loss = StopLoss(config=default_config)

        stop_loss.trigger()

        assert stop_loss.stop_triggered is True
        assert stop_loss.triggered is True
        assert stop_loss.is_active is False

    def test_reset(self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp) -> None:
        """Test reset method."""
        stop_loss = StopLoss(config=default_config)

        # Initialize and trigger
        stop_loss.initialize_position(100.0, sample_timestamp)
        stop_loss.update(80.0, sample_timestamp)  # Trigger stop
        assert stop_loss.triggered is True

        # Reset
        stop_loss.reset()

        assert stop_loss.activation_price == default_config.activation_price
        assert stop_loss.entry_price is None
        assert stop_loss.highest_price == 0.0
        assert stop_loss.is_active is True
        assert stop_loss.stop_triggered is False
        assert stop_loss.triggered is False
        assert stop_loss.trigger_price is None  # type: ignore[unreachable]
        assert stop_loss.stop_price is None

    def test_setup_trailing_stop_long(self, default_config: StopLossConfig) -> None:
        """Test setup_trailing_stop for long position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_loss.setup_trailing_stop(entry_price, 'long')

        assert stop_loss.entry_price == entry_price
        assert stop_loss.highest_price == entry_price
        assert stop_loss.stop_price is not None
        expected_stop = entry_price * (1 - stop_loss.trailing_stop_pct)
        assert abs(stop_loss.stop_price - expected_stop) < 0.001

    def test_setup_trailing_stop_short(self, default_config: StopLossConfig) -> None:
        """Test setup_trailing_stop for short position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_loss.setup_trailing_stop(entry_price, 'short')

        assert stop_loss.entry_price == entry_price
        assert stop_loss.highest_price == entry_price
        assert stop_loss.stop_price is not None
        expected_stop = entry_price * (1 + stop_loss.trailing_stop_pct)
        assert abs(stop_loss.stop_price - expected_stop) < 0.001

    def test_setup_trailing_stop_case_insensitive(self, default_config: StopLossConfig) -> None:
        """Test setup_trailing_stop with different case sides."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        stop_loss.setup_trailing_stop(entry_price, 'LONG')
        assert stop_loss.entry_price == entry_price

    def test_calculate_scaled_target_long(self, default_config: StopLossConfig) -> None:
        """Test calculate_scaled_target for long position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        # Test different confidence levels
        target_low = stop_loss.calculate_scaled_target(entry_price, 'low', 'long')
        target_medium = stop_loss.calculate_scaled_target(entry_price, 'medium', 'long')
        target_high = stop_loss.calculate_scaled_target(entry_price, 'high', 'long')

        assert abs(target_low - 102.0) < 0.001  # 100 * 1.02
        assert abs(target_medium - 105.0) < 0.001  # 100 * 1.05
        assert abs(target_high - 108.0) < 0.001  # 100 * 1.08

    def test_calculate_scaled_target_short(self, default_config: StopLossConfig) -> None:
        """Test calculate_scaled_target for short position."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        target_low = stop_loss.calculate_scaled_target(entry_price, 'low', 'short')
        target_medium = stop_loss.calculate_scaled_target(entry_price, 'medium', 'short')
        target_high = stop_loss.calculate_scaled_target(entry_price, 'high', 'short')

        # For short positions: entry * (2 - factor)
        assert abs(target_low - 98.0) < 0.001  # 100 * (2 - 1.02)
        assert abs(target_medium - 95.0) < 0.001  # 100 * (2 - 1.05)
        assert abs(target_high - 92.0) < 0.001  # 100 * (2 - 1.08)

    def test_calculate_scaled_target_unknown_confidence(
        self, default_config: StopLossConfig
    ) -> None:
        """Test calculate_scaled_target with unknown confidence level."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0
        target = stop_loss.calculate_scaled_target(entry_price, 'unknown', 'long')

        # Should default to medium confidence (1.05)
        assert abs(target - 105.0) < 0.001

    def test_calculate_scaled_target_case_insensitive(self, default_config: StopLossConfig) -> None:
        """Test calculate_scaled_target with different case confidence and side."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 100.0

        target1 = stop_loss.calculate_scaled_target(entry_price, 'high', 'long')
        target2 = stop_loss.calculate_scaled_target(entry_price, 'HIGH', 'LONG')

        assert abs(target1 - target2) < 0.001

    def test_update_with_stop_price_changes_logging(
        self, trailing_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test that trailing stop price changes are logged."""
        stop_loss = StopLoss(config=trailing_config)
        mock_logger = Mock(spec=logging.Logger)
        stop_loss.logger = mock_logger

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Price moves up - should update stop and log
        stop_loss.update(110.0, sample_timestamp)
        mock_logger.debug.assert_called()

        # Verify debug log was called with stop update message
        debug_calls = mock_logger.debug.call_args_list
        assert any('Trailing stop updated' in str(call) for call in debug_calls)

    def test_edge_case_zero_entry_price(self, default_config: StopLossConfig) -> None:
        """Test behavior with zero entry price."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 0.0
        result = stop_loss.check_trigger(entry_price, -1.0, 'long')

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_edge_case_negative_prices(self, default_config: StopLossConfig) -> None:
        """Test behavior with negative prices."""
        stop_loss = StopLoss(config=default_config)

        entry_price = -100.0
        result = stop_loss.check_trigger(entry_price, -110.0, 'long')

        # Should handle negative prices gracefully
        assert isinstance(result, bool)

    def test_edge_case_extreme_high_prices(self, default_config: StopLossConfig) -> None:
        """Test behavior with extremely high prices."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 1e10
        result = stop_loss.check_trigger(entry_price, 1e9, 'long')

        # Should handle large prices without overflow
        assert isinstance(result, bool)

    def test_edge_case_extreme_small_prices(self, default_config: StopLossConfig) -> None:
        """Test behavior with extremely small prices."""
        stop_loss = StopLoss(config=default_config)

        entry_price = 1e-10
        result = stop_loss.check_trigger(entry_price, 1e-11, 'long')

        # Should handle small prices without underflow
        assert isinstance(result, bool)

    def test_multiple_updates_with_no_change(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test multiple updates with no price change."""
        stop_loss = StopLoss(config=percentage_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        current_price = 99.0

        # Multiple updates with same price
        result1 = stop_loss.update(current_price, sample_timestamp)
        result2 = stop_loss.update(current_price, sample_timestamp)
        result3 = stop_loss.update(current_price, sample_timestamp)

        # All should behave the same
        assert result1['triggered'] == result2['triggered'] == result3['triggered']

    def test_rapid_price_changes(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test behavior with rapid price fluctuations."""
        stop_loss = StopLoss(config=percentage_config)

        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Rapid fluctuations - should trigger at 95.0 (5% stop from 100)
        prices = [105.0, 95.0, 98.0, 92.0, 96.0, 90.0]

        for price in prices:
            result = stop_loss.update(price, sample_timestamp)
            if result['triggered']:
                break

        # Should trigger at 95.0 (5% stop from 100)
        assert stop_loss.triggered is True
        assert stop_loss.trigger_price == 95.0

    def test_initialization_with_different_timestamps(self, default_config: StopLossConfig) -> None:
        """Test initialization with different timestamp types."""
        stop_loss = StopLoss(config=default_config)

        # Test with string timestamp
        ts1 = pd.Timestamp("2023-01-01 12:00:00")
        stop_loss.initialize_position(100.0, ts1)
        assert stop_loss.entry_price == 100.0

    def test_integration_scenario_complete_workflow(
        self, percentage_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test complete workflow from initialization to trigger."""
        stop_loss = StopLoss(config=percentage_config)

        # 1. Initialize position
        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)
        assert stop_loss.is_active is True
        assert stop_loss.triggered is False

        # 2. Check basic state
        assert stop_loss.entry_price == entry_price

        # 3. Update with favorable price movement
        result1 = stop_loss.update(105.0, sample_timestamp)
        assert result1['triggered'] is False
        assert stop_loss.is_active is True

        # 4. Update with unfavorable price movement (but not triggering)
        result2 = stop_loss.update(96.0, sample_timestamp)
        assert result2['triggered'] is False
        assert stop_loss.is_active is True

        # 5. Update with triggering price movement
        result3 = stop_loss.update(94.0, sample_timestamp)  # 6% drop triggers 5% stop
        assert result3['triggered'] is True
        assert stop_loss.is_active is False
        assert stop_loss.triggered is True  # type: ignore[unreachable]
        assert abs(result3['pnl_pct'] - (-0.06)) < 0.01

    def test_backtest_compatibility_attributes(
        self, default_config: StopLossConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test that backtest compatibility attributes work correctly."""
        stop_loss = StopLoss(config=default_config)

        # Initialize position
        entry_price = 100.0
        stop_loss.initialize_position(entry_price, sample_timestamp)

        # Test compatibility attributes
        assert hasattr(stop_loss, 'stop_loss_type')
        assert hasattr(stop_loss, 'stop_loss_value')
        assert hasattr(stop_loss, 'trailing_stop_pct')
        assert stop_loss.stop_loss_type == StopLossType.PERCENTAGE
        assert stop_loss.stop_loss_value == 0.02
        assert stop_loss.trailing_stop_pct == 0.05

        # Test triggered state compatibility
        assert hasattr(stop_loss, 'triggered')
        assert hasattr(stop_loss, 'triggered_price')
        assert hasattr(stop_loss, 'triggered_timestamp')
        assert hasattr(stop_loss, 'trigger_price')
        assert hasattr(stop_loss, 'trigger_time')

        # Initially not triggered
        assert stop_loss.triggered is False
        assert stop_loss.triggered_price is None

        # Trigger and check - note that trigger() doesn't set triggered_price
        stop_loss.trigger()
        assert stop_loss.triggered is True
        # triggered_price is only set during update() when price triggers
        assert stop_loss.triggered_price is None  # type: ignore[unreachable]

    def test_config_with_all_validation_constraints(self) -> None:
        """Test that config accepts all valid constraint combinations."""
        # Test minimum constraints
        config_min = StopLossConfig(
            stop_loss_value=0.001,
            trail_distance=0.001,
            trail_step=0.001,
            trailing_stop_pct=0.01,
        )
        assert config_min.stop_loss_value == 0.001

        # Test maximum constraints
        config_max = StopLossConfig(
            stop_loss_value=1.0,
            trail_distance=0.5,
            trail_step=0.1,
            trailing_stop_pct=0.2,
        )
        assert config_max.stop_loss_value == 1.0

        # Test boundary values
        config_boundary = StopLossConfig(
            max_loss_value=0.0,
            activation_price=0.0,
        )
        assert config_boundary.max_loss_value == 0.0
        assert config_boundary.activation_price == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
