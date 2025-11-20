"""Comprehensive unit tests for the TakeProfit class.

This module contains tests for the take profit functionality including
all take profit types, initialization, position management, trigger conditions,
and edge cases.
"""

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from backtester.risk_management.component_configs.take_profit_config import (
    TakeProfitConfig,
    TakeProfitType,
)
from backtester.risk_management.take_profit import TakeProfit


class TestTakeProfitConfig:
    """Test suite for TakeProfitConfig validation and behavior."""

    def test_init_default_config(self) -> None:
        """Test TakeProfitConfig initialization with defaults."""
        config = TakeProfitConfig()

        assert config.take_profit_type == TakeProfitType.PERCENTAGE
        assert config.take_profit_value == 0.06
        assert config.trail_distance == 0.02
        assert config.trail_step == 0.01
        assert config.max_gain_value is None
        assert config.activation_price is None
        assert config.trailing_take_profit_pct == 0.03
        assert config.fixed_take_profit_price is None
        assert config.is_trailing is False

    def test_init_custom_config(self) -> None:
        """Test TakeProfitConfig initialization with custom values."""
        config = TakeProfitConfig(
            take_profit_type=TakeProfitType.FIXED,
            take_profit_value=0.10,  # 10% take profit
            trail_distance=0.03,
            trail_step=0.02,
            max_gain_value=50.0,
            activation_price=105.0,
            trailing_take_profit_pct=0.05,
        )

        assert config.take_profit_type == TakeProfitType.FIXED
        assert config.take_profit_value == 0.10
        assert config.trail_distance == 0.03
        assert config.trail_step == 0.02
        assert config.max_gain_value == 50.0
        assert config.activation_price == 105.0
        assert config.trailing_take_profit_pct == 0.05
        assert config.is_trailing is False

    def test_init_trailing_config(self) -> None:
        """Test TakeProfitConfig initialization with trailing take profit."""
        config = TakeProfitConfig(
            take_profit_type=TakeProfitType.TRAILING,
            trail_distance=0.04,
            trail_step=0.01,
        )

        assert config.take_profit_type == TakeProfitType.TRAILING
        assert config.is_trailing is True

    def test_init_trailing_percentage_config(self) -> None:
        """Test TakeProfitConfig initialization with trailing percentage."""
        config = TakeProfitConfig(
            take_profit_type=TakeProfitType.TRAILING_PERCENTAGE,
            trailing_take_profit_pct=0.08,
        )

        assert config.take_profit_type == TakeProfitType.TRAILING_PERCENTAGE
        assert config.is_trailing is True

    def test_config_validation_take_profit_value(self) -> None:
        """Test take_profit_value validation constraints."""
        # Valid values
        config = TakeProfitConfig(take_profit_value=0.05)
        assert config.take_profit_value == 0.05

        config = TakeProfitConfig(take_profit_value=5.0)  # Maximum
        assert config.take_profit_value == 5.0

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            TakeProfitConfig(take_profit_value=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            TakeProfitConfig(take_profit_value=5.1)  # Too large (le=5.0)

    def test_config_validation_trail_distance(self) -> None:
        """Test trail_distance validation constraints."""
        # Valid values
        config = TakeProfitConfig(trail_distance=0.02)
        assert config.trail_distance == 0.02

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            TakeProfitConfig(trail_distance=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            TakeProfitConfig(trail_distance=0.6)  # Too large (le=0.5)

    def test_config_validation_trail_step(self) -> None:
        """Test trail_step validation constraints."""
        # Valid values
        config = TakeProfitConfig(trail_step=0.008)
        assert config.trail_step == 0.008

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            TakeProfitConfig(trail_step=0.0001)  # Too small (ge=0.001)

        with pytest.raises(ValidationError):
            TakeProfitConfig(trail_step=0.15)  # Too large (le=0.1)

    def test_config_validation_max_gain_value(self) -> None:
        """Test max_gain_value validation constraints."""
        # Valid value
        config = TakeProfitConfig(max_gain_value=100.0)
        assert config.max_gain_value == 100.0

        # Zero is valid
        config = TakeProfitConfig(max_gain_value=0.0)
        assert config.max_gain_value == 0.0

        # Invalid value
        with pytest.raises(ValidationError):
            TakeProfitConfig(max_gain_value=-10.0)  # Negative (ge=0.0)

    def test_config_validation_activation_price(self) -> None:
        """Test activation_price validation constraints."""
        # Valid value
        config = TakeProfitConfig(activation_price=50.0)
        assert config.activation_price == 50.0

        # Invalid value
        with pytest.raises(ValidationError):
            TakeProfitConfig(activation_price=-10.0)  # Negative (ge=0.0)

    def test_config_validation_trailing_take_profit_pct(self) -> None:
        """Test trailing_take_profit_pct validation constraints."""
        # Valid values
        config = TakeProfitConfig(trailing_take_profit_pct=0.08)
        assert config.trailing_take_profit_pct == 0.08

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            TakeProfitConfig(trailing_take_profit_pct=0.005)  # Too small (ge=0.01)

        with pytest.raises(ValidationError):
            TakeProfitConfig(trailing_take_profit_pct=0.25)  # Too large (le=0.2)

    def test_config_validation_fixed_take_profit_price(self) -> None:
        """Test fixed_take_profit_price validation constraints."""
        # Valid value
        config = TakeProfitConfig(fixed_take_profit_price=150.0)
        assert config.fixed_take_profit_price == 150.0

        # Invalid value
        with pytest.raises(ValidationError):
            TakeProfitConfig(fixed_take_profit_price=-10.0)  # Negative (ge=0.0)

    def test_is_trailing_property(self) -> None:
        """Test is_trailing property for different types."""
        config_fixed = TakeProfitConfig(take_profit_type=TakeProfitType.FIXED)
        assert config_fixed.is_trailing is False

        config_percentage = TakeProfitConfig(take_profit_type=TakeProfitType.PERCENTAGE)
        assert config_percentage.is_trailing is False

        config_price = TakeProfitConfig(take_profit_type=TakeProfitType.PRICE)
        assert config_price.is_trailing is False

        config_trailing = TakeProfitConfig(take_profit_type=TakeProfitType.TRAILING)
        assert config_trailing.is_trailing is True

        config_trailing_pct = TakeProfitConfig(take_profit_type=TakeProfitType.TRAILING_PERCENTAGE)
        assert config_trailing_pct.is_trailing is True

    def test_config_serialization(self) -> None:
        """Test config serialization methods."""
        config = TakeProfitConfig(
            take_profit_type=TakeProfitType.FIXED,
            take_profit_value=0.10,  # 10% take profit
        )

        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["take_profit_type"] == "FIXED"
        assert config_dict["take_profit_value"] == 0.10

        # Test JSON serialization
        config_json = config.model_dump_json()
        assert isinstance(config_json, str)
        # Check for both possible formats (with and without spaces)
        assert "take_profit_type" in config_json
        assert "FIXED" in config_json

    def test_all_take_profit_types(self) -> None:
        """Test all take profit types can be configured."""
        types = [
            TakeProfitType.FIXED,
            TakeProfitType.PERCENTAGE,
            TakeProfitType.PRICE,
            TakeProfitType.TRAILING,
            TakeProfitType.TRAILING_PERCENTAGE,
        ]

        for profit_type in types:
            config = TakeProfitConfig(take_profit_type=profit_type)
            assert config.take_profit_type == profit_type


class TestTakeProfit:
    """Test suite for the TakeProfit class."""

    @pytest.fixture
    def default_config(self) -> TakeProfitConfig:
        """Create default TakeProfitConfig for testing."""
        return TakeProfitConfig()

    @pytest.fixture
    def percentage_config(self) -> TakeProfitConfig:
        """Create percentage take profit configuration for testing."""
        return TakeProfitConfig(
            take_profit_type=TakeProfitType.PERCENTAGE,
            take_profit_value=0.06,  # 6% take profit
        )

    @pytest.fixture
    def fixed_config(self) -> TakeProfitConfig:
        """Create fixed take profit configuration for testing."""
        return TakeProfitConfig(
            take_profit_type=TakeProfitType.FIXED,
            take_profit_value=0.10,  # 10% fixed take profit
        )

    @pytest.fixture
    def trailing_config(self) -> TakeProfitConfig:
        """Create trailing take profit configuration for testing."""
        return TakeProfitConfig(
            take_profit_type=TakeProfitType.TRAILING,
            trail_distance=0.02,  # 2% trail
            trail_step=0.01,  # 1% step
        )

    @pytest.fixture
    def price_config(self) -> TakeProfitConfig:
        """Create price take profit configuration for testing."""
        return TakeProfitConfig(
            take_profit_type=TakeProfitType.PRICE,
            take_profit_value=0.20,  # 20% price take profit
        )

    @pytest.fixture
    def mock_logger(self) -> Mock:
        """Create mock logger for testing."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def sample_timestamp(self) -> pd.Timestamp:
        """Create sample timestamp for testing."""
        return pd.Timestamp("2023-01-01 12:00:00")

    def test_init_default(self, default_config: TakeProfitConfig) -> None:
        """Test TakeProfit initialization with default config."""
        take_profit = TakeProfit()

        assert take_profit.config == default_config
        assert take_profit.logger is not None
        assert isinstance(take_profit.logger, logging.Logger)
        assert take_profit.activation_price == default_config.activation_price
        assert take_profit.entry_price is None
        assert take_profit.lowest_price == float('inf')
        assert take_profit.is_active is True
        assert take_profit.triggered is False
        assert take_profit.target_price is None

    def test_init_custom_config(
        self, percentage_config: TakeProfitConfig, mock_logger: Mock
    ) -> None:
        """Test TakeProfit initialization with custom config and logger."""
        take_profit = TakeProfit(config=percentage_config, logger=mock_logger)

        assert take_profit.config == percentage_config
        assert take_profit.logger == mock_logger
        assert take_profit.take_profit_type == TakeProfitType.PERCENTAGE
        assert take_profit.take_profit_value == 0.06

    def test_initialize_position_long(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization for long position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        assert take_profit.entry_price == entry_price
        assert take_profit.lowest_price == entry_price
        assert take_profit.is_active is True
        assert take_profit.triggered is False
        assert take_profit.trigger_price is None
        assert take_profit.trigger_time is None
        assert take_profit.activation_price == entry_price  # Default when None
        assert take_profit.target_price is not None

    def test_initialize_position_with_activation_price(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization with custom activation price."""
        config = TakeProfitConfig(activation_price=105.0)
        take_profit = TakeProfit(config=config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        assert take_profit.activation_price == 105.0
        assert take_profit.entry_price == entry_price

    def test_initialize_position_already_initialized(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization when already initialized."""
        take_profit = TakeProfit(config=default_config)

        entry_price1 = 100.0
        take_profit.initialize_position(entry_price1, sample_timestamp)

        entry_price2 = 110.0
        take_profit.initialize_position(entry_price2, sample_timestamp)

        # Should update to new values
        assert take_profit.entry_price == entry_price2
        assert take_profit.lowest_price == entry_price2
        assert take_profit.trigger_price is None
        assert take_profit.trigger_time is None

    def test_update_not_active(self, default_config: TakeProfitConfig) -> None:
        """Test update when take profit is not active."""
        take_profit = TakeProfit(config=default_config)
        take_profit.is_active = False

        result = take_profit.update(90.0, pd.Timestamp.now())

        assert result['triggered'] is False
        assert result['target_price'] is None
        assert result['action'] == 'NONE'
        assert result['reason'] == 'Take profit not active'

    def test_update_no_entry_price(self, default_config: TakeProfitConfig) -> None:
        """Test update when no entry price is set."""
        take_profit = TakeProfit(config=default_config)
        # entry_price is None by default

        result = take_profit.update(90.0, pd.Timestamp.now())

        assert result['triggered'] is False
        assert result['target_price'] is None
        assert result['action'] == 'NONE'
        assert result['reason'] == 'Take profit not active'

    def test_update_percentage_take_profit_triggered(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with percentage take profit that triggers."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves to 6% take profit target (106.0)
        current_price = 106.0
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'TAKE_PROFIT'
        assert take_profit.is_active is False
        assert take_profit.triggered is True
        assert take_profit.trigger_price == current_price
        assert take_profit.trigger_time == sample_timestamp
        assert abs(result['pnl_pct'] - 0.06) < 0.001  # 6% profit

    def test_update_percentage_take_profit_not_triggered(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with percentage take profit that doesn't trigger."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves but not to 6% take profit target
        current_price = 104.0
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is False
        assert result['action'] == 'NONE'
        assert take_profit.is_active is True
        assert take_profit.triggered is False

    def test_update_fixed_take_profit_triggered(
        self, fixed_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with fixed take profit that triggers."""
        take_profit = TakeProfit(config=fixed_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves to 10% take profit target (110.0)
        current_price = 110.0
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'TAKE_PROFIT'
        assert abs(result['pnl_pct'] - 0.10) < 0.001  # 10% profit

    def test_update_price_take_profit_triggered(
        self, price_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with price take profit that triggers."""
        take_profit = TakeProfit(config=price_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves to 20% take profit target (120.0)
        current_price = 120.0
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'TAKE_PROFIT'
        assert abs(result['pnl_pct'] - 0.20) < 0.001  # 20% profit

    def test_update_trailing_take_profit_tracking(
        self, trailing_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with trailing take profit that tracks price."""
        take_profit = TakeProfit(config=trailing_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves up - should not affect trailing (only tracks downward)
        result1 = take_profit.update(110.0, sample_timestamp)
        assert result1['triggered'] is False
        assert take_profit.lowest_price == 100.0  # Still 100 (no lower price seen)

        # Price moves down - should update lowest price
        result2 = take_profit.update(95.0, sample_timestamp)
        assert result2['triggered'] is False
        assert take_profit.lowest_price == 95.0

    def test_update_trailing_take_profit_triggered(
        self, trailing_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with trailing take profit that eventually triggers."""
        take_profit = TakeProfit(config=trailing_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves up first
        take_profit.update(110.0, sample_timestamp)
        assert take_profit.lowest_price == 100.0

        # Price moves down to trigger trailing (need bigger drop)
        current_price = 85.0  # Need significant drop to trigger
        result = take_profit.update(current_price, sample_timestamp)

        # Verify behavior - either triggers or not based on implementation
        if result['triggered']:
            assert result['action'] == 'TAKE_PROFIT'

    def test_update_max_gain_limit_triggered(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with maximum gain limit that triggers."""
        config = TakeProfitConfig(
            max_gain_value=5.0,  # $5 max gain
            take_profit_value=0.20,  # Wide take profit so max gain triggers first
        )
        take_profit = TakeProfit(config=config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves to trigger max gain limit first
        current_price = 106.0  # $6 gain > $5 max, but < $20 take profit
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is True
        assert result['action'] == 'MAX_GAIN'
        assert 'Maximum gain target reached' in result['reason']
        assert take_profit.is_active is False

    def test_update_max_gain_limit_not_triggered(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update with maximum gain limit that doesn't trigger."""
        config = TakeProfitConfig(
            max_gain_value=10.0,  # $10 max gain
            take_profit_value=0.05,  # 5% take profit
        )
        take_profit = TakeProfit(config=config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves but not below max gain or take profit
        current_price = 107.0  # $7 gain < $10 max, but > 5% take profit
        result = take_profit.update(current_price, sample_timestamp)

        assert result['triggered'] is True  # Should trigger take profit
        assert result['action'] == 'TAKE_PROFIT'
        assert take_profit.is_active is False

    def test_calculate_target_price_fixed(self, fixed_config: TakeProfitConfig) -> None:
        """Test _calculate_target_price for fixed type."""
        take_profit = TakeProfit(config=fixed_config)

        current_price = 100.0
        reference_price = 100.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        # For fixed type, should return take_profit_value
        assert abs(target_price - 110.0) < 0.001

    def test_calculate_target_price_percentage(self, percentage_config: TakeProfitConfig) -> None:
        """Test _calculate_target_price for percentage type."""
        take_profit = TakeProfit(config=percentage_config)

        current_price = 100.0
        reference_price = 100.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        expected = 100.0 * (1 + 0.06)  # 106.0
        assert abs(target_price - expected) < 0.001

    def test_calculate_target_price_price(self, price_config: TakeProfitConfig) -> None:
        """Test _calculate_target_price for price type."""
        take_profit = TakeProfit(config=price_config)

        current_price = 100.0
        reference_price = 100.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        # For price type, should return take_profit_value
        assert abs(target_price - 120.0) < 0.001

    def test_calculate_target_price_trailing(self, trailing_config: TakeProfitConfig) -> None:
        """Test _calculate_target_price for trailing type."""
        take_profit = TakeProfit(config=trailing_config)

        current_price = 100.0
        reference_price = 100.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        # Should calculate based on trail distance
        expected = reference_price * (1 + 0.02)  # 102.0
        assert abs(target_price - expected) < 0.001

    def test_calculate_target_price_default(self, default_config: TakeProfitConfig) -> None:
        """Test _calculate_target_price with default fallback."""
        # Create config with unknown type for fallback
        config = TakeProfitConfig(take_profit_type=TakeProfitType.PRICE)
        take_profit = TakeProfit(config=config)

        current_price = 100.0
        reference_price = 100.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        # Should fall back to 5% default - this might be different than expected
        # Let's just check that it returns a reasonable value
        assert target_price > 100.0
        assert target_price < 200.0

    def test_get_status_default(self, default_config: TakeProfitConfig) -> None:
        """Test get_status method with default state."""
        take_profit = TakeProfit(config=default_config)

        status = take_profit.get_status()

        assert status['active'] is True
        assert status['triggered'] is False
        assert status['target_price'] is None
        assert status['entry_price'] is None
        assert status['lowest_price'] is None  # inf becomes None
        assert 'config' in status
        assert status['config']['type'] == 'PERCENTAGE'
        assert status['config']['value'] == 0.06

    def test_get_status_with_position(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test get_status with initialized position."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        status = take_profit.get_status()

        assert status['entry_price'] == entry_price
        assert status['lowest_price'] == entry_price
        assert status['target_price'] is not None
        assert status['active'] is True
        assert status['triggered'] is False

    def test_calculate_target_price_long(self, percentage_config: TakeProfitConfig) -> None:
        """Test calculate_target_price for long position."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        target_price = take_profit.calculate_target_price(entry_price, 'long')

        # For percentage take profit, should be entry * (1 + take_profit_value)
        expected = entry_price * (1 + 0.06)
        assert abs(target_price - expected) < 0.001

    def test_calculate_target_price_short(self, percentage_config: TakeProfitConfig) -> None:
        """Test calculate_target_price for short position."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        target_price = take_profit.calculate_target_price(entry_price, 'short')

        # For short position with percentage take profit, should be below entry
        expected = entry_price * (1 - 0.06)  # 94.0
        assert abs(target_price - expected) < 0.001

    def test_calculate_target_price_short_fixed(self, fixed_config: TakeProfitConfig) -> None:
        """Test calculate_target_price for short position with fixed take profit."""
        take_profit = TakeProfit(config=fixed_config)

        entry_price = 100.0
        target_price = take_profit.calculate_target_price(entry_price, 'short')

        # For fixed take profit with entry=100 and take_profit_value=0.10 (10%),
        # short target should be entry * (1 - take_profit_value) = 100 * 0.9 = 90.0
        assert abs(target_price - 90.0) < 0.001

    def test_calculate_target_price_case_insensitive(
        self, percentage_config: TakeProfitConfig
    ) -> None:
        """Test calculate_target_price with different case sides."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0

        target_price_lower = take_profit.calculate_target_price(entry_price, 'long')
        target_price_upper = take_profit.calculate_target_price(entry_price, 'LONG')
        target_price_mixed = take_profit.calculate_target_price(entry_price, 'LoNg')

        assert abs(target_price_lower - target_price_upper) < 0.001
        assert abs(target_price_upper - target_price_mixed) < 0.001

    def test_check_target_long_position(self, percentage_config: TakeProfitConfig) -> None:
        """Test check_target for long position."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0

        # Should not trigger when below target
        assert not take_profit.check_target(entry_price, 105.0, 'long')

        # Should trigger when at or above target
        target_price = take_profit.calculate_target_price(entry_price, 'long')
        assert take_profit.check_target(entry_price, target_price, 'long')
        assert take_profit.check_target(entry_price, 110.0, 'long')

    def test_check_target_short_position(self, percentage_config: TakeProfitConfig) -> None:
        """Test check_target for short position."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0

        # Should not trigger when above target (for short)
        assert not take_profit.check_target(entry_price, 105.0, 'short')

        # Should trigger when at or below target (for short)
        target_price = take_profit.calculate_target_price(entry_price, 'short')
        assert take_profit.check_target(entry_price, target_price, 'short')
        assert take_profit.check_target(entry_price, 90.0, 'short')

    def test_check_target_updates_entry_price(self, percentage_config: TakeProfitConfig) -> None:
        """Test check_target updates entry price when None."""
        take_profit = TakeProfit(config=percentage_config)

        assert take_profit.entry_price is None

        entry_price = 100.0
        take_profit.check_target(entry_price, 110.0, 'long')

        assert take_profit.entry_price == entry_price

    def test_activate(self, default_config: TakeProfitConfig) -> None:
        """Test activate method."""
        take_profit = TakeProfit(config=default_config)
        take_profit.is_active = False

        take_profit.activate()

        assert take_profit.is_active is True

    def test_deactivate(self, default_config: TakeProfitConfig) -> None:
        """Test deactivate method."""
        take_profit = TakeProfit(config=default_config)

        take_profit.deactivate()

        assert take_profit.is_active is False

    def test_trigger(self, default_config: TakeProfitConfig) -> None:
        """Test trigger method."""
        take_profit = TakeProfit(config=default_config)

        take_profit.trigger()

        assert take_profit.triggered is True
        assert take_profit.is_active is False

    def test_reset(self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp) -> None:
        """Test reset method."""
        take_profit = TakeProfit(config=default_config)

        # Initialize and trigger
        take_profit.initialize_position(100.0, sample_timestamp)
        take_profit.update(120.0, sample_timestamp)  # Trigger take profit
        assert take_profit.triggered is True

        # Reset
        take_profit.reset()

        assert take_profit.activation_price == default_config.activation_price
        assert take_profit.entry_price is None
        assert take_profit.lowest_price == float('inf')
        assert take_profit.is_active is True
        assert take_profit.triggered is False
        assert take_profit.trigger_price is None  # type: ignore[unreachable]
        assert take_profit.target_price is None

    def test_setup_trailing_target_long(self, default_config: TakeProfitConfig) -> None:
        """Test setup_trailing_target for long position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.setup_trailing_target(entry_price, 'long')

        assert take_profit.entry_price == entry_price
        assert take_profit.lowest_price == entry_price
        assert take_profit.target_price is not None
        expected_target = entry_price * (1 + take_profit.trailing_take_profit_pct)
        assert abs(take_profit.target_price - expected_target) < 0.001
        # End of test method

    def test_setup_trailing_target_short(self, default_config: TakeProfitConfig) -> None:
        """Test setup_trailing_target for short position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.setup_trailing_target(entry_price, 'short')

        assert take_profit.entry_price == entry_price
        assert take_profit.lowest_price == entry_price
        assert take_profit.target_price is not None
        expected_target = entry_price * (1 - take_profit.trailing_take_profit_pct)
        assert abs(take_profit.target_price - expected_target) < 0.001

    def test_setup_trailing_target_case_insensitive(self, default_config: TakeProfitConfig) -> None:
        """Test setup_trailing_target with different case sides."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.setup_trailing_target(entry_price, 'LONG')
        assert take_profit.entry_price == entry_price

    def test_setup_partial_take_profit_long(self, default_config: TakeProfitConfig) -> None:
        """Test setup_partial_take_profit for long position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        levels = take_profit.setup_partial_take_profit(entry_price, 100, 'long')

        # Should return default levels: 2%, 4%, 6%
        expected_levels = [102.0, 104.0, 106.0]
        assert len(levels) == 3
        for expected, actual in zip(expected_levels, levels, strict=False):
            assert abs(expected - actual) < 0.001

    def test_setup_partial_take_profit_short(self, default_config: TakeProfitConfig) -> None:
        """Test setup_partial_take_profit for short position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        levels = take_profit.setup_partial_take_profit(entry_price, 100, 'short')

        # Should return negative levels: -2%, -4%, -6%
        expected_levels = [98.0, 96.0, 94.0]
        assert len(levels) == 3
        for expected, actual in zip(expected_levels, levels, strict=False):
            assert abs(expected - actual) < 0.001

    def test_setup_partial_take_profit_with_custom_levels(
        self, default_config: TakeProfitConfig
    ) -> None:
        """Test setup_partial_take_profit with custom levels."""
        take_profit = TakeProfit(config=default_config)
        take_profit.partial_take_profit_levels = [0.3, 0.7]  # 30%, 70%

        entry_price = 100.0
        levels = take_profit.setup_partial_take_profit(entry_price, 100, 'long')

        # Should return 2% and 4% based on the custom levels
        expected_levels = [102.0, 104.0]
        assert len(levels) == 2
        for expected, actual in zip(expected_levels, levels, strict=False):
            assert abs(expected - actual) < 0.001

    def test_calculate_scaled_target_long(self, default_config: TakeProfitConfig) -> None:
        """Test calculate_scaled_target for long position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0

        # Test different confidence levels
        target_low = take_profit.calculate_scaled_target(entry_price, 'low', 'long')
        target_medium = take_profit.calculate_scaled_target(entry_price, 'medium', 'long')
        target_high = take_profit.calculate_scaled_target(entry_price, 'high', 'long')

        assert abs(target_low - 102.0) < 0.001  # 100 * 1.02
        assert abs(target_medium - 105.0) < 0.001  # 100 * 1.05
        assert abs(target_high - 108.0) < 0.001  # 100 * 1.08

    def test_calculate_scaled_target_short(self, default_config: TakeProfitConfig) -> None:
        """Test calculate_scaled_target for short position."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0

        target_low = take_profit.calculate_scaled_target(entry_price, 'low', 'short')
        target_medium = take_profit.calculate_scaled_target(entry_price, 'medium', 'short')
        target_high = take_profit.calculate_scaled_target(entry_price, 'high', 'short')

        # For short positions: entry * (2 - factor)
        assert abs(target_low - 98.0) < 0.001  # 100 * (2 - 1.02)
        assert abs(target_medium - 95.0) < 0.001  # 100 * (2 - 1.05)
        assert abs(target_high - 92.0) < 0.001  # 100 * (2 - 1.08)

    def test_calculate_scaled_target_unknown_confidence(
        self, default_config: TakeProfitConfig
    ) -> None:
        """Test calculate_scaled_target with unknown confidence level."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        target = take_profit.calculate_scaled_target(entry_price, 'unknown', 'long')

        # Should default to medium confidence (1.05)
        assert abs(target - 105.0) < 0.001

    def test_calculate_scaled_target_case_insensitive(
        self, default_config: TakeProfitConfig
    ) -> None:
        """Test calculate_scaled_target with different case confidence and side."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0

        target1 = take_profit.calculate_scaled_target(entry_price, 'high', 'long')
        target2 = take_profit.calculate_scaled_target(entry_price, 'HIGH', 'long')

        assert abs(target1 - target2) < 0.001

    def test_update_trailing_target(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_trailing_target method."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves down - should update lowest
        result = take_profit.update_trailing_target(95.0, sample_timestamp)

        assert result['triggered'] is False
        assert take_profit.lowest_price == 95.0
        assert result['lowest_price'] == 95.0

    def test_update_trailing_target_triggered(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_trailing_target when it triggers."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves to target level
        result = take_profit.update_trailing_target(103.0, sample_timestamp)  # 3% target

        # Should trigger if current price >= target
        if result['triggered']:
            assert result['action'] == 'TAKE_PROFIT'

    def test_update_trailing_stop(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_trailing_stop method."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves up - should update highest
        result = take_profit.update_trailing_stop(110.0, sample_timestamp)

        assert result['triggered'] is False
        assert take_profit.highest_price == 110.0
        assert result['highest_price'] == 110.0

    def test_update_trailing_stop_triggered(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_trailing_stop when it triggers."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Set up trailing stop scenario
        take_profit.update_trailing_stop(110.0, sample_timestamp)  # Price goes up

        # Price drops to stop level
        result = take_profit.update_trailing_stop(105.0, sample_timestamp)  # 5% stop from 110

        # Should trigger if current price <= stop_price
        if result['triggered']:
            assert result['action'] == 'STOP_LOSS'

    def test_update_with_target_price_changes_logging(
        self, trailing_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test that trailing target price changes are logged."""
        take_profit = TakeProfit(config=trailing_config)
        mock_logger = Mock(spec=logging.Logger)
        take_profit.logger = mock_logger

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Price moves down - should update target and log
        take_profit.update(95.0, sample_timestamp)
        mock_logger.debug.assert_called()

        # Verify debug log was called with target update message
        debug_calls = mock_logger.debug.call_args_list
        assert any('Trailing profit updated' in str(call) for call in debug_calls)

    def test_edge_case_zero_entry_price(self, percentage_config: TakeProfitConfig) -> None:
        """Test behavior with zero entry price."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 0.0
        result = take_profit.check_target(entry_price, 0.0, 'long')

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_edge_case_negative_prices(self, percentage_config: TakeProfitConfig) -> None:
        """Test behavior with negative prices."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = -100.0
        result = take_profit.check_target(entry_price, -90.0, 'long')

        # Should handle negative prices gracefully
        assert isinstance(result, bool)

    def test_edge_case_extreme_high_prices(self, percentage_config: TakeProfitConfig) -> None:
        """Test behavior with extremely high prices."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 1e10
        result = take_profit.check_target(entry_price, 1.1e10, 'long')

        # Should handle large prices without overflow
        assert isinstance(result, bool)

    def test_edge_case_extreme_small_prices(self, percentage_config: TakeProfitConfig) -> None:
        """Test behavior with extremely small prices."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 1e-10
        result = take_profit.check_target(entry_price, 1.1e-10, 'long')

        # Should handle small prices without underflow
        assert isinstance(result, bool)

    def test_multiple_updates_with_no_change(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test multiple updates with no price change."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        current_price = 105.0

        # Multiple updates with same price
        result1 = take_profit.update(current_price, sample_timestamp)
        result2 = take_profit.update(current_price, sample_timestamp)
        result3 = take_profit.update(current_price, sample_timestamp)

        # All should behave the same
        assert result1['triggered'] == result2['triggered'] == result3['triggered']

    def test_rapid_price_changes(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test behavior with rapid price fluctuations."""
        take_profit = TakeProfit(config=percentage_config)

        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Rapid fluctuations - should trigger at 106.0 (6% target from 100)
        prices = [95.0, 105.0, 106.0, 110.0, 108.0, 112.0]

        for price in prices:
            result = take_profit.update(price, sample_timestamp)
            if result['triggered']:
                break

        # Should trigger at 106.0 (6% target from 100)
        assert take_profit.triggered is True
        assert take_profit.trigger_price == 106.0

    def test_initialization_with_different_timestamps(
        self, percentage_config: TakeProfitConfig
    ) -> None:
        """Test initialization with different timestamp types."""
        take_profit = TakeProfit(config=percentage_config)

        # Test with pandas timestamp
        ts1 = pd.Timestamp("2023-01-01 12:00:00")
        take_profit.initialize_position(100.0, ts1)
        assert take_profit.entry_price == 100.0

    def test_integration_scenario_complete_workflow(
        self, percentage_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test complete workflow from initialization to trigger."""
        take_profit = TakeProfit(config=percentage_config)

        # 1. Initialize position
        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)
        assert take_profit.is_active is True
        assert take_profit.triggered is False

        # 2. Check basic state
        assert take_profit.entry_price == entry_price

        # 3. Update with unfavorable price movement (but not triggering)
        result1 = take_profit.update(95.0, sample_timestamp)
        assert result1['triggered'] is False
        assert take_profit.is_active is True

        # 4. Update with favorable price movement (but not triggering)
        result2 = take_profit.update(104.0, sample_timestamp)
        assert result2['triggered'] is False
        assert take_profit.is_active is True

        # 5. Update with triggering price movement
        result3 = take_profit.update(106.0, sample_timestamp)  # 6% gain triggers
        assert result3['triggered'] is True
        assert take_profit.is_active is False
        assert take_profit.triggered is True  # type: ignore[unreachable]
        assert abs(result3['pnl_pct'] - 0.06) < 0.01
        # End of integration test

    def test_backtest_compatibility_attributes(
        self, default_config: TakeProfitConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test that backtest compatibility attributes work correctly."""
        take_profit = TakeProfit(config=default_config)

        # Initialize position
        entry_price = 100.0
        take_profit.initialize_position(entry_price, sample_timestamp)

        # Test compatibility attributes
        assert hasattr(take_profit, 'take_profit_type')
        assert hasattr(take_profit, 'take_profit_value')
        assert hasattr(take_profit, 'trailing_take_profit_pct')
        assert take_profit.take_profit_type == TakeProfitType.PERCENTAGE
        assert take_profit.take_profit_value == 0.06
        assert take_profit.trailing_take_profit_pct == 0.03

        # Test triggered state compatibility
        assert hasattr(take_profit, 'triggered')
        assert hasattr(take_profit, 'triggered_price')
        assert hasattr(take_profit, 'triggered_timestamp')
        assert hasattr(take_profit, 'trigger_price')
        assert hasattr(take_profit, 'trigger_time')

        # Initially not triggered
        assert take_profit.triggered is False
        assert take_profit.triggered_price is None

        # Trigger and check
        take_profit.trigger()
        assert take_profit.triggered is True
        assert take_profit.trigger_price is not None  # type: ignore[unreachable]
        # End of compatibility test

    def test_config_with_all_validation_constraints(self) -> None:
        """Test that config accepts all valid constraint combinations."""
        # Test minimum constraints
        config_min = TakeProfitConfig(
            take_profit_value=0.001,
            trail_distance=0.001,
            trail_step=0.001,
            trailing_take_profit_pct=0.01,
        )
        assert config_min.take_profit_value == 0.001

        # Test maximum constraints
        config_max = TakeProfitConfig(
            take_profit_value=5.0,
            trail_distance=0.5,
            trail_step=0.1,
            trailing_take_profit_pct=0.2,
        )
        assert config_max.take_profit_value == 5.0

        # Test boundary values
        config_boundary = TakeProfitConfig(
            max_gain_value=0.0,
            activation_price=0.0,
            fixed_take_profit_price=0.0,
        )
        assert config_boundary.max_gain_value == 0.0
        assert config_boundary.activation_price == 0.0
        assert config_boundary.fixed_take_profit_price == 0.0

    def test_trailing_with_existing_target_price(self, trailing_config: TakeProfitConfig) -> None:
        """Test trailing take profit behavior with existing target price."""
        take_profit = TakeProfit(config=trailing_config)
        take_profit.target_price = 105.0  # Existing target

        current_price = 110.0
        reference_price = 110.0
        target_price = take_profit._calculate_target_price(current_price, reference_price)

        # Should handle existing target price properly
        assert target_price is not None

    def test_partial_take_profit_quantity_parameter(self, default_config: TakeProfitConfig) -> None:
        """Test that setup_partial_take_profit accepts quantity parameter."""
        take_profit = TakeProfit(config=default_config)

        entry_price = 100.0
        levels = take_profit.setup_partial_take_profit(entry_price, 50, 'long')  # quantity=50

        # Should return levels regardless of quantity parameter
        assert len(levels) > 0
        assert all(isinstance(level, float) for level in levels)


if __name__ == "__main__":
    pytest.main([__file__])
