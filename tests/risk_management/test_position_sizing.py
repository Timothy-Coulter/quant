"""Comprehensive unit tests for the PositionSizer class.

This module contains tests for the position sizing functionality including
all sizing methods, edge cases, error conditions, and boundary conditions.
"""

import logging
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from backtester.risk_management.component_configs.position_sizing_config import (
    PositionSizingConfig,
    SizingMethod,
)
from backtester.risk_management.position_sizing import PositionSizer


class TestPositionSizingConfig:
    """Test suite for PositionSizingConfig validation and behavior."""

    def test_init_default_config(self) -> None:
        """Test PositionSizingConfig initialization with defaults."""
        config = PositionSizingConfig()

        assert config.max_position_size == 0.10
        assert config.min_position_size == 0.01
        assert config.risk_per_trade == 0.02
        assert config.sizing_method == SizingMethod.FIXED_PERCENTAGE
        assert config.max_daily_trades == 5
        assert config.max_sector_exposure == 0.30
        assert config.max_correlation == 0.80
        assert config.volatility_adjustment is True
        assert config.conviction_factors == {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        assert config.kelly_win_rate is None
        assert config.kelly_avg_win is None
        assert config.kelly_avg_loss is None

    def test_init_custom_config(self) -> None:
        """Test PositionSizingConfig initialization with custom values."""
        config = PositionSizingConfig(
            max_position_size=0.20,
            min_position_size=0.005,
            risk_per_trade=0.01,
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.6,
            kelly_avg_win=100.0,
            kelly_avg_loss=50.0,
            conviction_factors={'low': 0.5, 'medium': 1.0, 'high': 1.5},
        )

        assert config.max_position_size == 0.20
        assert config.min_position_size == 0.005
        assert config.risk_per_trade == 0.01
        assert config.sizing_method == SizingMethod.KELLY
        assert config.kelly_win_rate == 0.6
        assert config.kelly_avg_win == 100.0
        assert config.kelly_avg_loss == 50.0
        assert config.conviction_factors == {'low': 0.5, 'medium': 1.0, 'high': 1.5}

    def test_config_validation_max_position_size(self) -> None:
        """Test that max_position_size validation works."""
        # Valid values
        config = PositionSizingConfig(max_position_size=0.15)
        assert config.max_position_size == 0.15

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            PositionSizingConfig(max_position_size=0.001)  # Too small

        with pytest.raises(ValidationError):
            PositionSizingConfig(max_position_size=1.5)  # Too large

    def test_config_validation_min_position_size(self) -> None:
        """Test that min_position_size validation works."""
        # Valid values
        config = PositionSizingConfig(min_position_size=0.005)
        assert config.min_position_size == 0.005

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            PositionSizingConfig(min_position_size=0.0001)  # Too small

        with pytest.raises(ValidationError):
            PositionSizingConfig(min_position_size=0.2)  # Too large

    def test_config_validation_risk_per_trade(self) -> None:
        """Test that risk_per_trade validation works."""
        # Valid values
        config = PositionSizingConfig(risk_per_trade=0.05)
        assert config.risk_per_trade == 0.05

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            PositionSizingConfig(risk_per_trade=0.0001)  # Too small

        with pytest.raises(ValidationError):
            PositionSizingConfig(risk_per_trade=0.2)  # Too large

    def test_kelly_criterion_validation(self) -> None:
        """Test Kelly criterion parameter validation."""
        # Valid Kelly parameters
        config = PositionSizingConfig(kelly_win_rate=0.5, kelly_avg_win=100.0, kelly_avg_loss=50.0)
        assert config.kelly_win_rate == 0.5
        assert config.kelly_avg_win == 100.0
        assert config.kelly_avg_loss == 50.0

        # Invalid Kelly parameters
        with pytest.raises(ValidationError):
            PositionSizingConfig(kelly_win_rate=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            PositionSizingConfig(kelly_avg_win=-10.0)  # Negative

    def test_requires_historical_data(self) -> None:
        """Test requires_historical_data property."""
        config_kelly = PositionSizingConfig(sizing_method=SizingMethod.KELLY)
        assert config_kelly.requires_historical_data is True

        config_volatility = PositionSizingConfig(sizing_method=SizingMethod.VOLATILITY_ADJUSTED)
        assert config_volatility.requires_historical_data is True

        config_fixed = PositionSizingConfig(sizing_method=SizingMethod.FIXED_PERCENTAGE)
        assert config_fixed.requires_historical_data is False

    def test_requires_correlation_data(self) -> None:
        """Test requires_correlation_data property."""
        config_correlation = PositionSizingConfig(sizing_method=SizingMethod.CORRELATION_ADJUSTED)
        assert config_correlation.requires_correlation_data is True

        config_other = PositionSizingConfig(sizing_method=SizingMethod.FIXED_PERCENTAGE)
        assert config_other.requires_correlation_data is False

    def test_config_serialization(self) -> None:
        """Test config serialization methods."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.RISK_BASED,
            kelly_win_rate=0.6,
            kelly_avg_win=120.0,
            kelly_avg_loss=60.0,
        )

        # Test dict conversion
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["sizing_method"] == "risk_based"
        assert config_dict["kelly_win_rate"] == 0.6

        # Test JSON serialization
        config_json = config.model_dump_json()
        assert isinstance(config_json, str)
        assert '"sizing_method":"risk_based"' in config_json


class TestPositionSizer:
    """Test suite for the PositionSizer class."""

    @pytest.fixture
    def default_config(self) -> PositionSizingConfig:
        """Create default PositionSizingConfig for testing."""
        return PositionSizingConfig()

    @pytest.fixture
    def kelly_config(self) -> PositionSizingConfig:
        """Create Kelly criterion configuration for testing."""
        return PositionSizingConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.6,
            kelly_avg_win=100.0,
            kelly_avg_loss=50.0,
        )

    @pytest.fixture
    def risk_based_config(self) -> PositionSizingConfig:
        """Create risk-based configuration for testing."""
        return PositionSizingConfig(
            sizing_method=SizingMethod.RISK_BASED,
            risk_per_trade=0.02,
            max_position_size=0.15,
        )

    @pytest.fixture
    def mock_logger(self) -> Mock:
        """Create mock logger for testing."""
        return Mock(spec=logging.Logger)

    def test_init_default(self, default_config: PositionSizingConfig) -> None:
        """Test PositionSizer initialization with default config."""
        sizer = PositionSizer()

        assert sizer.config == default_config
        assert sizer.logger is not None
        assert isinstance(sizer.logger, logging.Logger)

    def test_init_custom_config(
        self, risk_based_config: PositionSizingConfig, mock_logger: Mock
    ) -> None:
        """Test PositionSizer initialization with custom config and logger."""
        sizer = PositionSizer(config=risk_based_config, logger=mock_logger)

        assert sizer.config == risk_based_config
        assert sizer.logger == mock_logger

    def test_calculate_position_size_default_fixed_percentage(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with fixed percentage method."""
        sizer = PositionSizer(config=default_config)

        # Test basic calculation
        result = sizer.calculate_position_size(
            portfolio_value=10000.0, volatility=0.0, conviction=1.0
        )

        # Should be max_position_size * conviction = 0.10 * 1.0 = 0.10
        assert result == 0.10

    def test_calculate_position_size_with_conviction(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with different conviction levels."""
        sizer = PositionSizer(config=default_config)

        # Low conviction
        result_low = sizer.calculate_position_size(portfolio_value=10000.0, conviction=0.5)
        assert result_low == 0.05  # 0.10 * 0.5

        # High conviction - should be capped at max_position_size
        result_high = sizer.calculate_position_size(portfolio_value=10000.0, conviction=1.5)
        assert result_high == 0.10  # Capped at max_position_size (0.10)

    def test_calculate_position_size_with_volatility_adjustment(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with volatility adjustment."""
        sizer = PositionSizer(config=default_config)

        # Normal volatility
        result_normal = sizer.calculate_position_size(portfolio_value=10000.0, volatility=0.1)

        # High volatility should reduce position size
        result_high_vol = sizer.calculate_position_size(portfolio_value=10000.0, volatility=0.3)

        assert result_high_vol < result_normal

        # Very high volatility should cap at minimum
        result_very_high_vol = sizer.calculate_position_size(
            portfolio_value=10000.0, volatility=0.5
        )
        # Should be close to minimum due to volatility adjustment
        assert result_very_high_vol < 0.05

    def test_calculate_position_size_risk_based_method(
        self, risk_based_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with risk-based method."""
        sizer = PositionSizer(config=risk_based_config)

        result = sizer.calculate_position_size(
            portfolio_value=10000.0,
            entry_price=100.0,
            volatility=0.0,
        )

        # Should be based on risk_per_trade = 0.02
        # Position value = 10000 * 0.02 = 200
        # Position size = 200 / 100 = 2.0 (shares), but we return as fraction
        # The method returns fraction, so it should be around 0.02
        assert abs(result - 0.02) < 0.01

    def test_calculate_position_size_volatility_adjusted_method(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with volatility-adjusted method."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
            max_position_size=0.20,
        )
        sizer = PositionSizer(config=config)

        result_low_vol = sizer.calculate_position_size(portfolio_value=10000.0, volatility=0.05)

        result_high_vol = sizer.calculate_position_size(portfolio_value=10000.0, volatility=0.25)

        # High volatility should result in smaller position
        assert result_high_vol < result_low_vol

    def test_calculate_position_size_correlation_adjusted_method(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test position size calculation with correlation-adjusted method."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.CORRELATION_ADJUSTED,
            max_position_size=0.15,
        )
        sizer = PositionSizer(config=config)

        result_no_corr = sizer.calculate_position_size(portfolio_value=10000.0, entry_price=100.0)

        # High correlation should reduce position size
        # Note: The method signature doesn't have correlation parameter in calculate_position_size
        # This is a limitation in the implementation
        assert result_no_corr <= config.max_position_size

    def test_calculate_position_size_kelly_method(self, kelly_config: PositionSizingConfig) -> None:
        """Test position size calculation with Kelly criterion method."""
        sizer = PositionSizer(config=kelly_config)

        result = sizer.calculate_position_size(portfolio_value=10000.0)

        # Kelly calculation: f* = (bp - q) / b
        # where b = avg_win/avg_loss = 100/50 = 2, p = 0.6, q = 0.4
        # f* = (2 * 0.6 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # Should be capped at max_position_size = 0.10
        expected_kelly = min(0.4, kelly_config.max_position_size)
        assert result == expected_kelly

    def test_calculate_position_size_kelly_incomplete_params(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test Kelly method with incomplete parameters falls back to risk_per_trade."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.6,
            # Missing avg_win and avg_loss
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_position_size(portfolio_value=10000.0)

        # Should fall back to risk_per_trade
        assert result == config.risk_per_trade

    def test_calculate_position_size_kelly_zero_loss(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test Kelly method with zero average loss."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.6,
            kelly_avg_win=100.0,
            kelly_avg_loss=0.0,  # Zero loss
        )
        sizer = PositionSizer(config=config)

        result = sizer.calculate_position_size(portfolio_value=10000.0)

        # Should handle zero loss gracefully
        assert result == config.risk_per_trade  # Fallback

    def test_calculate_position_size_bounds_enforcement(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test that position size respects min/max bounds."""
        config = PositionSizingConfig(
            max_position_size=0.05,
            min_position_size=0.01,
        )
        sizer = PositionSizer(config=config)

        # Test with very high conviction that would exceed max
        result_high_conviction = sizer.calculate_position_size(
            portfolio_value=10000.0, conviction=10.0
        )
        assert result_high_conviction <= config.max_position_size

        # Test with very low conviction that would be below min
        result_low_conviction = sizer.calculate_position_size(
            portfolio_value=10000.0, conviction=0.01
        )
        assert result_low_conviction >= config.min_position_size

    def test_calculate_position_size_edge_cases(self, default_config: PositionSizingConfig) -> None:
        """Test position size calculation edge cases."""
        sizer = PositionSizer(config=default_config)

        # Zero portfolio value
        result_zero_portfolio = sizer.calculate_position_size(portfolio_value=0.0)
        assert result_zero_portfolio == 0.0

        # Negative portfolio value
        result_negative_portfolio = sizer.calculate_position_size(portfolio_value=-1000.0)
        assert result_negative_portfolio >= 0.0

        # Negative conviction
        result_negative_conviction = sizer.calculate_position_size(
            portfolio_value=10000.0, conviction=-0.5
        )
        assert result_negative_conviction >= 0.0

        # Very high volatility
        result_very_high_vol = sizer.calculate_position_size(
            portfolio_value=10000.0, volatility=10.0
        )
        assert result_very_high_vol >= 0.0

    def test_calculate_risk_based_size(self, risk_based_config: PositionSizingConfig) -> None:
        """Test _calculate_risk_based_size method."""
        sizer = PositionSizer(config=risk_based_config)

        result = sizer._calculate_risk_based_size(
            account_value=10000.0, entry_price=100.0, volatility=0.1
        )

        # Base position value = 10000 * 0.02 = 200
        # Volatility factor = max(0.1, 1.0 - 0.1 * 5) = 0.5
        # Position size = 200 * 0.5 / 10000 = 0.01
        expected = (10000.0 * 0.02 * 0.5) / 10000.0
        assert abs(result - expected) < 0.001

    def test_calculate_volatility_adjusted_size(self, default_config: PositionSizingConfig) -> None:
        """Test _calculate_volatility_adjusted_size method."""
        sizer = PositionSizer(config=default_config)

        result = sizer._calculate_volatility_adjusted_size(
            account_value=10000.0, entry_price=100.0, volatility=0.2
        )

        # Base size = 0.10
        # Volatility adjustment = max(0.1, 1.0 - 0.2 * 5) = max(0.1, 0.0) = 0.1
        # Result = 0.10 * 0.1 = 0.01
        expected = 0.10 * 0.1
        assert result == expected

    def test_calculate_correlation_adjusted_size(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test _calculate_correlation_adjusted_size method."""
        sizer = PositionSizer(config=default_config)

        result_low_corr = sizer._calculate_correlation_adjusted_size(
            account_value=10000.0, entry_price=100.0, portfolio_correlation=0.2
        )

        result_high_corr = sizer._calculate_correlation_adjusted_size(
            account_value=10000.0, entry_price=100.0, portfolio_correlation=0.8
        )

        # Lower correlation should allow larger positions
        assert result_low_corr > result_high_corr

        # Base size = 0.10
        # Low correlation adjustment = max(0.5, 1.0 - 0.2) = 0.8
        # High correlation adjustment = max(0.5, 1.0 - 0.8) = 0.5
        assert result_low_corr == 0.10 * 0.8
        assert result_high_corr == 0.10 * 0.5

    def test_calculate_kelly_size(self, kelly_config: PositionSizingConfig) -> None:
        """Test _calculate_kelly_size method."""
        sizer = PositionSizer(config=kelly_config)

        result = sizer._calculate_kelly_size(account_value=10000.0)

        # Kelly calculation: (b * p - q) / b
        # b = 100/50 = 2, p = 0.6, q = 0.4
        # (2 * 0.6 - 0.4) / 2 = 0.4
        # Capped at max_position_size = 0.10
        assert result == 0.10

    def test_calculate_kelly_size_incomplete_params(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test _calculate_kelly_size with incomplete parameters."""
        config = PositionSizingConfig(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.6,
            # Missing avg_win and avg_loss
        )
        sizer = PositionSizer(config=config)

        result = sizer._calculate_kelly_size(account_value=10000.0)

        # Should fall back to risk_per_trade
        assert result == config.risk_per_trade

    def test_calculate_position_size_fixed_risk(self, default_config: PositionSizingConfig) -> None:
        """Test calculate_position_size_fixed_risk method."""
        sizer = PositionSizer(config=default_config)

        result = sizer.calculate_position_size_fixed_risk(
            account_value=10000.0, entry_price=100.0, stop_price=95.0
        )

        # Risk amount = 10000 * 0.02 = 200
        # Stop distance = 100 - 95 = 5
        # Position value = 200 / (5/100) = 200 / 0.05 = 4000
        # Max position value = 10000 * 0.10 = 1000
        # Result = min(4000, 1000) = 1000
        expected = min(10000.0 * 0.02 / 0.05, 10000.0 * 0.10)
        assert result == expected

    def test_calculate_position_size_fixed_risk_zero_stop_distance(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test calculate_position_size_fixed_risk with zero stop distance."""
        sizer = PositionSizer(config=default_config)

        result = sizer.calculate_position_size_fixed_risk(
            account_value=10000.0, entry_price=100.0, stop_price=100.0
        )

        # Should return 0 for zero stop distance
        assert result == 0.0

    def test_calculate_position_size_percentage(self, default_config: PositionSizingConfig) -> None:
        """Test calculate_position_size_percentage method."""
        sizer = PositionSizer(config=default_config)

        result = sizer.calculate_position_size_percentage(
            account_value=10000.0, entry_price=100.0, percentage=0.05
        )

        # Position value = 10000 * 0.05 = 500
        # Shares = 500 / 100 = 5
        expected = 5.0
        assert result == expected

    def test_calculate_kelly_fraction(self) -> None:
        """Test calculate_kelly_fraction standalone method."""
        config = PositionSizingConfig(max_position_size=0.20)
        sizer = PositionSizer(config=config)

        result = sizer.calculate_kelly_fraction(win_rate=0.6, avg_win=100.0, avg_loss=50.0)

        # Kelly calculation: (b * p - q) / b
        # b = 100/50 = 2, p = 0.6, q = 0.4
        # (2 * 0.6 - 0.4) / 2 = 0.4
        # Capped at max_position_size = 0.20
        expected = 0.20
        assert result == expected

    def test_calculate_kelly_fraction_zero_loss(self) -> None:
        """Test calculate_kelly_fraction with zero loss."""
        config = PositionSizingConfig()
        sizer = PositionSizer(config=config)

        result = sizer.calculate_kelly_fraction(win_rate=0.6, avg_win=100.0, avg_loss=0.0)

        # Should handle zero loss gracefully
        assert result == 0.0

    def test_calculate_position_size_risk_based_comprehensive(
        self, default_config: PositionSizingConfig
    ) -> None:
        """Test calculate_position_size_risk_based comprehensive method."""
        sizer = PositionSizer(config=default_config)

        result = sizer.calculate_position_size_risk_based(
            account_value=10000.0,
            entry_price=100.0,
            stop_price=95.0,
            volatility=0.1,
            correlation=0.3,
            conviction_level='high',
        )

        # Should return a positive value within bounds
        assert result > 0.0
        assert result < (10000.0 * default_config.max_position_size) / 100.0

    def test_enforce_constraints(self, default_config: PositionSizingConfig) -> None:
        """Test enforce_constraints method."""
        sizer = PositionSizer(config=default_config)

        # Test minimum constraint
        result_min = sizer.enforce_constraints(
            position_size=0.001, account_value=10000.0, entry_price=100.0
        )
        # Minimum position value = 10000 * 0.01 = 100
        # Minimum position size = 100 / 100 = 1.0
        expected_min = 100.0 / 100.0
        assert result_min == expected_min

        # Test maximum constraint
        result_max = sizer.enforce_constraints(
            position_size=100.0, account_value=10000.0, entry_price=100.0
        )
        # Maximum position value = 10000 * 0.10 = 1000
        # Maximum position size = 1000 / 100 = 10.0
        expected_max = 1000.0 / 100.0
        assert result_max == expected_max

        # Test no constraint needed
        result_normal = sizer.enforce_constraints(
            position_size=5.0, account_value=10000.0, entry_price=100.0
        )
        # Position value = 5.0 * 100 = 500, which is between min (100) and max (1000)
        assert result_normal == 5.0

    def test_get_sizing_metrics(self, risk_based_config: PositionSizingConfig) -> None:
        """Test get_sizing_metrics method."""
        sizer = PositionSizer(config=risk_based_config)

        metrics = sizer.get_sizing_metrics()

        expected_keys = {
            'max_position_size',
            'min_position_size',
            'risk_per_trade',
            'sizing_method',
            'volatility_adjustment',
            'max_daily_trades',
            'max_sector_exposure',
            'max_correlation',
            'conviction_factors',
        }

        assert set(metrics.keys()) == expected_keys
        assert metrics['max_position_size'] == 0.15
        assert metrics['sizing_method'] == 'risk_based'
        assert metrics['risk_per_trade'] == 0.02
        assert metrics['volatility_adjustment'] is True

    def test_all_sizing_methods_existence(self) -> None:
        """Test that all sizing methods are properly implemented."""
        methods = [
            SizingMethod.FIXED_PERCENTAGE,
            SizingMethod.KELLY,
            SizingMethod.RISK_BASED,
            SizingMethod.VOLATILITY_ADJUSTED,
            SizingMethod.CORRELATION_ADJUSTED,
        ]

        for method in methods:
            config = PositionSizingConfig(sizing_method=method)
            sizer = PositionSizer(config=config)

            # Should be able to calculate position size without errors
            result = sizer.calculate_position_size(portfolio_value=10000.0)
            assert result >= 0.0
            assert result <= config.max_position_size

    def test_conviction_factors_integration(self, default_config: PositionSizingConfig) -> None:
        """Test integration of conviction factors in position sizing."""
        # Use risk-based method for this test since it applies conviction factors
        # Use more extreme conviction factors to ensure they don't hit max constraints
        config = PositionSizingConfig(
            sizing_method=SizingMethod.RISK_BASED,
            conviction_factors={'low': 0.2, 'medium': 1.0, 'high': 4.0},
            max_position_size=0.50,  # Large max to avoid constraints
        )
        sizer = PositionSizer(config=config)

        # Test different conviction levels (low, medium, high mappings)
        result_low = sizer.calculate_position_size_risk_based(
            account_value=10000.0, entry_price=100.0, stop_price=95.0, conviction_level='low'
        )
        result_medium = sizer.calculate_position_size_risk_based(
            account_value=10000.0, entry_price=100.0, stop_price=95.0, conviction_level='medium'
        )
        result_high = sizer.calculate_position_size_risk_based(
            account_value=10000.0, entry_price=100.0, stop_price=95.0, conviction_level='high'
        )

        # Should reflect conviction factors
        assert result_low < result_medium
        assert result_medium < result_high

    def test_different_account_values(self, default_config: PositionSizingConfig) -> None:
        """Test position sizing with different account values."""
        sizer = PositionSizer(config=default_config)

        # Small account
        result_small = sizer.calculate_position_size(portfolio_value=1000.0)
        # Large account
        result_large = sizer.calculate_position_size(portfolio_value=1000000.0)

        # Results should be proportional to account size
        # Both should be fractions of portfolio, so should be similar
        assert abs(result_small - result_large) < 0.01

    def test_various_entry_prices(self, default_config: PositionSizingConfig) -> None:
        """Test position sizing with various entry prices."""
        sizer = PositionSizer(config=default_config)

        # Low entry price
        result_low_price = sizer.calculate_position_size(portfolio_value=10000.0, entry_price=10.0)
        # High entry price
        result_high_price = sizer.calculate_position_size(
            portfolio_value=10000.0, entry_price=1000.0
        )

        # Results should be similar fractions for fixed percentage method
        assert abs(result_low_price - result_high_price) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
