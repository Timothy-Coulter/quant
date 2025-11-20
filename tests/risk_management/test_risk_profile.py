"""Comprehensive unit tests for the RiskProfile class.

This module contains tests for the risk profile functionality including
profile types, initialization, validation, suitability checking, and integration.
"""

import pytest
from pydantic import ValidationError

from backtester.risk_management.risk_profile import (
    RiskProfile,
    RiskProfileType,
)


class TestRiskProfileTypeEnum:
    """Test suite for RiskProfileType enum."""

    def test_enum_values(self) -> None:
        """Test RiskProfileType enum values."""
        assert RiskProfileType.CONSERVATIVE.value == "conservative"
        assert RiskProfileType.MODERATE.value == "moderate"
        assert RiskProfileType.AGGRESSIVE.value == "aggressive"

    def test_enum_is_string(self) -> None:
        """Test that RiskProfileType inherits from str."""
        assert isinstance(RiskProfileType.CONSERVATIVE, str)
        assert isinstance(RiskProfileType.MODERATE, str)
        assert isinstance(RiskProfileType.AGGRESSIVE, str)

    def test_enum_comparison(self) -> None:
        """Test RiskProfileType enum comparisons."""
        # Test that same enum is equal to itself
        assert RiskProfileType.CONSERVATIVE == RiskProfileType.CONSERVATIVE
        assert RiskProfileType.MODERATE == RiskProfileType.MODERATE
        assert RiskProfileType.AGGRESSIVE == RiskProfileType.AGGRESSIVE

    def test_enum_string_conversion(self) -> None:
        """Test string conversion of RiskProfileType enum."""
        assert str(RiskProfileType.CONSERVATIVE) == "RiskProfileType.CONSERVATIVE"
        assert str(RiskProfileType.MODERATE) == "RiskProfileType.MODERATE"
        assert str(RiskProfileType.AGGRESSIVE) == "RiskProfileType.AGGRESSIVE"
        # Test the .value attribute for actual string values
        assert RiskProfileType.CONSERVATIVE.value == "conservative"
        assert RiskProfileType.MODERATE.value == "moderate"
        assert RiskProfileType.AGGRESSIVE.value == "aggressive"


class TestRiskProfileInitialization:
    """Test suite for RiskProfile initialization and validation."""

    def test_init_default_values(self) -> None:
        """Test RiskProfile initialization with default values."""
        profile = RiskProfile(
            name="Test Profile",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.20,
            max_drawdown=0.15,
            target_sharpe_ratio=0.8,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        assert profile.name == "Test Profile"
        assert profile.profile_type == RiskProfileType.MODERATE
        assert profile.max_volatility == 0.20
        assert profile.max_drawdown == 0.15
        assert profile.target_sharpe_ratio == 0.8
        assert profile.max_position_size == 0.10
        assert profile.max_sector_exposure == 0.30
        assert profile.max_leverage == 2.0
        assert profile.risk_tolerance == "moderate"  # Default value
        assert profile.diversification_preference == "medium"  # Default value
        assert profile.rebalance_frequency == "monthly"  # Default value

    def test_init_with_all_values(self) -> None:
        """Test RiskProfile initialization with all values provided."""
        profile = RiskProfile(
            name="Custom Profile",
            profile_type=RiskProfileType.AGGRESSIVE,
            description="Custom aggressive profile",
            max_volatility=0.35,
            max_drawdown=0.20,
            target_sharpe_ratio=1.0,
            max_position_size=0.15,
            max_sector_exposure=0.35,
            max_leverage=2.5,
            risk_tolerance="high",
            diversification_preference="low",
            rebalance_frequency="quarterly",
            target_return=0.10,
            min_return=0.05,
        )

        assert profile.name == "Custom Profile"
        assert profile.profile_type == RiskProfileType.AGGRESSIVE
        assert profile.description == "Custom aggressive profile"
        assert profile.max_volatility == 0.35
        assert profile.max_drawdown == 0.20
        assert profile.target_sharpe_ratio == 1.0
        assert profile.max_position_size == 0.15
        assert profile.max_sector_exposure == 0.35
        assert profile.max_leverage == 2.5
        assert profile.risk_tolerance == "high"
        assert profile.diversification_preference == "low"
        assert profile.rebalance_frequency == "quarterly"
        assert profile.target_return == 0.10
        assert profile.min_return == 0.05

    def test_init_predefined_conservative(self) -> None:
        """Test RiskProfile initialization with predefined conservative profile."""
        profile = RiskProfile("Conservative")

        assert profile.name == "Conservative"
        assert profile.profile_type == RiskProfileType.CONSERVATIVE
        assert profile.description == "Low risk tolerance with focus on capital preservation"
        assert profile.max_volatility == 0.15
        assert profile.max_drawdown == 0.10
        assert profile.target_sharpe_ratio == 0.5
        assert profile.max_position_size == 0.05
        assert profile.max_sector_exposure == 0.20
        assert profile.max_leverage == 1.5
        assert profile.risk_tolerance == "low"
        assert profile.diversification_preference == "high"
        assert profile.rebalance_frequency == "weekly"
        assert profile.target_return == 0.05
        assert profile.min_return == 0.02

    def test_init_predefined_moderate(self) -> None:
        """Test RiskProfile initialization with predefined moderate profile."""
        profile = RiskProfile("Moderate")

        assert profile.name == "Moderate"
        assert profile.profile_type == RiskProfileType.MODERATE
        assert profile.description == "Balanced risk and return approach"
        assert profile.max_volatility == 0.25
        assert profile.max_drawdown == 0.15
        assert profile.target_sharpe_ratio == 0.8
        assert profile.max_position_size == 0.10
        assert profile.max_sector_exposure == 0.30
        assert profile.max_leverage == 2.0
        assert profile.risk_tolerance == "medium"
        assert profile.diversification_preference == "medium"
        assert profile.rebalance_frequency == "monthly"
        assert profile.target_return == 0.08
        assert profile.min_return == 0.04

    def test_init_predefined_aggressive(self) -> None:
        """Test RiskProfile initialization with predefined aggressive profile."""
        profile = RiskProfile("Aggressive")

        assert profile.name == "Aggressive"
        assert profile.profile_type == RiskProfileType.AGGRESSIVE
        assert profile.description == "High risk tolerance for maximum growth potential"
        assert profile.max_volatility == 0.40
        assert profile.max_drawdown == 0.25
        assert profile.target_sharpe_ratio == 1.2
        assert profile.max_position_size == 0.20
        assert profile.max_sector_exposure == 0.40
        assert profile.max_leverage == 3.0
        assert profile.risk_tolerance == "high"
        assert profile.diversification_preference == "low"
        assert profile.rebalance_frequency == "quarterly"
        assert profile.target_return == 0.12
        assert profile.min_return == 0.06

    def test_init_predefined_case_insensitive(self) -> None:
        """Test RiskProfile initialization with case-insensitive predefined names."""
        conservative_lower = RiskProfile("conservative")
        conservative_title = RiskProfile("Conservative")
        conservative_upper = RiskProfile("CONSERVATIVE")

        assert conservative_lower.profile_type == RiskProfileType.CONSERVATIVE
        assert conservative_title.profile_type == RiskProfileType.CONSERVATIVE
        assert conservative_upper.profile_type == RiskProfileType.CONSERVATIVE

    def test_init_with_additional_data(self) -> None:
        """Test RiskProfile initialization with name and additional data."""
        # This test should fail because we need to provide all required fields
        # when adding additional data beyond the name
        with pytest.raises(ValidationError):
            RiskProfile(
                name="Custom Conservative",
                max_volatility=0.12,
                target_return=0.04,
            )

    def test_init_invalid_volatility(self) -> None:
        """Test RiskProfile validation for invalid max_volatility."""
        with pytest.raises(ValidationError) as exc_info:
            RiskProfile(
                name="Test",
                profile_type=RiskProfileType.MODERATE,
                max_volatility=0.0,  # Below minimum
            )

        assert "max_volatility" in str(exc_info.value)

    def test_init_invalid_drawdown(self) -> None:
        """Test RiskProfile validation for invalid max_drawdown."""
        with pytest.raises(ValidationError) as exc_info:
            RiskProfile(
                name="Test",
                profile_type=RiskProfileType.MODERATE,
                max_drawdown=1.5,  # Above maximum
            )

        assert "max_drawdown" in str(exc_info.value)

    def test_init_invalid_sharpe_ratio(self) -> None:
        """Test RiskProfile validation for invalid target_sharpe_ratio."""
        with pytest.raises(ValidationError) as exc_info:
            RiskProfile(
                name="Test",
                profile_type=RiskProfileType.MODERATE,
                target_sharpe_ratio=-1.0,  # Below minimum
            )

        assert "target_sharpe_ratio" in str(exc_info.value)

    def test_init_invalid_leverage(self) -> None:
        """Test RiskProfile validation for invalid max_leverage."""
        with pytest.raises(ValidationError) as exc_info:
            RiskProfile(
                name="Test",
                profile_type=RiskProfileType.MODERATE,
                max_leverage=0.5,  # Below minimum
            )

        assert "max_leverage" in str(exc_info.value)


class TestPredefinedProfileMethods:
    """Test suite for predefined profile class methods."""

    def test_conservative_classmethod(self) -> None:
        """Test RiskProfile.conservative() class method."""
        profile = RiskProfile.conservative()

        assert profile.name == "Conservative"
        assert profile.profile_type == RiskProfileType.CONSERVATIVE
        assert profile.description == "Low risk tolerance with focus on capital preservation"
        assert profile.max_volatility == 0.15
        assert profile.max_drawdown == 0.10
        assert profile.target_sharpe_ratio == 0.5
        assert profile.max_position_size == 0.05
        assert profile.max_sector_exposure == 0.20
        assert profile.max_leverage == 1.5

    def test_moderate_classmethod(self) -> None:
        """Test RiskProfile.moderate() class method."""
        profile = RiskProfile.moderate()

        assert profile.name == "Moderate"
        assert profile.profile_type == RiskProfileType.MODERATE
        assert profile.description == "Balanced risk and return approach"
        assert profile.max_volatility == 0.25
        assert profile.max_drawdown == 0.15
        assert profile.target_sharpe_ratio == 0.8
        assert profile.max_position_size == 0.10
        assert profile.max_sector_exposure == 0.30
        assert profile.max_leverage == 2.0

    def test_aggressive_classmethod(self) -> None:
        """Test RiskProfile.aggressive() class method."""
        profile = RiskProfile.aggressive()

        assert profile.name == "Aggressive"
        assert profile.profile_type == RiskProfileType.AGGRESSIVE
        assert profile.description == "High risk tolerance for maximum growth potential"
        assert profile.max_volatility == 0.40
        assert profile.max_drawdown == 0.25
        assert profile.target_sharpe_ratio == 1.2
        assert profile.max_position_size == 0.20
        assert profile.max_sector_exposure == 0.40
        assert profile.max_leverage == 3.0

    def test_classmethod_consistency(self) -> None:
        """Test that class methods are consistent with constructor."""
        conservative_from_constructor = RiskProfile("Conservative")
        conservative_from_method = RiskProfile.conservative()

        assert (
            conservative_from_constructor.max_volatility == conservative_from_method.max_volatility
        )
        assert conservative_from_constructor.max_drawdown == conservative_from_method.max_drawdown
        assert (
            conservative_from_constructor.max_position_size
            == conservative_from_method.max_position_size
        )


class TestSuitabilityChecking:
    """Test suite for check_suitability method."""

    def test_check_suitability_all_within_limits(self) -> None:
        """Test suitability check when all metrics are within limits."""
        profile = RiskProfile.conservative()

        result = profile.check_suitability(
            portfolio_volatility=0.10,  # Below max_volatility (0.15)
            portfolio_max_drawdown=-0.08,  # Above max_drawdown (0.10)
            portfolio_sharpe=0.6,  # Above target_sharpe_ratio (0.5)
        )

        assert result['suitable'] is True
        assert result['suitability_score'] > 0
        assert 'risk_scores' in result
        assert 'volatility' in result['risk_scores']
        assert 'drawdown' in result['risk_scores']
        assert 'sharpe' in result['risk_scores']
        assert 'recommendations' in result
        assert result['profile_name'] == "Conservative"
        assert 'current_metrics' in result
        assert 'profile_limits' in result

    def test_check_suitability_exceeds_volatility(self) -> None:
        """Test suitability check when volatility exceeds limit."""
        profile = RiskProfile.conservative()

        result = profile.check_suitability(
            portfolio_volatility=0.20,  # Above max_volatility (0.15)
            portfolio_max_drawdown=-0.05,
            portfolio_sharpe=0.7,
        )

        assert result['suitable'] is False
        assert (
            "Reduce position sizes or volatility to meet profile limits"
            in result['recommendations']
        )

    def test_check_suitability_exceeds_drawdown(self) -> None:
        """Test suitability check when drawdown exceeds limit."""
        profile = RiskProfile.conservative()

        result = profile.check_suitability(
            portfolio_volatility=0.12,
            portfolio_max_drawdown=-0.15,  # Above max_drawdown (0.10)
            portfolio_sharpe=0.7,
        )

        assert result['suitable'] is False
        assert "Implement tighter stop losses to limit drawdowns" in result['recommendations']

    def test_check_suitability_below_sharpe_target(self) -> None:
        """Test suitability check when Sharpe ratio is below target."""
        profile = RiskProfile.conservative()

        result = profile.check_suitability(
            portfolio_volatility=0.12,
            portfolio_max_drawdown=-0.05,
            portfolio_sharpe=0.3,  # Below target_sharpe_ratio (0.5)
        )

        assert result['suitable'] is False
        assert (
            "Improve risk-adjusted returns through better strategy selection"
            in result['recommendations']
        )

    def test_check_suitability_all_within_limits_aggressive(self) -> None:
        """Test suitability check for aggressive profile with appropriate metrics."""
        profile = RiskProfile.aggressive()

        result = profile.check_suitability(
            portfolio_volatility=0.35,  # Below max_volatility (0.40)
            portfolio_max_drawdown=-0.20,  # Above max_drawdown (0.25)
            portfolio_sharpe=1.3,  # Above target_sharpe_ratio (1.2)
        )

        assert result['suitable'] is True
        assert "Portfolio well-suited for this risk profile" in result['recommendations']

    def test_check_suitability_boundary_values(self) -> None:
        """Test suitability check with boundary values."""
        profile = RiskProfile.moderate()

        result = profile.check_suitability(
            portfolio_volatility=0.25,  # Exactly at max_volatility
            portfolio_max_drawdown=-0.15,  # Exactly at max_drawdown
            portfolio_sharpe=0.8,  # Exactly at target_sharpe_ratio
        )

        assert result['suitable'] is True

    def test_check_suitability_risk_scores_calculation(self) -> None:
        """Test risk scores calculation in suitability check."""
        profile = RiskProfile.moderate()

        result = profile.check_suitability(
            portfolio_volatility=0.20,  # 80% of max_volatility
            portfolio_max_drawdown=-0.10,  # 67% of max_drawdown
            portfolio_sharpe=1.0,  # 125% of target_sharpe_ratio
        )

        volatility_score = result['risk_scores']['volatility']
        drawdown_score = result['risk_scores']['drawdown']
        sharpe_score = result['risk_scores']['sharpe']

        # Check that scores are calculated correctly
        assert 0 < volatility_score <= 1.0
        assert 0 < drawdown_score <= 1.0
        assert 0 < sharpe_score <= 1.0

        # Overall suitability score should be average of individual scores
        expected_suitability = (volatility_score + drawdown_score + sharpe_score) / 3
        assert result['suitability_score'] == expected_suitability

    def test_check_suitability_zero_portfolio_metrics(self) -> None:
        """Test suitability check with zero portfolio metrics."""
        profile = RiskProfile.moderate()

        result = profile.check_suitability(
            portfolio_volatility=0.0,
            portfolio_max_drawdown=0.0,
            portfolio_sharpe=0.0,
        )

        # Should still return valid result structure
        assert isinstance(result, dict)
        assert 'suitable' in result
        assert 'suitability_score' in result
        assert 'risk_scores' in result

    def test_check_suitability_negative_drawdown(self) -> None:
        """Test suitability check with negative drawdown values."""
        profile = RiskProfile.moderate()

        result = profile.check_suitability(
            portfolio_volatility=0.15,
            portfolio_max_drawdown=-0.08,
            portfolio_sharpe=0.9,
        )

        # Drawdown should be handled as absolute value
        drawdown_score = result['risk_scores']['drawdown']
        assert 0 < drawdown_score <= 1.0


class TestRiskLimits:
    """Test suite for get_risk_limits method."""

    def test_get_risk_limits_conservative(self) -> None:
        """Test get_risk_limits for conservative profile."""
        profile = RiskProfile.conservative()
        limits = profile.get_risk_limits()

        expected_keys = {
            'max_volatility',
            'max_drawdown',
            'max_position_size',
            'max_sector_exposure',
            'max_leverage',
            'target_sharpe_ratio',
        }
        assert set(limits.keys()) == expected_keys
        assert limits['max_volatility'] == 0.15
        assert limits['max_drawdown'] == 0.10
        assert limits['max_position_size'] == 0.05
        assert limits['max_sector_exposure'] == 0.20
        assert limits['max_leverage'] == 1.5
        assert limits['target_sharpe_ratio'] == 0.5

    def test_get_risk_limits_moderate(self) -> None:
        """Test get_risk_limits for moderate profile."""
        profile = RiskProfile.moderate()
        limits = profile.get_risk_limits()

        assert limits['max_volatility'] == 0.25
        assert limits['max_drawdown'] == 0.15
        assert limits['max_position_size'] == 0.10
        assert limits['max_sector_exposure'] == 0.30
        assert limits['max_leverage'] == 2.0
        assert limits['target_sharpe_ratio'] == 0.8

    def test_get_risk_limits_aggressive(self) -> None:
        """Test get_risk_limits for aggressive profile."""
        profile = RiskProfile.aggressive()
        limits = profile.get_risk_limits()

        assert limits['max_volatility'] == 0.40
        assert limits['max_drawdown'] == 0.25
        assert limits['max_position_size'] == 0.20
        assert limits['max_sector_exposure'] == 0.40
        assert limits['max_leverage'] == 3.0
        assert limits['target_sharpe_ratio'] == 1.2

    def test_get_risk_limits_custom_profile(self) -> None:
        """Test get_risk_limits for custom profile."""
        profile = RiskProfile(
            name="Custom",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.30,
            max_drawdown=0.18,
            max_position_size=0.12,
            max_sector_exposure=0.35,
            max_leverage=2.5,
        )
        limits = profile.get_risk_limits()

        assert limits['max_volatility'] == 0.30
        assert limits['max_drawdown'] == 0.18
        assert limits['max_position_size'] == 0.12
        assert limits['max_sector_exposure'] == 0.35
        assert limits['max_leverage'] == 2.5
        # target_sharpe_ratio should have default value
        assert limits['target_sharpe_ratio'] == 0.8


class TestProfileTypeChecking:
    """Test suite for profile type checking methods."""

    def test_is_conservative_true(self) -> None:
        """Test is_conservative returns True for conservative profile."""
        profile = RiskProfile.conservative()
        assert profile.is_conservative() is True
        assert profile.is_moderate() is False
        assert profile.is_aggressive() is False

    def test_is_moderate_true(self) -> None:
        """Test is_moderate returns True for moderate profile."""
        profile = RiskProfile.moderate()
        assert profile.is_conservative() is False
        assert profile.is_moderate() is True
        assert profile.is_aggressive() is False

    def test_is_aggressive_true(self) -> None:
        """Test is_aggressive returns True for aggressive profile."""
        profile = RiskProfile.aggressive()
        assert profile.is_conservative() is False
        assert profile.is_moderate() is False
        assert profile.is_aggressive() is True

    def test_type_checking_custom_profile(self) -> None:
        """Test type checking for custom profile."""
        # Use a valid enum value but test that type checking methods work
        profile = RiskProfile(
            name="Custom",
            profile_type=RiskProfileType.MODERATE,  # Use valid enum value
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        assert profile.is_conservative() is False
        assert profile.is_moderate() is True  # Should be true since we used MODERATE
        assert profile.is_aggressive() is False

    def test_type_checking_case_sensitivity(self) -> None:
        """Test that type checking is case sensitive."""
        # Test with different case profile types
        conservative = RiskProfile(
            name="Test",
            profile_type=RiskProfileType.CONSERVATIVE,
            max_volatility=0.15,
            max_drawdown=0.10,
            max_position_size=0.05,
            max_sector_exposure=0.20,
            max_leverage=1.5,
        )

        assert conservative.is_conservative() is True  # Should match exactly
        assert conservative.profile_type == RiskProfileType.CONSERVATIVE


class TestProfileSummary:
    """Test suite for get_profile_summary method."""

    def test_get_profile_summary_conservative(self) -> None:
        """Test get_profile_summary for conservative profile."""
        profile = RiskProfile.conservative()
        summary = profile.get_profile_summary()

        expected_keys = {
            'name',
            'type',
            'description',
            'risk_limits',
            'preferences',
            'targets',
        }
        assert set(summary.keys()) == expected_keys
        assert summary['name'] == "Conservative"
        assert summary['type'] == "conservative"
        assert summary['description'] == "Low risk tolerance with focus on capital preservation"
        assert 'risk_limits' in summary
        assert 'preferences' in summary
        assert 'targets' in summary

    def test_get_profile_summary_risk_limits_structure(self) -> None:
        """Test risk_limits structure in profile summary."""
        profile = RiskProfile.moderate()
        summary = profile.get_profile_summary()
        risk_limits = summary['risk_limits']

        expected_keys = {
            'max_volatility',
            'max_drawdown',
            'max_position_size',
            'max_sector_exposure',
            'max_leverage',
            'target_sharpe_ratio',
        }
        assert set(risk_limits.keys()) == expected_keys

    def test_get_profile_summary_preferences_structure(self) -> None:
        """Test preferences structure in profile summary."""
        profile = RiskProfile.aggressive()
        summary = profile.get_profile_summary()
        preferences = summary['preferences']

        expected_keys = {
            'risk_tolerance',
            'diversification_preference',
            'rebalance_frequency',
        }
        assert set(preferences.keys()) == expected_keys
        assert preferences['risk_tolerance'] == "high"
        assert preferences['diversification_preference'] == "low"
        assert preferences['rebalance_frequency'] == "quarterly"

    def test_get_profile_summary_targets_structure(self) -> None:
        """Test targets structure in profile summary."""
        profile = RiskProfile.moderate()
        summary = profile.get_profile_summary()
        targets = summary['targets']

        expected_keys = {
            'target_return',
            'min_return',
            'target_sharpe_ratio',
        }
        assert set(targets.keys()) == expected_keys
        assert targets['target_return'] == 0.08
        assert targets['min_return'] == 0.04
        assert targets['target_sharpe_ratio'] == 0.8

    def test_get_profile_summary_custom_profile(self) -> None:
        """Test get_profile_summary for custom profile."""
        profile = RiskProfile(
            name="Custom Profile",
            profile_type=RiskProfileType.MODERATE,
            description="Custom test profile",
            max_volatility=0.22,
            max_drawdown=0.16,
            max_position_size=0.11,
            max_sector_exposure=0.32,
            max_leverage=2.1,
            target_return=0.09,
            min_return=0.045,
        )
        summary = profile.get_profile_summary()

        assert summary['name'] == "Custom Profile"
        assert summary['type'] == "moderate"
        assert summary['description'] == "Custom test profile"
        assert summary['risk_limits']['max_volatility'] == 0.22
        assert summary['targets']['target_return'] == 0.09


class TestPredefinedProfileHelper:
    """Test suite for _get_predefined_profile helper method."""

    def test_get_predefined_profile_conservative(self) -> None:
        """Test _get_predefined_profile for conservative profile."""
        profile = RiskProfile.moderate()  # Using moderate instance to call helper
        conservative_data = profile._get_predefined_profile('conservative')

        assert isinstance(conservative_data, dict)
        assert conservative_data['max_volatility'] == 0.15
        assert conservative_data['max_drawdown'] == 0.10
        assert conservative_data['target_sharpe_ratio'] == 0.5
        assert conservative_data['risk_tolerance'] == 'low'
        assert conservative_data['diversification_preference'] == 'high'

    def test_get_predefined_profile_moderate(self) -> None:
        """Test _get_predefined_profile for moderate profile."""
        profile = RiskProfile.conservative()  # Using conservative instance to call helper
        moderate_data = profile._get_predefined_profile('moderate')

        assert isinstance(moderate_data, dict)
        assert moderate_data['max_volatility'] == 0.25
        assert moderate_data['max_drawdown'] == 0.15
        assert moderate_data['target_sharpe_ratio'] == 0.8
        assert moderate_data['risk_tolerance'] == 'medium'
        assert moderate_data['diversification_preference'] == 'medium'

    def test_get_predefined_profile_aggressive(self) -> None:
        """Test _get_predefined_profile for aggressive profile."""
        profile = RiskProfile.moderate()  # Using moderate instance to call helper
        aggressive_data = profile._get_predefined_profile('aggressive')

        assert isinstance(aggressive_data, dict)
        assert aggressive_data['max_volatility'] == 0.40
        assert aggressive_data['max_drawdown'] == 0.25
        assert aggressive_data['target_sharpe_ratio'] == 1.2
        assert aggressive_data['risk_tolerance'] == 'high'
        assert aggressive_data['diversification_preference'] == 'low'

    def test_get_predefined_profile_unknown(self) -> None:
        """Test _get_predefined_profile for unknown profile (should fall back to moderate)."""
        profile = RiskProfile.conservative()  # Using conservative instance to call helper
        unknown_data = profile._get_predefined_profile('unknown')

        # Should fall back to moderate profile
        assert isinstance(unknown_data, dict)
        assert unknown_data['max_volatility'] == 0.25
        assert unknown_data['max_drawdown'] == 0.15
        assert unknown_data['target_sharpe_ratio'] == 0.8
        assert unknown_data['risk_tolerance'] == 'medium'

    def test_get_predefined_profile_case_insensitive(self) -> None:
        """Test _get_predefined_profile case insensitive handling."""
        profile = RiskProfile.moderate()

        # The _get_predefined_profile method itself is case-sensitive
        # but the __init__ method converts to lowercase before calling it
        conservative_lower = profile._get_predefined_profile('conservative')
        conservative_upper = profile._get_predefined_profile(
            'CONSERVATIVE'
        )  # This should get moderate fallback
        conservative_mixed = profile._get_predefined_profile(
            'ConSeRvAtIvE'
        )  # This should get moderate fallback

        # 'conservative' should return conservative data
        # 'CONSERVATIVE' and 'ConSeRvAtIvE' should fall back to moderate data
        assert conservative_lower['max_volatility'] == 0.15
        assert conservative_lower['max_leverage'] == 1.5

        # These should be moderate profile values (fallback)
        assert conservative_upper['max_volatility'] == 0.25
        assert conservative_upper['max_leverage'] == 2.0
        assert conservative_mixed['max_volatility'] == 0.25
        assert conservative_mixed['max_leverage'] == 2.0


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_name(self) -> None:
        """Test RiskProfile with empty name."""
        profile = RiskProfile(
            name="",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        assert profile.name == ""

    def test_extreme_values_boundary(self) -> None:
        """Test RiskProfile with boundary extreme values."""
        profile = RiskProfile(
            name="Boundary Test",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.01,  # Minimum
            max_drawdown=1.0,  # Maximum
            target_sharpe_ratio=0.0,  # Minimum
            max_position_size=1.0,  # Maximum
            max_sector_exposure=0.05,  # Minimum
            max_leverage=20.0,  # Maximum
        )

        assert profile.max_volatility == 0.01
        assert profile.max_drawdown == 1.0
        assert profile.target_sharpe_ratio == 0.0
        assert profile.max_position_size == 1.0
        assert profile.max_sector_exposure == 0.05
        assert profile.max_leverage == 20.0

    def test_very_large_values(self) -> None:
        """Test RiskProfile with very large target_return values."""
        profile = RiskProfile(
            name="Large Returns",
            profile_type=RiskProfileType.AGGRESSIVE,
            max_volatility=0.40,
            max_drawdown=0.25,
            target_sharpe_ratio=1.2,
            max_position_size=0.20,
            max_sector_exposure=0.40,
            max_leverage=3.0,
            target_return=1.0,  # Maximum
            min_return=0.5,
        )

        assert profile.target_return == 1.0
        assert profile.min_return == 0.5

    def test_negative_min_return(self) -> None:
        """Test RiskProfile with negative min_return."""
        profile = RiskProfile(
            name="Negative Min Return",
            profile_type=RiskProfileType.AGGRESSIVE,
            max_volatility=0.40,
            max_drawdown=0.25,
            target_sharpe_ratio=1.2,
            max_position_size=0.20,
            max_sector_exposure=0.40,
            max_leverage=3.0,
            min_return=-0.1,  # Below zero
        )

        assert profile.min_return == -0.1

    def test_none_optional_values(self) -> None:
        """Test RiskProfile with None values for optional fields."""
        profile = RiskProfile(
            name="None Test",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
            target_return=None,
            min_return=None,
        )

        assert profile.target_return is None
        assert profile.min_return is None

    def test_string_type_with_enum_value(self) -> None:
        """Test RiskProfile with string profile_type and enum comparison."""
        # String values are accepted due to use_enum_values=True in model_config
        profile = RiskProfile(
            name="String Type",
            profile_type="moderate",  # String should be accepted and converted to enum
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        # Should work with string comparison
        assert profile.profile_type == "moderate"
        # Type checking should work with enum
        assert profile.is_moderate() is True

    def test_special_characters_in_name(self) -> None:
        """Test RiskProfile with special characters in name."""
        profile = RiskProfile(
            name="Test@#$%^&*()",
            profile_type=RiskProfileType.MODERATE,
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        assert profile.name == "Test@#$%^&*()"

    def test_unicode_in_description(self) -> None:
        """Test RiskProfile with unicode characters in description."""
        profile = RiskProfile(
            name="Unicode Test",
            profile_type=RiskProfileType.MODERATE,
            description="Descripción con carácteres 日本語 🎯",
            max_volatility=0.20,
            max_drawdown=0.15,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
        )

        assert "日本語" in profile.description
        assert "🎯" in profile.description


class TestIntegrationScenarios:
    """Test suite for real-world integration scenarios."""

    def test_profile_comparison_scenarios(self) -> None:
        """Test comparison between different profile types."""
        conservative = RiskProfile.conservative()
        moderate = RiskProfile.moderate()
        aggressive = RiskProfile.aggressive()

        # Conservative should have lowest risk limits
        assert conservative.max_volatility < moderate.max_volatility
        assert conservative.max_volatility < aggressive.max_volatility
        assert conservative.max_drawdown < moderate.max_drawdown
        assert conservative.max_drawdown < aggressive.max_drawdown

        # Aggressive should have highest risk limits
        assert aggressive.max_volatility > moderate.max_volatility
        assert aggressive.max_volatility > conservative.max_volatility
        assert aggressive.max_leverage > moderate.max_leverage
        assert aggressive.max_leverage > conservative.max_leverage

    def test_suitability_workflow(self) -> None:
        """Test complete suitability assessment workflow."""
        # Portfolio with moderate risk metrics that meet all criteria
        portfolio_vol = 0.18
        portfolio_dd = -0.12
        portfolio_sharpe = 0.9  # High enough for conservative and moderate, not aggressive

        # Test against different profiles
        conservative = RiskProfile.conservative()
        moderate = RiskProfile.moderate()
        aggressive = RiskProfile.aggressive()

        conservative_suitability = conservative.check_suitability(
            portfolio_vol, portfolio_dd, portfolio_sharpe
        )
        moderate_suitability = moderate.check_suitability(
            portfolio_vol, portfolio_dd, portfolio_sharpe
        )
        aggressive_suitability = aggressive.check_suitability(
            portfolio_vol, portfolio_dd, portfolio_sharpe
        )

        # Conservative should not be suitable (vol and dd too high)
        assert conservative_suitability['suitable'] is False

        # Moderate should be suitable (all within limits)
        assert moderate_suitability['suitable'] is True

        # Aggressive should not be suitable (Sharpe ratio 0.9 < 1.2 target)
        assert aggressive_suitability['suitable'] is False

        # Test with a portfolio that meets aggressive criteria
        high_performance_portfolio = RiskProfile.aggressive()
        high_perf_suitability = high_performance_portfolio.check_suitability(
            0.30,
            -0.20,
            1.3,  # Within aggressive limits
        )
        assert high_perf_suitability['suitable'] is True

        # Test with a portfolio that exceeds conservative limits
        high_vol_portfolio = RiskProfile.conservative()
        high_vol_suitability = high_vol_portfolio.check_suitability(
            0.20,
            -0.15,
            0.9,  # Too high vol and dd for conservative
        )
        assert high_vol_suitability['suitable'] is False

    def test_profile_limits_application(self) -> None:
        """Test application of profile risk limits in decision making."""
        aggressive = RiskProfile.aggressive()

        # Get risk limits
        _ = aggressive.get_risk_limits()

        # Test various portfolio metrics against these limits
        test_portfolios = [
            {'vol': 0.35, 'dd': -0.20, 'sharpe': 1.1},  # Within limits
            {'vol': 0.45, 'dd': -0.30, 'sharpe': 0.9},  # Exceeds limits
            {'vol': 0.30, 'dd': -0.15, 'sharpe': 1.5},  # Well within limits
        ]

        for portfolio in test_portfolios:
            suitability = aggressive.check_suitability(
                portfolio['vol'], portfolio['dd'], portfolio['sharpe']
            )

            # Should always return a valid suitability result
            assert isinstance(suitability, dict)
            assert 'suitable' in suitability
            assert 'suitability_score' in suitability

    def test_profile_serialization(self) -> None:
        """Test profile data serialization and deserialization."""
        original = RiskProfile.aggressive()

        # Test model_dump for JSON serialization
        profile_dict = original.model_dump()
        profile_json = original.model_dump_json()

        assert isinstance(profile_dict, dict)
        assert isinstance(profile_json, str)
        assert profile_dict['name'] == "Aggressive"
        assert profile_dict['profile_type'] == "aggressive"

        # Test that we can recreate from dict
        recreated = RiskProfile(**profile_dict)
        assert recreated.name == original.name
        assert recreated.profile_type == original.profile_type
        assert recreated.max_volatility == original.max_volatility

    def test_predefined_profile_consistency(self) -> None:
        """Test consistency between predefined profiles and constructor patterns."""
        # Test that constructor with string name gives same result as class method
        conservative_name = RiskProfile("Conservative")
        conservative_method = RiskProfile.conservative()

        assert conservative_name.max_volatility == conservative_method.max_volatility
        assert conservative_name.max_drawdown == conservative_method.max_drawdown
        assert conservative_name.max_position_size == conservative_method.max_position_size
        assert conservative_name.profile_type == conservative_method.profile_type


if __name__ == "__main__":
    pytest.main([__file__])
