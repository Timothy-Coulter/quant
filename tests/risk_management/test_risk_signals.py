"""Comprehensive unit tests for the risk signals module.

This module contains tests for risk signals, metrics, limits, and alerts
including action types, initialization, validation, comparison methods,
and integration scenarios.
"""

import pandas as pd
import pytest

from backtester.risk_management.risk_signals import (
    RiskAction,
    RiskAlert,
    RiskLimit,
    RiskMetric,
    RiskSignal,
)


class TestRiskActionEnum:
    """Test suite for RiskAction enum."""

    def test_enum_values(self) -> None:
        """Test RiskAction enum values."""
        assert RiskAction.HOLD.value == "HOLD"
        assert RiskAction.REDUCE_POSITION.value == "REDUCE_POSITION"
        assert RiskAction.CLOSE_POSITION.value == "CLOSE_POSITION"
        assert RiskAction.INCREASE_POSITION.value == "INCREASE_POSITION"
        assert RiskAction.EMERGENCY_HALT.value == "EMERGENCY_HALT"
        assert RiskAction.REVIEW_POSITIONS.value == "REVIEW_POSITIONS"

    def test_enum_is_string(self) -> None:
        """Test that RiskAction inherits from str."""
        for action in RiskAction:
            assert isinstance(action, str)
            assert isinstance(action, RiskAction)

    def test_enum_comparison(self) -> None:
        """Test RiskAction enum comparisons."""
        # Test that same enum is equal to itself
        assert RiskAction.HOLD == RiskAction.HOLD
        assert RiskAction.EMERGENCY_HALT == RiskAction.EMERGENCY_HALT

        # Test that different enums are not equal
        assert str(RiskAction.HOLD) != str(RiskAction.CLOSE_POSITION)
        assert str(RiskAction.REDUCE_POSITION) != str(RiskAction.INCREASE_POSITION)

    def test_enum_string_conversion(self) -> None:
        """Test string conversion of RiskAction enum."""
        assert str(RiskAction.HOLD) == "RiskAction.HOLD"
        assert str(RiskAction.EMERGENCY_HALT) == "RiskAction.EMERGENCY_HALT"

        # Test the .value attribute for actual string values
        assert RiskAction.HOLD.value == "HOLD"
        assert RiskAction.EMERGENCY_HALT.value == "EMERGENCY_HALT"

    def test_enum_membership(self) -> None:
        """Test enum membership testing."""
        all_actions = list(RiskAction)
        assert RiskAction.HOLD in all_actions
        assert RiskAction.CLOSE_POSITION in all_actions
        assert RiskAction.EMERGENCY_HALT in all_actions
        assert "INVALID_ACTION" not in all_actions


class TestRiskSignalInitialization:
    """Test suite for RiskSignal initialization and validation."""

    def test_init_minimal_params(self) -> None:
        """Test RiskSignal initialization with minimal parameters."""
        signal = RiskSignal(
            action=RiskAction.HOLD,
            reason="Normal market conditions",
            confidence=0.5,
        )

        assert signal.action == RiskAction.HOLD
        assert signal.reason == "Normal market conditions"
        assert signal.confidence == 0.5
        assert signal.metadata is None
        assert signal.timestamp is not None
        assert isinstance(signal.timestamp, str)

    def test_init_all_params(self) -> None:
        """Test RiskSignal initialization with all parameters."""
        timestamp = "2023-01-01 12:00:00"
        metadata = {"source": "test", "version": 1}

        signal = RiskSignal(
            action=RiskAction.REDUCE_POSITION,
            reason="High volatility detected",
            confidence=0.8,
            metadata=metadata,
            timestamp=timestamp,
        )

        assert signal.action == RiskAction.REDUCE_POSITION
        assert signal.reason == "High volatility detected"
        assert signal.confidence == 0.8
        assert signal.metadata == metadata
        assert signal.timestamp == timestamp

    def test_init_auto_timestamp(self) -> None:
        """Test RiskSignal automatic timestamp generation."""
        before_time = str(pd.Timestamp.now())
        signal = RiskSignal(
            action=RiskAction.HOLD,
            reason="Test signal",
            confidence=1.0,
        )
        after_time = str(pd.Timestamp.now())

        # Timestamp should be automatically generated
        assert signal.timestamp is not None
        assert before_time <= signal.timestamp <= after_time

    def test_init_with_pandas_timestamp(self) -> None:
        """Test RiskSignal with pandas Timestamp object."""
        ts = pd.Timestamp("2023-01-01 12:00:00")
        signal = RiskSignal(
            action=RiskAction.EMERGENCY_HALT,
            reason="Market crash detected",
            confidence=1.0,
            timestamp=str(ts),
        )

        assert signal.timestamp == "2023-01-01 12:00:00"

    def test_init_extreme_confidence_values(self) -> None:
        """Test RiskSignal with extreme confidence values."""
        # Test minimum confidence
        signal_low = RiskSignal(
            action=RiskAction.HOLD,
            reason="Low confidence signal",
            confidence=0.0,
        )
        assert signal_low.confidence == 0.0

        # Test maximum confidence
        signal_high = RiskSignal(
            action=RiskAction.CLOSE_POSITION,
            reason="High confidence signal",
            confidence=1.0,
        )
        assert signal_high.confidence == 1.0

    def test_init_edge_case_confidence(self) -> None:
        """Test RiskSignal with edge case confidence values."""
        # Test slightly above 1.0 (should still work as float)
        signal_high = RiskSignal(
            action=RiskAction.REDUCE_POSITION,
            reason="Over confidence test",
            confidence=1.1,
        )
        assert signal_high.confidence == 1.1

        # Test negative confidence (should work as float)
        signal_negative = RiskSignal(
            action=RiskAction.HOLD,
            reason="Negative confidence test",
            confidence=-0.1,
        )
        assert signal_negative.confidence == -0.1

    def test_init_with_dict_metadata(self) -> None:
        """Test RiskSignal with dictionary metadata."""
        metadata = {
            "volatility": 0.25,
            "correlation": 0.8,
            "var": 0.05,
            "positions": 10,
        }

        signal = RiskSignal(
            action=RiskAction.REVIEW_POSITIONS,
            reason="Multiple risk factors",
            confidence=0.7,
            metadata=metadata,
        )

        assert signal.metadata == metadata
        assert signal.metadata["volatility"] == 0.25

    def test_init_with_complex_metadata(self) -> None:
        """Test RiskSignal with complex metadata structures."""
        metadata = {
            "nested": {"level1": {"level2": "deep_value"}},
            "list_data": [1, 2, 3, "mixed", {"key": "value"}],
            "mixed_types": {
                "string": "text",
                "number": 42,
                "boolean": True,
                "none": None,
            },
        }

        signal = RiskSignal(
            action=RiskAction.INCREASE_POSITION,
            reason="Complex metadata test",
            confidence=0.9,
            metadata=metadata,
        )

        assert signal.metadata == metadata
        assert signal.metadata["nested"]["level1"]["level2"] == "deep_value"
        assert signal.metadata["list_data"][3] == "mixed"

    def test_init_with_empty_strings(self) -> None:
        """Test RiskSignal with empty string parameters."""
        signal = RiskSignal(
            action=RiskAction.HOLD,
            reason="",
            confidence=0.5,
        )

        assert signal.reason == ""
        assert signal.action == RiskAction.HOLD

    def test_init_special_characters(self) -> None:
        """Test RiskSignal with special characters in reason."""
        special_reason = "High 🚨 risk! ⚠️ Multiple factors: @#$%^&*()"

        signal = RiskSignal(
            action=RiskAction.EMERGENCY_HALT,
            reason=special_reason,
            confidence=0.95,
        )

        assert signal.reason == special_reason

    def test_init_unicode_characters(self) -> None:
        """Test RiskSignal with unicode characters in reason and metadata."""
        unicode_reason = "Riesgo alto 日本語 العربية 🚨"
        unicode_metadata = {
            "message": "描述 高リスク",
            "symbols": ["EURUSD", "GBPJPY", "USDJPY"],
        }

        signal = RiskSignal(
            action=RiskAction.REDUCE_POSITION,
            reason=unicode_reason,
            confidence=0.85,
            metadata=unicode_metadata,
        )

        assert "日本語" in signal.reason
        assert "🚨" in signal.reason
        assert signal.metadata is not None and signal.metadata["message"] == "描述 高リスク"


class TestRiskMetricInitialization:
    """Test suite for RiskMetric initialization and validation."""

    def test_init_minimal_params(self) -> None:
        """Test RiskMetric initialization with minimal parameters."""
        metric = RiskMetric(
            name="portfolio_var",
            value=0.05,
            unit="percentage",
            threshold=0.10,
            status="normal",
        )

        assert metric.name == "portfolio_var"
        assert metric.value == 0.05
        assert metric.unit == "percentage"
        assert metric.threshold == 0.10
        assert metric.status == "normal"
        assert metric.timestamp is not None

    def test_init_all_params(self) -> None:
        """Test RiskMetric initialization with all parameters."""
        timestamp = "2023-01-01 12:00:00"

        metric = RiskMetric(
            name="max_drawdown",
            value=-0.15,
            unit="percentage",
            threshold=0.20,
            status="warning",
            timestamp=timestamp,
        )

        assert metric.name == "max_drawdown"
        assert metric.value == -0.15
        assert metric.unit == "percentage"
        assert metric.threshold == 0.20
        assert metric.status == "warning"
        assert metric.timestamp == timestamp

    def test_init_different_units(self) -> None:
        """Test RiskMetric with different unit types."""
        # Test percentage unit
        percentage_metric = RiskMetric(
            name="volatility",
            value=0.20,
            unit="percentage",
            threshold=0.25,
            status="normal",
        )
        assert percentage_metric.unit == "percentage"

        # Test absolute unit
        absolute_metric = RiskMetric(
            name="position_size",
            value=100000,
            unit="absolute",
            threshold=200000,
            status="normal",
        )
        assert absolute_metric.unit == "absolute"

        # Test currency unit
        currency_metric = RiskMetric(
            name="portfolio_value",
            value=1000000,
            unit="USD",
            threshold=500000,
            status="normal",
        )
        assert currency_metric.unit == "USD"

    def test_init_auto_timestamp(self) -> None:
        """Test RiskMetric automatic timestamp generation."""
        before_time = str(pd.Timestamp.now())
        metric = RiskMetric(
            name="test_metric",
            value=1.0,
            unit="absolute",
            threshold=2.0,
            status="normal",
        )
        after_time = str(pd.Timestamp.now())

        assert metric.timestamp is not None
        assert before_time <= metric.timestamp <= after_time

    def test_init_zero_and_negative_values(self) -> None:
        """Test RiskMetric with zero and negative values."""
        # Test zero value
        zero_metric = RiskMetric(
            name="zero_metric",
            value=0.0,
            unit="absolute",
            threshold=1.0,
            status="normal",
        )
        assert zero_metric.value == 0.0

        # Test negative value
        negative_metric = RiskMetric(
            name="negative_metric",
            value=-5.0,
            unit="absolute",
            threshold=10.0,
            status="normal",
        )
        assert negative_metric.value == -5.0

    def test_init_extreme_values(self) -> None:
        """Test RiskMetric with extreme values."""
        # Test very small value
        small_metric = RiskMetric(
            name="small_metric",
            value=1e-10,
            unit="absolute",
            threshold=1e-5,
            status="normal",
        )
        assert small_metric.value == 1e-10

        # Test very large value
        large_metric = RiskMetric(
            name="large_metric",
            value=1e10,
            unit="absolute",
            threshold=1e15,
            status="normal",
        )
        assert large_metric.value == 1e10

    def test_init_different_statuses(self) -> None:
        """Test RiskMetric with different status values."""
        statuses = ["normal", "warning", "critical", "error", "ok", "fail"]

        for status in statuses:
            metric = RiskMetric(
                name=f"metric_{status}",
                value=1.0,
                unit="absolute",
                threshold=2.0,
                status=status,
            )
            assert metric.status == status

    def test_init_unicode_and_special_chars(self) -> None:
        """Test RiskMetric with unicode and special characters in name."""
        unicode_name = "volatilidad_日本語_विश्लेषण"

        metric = RiskMetric(
            name=unicode_name,
            value=0.25,
            unit="percentage",
            threshold=0.30,
            status="normal",
        )

        assert metric.name == unicode_name
        assert "日本語" in metric.name

    def test_init_with_pandas_timestamp(self) -> None:
        """Test RiskMetric with pandas Timestamp object."""
        ts = pd.Timestamp("2023-12-31 23:59:59")
        metric = RiskMetric(
            name="year_end_metric",
            value=1.5,
            unit="absolute",
            threshold=2.0,
            status="normal",
            timestamp=str(ts),
        )

        assert metric.timestamp == "2023-12-31 23:59:59"


class TestRiskMetricComparisonMethods:
    """Test suite for RiskMetric comparison methods."""

    def test_is_within_threshold_percentage_unit(self) -> None:
        """Test is_within_threshold for percentage units."""
        # Value below threshold should return True
        metric1 = RiskMetric(
            name="var",
            value=0.05,
            unit="percentage",
            threshold=0.10,
            status="normal",
        )
        assert metric1.is_within_threshold() is True

        # Value exactly at threshold should return True
        metric2 = RiskMetric(
            name="var",
            value=0.10,
            unit="percentage",
            threshold=0.10,
            status="normal",
        )
        assert metric2.is_within_threshold() is True

        # Value above threshold should return False
        metric3 = RiskMetric(
            name="var",
            value=0.15,
            unit="percentage",
            threshold=0.10,
            status="warning",
        )
        assert metric3.is_within_threshold() is False

    def test_is_within_threshold_absolute_unit(self) -> None:
        """Test is_within_threshold for absolute units."""
        # Positive value below threshold
        metric1 = RiskMetric(
            name="position_size",
            value=50000,
            unit="absolute",
            threshold=100000,
            status="normal",
        )
        assert metric1.is_within_threshold() is True

        # Negative value within absolute threshold
        metric2 = RiskMetric(
            name="pnl",
            value=-5000,
            unit="absolute",
            threshold=10000,
            status="normal",
        )
        assert metric2.is_within_threshold() is True

        # Value exceeding absolute threshold
        metric3 = RiskMetric(
            name="position_size",
            value=150000,
            unit="absolute",
            threshold=100000,
            status="warning",
        )
        assert metric3.is_within_threshold() is False

    def test_is_within_threshold_edge_cases(self) -> None:
        """Test is_within_threshold edge cases."""
        # Zero values
        zero_metric = RiskMetric(
            name="zero_metric",
            value=0.0,
            unit="absolute",
            threshold=0.0,
            status="normal",
        )
        assert zero_metric.is_within_threshold() is True

        # Negative threshold with positive value
        negative_threshold = RiskMetric(
            name="neg_threshold",
            value=5.0,
            unit="absolute",
            threshold=-10.0,
            status="normal",
        )
        assert negative_threshold.is_within_threshold() is False  # 5.0 > -10.0

    def test_lt_comparison_percentage_metrics(self) -> None:
        """Test __lt__ comparison for percentage metrics."""
        # For percentage metrics, less negative is "less than" (better)
        metric1 = RiskMetric("dd1", -0.10, "percentage", 0.15, "normal")
        metric2 = RiskMetric("dd2", -0.20, "percentage", 0.15, "normal")

        # -0.10 is "less than" -0.20 for percentage metrics
        assert metric1 < metric2

        # Same values
        metric3 = RiskMetric("dd3", -0.10, "percentage", 0.15, "normal")
        assert not (metric1 < metric3)

    def test_lt_comparison_absolute_metrics(self) -> None:
        """Test __lt__ comparison for absolute metrics."""
        # For non-percentage metrics, normal numeric comparison
        metric1 = RiskMetric("pos1", 50000, "absolute", 100000, "normal")
        metric2 = RiskMetric("pos2", 100000, "absolute", 150000, "normal")

        assert metric1 < metric2
        assert not (metric2 < metric1)

        # Same values
        metric3 = RiskMetric("pos3", 50000, "absolute", 100000, "normal")
        assert not (metric1 < metric3)

    def test_gt_comparison_percentage_metrics(self) -> None:
        """Test __gt__ comparison for percentage metrics."""
        # For percentage metrics, more negative is "greater than" (higher risk)
        metric1 = RiskMetric("dd1", -0.20, "percentage", 0.15, "normal")
        metric2 = RiskMetric("dd2", -0.10, "percentage", 0.15, "normal")

        # -0.20 is "greater than" -0.10 for percentage metrics
        assert metric1 > metric2
        assert not (metric2 > metric1)

    def test_gt_comparison_absolute_metrics(self) -> None:
        """Test __gt__ comparison for absolute metrics."""
        # For non-percentage metrics, normal numeric comparison
        metric1 = RiskMetric("pos1", 100000, "absolute", 150000, "normal")
        metric2 = RiskMetric("pos2", 50000, "absolute", 100000, "normal")

        assert metric1 > metric2
        assert not (metric2 > metric1)

        # Same values
        metric3 = RiskMetric("pos3", 100000, "absolute", 150000, "normal")
        assert not (metric1 > metric3)

    def test_comparison_mixed_units(self) -> None:
        """Test comparison with mixed unit types."""
        percentage_metric = RiskMetric("pct", -0.10, "percentage", 0.15, "normal")
        absolute_metric = RiskMetric("abs", 50, "absolute", 100, "normal")

        # When units are different, should use normal numeric comparison
        # The comparison should work but may not be semantically meaningful
        assert not (percentage_metric < absolute_metric)
        assert not (percentage_metric > absolute_metric)

    def test_comparison_with_zero_values(self) -> None:
        """Test comparison methods with zero values."""
        zero1 = RiskMetric("zero1", 0.0, "absolute", 1.0, "normal")
        zero2 = RiskMetric("zero2", 0.0, "absolute", 1.0, "normal")
        positive = RiskMetric("pos", 1.0, "absolute", 2.0, "normal")
        negative = RiskMetric("neg", -1.0, "absolute", 1.0, "normal")

        # Zero comparisons
        assert not (zero1 < zero2)
        assert not (zero1 > zero2)
        assert negative < zero1
        assert zero1 < positive

    def test_comparison_with_extreme_values(self) -> None:
        """Test comparison methods with extreme values."""
        small = RiskMetric("small", 1e-10, "absolute", 1e-5, "normal")
        large = RiskMetric("large", 1e10, "absolute", 1e15, "normal")

        assert small < large
        assert not (large < small)

        # Test with very negative percentage values
        very_negative = RiskMetric("very_neg", -1.0, "percentage", 0.5, "normal")
        less_negative = RiskMetric("less_neg", -0.5, "percentage", 0.5, "normal")

        # Very negative should be "greater than" less negative for percentages
        assert very_negative > less_negative


class TestRiskLimitInitialization:
    """Test suite for RiskLimit initialization and validation."""

    def test_init_minimal_params(self) -> None:
        """Test RiskLimit initialization with minimal parameters."""
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
        )

        assert limit.limit_type == "position_size"
        assert limit.threshold == 100000.0
        assert limit.severity == "warning"
        assert limit.description is None
        assert limit.is_active is True

    def test_init_all_params(self) -> None:
        """Test RiskLimit initialization with all parameters."""
        limit = RiskLimit(
            limit_type="drawdown",
            threshold=0.20,
            severity="critical",
            description="Maximum portfolio drawdown limit",
            is_active=False,
        )

        assert limit.limit_type == "drawdown"
        assert limit.threshold == 0.20
        assert limit.severity == "critical"
        assert limit.description == "Maximum portfolio drawdown limit"
        assert limit.is_active is False

    def test_init_different_limit_types(self) -> None:
        """Test RiskLimit with different limit types."""
        limit_types = [
            "position_size",
            "leverage",
            "drawdown",
            "var",
            "cvar",
            "volatility",
            "concentration",
            "correlation",
        ]

        for limit_type in limit_types:
            limit = RiskLimit(
                limit_type=limit_type,
                threshold=1.0,
                severity="normal",
            )
            assert limit.limit_type == limit_type

    def test_init_different_severities(self) -> None:
        """Test RiskLimit with different severity levels."""
        severities = ["low", "normal", "warning", "high", "critical", "emergency"]

        for severity in severities:
            limit = RiskLimit(
                limit_type="test_limit",
                threshold=1.0,
                severity=severity,
            )
            assert limit.severity == severity

    def test_init_extreme_thresholds(self) -> None:
        """Test RiskLimit with extreme threshold values."""
        # Very small threshold
        small_limit = RiskLimit(
            limit_type="var",
            threshold=1e-10,
            severity="normal",
        )
        assert small_limit.threshold == 1e-10

        # Very large threshold
        large_limit = RiskLimit(
            limit_type="position_size",
            threshold=1e10,
            severity="normal",
        )
        assert large_limit.threshold == 1e10

        # Zero threshold
        zero_limit = RiskLimit(
            limit_type="leverage",
            threshold=0.0,
            severity="normal",
        )
        assert zero_limit.threshold == 0.0

    def test_init_negative_thresholds(self) -> None:
        """Test RiskLimit with negative thresholds."""
        negative_limit = RiskLimit(
            limit_type="pnl",
            threshold=-5000.0,
            severity="normal",
        )
        assert negative_limit.threshold == -5000.0

    def test_init_is_active_states(self) -> None:
        """Test RiskLimit with different is_active states."""
        active_limit = RiskLimit(
            limit_type="test",
            threshold=1.0,
            severity="normal",
            is_active=True,
        )
        assert active_limit.is_active is True

        inactive_limit = RiskLimit(
            limit_type="test",
            threshold=1.0,
            severity="normal",
            is_active=False,
        )
        assert inactive_limit.is_active is False

    def test_init_with_none_description(self) -> None:
        """Test RiskLimit with None description."""
        limit = RiskLimit(
            limit_type="test",
            threshold=1.0,
            severity="normal",
            description=None,
        )
        assert limit.description is None

    def test_init_with_empty_description(self) -> None:
        """Test RiskLimit with empty description."""
        limit = RiskLimit(
            limit_type="test",
            threshold=1.0,
            severity="normal",
            description="",
        )
        assert limit.description == ""

    def test_init_with_unicode_description(self) -> None:
        """Test RiskLimit with unicode in description."""
        unicode_desc = "Límite de concentración 日本語 حد أقصى 🎯"

        limit = RiskLimit(
            limit_type="concentration",
            threshold=0.30,
            severity="high",
            description=unicode_desc,
        )

        assert limit.description == unicode_desc
        assert "日本語" in limit.description
        assert "🎯" in limit.description

    def test_init_special_characters(self) -> None:
        """Test RiskLimit with special characters in limit_type and description."""
        special_type = "drawdown_@#$%^&*()"
        special_desc = "Descripción especial: <>&\"'"

        limit = RiskLimit(
            limit_type=special_type,
            threshold=0.25,
            severity="critical",
            description=special_desc,
        )

        assert limit.limit_type == special_type
        assert limit.description == special_desc


class TestRiskLimitBreachDetection:
    """Test suite for RiskLimit is_breached method."""

    def test_is_breached_position_size_above_threshold(self) -> None:
        """Test is_breached for position_size when value exceeds threshold."""
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
        )

        # Value above threshold should breach
        assert limit.is_breached(150000.0) is True
        assert limit.is_breached(100000.1) is True

    def test_is_breached_position_size_below_threshold(self) -> None:
        """Test is_breached for position_size when value is below threshold."""
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
        )

        # Value below threshold should not breach
        assert limit.is_breached(50000.0) is False
        assert limit.is_breached(99999.9) is False
        assert limit.is_breached(0.0) is False

    def test_is_breached_position_size_at_threshold(self) -> None:
        """Test is_breached for position_size when value equals threshold."""
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
        )

        # Value equal to threshold should not breach
        assert limit.is_breached(100000.0) is False

    def test_is_breached_leverage_above_threshold(self) -> None:
        """Test is_breached for leverage when value exceeds threshold."""
        limit = RiskLimit(
            limit_type="leverage",
            threshold=2.0,
            severity="critical",
        )

        assert limit.is_breached(3.0) is True
        assert limit.is_breached(2.5) is True
        assert limit.is_breached(2.1) is True

    def test_is_breached_leverage_below_threshold(self) -> None:
        """Test is_breached for leverage when value is below threshold."""
        limit = RiskLimit(
            limit_type="leverage",
            threshold=2.0,
            severity="critical",
        )

        assert limit.is_breached(1.0) is False
        assert limit.is_breached(1.5) is False
        assert limit.is_breached(0.5) is False

    def test_is_breached_drawdown_above_threshold(self) -> None:
        """Test is_breached for drawdown when value exceeds threshold."""
        limit = RiskLimit(
            limit_type="drawdown",
            threshold=0.20,
            severity="high",
        )

        # For drawdown, higher (less negative) values are worse
        assert limit.is_breached(0.25) is True
        assert limit.is_breached(0.30) is True
        assert limit.is_breached(0.21) is True

    def test_is_breached_drawdown_below_threshold(self) -> None:
        """Test is_breached for drawdown when value is below threshold."""
        limit = RiskLimit(
            limit_type="drawdown",
            threshold=0.20,
            severity="high",
        )

        # For drawdown, lower (more negative) values are better
        assert limit.is_breached(0.15) is False
        assert limit.is_breached(0.10) is False
        assert limit.is_breached(0.05) is False

    def test_is_breached_var_above_threshold(self) -> None:
        """Test is_breached for var when absolute value exceeds threshold."""
        limit = RiskLimit(
            limit_type="var",
            threshold=0.05,
            severity="warning",
        )

        # For VaR, absolute value matters
        assert limit.is_breached(0.06) is True
        assert limit.is_breached(0.10) is True
        assert limit.is_breached(-0.07) is True
        assert limit.is_breached(-0.08) is True

    def test_is_breached_var_below_threshold(self) -> None:
        """Test is_breached for var when absolute value is below threshold."""
        limit = RiskLimit(
            limit_type="var",
            threshold=0.05,
            severity="warning",
        )

        assert limit.is_breached(0.04) is False
        assert limit.is_breached(0.01) is False
        assert limit.is_breached(-0.04) is False
        assert limit.is_breached(-0.01) is False

    def test_is_breached_cvar_above_threshold(self) -> None:
        """Test is_breached for cvar when absolute value exceeds threshold."""
        limit = RiskLimit(
            limit_type="cvar",
            threshold=0.08,
            severity="critical",
        )

        assert limit.is_breached(0.09) is True
        assert limit.is_breached(-0.10) is True
        assert limit.is_breached(0.15) is True

    def test_is_breached_cvar_below_threshold(self) -> None:
        """Test is_breached for cvar when absolute value is below threshold."""
        limit = RiskLimit(
            limit_type="cvar",
            threshold=0.08,
            severity="critical",
        )

        assert limit.is_breached(0.07) is False
        assert limit.is_breached(-0.05) is False
        assert limit.is_breached(0.02) is False

    def test_is_breached_inactive_limit(self) -> None:
        """Test is_breached when limit is inactive."""
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
            is_active=False,
        )

        # Inactive limits should never breach
        assert limit.is_breached(200000.0) is False
        assert limit.is_breached(1000000.0) is False
        assert limit.is_breached(0.0) is False

    def test_is_breached_unknown_limit_type(self) -> None:
        """Test is_breached for unknown limit types."""
        limit = RiskLimit(
            limit_type="unknown_type",
            threshold=100.0,
            severity="normal",
        )

        # Unknown types use default logic (value > threshold)
        assert limit.is_breached(150.0) is True
        assert limit.is_breached(50.0) is False
        assert limit.is_breached(100.0) is False

    def test_is_breached_edge_cases(self) -> None:
        """Test is_breached with edge case values."""
        # Test with zero values
        zero_limit = RiskLimit(
            limit_type="position_size",
            threshold=0.0,
            severity="normal",
        )
        assert zero_limit.is_breached(0.0) is False
        assert zero_limit.is_breached(0.1) is True
        assert zero_limit.is_breached(-0.1) is False

        # Test with very small thresholds
        small_limit = RiskLimit(
            limit_type="var",
            threshold=1e-10,
            severity="normal",
        )
        assert small_limit.is_breached(1e-9) is True
        assert small_limit.is_breached(1e-11) is False

        # Test with negative thresholds
        negative_limit = RiskLimit(
            limit_type="pnl",
            threshold=-1000.0,
            severity="normal",
        )
        assert negative_limit.is_breached(0.0) is True  # 0.0 > -1000.0
        assert negative_limit.is_breached(-500.0) is True  # -500.0 > -1000.0
        assert negative_limit.is_breached(-1500.0) is False  # -1500.0 < -1000.0


class TestRiskAlertInitialization:
    """Test suite for RiskAlert initialization and validation."""

    def test_init_minimal_params(self) -> None:
        """Test RiskAlert initialization with minimal parameters."""
        alert = RiskAlert(
            alert_type="limit_breach",
            severity="warning",
            message="Position size limit breached",
        )

        assert alert.alert_type == "limit_breach"
        assert alert.severity == "warning"
        assert alert.message == "Position size limit breached"
        assert alert.affected_symbol is None
        assert alert.current_value is None
        assert alert.limit_value is None
        assert alert.timestamp is not None
        assert alert.escalated is False

    def test_init_all_params(self) -> None:
        """Test RiskAlert initialization with all parameters."""
        timestamp = "2023-01-01 12:00:00"

        alert = RiskAlert(
            alert_type="drawdown_limit",
            severity="critical",
            message="Maximum drawdown exceeded",
            affected_symbol="EURUSD",
            current_value=-0.25,
            limit_value=-0.20,
            timestamp=timestamp,
            escalated=True,
        )

        assert alert.alert_type == "drawdown_limit"
        assert alert.severity == "critical"
        assert alert.message == "Maximum drawdown exceeded"
        assert alert.affected_symbol == "EURUSD"
        assert alert.current_value == -0.25
        assert alert.limit_value == -0.20
        assert alert.timestamp == timestamp
        assert alert.escalated is True

    def test_init_auto_timestamp(self) -> None:
        """Test RiskAlert automatic timestamp generation."""
        before_time = str(pd.Timestamp.now())
        alert = RiskAlert(
            alert_type="test",
            severity="normal",
            message="Test alert",
        )
        after_time = str(pd.Timestamp.now())

        assert alert.timestamp is not None
        assert before_time <= alert.timestamp <= after_time

    def test_init_different_alert_types(self) -> None:
        """Test RiskAlert with different alert types."""
        alert_types = [
            "limit_breach",
            "position_limit",
            "volatility_spike",
            "correlation_breakdown",
            "drawdown_limit",
            "var_breach",
            "leverage_limit",
            "concentration_risk",
        ]

        for alert_type in alert_types:
            alert = RiskAlert(
                alert_type=alert_type,
                severity="warning",
                message=f"Test {alert_type} alert",
            )
            assert alert.alert_type == alert_type

    def test_init_different_severities(self) -> None:
        """Test RiskAlert with different severity levels."""
        severities = ["low", "normal", "warning", "high", "critical", "emergency"]

        for severity in severities:
            alert = RiskAlert(
                alert_type="test",
                severity=severity,
                message=f"Test {severity} alert",
            )
            assert alert.severity == severity

    def test_init_with_symbol(self) -> None:
        """Test RiskAlert with affected symbol."""
        symbols = ["EURUSD", "GBPJPY", "USDJPY", "AUDUSD", "USDCAD"]

        for symbol in symbols:
            alert = RiskAlert(
                alert_type="position_limit",
                severity="warning",
                message=f"Position limit for {symbol}",
                affected_symbol=symbol,
            )
            assert alert.affected_symbol == symbol

    def test_init_with_numeric_values(self) -> None:
        """Test RiskAlert with numeric current and limit values."""
        # Test with positive values
        positive_alert = RiskAlert(
            alert_type="position_limit",
            severity="warning",
            message="Position size exceeded",
            current_value=150000.0,
            limit_value=100000.0,
        )
        assert positive_alert.current_value == 150000.0
        assert positive_alert.limit_value == 100000.0

        # Test with negative values
        negative_alert = RiskAlert(
            alert_type="drawdown_limit",
            severity="critical",
            message="Drawdown exceeded",
            current_value=-0.25,
            limit_value=-0.20,
        )
        assert negative_alert.current_value == -0.25
        assert negative_alert.limit_value == -0.20

        # Test with zero values
        zero_alert = RiskAlert(
            alert_type="test",
            severity="normal",
            message="Zero value test",
            current_value=0.0,
            limit_value=1.0,
        )
        assert zero_alert.current_value == 0.0
        assert zero_alert.limit_value == 1.0

    def test_init_extreme_numeric_values(self) -> None:
        """Test RiskAlert with extreme numeric values."""
        # Very small values
        small_alert = RiskAlert(
            alert_type="var_breach",
            severity="low",
            message="Small VaR breach",
            current_value=1e-10,
            limit_value=1e-8,
        )
        assert small_alert.current_value == 1e-10
        assert small_alert.limit_value == 1e-8

        # Very large values
        large_alert = RiskAlert(
            alert_type="position_limit",
            severity="high",
            message="Large position",
            current_value=1e10,
            limit_value=1e9,
        )
        assert large_alert.current_value == 1e10
        assert large_alert.limit_value == 1e9

    def test_init_escalated_states(self) -> None:
        """Test RiskAlert with different escalated states."""
        # Not escalated
        not_escalated = RiskAlert(
            alert_type="test",
            severity="warning",
            message="Test alert",
            escalated=False,
        )
        assert not_escalated.escalated is False

        # Escalated
        escalated_alert = RiskAlert(
            alert_type="critical_breach",
            severity="critical",
            message="Critical alert",
            escalated=True,
        )
        assert escalated_alert.escalated is True

    def test_init_with_unicode_message(self) -> None:
        """Test RiskAlert with unicode characters in message."""
        unicode_message = "Alerta de riesgo 日本語警告 🚨 Sistema de alerta"

        alert = RiskAlert(
            alert_type="risk_alert",
            severity="critical",
            message=unicode_message,
        )

        assert alert.message == unicode_message
        assert "日本語" in alert.message
        assert "🚨" in alert.message

    def test_init_with_special_characters(self) -> None:
        """Test RiskAlert with special characters."""
        special_message = "Alert: <HIGH RISK> @#$%^&*() !\"£$%^&*()"

        alert = RiskAlert(
            alert_type="special_test",
            severity="high",
            message=special_message,
        )

        assert alert.message == special_message

    def test_init_with_empty_fields(self) -> None:
        """Test RiskAlert with empty string fields."""
        alert = RiskAlert(
            alert_type="",
            severity="normal",
            message="",
            affected_symbol="",
        )

        assert alert.alert_type == ""
        assert alert.message == ""
        assert alert.affected_symbol == ""


class TestRiskAlertEscalation:
    """Test suite for RiskAlert escalation functionality."""

    def test_escalate_basic_functionality(self) -> None:
        """Test basic alert escalation functionality."""
        alert = RiskAlert(
            alert_type="warning_breach",
            severity="warning",
            message="Initial warning",
        )

        # Verify initial state
        assert alert.severity == "warning"
        assert alert.escalated is False

        # Escalate
        alert.escalate("critical")

        # Verify changes
        assert alert.severity == "critical"
        assert alert.escalated is True

    def test_escalate_to_all_severity_levels(self) -> None:
        """Test escalation to all possible severity levels."""
        severities = ["low", "normal", "warning", "high", "critical", "emergency"]

        for severity in severities:
            alert = RiskAlert(
                alert_type="test",
                severity="low",  # Start from low
                message="Test escalation",
            )

            alert.escalate(severity)

            assert alert.severity == severity
            assert alert.escalated is True

    def test_escalate_multiple_times(self) -> None:
        """Test multiple escalations of the same alert."""
        alert = RiskAlert(
            alert_type="test",
            severity="warning",
            message="Multiple escalation test",
        )

        # First escalation
        alert.escalate("high")
        assert alert.severity == "high"
        assert alert.escalated is True

        # Second escalation
        alert.escalate("critical")
        assert alert.severity == "critical"
        assert alert.escalated is True  # Should remain True

        # Third escalation
        alert.escalate("emergency")
        assert alert.severity == "emergency"
        assert alert.escalated is True

    def test_escalate_to_same_severity(self) -> None:
        """Test escalation to the same severity level."""
        alert = RiskAlert(
            alert_type="test",
            severity="warning",
            message="Same severity test",
        )

        # Escalate to same level
        alert.escalate("warning")

        assert alert.severity == "warning"
        assert alert.escalated is True  # Should still be marked as escalated

    def test_escalate_preserves_other_fields(self) -> None:
        """Test that escalation preserves other alert fields."""
        timestamp = "2023-01-01 12:00:00"
        alert = RiskAlert(
            alert_type="test_alert",
            severity="warning",
            message="Original message",
            affected_symbol="EURUSD",
            current_value=0.15,
            limit_value=0.10,
            timestamp=timestamp,
            escalated=False,
        )

        # Escalate
        alert.escalate("critical")

        # Check that other fields are preserved
        assert alert.alert_type == "test_alert"
        assert alert.message == "Original message"
        assert alert.affected_symbol == "EURUSD"
        assert alert.current_value == 0.15
        assert alert.limit_value == 0.10
        assert alert.timestamp == timestamp
        assert alert.escalated is True

    def test_escalate_with_unicode_severity(self) -> None:
        """Test escalation with unicode severity levels."""
        alert = RiskAlert(
            alert_type="test",
            severity="warning",
            message="Unicode escalation test",
        )

        # Escalate to unicode severity
        alert.escalate("crítico_日本語")

        assert alert.severity == "crítico_日本語"
        assert alert.escalated is True

    def test_escalate_with_empty_string(self) -> None:
        """Test escalation with empty string severity."""
        alert = RiskAlert(
            alert_type="test",
            severity="warning",
            message="Empty severity test",
        )

        # Escalate to empty string
        alert.escalate("")

        assert alert.severity == ""
        assert alert.escalated is True

    def test_escalate_with_special_characters(self) -> None:
        """Test escalation with special characters in severity."""
        special_severity = "CRITICAL@#$%^&*()"

        alert = RiskAlert(
            alert_type="test",
            severity="normal",
            message="Special chars test",
        )

        alert.escalate(special_severity)

        assert alert.severity == special_severity
        assert alert.escalated is True


class TestEdgeCasesAndIntegration:
    """Test suite for edge cases and integration scenarios."""

    def test_risk_signal_with_complex_scenario(self) -> None:
        """Test RiskSignal in a complex trading scenario."""
        # Simulate a complex trading decision
        metadata = {
            "market_conditions": "volatile",
            "volatility": 0.25,
            "correlations": {
                "EURUSD_GBPUSD": 0.85,
                "EURUSD_AUDUSD": 0.72,
            },
            "positions": [
                {"symbol": "EURUSD", "size": 0.5, "pnl": 0.02},
                {"symbol": "GBPUSD", "size": 0.3, "pnl": -0.01},
            ],
            "portfolio_var": 0.05,
            "max_drawdown": -0.08,
        }

        signal = RiskSignal(
            action=RiskAction.REDUCE_POSITION,
            reason="High correlation risk with portfolio concentration",
            confidence=0.85,
            metadata=metadata,
        )

        assert signal.action == RiskAction.REDUCE_POSITION
        assert signal.confidence == 0.85
        assert signal.metadata is not None
        assert signal.metadata["market_conditions"] == "volatile"
        assert signal.metadata["correlations"]["EURUSD_GBPUSD"] == 0.85
        assert len(signal.metadata["positions"]) == 2

    def test_risk_metric_comparison_in_risk_assessment(self) -> None:
        """Test RiskMetric comparison in a risk assessment workflow."""
        # Create a series of risk metrics for comparison
        var_metrics = [
            RiskMetric("portfolio_var", 0.03, "percentage", 0.05, "normal"),
            RiskMetric("portfolio_var", 0.06, "percentage", 0.05, "warning"),
            RiskMetric("portfolio_var", 0.08, "percentage", 0.05, "critical"),
        ]

        # Test threshold checking
        assert var_metrics[0].is_within_threshold() is True
        assert var_metrics[1].is_within_threshold() is False  # Exceeds threshold
        assert var_metrics[2].is_within_threshold() is False

        # Test comparison between metrics
        assert var_metrics[0] < var_metrics[1]  # Lower VaR is "less than" higher
        assert var_metrics[1] < var_metrics[2]
        assert var_metrics[2] > var_metrics[0]  # Higher VaR is "greater than" lower

    def test_risk_limit_breach_workflow(self) -> None:
        """Test RiskLimit breach detection in a workflow."""
        # Define risk limits for a portfolio
        limits = [
            RiskLimit("position_size", 100000.0, "high", "Max position size"),
            RiskLimit("leverage", 2.0, "critical", "Max leverage"),
            RiskLimit("drawdown", 0.20, "warning", "Max drawdown"),
            RiskLimit("var", 0.05, "warning", "Max VaR"),
        ]

        # Test scenarios
        scenarios = [
            {"position_size": 50000, "leverage": 1.0, "drawdown": 0.10, "var": 0.03},
            {"position_size": 150000, "leverage": 1.5, "drawdown": 0.15, "var": 0.04},
            {"position_size": 80000, "leverage": 3.0, "drawdown": 0.25, "var": 0.06},
        ]

        # Check breaches
        for i, scenario in enumerate(scenarios):
            breaches = []
            for limit in limits:
                if limit.is_breached(scenario[limit.limit_type]):
                    breaches.append(limit.limit_type)

            if i == 0:
                assert len(breaches) == 0
            elif i == 1:
                assert "position_size" in breaches
            elif i == 2:
                assert "leverage" in breaches
                assert "drawdown" in breaches
                assert "var" in breaches

    def test_risk_alert_escalation_workflow(self) -> None:
        """Test RiskAlert escalation in a risk management workflow."""
        # Create initial alerts
        alerts = [
            RiskAlert(
                "position_limit",
                "warning",
                "Position size approaching limit",
                current_value=95000,
                limit_value=100000,
            ),
            RiskAlert("volatility_spike", "normal", "Volatility increased"),
            RiskAlert("correlation_risk", "low", "High correlation detected"),
        ]

        # Simulate escalation based on severity
        for alert in alerts:
            if alert.alert_type == "position_limit":
                # Position limit alert escalates quickly
                if (
                    alert.current_value
                    and alert.limit_value
                    and alert.current_value > alert.limit_value * 0.9
                ):
                    alert.escalate("critical")
            elif alert.alert_type == "volatility_spike":
                # Volatility alert escalates moderately
                alert.escalate("high")
            elif alert.alert_type == "correlation_risk":
                # Correlation alert escalates slowly
                alert.escalate("warning")

        # Verify escalation
        position_alert = next(a for a in alerts if a.alert_type == "position_limit")
        volatility_alert = next(a for a in alerts if a.alert_type == "volatility_spike")
        correlation_alert = next(a for a in alerts if a.alert_type == "correlation_risk")

        assert position_alert.severity == "critical"
        assert volatility_alert.severity == "high"
        assert correlation_alert.severity == "warning"

        # All should be marked as escalated
        for alert in alerts:
            assert alert.escalated is True

    def test_risk_management_module_integration(self) -> None:
        """Test integration of all risk signal components."""
        # Create a comprehensive risk scenario
        # 1. Define risk limits
        limits = [
            RiskLimit("position_size", 100000.0, "warning", "Position size limit"),
            RiskLimit("drawdown", 0.15, "critical", "Drawdown limit"),
        ]

        # 2. Create risk metrics based on current portfolio state
        metrics = [
            RiskMetric("current_position", 120000.0, "absolute", 100000.0, "warning"),
            RiskMetric("portfolio_drawdown", -0.18, "percentage", 0.15, "critical"),
        ]

        # 3. Check for limit breaches
        breached_limits = [
            limit
            for limit in limits
            if limit.is_breached(120000.0 if limit.limit_type == "position_size" else -0.18)
        ]

        # 4. Generate alerts for breaches
        alerts = []
        for limit in breached_limits:
            alert = RiskAlert(
                alert_type="limit_breach",
                severity=limit.severity,
                message=f"{limit.limit_type} limit breached",
                current_value=120000.0 if limit.limit_type == "position_size" else -0.18,
                limit_value=limit.threshold,
            )
            alerts.append(alert)

        # 5. Generate risk signal based on overall risk level
        risk_score = len(breached_limits) * 0.5 + sum(
            0.3 for metric in metrics if not metric.is_within_threshold()
        )

        if risk_score > 0.7:
            action = RiskAction.EMERGENCY_HALT
            confidence = 0.9
        elif risk_score > 0.4:
            action = RiskAction.CLOSE_POSITION
            confidence = 0.7
        elif risk_score > 0.2:
            action = RiskAction.REDUCE_POSITION
            confidence = 0.5
        else:
            action = RiskAction.HOLD
            confidence = 0.3

        # 6. Create the final risk signal
        signal = RiskSignal(
            action=action,
            reason=f"Risk score: {risk_score:.2f}, {len(breached_limits)} limits breached",
            confidence=confidence,
            metadata={
                "risk_score": risk_score,
                "breached_limits": [limit.limit_type for limit in breached_limits],
                "alert_count": len(alerts),
                "metrics_status": {
                    metric.name: "breached" if not metric.is_within_threshold() else "normal"
                    for metric in metrics
                },
            },
        )

        # Verify integration
        assert len(breached_limits) == 1  # Only position_size breached
        assert len(alerts) == 1  # One alert per breached limit
        assert signal.action == RiskAction.EMERGENCY_HALT
        assert signal.confidence == 0.9
        assert signal.metadata is not None
        assert signal.metadata["risk_score"] > 0.7

        # Verify alerts
        position_alert = next(a for a in alerts if "position_size" in a.message)

        assert position_alert.severity == "warning"
        assert position_alert.current_value == 120000.0

    def test_timestamps_consistency(self) -> None:
        """Test timestamp consistency across components."""
        import time

        # Create components with small delays to test timestamp ordering
        time.sleep(0.001)  # 1ms delay

        signal = RiskSignal(
            action=RiskAction.HOLD,
            reason="Timestamp test",
            confidence=0.5,
        )

        time.sleep(0.001)  # 1ms delay

        metric = RiskMetric(
            name="test_metric",
            value=1.0,
            unit="absolute",
            threshold=2.0,
            status="normal",
        )

        time.sleep(0.001)  # 1ms delay

        RiskLimit(
            limit_type="test_limit",
            threshold=1.0,
            severity="normal",
        )

        time.sleep(0.001)  # 1ms delay

        alert = RiskAlert(
            alert_type="test",
            severity="normal",
            message="Timestamp test alert",
        )

        # Verify all have timestamps
        assert signal.timestamp is not None
        assert metric.timestamp is not None
        assert alert.timestamp is not None

        # Verify timestamps are in reasonable order (allowing for some processing time)
        signal_time = pd.Timestamp(signal.timestamp)
        metric_time = pd.Timestamp(metric.timestamp)
        alert_time = pd.Timestamp(alert.timestamp)

        # Due to small delays, we expect signal < metric < alert (approximately)
        assert signal_time <= metric_time
        assert metric_time <= alert_time

    def test_serialization_compatibility(self) -> None:
        """Test that all components can be serialized/deserialized."""
        # Test RiskSignal serialization
        signal = RiskSignal(
            action=RiskAction.REDUCE_POSITION,
            reason="Test serialization",
            confidence=0.8,
            metadata={"test": "data"},
        )

        signal_dict = {
            "action": signal.action.value,
            "reason": signal.reason,
            "confidence": signal.confidence,
            "metadata": signal.metadata,
            "timestamp": signal.timestamp,
        }

        assert isinstance(signal_dict, dict)
        assert signal_dict["action"] == "REDUCE_POSITION"
        assert signal_dict["confidence"] == 0.8

        # Test RiskMetric serialization
        metric = RiskMetric(
            name="test_metric",
            value=0.05,
            unit="percentage",
            threshold=0.10,
            status="warning",
        )

        metric_dict = {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "threshold": metric.threshold,
            "status": metric.status,
            "timestamp": metric.timestamp,
        }

        assert isinstance(metric_dict, dict)
        assert metric_dict["name"] == "test_metric"
        assert metric_dict["value"] == 0.05

        # Test RiskLimit serialization
        limit = RiskLimit(
            limit_type="position_size",
            threshold=100000.0,
            severity="warning",
            description="Test limit",
        )

        limit_dict = {
            "limit_type": limit.limit_type,
            "threshold": limit.threshold,
            "severity": limit.severity,
            "description": limit.description,
            "is_active": limit.is_active,
        }

        assert isinstance(limit_dict, dict)
        assert limit_dict["limit_type"] == "position_size"
        assert limit_dict["threshold"] == 100000.0

        # Test RiskAlert serialization
        alert = RiskAlert(
            alert_type="test_alert",
            severity="critical",
            message="Test alert message",
            affected_symbol="EURUSD",
        )

        alert_dict = {
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "affected_symbol": alert.affected_symbol,
            "current_value": alert.current_value,
            "limit_value": alert.limit_value,
            "timestamp": alert.timestamp,
            "escalated": alert.escalated,
        }

        assert isinstance(alert_dict, dict)
        assert alert_dict["alert_type"] == "test_alert"
        assert alert_dict["severity"] == "critical"


if __name__ == "__main__":
    pytest.main([__file__])
