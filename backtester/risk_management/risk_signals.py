"""Risk Signals and Actions.

This module provides enums and data structures for risk signals and actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class RiskAction(str, Enum):
    """Risk management action types."""

    HOLD = "HOLD"
    REDUCE_POSITION = "REDUCE_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    INCREASE_POSITION = "INCREASE_POSITION"
    EMERGENCY_HALT = "EMERGENCY_HALT"
    REVIEW_POSITIONS = "REVIEW_POSITIONS"


@dataclass
class RiskSignal:
    """Risk management signal."""

    action: RiskAction
    reason: str
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())


@dataclass
class RiskMetric:
    """Risk metric with threshold and status tracking."""

    name: str
    value: float
    unit: str
    threshold: float
    status: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())

    def is_within_threshold(self) -> bool:
        """Check if metric is within acceptable threshold."""
        if self.unit == 'percentage':
            # For percentage metrics, check if the absolute value is within threshold
            # This handles both positive (VaR) and negative (drawdown) values
            return abs(self.value) <= abs(self.threshold)
        else:
            return self.value <= self.threshold

    def __lt__(self, other: 'RiskMetric') -> bool:
        """Less than comparison for RiskMetric.

        For percentage metrics, the comparison depends on the metric type:
        - VaR metrics: lower values are "less than" higher values (less risk)
        - Drawdown metrics: more negative values are "less than" less negative values (more risk)

        Args:
            other: Another RiskMetric to compare with

        Returns:
            bool: True if this metric is "less than" other
        """
        if self.unit == 'percentage' and other.unit == 'percentage':
            # Check if this is a VaR-type metric (positive values, lower is better)
            if 'var' in self.name.lower() and 'var' in other.name.lower():
                return self.value < other.value  # Normal comparison for VaR
            # For drawdown and other negative metrics, more negative is "less than"
            else:
                return self.value > other.value  # More negative is "less than" (worse)
        # For mixed units, prevent meaningless comparisons by returning False
        if self.unit != other.unit:
            return False
        return self.value < other.value

    def __gt__(self, other: 'RiskMetric') -> bool:
        """Greater than comparison for RiskMetric.

        For percentage metrics, the comparison depends on the metric type:
        - VaR metrics: higher values are "greater than" lower values (more risk)
        - Drawdown metrics: less negative values are "greater than" more negative values (less risk)

        Args:
            other: Another RiskMetric to compare with

        Returns:
            bool: True if this metric is "greater than" other
        """
        if self.unit == 'percentage' and other.unit == 'percentage':
            # Check if this is a VaR-type metric (positive values, lower is better)
            if 'var' in self.name.lower() and 'var' in other.name.lower():
                return self.value > other.value  # Normal comparison for VaR
            # For drawdown and other negative metrics, less negative is "greater than"
            else:
                return self.value < other.value  # Less negative is "greater than" (better)
        # For mixed units, prevent meaningless comparisons by returning False
        if self.unit != other.unit:
            return False
        return self.value > other.value


@dataclass
class RiskLimit:
    """Risk limit configuration."""

    limit_type: str
    threshold: float
    severity: str
    description: str | None = None
    is_active: bool = True

    def is_breached(self, current_value: float) -> bool:
        """Check if limit is breached.

        Args:
            current_value: Current metric value

        Returns:
            True if limit is breached
        """
        if not self.is_active:
            return False

        if self.limit_type in ['position_size', 'leverage']:
            return current_value > self.threshold
        elif self.limit_type == 'drawdown':
            # For drawdown, more negative values are better (less loss)
            # So we breach when current_value is less negative than threshold
            return current_value > self.threshold
        elif self.limit_type in ['var', 'cvar']:
            return abs(current_value) > abs(self.threshold)
        else:
            return current_value > self.threshold


@dataclass
class RiskAlert:
    """Risk alert for limit breaches."""

    alert_type: str
    severity: str
    message: str
    affected_symbol: str | None = None
    current_value: float | None = None
    limit_value: float | None = None
    timestamp: str | None = None
    escalated: bool = False

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = str(pd.Timestamp.now())

    def escalate(self, new_severity: str) -> None:
        """Escalate alert severity.

        Args:
            new_severity: New severity level
        """
        self.severity = new_severity
        self.escalated = True
