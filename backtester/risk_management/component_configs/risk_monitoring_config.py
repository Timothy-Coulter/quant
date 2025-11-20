"""Risk Monitoring Configuration Models.

This module provides Pydantic models for risk monitoring configuration.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskMonitoringConfig(BaseModel):
    """Configuration for risk monitoring using pydantic BaseModel.

    This class defines all parameters needed to configure risk monitoring behavior,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core monitoring settings
    check_interval: int = Field(
        default=60, ge=1, le=3600, description="Time between risk checks in seconds"
    )
    enable_real_time_alerts: bool = Field(
        default=True, description="Whether to enable real-time alerts"
    )
    max_history_size: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Maximum number of historical measurements to keep",
    )

    # Alert thresholds
    volatility_threshold: float = Field(
        default=0.30, ge=0.05, le=1.0, description="Volatility threshold for alerts"
    )
    drawdown_threshold: float = Field(
        default=0.15, ge=0.05, le=1.0, description="Drawdown threshold for alerts"
    )
    leverage_threshold: float = Field(
        default=3.0, ge=1.0, le=10.0, description="Leverage threshold for alerts"
    )

    # Risk metric specific thresholds
    var_threshold: float = Field(
        default=0.06, ge=0.01, le=0.5, description="VaR threshold for alerts"
    )
    correlation_threshold: float = Field(
        default=0.85, ge=0.1, le=1.0, description="Average correlation threshold for alerts"
    )

    # Risk monitoring parameters
    lookback_period: int = Field(
        default=252, ge=20, le=1000, description="Lookback period for risk calculations"
    )
    confidence_level: float = Field(
        default=0.95, ge=0.90, le=0.99, description="Confidence level for VaR calculations"
    )

    # Alert management
    alert_escalation_rules: dict[AlertSeverity, list[str]] = Field(
        default_factory=lambda: {
            AlertSeverity.LOW: [],
            AlertSeverity.MEDIUM: ["reduce_position_sizes"],
            AlertSeverity.HIGH: ["reduce_position_sizes", "increase_cash_position"],
            AlertSeverity.CRITICAL: ["emergency_reduction", "halt_trading"],
        },
        description="Escalation rules by alert severity",
    )

    # Risk metric tracking
    track_metrics: list[str] = Field(
        default_factory=lambda: [
            "volatility",
            "drawdown",
            "leverage",
            "var",
            "correlation",
            "concentration",
        ],
        description="List of metrics to track",
    )

    # Dashboard and reporting
    generate_dashboard_data: bool = Field(
        default=True, description="Whether to generate dashboard data"
    )
    reporting_frequency: str = Field(
        default="daily", description="Frequency for generating reports"
    )

    @property
    def monitoring_enabled(self) -> bool:
        """Check if risk monitoring is enabled."""
        return self.check_interval > 0

    def get_alert_severity(self, metric_value: float, threshold: float) -> AlertSeverity:
        """Determine alert severity based on metric value and threshold."""
        ratio = metric_value / threshold if threshold > 0 else 0

        if ratio >= 1.5:
            return AlertSeverity.CRITICAL
        elif ratio >= 1.2:
            return AlertSeverity.HIGH
        elif ratio >= 1.0:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
