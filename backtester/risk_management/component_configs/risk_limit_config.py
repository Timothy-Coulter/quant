"""Risk Limits Configuration Models.

This module provides Pydantic models for risk limits configuration.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..risk_profile import RiskProfile


class LimitSeverity(str, Enum):
    """Risk limit severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLimitConfig(BaseModel):
    """Configuration for risk limits using pydantic BaseModel.

    This class defines all parameters needed to configure risk limits behavior,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core risk limits
    max_drawdown: float = Field(
        default=0.20, ge=0.01, le=1.0, description="Maximum portfolio drawdown limit"
    )
    max_leverage: float = Field(
        default=3.0, ge=1.0, le=20.0, description="Maximum leverage ratio limit"
    )
    max_single_position: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="Maximum single position size as fraction of portfolio",
    )
    max_portfolio_var: float = Field(
        default=0.05, ge=0.01, le=0.5, description="Maximum portfolio Value at Risk (VaR)"
    )
    max_daily_loss: float = Field(
        default=0.05, ge=0.01, le=0.5, description="Maximum daily loss limit"
    )
    max_sector_exposure: float = Field(
        default=0.30, ge=0.05, le=1.0, description="Maximum sector exposure limit"
    )
    max_correlation: float = Field(
        default=0.80, ge=0.1, le=1.0, description="Maximum correlation for position sizing"
    )
    max_volatility: float = Field(
        default=0.25, ge=0.05, le=1.0, description="Maximum portfolio volatility"
    )

    # Concentration and diversification limits
    concentration_limit: float = Field(
        default=0.25, ge=0.05, le=1.0, description="Maximum concentration in single position"
    )
    min_diversification_ratio: float = Field(
        default=0.60, ge=0.1, le=1.0, description="Minimum diversification ratio"
    )
    max_country_exposure: float = Field(
        default=0.60, ge=0.1, le=1.0, description="Maximum country/region exposure"
    )
    max_industry_exposure: float = Field(
        default=0.25, ge=0.05, le=1.0, description="Maximum industry exposure"
    )

    # Position-specific limits
    max_positions: int = Field(default=50, ge=1, le=1000, description="Maximum number of positions")
    min_position_size: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Minimum position size as fraction of portfolio"
    )

    # Risk profile integration
    risk_profile: RiskProfile = Field(
        default_factory=lambda: RiskProfile.moderate(), description="Associated risk profile"
    )

    # Alert and escalation settings
    alert_threshold_pct: float = Field(
        default=0.80,
        ge=0.1,
        le=1.0,
        description="Alert threshold as percentage of limit (0.80 = alert at 80% of limit)",
    )
    emergency_halt_threshold: float = Field(
        default=0.95, ge=0.1, le=1.0, description="Emergency halt threshold as percentage of limit"
    )
    auto_rebalance_trigger: bool = Field(
        default=True, description="Whether to trigger automatic rebalancing when limits breached"
    )

    # Time-based limits
    max_loss_streak_days: int = Field(
        default=5, ge=1, le=30, description="Maximum number of consecutive losing days"
    )
    cooling_off_period_days: int = Field(
        default=2, ge=0, le=30, description="Cooling off period after limit breach in days"
    )

    # Stress testing parameters
    stress_test_frequency: str = Field(
        default="monthly", description="Frequency for stress testing"
    )
    monte_carlo_simulations: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo simulations for stress testing",
    )
    var_confidence_level: float = Field(
        default=0.95, ge=0.90, le=0.99, description="Confidence level for VaR calculations"
    )

    # Limit enforcement
    hard_limits: bool = Field(
        default=True, description="Whether limits are hard (enforced) or soft (advisory)"
    )
    require_approval_above: float = Field(
        default=0.90,
        ge=0.1,
        le=1.0,
        description="Require approval when exceeding this percentage of limit",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize RiskLimitConfig with risk profile integration."""
        # Handle risk profile initialization
        if 'risk_profile' in data and isinstance(data['risk_profile'], str):
            # Convert string profile name to RiskProfile object
            profile_name = data['risk_profile'].lower()
            if profile_name == 'conservative':
                data['risk_profile'] = RiskProfile.conservative()
            elif profile_name == 'aggressive':
                data['risk_profile'] = RiskProfile.aggressive()
            else:  # moderate or any other value
                data['risk_profile'] = RiskProfile.moderate()

        super().__init__(**data)

    def get_profile_limits(self) -> dict[str, float]:
        """Get risk limits from the associated risk profile.

        Returns:
            Dictionary with risk limits from profile
        """
        return {
            'max_drawdown': self.risk_profile.max_drawdown,
            'max_leverage': self.risk_profile.max_leverage,
            'max_single_position': self.risk_profile.max_position_size,
            'max_sector_exposure': self.risk_profile.max_sector_exposure,
            'max_volatility': self.risk_profile.max_volatility,
        }

    def get_enforcement_thresholds(self) -> dict[str, float]:
        """Get enforcement thresholds for monitoring.

        Returns:
            Dictionary with enforcement thresholds
        """
        return {
            'alert_threshold': self.alert_threshold_pct,
            'emergency_halt_threshold': self.emergency_halt_threshold,
            'approval_threshold': self.require_approval_above,
        }

    def check_limit_status(self, current_value: float, limit_value: float) -> dict[str, Any]:
        """Check the status of a limit based on current value.

        Args:
            current_value: Current metric value
            limit_value: Limit threshold value

        Returns:
            Dictionary with limit status information
        """
        ratio = current_value / limit_value if limit_value > 0 else 0

        if ratio >= self.emergency_halt_threshold:
            status = 'emergency_halt'
            severity = LimitSeverity.CRITICAL
        elif ratio >= self.require_approval_above:
            status = 'requires_approval'
            severity = LimitSeverity.HIGH
        elif ratio >= self.alert_threshold_pct:
            status = 'alert'
            severity = LimitSeverity.MEDIUM
        else:
            status = 'normal'
            severity = LimitSeverity.LOW

        return {
            'status': status,
            'severity': severity,
            'ratio': ratio,
            'current_value': current_value,
            'limit_value': limit_value,
            'distance_to_limit': limit_value - current_value,
        }

    def is_hard_limit_breached(self, current_value: float, limit_value: float) -> bool:
        """Check if a hard limit is breached.

        Args:
            current_value: Current metric value
            limit_value: Limit threshold value

        Returns:
            True if hard limit is breached
        """
        if not self.hard_limits:
            return False

        return current_value > limit_value

    def get_breach_severity(self, current_value: float, limit_value: float) -> LimitSeverity:
        """Get the severity of a limit breach.

        Args:
            current_value: Current metric value
            limit_value: Limit threshold value

        Returns:
            Severity level of the breach
        """
        ratio = current_value / limit_value if limit_value > 0 else 0

        if ratio >= 1.5:
            return LimitSeverity.CRITICAL
        elif ratio >= 1.2:
            return LimitSeverity.HIGH
        elif ratio >= 1.0:
            return LimitSeverity.MEDIUM
        else:
            return LimitSeverity.LOW

    def requires_approval(self, current_value: float, limit_value: float) -> bool:
        """Check if approval is required for the current value.

        Args:
            current_value: Current metric value
            limit_value: Limit threshold value

        Returns:
            True if approval is required
        """
        ratio = current_value / limit_value if limit_value > 0 else 0
        return ratio >= self.require_approval_above

    def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk limits summary.

        Returns:
            Dictionary with risk limits summary
        """
        return {
            'core_limits': {
                'max_drawdown': self.max_drawdown,
                'max_leverage': self.max_leverage,
                'max_single_position': self.max_single_position,
                'max_portfolio_var': self.max_portfolio_var,
                'max_daily_loss': self.max_daily_loss,
            },
            'exposure_limits': {
                'max_sector_exposure': self.max_sector_exposure,
                'max_correlation': self.max_correlation,
                'max_country_exposure': self.max_country_exposure,
                'max_industry_exposure': self.max_industry_exposure,
                'concentration_limit': self.concentration_limit,
            },
            'enforcement': {
                'hard_limits': self.hard_limits,
                'alert_threshold_pct': self.alert_threshold_pct,
                'emergency_halt_threshold': self.emergency_halt_threshold,
                'require_approval_above': self.require_approval_above,
                'auto_rebalance_trigger': self.auto_rebalance_trigger,
            },
            'risk_profile': {
                'name': self.risk_profile.name,
                'type': self.risk_profile.profile_type.value,
                'max_volatility': self.risk_profile.max_volatility,
                'max_drawdown': self.risk_profile.max_drawdown,
                'target_sharpe_ratio': self.risk_profile.target_sharpe_ratio,
            },
        }

    def update_from_profile(self, profile: RiskProfile) -> None:
        """Update risk limits from a risk profile.

        Args:
            profile: Risk profile to apply
        """
        self.risk_profile = profile

        # Update core limits to match profile
        if profile.is_conservative():
            self.max_drawdown = min(self.max_drawdown, 0.15)
            self.max_leverage = min(self.max_leverage, 2.0)
            self.max_single_position = min(self.max_single_position, 0.08)
        elif profile.is_aggressive():
            self.max_drawdown = max(self.max_drawdown, 0.25)
            self.max_leverage = max(self.max_leverage, 3.0)
            self.max_single_position = max(self.max_single_position, 0.20)

    def create_profile_limits(self, profile_type: str) -> 'RiskLimitConfig':
        """Create a new RiskLimitConfig based on a profile type.

        Args:
            profile_type: Type of profile ('conservative', 'moderate', 'aggressive')

        Returns:
            New RiskLimitConfig instance
        """
        if profile_type.lower() == 'conservative':
            profile = RiskProfile.conservative()
        elif profile_type.lower() == 'aggressive':
            profile = RiskProfile.aggressive()
        else:
            profile = RiskProfile.moderate()

        # Create new config with profile-based limits
        config_data = {
            'risk_profile': profile,
            'max_drawdown': profile.max_drawdown,
            'max_leverage': profile.max_leverage,
            'max_single_position': profile.max_position_size,
            'max_sector_exposure': profile.max_sector_exposure,
            'max_volatility': profile.max_volatility,
        }

        return RiskLimitConfig(**config_data)
