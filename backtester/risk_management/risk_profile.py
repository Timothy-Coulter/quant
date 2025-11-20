"""Risk Profile Configuration System.

This module provides risk profile definitions and suitability checking for different
risk tolerance levels (conservative, moderate, aggressive).
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskProfileType(str, Enum):
    """Risk profile types."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskProfile(BaseModel):
    """Risk profile configuration using pydantic BaseModel.

    This class defines risk profiles with different risk tolerance levels,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Profile identification
    name: str = Field(description="Risk profile name")
    profile_type: RiskProfileType = Field(description="Risk profile type enum")
    description: str = Field(default="", description="Description of the risk profile")

    # Core risk limits
    max_volatility: float = Field(ge=0.01, le=1.0, description="Maximum portfolio volatility")
    max_drawdown: float = Field(ge=0.01, le=1.0, description="Maximum portfolio drawdown")
    target_sharpe_ratio: float = Field(
        default=0.8, ge=0.0, le=5.0, description="Target Sharpe ratio"
    )

    # Position limits
    max_position_size: float = Field(ge=0.01, le=1.0, description="Maximum single position size")
    max_sector_exposure: float = Field(ge=0.05, le=1.0, description="Maximum sector exposure")
    max_leverage: float = Field(ge=1.0, le=20.0, description="Maximum leverage ratio")

    # Risk management preferences
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    diversification_preference: str = Field(
        default="medium", description="Diversification preference level"
    )
    rebalance_frequency: str = Field(
        default="monthly", description="Preferred rebalancing frequency"
    )

    # Performance targets
    target_return: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Target annual return"
    )
    min_return: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum acceptable return"
    )

    def __init__(self, name: str, **data: Any) -> None:
        """Initialize risk profile with predefined settings if not all parameters provided."""
        # If only name is provided, use predefined settings
        if name.lower() in ['conservative', 'moderate', 'aggressive'] and len(data) == 0:
            data = self._get_predefined_profile(name.lower())
            data['name'] = name
            # Set the correct enum value instead of string
            profile_type_map = {
                'conservative': RiskProfileType.CONSERVATIVE,
                'moderate': RiskProfileType.MODERATE,
                'aggressive': RiskProfileType.AGGRESSIVE,
            }
            data['profile_type'] = profile_type_map[name.lower()]
            # Don't pass name as both argument and in data
            super().__init__(**data)
        else:
            super().__init__(name=name, **data)

    @classmethod
    def conservative(cls) -> 'RiskProfile':
        """Create a conservative risk profile."""
        return cls(
            name="Conservative",
            profile_type=RiskProfileType.CONSERVATIVE,
            description="Low risk tolerance with focus on capital preservation",
            max_volatility=0.15,
            max_drawdown=0.10,
            target_sharpe_ratio=0.5,
            max_position_size=0.05,
            max_sector_exposure=0.20,
            max_leverage=1.5,
            risk_tolerance="low",
            diversification_preference="high",
            rebalance_frequency="weekly",
            target_return=0.05,
            min_return=0.02,
        )

    @classmethod
    def moderate(cls) -> 'RiskProfile':
        """Create a moderate risk profile."""
        return cls(
            name="Moderate",
            profile_type=RiskProfileType.MODERATE,
            description="Balanced risk and return approach",
            max_volatility=0.25,
            max_drawdown=0.15,
            target_sharpe_ratio=0.8,
            max_position_size=0.10,
            max_sector_exposure=0.30,
            max_leverage=2.0,
            risk_tolerance="medium",
            diversification_preference="medium",
            rebalance_frequency="monthly",
            target_return=0.08,
            min_return=0.04,
        )

    @classmethod
    def aggressive(cls) -> 'RiskProfile':
        """Create an aggressive risk profile."""
        return cls(
            name="Aggressive",
            profile_type=RiskProfileType.AGGRESSIVE,
            description="High risk tolerance for maximum growth potential",
            max_volatility=0.40,
            max_drawdown=0.25,
            target_sharpe_ratio=1.2,
            max_position_size=0.20,
            max_sector_exposure=0.40,
            max_leverage=3.0,
            risk_tolerance="high",
            diversification_preference="low",
            rebalance_frequency="quarterly",
            target_return=0.12,
            min_return=0.06,
        )

    def _get_predefined_profile(self, profile_name: str) -> dict[str, Any]:
        """Get predefined profile settings.

        Args:
            profile_name: Name of the predefined profile

        Returns:
            Dictionary with profile settings
        """
        profiles = {
            'conservative': {
                'description': 'Low risk tolerance with focus on capital preservation',
                'max_volatility': 0.15,
                'max_drawdown': 0.10,
                'target_sharpe_ratio': 0.5,
                'max_position_size': 0.05,
                'max_sector_exposure': 0.20,
                'max_leverage': 1.5,
                'risk_tolerance': 'low',
                'diversification_preference': 'high',
                'rebalance_frequency': 'weekly',
                'target_return': 0.05,
                'min_return': 0.02,
            },
            'moderate': {
                'description': 'Balanced risk and return approach',
                'max_volatility': 0.25,
                'max_drawdown': 0.15,
                'target_sharpe_ratio': 0.8,
                'max_position_size': 0.10,
                'max_sector_exposure': 0.30,
                'max_leverage': 2.0,
                'risk_tolerance': 'medium',
                'diversification_preference': 'medium',
                'rebalance_frequency': 'monthly',
                'target_return': 0.08,
                'min_return': 0.04,
            },
            'aggressive': {
                'description': 'High risk tolerance for maximum growth potential',
                'max_volatility': 0.40,
                'max_drawdown': 0.25,
                'target_sharpe_ratio': 1.2,
                'max_position_size': 0.20,
                'max_sector_exposure': 0.40,
                'max_leverage': 3.0,
                'risk_tolerance': 'high',
                'diversification_preference': 'low',
                'rebalance_frequency': 'quarterly',
                'target_return': 0.12,
                'min_return': 0.06,
            },
        }

        return profiles.get(profile_name, profiles['moderate'])

    def check_suitability(
        self,
        portfolio_volatility: float,
        portfolio_max_drawdown: float,
        portfolio_sharpe: float,
    ) -> dict[str, Any]:
        """Check if portfolio is suitable for this risk profile.

        Args:
            portfolio_volatility: Current portfolio volatility
            portfolio_max_drawdown: Current portfolio maximum drawdown
            portfolio_sharpe: Current portfolio Sharpe ratio

        Returns:
            Dictionary with suitability analysis
        """
        volatility_ok = portfolio_volatility <= self.max_volatility
        drawdown_ok = abs(portfolio_max_drawdown) <= self.max_drawdown
        sharpe_ok = portfolio_sharpe >= self.target_sharpe_ratio

        suitable = volatility_ok and drawdown_ok and sharpe_ok

        # Calculate risk scores (0.0 to 1.0, where 1.0 is at or below limits)
        volatility_score = min(1.0, self.max_volatility / max(0.01, portfolio_volatility))
        drawdown_score = min(1.0, self.max_drawdown / max(0.01, abs(portfolio_max_drawdown)))
        sharpe_score = min(1.0, portfolio_sharpe / max(0.1, self.target_sharpe_ratio))

        # Overall suitability score
        suitability_score = (volatility_score + drawdown_score + sharpe_score) / 3

        return {
            'suitable': suitable,
            'suitability_score': suitability_score,
            'risk_scores': {
                'volatility': volatility_score,
                'drawdown': drawdown_score,
                'sharpe': sharpe_score,
            },
            'profile_name': self.name,
            'recommendations': self._generate_recommendations(
                volatility_ok, drawdown_ok, sharpe_ok
            ),
            'current_metrics': {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_max_drawdown': portfolio_max_drawdown,
                'portfolio_sharpe': portfolio_sharpe,
            },
            'profile_limits': {
                'max_volatility': self.max_volatility,
                'max_drawdown': self.max_drawdown,
                'target_sharpe_ratio': self.target_sharpe_ratio,
            },
        }

    def _generate_recommendations(
        self, volatility_ok: bool, drawdown_ok: bool, sharpe_ok: bool
    ) -> list[str]:
        """Generate recommendations based on profile suitability.

        Args:
            volatility_ok: Whether volatility is within limits
            drawdown_ok: Whether drawdown is within limits
            sharpe_ok: Whether Sharpe ratio meets target

        Returns:
            List of recommendations
        """
        recommendations = []

        if not volatility_ok:
            recommendations.append("Reduce position sizes or volatility to meet profile limits")
        if not drawdown_ok:
            recommendations.append("Implement tighter stop losses to limit drawdowns")
        if not sharpe_ok:
            recommendations.append(
                "Improve risk-adjusted returns through better strategy selection"
            )

        if volatility_ok and drawdown_ok and sharpe_ok:
            recommendations.append("Portfolio well-suited for this risk profile")

        return recommendations

    def get_risk_limits(self) -> dict[str, float]:
        """Get risk limits for this profile.

        Returns:
            Dictionary with risk limits
        """
        return {
            'max_volatility': self.max_volatility,
            'max_drawdown': self.max_drawdown,
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_leverage': self.max_leverage,
            'target_sharpe_ratio': self.target_sharpe_ratio,
        }

    def is_conservative(self) -> bool:
        """Check if this is a conservative profile.

        Returns:
            True if conservative profile
        """
        return self.profile_type == RiskProfileType.CONSERVATIVE

    def is_moderate(self) -> bool:
        """Check if this is a moderate profile.

        Returns:
            True if moderate profile
        """
        return self.profile_type == RiskProfileType.MODERATE

    def is_aggressive(self) -> bool:
        """Check if this is an aggressive profile.

        Returns:
            True if aggressive profile
        """
        return self.profile_type == RiskProfileType.AGGRESSIVE

    def get_profile_summary(self) -> dict[str, Any]:
        """Get comprehensive profile summary.

        Returns:
            Dictionary with profile summary
        """
        # Handle both enum and string profile types
        profile_type_value = (
            self.profile_type.value if hasattr(self.profile_type, 'value') else self.profile_type
        )

        return {
            'name': self.name,
            'type': profile_type_value,
            'description': self.description,
            'risk_limits': self.get_risk_limits(),
            'preferences': {
                'risk_tolerance': self.risk_tolerance,
                'diversification_preference': self.diversification_preference,
                'rebalance_frequency': self.rebalance_frequency,
            },
            'targets': {
                'target_return': self.target_return,
                'min_return': self.min_return,
                'target_sharpe_ratio': self.target_sharpe_ratio,
            },
        }
