"""Position Sizing Configuration Models.

This module provides Pydantic models for position sizing configuration.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SizingMethod(str, Enum):
    """Position sizing methods."""

    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY = "kelly"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_ADJUSTED = "correlation_adjusted"


class PositionSizingConfig(BaseModel):
    """Configuration for position sizing using pydantic BaseModel.

    This class defines all parameters needed to configure position sizing behavior,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core sizing parameters
    max_position_size: float = Field(
        default=0.10, ge=0.01, le=1.0, description="Maximum position size as fraction of portfolio"
    )
    min_position_size: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Minimum position size as fraction of portfolio"
    )
    risk_per_trade: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Risk per trade as fraction of portfolio"
    )
    sizing_method: SizingMethod = Field(
        default=SizingMethod.FIXED_PERCENTAGE, description="Method for position sizing"
    )

    # Advanced sizing parameters
    max_daily_trades: int = Field(
        default=5, ge=1, le=100, description="Maximum number of trades per day"
    )
    max_sector_exposure: float = Field(
        default=0.30, ge=0.05, le=1.0, description="Maximum sector exposure"
    )
    max_correlation: float = Field(
        default=0.80, ge=0.1, le=1.0, description="Maximum correlation for position sizing"
    )
    volatility_adjustment: bool = Field(
        default=True, description="Whether to adjust for volatility"
    )

    # Confidence and conviction parameters
    conviction_factors: dict[str, float] = Field(
        default_factory=lambda: {'low': 0.7, 'medium': 1.0, 'high': 1.3},
        description="Conviction adjustment factors",
    )

    # Kelly Criterion specific parameters (if using Kelly method)
    kelly_win_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Historical win rate for Kelly calculation"
    )
    kelly_avg_win: float | None = Field(
        default=None, ge=0.0, description="Average win amount for Kelly calculation"
    )
    kelly_avg_loss: float | None = Field(
        default=None, ge=0.0, description="Average loss amount for Kelly calculation"
    )

    @property
    def requires_historical_data(self) -> bool:
        """Check if this sizing method requires historical data."""
        return self.sizing_method in [SizingMethod.KELLY, SizingMethod.VOLATILITY_ADJUSTED]

    @property
    def requires_correlation_data(self) -> bool:
        """Check if this sizing method requires correlation data."""
        return self.sizing_method == SizingMethod.CORRELATION_ADJUSTED
