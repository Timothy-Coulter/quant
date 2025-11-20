"""Stop Loss Configuration Models.

This module provides Pydantic models for stop loss configuration.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StopLossType(str, Enum):
    """Types of stop loss mechanisms."""

    FIXED = "FIXED"
    PERCENTAGE = "PERCENTAGE"
    PRICE = "PRICE"
    TRAILING = "TRAILING"
    TRAILING_PERCENTAGE = "TRAILING_PERCENTAGE"


class StopLossConfig(BaseModel):
    """Configuration for stop loss mechanisms using pydantic BaseModel.

    This class defines all parameters needed to configure stop loss behavior,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core stop loss settings
    stop_loss_type: StopLossType = Field(
        default=StopLossType.PERCENTAGE, description="Type of stop loss mechanism"
    )
    stop_loss_value: float = Field(
        default=0.02,
        ge=0.001,
        le=1.0,
        description="Stop loss value (percentage, price, or fixed amount)",
    )

    # Trailing stop settings
    trail_distance: float = Field(
        default=0.01, ge=0.001, le=0.5, description="Trailing distance for trailing stop loss"
    )
    trail_step: float = Field(
        default=0.005, ge=0.001, le=0.1, description="Minimum step for trailing stop adjustment"
    )

    # Advanced settings
    max_loss_value: float | None = Field(
        default=None, ge=0.0, description="Maximum absolute loss allowed"
    )
    activation_price: float | None = Field(
        default=None, ge=0.0, description="Price at which stop loss activates"
    )
    trailing_stop_pct: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Trailing stop percentage"
    )

    @property
    def is_trailing(self) -> bool:
        """Check if this is a trailing stop loss configuration."""
        return self.stop_loss_type in [StopLossType.TRAILING, StopLossType.TRAILING_PERCENTAGE]
