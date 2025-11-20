"""Take Profit Configuration Models.

This module provides Pydantic models for take profit configuration.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TakeProfitType(str, Enum):
    """Types of take profit mechanisms."""

    FIXED = "FIXED"
    PERCENTAGE = "PERCENTAGE"
    PRICE = "PRICE"
    TRAILING = "TRAILING"
    TRAILING_PERCENTAGE = "TRAILING_PERCENTAGE"


class TakeProfitConfig(BaseModel):
    """Configuration for take profit mechanisms using pydantic BaseModel.

    This class defines all parameters needed to configure take profit behavior,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core take profit settings
    take_profit_type: TakeProfitType = Field(
        default=TakeProfitType.PERCENTAGE, description="Type of take profit mechanism"
    )
    take_profit_value: float = Field(
        default=0.06,
        ge=0.001,
        le=5.0,
        description="Take profit value (percentage, price, or fixed amount)",
    )

    # Trailing profit settings
    trail_distance: float = Field(
        default=0.02, ge=0.001, le=0.5, description="Trailing distance for trailing take profit"
    )
    trail_step: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Minimum step for trailing profit adjustment"
    )

    # Advanced settings
    max_gain_value: float | None = Field(
        default=None, ge=0.0, description="Maximum absolute gain target"
    )
    activation_price: float | None = Field(
        default=None, ge=0.0, description="Price at which take profit activates"
    )
    trailing_take_profit_pct: float = Field(
        default=0.03, ge=0.01, le=0.2, description="Trailing take profit percentage"
    )
    fixed_take_profit_price: float | None = Field(
        default=None, ge=0.0, description="Fixed take profit price"
    )

    @property
    def is_trailing(self) -> bool:
        """Check if this is a trailing take profit configuration."""
        return self.take_profit_type in [
            TakeProfitType.TRAILING,
            TakeProfitType.TRAILING_PERCENTAGE,
        ]
