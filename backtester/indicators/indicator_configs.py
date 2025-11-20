"""Indicator configuration system for the backtester.

This module provides standardized configuration classes for all technical indicators,
following the established pydantic patterns from the core configuration system.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_INDICATOR_TYPE_HINTS: dict[str, str] = {
    'sma': 'trend',
    'ema': 'trend',
    'wma': 'trend',
    'rsi': 'momentum',
    'macd': 'momentum',
    'stochastic': 'momentum',
    'williamsr': 'momentum',
    'williams_r': 'momentum',
    'cci': 'momentum',
    'obv': 'volume',
    'bollinger': 'volatility',
    'bollinger_bands': 'volatility',
    'atr': 'volatility',
}


class IndicatorConfig(BaseModel):
    """Configuration for indicator parameters following established patterns.

    This class defines all the parameters needed to configure any technical indicator,
    with validation and defaults that ensure proper operation across all indicator types.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            'component': 'indicator',
            'yaml_example': {
                '__config_class__': 'IndicatorConfig',
                'indicator_name': 'rsi',
                'factory_name': 'rsi',
                'indicator_type': 'momentum',
                'period': 14,
                'overbought_threshold': 70,
                'oversold_threshold': 30,
            },
        },
    )

    # Core indicator settings
    indicator_name: str = Field(description="Name of the indicator")
    indicator_type: str = Field(default="trend", description="Type/category of indicator")
    factory_name: str | None = Field(
        default=None,
        description="Optional IndicatorFactory registration name",
    )
    period: int = Field(default=14, description="Lookback period for calculations")

    # Moving Average Indicators
    short_period: int = Field(default=5, description="Short period for dual MA indicators")
    long_period: int = Field(default=20, description="Long period for dual MA indicators")
    ma_type: str = Field(default="simple", description="MA type: simple, exponential, weighted")
    price_column: str = Field(default="close", description="Price column to use for calculations")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    # Oscillator Indicators
    overbought_threshold: float = Field(default=70.0, description="Overbought level")
    oversold_threshold: float = Field(default=30.0, description="Oversold level")

    # Volatility Indicators
    standard_deviations: float = Field(default=2.0, description="Standard deviation multiplier")
    atr_multiplier: float = Field(default=2.0, description="ATR multiplier for bands")

    # MACD specific parameters
    fast_period: int = Field(default=12, description="Fast period for MACD")
    slow_period: int = Field(default=26, description="Slow period for MACD")
    signal_period: int = Field(default=9, description="Signal line period for MACD")

    # Stochastic specific parameters
    k_period: int = Field(default=14, description="%K period for stochastic oscillator")
    d_period: int = Field(default=3, description="%D period for stochastic oscillator")

    # Williams %R specific parameters
    williams_r_period: int = Field(default=14, description="Period for Williams %R")

    # CCI specific parameters
    cci_period: int = Field(default=20, description="Period for CCI")
    cci_constant: float = Field(default=0.015, description="Constant multiplier for CCI")

    # Signal Generation
    signal_sensitivity: float = Field(default=1.0, description="Signal threshold sensitivity")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence for signals")

    # Performance and Behavior
    cache_calculations: bool = Field(default=True, description="Cache intermediate calculations")
    calculate_realtime: bool = Field(default=False, description="Calculate in real-time")

    # Volume specific parameters
    volume_column: str = Field(default="volume", description="Volume column name")

    @field_validator('indicator_name')
    @classmethod
    def validate_indicator_name(cls, v: str) -> str:
        """Ensure indicator name is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("indicator_name must be a non-empty string")
        return v.strip()

    @field_validator('indicator_type', mode='before')
    @classmethod
    def validate_indicator_type(cls, v: str) -> str:
        """Validate indicator type is one of the allowed values."""
        valid_types = ['trend', 'momentum', 'volume', 'volatility', 'oscillator']
        value = v.lower().strip()
        if value not in valid_types:
            raise ValueError(f"indicator_type must be one of {valid_types}")
        return value

    @field_validator('ma_type')
    @classmethod
    def validate_ma_type(cls, v: str) -> str:
        """Validate moving average type is supported."""
        valid_types = ['simple', 'exponential', 'weighted']
        if v not in valid_types:
            raise ValueError(f"ma_type must be one of {valid_types}")
        return v

    @field_validator(
        'period',
        'short_period',
        'long_period',
        'fast_period',
        'slow_period',
        'signal_period',
        'k_period',
        'd_period',
        'williams_r_period',
        'cci_period',
    )
    @classmethod
    def validate_positive_periods(cls, v: int) -> int:
        """Validate that all period parameters are positive."""
        if v <= 0:
            raise ValueError("Period must be positive")
        return v

    @field_validator(
        'overbought_threshold',
        'oversold_threshold',
        'standard_deviations',
        'atr_multiplier',
        'signal_sensitivity',
        'confidence_threshold',
        'cci_constant',
    )
    @classmethod
    def validate_float_ranges(cls, v: float) -> float:
        """Validate float parameters are within reasonable ranges."""
        if (
            'threshold' in str(v) or 'sensitivity' in str(v) or 'confidence' in str(v)
        ) and not 0.0 <= v <= 1.0:
            raise ValueError(f"Value {v} must be between 0.0 and 1.0")
        return v

    @model_validator(mode='after')
    def _normalise_indicator(self) -> 'IndicatorConfig':
        """Normalise common indicator metadata for downstream loaders."""
        normalized_name = self.indicator_name.strip()
        self.indicator_name = normalized_name
        normalized_lower = normalized_name.lower()

        for key, hint in _INDICATOR_TYPE_HINTS.items():
            if (
                normalized_lower == key
                or normalized_lower.startswith(f"{key}_")
                or key in normalized_lower
            ):
                self.indicator_type = hint
                break

        if self.factory_name is None and normalized_name:
            self.factory_name = normalized_lower

        if normalized_lower.startswith('ema'):
            self.ma_type = 'exponential'
        elif normalized_lower.startswith('sma'):
            self.ma_type = 'simple'

        return self

    def get_indicator_columns(self) -> list[str]:
        """Get list of columns that this indicator will add to the DataFrame.

        Returns:
            List of column names that will be added
        """
        base_name = self.indicator_name.lower()

        if self.indicator_type == "trend":
            return self._get_trend_columns(base_name)
        elif self.indicator_type == "momentum":
            return self._get_momentum_columns(base_name)
        elif self.indicator_type == "volatility":
            return self._get_volatility_columns(base_name)
        elif self.indicator_type == "volume":
            return self._get_volume_columns(base_name)

        # Default fallback
        return [f"{base_name}_value"]

    def _get_trend_columns(self, base_name: str) -> list[str]:
        """Get column names for trend indicators."""
        if self.ma_type == "simple":
            return [f"{base_name}_sma"]
        elif self.ma_type == "exponential":
            return [f"{base_name}_ema"]
        else:
            return [f"{base_name}_wma"]

    def _get_momentum_columns(self, base_name: str) -> list[str]:
        """Get column names for momentum indicators."""
        if "rsi" in base_name:
            return [f"{base_name}_rsi"]
        elif "macd" in base_name:
            return [f"{base_name}_macd", f"{base_name}_signal", f"{base_name}_histogram"]
        elif "stochastic" in base_name:
            return [f"{base_name}_k", f"{base_name}_d"]
        elif "williams" in base_name:
            return [f"{base_name}_wr"]
        elif "cci" in base_name:
            return [f"{base_name}_cci"]
        return [f"{base_name}_value"]

    def _get_volatility_columns(self, base_name: str) -> list[str]:
        """Get column names for volatility indicators."""
        if "bollinger" in base_name:
            return [f"{base_name}_upper", f"{base_name}_middle", f"{base_name}_lower"]
        elif "atr" in base_name:
            return [f"{base_name}_atr"]
        return [f"{base_name}_value"]

    def _get_volume_columns(self, base_name: str) -> list[str]:
        """Get column names for volume indicators."""
        if "obv" in base_name:
            return [f"{base_name}_obv"]
        return [f"{base_name}_value"]

    def validate_for_indicator(self) -> None:
        """Validate configuration is appropriate for the specified indicator type.

        Raises:
            ValueError: If configuration is invalid for the indicator type
        """
        # Cross-validate parameters based on indicator type
        if self.indicator_type == "trend":
            self._validate_trend_indicators()
        elif self.indicator_type == "momentum":
            self._validate_momentum_indicators()
        elif self.indicator_type == "volatility":
            self._validate_volatility_indicators()

        # Additional validations can be added here for other indicator types

    def _validate_trend_indicators(self) -> None:
        """Validate trend indicator configurations."""
        if self.ma_type in ["exponential", "weighted"] and self.period <= 1:
            raise ValueError(f"For {self.ma_type} moving average, period must be > 1")

    def _validate_momentum_indicators(self) -> None:
        """Validate momentum indicator configurations."""
        if "macd" in self.indicator_name.lower():
            if self.fast_period >= self.slow_period:
                raise ValueError("Fast period must be less than slow period for MACD")
            if self.signal_period <= 0:
                raise ValueError("Signal period must be positive for MACD")

        elif "rsi" in self.indicator_name.lower():
            if self.overbought_threshold <= self.oversold_threshold:
                raise ValueError("Overbought threshold must be greater than oversold threshold")

    def _validate_volatility_indicators(self) -> None:
        """Validate volatility indicator configurations."""
        if "bollinger" in self.indicator_name.lower() and self.standard_deviations <= 0:
            raise ValueError("Standard deviations must be positive for Bollinger Bands")
