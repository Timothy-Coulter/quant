"""Base indicator class and factory pattern for the indicator system.

This module provides the abstract base class that all technical indicators must implement,
along with a factory pattern for creating indicator instances. It follows the established
modular component architecture with proper typing and logging integration.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
from pydantic import ValidationError

from backtester.core.logger import BacktesterLogger
from backtester.signal.signal_types import SignalGenerator, SignalType

from .indicator_configs import IndicatorConfig

T = TypeVar('T')


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators.

    This class provides the common interface and functionality that all technical
    indicators must implement. It follows the modular component architecture
    with proper typing, logging, and validation integration.
    """

    def __init__(self, config: IndicatorConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the indicator with configuration.

        Args:
            config: Indicator configuration parameters
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or BacktesterLogger.get_logger(__name__)
        self.name = config.indicator_name
        self.type = config.indicator_type
        self._cache: dict[str, Any] = {}
        self._is_initialized = False

        # Validate configuration
        self._validate_configuration()

        self.logger.debug(f"Initialized indicator: {self.name} (type: {self.type})")

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the default configuration for the indicator implementation."""
        raise NotImplementedError(f"{cls.__name__} must define default_config()")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values.

        Args:
            data: DataFrame with OHLCV data (datetime indexed)
                Required columns: 'open', 'high', 'low', 'close', 'volume'

        Returns:
            DataFrame with indicator values added as columns
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on indicator values.

        Args:
            data: DataFrame with market data and calculated indicators

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, raises exception otherwise
        """
        self._validate_data_structure(data)
        self._validate_ohlc_relationships(data)
        self._validate_data_sufficiency(data)
        return True

    def _validate_data_structure(self, data: pd.DataFrame) -> None:
        """Validate basic data structure and required columns."""
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must be datetime indexed")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> None:
        """Validate OHLC price relationships."""
        # Only validate OHLC relationships for non-NaN rows
        valid_rows = ~(
            data['high'].isna() | data['low'].isna() | data['open'].isna() | data['close'].isna()
        )

        if valid_rows.any():
            if not (data.loc[valid_rows, 'high'] >= data.loc[valid_rows, 'low']).all():
                raise ValueError("High prices must be >= low prices")
            if not (data.loc[valid_rows, 'high'] >= data.loc[valid_rows, 'open']).all():
                raise ValueError("High prices must be >= open prices")
            if not (data.loc[valid_rows, 'high'] >= data.loc[valid_rows, 'close']).all():
                raise ValueError("High prices must be >= close prices")
            if not (data.loc[valid_rows, 'low'] <= data.loc[valid_rows, 'open']).all():
                raise ValueError("Low prices must be <= open prices")
            if not (data.loc[valid_rows, 'low'] <= data.loc[valid_rows, 'close']).all():
                raise ValueError("Low prices must be <= close prices")

    def _validate_data_sufficiency(self, data: pd.DataFrame) -> None:
        """Validate that sufficient data is available."""
        if len(data) < self.config.period:
            raise ValueError(f"Insufficient data: need at least {self.config.period} periods")

    def get_required_columns(self) -> list[str]:
        """Get list of required data columns.

        Returns:
            List of required column names
        """
        return ['open', 'high', 'low', 'close', 'volume']

    def reset(self) -> None:
        """Reset indicator state for reuse."""
        self._cache.clear()
        self._is_initialized = False
        self.logger.debug(f"Indicator {self.name} reset")

    def _validate_configuration(self) -> None:
        """Validate indicator-specific configuration parameters."""
        try:
            # Validate required fields
            if not self.config.indicator_name:
                raise ValueError("indicator_name is required")
            if not self.config.indicator_type:
                raise ValueError("indicator_type is required")
            if self.config.period <= 0:
                raise ValueError("period must be positive")

            # Run type-specific validation
            self.config.validate_for_indicator()

        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    def get_indicator_info(self) -> dict[str, Any]:
        """Get indicator information and current configuration.

        Returns:
            Dictionary with indicator information
        """
        return {
            "name": self.name,
            "type": self.type,
            "period": self.config.period,
            "is_initialized": self._is_initialized,
            "config": self.config.model_dump(),
        }

    def _get_cached_result(self, cache_key: str) -> Any | None:
        """Get cached result if available and caching is enabled.

        Args:
            cache_key: Key for the cached result

        Returns:
            Cached result or None if not found
        """
        if self.config.cache_calculations and cache_key in self._cache:
            return self._cache[cache_key]
        return None

    def _set_cached_result(self, cache_key: str, result: Any) -> None:
        """Set cached result if caching is enabled.

        Args:
            cache_key: Key for the cached result
            result: Result to cache
        """
        if self.config.cache_calculations:
            self._cache[cache_key] = result

    def _create_standard_signal(
        self,
        signal_type: SignalType,
        action: str,
        confidence: float,
        timestamp: Any,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Create a standardized trading signal.

        Args:
            signal_type: Type of signal
            action: Action description
            confidence: Confidence level
            timestamp: Signal timestamp
            **metadata: Additional metadata

        Returns:
            Standardized signal dictionary
        """
        # Ensure all required metadata is present
        full_metadata = {'indicator_name': self.name, 'indicator_type': self.type, **metadata}

        return SignalGenerator.create_signal(
            signal_type=signal_type,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            metadata=full_metadata,
        )


class IndicatorFactory:
    """Factory for creating indicator instances.

    This factory pattern allows for easy registration and creation of indicator
    instances by name, following the established architectural patterns.
    """

    _indicators: dict[str, type[BaseIndicator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseIndicator]], type[BaseIndicator]]:
        """Register an indicator class with the factory.

        Args:
            name: Name to register the indicator under

        Returns:
            Decorator function for registering the indicator class
        """

        def decorator(indicator_class: type[BaseIndicator]) -> type[BaseIndicator]:
            cls._indicators[name.lower()] = indicator_class
            return indicator_class

        return decorator

    @classmethod
    def _get_indicator_class(cls, name: str) -> type[BaseIndicator]:
        slug = name.lower()
        indicator_class = cls._indicators.get(slug)
        if indicator_class is None:
            available = list(cls._indicators.keys())
            raise ValueError(f"Unknown indicator: {slug}. Available indicators: {available}")
        return indicator_class

    @classmethod
    def default_config(cls, name: str) -> IndicatorConfig:
        """Return the registered indicator's default configuration."""
        indicator_class = cls._get_indicator_class(name)
        return indicator_class.default_config()

    @classmethod
    def create(cls, name: str, config: IndicatorConfig | None = None) -> BaseIndicator:
        """Create an indicator instance by name.

        Args:
            name: Name of the indicator to create
            config: Configuration for the indicator. When omitted the registered
                indicator default will be used.

        Returns:
            Indicator instance

        Raises:
            ValueError: If the indicator name is not registered
        """
        indicator_class = cls._get_indicator_class(name)
        resolved_config = config or indicator_class.default_config()
        if resolved_config.factory_name is None:
            resolved_config.factory_name = name.lower()
        return indicator_class(resolved_config)

    @classmethod
    def get_available_indicators(cls) -> list[str]:
        """Get list of available indicator names.

        Returns:
            List of available indicator names
        """
        return list(cls._indicators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an indicator is registered.

        Args:
            name: Name of the indicator to check

        Returns:
            True if the indicator is registered
        """
        return name.lower() in cls._indicators
