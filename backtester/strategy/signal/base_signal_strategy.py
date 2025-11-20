"""Base signal strategy class for implementing signal generation strategies.

This module defines the abstract base class that all signal generation strategies
must inherit from. It provides the common interface and functionality required
for generating trading signals using technical analysis, machine learning models,
and various signal generation approaches.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from backtester.core.event_bus import EventBus, EventFilter
from backtester.core.events import MarketDataEvent, create_signal_event
from backtester.core.logger import get_backtester_logger
from backtester.indicators.base_indicator import BaseIndicator, IndicatorFactory
from backtester.indicators.config_loader import IndicatorConfigResolver
from backtester.model.base_model import BaseModel
from backtester.model.model_configs import ModelConfig
from backtester.signal.signal_types import SignalGenerator

from .signal_strategy_config import SignalStrategyConfig


class StrategyTypeLabel(str):
    """Case-insensitive string wrapper for strategy type labels."""

    def __new__(cls, value: str) -> 'StrategyTypeLabel':
        """Create a normalized label while preserving the original casing."""
        normalized = value.upper()
        obj = super().__new__(cls, normalized)
        obj._original = value
        return obj

    def __eq__(self, other: object) -> bool:
        """Compare labels case-insensitively while supporting raw string values."""
        if isinstance(other, str):
            return self.raw().lower() == other.lower()
        return super().__eq__(other)

    def lower(self) -> str:
        """Return the lowercase representation of the original label."""
        return self.raw().lower()

    def raw(self) -> str:
        """Return the original case-sensitive value."""
        return getattr(self, '_original', str(self))


class BaseSignalStrategy(ABC):
    """Abstract base class for signal generation strategies.

    This class provides the common interface and functionality that all signal
    generation strategies must implement. It follows the modular component
    architecture with proper typing, logging, and validation integration.
    """

    def __init__(self, config: SignalStrategyConfig, event_bus: EventBus) -> None:
        """Initialize the signal strategy.

        Args:
            config: Signal strategy configuration parameters
            event_bus: Event bus for event-driven communication
        """
        self.config = config
        self.event_bus = event_bus
        self.logger = get_backtester_logger(__name__)

        # Strategy state
        self.name = config.strategy_name
        self.type = config.strategy_type
        self.is_initialized = False
        self.current_step = 0
        self.signals: list[dict[str, Any]] = []
        self.signal_history: list[dict[str, Any]] = []  # For backward compatibility
        self.indicators: dict[str, BaseIndicator] = {}
        self.models: dict[str, BaseModel] = {}
        self._market_data_subscription_id: str | None = None

        # Performance tracking
        self.signal_count = 0
        self.valid_signal_count = 0
        self.invalid_signal_count = 0
        self.last_signal_time: float | None = None

        # Subscribe to market data events
        self._setup_event_subscriptions()

        # Initialize indicators and models
        self._initialize_components()

        self.is_initialized = True
        self.logger.info(f"Initialized signal strategy: {self.name} (type: {self.type})")

    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for the strategy."""
        symbol_filters: set[str] = set()
        for symbol in self.config.symbols:
            if symbol:
                symbol_filters.add(str(symbol).upper())

        metadata_filters: dict[str, Any] | None = None
        if symbol_filters and "*" not in symbol_filters and "ALL" not in symbol_filters:
            metadata_filters = {'symbols': symbol_filters}

        market_data_filter = EventFilter(
            event_types={'MARKET_DATA'},
            metadata_filters=metadata_filters,
        )

        self._market_data_subscription_id = self.event_bus.subscribe(
            self._handle_market_data_event, market_data_filter
        )
        self.logger.debug(
            "Subscribed to market data events with ID: %s", self._market_data_subscription_id
        )

    def _initialize_components(self) -> None:
        """Initialize indicators and models based on configuration."""
        resolver = IndicatorConfigResolver()
        for indicator_definition in self.config.indicators:
            try:
                config = resolver.resolve(indicator_definition)
                indicator_name = config.indicator_name
                factory_name = (config.factory_name or indicator_name).lower()
                indicator = IndicatorFactory.create(factory_name, config)
                self.indicators[indicator_name] = indicator
                self.logger.debug(
                    "Initialized indicator: %s (factory name: %s)",
                    indicator_name,
                    factory_name,
                )

            except Exception as e:
                fallback_name = getattr(indicator_definition, 'indicator_name', 'unknown')
                self.logger.error(f"Failed to initialize indicator {fallback_name}: {e}")

        # Initialize models (only if models attribute exists in config)
        if hasattr(self.config, 'models'):
            for model_config in self.config.models:
                try:
                    if hasattr(model_config, 'model_name'):
                        model_name = model_config.model_name
                        config = model_config
                    else:
                        model_name = model_config.get('model_name')
                        if not model_name:
                            self.logger.warning("Skipping model with no name")
                            continue
                        config = ModelConfig(**model_config)

                    self.models[model_name] = config
                    self.logger.debug(f"Initialized model: {model_name}")

                except Exception as e:
                    fallback_name = getattr(model_config, 'model_name', 'unknown')
                    self.logger.error(f"Failed to initialize model {fallback_name}: {e}")

    def _handle_market_data_event(self, event: MarketDataEvent) -> None:
        """Handle incoming market data events.

        Args:
            event: Market data event containing price and volume information
        """
        try:
            # Extract market data
            symbol = event.symbol
            timestamp = event.timestamp
            metadata_frame = event.metadata.get("data_frame")
            if isinstance(metadata_frame, pd.DataFrame) and not metadata_frame.empty:
                df = metadata_frame.copy()
            else:
                data = {
                    'open': event.open_price,
                    'high': event.high_price,
                    'low': event.low_price,
                    'close': event.close_price,
                    'volume': event.volume or 0,
                    'timestamp': timestamp,
                }
                df = pd.DataFrame([data]).set_index('timestamp')

            # Generate signals
            signals = self.generate_signals(df, symbol)

            # Process and publish signals
            for signal in signals:
                self._process_and_publish_signal(signal, symbol, timestamp)

        except Exception as e:
            self.logger.error(f"Error handling market data event: {e}")

    def _process_and_publish_signal(
        self, signal: dict[str, Any], symbol: str, timestamp: float
    ) -> None:
        """Process and publish a trading signal.

        Args:
            signal: Signal dictionary with signal information
            symbol: Trading symbol
            timestamp: Signal timestamp
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                self.invalid_signal_count += 1
                self.logger.warning(f"Invalid signal generated: {signal}")
                return

            # Apply signal filters
            if not self._apply_signal_filters(signal):
                self.logger.debug(f"Signal filtered out: {signal}")
                return

            # Create signal event
            signal_event = create_signal_event(
                symbol=symbol,
                signal_type=signal['signal_type'],
                strength=signal.get('confidence', 1.0),
                confidence=signal.get('confidence', 1.0),
                source=self.name,
                metadata=signal.get('metadata', {}),
            )

            # Publish signal event
            self.event_bus.publish(signal_event)

            # Update tracking
            self.signals.append(signal)
            self.signal_history.append(signal)  # For backward compatibility
            self.signal_count += 1
            self.valid_signal_count += 1
            self.last_signal_time = timestamp

            self.logger.debug(f"Published signal event: {signal_event}")

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self.invalid_signal_count += 1

    def _validate_signal(self, signal: dict[str, Any]) -> bool:
        """Validate a signal dictionary structure.

        Args:
            signal: Signal dictionary to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['signal_type', 'confidence', 'metadata']
            for field in required_fields:
                if field not in signal:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            # Validate signal type
            if not isinstance(signal['signal_type'], str):
                self.logger.error("Signal type must be a string")
                return False

            valid_signal_types = [
                'BUY',
                'SELL',
                'HOLD',
                'CLOSE',
                'INCREASE_POSITION',
                'DECREASE_POSITION',
            ]
            if signal['signal_type'] not in valid_signal_types:
                self.logger.error(f"Invalid signal type: {signal['signal_type']}")
                return False

            # Validate confidence
            confidence = signal['confidence']
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                self.logger.error(f"Confidence must be a number between 0.0 and 1.0: {confidence}")
                return False

            # Validate metadata
            metadata = signal['metadata']
            if not isinstance(metadata, dict):
                self.logger.error("Metadata must be a dictionary")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    def _apply_signal_filters(self, signal: dict[str, Any]) -> bool:
        """Apply signal filters to determine if signal should be processed.

        Args:
            signal: Signal dictionary to filter

        Returns:
            True if signal passes all filters, False otherwise
        """
        try:
            # Apply each configured filter
            for filter_config in self.config.signal_filters:
                if not self._apply_single_filter(signal, filter_config):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error applying signal filters: {e}")
            return False

    def _apply_single_filter(self, signal: dict[str, Any], filter_config: Any) -> bool:
        """Apply a single signal filter.

        Args:
            signal: Signal dictionary to filter
            filter_config: Filter configuration

        Returns:
            True if signal passes filter, False otherwise
        """
        try:
            if hasattr(filter_config, 'enabled') and not filter_config.enabled:
                return True

            min_confidence = getattr(filter_config, 'min_confidence', 0.0)
            max_confidence = getattr(filter_config, 'max_confidence', 1.0)
            min_strength = getattr(filter_config, 'min_strength', 0.0)
            max_strength = getattr(filter_config, 'max_strength', 1.0)
            min_duration = getattr(filter_config, 'min_signal_duration', 0)
            max_duration = getattr(filter_config, 'max_signal_duration', float('inf'))
            custom_filters = getattr(filter_config, 'custom_filters', {})

            # Confidence filter
            confidence = signal.get('confidence', 0.0)
            if not (min_confidence <= confidence <= max_confidence):
                return False

            # Strength filter
            strength = signal.get('strength', confidence)
            if not (min_strength <= strength <= max_strength):
                return False

            # Duration filter (if metadata contains duration)
            duration = signal.get('metadata', {}).get('duration', 1)
            if not (min_duration <= duration <= max_duration):
                return False

            # Custom filters
            for filter_name, filter_value in custom_filters.items():
                if (
                    filter_name in signal['metadata']
                    and signal['metadata'][filter_name] != filter_value
                ):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error applying single filter: {e}")
            return False

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        """Generate trading signals based on market data.

        Args:
            data: DataFrame with market data
            symbol: Trading symbol

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD', etc.)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)
        """
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the given data.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with indicator values added as columns
        """
        try:
            result_data = data.copy()

            for indicator_name, indicator in self.indicators.items():
                try:
                    # Validate data for indicator
                    indicator.validate_data(result_data)

                    # Calculate indicator values
                    indicator_data = indicator.calculate(result_data)

                    # Add indicator columns to result
                    for col in indicator_data.columns:
                        if col not in result_data.columns:
                            result_data[col] = indicator_data[col]

                    self.logger.debug(f"Calculated indicator: {indicator_name}")

                except Exception as e:
                    self.logger.error(f"Error calculating indicator {indicator_name}: {e}")

            return result_data

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return data

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction.

        Args:
            data: DataFrame with market data and indicators

        Returns:
            DataFrame with prepared features
        """
        try:
            result_data = data.copy()

            # Add basic features
            result_data['returns'] = data['close'].pct_change()
            result_data['price_change'] = data['close'] - data['open']
            result_data['high_low_spread'] = data['high'] - data['low']
            result_data['volume_price_trend'] = data['volume'] * result_data['returns']

            # Add moving averages
            for period in [5, 10, 20]:
                result_data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
                result_data[f'price_to_ma_{period}'] = data['close'] / result_data[f'ma_{period}']

            # Add volatility features
            result_data['volatility_5'] = result_data['returns'].rolling(window=5).std()
            result_data['volatility_20'] = result_data['returns'].rolling(window=20).std()

            # Drop rows with NaN values
            result_data = result_data.dropna()

            return result_data

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return data

    def get_required_columns(self) -> list[str]:
        """Get list of required data columns.

        Returns:
            List of required column names
        """
        raise NotImplementedError("Concrete strategies must define required data columns")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, raises exception otherwise
        """
        try:
            # Check if data is empty
            if data is None or data.empty:
                raise ValueError("Input data cannot be None or empty")

            # Check if data has required columns
            required_columns = self.get_required_columns()
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate OHLC relationships
            valid_rows = ~(
                data['high'].isna()
                | data['low'].isna()
                | data['open'].isna()
                | data['close'].isna()
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

            return True

        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False

    # ------------------------------------------------------------------#
    # Lifecycle hooks
    # ------------------------------------------------------------------#
    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked before the simulation loop starts."""
        return None

    def before_tick(self, context: dict[str, Any]) -> None:
        """Hook invoked before each processed market data tick."""
        return None

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        """Hook invoked after executing a tick."""
        return None

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked after the simulation loop completes."""
        return None

    def reset(self) -> None:
        """Reset strategy state for reuse."""
        self.current_step = 0
        self.signals.clear()
        self.signal_history.clear()  # Clear signal history for backward compatibility
        self.signal_count = 0
        self.valid_signal_count = 0
        self.invalid_signal_count = 0
        self.last_signal_time = None

        # Reset indicators
        for indicator in self.indicators.values():
            indicator.reset()

        # Reset models
        for model in self.models.values():
            if hasattr(model, 'reset'):
                model.reset()

        self.logger.info(f"Signal strategy {self.name} reset")

    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information and current state.

        Returns:
            Dictionary with strategy information
        """
        if hasattr(self.type, 'value'):
            type_raw = str(self.type.value)
        elif isinstance(self.type, str):
            type_raw = self.type
        else:
            type_raw = str(self.type)
        type_value = StrategyTypeLabel(type_raw)
        indicators_list = [
            name
            for name, indicator in self.indicators.items()
            if getattr(indicator, 'is_ready', False)
        ]

        return {
            "name": self.name,
            "type": type_value,
            "is_initialized": self.is_initialized,
            "symbols": self.config.symbols,
            "indicators": indicators_list,
            "models": list(self.models.keys()),
            "signal_count": self.signal_count,
            "valid_signal_count": self.valid_signal_count,
            "invalid_signal_count": self.invalid_signal_count,
            "last_signal_time": self.last_signal_time,
            "config": self.config.model_dump(),
            "performance_metrics": getattr(self, 'performance_metrics', {}),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        total_signals = self.signal_count
        valid_signals = self.valid_signal_count
        invalid_signals = self.invalid_signal_count

        return {
            "total_signals": total_signals,
            "valid_signals": valid_signals,
            "invalid_signals": invalid_signals,
            "signal_quality_ratio": valid_signals / total_signals if total_signals > 0 else 0.0,
            "last_signal_time": self.last_signal_time,
        }

    def _create_standard_signal(
        self,
        signal_type: str,
        confidence: float,
        action: str,
        symbol: str,
        timestamp: float,
        **metadata: Any,
    ) -> dict[str, Any]:
        """Create a standardized trading signal.

        Args:
            signal_type: Type of signal
            confidence: Confidence level
            action: Action description
            symbol: Trading symbol
            timestamp: Signal timestamp
            **metadata: Additional metadata

        Returns:
            Standardized signal dictionary
        """
        # Ensure all required metadata is present
        full_metadata = {
            'strategy_name': self.name,
            'strategy_type': self.type,
            'symbol': symbol,
            'timestamp': timestamp,
            **metadata,
        }

        return SignalGenerator.create_signal(
            signal_type=signal_type,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            metadata=full_metadata,
        )

    def should_retrain_based_on_performance(self) -> bool:
        """Determine if models should be retrained based on performance.

        Returns:
            True if models should be retrained
        """
        try:
            if not hasattr(self, 'models') or not self.models:
                return False

            # Check performance metrics
            if len(self.signal_history) < 10:
                return False

            # Calculate recent performance
            recent_signals = self.signal_history[-10:]
            successful_signals = sum(
                1 for signal in recent_signals if signal.get('signal_type') in ['BUY', 'SELL']
            )

            success_rate = successful_signals / len(recent_signals) if recent_signals else 0

            # Retrain if success rate is below threshold
            return success_rate < 0.3  # 30% threshold

        except Exception as e:
            self.logger.error(f"Error checking performance for retraining: {e}")
            return False

    def should_retrain_based_on_data_drift(self) -> bool:
        """Determine if models should be retrained based on data drift.

        Returns:
            True if models should be retrained due to data drift
        """
        try:
            if not hasattr(self, 'models') or not self.models:
                return False

            # Check for data drift by comparing recent statistics
            if len(self.signal_history) < 20:
                return False

            # Simple drift detection - compare recent vs older statistics
            recent_signals = self.signal_history[-10:]
            older_signals = self.signal_history[-20:-10]

            if not recent_signals or not older_signals:
                return False

            # Compare average confidence levels
            recent_confidence = np.mean([s.get('confidence', 0) for s in recent_signals])
            older_confidence = np.mean([s.get('confidence', 0) for s in older_signals])

            # Retrain if confidence has dropped significantly
            confidence_drop = (
                (older_confidence - recent_confidence) / older_confidence
                if older_confidence > 0
                else 0
            )

            return confidence_drop > 0.5  # 50% drop in confidence

        except Exception as e:
            self.logger.error(f"Error checking data drift for retraining: {e}")
            return False
