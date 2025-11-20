"""Base model class and framework adapter pattern for the model system.

This module provides the abstract base class that all machine learning models must implement,
along with a framework adapter pattern for integrating different ML libraries.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import ValidationError

from backtester.core.logger import BacktesterLogger
from backtester.data.data_retrieval import DataRetrieval
from backtester.signal.signal_types import SignalGenerator, SignalType

T = TypeVar('T')
ModelType = TypeVar('ModelType')


class ModelFrameworkAdapter(Protocol[ModelType]):
    """Abstract adapter for ML framework integration.

    This class defines the interface that all framework adapters must implement
    to provide consistent model operations across different ML frameworks.
    """

    def initialize_model(self, config: Any) -> ModelType:
        """Initialize framework-specific model.

        Args:
            config: Framework-specific configuration

        Returns:
            Initialized model instance
        """
        ...

    def train_model(self, model: ModelType, features: pd.DataFrame, target: pd.Series) -> ModelType:
        """Train the framework-specific model.

        Args:
            model: Framework-specific model instance
            features: Training features
            target: Training targets

        Returns:
            Trained model instance
        """
        ...

    def predict(self, model: ModelType, features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions using framework-specific model.

        Args:
            model: Framework-specific model instance
            features: Features for prediction

        Returns:
            Model predictions
        """
        ...

    def save_model(self, model: ModelType, filepath: str) -> None:
        """Save model to file.

        Args:
            model: Framework-specific model instance
            filepath: Path to save the model
        """
        ...

    def load_model(self, filepath: str) -> ModelType:
        """Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        ...


class BaseModel[ModelType](ABC):
    """Abstract base class for all machine learning models.

    This class provides the common interface and functionality that all machine learning
    models must implement. It follows the modular component architecture
    with proper typing, logging, and validation integration.
    """

    def __init__(self, config: Any, logger: logging.Logger | None = None) -> None:
        """Initialize the model with configuration.

        Args:
            config: Model configuration parameters
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or BacktesterLogger.get_logger(__name__)
        self.name = config.model_name
        self.type = config.model_type
        self.framework = config.framework
        self._model: ModelType | None = None
        self._is_trained = False
        self._feature_columns: list[str] = []
        self._target_column = config.target_column

        # Initialize data handler
        self.data_handler = (
            DataRetrieval(config.data_config) if hasattr(config, 'data_config') else None
        )

        # Validate configuration
        self._validate_configuration()

        self.logger.debug(
            f"Initialized model: {self.name} (type: {self.type}, framework: {self.framework})"
        )

    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare and engineer features from raw data.

        Args:
            data: DataFrame with OHLCV data (datetime indexed)
                Expected columns: 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Tuple of (features, target) where:
            - features: DataFrame with engineered features
            - target: Series with target values
        """
        pass

    @abstractmethod
    def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Train the model on provided data.

        Args:
            features: Training features
            target: Training targets

        Returns:
            Dictionary with training results and metrics
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions from features.

        Args:
            features: Features for prediction

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on model predictions.

        Args:
            data: DataFrame with market data and calculated features

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
        if len(data) < self.config.lookback_period:
            raise ValueError(
                f"Insufficient data: need at least {self.config.lookback_period} periods"
            )

    def get_required_columns(self) -> list[str]:
        """Get list of required data columns.

        Returns:
            List of required column names
        """
        return ['open', 'high', 'low', 'close', 'volume']

    def reset(self) -> None:
        """Reset model state for reuse."""
        self._model = None
        self._is_trained = False
        self._feature_columns.clear()
        self.logger.debug(f"Model {self.name} reset")

    def _validate_configuration(self) -> None:
        """Validate model-specific configuration parameters."""
        try:
            # Validate required fields
            if not self.config.model_name:
                raise ValueError("model_name is required")
            if not self.config.model_type:
                raise ValueError("model_type is required")
            if not self.config.framework:
                raise ValueError("framework is required")
            if self.config.lookback_period <= 0:
                raise ValueError("lookback_period must be positive")

        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get model information and current configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "type": self.type,
            "framework": self.framework,
            "is_trained": self._is_trained,
            "config": self.config.model_dump(),
            "feature_columns": self._feature_columns,
        }

    def _create_standard_signal(
        self,
        signal_type: SignalType | str,
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
        full_metadata = {
            'model_name': self.name,
            'model_type': self.type,
            'framework': self.framework,
            **metadata,
        }

        # Convert string to SignalType if needed
        if isinstance(signal_type, str):
            try:
                signal_type = SignalType(signal_type.upper())
            except ValueError:
                signal_type = SignalType.HOLD  # Default fallback
        # Convert integer to SignalType if needed
        elif isinstance(signal_type, int):
            if signal_type == 1:
                signal_type = SignalType.BUY
            elif signal_type == -1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD  # Default fallback

        signal_dict = SignalGenerator.create_signal(
            signal_type=signal_type,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            metadata=full_metadata,
        )
        if not isinstance(signal_dict, dict):
            raise ValueError("Signal generation failed")
        return signal_dict

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training/prediction.

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with engineered features
        """
        features = data.copy()

        # Add basic technical indicators as features
        features['returns'] = data['close'].pct_change()
        features['price_change'] = data['close'] - data['open']
        features['high_low_spread'] = data['high'] - data['low']
        features['volume_price_trend'] = data['volume'] * features['returns']

        # Add moving averages
        for period in [5, 10, 20]:
            features[f'ma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'price_to_ma_{period}'] = data['close'] / features[f'ma_{period}']

        # Add volatility features
        features['volatility_5'] = features['returns'].rolling(window=5).std()
        features['volatility_20'] = features['returns'].rolling(window=20).std()

        # Add RSI as a momentum indicator
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # Drop rows with NaN values (due to rolling windows)
        features = features.dropna()

        # Select only numeric columns for features
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_columns]

        # Store feature columns for later use
        self._feature_columns = list(features.columns)

        return features

    def _split_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            features: Feature DataFrame
            target: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(features) * self.config.train_test_split)

        x_train = features.iloc[:split_idx]
        x_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        return x_train, x_test, y_train, y_test
