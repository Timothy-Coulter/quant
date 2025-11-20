"""Scikit-learn model implementations for the backtester.

This module provides concrete implementations of scikit-learn models that extend
the BaseModel class and use the SklearnAdapter for framework integration.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from backtester.model.base_model import BaseModel
from backtester.model.framework_adapters import SklearnAdapter
from backtester.model.model_configs import SklearnModelConfig
from backtester.model.model_factory import ModelFactory


class SklearnModel(BaseModel[BaseEstimator]):
    """Base class for scikit-learn models.

    This class provides common functionality for all scikit-learn models
    while delegating framework-specific operations to the SklearnAdapter.
    """

    _expected_framework = "sklearn"

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the sklearn model.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.adapter = SklearnAdapter(config, logger)
        self._model = None

    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for sklearn model training.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (features, target)
        """
        self.validate_data(data)

        # Use the base model's feature preparation
        features = self._prepare_features(data)

        # Create target based on model type
        if self.type == 'regression':
            # Predict next period's close price
            target = data['close'].shift(-1)
        elif self.type == 'classification':
            # Predict price direction (1 = up, 0 = down)
            target = (data['close'].shift(-1) > data['close']).astype(int)
        else:
            # Default to next period return
            target = data['close'].pct_change().shift(-1)

        # Align features and target
        min_len = min(len(features), len(target))
        features = features.iloc[:min_len]
        target = target.iloc[:min_len]

        # Remove NaN values
        valid_indices = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_indices]
        target = target[valid_indices]

        self.logger.debug(f"Prepared {len(features)} samples with {len(features.columns)} features")
        return features, target

    def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Train the sklearn model.

        Args:
            features: Training features
            target: Training targets

        Returns:
            Dictionary with training results and metrics
        """
        try:
            # Initialize model if not already done
            if self._model is None:
                self._model = self.adapter.initialize_model(self.config)

            # Train the model
            self._model = self.adapter.train_model(self._model, features, target)
            self._is_trained = True

            # Calculate training metrics
            predictions = self.adapter.predict(self._model, features)
            metrics = self._calculate_metrics(target, predictions)

            self.logger.info(f"Trained sklearn model: {self.config.model_class}")
            return {
                'success': True,
                'model_class': self.config.model_class,
                'metrics': metrics,
                'feature_count': len(features.columns),
                'sample_count': len(features),
            }

        except Exception as e:
            self.logger.error(f"Failed to train sklearn model: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions using the sklearn model.

        Args:
            features: Features for prediction

        Returns:
            Model predictions
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")

        if self._model is None:
            raise ValueError("Model is not initialized")

        if self._model is None:
            raise ValueError("Model is not initialized")
        if not hasattr(self.adapter, 'predict'):
            raise ValueError("Adapter does not support prediction")
        if not hasattr(self._model, 'predict'):
            raise ValueError("Model does not support prediction")
        predictions = self.adapter.predict(self._model, features)
        return predictions

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on model predictions.

        Args:
            data: DataFrame with market data

        Returns:
            List of signal dictionaries
        """
        if not self._is_trained:
            return self._generate_no_signal("Model not trained")

        try:
            # Prepare features
            features, _ = self.prepare_data(data)
            if len(features) == 0:
                return self._generate_no_signal("No valid data for prediction")

            # Get predictions
            predictions = self.predict(features)

            # Generate signals based on model type
            if self.type == 'classification':
                return self._generate_classification_signals(features, predictions)
            else:
                return self._generate_regression_signals(features, predictions)

        except Exception as e:
            self.logger.error(f"Failed to generate signals: {e}")
            return self._generate_no_signal(f"Signal generation error: {e}")

    def _generate_classification_signals(
        self, features: pd.DataFrame, predictions: NDArray[Any]
    ) -> list[dict[str, Any]]:
        """Generate signals for classification models.

        Args:
            features: Feature DataFrame
            predictions: Model predictions

        Returns:
            List of signal dictionaries
        """
        signals = []

        if hasattr(self._model, 'predict_proba'):
            probabilities = self.adapter.predict_proba(self._model, features)
            buy_prob = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

            for i, (pred, prob) in enumerate(zip(predictions, buy_prob, strict=False)):
                confidence = abs(prob - 0.5) * 2  # Convert to 0-1 confidence

                if prob > self.config.signal_threshold:
                    signal_type = "STRONG_BUY" if prob > 0.8 else "BUY"
                elif prob < (1 - self.config.signal_threshold):
                    signal_type = "STRONG_SELL" if prob < 0.2 else "SELL"
                else:
                    signal_type = "HOLD"

                if confidence >= self.config.confidence_threshold:
                    timestamp = features.index[i]
                    signal = self._create_signal_from_prediction(
                        signal_type, confidence, timestamp, pred, prob
                    )
                    signals.append(signal)
        else:
            # Use binary predictions
            for i, pred in enumerate(predictions):
                confidence = 0.7  # Default confidence for binary predictions

                signal_type = "BUY" if pred == 1 else "SELL"

                timestamp = features.index[i]
                signal = self._create_signal_from_prediction(
                    signal_type, confidence, timestamp, pred
                )
                signals.append(signal)

        return signals

    def _generate_regression_signals(
        self, features: pd.DataFrame, predictions: NDArray[Any]
    ) -> list[dict[str, Any]]:
        """Generate signals for regression models.

        Args:
            features: Feature DataFrame
            predictions: Model predictions

        Returns:
            List of signal dictionaries
        """
        signals = []

        # Calculate prediction confidence based on deviation from mean
        prediction_std = np.std(predictions)
        prediction_mean = np.mean(predictions)

        for i, pred in enumerate(predictions):
            confidence = min(abs(pred - prediction_mean) / (prediction_std + 1e-8), 1.0)

            if pred > prediction_mean * (1 + self.config.signal_threshold):
                signal_type = "STRONG_BUY" if pred > prediction_mean * 1.2 else "BUY"
            elif pred < prediction_mean * (1 - self.config.signal_threshold):
                signal_type = "STRONG_SELL" if pred < prediction_mean * 0.8 else "SELL"
            else:
                signal_type = "HOLD"

            if confidence >= self.config.confidence_threshold:
                timestamp = features.index[i]
                signal = self._create_signal_from_prediction(
                    signal_type, confidence, timestamp, pred
                )
                signals.append(signal)

        return signals

    def _calculate_metrics(self, target: pd.Series, predictions: NDArray[Any]) -> dict[str, float]:
        """Calculate model performance metrics.

        Args:
            target: Actual values
            predictions: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Mean Squared Error
        mse = np.mean((target - predictions) ** 2)
        metrics['mse'] = float(mse)

        # Mean Absolute Error
        mae = np.mean(np.abs(target - predictions))
        metrics['mae'] = float(mae)

        # R-squared
        ss_res = np.sum((target - predictions) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r2'] = float(r2)

        return metrics

    def _create_signal_from_prediction(
        self,
        signal_type: str,
        confidence: float,
        timestamp: Any,
        prediction: float,
        probability: float | None = None,
    ) -> dict[str, Any]:
        """Create a signal dictionary from model prediction.

        Args:
            signal_type: Type of signal
            confidence: Confidence level
            timestamp: Signal timestamp
            prediction: Model prediction
            probability: Prediction probability (for classification)

        Returns:
            Signal dictionary
        """
        from backtester.signal.signal_types import SignalType

        # Map signal types to SignalType enum
        type_mapping = {
            'BUY': SignalType.BUY,
            'SELL': SignalType.SELL,
            'HOLD': SignalType.HOLD,
            'STRONG_BUY': SignalType.BUY,
            'STRONG_SELL': SignalType.SELL,
        }

        signal_type_enum = type_mapping.get(signal_type, SignalType.HOLD)

        # Create action description
        if signal_type in ['STRONG_BUY', 'BUY']:
            action = f"Price prediction suggests upward movement (prediction: {prediction:.4f})"
        elif signal_type in ['STRONG_SELL', 'SELL']:
            action = f"Price prediction suggests downward movement (prediction: {prediction:.4f})"
        else:
            action = f"Price prediction suggests holding (prediction: {prediction:.4f})"

        # Additional metadata
        metadata = {
            'model_prediction': float(prediction),
            'model_class': self.config.model_class,
        }

        if probability is not None:
            metadata['prediction_probability'] = float(probability)

        signal_result = self._create_standard_signal(
            signal_type=signal_type_enum,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            **metadata,
        )
        if not isinstance(signal_result, dict):
            raise ValueError("Signal generation failed")
        return signal_result

    def _generate_no_signal(self, reason: str) -> list[dict[str, Any]]:
        """Generate a no-signal response.

        Args:
            reason: Reason for no signal

        Returns:
            List containing a HOLD signal
        """
        import datetime

        from backtester.signal.signal_types import SignalType

        return [
            self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=f"No signal generated: {reason}",
                confidence=0.0,
                timestamp=datetime.datetime.now(),
                reason=reason,
            )
        ]

    def get_feature_importance(self) -> NDArray[Any] | None:
        """Get feature importance from the model if available.

        Returns:
            Feature importance array or None
        """
        if self._model is None:
            return None

        if self._model is None:
            return None
        if self._model is None:
            return None
        try:
            importance = self.adapter.get_feature_importance(self._model)
            if importance is not None:
                return importance
            else:
                return np.zeros(len(self._feature_columns))
        except Exception:
            if len(self._feature_columns) > 0:
                try:
                    return np.zeros(len(self._feature_columns))
                except Exception:
                    return np.zeros(1)
            else:
                return np.zeros(1)


@ModelFactory.register_model("sklearn_linear_regression")
class LinearRegressionModel(SklearnModel):
    """Linear Regression model using scikit-learn."""

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the Linear Regression model.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        # Set default values for Linear Regression
        config_dict = config.model_dump()
        config_dict['model_class'] = 'LinearRegression'
        config_dict['model_type'] = 'regression'
        config_dict['framework'] = 'sklearn'

        updated_config = SklearnModelConfig(**config_dict)
        super().__init__(updated_config, logger)


@ModelFactory.register_model("sklearn_random_forest")
class RandomForestModel(SklearnModel):
    """Random Forest model using scikit-learn."""

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the Random Forest model.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        # Set default values for Random Forest
        config_dict = config.model_dump()
        config_dict['model_class'] = 'RandomForestRegressor'
        config_dict['model_type'] = 'regression'
        config_dict['framework'] = 'sklearn'

        # Set default hyperparameters if not provided
        if not config_dict.get('hyperparameters'):
            config_dict['hyperparameters'] = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
            }

        updated_config = SklearnModelConfig(**config_dict)
        super().__init__(updated_config, logger)


@ModelFactory.register_model("sklearn_random_forest_classifier")
class RandomForestClassifierModel(SklearnModel):
    """Random Forest Classifier model using scikit-learn."""

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the Random Forest Classifier model.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        # Set default values for Random Forest Classifier
        config_dict = config.model_dump()
        config_dict['model_class'] = 'RandomForestClassifier'
        config_dict['model_type'] = 'classification'
        config_dict['framework'] = 'sklearn'

        # Set default hyperparameters if not provided
        if not config_dict.get('hyperparameters'):
            config_dict['hyperparameters'] = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
            }

        updated_config = SklearnModelConfig(**config_dict)
        super().__init__(updated_config, logger)


@ModelFactory.register_model("sklearn_svm")
class SVMModel(SklearnModel):
    """Support Vector Machine model using scikit-learn."""

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the Support Vector Machine model.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        # Set default values for SVM
        config_dict = config.model_dump()
        config_dict['model_class'] = 'SVR'
        config_dict['model_type'] = 'regression'
        config_dict['framework'] = 'sklearn'

        # Set default hyperparameters if not provided
        if not config_dict.get('hyperparameters'):
            config_dict['hyperparameters'] = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}

        updated_config = SklearnModelConfig(**config_dict)
        super().__init__(updated_config, logger)
