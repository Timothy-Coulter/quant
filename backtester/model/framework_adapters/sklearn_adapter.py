"""Scikit-learn adapter for ML model integration.

This module provides an adapter for integrating scikit-learn models with the backtester model system.
"""

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import dump, load
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.svm import SVC, SVR

from backtester.model.base_model import ModelFrameworkAdapter
from backtester.model.model_configs import SklearnModelConfig


class SklearnAdapter(ModelFrameworkAdapter[BaseEstimator]):
    """Adapter for scikit-learn models.

    This adapter provides a consistent interface for scikit-learn models while
    handling framework-specific implementation details.
    """

    def __init__(self, config: SklearnModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the sklearn adapter.

        Args:
            config: Configuration for the sklearn model
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model: BaseEstimator | None = None

    def initialize_model(self, config: SklearnModelConfig) -> BaseEstimator:
        """Initialize a scikit-learn model based on configuration.

        Args:
            config: Sklearn model configuration

        Returns:
            Initialized sklearn model instance
        """
        # Model mapping
        model_mapping = {
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'SVR': SVR,
            'SVC': SVC,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'Lasso': Lasso,
            'Ridge': Ridge,
            'ElasticNet': ElasticNet,
            'KMeans': KMeans,
        }

        if config.model_class not in model_mapping:
            raise ValueError(f"Unsupported sklearn model: {config.model_class}")

        model_class = model_mapping[config.model_class]

        # Initialize model with hyperparameters
        model = model_class(**config.hyperparameters)

        self._model = model
        self.logger.info(f"Initialized sklearn model: {config.model_class}")
        return model

    def train_model(
        self, model: BaseEstimator, features: pd.DataFrame, target: pd.Series
    ) -> BaseEstimator:
        """Train the sklearn model.

        Args:
            model: Sklearn model instance
            features: Training features
            target: Training targets

        Returns:
            Trained model instance
        """
        # Handle classification vs regression
        if hasattr(model, 'predict_proba'):
            # Classification model
            if target.dtype == 'object' or target.nunique() <= 10:
                y_train = target
            else:
                # Convert continuous target to binary classes for classification
                y_train = (target > target.median()).astype(int)
        else:
            # Regression model
            y_train = target

        # Train the model
        model.fit(features, y_train)

        self.logger.info(f"Trained sklearn model: {model.__class__.__name__}")
        return model

    def predict(self, model: BaseEstimator, features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions using sklearn model.

        Args:
            model: Sklearn model instance
            features: Features for prediction

        Returns:
            Model predictions
        """
        predictions_raw = model.predict(features)
        predictions = np.asarray(predictions_raw)

        # Handle different prediction formats
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions

    def predict_proba(self, model: BaseEstimator, features: pd.DataFrame) -> NDArray[Any]:
        """Generate probability predictions using sklearn classification model.

        Args:
            model: Sklearn classification model instance
            features: Features for prediction

        Returns:
            Model probability predictions
        """
        if hasattr(model, 'predict_proba'):
            proba_raw = model.predict_proba(features)
            proba = np.asarray(proba_raw)
            return proba
        else:
            # Fallback to regular predictions for regression models
            predictions = self.predict(model, features)
            proba = np.column_stack([1 - predictions, predictions])
            return np.asarray(proba)

    def save_model(self, model: BaseEstimator, filepath: str) -> None:
        """Save sklearn model to file.

        Args:
            model: Sklearn model instance
            filepath: Path to save the model
        """
        dump(model, filepath)
        self.logger.info(f"Saved sklearn model to: {filepath}")

    def load_model(self, filepath: str) -> BaseEstimator:
        """Load sklearn model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        model = cast(BaseEstimator, load(filepath))
        self.logger.info(f"Loaded sklearn model from: {filepath}")
        return model

    def get_feature_importance(self, model: BaseEstimator) -> NDArray[Any] | None:
        """Get feature importance from the model if available.

        Args:
            model: Sklearn model instance

        Returns:
            Feature importance array or None if not available
        """
        if hasattr(model, 'feature_importances_'):
            return np.asarray(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            coef_array = np.asarray(model.coef_)
            importance = np.abs(coef_array)
            return cast(NDArray[Any], importance)
        else:
            return None

    def get_model_params(self, model: BaseEstimator) -> dict[str, Any]:
        """Get model parameters.

        Args:
            model: Sklearn model instance

        Returns:
            Dictionary of model parameters
        """
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if isinstance(params, dict):
                return params
            else:
                return {}
        else:
            return {}

    @classmethod
    def create_model(cls, model_name: str, config: SklearnModelConfig) -> 'SklearnAdapter':
        """Create a sklearn model adapter with the specified model.

        Args:
            model_name: Name of the model to create
            config: Configuration for the model

        Returns:
            SklearnAdapter instance
        """
        adapter = cls(config)
        adapter.initialize_model(config)
        return adapter
