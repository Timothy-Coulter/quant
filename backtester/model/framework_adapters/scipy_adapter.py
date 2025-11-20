"""SciPy adapter for ML model integration.

This module provides an adapter for integrating SciPy statistical models with the backtester model system.
"""

import logging
import pickle
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import interpolate, optimize, signal, stats

from backtester.model.base_model import ModelFrameworkAdapter
from backtester.model.model_configs import SciPyModelConfig


class SciPyAdapter(ModelFrameworkAdapter[Any]):
    """Adapter for SciPy statistical models.

    This adapter provides a consistent interface for SciPy statistical functions
    while handling framework-specific implementation details.
    """

    def __init__(self, config: SciPyModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the SciPy adapter.

        Args:
            config: Configuration for the SciPy model
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model: Any = None
        self._function: Callable[..., Any] | None = None

    def initialize_model(self, config: SciPyModelConfig) -> Callable[..., Any]:
        """Initialize a SciPy function based on configuration.

        Args:
            config: SciPy model configuration

        Returns:
            Configured SciPy function or optimizer
        """
        function_path = config.scipy_function.split('.')
        if len(function_path) != 2:
            raise ValueError(f"Invalid scipy function path: {config.scipy_function}")

        module_name, function_name = function_path

        # Get the module and function
        if module_name == 'stats':
            module: Any = stats
        elif module_name == 'optimize':
            module = optimize
        elif module_name == 'signal':
            module = signal
        elif module_name == 'interpolate':
            module = interpolate
        else:
            raise ValueError(f"Unsupported SciPy module: {module_name}")

        function = getattr(module, function_name)

        self._function = function
        self.logger.info(f"Initialized SciPy function: {config.scipy_function}")
        if callable(function):
            return cast(Callable[..., Any], function)
        else:
            # Return a dummy function that raises an error
            def dummy_function(*args: Any, **kwargs: Any) -> Any:
                raise ValueError("Function not properly initialized")

            return dummy_function

    def train_model(
        self, model: Callable[..., Any], features: pd.DataFrame, target: pd.Series
    ) -> Callable[..., Any]:
        """Train/fir the SciPy function.

        Args:
            model: SciPy function or optimizer
            features: Training features
            target: Training targets

        Returns:
            Fitted model or function
        """
        function_name: str = self.config.scipy_function

        if 'stats.linregress' in function_name:
            # Linear regression
            x = features.iloc[:, 0].values
            y = target.values
            result = stats.linregress(x, y)
            self._model = result

        elif 'stats.pearsonr' in function_name:
            # Pearson correlation
            x = features.iloc[:, 0].values
            y = target.values
            result_pearson = stats.pearsonr(x, y)
            self._model = result_pearson

        elif 'optimize.minimize' in function_name:
            # Optimization problem
            def objective_function(params: Any) -> Any:
                # Simple quadratic objective for demonstration
                x, y = params
                return (x - 1) ** 2 + (y - 1) ** 2

            initial_guess = self.config.function_params.get('initial_guess', [0, 0])
            result = optimize.minimize(objective_function, initial_guess)
            self._model = result

        elif 'signal.find_peaks' in function_name:
            # Peak finding
            # This is typically used on time series data
            self._model = lambda data: signal.find_peaks(data, height=0)

        elif 'interpolate.interp1d' in function_name:
            # Interpolation
            x = features.index.values
            y = target.values
            interp_fn = interpolate.interp1d(x, y, kind='linear')
            self._model = interp_fn

        else:
            # Generic function execution
            if self.config.function_params:
                self._model = model(**self.config.function_params)
            else:
                self._model = model

        self.logger.info(f"Fitted SciPy model: {function_name}")
        return self._model  # type: ignore[no-any-return]

    def predict(self, model: Callable[..., Any], features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions using SciPy model.

        Args:
            model: SciPy fitted model or function
            features: Features for prediction

        Returns:
            Model predictions
        """
        function_name = self.config.scipy_function

        if hasattr(self._model, 'slope') and hasattr(self._model, 'intercept'):
            # Linear regression result (LinregressResult)
            x = features.iloc[:, 0].values
            predictions = self._model.slope * x + self._model.intercept

        elif hasattr(self._model, 'statistic') and hasattr(self._model, 'pvalue'):
            # Pearson correlation result
            x = features.iloc[:, 0].values
            predictions = np.full(len(x), float(self._model.statistic))

        elif function_name == 'optimize.minimize':
            # Optimization result
            predictions = np.array([self._model.x])

        elif function_name == 'signal.find_peaks':
            # Peak finding result
            if len(features) > 0:
                signal_data = features.iloc[:, 0].values
                peaks, _ = self._model(signal_data)
                predictions = np.zeros(len(signal_data))
                predictions[peaks] = 1
            else:
                predictions = np.array([])

        elif isinstance(self._model, interpolate.interp1d):
            # Interpolation function
            x = features.index.values if hasattr(features, 'index') else np.arange(len(features))
            predictions = self._model(x)

        elif callable(self._model):
            # Callable function
            x = features.index.values if hasattr(features, 'index') else np.arange(len(features))
            predictions = self._model(x)
            if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
                predictions = predictions.flatten()
            else:
                predictions = np.array([predictions])

        else:
            # Fallback: return zeros
            predictions = np.zeros(len(features))

        return np.asarray(predictions)

    def save_model(self, model: Callable[..., Any], filepath: str) -> None:
        """Save SciPy model to file.

        Args:
            model: SciPy model or function
            filepath: Path to save the model
        """
        # Save the model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(
                {
                    'model': model,
                    'function_name': self.config.scipy_function,
                    'function_params': self.config.function_params,
                },
                f,
            )

        self.logger.info(f"Saved SciPy model to: {filepath}")

    def load_model(self, filepath: str) -> dict[str, Any]:
        """Load SciPy model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Dictionary containing model and metadata
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.logger.info(f"Loaded SciPy model from: {filepath}")
        if isinstance(data, dict):
            return data
        else:
            return {
                'model': data,
                'function_name': self.config.scipy_function,
                'function_params': self.config.function_params,
            }

    def get_model_info(self, model: Callable[..., Any]) -> dict[str, Any]:
        """Get information about the SciPy model.

        Args:
            model: SciPy model or function

        Returns:
            Dictionary of model information
        """
        info = {
            'function_name': self.config.scipy_function,
            'function_params': self.config.function_params,
        }

        # Add function-specific information
        if hasattr(model, 'slope') and hasattr(model, 'intercept'):
            info.update(
                {
                    'slope': str(getattr(model, 'slope', 0.0)),
                    'intercept': str(getattr(model, 'intercept', 0.0)),
                    'r_value': str(getattr(model, 'rvalue', 0.0)),
                    'p_value': str(getattr(model, 'pvalue', 0.0)),
                    'stderr': str(getattr(model, 'stderr', 0.0)),
                }
            )
        elif hasattr(model, 'x'):  # Optimization result
            x_attr = getattr(model, 'x', [0])
            optimal_params = x_attr.tolist() if hasattr(x_attr, 'tolist') else list(x_attr)
            info.update(
                {
                    'optimal_params': optimal_params,
                    'success': str(getattr(model, 'success', False)),
                    'fun': str(getattr(model, 'fun', 0.0)),
                }
            )

        return info

    def get_correlation_coefficients(self) -> dict[str, float]:
        """Get correlation coefficients if available.

        Returns:
            Dictionary of correlation coefficients
        """
        if hasattr(self._model, 'rvalue'):
            return {'correlation': self._model.rvalue}
        elif hasattr(self._model, 'correlation'):
            return {'correlation': self._model.correlation}
        else:
            return {}

    def evaluate_model(
        self, features: pd.DataFrame, target: pd.Series, predictions: NDArray[Any]
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            target: Test targets
            predictions: Model predictions

        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate common metrics
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

    @classmethod
    def create_model(cls, model_name: str, config: SciPyModelConfig) -> 'SciPyAdapter':
        """Create a SciPy model adapter with the specified function.

        Args:
            model_name: Name of the model/function to create
            config: Configuration for the model

        Returns:
            SciPyAdapter instance
        """
        adapter = cls(config)
        adapter.initialize_model(config)
        return adapter
