"""Unit tests for the ModelFactory class."""

import logging
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.model.base_model import BaseModel
from backtester.model.model_configs import (
    ModelConfig,
)
from backtester.model.model_factory import ModelFactory


class MockModel:
    """Mock model class for testing."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the mock model.

        Args:
            config: Model configuration
        """
        self.config = config


class MockAdapter:
    """Mock adapter class for testing."""

    def __init__(self, config: ModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the mock adapter.

        Args:
            config: Adapter configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger


class TestModelFactory:
    """Test cases for ModelFactory class."""

    def setup_method(self) -> None:
        """Reset factory state before each test."""
        ModelFactory._models.clear()
        ModelFactory._adapters.clear()

    def test_register_model(self) -> None:
        """Test model registration."""
        assert "test_model" not in ModelFactory._models

        @ModelFactory.register_model("test_model")
        class TestModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                """Initialize the mock model.

                Args:
                    config: Model configuration
                """
                super().__init__(config)

        assert "test_model" in ModelFactory._models
        assert ModelFactory._models["test_model"] is TestModel

    def test_register_adapter(self) -> None:
        """Test adapter registration."""
        assert "test_framework" not in ModelFactory._adapters

        ModelFactory.register_adapter("test_framework", MockAdapter)

        assert "test_framework" in ModelFactory._adapters
        assert ModelFactory._adapters["test_framework"] is MockAdapter

    def test_create_model_success(self) -> None:
        """Test successful model creation."""

        # Register a test model
        @ModelFactory.register_model("test_model")
        class TestModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                """Initialize the mock model.

                Args:
                    config: Model configuration
                """
                super().__init__(config)

            def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                return data, data['close']

            def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
                return {"accuracy": 0.9}

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

            def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"signal_type": "HOLD", "action": "Hold", "confidence": 0.5, "metadata": {}}
                ]

        # Create config
        config = ModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
        )

        # Create model
        model = ModelFactory.create("test_model", config)

        assert isinstance(model, TestModel)
        assert model.config == config

    def test_create_model_unknown_name(self) -> None:
        """Test creating model with unknown name."""
        config = ModelConfig(
            model_name="unknown_model",
            model_type="regression",
            framework="sklearn",
        )

        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("unknown_model", config)

    def test_get_available_models(self) -> None:
        """Test getting available models."""

        # Register test models
        @ModelFactory.register_model("model1")
        class Model1(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        @ModelFactory.register_model("model2")
        class Model2(BaseModel[Any]):
            _expected_framework = "tensorflow"

            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        @ModelFactory.register_model("model3")
        class Model3(BaseModel[Any]):
            _expected_framework = "sklearn"

            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        available_models = ModelFactory.get_available_models()

        assert "unknown" in available_models
        assert "tensorflow" in available_models
        assert "sklearn" in available_models

        assert "model1" in available_models["unknown"]
        assert "model2" in available_models["tensorflow"]
        assert "model3" in available_models["sklearn"]

    def test_get_available_frameworks(self) -> None:
        """Test getting available frameworks."""
        ModelFactory.register_adapter("framework1", MockAdapter)
        ModelFactory.register_adapter("framework2", MockAdapter)

        frameworks = ModelFactory.get_available_frameworks()

        assert "framework1" in frameworks
        assert "framework2" in frameworks

    def test_is_registered(self) -> None:
        """Test checking if model is registered."""

        @ModelFactory.register_model("registered_model")
        class RegisteredModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        assert ModelFactory.is_registered("registered_model")
        assert not ModelFactory.is_registered("unregistered_model")

    def test_is_framework_supported(self) -> None:
        """Test checking if framework is supported."""
        ModelFactory.register_adapter("supported_framework", MockAdapter)

        assert ModelFactory.is_framework_supported("supported_framework")
        assert not ModelFactory.is_framework_supported("unsupported_framework")

    def test_create_from_config_dict(self) -> None:
        """Test creating model from configuration dictionary."""

        # Register a test model
        @ModelFactory.register_model("dict_model")
        class DictModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                """Initialize the mock model.

                Args:
                    config: Model configuration
                """
                super().__init__(config)

            def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                return data, data['close']

            def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
                return {"accuracy": 0.9}

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

            def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"signal_type": "HOLD", "action": "Hold", "confidence": 0.5, "metadata": {}}
                ]

        config_dict = {
            "model_name": "dict_model",
            "model_type": "regression",
            "framework": "sklearn",
            "model_class": "LinearRegression",  # Required for SklearnModelConfig
            "lookback_period": 30,
        }

        model = ModelFactory.create_from_config_dict(config_dict)

        assert isinstance(model, DictModel)
        assert model.config.model_name == "dict_model"
        assert model.config.model_type == "regression"

    def test_create_from_config_dict_invalid_config(self) -> None:
        """Test creating model from invalid configuration dictionary."""
        config_dict = {
            "model_name": "dict_model",
            "model_type": "invalid_type",  # Invalid model type
            "framework": "sklearn",
        }

        with pytest.raises(ValueError, match="Invalid configuration"):
            ModelFactory.create_from_config_dict(config_dict)

    def test_create_from_config_dict_sklearn(self) -> None:
        """Test creating sklearn model from configuration dictionary."""

        # First register a test model that we know works
        @ModelFactory.register_model("test_sklearn_model")
        class TestSklearnModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

            def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                return data, data['close']

            def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
                return {"accuracy": 0.9}

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

            def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"signal_type": "HOLD", "action": "Hold", "confidence": 0.5, "metadata": {}}
                ]

        config_dict = {
            "model_name": "test_sklearn_model",
            "model_type": "regression",
            "framework": "sklearn",
            "model_class": "LinearRegression",
            "lookback_period": 30,
        }

        # This should create a test model (since the actual sklearn models may not be loaded in test environment)
        model = ModelFactory.create_from_config_dict(config_dict)
        assert hasattr(model, 'config')
        assert model.config.model_name == "test_sklearn_model"
        assert model.config.model_class == "LinearRegression"

    def test_create_from_config_dict_tensorflow(self) -> None:
        """Test creating TensorFlow model from configuration dictionary."""

        # First register a test model that we know works
        @ModelFactory.register_model("test_tensorflow_model")
        class TestTensorFlowModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

            def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                return data, data['close']

            def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
                return {"accuracy": 0.9}

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

            def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"signal_type": "HOLD", "action": "Hold", "confidence": 0.5, "metadata": {}}
                ]

        config_dict = {
            "model_name": "test_tensorflow_model",
            "model_type": "regression",
            "framework": "tensorflow",
            "model_architecture": {"layers": []},
            "lookback_period": 30,
        }

        # This should create a test model (since the actual tensorflow models may not be loaded in test environment)
        model = ModelFactory.create_from_config_dict(config_dict)
        assert hasattr(model, 'config')
        assert model.config.model_name == "test_tensorflow_model"
        # The model was created successfully, which is the main test goal

    def test_get_factory_info(self) -> None:
        """Test getting comprehensive factory information."""

        # Register test models
        @ModelFactory.register_model("model1")
        class Model1(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        @ModelFactory.register_model("model2")
        class Model2(BaseModel[Any]):
            _expected_framework = "sklearn"

            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        # Register test adapters
        ModelFactory.register_adapter("sklearn", MockAdapter)
        ModelFactory.register_adapter("tensorflow", MockAdapter)

        info = ModelFactory.get_factory_info()

        assert "registered_models" in info
        assert "supported_frameworks" in info
        assert "models_by_framework" in info
        assert "total_models" in info
        assert "total_frameworks" in info

        assert "model1" in info["registered_models"]
        assert "model2" in info["registered_models"]
        assert "sklearn" in info["supported_frameworks"]
        assert "tensorflow" in info["supported_frameworks"]
        assert info["total_models"] == 2
        assert info["total_frameworks"] == 2

    def test_framework_validation(self) -> None:
        """Test framework validation during model creation."""

        # Register a model that expects a specific framework
        @ModelFactory.register_model("framework_specific_model")
        class FrameworkSpecificModel(BaseModel[Any]):
            _expected_framework = "sklearn"

            def __init__(self, config: ModelConfig) -> None:
                """Initialize the mock model.

                Args:
                    config: Model configuration
                """
                super().__init__(config)

        # Create config with wrong framework
        config = ModelConfig(
            model_name="framework_specific_model",
            model_type="regression",
            framework="tensorflow",  # Wrong framework
        )

        # Should raise ValueError due to framework mismatch
        with pytest.raises(ValueError, match="expects framework sklearn"):
            ModelFactory.create("framework_specific_model", config)

    @patch('backtester.core.logger.BacktesterLogger')
    def test_factory_error_handling(self, mock_logger: Mock) -> None:
        """Test factory error handling."""
        # Test ValueError for unknown model
        config = ModelConfig(
            model_name="unknown",
            model_type="regression",
            framework="sklearn",
        )

        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("unknown", config)

        # Test ValueError for unsupported framework
        ModelFactory.register_model("test_model")

        with pytest.raises(ValueError, match="Unsupported framework"):
            ModelFactory.create_with_framework("unsupported", "test_model", config)

    def test_convenience_functions(self) -> None:
        """Test convenience functions."""
        # Import convenience functions
        from backtester.model.model_factory import (
            create_model,
            get_available_frameworks,
            get_available_models,
            register_model,
        )

        # Register a test model
        @register_model("convenience_model")
        class ConvenienceModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                """Initialize the mock model.

                Args:
                    config: Model configuration
                """
                super().__init__(config)

            def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                return data, data['close']

            def train(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
                return {"accuracy": 0.9}

            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(features))

            def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
                return [
                    {"signal_type": "HOLD", "action": "Hold", "confidence": 0.5, "metadata": {}}
                ]

        # Test register_model function
        assert ModelFactory.is_registered("convenience_model")

        # Test create_model function
        config = ModelConfig(
            model_name="convenience_model",
            model_type="regression",
            framework="sklearn",
        )

        model = create_model("convenience_model", config)
        assert isinstance(model, ConvenienceModel)

        # Test get_available_models function
        models = get_available_models()
        assert "convenience_model" in str(models)

        # Test get_available_frameworks function
        frameworks = get_available_frameworks()
        assert isinstance(frameworks, list)

    def test_factory_singleton_behavior(self) -> None:
        """Test that factory maintains singleton behavior."""

        # Register a model in one instance
        @ModelFactory.register_model("singleton_model")
        class SingletonModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        # Get factory info from class method
        info1 = ModelFactory.get_factory_info()

        # Get factory info from global instance
        from backtester.model.model_factory import factory

        info2 = factory.get_factory_info()

        # Both should return the same registered models
        assert "singleton_model" in info1["registered_models"]
        assert "singleton_model" in info2["registered_models"]

    def test_adapter_registration_with_invalid_name(self) -> None:
        """Test that adapter registration works with any string name."""
        # This should work fine - the framework name is just a string identifier
        ModelFactory.register_adapter("", MockAdapter)  # Empty string
        ModelFactory.register_adapter("complex-name-123", MockAdapter)  # Complex name
        ModelFactory.register_adapter("123numeric", MockAdapter)  # Starting with number

        assert "" in ModelFactory._adapters
        assert "complex-name-123" in ModelFactory._adapters
        assert "123numeric" in ModelFactory._adapters

    def test_model_registration_decorator_return_value(self) -> None:
        """Test that the registration decorator returns the original class."""

        @ModelFactory.register_model("decorated_model")
        class DecoratedModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

        # The decorator should return the original class
        assert DecoratedModel is not None
        assert ModelFactory._models["decorated_model"] is DecoratedModel

    def test_multiple_registrations_same_name(self) -> None:
        """Test registering multiple classes with the same name."""
        first_model = self._create_mock_model("FirstModel")
        second_model = self._create_mock_model("SecondModel")

        # Register first model
        ModelFactory.register_model("duplicate_model")(first_model)

        # Register second model with same name - should overwrite
        ModelFactory.register_model("duplicate_model")(second_model)

        assert ModelFactory._models["duplicate_model"] is second_model

    def _create_mock_model(self, model_name: str) -> type[BaseModel[Any]]:
        """Create a mock model class for testing."""

        class MockModel(BaseModel[Any]):
            def __init__(self, config: ModelConfig) -> None:
                super().__init__(config)

            def prepare_data(self, data: Any) -> Any:
                raise NotImplementedError

            def train(self, features: Any, target: Any) -> Any:
                raise NotImplementedError

            def predict(self, features: Any) -> Any:
                raise NotImplementedError

            def generate_signals(self, data: Any) -> Any:
                raise NotImplementedError

        return MockModel

    def test_get_available_models_with_no_frameworks(self) -> None:
        """Test get_available_models when no frameworks are set."""
        # Clear adapters
        ModelFactory._adapters.clear()

        models = ModelFactory.get_available_models()

        # Should still return a dict, just without framework groups
        assert isinstance(models, dict)
