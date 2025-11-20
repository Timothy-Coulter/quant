"""Model Factory for creating and managing model instances.

This module provides a factory pattern for creating model instances by name,
following the established architectural patterns in the backtester framework.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError

from backtester.model.base_model import BaseModel
from backtester.model.model_configs import (
    ModelConfig,
    PyTorchModelConfig,
    SciPyModelConfig,
    SklearnModelConfig,
)

# TypeVar for models
ModelType = TypeVar('ModelType')


class ModelFactory:
    """Factory for creating model instances with framework adapters.

    This factory pattern allows for easy registration and creation of model
    instances by name, following the established architectural patterns.
    """

    _models: dict[str, type[BaseModel[Any]]] = {}
    _adapters: dict[str, Any] = {}

    @classmethod
    def register_model(cls, name: str) -> Callable[[type[BaseModel[Any]]], type[BaseModel[Any]]]:
        """Register a model class with the factory.

        Args:
            name: Name to register the model under

        Returns:
            Decorator function for registering the model class
        """

        def decorator(model_class: type[BaseModel[Any]]) -> type[BaseModel[Any]]:
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def register_adapter(cls, framework: str, adapter_class: type[Any]) -> None:
        """Register a framework adapter with the factory.

        Args:
            framework: Framework name (sklearn, tensorflow, scipy, pytorch)
            adapter_class: Adapter class for the framework
        """
        cls._adapters[framework] = adapter_class

    @classmethod
    def create(cls, name: str, config: ModelConfig) -> BaseModel[Any]:
        """Create a model instance by name.

        Args:
            name: Name of the model to create
            config: Configuration for the model

        Returns:
            Model instance

        Raises:
            ValueError: If the model name is not registered
            ValueError: If configuration doesn't match framework requirements
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available models: {available}")

        model_class = cls._models[name]

        # Validate that config framework matches model's expected framework
        # This is a simplified check - in practice, you'd want more sophisticated validation
        if (
            hasattr(model_class, '_expected_framework')
            and config.framework != model_class._expected_framework
        ):
            raise ValueError(
                f"Model {name} expects framework {model_class._expected_framework}, "
                f"but config specifies {config.framework}"
            )

        return model_class(config)

    @classmethod
    def create_with_framework(
        cls, framework: str, model_name: str, config: ModelConfig
    ) -> BaseModel[Any]:
        """Create a model instance using framework-specific factory method.

        Args:
            framework: ML framework name
            model_name: Name of the model within the framework
            config: Configuration for the model

        Returns:
            Model instance

        Raises:
            ValueError: If framework is not supported or adapter not found
        """
        if framework not in cls._adapters:
            supported_frameworks = list(cls._adapters.keys())
            raise ValueError(
                f"Unsupported framework: {framework}. Supported frameworks: {supported_frameworks}"
            )

        adapter_class = cls._adapters[framework]

        # Use the framework-specific factory method
        try:
            result = adapter_class.create_model(model_name, config)
            if isinstance(result, BaseModel):
                return result
            else:
                # Fallback to generic factory if adapter doesn't have proper create_model method
                return cls.create(model_name, config)
        except AttributeError:
            # Fallback to generic factory if adapter doesn't have create_model method
            return cls.create(model_name, config)

    @classmethod
    def get_available_models(cls) -> dict[str, list[str]]:
        """Get list of available models grouped by framework.

        Returns:
            Dictionary with framework -> list of model names mapping
        """
        models_by_framework: dict[str, list[str]] = {}

        for model_name, model_class in cls._models.items():
            # Try to determine framework from model class
            framework = getattr(model_class, '_expected_framework', 'unknown')

            if framework not in models_by_framework:
                models_by_framework[framework] = []
            models_by_framework[framework].append(model_name)

        return models_by_framework

    @classmethod
    def get_available_frameworks(cls) -> list[str]:
        """Get list of available frameworks.

        Returns:
            List of available framework names
        """
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if the model is registered
        """
        return name in cls._models

    @classmethod
    def is_framework_supported(cls, framework: str) -> bool:
        """Check if a framework is supported.

        Args:
            framework: Name of the framework to check

        Returns:
            True if the framework is supported
        """
        return framework in cls._adapters

    @classmethod
    def create_from_config_dict(cls, config_dict: dict[str, Any]) -> BaseModel[Any]:
        """Create a model instance from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Model instance

        Raises:
            ValueError: If configuration is invalid or model not found
        """
        try:
            # Determine the appropriate config class based on framework
            framework = config_dict.get('framework')
            if framework == 'sklearn':
                config: SklearnModelConfig | SciPyModelConfig | PyTorchModelConfig | ModelConfig = (
                    SklearnModelConfig(**config_dict)
                )
            elif framework == 'scipy':
                config = SciPyModelConfig(**config_dict)
            elif framework == 'pytorch':
                config = PyTorchModelConfig(**config_dict)
            else:
                config = ModelConfig(**config_dict)

            # Get model name from config
            model_name = config.model_name

            return cls.create(model_name, config)

        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to create model: {e}") from e

    @classmethod
    def get_factory_info(cls) -> dict[str, Any]:
        """Get comprehensive factory information.

        Returns:
            Dictionary with factory information including registered models and frameworks
        """
        return {
            'registered_models': list(cls._models.keys()),
            'supported_frameworks': list(cls._adapters.keys()),
            'models_by_framework': cls.get_available_models(),
            'total_models': len(cls._models),
            'total_frameworks': len(cls._adapters),
        }


class ModelRegistrationError(Exception):
    """Exception raised when model registration fails."""


class ModelCreationError(Exception):
    """Exception raised when model creation fails."""


# Global factory instance for convenience
factory = ModelFactory()


# Convenience functions for easy access
def register_model(name: str) -> Callable[[type[BaseModel[Any]]], type[BaseModel[Any]]]:
    """Register a model with the global factory.

    Args:
        name: Name to register the model under

    Returns:
        Decorator function for registering the model class
    """
    return factory.register_model(name)


def create_model(name: str, config: ModelConfig) -> BaseModel[Any]:
    """Create a model instance using the global factory.

    Args:
        name: Name of the model to create
        config: Configuration for the model

    Returns:
        Model instance
    """
    return factory.create(name, config)


def get_available_models() -> dict[str, list[str]]:
    """Get available models grouped by framework.

    Returns:
        Dictionary with framework -> list of model names mapping
    """
    return factory.get_available_models()


def get_available_frameworks() -> list[str]:
    """Get available frameworks.

    Returns:
        List of available framework names
    """
    return factory.get_available_frameworks()
