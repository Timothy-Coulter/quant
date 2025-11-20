"""Model System for the Backtester Framework.

This module provides a comprehensive machine learning model system that integrates
seamlessly with the backtesting framework, supporting multiple ML frameworks including
scikit-learn, TensorFlow, SciPy, and PyTorch.

The model system follows the established architectural patterns:
- Abstract base classes with clear interfaces
- Pydantic-based configuration models
- Factory pattern for model instantiation
- Framework adapters for compatibility
- Signal generation integration
"""

# Model implementations
from . import sklearn_models
from .base_model import BaseModel, ModelFrameworkAdapter
from .model_configs import (
    ModelConfig,
    PyTorchModelConfig,
    SciPyModelConfig,
    SklearnModelConfig,
)
from .model_factory import ModelFactory

__all__ = [
    "BaseModel",
    "ModelFrameworkAdapter",
    "ModelConfig",
    "SklearnModelConfig",
    "SciPyModelConfig",
    "PyTorchModelConfig",
    "ModelFactory",
    "sklearn_models",
]

# Model Factory Registration
_model_factory = ModelFactory()

# Auto-register built-in models
try:
    from .framework_adapters import PyTorchAdapter, SciPyAdapter, SklearnAdapter

    _model_factory.register_adapter("sklearn", SklearnAdapter)
    _model_factory.register_adapter("scipy", SciPyAdapter)
    _model_factory.register_adapter("pytorch", PyTorchAdapter)
except ImportError:
    # Framework not available, skip registration
    pass
