"""Framework Adapters for ML Model Integration.

This module provides adapters for integrating various machine learning frameworks
with the backtester model system. Each adapter handles framework-specific
implementation details while maintaining a consistent interface.

Supported frameworks:
- scikit-learn: Traditional ML algorithms
- TensorFlow: Deep learning and neural networks
- SciPy: Scientific computing and statistical models
- PyTorch: Dynamic neural networks and deep learning
"""

from .pytorch_adapter import PyTorchAdapter
from .scipy_adapter import SciPyAdapter
from .sklearn_adapter import SklearnAdapter

__all__ = [
    "SklearnAdapter",
    "SciPyAdapter",
    "PyTorchAdapter",
]
