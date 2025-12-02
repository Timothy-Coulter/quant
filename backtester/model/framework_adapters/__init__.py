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

from __future__ import annotations

from typing import Any

from .scipy_adapter import SciPyAdapter
from .sklearn_adapter import SklearnAdapter

__all__ = ["SklearnAdapter", "SciPyAdapter"]
PyTorchAdapter: type[Any]

try:
    from . import pytorch_adapter as _pytorch_adapter
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _PYTORCH_IMPORT_ERROR = exc

    class _MissingPyTorchAdapter:
        """Placeholder that raises a helpful error when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "PyTorchAdapter requires the optional 'torch' dependency. "
                "Install PyTorch to enable this adapter."
            ) from _PYTORCH_IMPORT_ERROR

    PyTorchAdapter = _MissingPyTorchAdapter
else:
    PyTorchAdapter = _pytorch_adapter.PyTorchAdapter
    __all__.append("PyTorchAdapter")
