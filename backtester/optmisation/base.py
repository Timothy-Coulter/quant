"""Base classes and types for optimization module.

This module provides the foundational classes and enums used across
the optimization system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OptimizationType(Enum):
    """Types of optimization approaches."""

    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINED = "constrained"


class OptimizationDirection(Enum):
    """Optimization direction types."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class OptimizationMetadata:
    """Metadata for optimization runs."""

    study_name: str
    optimization_type: OptimizationType
    direction: OptimizationDirection
    n_trials: int
    start_time: str
    parameters_count: int
    dataset_info: dict[str, Any] | None = None


class BaseOptimization(ABC):
    """Abstract base class for optimization implementations.

    This class defines the interface that all optimization implementations
    must follow, providing a consistent API for optimization operations.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the base optimization.

        Args:
            logger: Logger instance for recording optimization events
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def optimize(
        self,
        n_trials: int,
        timeout: int | None = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> Any:
        """Run the optimization process.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds for optimization
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar

        Returns:
            Optimization results
        """
        pass

    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """Get the best parameters found during optimization.

        Returns:
            Dictionary of best parameters
        """
        pass

    @abstractmethod
    def get_best_value(self) -> float:
        """Get the best objective value found.

        Returns:
            Best objective value
        """
        pass

    @abstractmethod
    def get_study_name(self) -> str:
        """Get the name of the optimization study.

        Returns:
            Study name
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Validate optimization parameters.

        Args:
            params: Parameters to validate

        Returns:
            True if parameters are valid

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary")

        required_types = (int, float, str, bool)
        for key, value in params.items():
            if not isinstance(key, str):
                raise ValueError(f"Parameter key must be string: {key}")
            if not isinstance(value, required_types):
                raise ValueError(f"Parameter value must be one of {required_types}: {key}={value}")

        return True

    def get_optimization_info(self) -> dict[str, Any]:
        """Get information about the optimization setup.

        Returns:
            Dictionary containing optimization information
        """
        return {
            'study_name': self.get_study_name(),
            'best_params': self.get_best_params(),
            'best_value': self.get_best_value(),
            'logger_level': self.logger.level,
        }
