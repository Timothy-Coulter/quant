"""Parameter space definition and optimization configuration.

This module provides functionality for defining parameter search spaces,
validation rules, and optimization configurations for Optuna studies.
"""

import logging
from dataclasses import dataclass
from typing import Any

import optuna


@dataclass
class ParameterDefinition:
    """Definition of a single optimization parameter."""

    name: str
    param_type: str  # 'float', 'int', 'categorical', 'loguniform'
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None
    q: float | int | None = None  # Quantization step for float parameters

    def __post_init__(self) -> None:
        """Validate parameter definition."""
        if self.param_type not in ['float', 'int', 'categorical', 'loguniform']:
            raise ValueError(f"Invalid parameter type: {self.param_type}")

        if self.param_type in ['float', 'int', 'loguniform']:
            if self.low is None or self.high is None:
                raise ValueError(f"Low and high bounds required for {self.param_type} parameters")

            if self.low >= self.high:
                raise ValueError(f"Low must be less than high: {self.low} >= {self.high}")


class ParameterSpace:
    """Manages parameter search spaces for optimization."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize parameter space manager.

        Args:
            logger: Logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self._parameters: dict[str, ParameterDefinition] = {}

    def add_parameter(self, param_def: ParameterDefinition) -> 'ParameterSpace':
        """Add a parameter to the search space.

        Args:
            param_def: Parameter definition

        Returns:
            Self for method chaining
        """
        self._parameters[param_def.name] = param_def
        self.logger.debug(f"Added parameter: {param_def.name}")
        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
        q: float | None = None,
    ) -> 'ParameterSpace':
        """Add a float parameter to the search space.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            step: Step size (for uniform sampling)
            log: Whether to use log scale
            q: Quantization step

        Returns:
            Self for method chaining
        """
        param_def = ParameterDefinition(
            name=name,
            param_type='float',
            low=low,
            high=high,
            step=step,
            log=log,
            q=q,
        )
        return self.add_parameter(param_def)

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int | None = None,
    ) -> 'ParameterSpace':
        """Add an integer parameter to the search space.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            step: Step size (for uniform sampling)

        Returns:
            Self for method chaining
        """
        param_def = ParameterDefinition(
            name=name,
            param_type='int',
            low=low,
            high=high,
            step=step,
        )
        return self.add_parameter(param_def)

    def add_categorical(self, name: str, choices: list[Any]) -> 'ParameterSpace':
        """Add a categorical parameter to the search space.

        Args:
            name: Parameter name
            choices: List of possible choices

        Returns:
            Self for method chaining
        """
        param_def = ParameterDefinition(
            name=name,
            param_type='categorical',
            choices=choices,
        )
        return self.add_parameter(param_def)

    def add_loguniform(
        self,
        name: str,
        low: float,
        high: float,
        q: float | None = None,
    ) -> 'ParameterSpace':
        """Add a loguniform parameter to the search space.

        Args:
            name: Parameter name
            low: Lower bound (positive)
            high: Upper bound
            q: Quantization step

        Returns:
            Self for method chaining
        """
        if low <= 0:
            raise ValueError("Loguniform parameters must have positive lower bound")

        param_def = ParameterDefinition(
            name=name,
            param_type='loguniform',
            low=low,
            high=high,
            q=q,
        )
        return self.add_parameter(param_def)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameter values for an Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        for param_def in self._parameters.values():
            value = self._suggest_param_value(trial, param_def)
            if value is not None:
                params[param_def.name] = value

        return params

    def _suggest_param_value(
        self, trial: optuna.Trial, param_def: ParameterDefinition
    ) -> float | int | Any:
        """Suggest a single parameter value for an Optuna trial.

        Args:
            trial: Optuna trial object
            param_def: Parameter definition

        Returns:
            Suggested parameter value
        """
        if param_def.param_type == 'float':
            return self._suggest_float_param(trial, param_def)
        elif param_def.param_type == 'int':
            return self._suggest_int_param(trial, param_def)
        elif param_def.param_type == 'categorical':
            if param_def.choices is None:
                raise ValueError(f"Categorical parameter '{param_def.name}' requires choices")
            return trial.suggest_categorical(
                param_def.name,
                list(param_def.choices),
            )
        elif param_def.param_type == 'loguniform':
            return self._suggest_loguniform_param(trial, param_def)
        return None

    def _suggest_float_param(self, trial: optuna.Trial, param_def: ParameterDefinition) -> float:
        """Suggest a float parameter value.

        Args:
            trial: Optuna trial object
            param_def: Parameter definition

        Returns:
            Suggested float value
        """
        low = float(param_def.low) if param_def.low is not None else None
        high = float(param_def.high) if param_def.high is not None else None
        if low is None or high is None:
            raise ValueError(f"Float parameter '{param_def.name}' requires low and high bounds")
        if param_def.step is not None:
            value = trial.suggest_float(
                param_def.name,
                low,
                high,
                step=float(param_def.step),
            )
            return float(value)
        elif param_def.q is not None:
            value = trial.suggest_float(
                param_def.name,
                low,
                high,
                step=param_def.q,
            )
            return float(value)
        else:
            value = trial.suggest_float(
                param_def.name,
                low,
                high,
                log=param_def.log,
            )
            return float(value)

    def _suggest_int_param(self, trial: optuna.Trial, param_def: ParameterDefinition) -> int:
        """Suggest an integer parameter value.

        Args:
            trial: Optuna trial object
            param_def: Parameter definition

        Returns:
            Suggested integer value
        """
        low = int(param_def.low) if param_def.low is not None else None
        high = int(param_def.high) if param_def.high is not None else None
        if low is None or high is None:
            raise ValueError(f"Int parameter '{param_def.name}' requires low and high bounds")
        if param_def.step is not None:
            value = trial.suggest_int(
                param_def.name,
                low,
                high,
                step=int(param_def.step),
            )
            return int(value)
        else:
            value = trial.suggest_int(
                param_def.name,
                low,
                high,
            )
            return int(value)

    def _suggest_loguniform_param(
        self, trial: optuna.Trial, param_def: ParameterDefinition
    ) -> float:
        """Suggest a loguniform parameter value.

        Args:
            trial: Optuna trial object
            param_def: Parameter definition

        Returns:
            Suggested loguniform value
        """
        low = float(param_def.low) if param_def.low is not None else None
        high = float(param_def.high) if param_def.high is not None else None
        if low is None or high is None:
            raise ValueError(
                f"Loguniform parameter '{param_def.name}' requires low and high bounds"
            )
        value = trial.suggest_float(
            param_def.name,
            low,
            high,
            log=True,
        )
        return float(value)

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names in the search space.

        Returns:
            List of parameter names
        """
        return list(self._parameters.keys())

    def get_parameter_count(self) -> int:
        """Get the number of parameters in the search space.

        Returns:
            Number of parameters
        """
        return len(self._parameters)

    def create_grid_space(self) -> dict[str, list[Any]]:
        """Create a grid search space from the parameter definitions.

        Returns:
            Dictionary mapping parameter names to value lists
        """
        grid_space = {}
        for param_def in self._parameters.values():
            grid_space[param_def.name] = self._create_parameter_grid_values(param_def)
        return grid_space

    def _create_parameter_grid_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create grid values for a single parameter definition.

        Args:
            param_def: Parameter definition

        Returns:
            List of grid values for the parameter
        """
        if param_def.param_type == 'float':
            return self._create_float_grid_values(param_def)
        elif param_def.param_type == 'int':
            return self._create_int_grid_values(param_def)
        elif param_def.param_type == 'categorical':
            return self._create_categorical_grid_values(param_def)
        elif param_def.param_type == 'loguniform':
            return self._create_loguniform_grid_values(param_def)
        else:
            raise ValueError(f"Unknown parameter type: {param_def.param_type}")

    def _create_float_grid_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create grid values for float parameters."""
        if param_def.step is not None and param_def.low is not None and param_def.high is not None:
            return self._create_stepped_float_values(param_def)
        else:
            return self._create_range_float_values(param_def)

    def _create_stepped_float_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create stepped float values."""
        start = float(param_def.low) if param_def.low is not None else 0.0
        stop = float(param_def.high) if param_def.high is not None else 0.0
        step = float(param_def.step) if param_def.step is not None else 0.0
        if step <= 0:
            raise ValueError(f"Step must be positive for parameter '{param_def.name}'")

        values = []
        current = start
        while current < stop:
            values.append(current)
            current += step
        return values

    def _create_range_float_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create range float values."""
        if param_def.low is None or param_def.high is None:
            raise ValueError(
                f"Float parameter '{param_def.name}' requires low and high for grid space"
            )
        return [float(param_def.low), float(param_def.high)]

    def _create_int_grid_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create grid values for integer parameters."""
        if param_def.low is None or param_def.high is None:
            raise ValueError(
                f"Int parameter '{param_def.name}' requires low and high for grid space"
            )
        return list(range(int(param_def.low), int(param_def.high)))

    def _create_categorical_grid_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create grid values for categorical parameters."""
        if param_def.choices is None:
            raise ValueError(
                f"Categorical parameter '{param_def.name}' requires choices for grid space"
            )
        return list(param_def.choices)

    def _create_loguniform_grid_values(self, param_def: ParameterDefinition) -> list[Any]:
        """Create grid values for loguniform parameters."""
        if param_def.low is None or param_def.high is None:
            raise ValueError(
                f"Loguniform parameter '{param_def.name}' requires low and high for grid space"
            )
        return [float(param_def.low), float(param_def.high)]


class OptimizationConfig:
    """Configuration for optimization runs."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize optimization configuration.

        Args:
            logger: Logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # Default optimization settings
        self.n_trials: int = 100
        self.timeout: int | None = None
        self.n_jobs: int = 1
        self.show_progress_bar: bool = True

        # Default Optuna sampler settings
        self.sampler_name: str = 'tpe'
        self.sampler_kwargs: dict[str, Any] = {}

        # Storage settings
        self.storage_url: str | None = None
        self.storage_backend: str = 'sqlite'  # 'sqlite', 'postgresql'

        # Study settings
        self.direction: str = 'maximize'
        self.study_name: str | None = None
        self.load_if_exists: bool = True

    def set_trials(self, n_trials: int) -> 'OptimizationConfig':
        """Set number of trials.

        Args:
            n_trials: Number of trials

        Returns:
            Self for method chaining
        """
        self.n_trials = n_trials
        return self

    def set_timeout(self, timeout: int | None) -> 'OptimizationConfig':
        """Set optimization timeout.

        Args:
            timeout: Timeout in seconds

        Returns:
            Self for method chaining
        """
        self.timeout = timeout
        return self

    def set_parallel_jobs(self, n_jobs: int) -> 'OptimizationConfig':
        """Set number of parallel jobs.

        Args:
            n_jobs: Number of parallel jobs

        Returns:
            Self for method chaining
        """
        self.n_jobs = n_jobs
        return self

    def set_sampler(self, sampler_name: str, **kwargs: Any) -> 'OptimizationConfig':
        """Set Optuna sampler configuration.

        Args:
            sampler_name: Name of the sampler ('tpe', 'random', 'grid', etc.)
            **kwargs: Sampler-specific parameters

        Returns:
            Self for method chaining
        """
        self.sampler_name = sampler_name
        self.sampler_kwargs = kwargs
        return self

    def set_storage(self, storage_url: str | None) -> 'OptimizationConfig':
        """Set storage configuration.

        Args:
            storage_url: Storage URL (None for in-memory SQLite)

        Returns:
            Self for method chaining
        """
        self.storage_url = storage_url
        return self

    def set_study_name(self, study_name: str) -> 'OptimizationConfig':
        """Set study name.

        Args:
            study_name: Study name

        Returns:
            Self for method chaining
        """
        self.study_name = study_name
        return self

    def set_direction(self, direction: str) -> 'OptimizationConfig':
        """Set optimization direction.

        Args:
            direction: 'maximize' or 'minimize'

        Returns:
            Self for method chaining
        """
        if direction not in ['maximize', 'minimize']:
            raise ValueError(f"Direction must be 'maximize' or 'minimize': {direction}")
        self.direction = direction
        return self

    def get_sampler(self) -> optuna.samplers.BaseSampler:
        """Get configured Optuna sampler.

        Returns:
            Configured sampler instance
        """
        sampler_map = {
            'tpe': optuna.samplers.TPESampler,
            'random': optuna.samplers.RandomSampler,
            'grid': optuna.samplers.GridSampler,
            'cmaes': optuna.samplers.CmaEsSampler,
            'nsgaii': optuna.samplers.NSGAIISampler,
        }

        if self.sampler_name not in sampler_map:
            raise ValueError(f"Unknown sampler: {self.sampler_name}")

        sampler_class = sampler_map[self.sampler_name]

        if self.sampler_name == 'grid':
            raise ValueError(
                "Grid sampler requires search space - use ParameterSpace.create_grid_space()"
            )

        sampler: optuna.samplers.BaseSampler = sampler_class(**self.sampler_kwargs)
        return sampler

    def get_storage_url(self) -> str | None:
        """Get storage URL with defaults applied.

        Returns:
            Storage URL
        """
        if self.storage_url:
            return self.storage_url

        if self.storage_backend == 'sqlite':
            return None  # In-memory SQLite
        elif self.storage_backend == 'postgresql':
            # Default PostgreSQL configuration
            return 'postgresql://localhost:5432/optuna_db'

        return None
