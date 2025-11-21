"""Optimization runner and result handling.

This module provides the main optimization runner that orchestrates
the entire optimization process using Optuna.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backtester.optmisation._optuna import require_optuna
from backtester.optmisation.base import BaseOptimization
from backtester.optmisation.objective import OptimizationObjective
from backtester.optmisation.parameter_space import OptimizationConfig, ParameterSpace
from backtester.optmisation.study_manager import OptunaStudyManager


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    best_params: dict[str, Any]
    best_value: float | None
    best_trial: Any | None
    n_trials: int
    optimization_time: float
    study_summary: dict[str, Any]
    trial_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_trial': (
                {
                    'number': self.best_trial.number if self.best_trial else None,
                    'value': self.best_trial.value if self.best_trial else None,
                    'params': self.best_trial.params if self.best_trial else None,
                }
                if self.best_trial
                else None
            ),
            'n_trials': self.n_trials,
            'optimization_time': self.optimization_time,
            'study_summary': self.study_summary,
            'trial_statistics': self.trial_statistics,
        }


class OptimizationRunner(BaseOptimization):
    """Main optimization runner that orchestrates the optimization process.

    This class brings together the study manager, parameter space,
    objective function, and runs the actual optimization.
    """

    def __init__(
        self,
        backtest_engine: Any,
        parameter_space: ParameterSpace,
        objective: OptimizationObjective,
        config: OptimizationConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the optimization runner.

        Args:
            backtest_engine: BacktestEngine instance
            parameter_space: Parameter search space
            objective: Optimization objective
            config: Optimization configuration
            logger: Logger instance
        """
        super().__init__(logger)
        self.backtest_engine = backtest_engine
        self.parameter_space = parameter_space
        self.objective = objective
        self.config = config

        # Create study manager
        self.study_manager = OptunaStudyManager(
            study_name=config.study_name,
            storage_url=config.get_storage_url(),
            direction=config.direction,
            logger=logger,
        )

        # Results storage
        self.optimization_result: OptimizationResult | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def optimize(
        self,
        n_trials: int | None = None,
        timeout: int | None = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> OptimizationResult:
        """Run the optimization process.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds for optimization
            n_jobs: Number of parallel jobs
            show_progress_bar: Whether to show progress bar

        Returns:
            OptimizationResult containing optimization results
        """
        # Use provided parameters or fall back to config
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout

        self.logger.info("Starting optimization process")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Timeout: {timeout} seconds")
        self.logger.info(f"Direction: {self.config.direction}")

        # Record start time
        self.start_time = time.time()

        try:
            # Create or load study
            study = self.study_manager.create_study(
                config=self.config,
                parameter_space=self.parameter_space,
                load_if_exists=True,
            )

            # Create objective function for Optuna
            optuna_objective = self._create_optuna_objective()

            # Handle different optimization types
            if self.objective.objective_type == "constrained":
                return self._run_constrained_optimization(
                    study, optuna_objective, n_trials, timeout, show_progress_bar
                )
            else:
                return self._run_standard_optimization(
                    study, optuna_objective, n_trials, timeout, show_progress_bar
                )

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
        finally:
            self.end_time = time.time()

    def _create_optuna_objective(self) -> Callable[[Any], float]:
        """Create objective function for Optuna.

        Returns:
            Objective function for Optuna
        """
        parameter_space = self.parameter_space
        objective = self.objective
        direction = self.config.direction

        def optuna_objective(trial: Any) -> float:
            """Objective function for Optuna trial.

            Args:
                trial: Optuna trial object

            Returns:
                Objective value to optimize
            """
            # Suggest parameters using the parameter space
            suggested_params = parameter_space.suggest_params(trial)

            # Validate parameters
            self.validate_params(suggested_params)

            # Run backtest with suggested parameters
            try:
                result = objective._run_backtest(suggested_params)

                # Calculate objective value using objective functions
                objective_functions = objective.get_objective_functions()
                if not objective_functions:
                    raise ValueError("No objective functions defined")

                # For single objective, return the first function result
                if len(objective_functions) == 1:
                    value: float = objective_functions[0](suggested_params)
                else:
                    # For multiple objectives, combine them
                    value = sum(func(suggested_params) for func in objective_functions)

                # Store result for later analysis
                objective.best_result = result
                result.value = value

                self.logger.debug(
                    f"Trial {trial.number}: value={value:.4f}, params={suggested_params}"
                )

                return value

            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                # Return a very poor value for failed trials
                if direction == "maximize":
                    return -1e10
                else:
                    return 1e10

        return optuna_objective

    def _run_standard_optimization(
        self,
        study: Any,
        optuna_objective: Any,
        n_trials: int,
        timeout: int | None,
        show_progress_bar: bool,
    ) -> OptimizationResult:
        """Run standard single or multi-objective optimization.

        Args:
            study: Optuna study
            optuna_objective: Objective function
            n_trials: Number of trials
            timeout: Timeout in seconds
            show_progress_bar: Whether to show progress bar

        Returns:
            OptimizationResult
        """
        self.logger.info("Running standard optimization")

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=show_progress_bar,
        )

        return self._create_optimization_result()

    def _run_constrained_optimization(
        self,
        study: Any,
        optuna_objective: Any,
        n_trials: int,
        timeout: int | None,
        show_progress_bar: bool,
    ) -> OptimizationResult:
        """Run constrained optimization.

        Args:
            study: Optuna study
            optuna_objective: Objective function
            n_trials: Number of trials
            timeout: Timeout in seconds
            show_progress_bar: Whether to show progress bar

        Returns:
            OptimizationResult
        """
        self.logger.info("Running constrained optimization")

        # Get constraint functions
        constraint_functions = self.objective.get_constraint_functions()

        if constraint_functions:
            # Create constraint function for Optuna
            def constraints(trial: Any) -> list[float]:
                params = self.parameter_space.suggest_params(trial)
                return [func(params) for func in constraint_functions]

            # Run optimization with constraints
            study.optimize(
                optuna_objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.config.n_jobs,
                show_progress_bar=show_progress_bar,
                gc_after_trial=True,
            )
        else:
            # Run as standard optimization if no constraints
            return self._run_standard_optimization(
                study, optuna_objective, n_trials, timeout, show_progress_bar
            )

        return self._create_optimization_result()

    def _create_optimization_result(self) -> OptimizationResult:
        """Create optimization result from completed study.

        Returns:
            OptimizationResult
        """
        study = self.study_manager.get_study()
        end_time = self.end_time or time.time()
        optimization_time = end_time - (self.start_time or end_time)

        # Get study information
        study_summary = self.study_manager.get_study_summary()
        trial_statistics = self.study_manager.get_trial_statistics()

        # Create result
        result = OptimizationResult(
            best_params=study.best_params if study.best_params else {},
            best_value=study.best_value,
            best_trial=study.best_trial,
            n_trials=trial_statistics.get("total_trials", 0),
            optimization_time=optimization_time,
            study_summary=study_summary,
            trial_statistics=trial_statistics,
        )

        self.optimization_result = result

        self.logger.info("Optimization completed")
        self.logger.info(f"Best value: {result.best_value}")
        self.logger.info(f"Best params: {result.best_params}")
        self.logger.info(f"Total trials: {result.n_trials}")
        self.logger.info(f"Optimization time: {result.optimization_time:.2f} seconds")

        return result

    def get_best_params(self) -> dict[str, Any]:
        """Get the best parameters found during optimization.

        Returns:
            Dictionary of best parameters
        """
        if self.optimization_result is not None:
            return self.optimization_result.best_params

        study = self.study_manager.get_study()
        return study.best_params if study.best_params else {}

    def get_best_value(self) -> float:
        """Get the best objective value found.

        Returns:
            Best objective value
        """
        if self.optimization_result is not None:
            return self.optimization_result.best_value or 0.0

        study = self.study_manager.get_study()
        return study.best_value or 0.0

    def get_study_name(self) -> str:
        """Get the name of the optimization study.

        Returns:
            Study name
        """
        return self.study_manager.study_name

    def get_optimization_result(self) -> OptimizationResult | None:
        """Get the complete optimization result.

        Returns:
            OptimizationResult or None if optimization hasn't completed
        """
        return self.optimization_result

    def run_final_validation(self, n_validation_trials: int = 10) -> dict[str, Any]:
        """Run final validation with best parameters.

        Args:
            n_validation_trials: Number of validation trials

        Returns:
            Validation results
        """
        best_params = self.get_best_params()
        if not best_params:
            raise ValueError("No best parameters available for validation")

        self.logger.info("Running final validation")
        validation_results: list[dict[str, Any]] = []

        for i in range(n_validation_trials):
            try:
                result = self.objective._run_backtest(best_params)
                validation_results.append(
                    {
                        'trial': i + 1,
                        'metrics': result.metrics,
                        'value': result.value,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Validation trial {i + 1} failed: {e}")

        # Calculate statistics
        if validation_results:
            metrics_values: dict[str, Any] = {}
            for metric in validation_results[0]['metrics']:
                values = [
                    r['metrics'][metric] for r in validation_results if metric in r['metrics']
                ]
                if values:
                    metrics_values[metric] = {
                        'mean': sum(values) / len(values),
                        'std': (
                            sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
                        )
                        ** 0.5,
                        'min': min(values),
                        'max': max(values),
                    }

            return {
                'n_validation_trials': len(validation_results),
                'parameter_stability': len(validation_results),
                'metrics_statistics': metrics_values,
                'validation_results': validation_results,
            }

        return {'error': 'No successful validation trials'}

    def plot_optimization_history(self, save_path: str | None = None) -> Any:
        """Plot optimization history and results.

        Args:
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt

            study = self.study_manager.get_study()
            optuna_module = require_optuna()
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Optuna Optimization Results', fontsize=16)

            # Plot optimization history
            optuna_module.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')

            # Plot parameter importance
            optuna_module.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
            axes[0, 1].set_title('Parameter Importance')

            # Plot parallel coordinates
            optuna_module.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1, 0])
            axes[1, 0].set_title('Parameter Relationships')

            # Plot contour
            if len(study.best_params) >= 2:
                param_names = list(study.best_params.keys())[:2]
                optuna_module.visualization.matplotlib.plot_contour(
                    study, params=param_names, ax=axes[1, 1]
                )
                axes[1, 1].set_title(f'Contour: {param_names[0]} vs {param_names[1]}')
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    'Not enough parameters\nfor contour plot',
                    ha='center',
                    va='center',
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title('Contour Plot')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Optimization plot saved to {save_path}")

            return fig

        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {e}")
            return None
