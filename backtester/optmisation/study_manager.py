"""Optuna study management and configuration.

This module provides functionality for creating, managing, and configuring
Optuna studies for backtest optimization.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from backtester.optmisation._optuna import require_optuna
from backtester.optmisation.base import OptimizationMetadata
from backtester.optmisation.parameter_space import OptimizationConfig, ParameterSpace


class OptunaStudyManager:
    """Manages Optuna studies for backtest optimization.

    This class handles study creation, configuration, and management
    of Optuna optimization studies.
    """

    def __init__(
        self,
        study_name: str | None = None,
        storage_url: str | None = None,
        direction: str = "maximize",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the study manager.

        Args:
            study_name: Name of the study (auto-generated if None)
            storage_url: Storage URL for study persistence
            direction: Optimization direction ('maximize' or 'minimize')
            logger: Logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.storage_url = storage_url
        self.direction = direction
        self._study: Any | None = None

        # Generate study name if not provided
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study_name = f"backtest_optimization_{timestamp}"
        else:
            self.study_name = study_name

        self.logger.info(f"Initialized study manager with name: {self.study_name}")

    def create_study(
        self,
        config: OptimizationConfig | None = None,
        parameter_space: ParameterSpace | None = None,
        load_if_exists: bool = True,
    ) -> Any:
        """Create a new Optuna study.

        Args:
            config: Optimization configuration
            parameter_space: Parameter search space
            load_if_exists: Whether to load existing study

        Returns:
            Created Optuna study
        """
        if config is None:
            config = OptimizationConfig(logger=self.logger)

        if config.study_name is None:
            config.set_study_name(self.study_name)

        if config.storage_url is None:
            config.set_storage(self.storage_url)

        # Create sampler
        sampler = config.get_sampler()

        # Create study
        try:
            optuna_module = require_optuna()
            self._study = optuna_module.create_study(
                study_name=config.study_name,
                direction=config.direction,
                storage=config.get_storage_url(),
                sampler=sampler,
                load_if_exists=load_if_exists,
            )

            self.logger.info(f"Created study: {self.study_name}")
            self.logger.info(f"Direction: {config.direction}")
            self.logger.info(f"Storage: {config.get_storage_url() or 'In-memory'}")

            return self._study

        except Exception as e:
            self.logger.error(f"Failed to create study: {e}")
            raise

    def load_study(
        self,
        config: OptimizationConfig | None = None,
    ) -> Any:
        """Load an existing Optuna study.

        Args:
            config: Optimization configuration

        Returns:
            Loaded Optuna study
        """
        if config is None:
            config = OptimizationConfig(logger=self.logger)

        if config.study_name is None:
            config.set_study_name(self.study_name)

        try:
            storage_url = config.get_storage_url()
            if storage_url:
                optuna_module = require_optuna()
                self._study = optuna_module.load_study(
                    study_name=config.study_name,
                    storage=storage_url,
                )
            else:
                # In-memory studies cannot be re-loaded without a storage URL; raise an error.
                raise ValueError(
                    "Cannot load study without a storage URL. "
                    "Configure OptimizationConfig.storage_url to use persistent storage."
                )

            self.logger.info(f"Loaded existing study: {self.study_name}")
            self.logger.info(f"Trials completed: {len(self._study.trials)}")

            return self._study
        except Exception as e:
            self.logger.error(f"Failed to load study: {e}")
            raise

    def get_study(self) -> Any:
        """Get the current study instance.

        Returns:
            Current study instance

        Raises:
            ValueError: If study hasn't been created or loaded
        """
        if self._study is None:
            raise ValueError(
                "Study not created or loaded. Call create_study() or load_study() first."
            )

        return self._study

    def delete_study(self) -> None:
        """Delete the current study.

        This is a destructive operation that removes all trial data.
        """
        if self._study is None:
            self.logger.warning("No study to delete")
            return

        try:
            if self.storage_url:
                optuna_module = require_optuna()
                optuna_module.delete_study(
                    study_name=self.study_name,
                    storage=self.storage_url,
                )
            else:
                raise ValueError(
                    "Cannot delete study without a storage URL. "
                    "Configure OptimizationConfig.storage_url for persistent studies."
                )

            self.logger.info(f"Deleted study: {self.study_name}")
            self._study = None
        except Exception as e:
            self.logger.error(f"Failed to delete study: {e}")
            raise

    def get_study_summary(self) -> dict[str, Any]:
        """Get summary of the current study.

        Returns:
            Dictionary containing study summary
        """
        if self._study is None:
            return {"error": "No study available"}

        summary = {
            "study_name": self._study.study_name,
            "direction": self._study.direction,
            "n_trials": len(self._study.trials),
            "best_trial": None,
            "best_params": None,
            "best_value": None,
        }

        if self._study.best_trial is not None:
            summary["best_trial"] = {
                "number": self._study.best_trial.number,
                "value": self._study.best_trial.value,
                "params": self._study.best_trial.params,
                "datetime_complete": self._study.best_trial.datetime_complete,
            }
            summary["best_params"] = self._study.best_params
            summary["best_value"] = self._study.best_value

        return summary

    def get_trial_statistics(self) -> dict[str, Any]:
        """Get statistics about trials in the study.

        Returns:
            Dictionary containing trial statistics
        """
        if self._study is None:
            return {"error": "No study available"}

        trials = self._study.trials

        if not trials:
            return {"total_trials": 0}

        values = [trial.value for trial in trials if trial.value is not None]
        optuna_module = require_optuna()
        complete_trials = [
            trial for trial in trials if trial.state == optuna_module.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            trial for trial in trials if trial.state == optuna_module.trial.TrialState.PRUNED
        ]
        failed_trials = [
            trial for trial in trials if trial.state == optuna_module.trial.TrialState.FAIL
        ]

        stats = {
            "total_trials": len(trials),
            "complete_trials": len(complete_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_value": self._study.best_value if self._study.best_value is not None else None,
            "completion_rate": len(complete_trials) / len(trials) if trials else 0.0,
        }

        if values:
            stats.update(
                {
                    "min_value": min(values),
                    "max_value": max(values),
                    "mean_value": sum(values) / len(values),
                    "value_std": (
                        sum((v - stats.get("mean_value", 0)) ** 2 for v in values) / len(values)
                    )
                    ** 0.5,
                }
            )

        return stats

    def export_study_data(self, format: str = "dataframe") -> Any:
        """Export study data in various formats.

        Args:
            format: Export format ('dataframe', 'json', 'csv')

        Returns:
            Study data in requested format
        """
        if self._study is None:
            raise ValueError("No study available to export")

        if format == "dataframe":
            return self._study.trials_dataframe()
        elif format == "json":
            import json

            trials_data = []
            for trial in self._study.trials:
                trial_dict = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                    "datetime_start": (
                        trial.datetime_start.isoformat() if trial.datetime_start else None
                    ),
                    "datetime_complete": (
                        trial.datetime_complete.isoformat() if trial.datetime_complete else None
                    ),
                }
                trials_data.append(trial_dict)
            return json.dumps(trials_data, indent=2)
        elif format == "csv":
            df = self._study.trials_dataframe()
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_optimization_metadata(self) -> OptimizationMetadata:
        """Get metadata about the optimization.

        Returns:
            OptimizationMetadata instance
        """
        from backtester.optmisation.base import OptimizationDirection, OptimizationType

        trial_stats = self.get_trial_statistics()

        return OptimizationMetadata(
            study_name=self.study_name,
            optimization_type=OptimizationType.SINGLE_OBJECTIVE,  # Default, can be enhanced
            direction=OptimizationDirection(self.direction),
            n_trials=trial_stats.get("total_trials", 0),
            start_time=datetime.now().isoformat(),
            parameters_count=(
                len(self._study.best_params) if self._study and self._study.best_params else 0
            ),
        )

    def get_best_parameters(self) -> dict[str, Any]:
        """Get the best parameters found in the study.

        Returns:
            Dictionary of best parameters
        """
        if self._study is None:
            raise ValueError("No study available")

        best_params = self._study.best_params
        return dict(best_params)

    def get_best_value(self) -> float | None:
        """Get the best objective value found.

        Returns:
            Best objective value or None
        """
        if self._study is None:
            raise ValueError("No study available")

        best_value = self._study.best_value
        return float(best_value) if best_value is not None else None

    def get_best_trial(self) -> Any | None:
        """Get the best trial from the study.

        Returns:
            Best trial or None
        """
        if self._study is None:
            raise ValueError("No study available")

        return self._study.best_trial
