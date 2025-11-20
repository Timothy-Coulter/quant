"""Optimization objective functions and results.

This module provides functionality for defining optimization objectives,
handling objective results, and integrating with the backtesting engine.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backtester.core.backtest_engine import BacktestEngine


@dataclass
class ObjectiveResult:
    """Container for objective function results."""

    value: float
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    trial_number: int | None = None
    duration: float | None = None
    constraint_values: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            'value': self.value,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'trial_number': self.trial_number,
            'duration': self.duration,
            'constraint_values': self.constraint_values,
        }


class OptimizationObjective:
    """Objective function for backtest optimization.

    This class encapsulates the objective function that Optuna will use
    to evaluate different parameter combinations during optimization.
    """

    def __init__(
        self,
        backtest_engine: BacktestEngine,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1mo",
        objective_type: str = "single",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the optimization objective.

        Args:
            backtest_engine: BacktestEngine instance to use for evaluations
            ticker: Trading symbol to optimize for
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval
            objective_type: Type of objective ('single', 'multi', 'constrained')
            logger: Logger instance
        """
        self.backtest_engine = backtest_engine
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.objective_type = objective_type
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # Load data once during initialization
        self.logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
        self.data = self.backtest_engine.load_data(ticker, start_date, end_date, interval)

        # Objective function components
        self.objective_functions: list[Callable[[dict[str, Any]], float]] = []
        self.constraint_functions: list[Callable[[dict[str, Any]], float]] = []

        # Result tracking
        self.best_result: ObjectiveResult | None = None
        self.trial_count = 0

    def add_objective_function(
        self, func: Callable[[dict[str, Any]], float], weight: float = 1.0
    ) -> 'OptimizationObjective':
        """Add an objective function component.

        Args:
            func: Objective function that takes parameters and returns a value
            weight: Weight for combining with other objective functions

        Returns:
            Self for method chaining
        """
        self.objective_functions.append(lambda p: func(p) * weight)
        return self

    def add_constraint_function(
        self, func: Callable[[dict[str, Any]], float]
    ) -> 'OptimizationObjective':
        """Add a constraint function.

        Args:
            func: Constraint function that takes parameters and returns constraint value

        Returns:
            Self for method chaining
        """
        self.constraint_functions.append(func)
        return self

    def set_single_objective(
        self,
        metric: str = "total_return",
        penalty_factor: float = 50.0,
        constraint_metric: str | None = "max_drawdown",
        constraint_limit: float = -0.20,
    ) -> 'OptimizationObjective':
        """Set up a single objective optimization.

        Args:
            metric: Primary metric to optimize
            penalty_factor: Factor for penalty-based constraints
            constraint_metric: Metric to use for penalty constraints
            constraint_limit: Maximum allowed value for constraint metric

        Returns:
            Self for method chaining
        """

        def single_objective(params: dict[str, Any]) -> float:
            result = self._run_backtest(params)
            primary_value: float = result.metrics.get(metric, 0.0)

            # Apply penalty constraints if specified
            if constraint_metric and constraint_metric in result.metrics:
                constraint_value = result.metrics[constraint_metric]
                if constraint_value < constraint_limit:
                    penalty = abs(constraint_value - constraint_limit) * penalty_factor
                    return float(primary_value - penalty)

            return float(primary_value)

        self.objective_functions = [single_objective]
        self.objective_type = "single"
        return self

    def set_multi_objective(
        self, metrics: list[str], weights: list[float] | None = None
    ) -> 'OptimizationObjective':
        """Set up multi-objective optimization.

        Args:
            metrics: List of metrics to optimize
            weights: Optional weights for each metric

        Returns:
            Self for method chaining
        """
        if weights is None:
            weights = [1.0] * len(metrics)

        if len(metrics) != len(weights):
            raise ValueError("Metrics and weights must have the same length")

        def multi_objective(params: dict[str, Any]) -> float:
            result = self._run_backtest(params)
            total_value = 0.0
            for metric, weight in zip(metrics, weights, strict=True):
                value = result.metrics.get(metric, 0.0)
                total_value += value * weight
            return total_value

        self.objective_functions = [multi_objective]
        self.objective_type = "multi"
        return self

    def set_constrained_objective(
        self,
        objective_metric: str = "total_return",
        constraint_functions: list[Callable[[dict[str, Any]], float]] | None = None,
    ) -> 'OptimizationObjective':
        """Set up constrained objective optimization.

        Args:
            objective_metric: Primary objective metric
            constraint_functions: List of constraint functions

        Returns:
            Self for method chaining
        """

        def constrained_objective(params: dict[str, Any]) -> float:
            result = self._run_backtest(params)
            return float(result.metrics.get(objective_metric, 0.0))

        self.objective_functions = [constrained_objective]
        if constraint_functions:
            self.constraint_functions = constraint_functions

        self.objective_type = "constrained"
        return self

    def _run_backtest(self, params: dict[str, Any]) -> ObjectiveResult:
        """Run backtest with given parameters.

        Args:
            params: Parameter dictionary

        Returns:
            ObjectiveResult containing backtest results
        """
        self.trial_count += 1

        # Create a copy of parameters for strategy/portfolio configuration
        strategy_params = {
            k: v
            for k, v in params.items()
            if k
            in [
                'ma_short',
                'ma_long',
                'leverage_base',
                'leverage_alpha',
                'base_to_alpha_split',
                'alpha_to_base_split',
                'stop_loss_base',
                'stop_loss_alpha',
                'take_profit_target',
            ]
        }

        portfolio_params = {
            k: v
            for k, v in params.items()
            if k in ['initial_capital', 'commission_rate', 'slippage_std']
        }

        # Run backtest
        try:
            results = self.backtest_engine.run_backtest(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
                strategy_params=strategy_params,
                portfolio_params=portfolio_params,
            )

            # Extract metrics
            metrics = results.get('performance', {})
            if not metrics:
                metrics = {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

            objective_result = ObjectiveResult(
                value=0.0,  # Will be calculated by objective functions
                metrics=metrics,
                parameters=params,
                trial_number=self.trial_count,
            )

            self.logger.debug(f"Trial {self.trial_count}: {params}")
            self.logger.debug(f"Trial {self.trial_count} metrics: {metrics}")

            return objective_result

        except Exception as e:
            self.logger.error(f"Error in trial {self.trial_count}: {e}")
            # Return a poor-performing result for failed trials
            return ObjectiveResult(
                value=-1000.0,  # Very poor score
                metrics={'total_return': -1.0, 'sharpe_ratio': -10.0, 'max_drawdown': -1.0},
                parameters=params,
                trial_number=self.trial_count,
            )

    def get_objective_functions(self) -> list[Callable[[dict[str, Any]], float]]:
        """Get the list of objective functions.

        Returns:
            List of objective functions
        """
        return self.objective_functions

    def get_constraint_functions(self) -> list[Callable[[dict[str, Any]], float]]:
        """Get the list of constraint functions.

        Returns:
            List of constraint functions
        """
        return self.constraint_functions

    def get_best_result(self) -> ObjectiveResult | None:
        """Get the best result seen so far.

        Returns:
            Best result or None if no trials completed
        """
        return self.best_result

    def get_trial_count(self) -> int:
        """Get the number of trials completed.

        Returns:
            Trial count
        """
        return self.trial_count


def create_optimization_objective(
    backtest_engine: BacktestEngine,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1mo",
    objective_type: str = "single",
    logger: logging.Logger | None = None,
) -> OptimizationObjective:
    """Create and configure an optimization objective.

    Args:
        backtest_engine: BacktestEngine instance
        ticker: Trading symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
        objective_type: Type of objective
        logger: Logger instance

    Returns:
        Configured OptimizationObjective instance
    """
    objective = OptimizationObjective(
        backtest_engine=backtest_engine,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        objective_type=objective_type,
        logger=logger,
    )

    # Set default objectives based on type
    if objective_type == "single":
        objective.set_single_objective()
    elif objective_type == "multi":
        objective.set_multi_objective(["total_return", "sharpe_ratio"])
    elif objective_type == "constrained":
        objective.set_constrained_objective()

    return objective
