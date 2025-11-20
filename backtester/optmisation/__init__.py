"""Optuna Optimization Module for the Backtester.

This module provides comprehensive optimization functionality using Optuna
for hyperparameter tuning and strategy optimization.
"""

from backtester.optmisation.base import BaseOptimization, OptimizationDirection, OptimizationType
from backtester.optmisation.objective import ObjectiveResult, OptimizationObjective
from backtester.optmisation.parameter_space import OptimizationConfig, ParameterSpace
from backtester.optmisation.results import ResultsAnalyzer
from backtester.optmisation.runner import OptimizationResult, OptimizationRunner
from backtester.optmisation.study_manager import OptunaStudyManager

__all__ = [
    'OptunaStudyManager',
    'OptimizationObjective',
    'ObjectiveResult',
    'ParameterSpace',
    'OptimizationConfig',
    'OptimizationRunner',
    'OptimizationResult',
    'ResultsAnalyzer',
    'BaseOptimization',
    'OptimizationType',
    'OptimizationDirection',
]
