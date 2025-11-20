"""Results analysis and reporting for optimization.

This module provides functionality for analyzing optimization results,
generating reports, and exporting data from optimization studies.
"""

import logging
from datetime import datetime
from typing import Any

from backtester.optmisation.runner import OptimizationResult


class ResultsAnalyzer:
    """Analyzes and reports on optimization results.

    This class provides comprehensive analysis of optimization results
    including statistics, comparisons, and visualizations.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the results analyzer.

        Args:
            logger: Logger instance
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def analyze_optimization_result(
        self,
        result: OptimizationResult,
        backtest_engine: Any,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1mo",
    ) -> dict[str, Any]:
        """Analyze optimization result with backtest validation.

        Args:
            result: Optimization result to analyze
            backtest_engine: BacktestEngine instance for validation
            ticker: Trading symbol
            start_date: Start date for validation
            end_date: End date for validation
            interval: Data interval

        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Analyzing optimization result")

        # Basic statistics
        analysis = {
            'optimization_summary': self._get_optimization_summary(result),
            'parameter_analysis': self._analyze_parameters(result),
            'performance_analysis': self._analyze_performance_metrics(result),
            'trial_analysis': self._analyze_trial_results(result),
            'validation_backtest': self._run_validation_backtest(
                result, backtest_engine, ticker, start_date, end_date, interval
            ),
            'recommendations': self._generate_recommendations(result),
        }

        return analysis

    def _get_optimization_summary(self, result: OptimizationResult) -> dict[str, Any]:
        """Get optimization summary statistics.

        Args:
            result: Optimization result

        Returns:
            Dictionary containing optimization summary
        """
        return {
            'study_name': result.study_summary.get('study_name', 'Unknown'),
            'optimization_direction': result.study_summary.get('direction', 'Unknown'),
            'total_trials': result.n_trials,
            'complete_trials': result.trial_statistics.get('complete_trials', 0),
            'completion_rate': result.trial_statistics.get('completion_rate', 0.0),
            'optimization_time_seconds': result.optimization_time,
            'best_value': result.best_value,
            'best_parameters': result.best_params,
            'optimization_efficiency': (
                result.n_trials / result.optimization_time if result.optimization_time > 0 else 0
            ),
        }

    def _analyze_parameters(self, result: OptimizationResult) -> dict[str, Any]:
        """Analyze the best parameters found.

        Args:
            result: Optimization result

        Returns:
            Dictionary containing parameter analysis
        """
        best_params = result.best_params
        if not best_params:
            return {'error': 'No parameters available for analysis'}

        # Categorize parameters by type
        strategy_params = {}
        portfolio_params = {}
        other_params = {}

        for key, value in best_params.items():
            if key in [
                'ma_short',
                'ma_long',
                'leverage_base',
                'leverage_alpha',
                'base_to_alpha_split',
                'alpha_to_base_split',
                'stop_loss_base',
                'stop_loss_alpha',
                'take_profit_target',
            ]:
                strategy_params[key] = value
            elif key in [
                'initial_capital',
                'commission_rate',
                'slippage_std',
                'maintenance_margin',
            ]:
                portfolio_params[key] = value
            else:
                other_params[key] = value

        # Parameter statistics (from trial statistics)
        trial_stats = result.trial_statistics
        if 'parameter_names' in trial_stats:
            # This would require more detailed trial data
            pass

        return {
            'best_parameters': best_params,
            'strategy_parameters': strategy_params,
            'portfolio_parameters': portfolio_params,
            'other_parameters': other_params,
            'parameter_count': len(best_params),
            'parameter_importance': self._estimate_parameter_importance(result),
        }

    def _analyze_performance_metrics(self, result: OptimizationResult) -> dict[str, Any]:
        """Analyze performance metrics from the best trial.

        Args:
            result: Optimization result

        Returns:
            Dictionary containing performance analysis
        """
        if result.best_trial is None:
            return {'error': 'No best trial available for analysis'}

        # Get the best trial's user attributes if available
        best_trial = result.best_trial

        # Extract any stored metrics from the trial
        metrics = {}
        if hasattr(best_trial, 'user_attrs') and best_trial.user_attrs:
            metrics = best_trial.user_attrs

        # Calculate metric statistics across all trials
        trial_stats = result.trial_statistics

        analysis = {
            'best_trial_metrics': metrics,
            'metric_statistics': {
                'value_range': {
                    'min': trial_stats.get('min_value', 0),
                    'max': trial_stats.get('max_value', 0),
                    'mean': trial_stats.get('mean_value', 0),
                    'std': trial_stats.get('value_std', 0),
                },
                'distribution_quality': self._assess_distribution_quality(trial_stats),
            },
        }

        return analysis

    def _analyze_trial_results(self, result: OptimizationResult) -> dict[str, Any]:
        """Analyze trial results and patterns.

        Args:
            result: Optimization result

        Returns:
            Dictionary containing trial analysis
        """
        trial_stats = result.trial_statistics

        analysis = {
            'trial_outcomes': {
                'total_trials': trial_stats.get('total_trials', 0),
                'successful_trials': trial_stats.get('complete_trials', 0),
                'pruned_trials': trial_stats.get('pruned_trials', 0),
                'failed_trials': trial_stats.get('failed_trials', 0),
            },
            'trial_quality': {
                'success_rate': trial_stats.get('completion_rate', 0.0),
                'optimization_progress': self._assess_optimization_progress(trial_stats),
            },
        }

        return analysis

    def _run_validation_backtest(
        self,
        result: OptimizationResult,
        backtest_engine: Any,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> dict[str, Any]:
        """Run validation backtest with best parameters.

        Args:
            result: Optimization result
            backtest_engine: BacktestEngine instance
            ticker: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary containing validation results
        """
        try:
            # Run backtest with best parameters
            backtest_results = backtest_engine.run_backtest(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                strategy_params=result.best_params,
            )

            # Extract key metrics
            performance = backtest_results.get('performance', {})

            validation_result = {
                'backtest_successful': True,
                'performance_metrics': performance,
                'summary': {
                    'total_return': performance.get('total_return', 0.0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
                    'max_drawdown': performance.get('max_drawdown', 0.0),
                    'volatility': performance.get('volatility', 0.0),
                    'calmar_ratio': performance.get('calmar_ratio', 0.0),
                },
                'trading_statistics': {
                    'total_trades': performance.get('total_trades', 0),
                    'win_rate': performance.get('win_rate', 0.0),
                    'profit_factor': performance.get('profit_factor', 0.0),
                },
            }

            self.logger.info("Validation backtest completed successfully")
            return validation_result

        except Exception as e:
            self.logger.error(f"Validation backtest failed: {e}")
            return {
                'backtest_successful': False,
                'error': str(e),
            }

    def _generate_recommendations(self, result: OptimizationResult) -> dict[str, Any]:
        """Generate recommendations based on optimization results.

        Args:
            result: Optimization result

        Returns:
            Dictionary containing recommendations
        """
        recommendations = []

        # Analyze optimization quality
        completion_rate = result.trial_statistics.get('completion_rate', 0.0)
        if completion_rate < 0.8:
            recommendations.append(
                {
                    'type': 'optimization_quality',
                    'priority': 'high',
                    'message': 'Low trial completion rate. Consider simplifying parameter space or fixing constraint violations.',
                }
            )

        # Analyze optimization time
        if result.optimization_time > 300:  # 5 minutes
            recommendations.append(
                {
                    'type': 'performance',
                    'priority': 'medium',
                    'message': 'Long optimization time. Consider reducing n_trials or using more efficient samplers.',
                }
            )

        # Analyze best value
        if (
            result.best_value is not None
            and result.study_summary.get('direction') == 'maximize'
            and result.best_value < 0
        ):
            recommendations.append(
                {
                    'type': 'objective_function',
                    'priority': 'high',
                    'message': 'Negative best value. Review objective function and parameter constraints.',
                }
            )

        # Analyze parameter stability
        param_count = len(result.best_params)
        if param_count < 3:
            recommendations.append(
                {
                    'type': 'parameter_space',
                    'priority': 'low',
                    'message': 'Limited parameter space. Consider expanding search for better optimization.',
                }
            )

        # Generate summary recommendations
        if result.trial_statistics.get('failed_trials', 0) > 0:
            recommendations.append(
                {
                    'type': 'robustness',
                    'priority': 'medium',
                    'message': 'Some trials failed. Investigate parameter ranges and backtest stability.',
                }
            )

        return {
            'recommendations': recommendations,
            'optimization_grade': self._grade_optimization(result),
            'next_steps': self._suggest_next_steps(result),
        }

    def _estimate_parameter_importance(self, result: OptimizationResult) -> dict[str, float]:
        """Estimate parameter importance based on optimization results.

        Args:
            result: Optimization result

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        # This is a simplified estimation - in practice, you'd want to
        # analyze the actual trial data to determine parameter importance
        best_params = result.best_params

        if not best_params:
            return {}

        # Simple heuristic based on parameter values
        importance = {}
        for param_name, param_value in best_params.items():
            # Higher importance for parameters with values away from defaults
            if param_name == 'leverage_base':
                importance[param_name] = min(abs(param_value - 1.0) / 5.0, 1.0)
            elif param_name == 'leverage_alpha':
                importance[param_name] = min(abs(param_value - 3.0) / 7.0, 1.0)
            elif 'split' in param_name:
                importance[param_name] = abs(param_value - 0.5) * 2
            elif 'loss' in param_name or 'profit' in param_name:
                importance[param_name] = param_value * 10
            else:
                importance[param_name] = 0.5  # Default moderate importance

        return importance

    def _assess_distribution_quality(self, trial_stats: dict[str, Any]) -> str:
        """Assess the quality of the value distribution.

        Args:
            trial_stats: Trial statistics

        Returns:
            Quality assessment string
        """
        mean_value = trial_stats.get('mean_value', 0)
        value_std = trial_stats.get('value_std', 0)

        if value_std == 0:
            return 'degenerate'
        elif abs(mean_value) < 0.01:
            return 'poor_dispersion'
        else:
            coefficient_of_variation = value_std / abs(mean_value)
            if coefficient_of_variation < 0.1:
                return 'low_dispersion'
            elif coefficient_of_variation > 2.0:
                return 'high_dispersion'
            else:
                return 'good_dispersion'

    def _assess_optimization_progress(self, trial_stats: dict[str, Any]) -> str:
        """Assess the optimization progress quality.

        Args:
            trial_stats: Trial statistics

        Returns:
            Progress assessment string
        """
        completion_rate = trial_stats.get('completion_rate', 0.0)
        total_trials = trial_stats.get('total_trials', 0)

        if total_trials < 10:
            return 'insufficient_trials'
        elif completion_rate < 0.5:
            return 'poor_progress'
        elif completion_rate < 0.8:
            return 'moderate_progress'
        else:
            return 'good_progress'

    def _grade_optimization(self, result: OptimizationResult) -> str:
        """Grade the overall optimization quality.

        Args:
            result: Optimization result

        Returns:
            Grade string
        """
        completion_rate = result.trial_statistics.get('completion_rate', 0.0)
        total_trials = result.n_trials

        # Simple grading logic
        if completion_rate >= 0.9 and total_trials >= 50:
            return 'A'
        elif completion_rate >= 0.8 and total_trials >= 30:
            return 'B'
        elif completion_rate >= 0.7 and total_trials >= 20:
            return 'C'
        elif completion_rate >= 0.5:
            return 'D'
        else:
            return 'F'

    def _suggest_next_steps(self, result: OptimizationResult) -> list[str]:
        """Suggest next steps based on optimization results.

        Args:
            result: Optimization result

        Returns:
            List of suggested next steps
        """
        suggestions = []

        # Based on trial statistics
        completion_rate = result.trial_statistics.get('completion_rate', 0.0)
        if completion_rate < 0.8:
            suggestions.append("Investigate and fix constraint violations")
            suggestions.append("Review parameter ranges for合理性")

        # Based on optimization time
        if result.optimization_time > 300:
            suggestions.append("Consider using more efficient optimization algorithms")
            suggestions.append("Reduce the number of trials or use early stopping")

        # Based on best value
        if result.best_value is not None:
            direction = result.study_summary.get('direction', 'maximize')
            if direction == 'maximize' and result.best_value < 0:
                suggestions.append("Review objective function - all values are negative")
            elif direction == 'minimize' and result.best_value > 1000:
                suggestions.append("Review objective function - values are very large")

        # General suggestions
        suggestions.append("Run validation backtests with best parameters")
        suggestions.append("Consider multi-objective optimization if needed")
        suggestions.append("Analyze parameter importance for model interpretation")

        return suggestions

    def export_analysis_report(self, analysis: dict[str, Any], format: str = "markdown") -> str:
        """Export analysis report in various formats.

        Args:
            analysis: Analysis results
            format: Export format ('markdown', 'json', 'html')

        Returns:
            Report string in requested format
        """
        if format == "markdown":
            return self._create_markdown_report(analysis)
        elif format == "json":
            import json

            return json.dumps(analysis, indent=2, default=str)
        elif format == "html":
            return self._create_html_report(analysis)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _create_markdown_report(self, analysis: dict[str, Any]) -> str:
        """Create markdown report from analysis.

        Args:
            analysis: Analysis results

        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Optimization Analysis Report")
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Optimization Summary
        opt_summary = analysis.get('optimization_summary', {})
        lines.append("## Optimization Summary")
        lines.append(f"- Study Name: {opt_summary.get('study_name', 'Unknown')}")
        lines.append(f"- Direction: {opt_summary.get('optimization_direction', 'Unknown')}")
        lines.append(f"- Total Trials: {opt_summary.get('total_trials', 0)}")
        lines.append(f"- Completion Rate: {opt_summary.get('completion_rate', 0):.2%}")
        lines.append(f"- Best Value: {opt_summary.get('best_value', 'N/A')}")
        lines.append(
            f"- Optimization Time: {opt_summary.get('optimization_time_seconds', 0):.2f} seconds"
        )
        lines.append("")

        # Best Parameters
        param_analysis = analysis.get('parameter_analysis', {})
        best_params = param_analysis.get('best_parameters', {})
        if best_params:
            lines.append("## Best Parameters")
            for param, value in best_params.items():
                lines.append(f"- {param}: {value}")
            lines.append("")

        # Performance Analysis
        perf_analysis = analysis.get('performance_analysis', {})
        if perf_analysis and 'validation_backtest' in perf_analysis:
            validation = perf_analysis['validation_backtest']
            if validation.get('backtest_successful'):
                lines.append("## Validation Backtest Results")
                summary = validation.get('summary', {})
                lines.append(f"- Total Return: {summary.get('total_return', 0):.2%}")
                lines.append(f"- Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
                lines.append(f"- Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
                lines.append(f"- Volatility: {summary.get('volatility', 0):.2%}")
                lines.append(f"- Calmar Ratio: {summary.get('calmar_ratio', 0):.3f}")
                lines.append("")

        # Recommendations
        recommendations = analysis.get('recommendations', {})
        recs = recommendations.get('recommendations', [])
        if recs:
            lines.append("## Recommendations")
            for rec in recs:
                lines.append(
                    f"- **{rec.get('priority', 'Unknown').title()} Priority**: {rec.get('message', '')}"
                )
            lines.append("")

        return "\n".join(lines)

    def _create_html_report(self, analysis: dict[str, Any]) -> str:
        """Create HTML report from analysis.

        Args:
            analysis: Analysis results

        Returns:
            HTML report string
        """
        # Simplified HTML report
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head><title>Optimization Analysis Report</title></head>")
        html.append("<body>")
        html.append("<h1>Optimization Analysis Report</h1>")
        html.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Add sections from markdown report
        markdown_report = self._create_markdown_report(analysis)
        # Basic conversion from markdown to HTML (simplified)
        for line in markdown_report.split('\n'):
            if line.startswith('# '):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('- '):
                html.append(f"<li>{line[2:]}</li>")
            else:
                if line.strip():
                    html.append(f"<p>{line}</p>")

        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)
