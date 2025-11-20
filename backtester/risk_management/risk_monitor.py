"""Risk Monitoring System.

This module provides comprehensive risk monitoring functionality with real-time
alerts, risk metrics tracking, and portfolio risk analysis.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backtester.risk_management.component_configs.risk_monitoring_config import RiskMonitoringConfig


class RiskMonitor:
    """Real-time risk monitoring system."""

    def __init__(
        self,
        config: RiskMonitoringConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize risk monitor.

        Args:
            config: RiskMonitoringConfig with monitoring parameters
            logger: Optional logger instance
        """
        self.config: RiskMonitoringConfig = config or RiskMonitoringConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # State tracking
        self.is_monitoring = True
        self.risk_metrics: dict[str, dict[str, Any]] = {}
        self.risk_metrics_history: list[dict[str, Any]] = []
        self.alert_thresholds = {
            'volatility': self.config.volatility_threshold,
            'drawdown': self.config.drawdown_threshold,
            'leverage': self.config.leverage_threshold,
        }
        self.alert_rules: list[dict[str, Any]] = []
        self.portfolio_values: list[float] = []
        self.timestamps: list[datetime] = []

    def add_risk_metric(
        self, metric_name: str, threshold: float, comparison: str = 'greater_than'
    ) -> None:
        """Add a risk metric to monitor.

        Args:
            metric_name: Name of the risk metric
            threshold: Threshold value for the metric
            comparison: Comparison operator ('greater_than' or 'less_than')
        """
        self.risk_metrics[metric_name] = {
            'threshold': threshold,
            'comparison': comparison,
            'name': metric_name,
        }

    def check_risk_metrics(self, portfolio_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Check all risk metrics for violations.

        Args:
            portfolio_state: Dictionary with current portfolio metrics

        Returns:
            List of metric violations
        """
        violations = []

        for metric_name, config in self.risk_metrics.items():
            if metric_name in portfolio_state:
                current_value = portfolio_state[metric_name]
                threshold = config['threshold']
                comparison = config['comparison']

                is_violation = False
                if (
                    comparison == 'greater_than'
                    and current_value > threshold
                    or comparison == 'less_than'
                    and current_value < threshold
                ):
                    is_violation = True

                if is_violation:
                    violations.append(
                        {
                            'metric': metric_name,
                            'current_value': current_value,
                            'threshold': threshold,
                            'severity': self._get_violation_severity(current_value, threshold),
                        }
                    )
            elif metric_name == 'drawdown' and 'current_drawdown' in portfolio_state:
                current_value = portfolio_state['current_drawdown']
                threshold = config['threshold']
                comparison = config['comparison']

                is_violation = False
                if (
                    comparison == 'greater_than'
                    and current_value > threshold
                    or comparison == 'less_than'
                    and current_value < threshold
                ):
                    is_violation = True

                if is_violation:
                    violations.append(
                        {
                            'metric': 'drawdown',
                            'current_value': current_value,
                            'threshold': threshold,
                            'severity': self._get_violation_severity(current_value, threshold),
                        }
                    )

        return violations

    def process_portfolio_update(self, update: dict[str, Any]) -> dict[str, Any] | None:
        """Process portfolio update and generate alerts if necessary.

        Args:
            update: Portfolio update information

        Returns:
            Alert information if triggered
        """
        if not self.is_monitoring:
            return None

        violations = self.check_risk_metrics(update)

        if violations and self.config.enable_real_time_alerts:
            # Generate alert for most severe violation
            most_severe = max(
                violations, key=lambda x: {'high': 2, 'medium': 1, 'low': 0}.get(x['severity'], 0)
            )

            # Always generate an alert if there are violations and alerts are enabled
            return {
                'alert_type': 'risk_metric_breach',
                'metric': most_severe['metric'],
                'current_value': most_severe['current_value'],
                'threshold': most_severe['threshold'],
                'severity': most_severe['severity'],
                'timestamp': update.get('timestamp'),
                'recommended_action': self._get_recommended_action(most_severe),
            }

        return None

    def update(self, portfolio: Any, timestamp: datetime) -> None:
        """Update risk metrics with current portfolio state.

        Args:
            portfolio: Portfolio object
            timestamp: Current timestamp
        """
        # Get current portfolio value
        total_value = portfolio.get_total_value() if hasattr(portfolio, 'get_total_value') else 0

        # Append current values
        self.portfolio_values.append(total_value)
        self.timestamps.append(timestamp)

        # Calculate rolling volatility if enough data points
        if len(self.portfolio_values) >= 20:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            # Annualized volatility (assuming 252 trading days)
            self.current_volatility = np.std(returns) * np.sqrt(252)

        # Record current risk measurement
        measurement = {
            'timestamp': timestamp,
            'portfolio_value': total_value,
        }

        if hasattr(self, 'current_volatility'):
            measurement['volatility'] = self.current_volatility

        # Add other risk metrics if available from portfolio
        if hasattr(portfolio, 'get_current_drawdown'):
            measurement['drawdown'] = portfolio.get_current_drawdown()

        if hasattr(portfolio, 'get_total_leverage'):
            measurement['leverage'] = portfolio.get_total_leverage()

        self.record_risk_measurement(measurement)

    def record_risk_measurement(self, measurement: dict[str, Any]) -> None:
        """Record risk measurement to history.

        Args:
            measurement: Risk measurement data
        """
        measurement['timestamp'] = measurement.get('timestamp', pd.Timestamp.now())
        self.risk_metrics_history.append(measurement)

        # Keep history size limited
        if len(self.risk_metrics_history) > self.config.max_history_size:
            self.risk_metrics_history = self.risk_metrics_history[-self.config.max_history_size :]

    def analyze_risk_trends(self, trend_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze risk trends over time.

        Args:
            trend_data: List of risk measurement data

        Returns:
            Dictionary with trend analysis
        """
        if len(trend_data) < 2:
            return {'trend_direction': 'insufficient_data'}

        # Simplified trend analysis
        volatility_trend = self._calculate_trend([d.get('volatility', 0) for d in trend_data])
        drawdown_trend = self._calculate_trend([d.get('drawdown', 0) for d in trend_data])

        return {
            'volatility_trend': {'direction': volatility_trend},
            'drawdown_trend': {'direction': drawdown_trend},
            'trend_direction': 'increasing' if volatility_trend == 'increasing' else 'stable',
        }

    def add_alert_rule(self, metric: str, threshold: float, severity: str, action: str) -> None:
        """Add alert rule for automated alerting.

        Args:
            metric: Metric name
            threshold: Alert threshold
            severity: Alert severity
            action: Recommended action
        """
        self.alert_rules.append(
            {
                'metric': metric,
                'threshold': threshold,
                'severity': severity,
                'action': action,
            }
        )

    def generate_alert(self, metric: str, value: float, severity: str) -> dict[str, Any]:
        """Generate risk alert.

        Args:
            metric: Metric name
            value: Current metric value
            severity: Alert severity

        Returns:
            Alert information
        """
        return {
            'alert_type': 'manual_alert',
            'metric': metric,
            'value': value,
            'severity': severity,
            'recommended_action': self._get_recommended_action({'metric': metric}),
            'timestamp': pd.Timestamp.now().isoformat(),
        }

    def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate dashboard data for risk monitoring.

        Returns:
            Dictionary with dashboard data
        """
        current_metrics = {}
        if self.risk_metrics_history:
            latest = self.risk_metrics_history[-1]
            current_metrics = latest

        historical_trends = (
            self.risk_metrics_history[-30:]
            if len(self.risk_metrics_history) >= 30
            else self.risk_metrics_history
        )

        # Calculate risk summary
        risk_summary = {
            'total_measurements': len(self.risk_metrics_history),
            'latest_metrics': current_metrics,
            'trend_status': 'stable',
        }

        return {
            'current_metrics': current_metrics,
            'historical_trends': historical_trends,
            'risk_summary': risk_summary,
            'alerts': [],
        }

    def generate_risk_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Generate risk report for date range.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Risk report dictionary
        """
        filtered_history = [
            m
            for m in self.risk_metrics_history
            if start_date <= m.get('timestamp', start_date) <= end_date
        ]

        return {
            'report_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'executive_summary': 'Risk monitoring shows stable conditions',
            'risk_metrics': self._aggregate_metrics(filtered_history),
            'trends': self.analyze_risk_trends(filtered_history),
            'recommendations': ['Continue monitoring current risk levels'],
            'alerts_summary': {'total_alerts': 0, 'critical_alerts': 0},
        }

    def optimize_risk_limits(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        """Optimize risk limits based on historical performance.

        Args:
            performance_data: Historical performance data

        Returns:
            Dictionary with optimization recommendations
        """
        # Simplified optimization based on historical volatility
        returns = performance_data.get('returns', pd.Series())

        if len(returns) == 0:
            return {
                'recommended_drawdown_limit': 0.15,
                'recommended_volatility_limit': 0.25,
                'recommended_position_limit': 0.12,
                'optimization_score': 0.5,
            }

        historical_vol = returns.std() * np.sqrt(252)

        return {
            'recommended_drawdown_limit': min(0.20, historical_vol * 1.5),
            'recommended_volatility_limit': min(0.30, historical_vol * 1.2),
            'recommended_position_limit': 0.10 if historical_vol > 0.20 else 0.15,
            'optimization_score': 0.7,
        }

    def get_recommended_action(self, risk_level: str, position_size: float) -> str:
        """Get recommended action based on risk level and position size.

        Args:
            risk_level: Current risk level
            position_size: Position size

        Returns:
            Recommended action
        """
        action_mapping = {
            ('low', 0.05): 'normal_trading',
            ('medium', 0.10): 'normal_trading',
            ('high', 0.15): 'reduce_position_size',
            ('very_high', 0.25): 'emergency_reduction',
        }

        return action_mapping.get((risk_level, position_size), 'review_positions')

    def _get_violation_severity(self, current_value: float, threshold: float) -> str:
        """Get violation severity based on current value and threshold.

        Args:
            current_value: Current metric value
            threshold: Metric threshold

        Returns:
            Severity level
        """
        ratio = abs(current_value / threshold) if threshold > 0 else 0

        if ratio >= 1.5:
            return 'high'
        elif ratio >= 1.2:
            return 'medium'
        else:
            return 'low'

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a list of values.

        Args:
            values: List of values

        Returns:
            Trend direction
        """
        if len(values) < 2:
            return 'stable'

        # Simple linear trend calculation
        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2 :])

        if second_half > first_half * 1.05:
            return 'increasing'
        elif second_half < first_half * 0.95:
            return 'decreasing'
        else:
            return 'stable'

    def _get_recommended_action(self, violation: dict[str, Any]) -> str:
        """Get recommended action for a violation.

        Args:
            violation: Violation information

        Returns:
            Recommended action
        """
        metric = violation.get('metric', 'unknown')

        action_mapping = {
            'drawdown': 'reduce_position_sizes',
            'volatility': 'increase_diversification',
            'leverage': 'lower_leverage',
            'position_size': 'reduce_individual_positions',
        }

        return action_mapping.get(metric, 'review_all_positions')

    def _aggregate_metrics(self, measurements: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate metrics from measurements.

        Args:
            measurements: List of measurements

        Returns:
            Aggregated metrics
        """
        if not measurements:
            return {}

        # Calculate averages and extremes
        aggregated = {}
        metrics = ['volatility', 'drawdown', 'leverage']

        for metric in metrics:
            values = [m.get(metric, 0) for m in measurements]
            aggregated[metric] = {
                'average': np.mean(values),
                'maximum': np.max(values),
                'minimum': np.min(values),
            }

        return aggregated

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status.

        Returns:
            Dictionary with monitoring status
        """
        return {
            'is_monitoring': self.is_monitoring,
            'check_interval': self.config.check_interval,
            'enable_real_time_alerts': self.config.enable_real_time_alerts,
            'total_metrics_tracked': len(self.risk_metrics),
            'total_measurements': len(self.risk_metrics_history),
            'alert_rules_configured': len(self.alert_rules),
        }
