"""Risk Management System.

This module provides comprehensive risk management functionality including
stop-loss, take-profit, position sizing, risk limits, monitoring, and metrics calculation.
"""

from __future__ import annotations

# Configuration classes
from backtester.risk_management.component_configs.comprehensive_risk_config import (
    ComprehensiveRiskConfig,
)
from backtester.risk_management.component_configs.position_sizing_config import PositionSizingConfig
from backtester.risk_management.component_configs.risk_limit_config import RiskLimitConfig
from backtester.risk_management.component_configs.risk_monitoring_config import RiskMonitoringConfig
from backtester.risk_management.component_configs.stop_loss_config import StopLossConfig
from backtester.risk_management.component_configs.take_profit_config import TakeProfitConfig

# Core risk management classes
from backtester.risk_management.position_sizing import PositionSizer
from backtester.risk_management.risk_limits import RiskLimits
from backtester.risk_management.risk_metrics_calculator import RiskMetricsCalculator
from backtester.risk_management.risk_monitor import RiskMonitor
from backtester.risk_management.risk_signals import (
    RiskAction,
    RiskAlert,
    RiskLimit,
    RiskMetric,
    RiskSignal,
)
from backtester.risk_management.stop_loss import StopLoss
from backtester.risk_management.take_profit import TakeProfit

__all__ = [
    # Configuration classes
    'ComprehensiveRiskConfig',
    'PositionSizingConfig',
    'RiskLimitConfig',
    'RiskMonitoringConfig',
    'StopLossConfig',
    'TakeProfitConfig',
    # Core classes
    'PositionSizer',
    'RiskControlManager',
    'RiskLimits',
    'RiskMetricsCalculator',
    'RiskMonitor',
    'StopLoss',
    'TakeProfit',
    # Signal and alert classes
    'RiskAction',
    'RiskAlert',
    'RiskLimit',
    'RiskMetric',
    'RiskSignal',
]


def __getattr__(name: str) -> object:
    """Lazy-load heavy modules to avoid circular imports during package init."""
    if name == "RiskControlManager":
        from backtester.risk_management.risk_control_manager import RiskControlManager

        return RiskControlManager
    raise AttributeError(name)
