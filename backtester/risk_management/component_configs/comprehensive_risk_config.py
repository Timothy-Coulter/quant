"""Comprehensive Risk Management Configuration.

This module provides the main risk management configuration model that combines
all individual risk management configurations into a unified interface.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .position_sizing_config import PositionSizingConfig
from .risk_limit_config import RiskLimitConfig
from .risk_monitoring_config import RiskMonitoringConfig
from .stop_loss_config import StopLossConfig
from .take_profit_config import TakeProfitConfig


class ComprehensiveRiskConfig(BaseModel):
    """Comprehensive risk management configuration using pydantic BaseModel.

    This class combines all individual risk management configurations into a
    unified interface, inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Individual configuration components
    stop_loss_config: StopLossConfig | None = Field(
        default=None, description="Stop loss configuration"
    )
    take_profit_config: TakeProfitConfig | None = Field(
        default=None, description="Take profit configuration"
    )
    position_sizing_config: PositionSizingConfig | None = Field(
        default=None, description="Position sizing configuration"
    )
    risk_limits_config: RiskLimitConfig | None = Field(
        default=None, description="Risk limits configuration"
    )
    risk_monitoring_config: RiskMonitoringConfig | None = Field(
        default=None, description="Risk monitoring configuration"
    )

    # Legacy compatibility fields
    max_portfolio_risk: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Maximum portfolio risk (legacy compatibility)"
    )
    max_position_size: float = Field(
        default=0.10, ge=0.01, le=1.0, description="Maximum position size (legacy compatibility)"
    )
    max_leverage: float = Field(
        default=5.0, ge=1.0, le=20.0, description="Maximum leverage (legacy compatibility)"
    )
    max_drawdown: float = Field(
        default=0.20, ge=0.05, le=1.0, description="Maximum drawdown (legacy compatibility)"
    )
    stop_loss_pct: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Stop loss percentage (legacy compatibility)"
    )
    take_profit_pct: float = Field(
        default=0.06, ge=0.001, le=1.0, description="Take profit percentage (legacy compatibility)"
    )
    max_daily_loss: float = Field(
        default=0.05, ge=0.01, le=0.5, description="Maximum daily loss (legacy compatibility)"
    )
    volatility_threshold: float = Field(
        default=0.03, ge=0.01, le=0.5, description="Volatility threshold (legacy compatibility)"
    )
    correlation_limit: float = Field(
        default=0.7, ge=0.1, le=1.0, description="Correlation limit (legacy compatibility)"
    )

    # Advanced risk management settings
    enable_dynamic_hedging: bool = Field(
        default=False, description="Whether to enable dynamic hedging"
    )
    stress_test_frequency: str = Field(
        default="monthly", description="Frequency for stress testing"
    )
    rebalance_frequency: str = Field(
        default="weekly", description="Rebalancing frequency for risk management"
    )

    # Risk attribution and analysis
    risk_attribution_enabled: bool = Field(
        default=True, description="Whether to enable risk attribution"
    )
    factor_analysis_enabled: bool = Field(
        default=False, description="Whether to enable factor analysis"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ComprehensiveRiskConfig with default configurations if not provided."""
        super().__init__(**data)

        # Track which configs were explicitly provided
        self._explicit_configs = {
            'stop_loss_config': 'stop_loss_config' in data,
            'take_profit_config': 'take_profit_config' in data,
            'position_sizing_config': 'position_sizing_config' in data,
            'risk_limits_config': 'risk_limits_config' in data,
            'risk_monitoring_config': 'risk_monitoring_config' in data,
        }

        # Initialize default configurations only if they were not explicitly provided
        # in the input data AND are currently None
        self._initialize_default_config('stop_loss_config', StopLossConfig)
        self._initialize_default_config('take_profit_config', TakeProfitConfig)
        self._initialize_default_config('position_sizing_config', PositionSizingConfig)
        self._initialize_default_config('risk_limits_config', RiskLimitConfig)
        self._initialize_default_config('risk_monitoring_config', RiskMonitoringConfig)

    def _initialize_default_config(self, config_name: str, config_class: type) -> None:
        """Initialize a default configuration if not explicitly provided."""
        if not self._explicit_configs[config_name] and getattr(self, config_name) is None:
            setattr(self, config_name, config_class())

    def get_effective_risk_limits(self) -> dict[str, float]:
        """Get effective risk limits from the risk limits configuration."""
        # Check if risk_limits_config was explicitly provided (not auto-generated)
        if (
            self._explicit_configs.get('risk_limits_config', False)
            and self.risk_limits_config is not None
        ):
            return {
                'max_drawdown': self.risk_limits_config.max_drawdown,
                'max_leverage': self.risk_limits_config.max_leverage,
                'max_single_position': self.risk_limits_config.max_single_position,
                'max_portfolio_var': self.risk_limits_config.max_portfolio_var,
                'max_daily_loss': self.max_daily_loss,
                'max_correlation': self.risk_limits_config.max_correlation,
            }

        # Fallback to legacy fields
        return {
            'max_drawdown': self.max_drawdown,
            'max_leverage': self.max_leverage,
            'max_single_position': self.max_position_size,
            'max_portfolio_var': 0.05,  # Default VaR
            'max_daily_loss': self.max_daily_loss,
            'max_correlation': self.correlation_limit,
        }

    def get_position_sizing_params(self) -> dict[str, Any]:
        """Get position sizing parameters."""
        # Check if position_sizing_config was explicitly provided (not auto-generated)
        if (
            self._explicit_configs.get('position_sizing_config', False)
            and self.position_sizing_config is not None
        ):
            return {
                'max_position_size': self.position_sizing_config.max_position_size,
                'min_position_size': self.position_sizing_config.min_position_size,
                'risk_per_trade': self.position_sizing_config.risk_per_trade,
                'sizing_method': self.position_sizing_config.sizing_method,
                'volatility_adjustment': self.position_sizing_config.volatility_adjustment,
                'max_daily_trades': self.position_sizing_config.max_daily_trades,
            }

        # Fallback to legacy fields
        return {
            'max_position_size': self.max_position_size,
            'min_position_size': 0.01,  # Default min
            'risk_per_trade': 0.02,  # Default risk per trade
            'sizing_method': 'fixed_percentage',
            'volatility_adjustment': True,
            'max_daily_trades': 5,
        }

    def get_stop_loss_params(self) -> dict[str, Any]:
        """Get stop loss parameters."""
        # Check if stop_loss_config was explicitly provided (not auto-generated)
        if (
            self._explicit_configs.get('stop_loss_config', False)
            and self.stop_loss_config is not None
        ):
            return {
                'stop_loss_type': self.stop_loss_config.stop_loss_type,
                'stop_loss_value': self.stop_loss_config.stop_loss_value,
                'trail_distance': self.stop_loss_config.trail_distance,
                'trail_step': self.stop_loss_config.trail_step,
                'max_loss_value': self.stop_loss_config.max_loss_value,
            }

        # Fallback to legacy fields
        return {
            'stop_loss_type': 'PERCENTAGE',
            'stop_loss_value': self.stop_loss_pct,
            'trail_distance': 0.01,  # Default trail distance
            'trail_step': 0.005,  # Default trail step
            'max_loss_value': None,
        }

    def get_take_profit_params(self) -> dict[str, Any]:
        """Get take profit parameters."""
        # Check if take_profit_config was explicitly provided (not auto-generated)
        if (
            self._explicit_configs.get('take_profit_config', False)
            and self.take_profit_config is not None
        ):
            return {
                'take_profit_type': self.take_profit_config.take_profit_type,
                'take_profit_value': self.take_profit_config.take_profit_value,
                'trail_distance': self.take_profit_config.trail_distance,
                'trail_step': self.take_profit_config.trail_step,
                'max_gain_value': self.take_profit_config.max_gain_value,
            }

        # Fallback to legacy fields
        return {
            'take_profit_type': 'PERCENTAGE',
            'take_profit_value': self.take_profit_pct,
            'trail_distance': 0.02,  # Default trail distance
            'trail_step': 0.01,  # Default trail step
            'max_gain_value': None,
        }

    def get_monitoring_params(self) -> dict[str, Any]:
        """Get risk monitoring parameters."""
        # Check if risk_monitoring_config was explicitly provided (not auto-generated)
        if (
            self._explicit_configs.get('risk_monitoring_config', False)
            and self.risk_monitoring_config is not None
        ):
            return {
                'check_interval': self.risk_monitoring_config.check_interval,
                'enable_real_time_alerts': self.risk_monitoring_config.enable_real_time_alerts,
                'max_history_size': self.risk_monitoring_config.max_history_size,
                'volatility_threshold': self.risk_monitoring_config.volatility_threshold,
                'drawdown_threshold': self.risk_monitoring_config.drawdown_threshold,
                'var_threshold': self.risk_monitoring_config.var_threshold,
                'lookback_period': self.risk_monitoring_config.lookback_period,
                'confidence_level': self.risk_monitoring_config.confidence_level,
            }

        # Fallback to legacy fields and defaults
        return {
            'check_interval': 60,
            'enable_real_time_alerts': True,
            'max_history_size': 500,
            'volatility_threshold': self.volatility_threshold,  # Use ComprehensiveRiskConfig value
            'drawdown_threshold': 0.15,  # Default drawdown threshold
            'var_threshold': 0.06,  # Default VaR threshold
            'lookback_period': 252,
            'confidence_level': 0.95,
        }
