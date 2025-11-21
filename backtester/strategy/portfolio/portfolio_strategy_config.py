"""Portfolio strategy configuration models.

This module contains Pydantic configuration models for portfolio strategies,
including constraints, optimization parameters, and risk budgets.
"""

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class PortfolioStrategyType(str, Enum):
    """Portfolio strategy types."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    MODERN_PORTFOLIO_THEORY = "modern_portfolio_theory"
    CONCENTRATED = "concentrated"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    CUSTOM = "custom"


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency options."""

    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    THRESHOLD_BASED = "threshold_based"


class RiskBudgetMethod(str, Enum):
    """Risk budget allocation methods."""

    EQUAL = "equal"
    INVERSE_VOLATILITY = "inverse_volatility"
    MARKET_CAP = "market_cap"
    CUSTOM = "custom"


class ConstraintType(str, Enum):
    """Constraint types for portfolio optimization."""

    WEIGHT = "weight"
    POSITION_SIZE = "position_size"
    SECTOR = "sector"
    CORRELATION = "correlation"
    TURNOVER = "turnover"
    LIQUIDITY = "liquidity"
    CUSTOM = "custom"


class OptimizationMethod(str, Enum):
    """Portfolio optimization methods."""

    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    CUSTOM = "custom"


class AllocationMethod(str, Enum):
    """Portfolio allocation methods."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    MODERN_PORTFOLIO_THEORY = "modern_portfolio_theory"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    CUSTOM = "custom"


class SignalFilterConfig(BaseModel):
    """Configuration for signal filtering."""

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    min_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    max_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    min_signal_duration: int = Field(default=1, ge=1)
    max_signal_duration: int = Field(default=100, ge=1)
    custom_filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator('max_confidence')
    @classmethod
    def max_confidence_must_be_ge_min(cls, v: float, info: ValidationInfo) -> float:
        """Ensure `max_confidence` is not below `min_confidence`."""
        if info.data.get('min_confidence') is not None and v < info.data['min_confidence']:
            raise ValueError('max_confidence must be >= min_confidence')
        return v

    @field_validator('max_strength')
    @classmethod
    def max_strength_must_be_ge_min(cls, v: float, info: ValidationInfo) -> float:
        """Ensure `max_strength` is not below `min_strength`."""
        if info.data.get('min_strength') is not None and v < info.data['min_strength']:
            raise ValueError('max_strength must be >= min_strength')
        return v

    @field_validator('max_signal_duration')
    @classmethod
    def max_signal_duration_must_be_ge_min(cls, v: int, info: ValidationInfo) -> int:
        """Ensure `max_signal_duration` is not below `min_signal_duration`."""
        if (
            info.data.get('min_signal_duration') is not None
            and v < info.data['min_signal_duration']
        ):
            raise ValueError('max_signal_duration must be >= min_signal_duration')
        return v


class PortfolioConstraints(BaseModel):
    """Portfolio constraints configuration."""

    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    min_position_size: float = Field(default=0.001, ge=0.0)
    max_position_size: float = Field(default=0.3, ge=0.0)
    sector_constraints: dict[str, dict[str, float]] = Field(default_factory=dict)
    correlation_constraints: dict[str, float] = Field(default_factory=dict)
    turnover_constraints: dict[str, float] = Field(default_factory=dict)
    liquidity_constraints: dict[str, float] = Field(default_factory=dict)

    @field_validator('max_weight')
    @classmethod
    def max_weight_must_be_ge_min(cls, v: float, info: ValidationInfo) -> float:
        """Ensure `max_weight` is not below `min_weight`."""
        if info.data.get('min_weight') is not None and v < info.data['min_weight']:
            raise ValueError('max_weight must be >= min_weight')
        return v

    @field_validator('max_position_size')
    @classmethod
    def max_position_size_must_be_ge_min(cls, v: float, info: ValidationInfo) -> float:
        """Ensure `max_position_size` is not below `min_position_size`."""
        if info.data.get('min_position_size') is not None and v < info.data['min_position_size']:
            raise ValueError('max_position_size must be >= min_position_size')
        return v

    def get_constraint_for_symbol(self, symbol: str) -> dict[str, float]:
        """Get constraints for a specific symbol."""
        symbol_constraints = {
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
        }

        # Add symbol-specific constraints if they exist
        if symbol in self.sector_constraints:
            symbol_constraints.update(self.sector_constraints[symbol])

        if symbol in self.correlation_constraints:
            symbol_constraints['max_correlation'] = self.correlation_constraints[symbol]

        if symbol in self.liquidity_constraints:
            symbol_constraints['min_liquidity'] = self.liquidity_constraints[symbol]

        return symbol_constraints


class PortfolioOptimizationParams(BaseModel):
    """Portfolio optimization parameters."""

    lookback_period: int = Field(default=252, ge=1)
    risk_free_rate: float = Field(default=0.02, ge=0.0)
    target_return: float | None = Field(default=None, ge=0.0)
    target_risk: float | None = Field(default=None, ge=0.0)
    optimization_method: str = Field(default="mean_variance")
    risk_aversion: float = Field(default=1.0, ge=0.0)
    transaction_costs: float = Field(default=0.001, ge=0.0)
    rebalance_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    min_samples: int = Field(default=30, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    max_iterations: int = Field(default=500, ge=1)
    convergence_tolerance: float = Field(default=1e-6, ge=0.0)

    @field_validator('max_samples')
    @classmethod
    def max_samples_must_be_ge_min(cls, v: int | None, info: ValidationInfo) -> int | None:
        """Ensure `max_samples` is not below `min_samples`."""
        if (
            info.data.get('min_samples') is not None
            and v is not None
            and v < info.data['min_samples']
        ):
            raise ValueError('max_samples must be >= min_samples')
        return v


class RiskBudget(BaseModel):
    """Risk budget configuration."""

    risk_budget_method: RiskBudgetMethod = Field(default=RiskBudgetMethod.EQUAL)
    risk_budgets: dict[str, float] = Field(default_factory=dict)
    risk_contribution_targets: dict[str, float] = Field(default_factory=dict)
    volatility_target: float | None = Field(default=None, ge=0.0)
    max_drawdown: float | None = Field(default=None, ge=0.0)
    var_confidence: float = Field(default=0.95, ge=0.5, le=1.0)
    var_horizon: int = Field(default=1, ge=1)

    @field_validator('risk_budgets')
    @classmethod
    def risk_budgets_must_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure risk budget allocations sum to 1.0."""
        if v and abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError('risk_budgets must sum to 1.0')
        return v

    @field_validator('risk_contribution_targets')
    @classmethod
    def risk_contribution_targets_must_be_valid(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure risk contribution targets are non-negative."""
        if v and any(val < 0 for val in v.values()):
            raise ValueError('risk_contribution_targets must be non-negative')
        return v

    def get_risk_budget(self) -> dict[str, float]:
        """Get risk budget for symbols."""
        if self.risk_budget_method == RiskBudgetMethod.EQUAL:
            return {symbol: 1.0 / len(self.risk_budgets) for symbol in self.risk_budgets}
        else:
            return self.risk_budgets.copy()


class RiskParameters(BaseModel):
    """Risk management parameters."""

    max_drawdown: float = Field(default=0.2, ge=0.0, le=1.0)
    var_confidence: float = Field(default=0.95, ge=0.5, le=1.0)
    var_horizon: int = Field(default=1, ge=1)
    max_position_size: float = Field(default=0.3, ge=0.0, le=1.0)
    max_sector_exposure: float = Field(default=0.5, ge=0.0, le=1.0)
    max_correlation: float = Field(default=0.8, ge=0.0, le=1.0)
    volatility_target: float | None = Field(default=None, ge=0.0)
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_win_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    risk_parity_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)
    rebalance_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class PerformanceMetrics(BaseModel):
    """Performance metrics configuration."""

    benchmark_symbol: str | None = Field(default=None)
    risk_free_rate: float = Field(default=0.02, ge=0.0)
    annualization_factor: int = Field(default=252, ge=1)
    calculate_sharpe_ratio: bool = Field(default=True)
    calculate_sortino_ratio: bool = Field(default=True)
    calculate_calmar_ratio: bool = Field(default=True)
    calculate_information_ratio: bool = Field(default=True)
    calculate_max_drawdown: bool = Field(default=True)
    calculate_var: bool = Field(default=True)
    calculate_cvar: bool = Field(default=True)
    calculate_tracking_error: bool = Field(default=True)
    calculate_beta: bool = Field(default=True)
    calculate_alpha: bool = Field(default=True)
    calculate_turnover: bool = Field(default=True)
    calculate_slippage: bool = Field(default=True)
    calculate_commission: bool = Field(default=True)


SignalFilterConfigInput = SignalFilterConfig | Mapping[str, Any]
RiskBudgetInput = RiskBudget | Mapping[str, Any]


class PortfolioStrategyConfig(BaseModel):
    """Portfolio strategy configuration."""

    strategy_name: str = Field(..., min_length=1, max_length=100)
    strategy_type: PortfolioStrategyType = Field(default=PortfolioStrategyType.EQUAL_WEIGHT)
    symbols: list[str] = Field(..., min_length=1, max_length=50)
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    optimization_params: PortfolioOptimizationParams = Field(
        default_factory=PortfolioOptimizationParams
    )
    risk_budget: RiskBudgetInput = Field(default_factory=RiskBudget)
    risk_parameters: RiskParameters = Field(default_factory=RiskParameters)
    signal_filters: SignalFilterConfigInput = Field(default_factory=SignalFilterConfig)
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    rebalance_frequency: RebalanceFrequency = Field(default=RebalanceFrequency.WEEKLY)
    enable_rebalancing: bool = Field(default=True)
    min_position_size: float = Field(default=0.001, ge=0.0)
    max_position_size: float = Field(default=0.3, ge=0.0)
    allocation_method: AllocationMethod = Field(default=AllocationMethod.EQUAL_WEIGHT)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator('symbols')
    @classmethod
    def symbols_must_be_unique(cls, v: list[str]) -> list[str]:
        """Ensure the configured symbols list does not contain duplicates."""
        if len(v) != len(set(v)):
            raise ValueError('symbols must be unique')
        return v

    @field_validator('max_position_size')
    @classmethod
    def max_position_size_must_be_ge_min(cls, v: float, info: ValidationInfo) -> float:
        """Ensure global max position size is at least the configured minimum."""
        if info.data.get('min_position_size') is not None and v < info.data['min_position_size']:
            raise ValueError('max_position_size must be >= min_position_size')
        return v

    @field_validator('custom_parameters')
    @classmethod
    def custom_parameters_must_be_valid(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure custom parameters do not clash with reserved configuration keys."""
        # Validate that custom parameters don't conflict with standard parameters
        reserved_keys = {
            'strategy_name',
            'strategy_type',
            'symbols',
            'constraints',
            'optimization_params',
            'risk_budget',
            'risk_parameters',
            'signal_filters',
            'performance_metrics',
            'rebalance_frequency',
            'enable_rebalancing',
            'min_position_size',
            'max_position_size',
            'allocation_method',
        }

        conflicting_keys = set(v.keys()) & reserved_keys
        if conflicting_keys:
            raise ValueError(f'custom_parameters conflicts with reserved keys: {conflicting_keys}')

        return v

    def get_constraint_for_symbol(self, symbol: str) -> dict[str, float]:
        """Get constraints for a specific symbol."""
        return self.constraints.get_constraint_for_symbol(symbol)

    def should_rebalance(self, current_weights: dict[str, float], current_step: int) -> bool:
        """Determine if portfolio should be rebalanced."""
        if not self.enable_rebalancing:
            return False

        if self.rebalance_frequency == RebalanceFrequency.CONTINUOUS:
            return True

        if self.rebalance_frequency == RebalanceFrequency.THRESHOLD_BASED:
            # Check if any weight has deviated significantly from target
            for _, current_weight in current_weights.items():
                target_weight = 1.0 / len(self.symbols)  # Default equal weight
                deviation = abs(current_weight - target_weight)
                if deviation > self.optimization_params.rebalance_threshold:
                    return True
            return False

        # For time-based rebalancing, this would be implemented in the strategy
        # For now, return True to allow the strategy to handle time-based logic
        return True

    def get_risk_budget(self) -> dict[str, float]:
        """Get risk budget for symbols."""
        budget = cast(RiskBudget, self.risk_budget)
        return budget.get_risk_budget()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'PortfolioStrategyConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def validate_config(self) -> bool:
        """Validate the configuration."""
        try:
            if not self.strategy_name or not self.symbols:
                return False

            type_validations = (
                isinstance(self.constraints, PortfolioConstraints),
                isinstance(self.optimization_params, PortfolioOptimizationParams),
                isinstance(self.risk_budget, RiskBudget),
                isinstance(self.risk_parameters, RiskParameters),
                isinstance(self.signal_filters, SignalFilterConfig),
                isinstance(self.performance_metrics, PerformanceMetrics),
            )

            return all(type_validations)

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> 'PortfolioStrategyConfig':
        """Create a copy of the configuration."""
        snapshot = self.model_dump(include=include, exclude=exclude)
        if update:
            snapshot.update(update)
        payload = deepcopy(snapshot) if deep else dict(snapshot)
        return self.model_validate(payload)

    def update(self, **kwargs: Any) -> 'PortfolioStrategyConfig':
        """Update configuration with new values."""
        config_dict = self.model_dump()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)

    def __str__(self) -> str:
        """String representation of the configuration."""
        strategy_type = getattr(self.strategy_type, 'value', self.strategy_type)
        return (
            "PortfolioStrategyConfig("
            f"name='{self.strategy_name}', type='{strategy_type}', symbols={self.symbols})"
        )

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return self.__str__()

    @field_validator('signal_filters', mode='before')
    @classmethod
    def normalize_signal_filters(cls, value: SignalFilterConfigInput) -> SignalFilterConfig:
        """Allow plain dictionaries for signal filter configuration."""
        if isinstance(value, Mapping):
            return SignalFilterConfig(**dict(value))
        return value

    @field_validator('risk_budget', mode='before')
    @classmethod
    def normalize_risk_budget(cls, value: RiskBudgetInput) -> RiskBudget:
        """Allow dictionaries for risk budget configuration."""
        if isinstance(value, Mapping):
            return RiskBudget(**dict(value))
        return value
