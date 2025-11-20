"""Configuration System for the Backtester.

This module provides a centralized configuration system that can be used
globally throughout the backtesting framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtester.risk_management.component_configs.comprehensive_risk_config import (
    ComprehensiveRiskConfig as ComprehensiveRiskConfig,
)


class DataRetrievalConfig(BaseModel):
    """Configuration class for data retrieval parameters using pydantic BaseModel.

    This class defines all the parameters needed to configure a data retrieval request,
    inheriting from pydantic.BaseModel for validation and serialization.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Data source configuration
    data_source: str = Field(
        default="yahoo", description="Data source (e.g., yahoo, bloomberg, fred)"
    )
    start_date: str | Any = Field(default="year", description="Start date for data retrieval")
    finish_date: str | Any | None = Field(
        default=None, description="Finish date for data retrieval"
    )

    # Ticker and field configuration
    tickers: str | list[str] | None = Field(
        default_factory=lambda: ["SPY"], description="List of ticker symbols"
    )
    fields: list[str] = Field(
        default_factory=lambda: ["close"], description="List of fields to retrieve"
    )
    vendor_tickers: list[str] | None = Field(
        default=None, description="Vendor-specific ticker symbols"
    )
    vendor_fields: list[str] | None = Field(default=None, description="Vendor-specific field names")

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v: str | list[str] | None) -> str | list[str] | None:
        """Convert single string to list for tickers."""
        if isinstance(v, str):
            return [v]
        return v

    # Frequency and granularity
    freq: str = Field(default="daily", description="Data frequency (daily, intraday, etc.)")
    gran_freq: str | None = Field(default=None, description="Granular frequency")
    freq_mult: int = Field(default=1, description="Frequency multiplier")

    # Cache and environment configuration
    cache_algo: str = Field(default="internet_load_return", description="Cache algorithm to use")
    environment: str | None = Field(default=None, description="Data environment (prod, backtest)")
    cut: str = Field(default="NYC", description="Cut time for data")

    # API keys and authentication
    fred_api_key: str | None = Field(default=None, description="FRED API key")
    alpha_vantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")
    eikon_api_key: str | None = Field(default=None, description="Eikon API key")

    # Additional parameters
    category: str | None = Field(default=None, description="Data category")
    dataset: str | None = Field(default=None, description="Dataset name")
    trade_side: str = Field(default="trade", description="Trade side (trade, bid, ask)")
    resample: str | None = Field(default=None, description="Resample frequency")
    resample_how: str = Field(default="last", description="Resample method")

    # Threading and performance
    split_request_chunks: int = Field(default=0, description="Split request into chunks")
    list_threads: int = Field(default=1, description="Number of threads for data loading")

    # Cache behavior
    push_to_cache: bool = Field(default=True, description="Whether to push data to cache")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Data overrides")
    cache_ttl_seconds: float | None = Field(
        default=300.0,
        description="In-memory cache time-to-live per entry in seconds",
        ge=0.0,
    )
    cache_max_entries: int | None = Field(
        default=256,
        description="Maximum number of cached frames kept in memory",
        ge=1,
    )


class StrategyConfig(BaseModel):
    """Strategy-related configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    strategy_name: str = Field(default="momentum_strategy", description="Name of the strategy")
    ma_short: int = Field(default=5, description="Short moving average period")
    ma_long: int = Field(default=20, description="Long moving average period")
    leverage_base: float = Field(default=1.0, description="Base leverage")
    leverage_alpha: float = Field(default=3.0, description="Alpha leverage")
    base_to_alpha_split: float = Field(default=0.2, description="Base to alpha split ratio")
    alpha_to_base_split: float = Field(default=0.2, description="Alpha to base split ratio")
    stop_loss_base: float = Field(default=0.025, description="Base stop loss percentage")
    stop_loss_alpha: float = Field(default=0.025, description="Alpha stop loss percentage")
    take_profit_target: float = Field(default=0.10, description="Take profit target percentage")


class PortfolioConfig(BaseModel):
    """Portfolio-related configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core portfolio parameters
    portfolio_strategy_name: str = Field(
        default="kelly_criterion", description="Default portfolio strategy identifier"
    )
    initial_capital: float = Field(default=100.0, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    interest_rate_daily: float = Field(default=0.00025, description="Daily interest rate")
    spread_rate: float = Field(default=0.0002, description="Spread rate")
    slippage_std: float = Field(default=0.0005, description="Slippage standard deviation")
    funding_enabled: bool = Field(default=True, description="Whether funding is enabled")
    tax_rate: float = Field(default=0.45, description="Tax rate")

    # General Portfolio specific
    max_positions: int = Field(default=10, description="Maximum number of concurrent positions")

    # Dual Pool Portfolio specific
    leverage_base: float = Field(default=1.0, description="Leverage factor for base pool")
    leverage_alpha: float = Field(default=3.0, description="Leverage factor for alpha pool")
    base_to_alpha_split: float = Field(default=0.2, description="Base to alpha split ratio")
    alpha_to_base_split: float = Field(default=0.2, description="Alpha to base split ratio")
    stop_loss_base: float = Field(default=0.025, description="Base stop loss percentage")
    stop_loss_alpha: float = Field(default=0.025, description="Alpha stop loss percentage")
    take_profit_target: float = Field(default=0.10, description="Take profit target percentage")
    maintenance_margin: float = Field(
        default=0.5, description="Maintenance margin for leveraged positions"
    )
    max_total_leverage: float = Field(
        default=4.0, description="Maximum total portfolio leverage allowed"
    )
    cash: float = Field(default=0.0, description="Cash allocation for the portfolio")

    @field_validator('base_to_alpha_split', 'alpha_to_base_split')
    @classmethod
    def validate_split_ratios(cls, v: float) -> float:
        """Validate split ratios are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Split ratios must be between 0 and 1")
        return v

    @field_validator('leverage_base', 'leverage_alpha')
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Validate leverage is positive."""
        if v <= 0:
            raise ValueError("Leverage must be positive")
        return v

    def create_general_portfolio(self) -> dict[str, Any]:
        """Create a dictionary of parameters for GeneralPortfolio initialization.

        Returns:
            Dictionary of parameters for GeneralPortfolio
        """
        return {
            'initial_capital': self.initial_capital,
            'commission_rate': self.commission_rate,
            'interest_rate_daily': self.interest_rate_daily,
            'spread_rate': self.spread_rate,
            'slippage_std': self.slippage_std,
            'funding_enabled': self.funding_enabled,
            'tax_rate': self.tax_rate,
            'max_positions': self.max_positions,
        }

    def create_dual_pool_portfolio(self) -> dict[str, Any]:
        """Create a dictionary of parameters for DualPoolPortfolio initialization.

        Returns:
            Dictionary of parameters for DualPoolPortfolio
        """
        return {
            'initial_capital': self.initial_capital,
            'leverage_base': self.leverage_base,
            'leverage_alpha': self.leverage_alpha,
            'base_to_alpha_split': self.base_to_alpha_split,
            'alpha_to_base_split': self.alpha_to_base_split,
            'stop_loss_base': self.stop_loss_base,
            'stop_loss_alpha': self.stop_loss_alpha,
            'take_profit_target': self.take_profit_target,
            'maintenance_margin': self.maintenance_margin,
            'commission_rate': self.commission_rate,
            'interest_rate_daily': self.interest_rate_daily,
            'spread_rate': self.spread_rate,
            'slippage_std': self.slippage_std,
            'funding_enabled': self.funding_enabled,
            'max_total_leverage': self.max_total_leverage,
            'cash': self.cash,
            'tax_rate': self.tax_rate,
        }


class SimulatedBrokerConfig(BaseModel):
    """Configuration for SimulatedBroker parameters."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    commission_rate: float = Field(
        default=0.001, description="Commission rate for trades (as decimal)"
    )
    min_commission: float = Field(default=1.0, description="Minimum commission per trade")
    spread: float = Field(default=0.0001, description="Bid-ask spread (as decimal)")
    slippage_model: str = Field(
        default="normal", description="Type of slippage model ('normal', 'fixed', 'none')"
    )
    slippage_distribution: str | None = Field(
        default=None,
        description="Optional distribution applied when sampling slippage adjustments",
    )
    slippage_std: float = Field(
        default=0.0005, description="Standard deviation for slippage simulation"
    )
    latency_ms: float = Field(default=0.0, description="Baseline simulated latency in milliseconds")
    latency_jitter_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional random latency applied on top of the baseline",
    )
    max_orders_per_minute: int = Field(
        default=0,
        ge=0,
        description="Maximum number of orders allowed per rolling minute (0 disables throttling)",
    )
    order_cooldown_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum number of seconds that must elapse between orders (0 disables cooldown)",
    )

    @field_validator('slippage_model')
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        """Validate slippage model is one of the allowed values."""
        valid_models = ['normal', 'fixed', 'none']
        if v not in valid_models:
            raise ValueError(f"slippage_model must be one of {valid_models}")
        return v

    @field_validator('slippage_distribution')
    @classmethod
    def validate_slippage_distribution(cls, v: str | None) -> str | None:
        """Validate slippage distribution aliases when explicitly provided."""
        if v is None:
            return v
        valid_distributions = ['normal', 'lognormal', 'fixed', 'none']
        if v not in valid_distributions:
            raise ValueError(f"slippage_distribution must be one of {valid_distributions}")
        return v

    @field_validator('order_cooldown_seconds')
    @classmethod
    def validate_cooldown(cls, v: float) -> float:
        """Ensure cooldown is non-negative."""
        if v < 0:
            raise ValueError("order_cooldown_seconds must be non-negative")
        return v

    @field_validator('max_orders_per_minute')
    @classmethod
    def validate_order_rate(cls, v: int) -> int:
        """Ensure throttling limit is non-negative."""
        if v < 0:
            raise ValueError("max_orders_per_minute must be greater than or equal to zero")
        return v

    @model_validator(mode='after')
    def _sync_slippage_distribution(self) -> SimulatedBrokerConfig:
        """Default distribution to match the declared slippage model when unset."""
        if self.slippage_distribution is None:
            self.slippage_distribution = self.slippage_model
        return self


class ExecutionConfig(SimulatedBrokerConfig):
    """Backward-compatible alias for execution component configuration."""

    pass


class PerformanceConfig(BaseModel):
    """Performance analysis configuration settings."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")
    benchmark_enabled: bool = Field(default=False, description="Whether benchmark is enabled")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")


class BacktesterConfig(BaseModel):
    """Main configuration class for the backtester."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    data: DataRetrievalConfig | None = Field(
        default=None, description="Data retrieval configuration"
    )
    strategy: StrategyConfig | None = Field(default=None, description="Strategy configuration")
    portfolio: PortfolioConfig | None = Field(default=None, description="Portfolio configuration")
    execution: ExecutionConfig | None = Field(default=None, description="Execution configuration")
    risk: ComprehensiveRiskConfig | None = Field(default=None, description="Risk configuration")
    performance: PerformanceConfig | None = Field(
        default=None, description="Performance configuration"
    )

    # Data period and trading behavior settings
    data_period_days: int = Field(
        default=1, description="Time interval between data points in days"
    )
    maximum_period_between_trade: int = Field(
        default=30, description="Maximum periods to wait between trades"
    )
    trade_immediately_after_stop: bool = Field(
        default=True, description="Whether to trade immediately after stop"
    )

    def __init__(self, **data: Any) -> None:
        """Initialize BacktesterConfig and set default configurations if not provided."""
        super().__init__(**data)
        # Initialize default configurations if not provided
        if self.data is None:
            self.data = DataRetrievalConfig()
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.portfolio is None:
            self.portfolio = PortfolioConfig()
        if self.execution is None:
            self.execution = ExecutionConfig()
        if self.risk is None:
            self.risk = ComprehensiveRiskConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()


# Global configuration instance
_global_config: BacktesterConfig | None = None


def get_config() -> BacktesterConfig:
    """Get the global configuration instance.

    Returns:
        Global BacktesterConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = BacktesterConfig()
    assert _global_config is not None
    return _global_config


def set_config(config: BacktesterConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: BacktesterConfig instance to set as global
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = BacktesterConfig()


def _coerce_datetime(value: Any) -> datetime | None:
    """Attempt to convert the provided value into a datetime for validation."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        # Allow relative keywords that findatapy understands without validation
        if normalized in {"year", "ytd", "max", "month", "week"}:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def validate_run_config(config: BacktesterConfig) -> BacktesterConfig:
    """Validate a backtest configuration and raise if any invalid combinations exist."""
    errors: list[str] = []

    data = config.data
    if data is None:
        errors.append("data configuration is required")
    else:
        tickers = data.tickers
        if isinstance(tickers, str):
            tickers = [tickers]
        if not tickers:
            errors.append("data.tickers must contain at least one symbol")

        start_dt = _coerce_datetime(data.start_date)
        finish_dt = _coerce_datetime(data.finish_date)
        if start_dt and finish_dt and finish_dt < start_dt:
            errors.append(
                f"data.finish_date ({data.finish_date}) must not be earlier than data.start_date ({data.start_date})"
            )

    portfolio = config.portfolio
    if portfolio is None:
        errors.append("portfolio configuration is required")
    else:
        if portfolio.leverage_base <= 0:
            errors.append("portfolio.leverage_base must be positive")
        if portfolio.leverage_alpha <= 0:
            errors.append("portfolio.leverage_alpha must be positive")

    if errors:
        formatted = "\n - ".join(errors)
        raise ValueError(f"Invalid backtest configuration:\n - {formatted}")

    return config


# ---------------------------------------------------------------------------#
# Immutable config views
# ---------------------------------------------------------------------------#


def _as_tuple(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _freeze_mapping(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return MappingProxyType({})
    return MappingProxyType(dict(payload))


@dataclass(frozen=True, slots=True)
class DataConfigView:
    """Immutable view for data configuration passed to downstream readers."""

    data_source: str
    tickers: tuple[str, ...]
    start_date: Any
    finish_date: Any | None
    freq: str
    fields: tuple[str, ...]
    vendor_tickers: tuple[str, ...]
    vendor_fields: tuple[str, ...]
    cache_ttl_seconds: float | None
    cache_max_entries: int | None


@dataclass(frozen=True, slots=True)
class PortfolioConfigView:
    """Immutable portfolio configuration slice."""

    initial_capital: float
    commission_rate: float
    interest_rate_daily: float
    spread_rate: float
    slippage_std: float
    funding_enabled: bool
    tax_rate: float
    max_positions: int


@dataclass(frozen=True, slots=True)
class RiskConfigView:
    """Immutable wrapper around comprehensive risk configuration."""

    _payload: Mapping[str, Any]

    @classmethod
    def from_model(cls, model: ComprehensiveRiskConfig) -> Self:
        """Create a view from an existing ComprehensiveRiskConfig."""
        return cls(_payload=_freeze_mapping(model.model_dump(mode="python")))

    def materialize(self) -> ComprehensiveRiskConfig:
        """Return a deep copy of the underlying configuration."""
        return ComprehensiveRiskConfig(**dict(self._payload))


def build_data_config_view(config: BacktesterConfig) -> DataConfigView:
    """Return an immutable data configuration view for downstream consumers."""
    assert config.data is not None
    data = config.data
    return DataConfigView(
        data_source=data.data_source,
        tickers=_as_tuple(data.tickers) or ("SPY",),
        start_date=data.start_date,
        finish_date=data.finish_date,
        freq=data.freq,
        fields=tuple(data.fields),
        vendor_tickers=_as_tuple(data.vendor_tickers),
        vendor_fields=_as_tuple(data.vendor_fields),
        cache_ttl_seconds=data.cache_ttl_seconds,
        cache_max_entries=data.cache_max_entries,
    )


def build_portfolio_config_view(config: BacktesterConfig) -> PortfolioConfigView:
    """Return an immutable portfolio configuration view."""
    assert config.portfolio is not None
    portfolio = config.portfolio
    funding_enabled = (
        bool(portfolio.funding_enabled) if portfolio.funding_enabled is not None else True
    )
    return PortfolioConfigView(
        initial_capital=float(portfolio.initial_capital or 100.0),
        commission_rate=float(portfolio.commission_rate or 0.001),
        interest_rate_daily=float(portfolio.interest_rate_daily or 0.00025),
        spread_rate=float(portfolio.spread_rate or 0.0002),
        slippage_std=float(portfolio.slippage_std or 0.0005),
        funding_enabled=funding_enabled,
        tax_rate=float(portfolio.tax_rate or 0.45),
        max_positions=int(portfolio.max_positions or 10),
    )


def build_risk_config_view(config: BacktesterConfig) -> RiskConfigView:
    """Return an immutable risk configuration view."""
    assert config.risk is not None
    return RiskConfigView.from_model(config.risk)


class BacktestRunConfig:
    """Builder that produces immutable BacktesterConfig snapshots for each run."""

    _COMPONENT_MODELS = {
        "data": DataRetrievalConfig,
        "strategy": StrategyConfig,
        "portfolio": PortfolioConfig,
        "execution": ExecutionConfig,
        "risk": ComprehensiveRiskConfig,
        "performance": PerformanceConfig,
    }

    def __init__(self, base_config: BacktesterConfig | None = None) -> None:
        """Initialise the builder with a deep copy of the provided base config."""
        base = base_config or get_config()
        self._base_config = base.model_copy(deep=True)
        self._overrides: dict[str, dict[str, Any]] = {}

    def with_data_overrides(
        self, config: DataRetrievalConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the data component."""
        return self._apply_override("data", config, kwargs)

    def with_strategy_overrides(
        self, config: StrategyConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the strategy component."""
        return self._apply_override("strategy", config, kwargs)

    def with_portfolio_overrides(
        self, config: PortfolioConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the portfolio component."""
        return self._apply_override("portfolio", config, kwargs)

    def with_execution_overrides(
        self, config: ExecutionConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the execution component."""
        return self._apply_override("execution", config, kwargs)

    def with_risk_overrides(
        self, config: ComprehensiveRiskConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the risk component."""
        return self._apply_override("risk", config, kwargs)

    def with_performance_overrides(
        self, config: PerformanceConfig | dict[str, Any] | None = None, **kwargs: Any
    ) -> BacktestRunConfig:
        """Queue overrides for the performance component."""
        return self._apply_override("performance", config, kwargs)

    def _apply_override(
        self,
        component: str,
        config: BaseModel | dict[str, Any] | None,
        extra_kwargs: dict[str, Any],
    ) -> BacktestRunConfig:
        payload: dict[str, Any] = {}
        if isinstance(config, BaseModel):
            payload.update(config.model_dump(exclude_unset=True))
        elif isinstance(config, dict):
            payload.update(config)
        payload.update(extra_kwargs)
        if payload:
            self._overrides.setdefault(component, {}).update(payload)
        return self

    def build(self, *, validate: bool = True) -> BacktesterConfig:
        """Produce a BacktesterConfig snapshot with the queued overrides applied."""
        resolved = self._base_config.model_copy(deep=True)

        for component, overrides in self._overrides.items():
            model_cls = self._COMPONENT_MODELS.get(component)
            if model_cls is None:
                continue
            current = getattr(resolved, component)
            if current is None:
                current = model_cls()
            updated = current.model_copy(update=overrides)
            setattr(resolved, component, updated)

        if validate:
            return validate_run_config(resolved)
        return resolved
