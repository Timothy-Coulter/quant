"""Signal strategy configuration models using Pydantic.

This module defines the configuration models for all signal strategies,
providing type safety, validation, and serialization capabilities.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Import the correct IndicatorConfig from the existing indicators module
from backtester.indicators.indicator_configs import IndicatorConfig


class SignalStrategyType(str, Enum):
    """Enumeration of signal strategy types."""

    TECHNICAL_ANALYSIS = "technical_analysis"
    ML_MODEL = "ml_model"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """Configuration for machine learning models."""

    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model")
    framework: str = Field(..., description="ML framework used")
    target_column: str = Field("close", description="Target column for prediction")
    lookback_period: int = Field(20, ge=1, description="Lookback period for features")
    train_test_split: float = Field(0.8, ge=0.1, le=0.9, description="Train/test split ratio")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    data_config: dict[str, Any] | None = Field(None, description="Data configuration")

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v.strip()

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        """Validate model type."""
        if not v or not isinstance(v, str):
            raise ValueError("Model type must be a non-empty string")
        return v.strip()

    @field_validator('framework')
    @classmethod
    def validate_framework(cls, v):
        """Validate framework."""
        if not v or not isinstance(v, str):
            raise ValueError("Framework must be a non-empty string")
        return v.strip()


class SignalFilterConfig(BaseModel):
    """Configuration for signal filters."""

    filter_name: str = Field(..., description="Name of the filter")
    filter_type: str = Field(..., description="Type of filter")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Filter parameters")
    enabled: bool = Field(True, description="Whether the filter is enabled")

    @field_validator('filter_name')
    @classmethod
    def validate_filter_name(cls, v):
        """Validate filter name."""
        if not v or not isinstance(v, str):
            raise ValueError("Filter name must be a non-empty string")
        return v.strip()

    @field_validator('filter_type')
    @classmethod
    def validate_filter_type(cls, v):
        """Validate filter type."""
        if not v or not isinstance(v, str):
            raise ValueError("Filter type must be a non-empty string")
        return v.strip()


class RiskParameters(BaseModel):
    """Risk management parameters for signal strategies."""

    max_position_size: float = Field(0.1, ge=0.01, le=1.0, description="Maximum position size")
    stop_loss_percentage: float = Field(0.05, ge=0.0, le=0.5, description="Stop loss percentage")
    take_profit_percentage: float = Field(0.1, ge=0.0, le=1.0, description="Take profit percentage")
    max_drawdown: float = Field(0.2, ge=0.0, le=1.0, description="Maximum drawdown")
    risk_per_trade: float = Field(0.02, ge=0.0, le=0.1, description="Risk per trade")
    correlation_limit: float = Field(0.8, ge=0.0, le=1.0, description="Correlation limit")

    @field_validator('max_position_size')
    @classmethod
    def validate_max_position_size(cls, v):
        """Validate max position size."""
        if v <= 0 or v > 1:
            raise ValueError("Max position size must be between 0 and 1")
        return v


class ExecutionParameters(BaseModel):
    """Execution parameters for signal strategies."""

    execution_type: str = Field("market", description="Execution type")
    slippage_percentage: float = Field(0.001, ge=0.0, le=0.1, description="Slippage percentage")
    commission_percentage: float = Field(0.001, ge=0.0, le=0.1, description="Commission percentage")
    min_order_size: float = Field(100.0, ge=0.0, description="Minimum order size")
    max_order_size: float = Field(1000000.0, ge=0.0, description="Maximum order size")
    timeout_seconds: int = Field(30, ge=1, description="Execution timeout in seconds")

    @field_validator('execution_type')
    @classmethod
    def validate_execution_type(cls, v):
        """Validate execution type."""
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if v not in valid_types:
            raise ValueError(f"Execution type must be one of: {valid_types}")
        return v


class TechnicalAnalysisStrategyConfig(BaseModel):
    """Configuration for technical analysis strategy."""

    name: str = Field(default="technical_analysis_strategy", description="Strategy name")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.TECHNICAL_ANALYSIS, description="Strategy type"
    )
    strategy_name: str = Field(default="technical_analysis_strategy")
    symbols: list[str] = Field(
        default_factory=lambda: ['AAPL'], description="List of symbols to trade"
    )
    indicators: list[IndicatorConfig] = Field(..., description="List of indicators to use")
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Technical analysis specific parameters
    confirmation_period: int = Field(3, ge=1, description="Confirmation period for signals")
    min_signal_strength: float = Field(0.5, ge=0.0, le=1.0, description="Minimum signal strength")
    use_volume_confirmation: bool = Field(True, description="Use volume confirmation")
    volume_multiplier: float = Field(1.2, ge=1.0, description="Volume multiplier for confirmation")

    # Strategy parameters
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )

    # For backward compatibility with tests
    trend_indicators: list[str] = Field(
        default_factory=lambda: ['sma_20', 'sma_50', 'ema_12', 'ema_26'],
        description='List of trend indicators to use',
    )
    momentum_indicators: list[str] = Field(
        default_factory=lambda: ['rsi', 'macd', 'stochastic', 'cci'],
        description='List of momentum indicators to use',
    )
    volatility_indicators: list[str] = Field(
        default_factory=lambda: ['bb', 'atr'], description='List of volatility indicators to use'
    )
    volume_indicators: list[str] = Field(
        default_factory=lambda: ['volume'], description='List of volume indicators to use'
    )
    signal_generation_rules: list[str] = Field(
        default_factory=list, description='Rules used for generating signals'
    )
    signal_aggregation: str = Field(
        default='weighted_vote', description='Method for aggregating signals'
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description='Minimum confidence threshold for signals'
    )

    @field_validator('indicators')
    @classmethod
    def validate_indicators(cls, v):
        """Validate indicators list."""
        # Check for duplicate indicator names
        names = [ind.indicator_name for ind in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate indicator names found")

        return v

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Ensure at least one trading symbol is provided."""
        if not v:
            raise ValueError("At least one symbol is required")
        normalized: list[str] = []
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
            normalized.append(symbol.strip().upper())
        return normalized

    @field_validator('signal_generation_rules', mode='before')
    @classmethod
    def normalize_signal_generation_rules(cls, v):
        """Allow a single string or list of strings for signal generation rules."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            normalized = []
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("Signal generation rules must be non-empty strings")
                normalized.append(item.strip())
            return normalized
        raise ValueError("Signal generation rules must be provided as a string or list of strings")


class MLModelStrategyConfig(BaseModel):
    """Configuration for ML model strategy."""

    name: str = Field(default="ml_model_strategy", description="Strategy name")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.ML_MODEL, description="Strategy type"
    )
    strategy_name: str = Field(default="ml_model_strategy")
    symbols: list[str] = Field(
        default_factory=lambda: ['AAPL'],
        description="List of symbols to trade",
    )
    models: list[ModelConfig] = Field(..., description="List of ML models to use")
    indicators: list[IndicatorConfig] = Field(
        default_factory=list, description="Supporting indicators"
    )
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # ML model specific parameters
    prediction_horizon: int = Field(1, ge=1, description="Prediction horizon")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Confidence threshold")
    min_prediction_strength: float = Field(0.1, ge=0.0, description="Minimum prediction strength")
    use_ensemble: bool = Field(True, description="Use ensemble of models")
    ensemble_weights: dict[str, float] | None = Field(None, description="Ensemble model weights")

    # Feature engineering parameters
    feature_columns: list[str] | None = Field(None, description="Feature columns to use")
    target_column: str = Field("close", description="Target column")
    normalize_features: bool = Field(True, description="Normalize features")

    # Aggregation parameters
    aggregation_method: str = Field("weighted_average", description="Aggregation method")
    retrain_frequency: str = Field(
        "daily", description="How frequently the models should be retrained"
    )
    feature_importance_threshold: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum feature importance to keep a feature"
    )

    # Strategy parameters
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )

    # For backward compatibility with tests
    model_ensemble_method: str = Field(
        default="weighted_average", description="Model ensemble method"
    )

    @field_validator('models')
    @classmethod
    def validate_models(cls, v):
        """Validate models list."""
        # Check for duplicate model names
        names = [model.model_name for model in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate model names found")

        return v

    @field_validator('aggregation_method')
    @classmethod
    def validate_aggregation_method(cls, v):
        """Validate aggregation method."""
        valid_methods = [
            'weighted_average',
            'majority_vote',
            'confidence_weighted',
            'simple_average',
        ]
        if v not in valid_methods:
            raise ValueError(f"Aggregation method must be one of: {valid_methods}")
        return v

    @field_validator('retrain_frequency')
    @classmethod
    def validate_retrain_frequency(cls, v):
        """Validate retraining frequency string."""
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'custom']
        value = v.lower().strip()
        if value not in valid_frequencies:
            raise ValueError(f"Retrain frequency must be one of: {valid_frequencies}")
        return value

    @field_validator('feature_importance_threshold')
    @classmethod
    def validate_feature_importance_threshold(cls, v):
        """Validate feature importance threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Feature importance threshold must be between 0.0 and 1.0")
        return v

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Ensure at least one trading symbol is provided."""
        if not v:
            raise ValueError("At least one symbol is required")
        normalized: list[str] = []
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
            normalized.append(symbol.strip().upper())
        return normalized


class MomentumStrategyConfig(BaseModel):
    """Configuration for momentum strategy."""

    name: str = Field(default="momentum_strategy", description="Strategy name")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.MOMENTUM, description="Strategy type"
    )
    strategy_name: str = Field(default="momentum_strategy")
    symbols: list[str] = Field(
        default_factory=lambda: ['AAPL'],
        description="List of symbols to trade",
    )
    indicators: list[IndicatorConfig] = Field(
        default_factory=list, description="Supporting indicators"
    )
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Momentum specific parameters
    momentum_period: int = Field(20, ge=1, description="Momentum period")
    momentum_threshold: float = Field(0.02, ge=0.0, description="Momentum threshold")
    acceleration_threshold: float = Field(0.01, ge=0.0, description="Acceleration threshold")
    volume_confirmation: bool = Field(True, description="Use volume confirmation")
    min_volume_multiplier: float = Field(1.2, ge=1.0, description="Minimum volume multiplier")

    # Multi-timeframe parameters
    use_multiple_timeframes: bool = Field(True, description="Use multiple timeframes")
    timeframes: list[int] = Field(
        default_factory=lambda: [5, 10, 20], description="Timeframes to use"
    )
    momentum_weighting: str = Field("equal", description="Momentum weighting method")

    # Signal generation parameters
    use_breakout_signals: bool = Field(True, description="Use breakout signals")
    use_divergence_signals: bool = Field(True, description="Use divergence signals")
    use_trend_following: bool = Field(True, description="Use trend following")
    trend_confirmation: bool = Field(True, description="Require higher timeframe confirmation")
    volatility_filter: bool = Field(False, description="Apply a volatility filter to signals")
    min_volume_threshold: float = Field(
        0.0, ge=0.0, description="Minimum volume required to act on a signal"
    )

    # Strategy parameters
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )

    # For backward compatibility with tests
    momentum_periods: list[int] = Field(
        default_factory=lambda: [20], description="Momentum periods to use"
    )

    @field_validator('timeframes')
    @classmethod
    def validate_timeframes(cls, v):
        """Validate timeframes."""
        if not v:
            raise ValueError("At least one timeframe is required")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate timeframes found")

        # Check for reasonable values
        for tf in v:
            if tf <= 0 or tf > 252:  # Max one year of daily data
                raise ValueError("Timeframes must be between 1 and 252")

        return v

    @field_validator('momentum_weighting')
    @classmethod
    def validate_momentum_weighting(cls, v):
        """Validate momentum weighting method."""
        valid_methods = ['equal', 'linear', 'exponential', 'custom']
        if v not in valid_methods:
            raise ValueError(f"Momentum weighting must be one of: {valid_methods}")
        return v

    @field_validator('momentum_periods')
    @classmethod
    def validate_momentum_periods(cls, v):
        """Validate collection of momentum periods."""
        if not v:
            raise ValueError("At least one momentum period is required")
        unique_periods = set()
        for period in v:
            if period <= 0:
                raise ValueError("Momentum periods must be positive")
            unique_periods.add(period)
        if len(unique_periods) != len(v):
            raise ValueError("Duplicate momentum periods found")
        return v

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Ensure at least one trading symbol is provided."""
        if not v:
            raise ValueError("At least one symbol is required")
        normalized: list[str] = []
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
            normalized.append(symbol.strip().upper())
        return normalized


class MeanReversionStrategyConfig(BaseModel):
    """Configuration for mean reversion strategy."""

    name: str = Field(default="mean_reversion_strategy", description="Strategy name")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.MEAN_REVERSION, description="Strategy type"
    )
    strategy_name: str = Field(default="mean_reversion_strategy")
    symbols: list[str] = Field(..., description="List of symbols to trade")
    indicators: list[IndicatorConfig] = Field(
        default_factory=list, description="Supporting indicators"
    )
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Mean reversion specific parameters
    mean_period: int = Field(20, ge=1, description="Mean calculation period")
    std_dev_period: int = Field(20, ge=1, description="Standard deviation period")
    z_score_threshold: float = Field(2.0, ge=0.5, description="Z-score threshold")
    bollinger_band_period: int = Field(20, ge=1, description="Bollinger Band period")
    bollinger_band_std: float = Field(2.0, ge=0.5, description="Bollinger Band standard deviations")

    # Mean reversion detection parameters
    use_multiple_means: bool = Field(True, description="Use multiple timeframe means")
    mean_periods: list[int] = Field(
        default_factory=lambda: [10, 20, 50], description="Mean calculation periods"
    )
    std_dev_periods: list[int] = Field(
        default_factory=lambda: [20], description="Standard deviation calculation periods"
    )
    use_hurst_exponent: bool = Field(False, description="Use Hurst exponent")
    hurst_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Hurst exponent threshold")

    # Signal generation parameters
    use_statistical_arbitrage: bool = Field(True, description="Use statistical arbitrage")
    use_pairs_trading: bool = Field(False, description="Use pairs trading")
    mean_reversion_speed: float = Field(0.1, ge=0.0, le=1.0, description="Mean reversion speed")

    # Risk management parameters
    max_deviation: float = Field(3.0, ge=1.0, description="Maximum deviation threshold")
    min_reversion_strength: float = Field(0.05, ge=0.0, description="Minimum reversion strength")
    volume_confirmation: bool = Field(True, description="Use volume confirmation")
    volatility_adjustment: bool = Field(
        False, description="Adjust signal strength based on volatility regimes"
    )
    regime_filter: bool = Field(False, description="Apply regime-based filtering to signals")
    correlation_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Correlation threshold for pair selection"
    )

    # Strategy parameters
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )

    # For backward compatibility with tests
    min_volume_multiplier: float = Field(
        default=1.2, ge=1.0, description="Minimum volume multiplier"
    )

    @field_validator('mean_periods')
    @classmethod
    def validate_mean_periods(cls, v):
        """Validate mean periods."""
        if not v:
            raise ValueError("At least one mean period is required")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate mean periods found")

        # Check for reasonable values
        for period in v:
            if period <= 0 or period > 252:
                raise ValueError("Mean periods must be between 1 and 252")

        return v

    @field_validator('std_dev_periods')
    @classmethod
    def validate_std_dev_periods(cls, v):
        """Validate standard deviation periods."""
        if not v:
            raise ValueError("At least one standard deviation period is required")
        unique_periods = set()
        for period in v:
            if period <= 0 or period > 252:
                raise ValueError("Standard deviation periods must be between 1 and 252")
            unique_periods.add(period)
        if len(unique_periods) != len(v):
            raise ValueError("Duplicate standard deviation periods found")
        return v

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Ensure at least one trading symbol is provided."""
        if not v:
            raise ValueError("At least one symbol is required")
        normalized: list[str] = []
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
            normalized.append(symbol.strip().upper())
        return normalized


class ArbitrageStrategyConfig(BaseModel):
    """Configuration for arbitrage strategy."""

    name: str = Field(default="arbitrage_strategy", description="Strategy name")
    strategy_name: str = Field(default="arbitrage_strategy", description="Strategy identifier")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.ARBITRAGE, description="Strategy type"
    )
    symbols: list[str] = Field(..., description="List of symbols to trade")
    indicators: list[IndicatorConfig] = Field(
        default_factory=list, description="Supporting indicators"
    )
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Arbitrage specific parameters
    arbitrage_threshold: float = Field(0.01, ge=0.0, description="Arbitrage threshold")
    execution_speed_ms: int = Field(100, ge=1, description="Execution speed in milliseconds")
    min_profit_margin: float = Field(0.005, ge=0.0, description="Minimum profit margin")
    max_holding_period: int = Field(3600, ge=1, description="Maximum holding period in seconds")

    # Strategy parameters
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )


class CustomStrategyConfig(BaseModel):
    """Configuration for custom strategies."""

    name: str = Field(default="custom_strategy", description="Strategy name")
    strategy_name: str = Field(default="custom_strategy", description="Strategy identifier")
    strategy_type: SignalStrategyType = Field(
        SignalStrategyType.CUSTOM, description="Strategy type"
    )
    symbols: list[str] = Field(..., description="List of symbols to trade")
    indicators: list[IndicatorConfig] = Field(
        default_factory=list, description="Supporting indicators"
    )
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Custom strategy parameters
    custom_module: str = Field(..., description="Custom strategy module path")
    custom_class: str = Field(..., description="Custom strategy class name")
    custom_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Custom strategy parameters"
    )
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional strategy parameters"
    )

    @model_validator(mode='after')
    def _synchronize_parameters(self) -> 'CustomStrategyConfig':
        """Keep legacy custom_parameters and strategy_parameters aligned."""
        if self.strategy_parameters and not self.custom_parameters:
            self.custom_parameters = dict(self.strategy_parameters)
        elif self.custom_parameters and not self.strategy_parameters:
            self.strategy_parameters = dict(self.custom_parameters)
        return self

    @field_validator('custom_module')
    @classmethod
    def validate_custom_module(cls, v):
        """Validate custom module."""
        if not v or not isinstance(v, str):
            raise ValueError("Custom module must be a non-empty string")
        return v.strip()

    @field_validator('custom_class')
    @classmethod
    def validate_custom_class(cls, v):
        """Validate custom class."""
        if not v or not isinstance(v, str):
            raise ValueError("Custom class must be a non-empty string")
        return v.strip()


class SignalStrategyConfig(BaseModel):
    """Base configuration for signal strategies."""

    # Common configuration
    name: str = Field(..., description="Strategy name")
    description: str = Field("", description="Strategy description")
    enabled: bool = Field(True, description="Whether the strategy is enabled")

    # Strategy type specific configuration
    strategy_type: SignalStrategyType = Field(..., description="Strategy type")

    # Common components
    symbols: list[str] = Field(..., description="List of symbols to trade")
    indicators: list[IndicatorConfig] = Field(default_factory=list, description="Indicators to use")
    signal_filters: list[SignalFilterConfig] = Field(
        default_factory=list, description="Signal filters"
    )
    models: list[ModelConfig] = Field(default_factory=list, description="Model configurations")
    risk_parameters: RiskParameters = Field(
        default_factory=RiskParameters, description="Risk parameters"
    )
    execution_params: ExecutionParameters = Field(
        default_factory=ExecutionParameters, description="Execution parameters"
    )

    # Strategy-specific configuration (union of all strategy configs)
    strategy_config: (
        TechnicalAnalysisStrategyConfig
        | MLModelStrategyConfig
        | MomentumStrategyConfig
        | MeanReversionStrategyConfig
        | ArbitrageStrategyConfig
        | CustomStrategyConfig
    ) = Field(..., description="Strategy-specific configuration")

    # Additional parameters
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    # Logging and monitoring
    log_level: str = Field("INFO", description="Logging level")
    monitor_performance: bool = Field(True, description="Monitor strategy performance")
    performance_metrics: list[str] = Field(
        default_factory=list, description="Performance metrics to track"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate strategy name."""
        if not v or not isinstance(v, str):
            raise ValueError("Strategy name must be a non-empty string")
        return v.strip()

    @field_validator('strategy_type')
    @classmethod
    def validate_strategy_type(cls, v):
        """Validate strategy type."""
        if isinstance(v, SignalStrategyType):
            return v
        if isinstance(v, str):
            try:
                normalized = v.strip()
                return SignalStrategyType(normalized)
            except ValueError as exc:
                raise ValueError("Strategy type must be a valid SignalStrategyType") from exc
        raise ValueError("Strategy type must be a valid SignalStrategyType")

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Ensure at least one trading symbol is provided."""
        if not v:
            raise ValueError("At least one symbol is required")
        normalized = []
        for symbol in v:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Symbols must be non-empty strings")
            normalized.append(symbol.strip().upper())
        return normalized

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator('performance_metrics')
    @classmethod
    def validate_performance_metrics(cls, v):
        """Validate performance metrics."""
        valid_metrics = [
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'calmar_ratio',
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid performance metric: {metric}")
        return v

    @field_validator('strategy_config', mode='before')
    @classmethod
    def validate_strategy_config(cls, v, info):
        """Validate and create strategy-specific configuration."""
        strategy_type = info.data.get('strategy_type')

        config_map = {
            SignalStrategyType.TECHNICAL_ANALYSIS: TechnicalAnalysisStrategyConfig,
            SignalStrategyType.ML_MODEL: MLModelStrategyConfig,
            SignalStrategyType.MOMENTUM: MomentumStrategyConfig,
            SignalStrategyType.MEAN_REVERSION: MeanReversionStrategyConfig,
            SignalStrategyType.ARBITRAGE: ArbitrageStrategyConfig,
            SignalStrategyType.CUSTOM: CustomStrategyConfig,
        }

        config_cls = config_map.get(strategy_type)
        if config_cls is None:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        config_instance = v if isinstance(v, config_cls) else config_cls(**v)

        outer_name = info.data.get('name')
        if outer_name:
            if hasattr(config_instance, 'strategy_name'):
                config_instance.strategy_name = outer_name
            if hasattr(config_instance, 'name'):
                config_instance.name = outer_name

        return config_instance

    @property
    def strategy_name(self) -> str:
        """Expose the underlying strategy configuration name."""
        name_attr = getattr(self.strategy_config, 'strategy_name', None)
        if isinstance(name_attr, str) and name_attr:
            return name_attr
        return self.name

    @property
    def strategy_parameters(self) -> dict[str, Any]:
        """Delegate strategy parameters to the underlying configuration."""
        params = getattr(self.strategy_config, 'strategy_parameters', None)
        if isinstance(params, dict):
            return params
        return self.parameters

    def get_strategy_parameters(self) -> dict[str, Any]:
        """Get strategy-specific parameters.

        Returns:
            Dictionary of strategy parameters
        """
        return self.strategy_parameters

    def get_indicators_config(self) -> list[IndicatorConfig]:
        """Get indicators configuration.

        Returns:
            List of indicator configurations
        """
        return self.strategy_config.indicators

    def get_signal_filters_config(self) -> list[SignalFilterConfig]:
        """Get signal filters configuration.

        Returns:
            List of signal filter configurations
        """
        return self.strategy_config.signal_filters

    def get_risk_parameters(self) -> RiskParameters:
        """Get risk parameters.

        Returns:
            Risk parameters
        """
        return self.strategy_config.risk_parameters

    def get_execution_parameters(self) -> ExecutionParameters:
        """Get execution parameters.

        Returns:
            Execution parameters
        """
        return self.strategy_config.execution_params

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'SignalStrategyConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary representation of configuration

        Returns:
            SignalStrategyConfig instance
        """
        return cls(**config_dict)

    def validate_configuration(self) -> bool:
        """Validate the complete configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate the configuration
            self.model_validate(self.model_dump())

            # Strategy-specific validation
            if hasattr(self.strategy_config, 'validate_configuration'):
                self.strategy_config.validate_configuration()

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def get_required_columns(self) -> list[str]:
        """Get list of required data columns.

        Returns:
            List of required column names
        """
        # Common required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Add columns required by indicators
        for indicator in self.get_indicators_config():
            source_column = getattr(indicator, 'source_column', indicator.price_column)
            if source_column not in required_columns:
                required_columns.append(source_column)

        return required_columns

    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information.

        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type.value,
            'enabled': self.enabled,
            'indicators_count': len(self.get_indicators_config()),
            'filters_count': len(self.get_signal_filters_config()),
            'risk_parameters': self.get_risk_parameters().model_dump(),
            'execution_parameters': self.get_execution_parameters().model_dump(),
            'log_level': self.log_level,
            'monitor_performance': self.monitor_performance,
            'performance_metrics': self.performance_metrics,
        }


# Type aliases for convenience
SignalStrategyConfigDict = dict[str, Any]
SignalStrategyConfigList = list[SignalStrategyConfig]

__all__ = [
    'SignalStrategyType',
    'ModelConfig',
    'SignalFilterConfig',
    'RiskParameters',
    'ExecutionParameters',
    'TechnicalAnalysisStrategyConfig',
    'MLModelStrategyConfig',
    'MomentumStrategyConfig',
    'MeanReversionStrategyConfig',
    'ArbitrageStrategyConfig',
    'CustomStrategyConfig',
    'SignalStrategyConfig',
    'IndicatorConfig',
]
