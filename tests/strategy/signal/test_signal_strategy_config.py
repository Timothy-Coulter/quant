"""Tests for signal strategy configuration models."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    MeanReversionStrategyConfig,
    MLModelStrategyConfig,
    ModelConfig,
    MomentumStrategyConfig,
    RiskParameters,
    SignalFilterConfig,
    SignalStrategyConfig,
    SignalStrategyType,
    TechnicalAnalysisStrategyConfig,
)


def make_indicator(
    name: str,
    indicator_type: str = "momentum",
    period: int = 14,
) -> IndicatorConfig:
    """Helper to build an indicator config with required fields."""
    return IndicatorConfig(
        indicator_name=name,
        indicator_type=indicator_type,
        period=period,
    )


def make_risk_parameters() -> RiskParameters:
    """Helper to build risk parameters with required fields."""
    return RiskParameters(
        max_position_size=0.1,
        stop_loss_percentage=0.05,
        take_profit_percentage=0.1,
        max_drawdown=0.2,
        risk_per_trade=0.02,
        correlation_limit=0.8,
    )


def make_execution_parameters() -> ExecutionParameters:
    """Helper to build execution parameters with required fields."""
    return ExecutionParameters(
        execution_type="market",
        slippage_percentage=0.001,
        commission_percentage=0.001,
        min_order_size=100.0,
        max_order_size=1000000.0,
        timeout_seconds=30,
    )


def make_ta_strategy_config(symbols: list[str]) -> TechnicalAnalysisStrategyConfig:
    """Helper to build a technical analysis strategy config for tests."""
    indicator = make_indicator(name="RSI")
    return TechnicalAnalysisStrategyConfig(
        symbols=symbols,
        indicators=[indicator],
        signal_filters=[],
        risk_parameters=make_risk_parameters(),
        execution_params=make_execution_parameters(),
    )


class TestSignalStrategyConfig:
    """Test SignalStrategyConfig model."""

    def test_signal_strategy_config_creation(self):
        """Test creating a basic signal strategy config."""
        strategy_config = make_ta_strategy_config(["AAPL", "GOOGL"])
        indicator = strategy_config.indicators[0]
        risk_params = strategy_config.risk_parameters
        execution_params = strategy_config.execution_params
        config = SignalStrategyConfig(
            name="test_strategy",
            strategy_type=SignalStrategyType.TECHNICAL_ANALYSIS,
            symbols=["AAPL", "GOOGL"],
            indicators=[indicator],
            signal_filters=[],
            risk_parameters=risk_params,
            execution_params=execution_params,
            strategy_config=strategy_config,
        )

        assert config.name == "test_strategy"
        assert config.strategy_type == SignalStrategyType.TECHNICAL_ANALYSIS
        assert config.symbols == ["AAPL", "GOOGL"]
        assert len(config.indicators) == 1
        assert config.indicators[0].indicator_name == "RSI"
        assert config.indicators[0].period == 14

    def test_signal_strategy_config_validation_invalid_name(self):
        """Test validation fails with invalid name."""
        with pytest.raises(ValidationError) as exc_info:
            SignalStrategyConfig(
                name="",  # Empty name
                strategy_type=SignalStrategyType.TECHNICAL_ANALYSIS,
                symbols=["AAPL"],
                indicators=make_ta_strategy_config(["AAPL"]).indicators,
                signal_filters=[],
                risk_parameters=make_risk_parameters(),
                execution_params=make_execution_parameters(),
                strategy_config=make_ta_strategy_config(["AAPL"]),
            )

        assert "name" in str(exc_info.value)

    def test_signal_strategy_config_validation_invalid_strategy_type(self):
        """Test validation fails with invalid strategy type."""
        with pytest.raises(ValidationError) as exc_info:
            SignalStrategyConfig(
                name="test_strategy",
                strategy_type=cast(Any, "invalid_type"),  # Invalid strategy type
                symbols=["AAPL"],
                indicators=make_ta_strategy_config(["AAPL"]).indicators,
                signal_filters=[],
                risk_parameters=make_risk_parameters(),
                execution_params=make_execution_parameters(),
                strategy_config=make_ta_strategy_config(["AAPL"]),
            )

        assert "strategy_type" in str(exc_info.value)

    def test_signal_strategy_config_validation_no_symbols(self):
        """Test validation fails with no symbols."""
        with pytest.raises(ValidationError) as exc_info:
            SignalStrategyConfig(
                name="test_strategy",
                strategy_type=SignalStrategyType.TECHNICAL_ANALYSIS,
                symbols=[],  # Empty symbols list
                indicators=make_ta_strategy_config(["AAPL"]).indicators,
                signal_filters=[],
                risk_parameters=make_risk_parameters(),
                execution_params=make_execution_parameters(),
                strategy_config=make_ta_strategy_config(["AAPL"]),
            )

        assert "symbols" in str(exc_info.value)


class TestTechnicalAnalysisStrategyConfig:
    """Test TechnicalAnalysisStrategyConfig model."""

    def test_creation(self):
        """Test creating technical analysis strategy config."""
        config = TechnicalAnalysisStrategyConfig(
            name="ta_strategy",
            strategy_type=SignalStrategyType.TECHNICAL_ANALYSIS,
            symbols=["AAPL"],
            indicators=[
                make_indicator(indicator_type="momentum", name="RSI"),
                make_indicator(indicator_type="trend", name="MACD", period=26),
            ],
            signal_filters=[],
            risk_parameters=make_risk_parameters(),
            execution_params=make_execution_parameters(),
            trend_indicators=["MA", "ADX"],
            momentum_indicators=["RSI", "Stochastic"],
            volatility_indicators=["BollingerBands", "ATR"],
            volume_indicators=["OBV", "VolumeSMA"],
            signal_generation_rules=["trend_following", "momentum_breakout"],
        )

        assert config.name == "ta_strategy"
        assert len(config.trend_indicators) == 2
        assert len(config.momentum_indicators) == 2
        assert len(config.volatility_indicators) == 2
        assert len(config.volume_indicators) == 2
        assert len(config.signal_generation_rules) == 2


class TestMLModelStrategyConfig:
    """Test MLModelStrategyConfig model."""

    def test_creation(self):
        """Test creating ML model strategy config."""
        config = MLModelStrategyConfig(
            name="ml_strategy",
            strategy_type=SignalStrategyType.ML_MODEL,
            symbols=["AAPL"],
            indicators=[],
            models=[
                ModelConfig(
                    model_name="random_forest",
                    model_type="classification",
                    framework="sklearn",
                    lookback_period=20,
                    train_test_split=0.8,
                    parameters={"n_estimators": 100},
                )
            ],
            signal_filters=[],
            risk_parameters=make_risk_parameters(),
            execution_params=make_execution_parameters(),
            model_ensemble_method="voting",
            retrain_frequency="monthly",
            feature_importance_threshold=0.01,
        )

        assert config.name == "ml_strategy"
        assert config.model_ensemble_method == "voting"
        assert config.retrain_frequency == "monthly"
        assert config.feature_importance_threshold == 0.01
        assert len(config.models) == 1
        assert config.models[0].model_name == "random_forest"


class TestMomentumStrategyConfig:
    """Test MomentumStrategyConfig model."""

    def test_creation(self):
        """Test creating momentum strategy config."""
        config = MomentumStrategyConfig(
            name="momentum_strategy",
            strategy_type=SignalStrategyType.MOMENTUM,
            symbols=["AAPL"],
            indicators=[],
            signal_filters=[],
            risk_parameters=make_risk_parameters(),
            execution_params=make_execution_parameters(),
            momentum_periods=[5, 10, 20],
            momentum_weighting="linear",
            momentum_threshold=0.02,
            trend_confirmation=True,
            volatility_filter=True,
            min_volume_threshold=1000000,
        )

        assert config.name == "momentum_strategy"
        assert config.momentum_periods == [5, 10, 20]
        assert config.momentum_weighting == "linear"
        assert config.momentum_threshold == 0.02
        assert config.trend_confirmation is True
        assert config.volatility_filter is True
        assert config.min_volume_threshold == 1000000


class TestMeanReversionStrategyConfig:
    """Test MeanReversionStrategyConfig model."""

    def test_creation(self):
        """Test creating mean reversion strategy config."""
        config = MeanReversionStrategyConfig(
            name="mean_reversion_strategy",
            strategy_type=SignalStrategyType.MEAN_REVERSION,
            symbols=["AAPL"],
            indicators=[],
            signal_filters=[],
            risk_parameters=make_risk_parameters(),
            execution_params=make_execution_parameters(),
            mean_periods=[20, 50],
            std_dev_periods=[2, 3],
            min_reversion_strength=0.1,
            hurst_threshold=0.5,
            volatility_adjustment=True,
            regime_filter=True,
            correlation_threshold=0.7,
        )

        assert config.name == "mean_reversion_strategy"
        assert config.mean_periods == [20, 50]
        assert config.std_dev_periods == [2, 3]
        assert config.min_reversion_strength == 0.1
        assert config.hurst_threshold == 0.5
        assert config.volatility_adjustment is True
        assert config.regime_filter is True
        assert config.correlation_threshold == 0.7


class TestIndicatorConfig:
    """Test IndicatorConfig model."""

    def test_creation(self):
        """Test creating indicator config."""
        config = IndicatorConfig(
            indicator_name="RSI",
            indicator_type="momentum",
            period=14,
            parameters={"overbought": 70, "oversold": 30},
        )

        assert config.indicator_name == "RSI"
        assert config.indicator_type == "momentum"
        assert config.period == 14
        assert config.parameters == {"overbought": 70, "oversold": 30}

    def test_validation_invalid_indicator_name(self):
        """Test validation fails with invalid indicator name."""
        with pytest.raises(ValidationError):
            IndicatorConfig(
                indicator_name="",  # Empty name
                indicator_type="momentum",
                period=14,
            )

    def test_validation_invalid_period(self):
        """Test validation fails with invalid period."""
        with pytest.raises(ValidationError):
            IndicatorConfig(
                indicator_name="RSI",
                indicator_type="momentum",
                period=0,  # Invalid period
            )


class TestModelConfig:
    """Test ModelConfig model."""

    def test_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            model_name="random_forest",
            model_type="classification",
            framework="sklearn",
            lookback_period=20,
            train_test_split=0.8,
            parameters={"n_estimators": 100, "max_depth": 10},
            data_config={"source": "yahoo", "symbols": ["AAPL"]},
        )

        assert config.model_name == "random_forest"
        assert config.model_type == "classification"
        assert config.framework == "sklearn"
        assert config.lookback_period == 20
        assert config.train_test_split == 0.8
        assert config.parameters == {"n_estimators": 100, "max_depth": 10}
        assert config.data_config == {"source": "yahoo", "symbols": ["AAPL"]}

    def test_validation_invalid_model_name(self):
        """Test validation fails with invalid model name."""
        with pytest.raises(ValidationError):
            ModelConfig(
                model_name="",  # Empty name
                model_type="classification",
                framework="sklearn",
                lookback_period=20,
            )


class TestSignalFilterConfig:
    """Test SignalFilterConfig model."""

    def test_creation(self):
        """Test creating signal filter config."""
        config = SignalFilterConfig(
            filter_name="volatility_filter",
            filter_type="threshold",
            parameters={"max_volatility": 0.05},
            enabled=True,
        )

        assert config.filter_name == "volatility_filter"
        assert config.filter_type == "threshold"
        assert config.parameters == {"max_volatility": 0.05}
        assert config.enabled is True

    def test_validation_invalid_filter_name(self):
        """Test validation fails with invalid filter name."""
        with pytest.raises(ValidationError):
            SignalFilterConfig(
                filter_name="",  # Empty name
                filter_type="threshold",
                parameters={},
            )


class TestRiskParameters:
    """Test RiskParameters model."""

    def test_creation(self):
        """Test creating risk parameters."""
        config = RiskParameters(
            max_position_size=0.1,
            stop_loss_percentage=0.02,
            take_profit_percentage=0.05,
            max_drawdown=0.2,
            risk_per_trade=0.01,
            correlation_limit=0.8,
        )

        assert config.max_position_size == 0.1
        assert config.stop_loss_percentage == 0.02
        assert config.take_profit_percentage == 0.05
        assert config.max_drawdown == 0.2
        assert config.risk_per_trade == 0.01
        assert config.correlation_limit == 0.8

    def test_validation_invalid_max_position_size(self):
        """Test validation fails with invalid max position size."""
        with pytest.raises(ValidationError):
            RiskParameters(max_position_size=1.5)  # Invalid size


class TestExecutionParameters:
    """Test ExecutionParameters model."""

    def test_creation(self):
        """Test creating execution parameters."""
        config = ExecutionParameters(
            execution_type="market",
            slippage_percentage=0.001,
            commission_percentage=0.001,
            min_order_size=100.0,
            max_order_size=1000000.0,
            timeout_seconds=30,
        )

        assert config.execution_type == "market"
        assert config.slippage_percentage == 0.001
        assert config.commission_percentage == 0.001
        assert config.min_order_size == 100.0
        assert config.max_order_size == 1000000.0
        assert config.timeout_seconds == 30

    def test_validation_invalid_execution_type(self):
        """Test validation fails with invalid execution type."""
        with pytest.raises(ValidationError):
            ExecutionParameters(execution_type="invalid_type")


class TestSignalStrategyType:
    """Test SignalStrategyType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert SignalStrategyType.TECHNICAL_ANALYSIS.value == "technical_analysis"
        assert SignalStrategyType.ML_MODEL.value == "ml_model"
        assert SignalStrategyType.MOMENTUM.value == "momentum"
        assert SignalStrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert SignalStrategyType.ARBITRAGE.value == "arbitrage"
        assert SignalStrategyType.CUSTOM.value == "custom"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        strategy_type = SignalStrategyType("technical_analysis")
        assert strategy_type == SignalStrategyType.TECHNICAL_ANALYSIS

    def test_enum_invalid_string(self):
        """Test invalid string raises error."""
        with pytest.raises(ValueError):
            SignalStrategyType("INVALID_TYPE")
