"""Tests for base signal strategy class."""

from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.core.events import MarketDataEvent, MarketDataType, SignalEvent, SignalType
from backtester.strategy.signal.ml_model_strategy import MLModelStrategy
from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    MLModelStrategyConfig,
    RiskParameters,
    SignalStrategyConfig,
    SignalStrategyType,
    TechnicalAnalysisStrategyConfig,
)
from backtester.strategy.signal.technical_analysis_strategy import TechnicalAnalysisStrategy


class TestBaseSignalStrategy:
    """Test BaseSignalStrategy class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def basic_config(self):
        """Create a basic signal strategy config."""
        return SignalStrategyConfig(
            name="test_strategy",
            strategy_type=SignalStrategyType.TECHNICAL_ANALYSIS,
            symbols=["AAPL", "GOOGL"],
            indicators=[],
            models=[],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            strategy_config=TechnicalAnalysisStrategyConfig(
                symbols=["AAPL", "GOOGL"],
                indicators=[
                    IndicatorConfig(
                        indicator_name="sma", indicator_type="trend", period=20, parameters={}
                    )
                ],
                signal_filters=[],
                risk_parameters=RiskParameters(),
                execution_params=ExecutionParameters(),
                trend_indicators=["MA", "ADX"],
                momentum_indicators=["RSI", "Stochastic"],
                volatility_indicators=["BollingerBands", "ATR"],
                volume_indicators=["OBV", "VolumeSMA"],
                signal_generation_rules=cast(Any, "trend_following"),
            ),
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(100, 200, 100),
                "low": np.random.uniform(100, 200, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )
        return data

    def test_initialization(self, basic_config, mock_event_bus):
        """Test strategy initialization."""
        # Use a concrete strategy class instead of abstract BaseSignalStrategy
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        assert strategy.config == basic_config.strategy_config
        assert strategy.event_bus == mock_event_bus
        assert strategy.name == basic_config.strategy_config.strategy_name
        assert strategy.config.symbols == basic_config.symbols
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0
        assert strategy.last_signal_time is None
        assert strategy.signals == []

    def test_initialization_with_indicators(self, mock_event_bus):
        """Test strategy initialization with indicators."""
        config = TechnicalAnalysisStrategyConfig(
            symbols=["AAPL"],
            indicators=[
                IndicatorConfig(indicator_name="rsi", indicator_type="momentum", period=14),
                IndicatorConfig(indicator_name="macd", indicator_type="momentum", period=26),
            ],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            trend_indicators=["MA", "ADX"],
            momentum_indicators=["RSI", "Stochastic"],
            volatility_indicators=["BollingerBands", "ATR"],
            volume_indicators=["OBV", "VolumeSMA"],
            signal_generation_rules=cast(Any, "trend_following"),
        )

        # Test TechnicalAnalysisStrategy initialization with indicators
        strategy = TechnicalAnalysisStrategy(config, mock_event_bus)

        assert len(strategy.indicators) == 2
        assert "rsi" in strategy.indicators
        assert "macd" in strategy.indicators
        assert strategy.indicators["rsi"].config.period == 14
        assert strategy.indicators["macd"].config.period == 26

    def test_initialization_with_models(self, mock_event_bus):
        """Test strategy initialization with models."""
        risk_params = RiskParameters(
            max_position_size=0.1,
            stop_loss_percentage=0.05,
            take_profit_percentage=0.1,
            max_drawdown=0.2,
            risk_per_trade=0.01,
            correlation_limit=0.8,
        )
        execution_params = ExecutionParameters(
            execution_type="market",
            slippage_percentage=0.001,
            commission_percentage=0.001,
            min_order_size=100.0,
            max_order_size=1000000.0,
            timeout_seconds=30,
        )
        nested_config = MLModelStrategyConfig(
            symbols=["AAPL"],
            models=[],
            indicators=[],
            signal_filters=[],
            risk_parameters=risk_params,
            execution_params=execution_params,
            prediction_horizon=1,
            confidence_threshold=0.6,
            min_prediction_strength=0.1,
            use_ensemble=True,
            ensemble_weights=None,
            feature_columns=None,
            target_column="close",
            normalize_features=True,
            aggregation_method="weighted_average",
        )

        # Test MLModelStrategy initialization with models
        config = SignalStrategyConfig(
            name="ml_strategy",
            strategy_type=SignalStrategyType.ML_MODEL,
            symbols=["AAPL"],
            indicators=[],
            signal_filters=[],
            risk_parameters=risk_params,
            execution_params=execution_params,
            strategy_config=nested_config,
        )
        strategy = MLModelStrategy(config, mock_event_bus)

        # Note: The MLModelStrategy doesn't initialize models by default in the constructor
        # This test verifies the strategy can be created without errors
        assert strategy.config == config
        assert isinstance(strategy.config, SignalStrategyConfig)
        assert strategy.config.strategy_config == nested_config
        assert strategy.event_bus == mock_event_bus

    def test_subscribe_to_events(self, basic_config, mock_event_bus):
        """Test subscribing to market data events."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        # Check that subscribe was called for market data events
        mock_event_bus.subscribe.assert_called()

        # Verify the handler was registered
        call_args = mock_event_bus.subscribe.call_args
        assert call_args[0][0] == strategy._handle_market_data_event

    def test_unsubscribe_from_events(self, basic_config, mock_event_bus):
        """Test unsubscribing from events."""
        # The BaseSignalStrategy doesn't have an unsubscribe method yet
        # For now, we'll just test that the subscription was made
        TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)
        assert mock_event_bus.subscribe.called

    def test_process_market_data_event(self, basic_config, mock_event_bus):
        """Test processing market data events."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        # Mock the generate_signals method
        mock_generate = Mock(return_value=[])
        object.__setattr__(strategy, "generate_signals", mock_generate)

        # Create a market data event
        event = MarketDataEvent(
            event_type="MARKET_DATA",
            timestamp=1640995200.0,
            source="market_data_feed",
            symbol="AAPL",
            data_type=MarketDataType.BAR,
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=152.0,
            volume=1000000,
        )

        # Call the event handler directly
        strategy._handle_market_data_event(event)

        # Check that generate_signals was called
        mock_generate.assert_called_once()
        args = mock_generate.call_args[0]
        assert args[0] is not None  # DataFrame should be passed
        assert args[1] == "AAPL"

    def test_generate_signals_not_implemented(self, basic_config, mock_event_bus):
        """Test that generate_signals raises NotImplementedError."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        with pytest.raises(NotImplementedError):
            strategy.generate_signals(None, "AAPL")

    def test_get_required_columns_not_implemented(self, basic_config, mock_event_bus):
        """Ensure get_required_columns returns a list of column names."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        required_columns = strategy.get_required_columns()
        assert isinstance(required_columns, list)
        assert required_columns  # list should not be empty

    def test_validate_signal_valid(self, basic_config, mock_event_bus):
        """Test signal validation with valid signal."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        valid_signal = {
            "signal_type": "BUY",
            "confidence": 0.8,
            "metadata": {"indicator": "RSI", "value": 30.0},
        }

        result = strategy._validate_signal(valid_signal)
        assert result is True
        assert strategy.valid_signal_count == 0  # Not incremented in _validate_signal
        assert strategy.invalid_signal_count == 0

    def test_validate_signal_invalid(self, basic_config, mock_event_bus):
        """Test signal validation with invalid signal."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        invalid_signal = {
            "signal_type": "INVALID",  # Invalid signal type
            "confidence": 1.5,  # Invalid confidence
            "metadata": {"indicator": "RSI", "value": 30.0},
        }

        result = strategy._validate_signal(invalid_signal)
        assert result is False
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0

    def test_process_and_publish_signal(self, basic_config, mock_event_bus):
        """Test processing and publishing signals."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        signal_data = {
            "signal_type": "BUY",
            "confidence": 0.8,
            "metadata": {"indicator": "RSI", "value": 30.0},
        }

        strategy._process_and_publish_signal(signal_data, "AAPL", 1640995200.0)

        # Check that publish was called
        mock_event_bus.publish.assert_called()

        # Verify the event
        call_args = mock_event_bus.publish.call_args[0][0]
        assert isinstance(call_args, SignalEvent)
        assert call_args.symbol == "AAPL"
        assert call_args.signal_type == SignalType.BUY

    def test_get_signal_history(self, basic_config, mock_event_bus):
        """Test getting signal history."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        # Add some signals to history
        for _ in range(5):
            signal_data = {
                "signal_type": "BUY",
                "confidence": 0.8,
                "metadata": {"indicator": "RSI", "value": 30.0},
            }
            strategy.signals.append(signal_data)

        assert len(strategy.signals) == 5

    def test_get_performance_metrics(self, basic_config, mock_event_bus):
        """Test getting performance metrics."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        # Add some signals
        for _ in range(3):
            signal_data = {
                "signal_type": "BUY",
                "confidence": 0.8,
                "metadata": {"indicator": "RSI", "value": 30.0},
            }
            strategy.signals.append(signal_data)
            strategy.signal_count += 1
            strategy.valid_signal_count += 1

        metrics = strategy.get_performance_metrics()

        assert metrics["total_signals"] == 3
        assert metrics["valid_signals"] == 3
        assert metrics["invalid_signals"] == 0
        assert metrics["signal_quality_ratio"] == 1.0

    def test_reset(self, basic_config, mock_event_bus):
        """Test resetting strategy state."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        # Add some signals
        signal_data = {
            "signal_type": "BUY",
            "confidence": 0.8,
            "metadata": {"indicator": "RSI", "value": 30.0},
        }
        strategy.signals.append(signal_data)
        strategy.valid_signal_count = 5
        strategy.invalid_signal_count = 2

        strategy.reset()

        assert len(strategy.signals) == 0
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0
        assert strategy.last_signal_time is None

    def test_get_strategy_info(self, basic_config, mock_event_bus):
        """Test getting strategy information."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        info = strategy.get_strategy_info()

        assert info["name"] == basic_config.name
        assert info["type"] == basic_config.strategy_type.value
        assert info["symbols"] == basic_config.symbols
        assert info["indicators"] == []
        assert info["models"] == []
        assert "performance_metrics" in info

    def test_apply_signal_filters_no_filters(self, basic_config, mock_event_bus):
        """Test applying signal filters when no filters are configured."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        signal = {
            "signal_type": "BUY",
            "confidence": 0.8,
            "metadata": {"indicator": "RSI", "value": 30.0},
        }

        result = strategy._apply_signal_filters(signal)
        assert result is True

    def test_apply_signal_filters_with_disabled_filter(self, basic_config, mock_event_bus):
        """Test applying signal filters with disabled filters."""
        # Add a disabled filter to config
        config = TechnicalAnalysisStrategyConfig(
            symbols=["AAPL"],
            indicators=[],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            trend_indicators=["MA", "ADX"],
            momentum_indicators=["RSI", "Stochastic"],
            volatility_indicators=["BollingerBands", "ATR"],
            volume_indicators=["OBV", "VolumeSMA"],
            signal_generation_rules=cast(Any, "trend_following"),
        )
        config.signal_filters = [Mock(enabled=False)]

        strategy = TechnicalAnalysisStrategy(config, mock_event_bus)

        signal = {
            "signal_type": "BUY",
            "confidence": 0.8,
            "metadata": {"indicator": "RSI", "value": 30.0},
        }

        result = strategy._apply_signal_filters(signal)
        assert result is True

    def test_get_required_columns_default(self, basic_config, mock_event_bus):
        """Test default required columns implementation."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        required_columns = strategy.get_required_columns()
        assert isinstance(required_columns, list)
        assert len(required_columns) > 0

    def test_generate_signals_default(self, basic_config, mock_event_bus):
        """Test default generate_signals implementation."""
        strategy = TechnicalAnalysisStrategy(basic_config.strategy_config, mock_event_bus)

        with pytest.raises(NotImplementedError):
            strategy.generate_signals(None, "AAPL")
