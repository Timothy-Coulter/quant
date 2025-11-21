"""Tests for technical analysis strategy."""

from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    RiskParameters,
    TechnicalAnalysisStrategyConfig,
)
from backtester.strategy.signal.technical_analysis_strategy import TechnicalAnalysisStrategy


class TestTechnicalAnalysisStrategy:
    """Test TechnicalAnalysisStrategy class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def basic_config(self):
        """Create a basic technical analysis strategy config."""
        return TechnicalAnalysisStrategyConfig(
            symbols=["AAPL", "GOOGL"],
            indicators=[
                IndicatorConfig(indicator_name="RSI", indicator_type="momentum", period=14),
                IndicatorConfig(indicator_name="MACD", indicator_type="momentum", period=26),
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

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)  # For reproducible results
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
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        assert strategy.config == basic_config
        assert strategy.name == basic_config.name
        assert strategy.trend_indicators == basic_config.trend_indicators
        assert strategy.momentum_indicators == basic_config.momentum_indicators
        assert strategy.volatility_indicators == basic_config.volatility_indicators
        assert strategy.volume_indicators == basic_config.volume_indicators
        assert strategy.signal_generation_rules == basic_config.signal_generation_rules

    def test_get_required_columns(self, basic_config, mock_event_bus):
        """Test getting required columns."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        required_columns = strategy.get_required_columns()

        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "adx",
            "di_plus",
            "di_minus",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
            "atr",
            "obv",
            "volume_sma",
        ]

        assert all(col in required_columns for col in expected_columns)

    @patch('backtester.strategy.signal.technical_analysis_strategy.IndicatorFactory')
    def test_generate_signals_with_indicators(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with indicators."""
        # Mock indicator factory
        mock_indicator = Mock()
        mock_indicator.name = "RSI"
        mock_indicator.calculate.return_value = pd.Series(
            [30, 45, 60, 70, 80], index=sample_market_data.index
        )
        mock_indicator.get_required_columns.return_value = ["close"]
        mock_indicator.get_signal_type.return_value = "BUY"
        mock_factory.create.return_value = mock_indicator

        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        signals = strategy.generate_signals(sample_market_data, "AAPL")

        # Check that indicators were created and calculated
        assert mock_factory.create.call_count == 2  # RSI and MACD
        mock_indicator.calculate.assert_called()

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_generate_signals_empty_data(self, basic_config, mock_event_bus):
        """Test generating signals with empty data."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        empty_data = pd.DataFrame()

        signals = strategy.generate_signals(empty_data, "AAPL")

        assert signals == []

    def test_generate_signals_insufficient_data(self, basic_config, mock_event_bus):
        """Test generating signals with insufficient data."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Create data with fewer rows than required
        short_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000000, 2000000],
            },
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )

        signals = strategy.generate_signals(short_data, "AAPL")

        assert signals == []

    def test_calculate_trend_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating trend signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock indicator data
        sample_market_data["ma_20"] = sample_market_data["close"].rolling(20).mean()
        sample_market_data["ma_50"] = sample_market_data["close"].rolling(50).mean()
        sample_market_data["adx"] = np.random.uniform(20, 40, len(sample_market_data))

        signals = strategy._calculate_trend_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_calculate_momentum_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating momentum signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock indicator data
        sample_market_data["rsi"] = np.random.uniform(20, 80, len(sample_market_data))
        sample_market_data["stochastic_k"] = np.random.uniform(20, 80, len(sample_market_data))
        sample_market_data["stochastic_d"] = np.random.uniform(20, 80, len(sample_market_data))

        signals = strategy._calculate_momentum_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_calculate_volatility_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating volatility signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock indicator data
        sample_market_data["bollinger_upper"] = sample_market_data["close"] * 1.02
        sample_market_data["bollinger_middle"] = sample_market_data["close"]
        sample_market_data["bollinger_lower"] = sample_market_data["close"] * 0.98
        sample_market_data["atr"] = np.random.uniform(1, 5, len(sample_market_data))

        signals = strategy._calculate_volatility_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_calculate_volume_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating volume signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock indicator data
        sample_market_data["obv"] = np.cumsum(
            np.random.uniform(-1000000, 1000000, len(sample_market_data))
        )
        sample_market_data["volume_sma"] = sample_market_data["volume"].rolling(20).mean()

        signals = strategy._calculate_volume_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_generate_trend_following_signals(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating trend following signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock trend indicators
        sample_market_data["ma_20"] = sample_market_data["close"].rolling(20).mean()
        sample_market_data["ma_50"] = sample_market_data["close"].rolling(50).mean()
        sample_market_data["adx"] = np.random.uniform(25, 35, len(sample_market_data))

        signals = strategy._generate_trend_following_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_generate_momentum_breakout_signals(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating momentum breakout signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock momentum indicators
        sample_market_data["rsi"] = np.random.uniform(30, 70, len(sample_market_data))
        sample_market_data["stochastic_k"] = np.random.uniform(30, 70, len(sample_market_data))
        sample_market_data["stochastic_d"] = np.random.uniform(30, 70, len(sample_market_data))

        signals = strategy._generate_momentum_breakout_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_generate_mean_reversion_signals(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating mean reversion signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock mean reversion indicators
        sample_market_data["bollinger_upper"] = sample_market_data["close"] * 1.02
        sample_market_data["bollinger_middle"] = sample_market_data["close"]
        sample_market_data["bollinger_lower"] = sample_market_data["close"] * 0.98

        signals = strategy._generate_mean_reversion_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_generate_volume_breakout_signals(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating volume breakout signals."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Mock volume indicators
        sample_market_data["obv"] = np.cumsum(
            np.random.uniform(-1000000, 1000000, len(sample_market_data))
        )
        sample_market_data["volume_sma"] = sample_market_data["volume"].rolling(20).mean()

        signals = strategy._generate_volume_breakout_signals(sample_market_data)

        assert isinstance(signals, list)
        if signals:
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)

    def test_combine_signals(self, basic_config, mock_event_bus):
        """Test combining signals from different sources."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        signals1 = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        signals2 = [
            {"signal_type": "BUY", "confidence": 0.8, "action": "Enter long"},
        ]

        combined = strategy._combine_signals(signals1, signals2)

        assert isinstance(combined, list)
        assert len(combined) == 3  # All signals should be combined

    def test_aggregate_signals(self, basic_config, mock_event_bus):
        """Test aggregating signals with voting."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "BUY", "confidence": 0.8, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        aggregated = strategy._aggregate_signals(signals)

        assert isinstance(aggregated, list)
        if aggregated:
            assert all("signal_type" in signal for signal in aggregated)
            assert all("action" in signal for signal in aggregated)

    def test_validate_signal_data_valid(self, basic_config, mock_event_bus, sample_market_data):
        """Test validating signal data with valid data."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Add required columns
        required_columns = strategy.get_required_columns()
        for col in required_columns:
            if col not in sample_market_data.columns:
                sample_market_data[col] = np.random.uniform(0, 100, len(sample_market_data))

        result = strategy.validate_signal_data(sample_market_data)

        assert result is True

    def test_validate_signal_data_missing_columns(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test validating signal data with missing columns."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Remove required columns
        sample_market_data = sample_market_data.drop(columns=["close", "volume"])

        result = strategy.validate_signal_data(sample_market_data)

        assert result is False

    def test_validate_signal_data_insufficient_rows(self, basic_config, mock_event_bus):
        """Test validating signal data with insufficient rows."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Create data with too few rows
        small_data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000000, 2000000],
            }
        )

        result = strategy.validate_signal_data(small_data)

        assert result is False

    def test_get_strategy_info(self, basic_config, mock_event_bus):
        """Test getting strategy information."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        info = strategy.get_strategy_info()

        assert info["name"] == basic_config.name
        assert info["type"] == "TECHNICAL_ANALYSIS"
        assert info["trend_indicators"] == basic_config.trend_indicators
        assert info["momentum_indicators"] == basic_config.momentum_indicators
        assert info["volatility_indicators"] == basic_config.volatility_indicators
        assert info["volume_indicators"] == basic_config.volume_indicators
        assert info["signal_generation_rules"] == basic_config.signal_generation_rules

    def test_reset(self, basic_config, mock_event_bus):
        """Test resetting strategy state."""
        strategy = TechnicalAnalysisStrategy(basic_config, mock_event_bus)

        # Add some state
        strategy.signal_history = [{"test": "data"}]
        strategy.valid_signal_count = 5
        strategy.invalid_signal_count = 2

        strategy.reset()

        assert strategy.signal_history == []
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0
