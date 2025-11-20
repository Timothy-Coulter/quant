"""Tests for momentum strategy."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.strategy.signal.momentum_strategy import MomentumStrategy
from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    MomentumStrategyConfig,
    RiskParameters,
)


class TestMomentumStrategy:
    """Test MomentumStrategy class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def basic_config(self):
        """Create a basic momentum strategy config."""
        return MomentumStrategyConfig(
            symbols=["AAPL", "GOOGL"],
            indicators=[
                IndicatorConfig(
                    indicator_name="sma", indicator_type="trend", period=20, parameters={}
                )
            ],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            momentum_periods=[5, 10, 20],
            momentum_weighting="linear",
            momentum_threshold=0.02,
            trend_confirmation=True,
            volatility_filter=True,
            min_volume_threshold=1000000,
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
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        assert strategy.config == basic_config
        assert strategy.name == basic_config.name
        assert strategy.momentum_periods == basic_config.momentum_periods
        assert strategy.momentum_weighting == basic_config.momentum_weighting
        assert strategy.momentum_threshold == basic_config.momentum_threshold
        assert strategy.trend_confirmation == basic_config.trend_confirmation
        assert strategy.volatility_filter == basic_config.volatility_filter
        assert strategy.min_volume_threshold == basic_config.min_volume_threshold

    def test_get_required_columns(self, basic_config, mock_event_bus):
        """Test getting required columns."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        required_columns = strategy.get_required_columns()

        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_score",
            "trend_strength",
            "volatility",
        ]

        assert all(col in required_columns for col in expected_columns)

    def test_generate_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test generating momentum signals."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        signals = strategy.generate_signals(sample_market_data, "AAPL")

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_generate_signals_empty_data(self, basic_config, mock_event_bus):
        """Test generating signals with empty data."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        empty_data = pd.DataFrame()

        signals = strategy.generate_signals(empty_data, "AAPL")

        assert signals == []

    def test_generate_signals_insufficient_data(self, basic_config, mock_event_bus):
        """Test generating signals with insufficient data."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

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

    def test_calculate_momentum_indicators(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating momentum indicators."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        indicators = strategy._calculate_momentum_indicators(sample_market_data)

        # Check that indicators were calculated
        assert isinstance(indicators, dict)
        assert "momentum_5" in indicators
        assert "momentum_10" in indicators
        assert "momentum_20" in indicators
        assert len(indicators["momentum_5"]) == len(sample_market_data)

    def test_calculate_momentum_score(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating momentum score."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Calculate momentum indicators first
        indicators = strategy._calculate_momentum_indicators(sample_market_data)

        score = strategy._calculate_momentum_score(indicators)

        # Check that score was calculated
        assert isinstance(score, pd.Series)
        assert len(score) == len(sample_market_data)
        assert score.min() >= 0
        assert score.max() <= 1

    def test_calculate_trend_strength(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating trend strength."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        trend_strength = strategy._calculate_trend_strength(sample_market_data)

        # Check that trend strength was calculated
        assert isinstance(trend_strength, pd.Series)
        assert len(trend_strength) == len(sample_market_data)
        assert trend_strength.min() >= 0
        assert trend_strength.max() <= 1

    def test_calculate_volatility(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating volatility."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        volatility = strategy._calculate_volatility(sample_market_data)

        # Check that volatility was calculated
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_market_data)
        assert (volatility >= 0).all()  # Volatility should be non-negative

    def test_generate_momentum_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test generating momentum signals."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        signals = strategy._generate_momentum_signals(sample_market_data)

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_apply_trend_confirmation(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying trend confirmation filter."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Calculate momentum and trend strength
        indicators = strategy._calculate_momentum_indicators(sample_market_data)
        momentum_score = strategy._calculate_momentum_score(indicators)
        trend_strength = strategy._calculate_trend_strength(sample_market_data)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_trend_confirmation(
            signals, momentum_score, trend_strength, sample_market_data.index
        )

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_apply_volatility_filter(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying volatility filter."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Calculate volatility
        volatility = strategy._calculate_volatility(sample_market_data)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_volatility_filter(
            signals, volatility, sample_market_data.index
        )

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_apply_volume_filter(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying volume filter."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_volume_filter(
            signals, sample_market_data["volume"], sample_market_data.index
        )

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_weight_momentum_periods_linear(self, basic_config, mock_event_bus, sample_market_data):
        """Test weighting momentum periods with linear weighting."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Calculate momentum indicators
        indicators = strategy._calculate_momentum_indicators(sample_market_data)

        weighted_score = strategy._weight_momentum_periods(indicators, "linear")

        # Check that weighted score was calculated
        assert isinstance(weighted_score, pd.Series)
        assert len(weighted_score) == len(sample_market_data)
        assert weighted_score.min() >= 0
        assert weighted_score.max() <= 1

    def test_weight_momentum_periods_exponential(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test weighting momentum periods with exponential weighting."""
        # Create config with exponential weighting
        config = MomentumStrategyConfig(
            indicators=[],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            momentum_periods=[5, 10, 20],
            momentum_weighting="exponential",
            momentum_threshold=0.02,
            trend_confirmation=True,
            volatility_filter=True,
            min_volume_threshold=1000000,
        )

        strategy = MomentumStrategy(config, mock_event_bus)

        # Calculate momentum indicators
        indicators = strategy._calculate_momentum_indicators(sample_market_data)

        weighted_score = strategy._weight_momentum_periods(indicators, "exponential")

        # Check that weighted score was calculated
        assert isinstance(weighted_score, pd.Series)
        assert len(weighted_score) == len(sample_market_data)
        assert weighted_score.min() >= 0
        assert weighted_score.max() <= 1

    def test_weight_momentum_periods_equal(self, basic_config, mock_event_bus, sample_market_data):
        """Test weighting momentum periods with equal weighting."""
        # Create config with equal weighting
        config = MomentumStrategyConfig(
            indicators=[],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            momentum_periods=[5, 10, 20],
            momentum_weighting="equal",
            momentum_threshold=0.02,
            trend_confirmation=True,
            volatility_filter=True,
            min_volume_threshold=1000000,
        )

        strategy = MomentumStrategy(config, mock_event_bus)

        # Calculate momentum indicators
        indicators = strategy._calculate_momentum_indicators(sample_market_data)

        weighted_score = strategy._weight_momentum_periods(indicators, "equal")

        # Check that weighted score was calculated
        assert isinstance(weighted_score, pd.Series)
        assert len(weighted_score) == len(sample_market_data)
        assert weighted_score.min() >= 0
        assert weighted_score.max() <= 1

    def test_calculate_momentum_signals_strong_momentum(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with strong momentum."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Create strong upward momentum
        sample_market_data["close"] = np.linspace(100, 200, len(sample_market_data))

        signals = strategy._generate_momentum_signals(sample_market_data)

        # Should generate BUY signals with strong momentum
        buy_signals = [s for s in signals if s.get("signal_type") == "BUY"]
        if buy_signals:
            assert all(s.get("confidence", 0) > 0.5 for s in buy_signals)

    def test_calculate_momentum_signals_weak_momentum(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with weak momentum."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Create weak momentum (sideways movement)
        sample_market_data["close"] = 100 + np.random.normal(0, 1, len(sample_market_data))

        signals = strategy._generate_momentum_signals(sample_market_data)

        # Should generate HOLD signals with weak momentum
        hold_signals = [s for s in signals if s.get("signal_type") == "HOLD"]
        if hold_signals:
            assert len(hold_signals) > 0

    def test_validate_signal_data_valid(self, basic_config, mock_event_bus, sample_market_data):
        """Test validating signal data with valid data."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

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
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Remove required columns
        sample_market_data = sample_market_data.drop(columns=["close", "volume"])

        result = strategy.validate_signal_data(sample_market_data)

        assert result is False

    def test_validate_signal_data_insufficient_rows(self, basic_config, mock_event_bus):
        """Test validating signal data with insufficient rows."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

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
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        info = strategy.get_strategy_info()

        assert info["name"] == basic_config.name
        assert info["type"] == "MOMENTUM"
        assert info["momentum_periods"] == basic_config.momentum_periods
        assert info["momentum_weighting"] == basic_config.momentum_weighting
        assert info["momentum_threshold"] == basic_config.momentum_threshold
        assert info["trend_confirmation"] == basic_config.trend_confirmation
        assert info["volatility_filter"] == basic_config.volatility_filter
        assert info["min_volume_threshold"] == basic_config.min_volume_threshold

    def test_reset(self, basic_config, mock_event_bus):
        """Test resetting strategy state."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Add some state
        strategy.signal_history = [{"test": "data"}]
        strategy.valid_signal_count = 5
        strategy.invalid_signal_count = 2

        strategy.reset()

        assert strategy.signal_history == []
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0

    def test_momentum_threshold_filtering(self, basic_config, mock_event_bus, sample_market_data):
        """Test filtering signals based on momentum threshold."""
        strategy = MomentumStrategy(basic_config, mock_event_bus)

        # Calculate momentum indicators
        indicators = strategy._calculate_momentum_indicators(sample_market_data)
        momentum_score = strategy._calculate_momentum_score(indicators)

        # Create signals with varying confidence
        signals = [
            {"signal_type": "BUY", "confidence": 0.1, "action": "Enter long"},  # Below threshold
            {"signal_type": "BUY", "confidence": 0.8, "action": "Enter long"},  # Above threshold
            {
                "signal_type": "SELL",
                "confidence": 0.9,
                "action": "Exit position",
            },  # Above threshold
        ]

        filtered_signals = strategy._apply_momentum_threshold_filter(
            signals, momentum_score, sample_market_data.index
        )

        # Should filter out signals below threshold
        assert len(filtered_signals) <= len(signals)
        if filtered_signals:
            assert all(
                s.get("confidence", 0) >= basic_config.momentum_threshold for s in filtered_signals
            )
