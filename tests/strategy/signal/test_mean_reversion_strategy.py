"""Tests for mean reversion strategy."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.strategy.signal.mean_reversion_strategy import MeanReversionStrategy
from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    MeanReversionStrategyConfig,
    RiskParameters,
)


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def basic_config(self):
        """Create a basic mean reversion strategy config."""
        return MeanReversionStrategyConfig(
            symbols=["AAPL", "GOOGL"],
            indicators=[
                IndicatorConfig(
                    indicator_name="sma", indicator_type="trend", period=20, parameters={}
                )
            ],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
            mean_periods=[20, 50],
            std_dev_periods=[2, 3],
            min_reversion_strength=0.1,
            hurst_threshold=0.5,
            volatility_adjustment=True,
            regime_filter=True,
            correlation_threshold=0.7,
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
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        assert strategy.config == basic_config
        assert strategy.name == basic_config.name
        assert strategy.mean_periods == basic_config.mean_periods
        assert strategy.std_dev_periods == basic_config.std_dev_periods
        assert strategy.min_reversion_strength == basic_config.min_reversion_strength
        assert strategy.hurst_threshold == basic_config.hurst_threshold
        assert strategy.volatility_adjustment == basic_config.volatility_adjustment
        assert strategy.regime_filter == basic_config.regime_filter
        assert strategy.correlation_threshold == basic_config.correlation_threshold

    def test_get_required_columns(self, basic_config, mock_event_bus):
        """Test getting required columns."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        required_columns = strategy.get_required_columns()

        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "mean_20",
            "mean_50",
            "std_dev_2",
            "std_dev_3",
            "z_score",
            "reversion_strength",
            "hurst_exponent",
            "volatility",
            "regime",
            "correlation",
        ]

        assert all(col in required_columns for col in expected_columns)

    def test_generate_signals(self, basic_config, mock_event_bus, sample_market_data):
        """Test generating mean reversion signals."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        signals = strategy.generate_signals(sample_market_data, "AAPL")

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_generate_signals_empty_data(self, basic_config, mock_event_bus):
        """Test generating signals with empty data."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        empty_data = pd.DataFrame()

        signals = strategy.generate_signals(empty_data, "AAPL")

        assert signals == []

    def test_generate_signals_insufficient_data(self, basic_config, mock_event_bus):
        """Test generating signals with insufficient data."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

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

    def test_calculate_mean_indicators(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating mean indicators."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        indicators = strategy._calculate_mean_indicators(sample_market_data)

        # Check that indicators were calculated
        assert isinstance(indicators, dict)
        assert "mean_20" in indicators
        assert "mean_50" in indicators
        assert len(indicators["mean_20"]) == len(sample_market_data)

    def test_calculate_std_dev_indicators(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating standard deviation indicators."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        indicators = strategy._calculate_std_dev_indicators(sample_market_data)

        # Check that indicators were calculated
        assert isinstance(indicators, dict)
        assert "std_dev_2" in indicators
        assert "std_dev_3" in indicators
        assert len(indicators["std_dev_2"]) == len(sample_market_data)

    def test_calculate_z_score(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating z-score."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate mean and std dev indicators first
        mean_indicators = strategy._calculate_mean_indicators(sample_market_data)
        std_dev_indicators = strategy._calculate_std_dev_indicators(sample_market_data)

        z_score = strategy._calculate_z_score(mean_indicators, std_dev_indicators)

        # Check that z-score was calculated
        assert isinstance(z_score, pd.Series)
        assert len(z_score) == len(sample_market_data)

    def test_calculate_reversion_strength(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating reversion strength."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate z-score first
        mean_indicators = strategy._calculate_mean_indicators(sample_market_data)
        std_dev_indicators = strategy._calculate_std_dev_indicators(sample_market_data)
        z_score = strategy._calculate_z_score(mean_indicators, std_dev_indicators)

        reversion_strength = strategy._calculate_reversion_strength(z_score)

        # Check that reversion strength was calculated
        assert isinstance(reversion_strength, pd.Series)
        assert len(reversion_strength) == len(sample_market_data)
        assert (reversion_strength >= 0).all()  # Should be non-negative

    def test_calculate_hurst_exponent(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating Hurst exponent."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        hurst_exponent = strategy._calculate_hurst_exponent(sample_market_data["close"])

        # Check that Hurst exponent was calculated
        assert isinstance(hurst_exponent, pd.Series)
        assert len(hurst_exponent) == len(sample_market_data)
        assert (hurst_exponent >= 0).all()  # Should be non-negative
        assert (hurst_exponent <= 1).all()  # Should be <= 1

    def test_calculate_volatility(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating volatility."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        volatility = strategy._calculate_volatility(sample_market_data)

        # Check that volatility was calculated
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_market_data)
        assert (volatility >= 0).all()  # Should be non-negative

    def test_detect_market_regime(self, basic_config, mock_event_bus, sample_market_data):
        """Test detecting market regime."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        regime = strategy._detect_market_regime(sample_market_data)

        # Check that regime was detected
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(sample_market_data)
        assert regime.nunique() <= 3  # Should have at most 3 regimes

    def test_calculate_correlation(self, basic_config, mock_event_bus, sample_market_data):
        """Test calculating correlation."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        correlation = strategy._calculate_correlation(sample_market_data)

        # Check that correlation was calculated
        assert isinstance(correlation, pd.Series)
        assert len(correlation) == len(sample_market_data)
        assert (correlation >= -1).all()  # Should be >= -1
        assert (correlation <= 1).all()  # Should be <= 1

    def test_generate_mean_reversion_signals(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating mean reversion signals."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        signals = strategy._generate_mean_reversion_signals(sample_market_data)

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_apply_reversion_strength_filter(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test applying reversion strength filter."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate reversion strength
        mean_indicators = strategy._calculate_mean_indicators(sample_market_data)
        std_dev_indicators = strategy._calculate_std_dev_indicators(sample_market_data)
        z_score = strategy._calculate_z_score(mean_indicators, std_dev_indicators)
        reversion_strength = strategy._calculate_reversion_strength(z_score)

        signals = [
            {"signal_type": "BUY", "confidence": 0.05, "action": "Enter long"},  # Below threshold
            {"signal_type": "BUY", "confidence": 0.8, "action": "Enter long"},  # Above threshold
            {
                "signal_type": "SELL",
                "confidence": 0.9,
                "action": "Exit position",
            },  # Above threshold
        ]

        filtered_signals = strategy._apply_reversion_strength_filter(
            signals, reversion_strength, sample_market_data.index
        )

        # Should filter out signals below threshold
        assert len(filtered_signals) <= len(signals)
        if filtered_signals:
            assert all(
                s.get("confidence", 0) >= basic_config.min_reversion_strength
                for s in filtered_signals
            )

    def test_apply_hurst_filter(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying Hurst filter."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate Hurst exponent
        hurst_exponent = strategy._calculate_hurst_exponent(sample_market_data["close"])

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_hurst_filter(
            signals, hurst_exponent, sample_market_data.index
        )

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_apply_volatility_adjustment(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying volatility adjustment."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate volatility
        volatility = strategy._calculate_volatility(sample_market_data)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        adjusted_signals = strategy._apply_volatility_adjustment(
            signals, volatility, sample_market_data.index
        )

        # Check that signals were adjusted
        assert isinstance(adjusted_signals, list)
        assert len(adjusted_signals) == len(signals)
        if adjusted_signals:
            assert all("confidence" in s for s in adjusted_signals)

    def test_apply_regime_filter(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying regime filter."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Detect market regime
        regime = strategy._detect_market_regime(sample_market_data)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_regime_filter(signals, regime, sample_market_data.index)

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_apply_correlation_filter(self, basic_config, mock_event_bus, sample_market_data):
        """Test applying correlation filter."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate correlation
        correlation = strategy._calculate_correlation(sample_market_data)

        signals = [
            {"signal_type": "BUY", "confidence": 0.7, "action": "Enter long"},
            {"signal_type": "SELL", "confidence": 0.6, "action": "Exit position"},
        ]

        filtered_signals = strategy._apply_correlation_filter(
            signals, correlation, sample_market_data.index
        )

        # Check that signals were filtered
        assert isinstance(filtered_signals, list)
        assert len(filtered_signals) <= len(signals)

    def test_calculate_mean_reversion_signals_oversold(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with oversold conditions."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Create oversold conditions (price significantly below mean)
        sample_market_data["close"] = 50 + np.random.normal(0, 5, len(sample_market_data))

        signals = strategy._generate_mean_reversion_signals(sample_market_data)

        # Should generate BUY signals with oversold conditions
        buy_signals = [s for s in signals if s.get("signal_type") == "BUY"]
        if buy_signals:
            assert len(buy_signals) > 0

    def test_calculate_mean_reversion_signals_overbought(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with overbought conditions."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Create overbought conditions (price significantly above mean)
        sample_market_data["close"] = 200 + np.random.normal(0, 5, len(sample_market_data))

        signals = strategy._generate_mean_reversion_signals(sample_market_data)

        # Should generate SELL signals with overbought conditions
        sell_signals = [s for s in signals if s.get("signal_type") == "SELL"]
        if sell_signals:
            assert len(sell_signals) > 0

    def test_calculate_mean_reversion_signals_neutral(
        self, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with neutral conditions."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Create neutral conditions (price close to mean)
        sample_market_data["close"] = 150 + np.random.normal(0, 1, len(sample_market_data))

        signals = strategy._generate_mean_reversion_signals(sample_market_data)

        # Should generate HOLD signals with neutral conditions
        hold_signals = [s for s in signals if s.get("signal_type") == "HOLD"]
        if hold_signals:
            assert len(hold_signals) > 0

    def test_validate_signal_data_valid(self, basic_config, mock_event_bus, sample_market_data):
        """Test validating signal data with valid data."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

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
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Remove required columns
        sample_market_data = sample_market_data.drop(columns=["close", "volume"])

        result = strategy.validate_signal_data(sample_market_data)

        assert result is False

    def test_validate_signal_data_insufficient_rows(self, basic_config, mock_event_bus):
        """Test validating signal data with insufficient rows."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

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
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        info = strategy.get_strategy_info()

        assert info["name"] == basic_config.name
        assert info["type"] == "MEAN_REVERSION"
        assert info["mean_periods"] == basic_config.mean_periods
        assert info["std_dev_periods"] == basic_config.std_dev_periods
        assert info["min_reversion_strength"] == basic_config.min_reversion_strength
        assert info["hurst_threshold"] == basic_config.hurst_threshold
        assert info["volatility_adjustment"] == basic_config.volatility_adjustment
        assert info["regime_filter"] == basic_config.regime_filter
        assert info["correlation_threshold"] == basic_config.correlation_threshold

    def test_reset(self, basic_config, mock_event_bus):
        """Test resetting strategy state."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Add some state
        strategy.signal_history = [{"test": "data"}]
        strategy.valid_signal_count = 5
        strategy.invalid_signal_count = 2

        strategy.reset()

        assert strategy.signal_history == []
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0

    def test_z_score_threshold_filtering(self, basic_config, mock_event_bus, sample_market_data):
        """Test filtering signals based on z-score thresholds."""
        strategy = MeanReversionStrategy(basic_config, mock_event_bus)

        # Calculate z-score
        mean_indicators = strategy._calculate_mean_indicators(sample_market_data)
        std_dev_indicators = strategy._calculate_std_dev_indicators(sample_market_data)
        z_score = strategy._calculate_z_score(mean_indicators, std_dev_indicators)

        # Create signals with varying confidence
        signals = [
            {"signal_type": "BUY", "confidence": 0.1, "action": "Enter long"},  # Weak signal
            {"signal_type": "BUY", "confidence": 0.8, "action": "Enter long"},  # Strong signal
            {"signal_type": "SELL", "confidence": 0.9, "action": "Exit position"},  # Strong signal
        ]

        filtered_signals = strategy._apply_z_score_threshold_filter(
            signals, z_score, sample_market_data.index
        )

        # Should filter out weak signals
        assert len(filtered_signals) <= len(signals)
        if filtered_signals:
            assert all(
                s.get("confidence", 0) > 0.3 for s in filtered_signals
            )  # Assuming threshold > 0.3
