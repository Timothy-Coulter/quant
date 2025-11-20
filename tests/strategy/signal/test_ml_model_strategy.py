"""Tests for ML model strategy."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from backtester.core.event_bus import EventBus
from backtester.strategy.signal.ml_model_strategy import MLModelStrategy
from backtester.strategy.signal.signal_strategy_config import (
    ExecutionParameters,
    IndicatorConfig,
    MLModelStrategyConfig,
    ModelConfig,
    RiskParameters,
)


class TestMLModelStrategy:
    """Test MLModelStrategy class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def basic_config(self):
        """Create a basic ML model strategy config."""
        return MLModelStrategyConfig(
            symbols=["AAPL", "GOOGL"],
            models=[
                ModelConfig(
                    model_name="random_forest",
                    model_type="classification",
                    framework="sklearn",
                    lookback_period=20,
                    train_test_split=0.8,
                    parameters={"n_estimators": 100, "max_depth": 10},
                )
            ],
            indicators=[
                IndicatorConfig(
                    indicator_name="sma", indicator_type="trend", period=20, parameters={}
                )
            ],
            signal_filters=[],
            risk_parameters=RiskParameters(),
            execution_params=ExecutionParameters(),
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
        strategy = MLModelStrategy(basic_config, mock_event_bus)

        assert strategy.config == basic_config
        assert strategy.name == basic_config.name
        assert strategy.model_ensemble_method == basic_config.model_ensemble_method
        assert strategy.retrain_frequency == basic_config.retrain_frequency
        assert strategy.feature_importance_threshold == basic_config.feature_importance_threshold

    def test_get_required_columns(self, basic_config, mock_event_bus):
        """Test getting required columns."""
        strategy = MLModelStrategy(basic_config, mock_event_bus)

        required_columns = strategy.get_required_columns()

        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "price_change",
            "high_low_spread",
            "volume_price_trend",
            "ma_5",
            "price_to_ma_5",
            "ma_10",
            "price_to_ma_10",
            "ma_20",
            "price_to_ma_20",
            "volatility_5",
            "volatility_20",
            "rsi",
        ]

        assert all(col in required_columns for col in expected_columns)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_generate_signals_with_model(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test generating signals with ML model."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.predict.return_value = np.array([1, 0, 1, 0, 1])  # BUY, HOLD, BUY, HOLD, BUY
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_model.get_model_info.return_value = {"name": "random_forest", "type": "classification"}
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        signals = strategy.generate_signals(sample_market_data, "AAPL")

        # Check that model was created and predict was called
        mock_factory.create.assert_called_once()
        mock_model.predict.assert_called()

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    def test_generate_signals_empty_data(self, basic_config, mock_event_bus):
        """Test generating signals with empty data."""
        strategy = MLModelStrategy(basic_config, mock_event_bus)

        empty_data = pd.DataFrame()

        signals = strategy.generate_signals(empty_data, "AAPL")

        assert signals == []

    def test_generate_signals_insufficient_data(self, basic_config, mock_event_bus):
        """Test generating signals with insufficient data."""
        strategy = MLModelStrategy(basic_config, mock_event_bus)

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

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_prepare_features(self, mock_factory, basic_config, mock_event_bus, sample_market_data):
        """Test preparing features for ML model."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        features = strategy._prepare_features(sample_market_data)

        # Check that features were prepared
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert "returns" in features.columns
        assert "price_change" in features.columns
        assert "high_low_spread" in features.columns
        assert "volume_price_trend" in features.columns

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_train_models(self, mock_factory, basic_config, mock_event_bus, sample_market_data):
        """Test training ML models."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.train.return_value = {"accuracy": 0.85, "precision": 0.80}
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features and target
        features = strategy._prepare_features(sample_market_data)
        target = pd.Series(np.random.choice([0, 1], len(features)))  # Binary classification target

        results = strategy._train_models(features, target)

        # Check that models were trained
        mock_model.train.assert_called_once()
        assert isinstance(results, dict)
        assert "random_forest" in results

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_predict_with_model(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test making predictions with ML model."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.predict.return_value = np.array([1, 0, 1, 0, 1])  # BUY, HOLD, BUY, HOLD, BUY
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features
        features = strategy._prepare_features(sample_market_data)

        predictions = strategy._predict_with_model(mock_model, features)

        # Check that predictions were made
        mock_model.predict.assert_called_once()
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(features)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_convert_predictions_to_signals(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test converting model predictions to trading signals."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.predict.return_value = np.array([1, 0, 1, 0, 1])  # BUY, HOLD, BUY, HOLD, BUY
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features and get predictions
        features = strategy._prepare_features(sample_market_data)
        predictions = strategy._predict_with_model(mock_model, features)

        signals = strategy._convert_predictions_to_signals(
            predictions, mock_model, sample_market_data
        )

        # Check that signals were generated
        assert isinstance(signals, list)
        if signals:  # If signals were generated
            assert all("signal_type" in signal for signal in signals)
            assert all("action" in signal for signal in signals)
            assert all("confidence" in signal for signal in signals)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_ensemble_predictions_voting(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test ensemble predictions with voting method."""
        # Create multiple models
        models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.name = f"model_{i}"
            mock_model.model_type = "classification"
            mock_model.framework = "sklearn"
            mock_model.predict.return_value = np.random.choice([0, 1], len(sample_market_data))
            mock_model.get_required_columns.return_value = ["feature1", "feature2"]
            models.append(mock_model)
            mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features
        features = strategy._prepare_features(sample_market_data)

        predictions = strategy._ensemble_predictions(models, features, "voting")

        # Check that ensemble predictions were made
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(features)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_ensemble_predictions_averaging(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test ensemble predictions with averaging method."""
        # Create multiple models
        models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.name = f"model_{i}"
            mock_model.model_type = "regression"
            mock_model.framework = "sklearn"
            mock_model.predict.return_value = np.random.uniform(0, 1, len(sample_market_data))
            mock_model.get_required_columns.return_value = ["feature1", "feature2"]
            models.append(mock_model)
            mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features
        features = strategy._prepare_features(sample_market_data)

        predictions = strategy._ensemble_predictions(models, features, "averaging")

        # Check that ensemble predictions were made
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(features)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_calculate_feature_importance(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test calculating feature importance."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.get_required_columns.return_value = [
            "feature1",
            "feature2",
            "feature3",
            "feature4",
        ]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features
        features = strategy._prepare_features(sample_market_data)

        importance = strategy._calculate_feature_importance(mock_model, features)

        # Check that feature importance was calculated
        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_validate_model_data_valid(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test validating model data with valid data."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Prepare features
        features = strategy._prepare_features(sample_market_data)

        result = strategy._validate_model_data(features)

        assert result is True

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_validate_model_data_insufficient_samples(
        self, mock_factory, basic_config, mock_event_bus
    ):
        """Test validating model data with insufficient samples."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Create insufficient data
        small_features = pd.DataFrame(
            {
                "feature1": [1, 2],
                "feature2": [3, 4],
            }
        )

        result = strategy._validate_model_data(small_features)

        assert result is False

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_validate_model_data_missing_features(self, mock_factory, basic_config, mock_event_bus):
        """Test validating model data with missing features."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2", "feature3"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Create data with missing features
        incomplete_features = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                # Missing feature3
            }
        )

        result = strategy._validate_model_data(incomplete_features)

        assert result is False

    def test_get_strategy_info(self, basic_config, mock_event_bus):
        """Test getting strategy information."""
        strategy = MLModelStrategy(basic_config, mock_event_bus)

        info = strategy.get_strategy_info()

        assert info["name"] == basic_config.name
        assert info["type"] == "ML_MODEL"
        assert info["model_ensemble_method"] == basic_config.model_ensemble_method
        assert info["retrain_frequency"] == basic_config.retrain_frequency
        assert info["feature_importance_threshold"] == basic_config.feature_importance_threshold

    def test_reset(self, basic_config, mock_event_bus):
        """Test resetting strategy state."""
        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Add some state
        strategy.signal_history = [{"test": "data"}]
        strategy.valid_signal_count = 5
        strategy.invalid_signal_count = 2

        strategy.reset()

        assert strategy.signal_history == []
        assert strategy.valid_signal_count == 0
        assert strategy.invalid_signal_count == 0

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_should_retrain_default(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test default retrain logic."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Should retrain by default
        result = strategy._should_retrain(mock_model, sample_market_data)

        assert result is True

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_should_retrain_based_on_performance(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test retrain logic based on model performance."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_model.performance_metrics = {"accuracy": 0.95}  # High performance
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Should not retrain if performance is high
        result = strategy._should_retrain(mock_model, sample_market_data)

        assert result is False

    @patch('backtester.strategy.signal.ml_model_strategy.ModelFactory')
    def test_should_retrain_based_on_data_drift(
        self, mock_factory, basic_config, mock_event_bus, sample_market_data
    ):
        """Test retrain logic based on data drift."""
        # Mock model factory
        mock_model = Mock()
        mock_model.name = "random_forest"
        mock_model.model_type = "classification"
        mock_model.framework = "sklearn"
        mock_model.get_required_columns.return_value = ["feature1", "feature2"]
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        mock_factory.create.return_value = mock_model

        strategy = MLModelStrategy(basic_config, mock_event_bus)

        # Should retrain if data drift is detected
        result = strategy._should_retrain(mock_model, sample_market_data)

        assert result is True
