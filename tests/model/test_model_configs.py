"""Unit tests for the model configuration classes."""

import pytest
from pydantic import ValidationError

from backtester.model.model_configs import (
    ModelConfig,
    PyTorchModelConfig,
    SciPyModelConfig,
    SklearnModelConfig,
)


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_valid_config(self) -> None:
        """Test valid model configuration."""
        config = ModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            lookback_period=30,
            prediction_horizon=1,
        )

        assert config.model_name == "test_model"
        assert config.model_type == "regression"
        assert config.framework == "sklearn"
        assert config.lookback_period == 30
        assert config.prediction_horizon == 1
        assert config.signal_threshold == 0.5
        assert config.confidence_threshold == 0.6

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
        )

        assert config.lookback_period == 30
        assert config.prediction_horizon == 1
        assert config.train_test_split == 0.8
        assert config.validation_split == 0.2
        assert config.signal_threshold == 0.5
        assert config.confidence_threshold == 0.6
        assert config.enable_monitoring is True
        assert config.retrain_threshold == 0.7
        assert config.target_column == "close"
        assert config.normalize_features is True
        assert config.cv_folds == 5
        assert config.random_state is None

    def test_invalid_model_type(self) -> None:
        """Test validation of model_type field."""
        with pytest.raises(ValidationError, match="model_type must be one of"):
            ModelConfig(
                model_name="test_model",
                model_type="invalid_type",
                framework="sklearn",
            )

    def test_invalid_framework(self) -> None:
        """Test validation of framework field."""
        with pytest.raises(ValidationError, match="framework must be one of"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="invalid_framework",
            )

    def test_positive_integer_validation(self) -> None:
        """Test validation of positive integer fields."""
        # Test negative lookback_period
        with pytest.raises(ValidationError, match="Period must be positive"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                lookback_period=-10,
            )

        # Test zero prediction_horizon
        with pytest.raises(ValidationError, match="Period must be positive"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                prediction_horizon=0,
            )

    def test_ratio_validation(self) -> None:
        """Test validation of ratio fields."""
        # Test train_test_split > 1
        with pytest.raises(ValidationError, match="Split ratio"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                train_test_split=1.5,
            )

        # Test signal_threshold < 0
        with pytest.raises(ValidationError, match="Threshold"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                signal_threshold=-0.1,
            )

        # Test confidence_threshold > 1
        with pytest.raises(ValidationError, match="Threshold"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                confidence_threshold=1.5,
            )

    def test_cv_folds_validation(self) -> None:
        """Test validation of cv_folds field."""
        # Test zero cv_folds
        with pytest.raises(ValidationError, match="Period must be positive"):
            ModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                cv_folds=0,
            )

    def test_feature_columns_optional(self) -> None:
        """Test that feature_columns is optional."""
        config = ModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            feature_columns=None,
        )
        assert config.feature_columns is None

    def test_serialization(self) -> None:
        """Test model serialization."""
        config = ModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            lookback_period=30,
        )

        # Test model_dump
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert dumped['model_name'] == "test_model"

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)


class TestSklearnModelConfig:
    """Test cases for SklearnModelConfig class."""

    def test_valid_sklearn_config(self) -> None:
        """Test valid sklearn model configuration."""
        config = SklearnModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            model_class="LinearRegression",
            hyperparameters={'fit_intercept': True},
        )

        assert config.framework == "sklearn"
        assert config.model_class == "LinearRegression"
        assert config.hyperparameters == {'fit_intercept': True}

    def test_default_sklearn_config(self) -> None:
        """Test default sklearn model configuration."""
        config = SklearnModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            model_class="LinearRegression",
        )

        assert config.hyperparameters == {}
        assert config.preprocessing_steps == []

    def test_invalid_sklearn_model_class(self) -> None:
        """Test validation of sklearn model class."""
        with pytest.raises(ValidationError, match="Unknown sklearn model class"):
            SklearnModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                model_class="InvalidModel",
            )

    def test_valid_sklearn_model_classes(self) -> None:
        """Test that valid sklearn model classes are accepted."""
        valid_classes = [
            'LinearRegression',
            'LogisticRegression',
            'RandomForestRegressor',
            'RandomForestClassifier',
            'SVR',
            'SVC',
            'GradientBoostingRegressor',
            'GradientBoostingClassifier',
            'Lasso',
            'Ridge',
            'ElasticNet',
            'KMeans',
        ]

        for model_class in valid_classes:
            config = SklearnModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                model_class=model_class,
            )
            assert config.model_class == model_class

    def test_hyperparameters_dict(self) -> None:
        """Test hyperparameters dictionary handling."""
        hyperparameters = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
        }

        config = SklearnModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            model_class="RandomForestRegressor",
            hyperparameters=hyperparameters,
        )

        assert config.hyperparameters == hyperparameters


class TestSciPyModelConfig:
    """Test cases for SciPyModelConfig class."""

    def test_valid_scipy_config(self) -> None:
        """Test valid scipy model configuration."""
        config = SciPyModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="scipy",
            scipy_function="stats.linregress",
            function_params={'alternative': 'two-sided'},
            optimization_method="BFGS",
        )

        assert config.framework == "scipy"
        assert config.scipy_function == "stats.linregress"
        assert config.function_params == {'alternative': 'two-sided'}
        assert config.optimization_method == "BFGS"

    def test_invalid_scipy_function(self) -> None:
        """Test validation of scipy function."""
        with pytest.raises(ValidationError, match="Unknown scipy function"):
            SciPyModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="scipy",
                scipy_function="invalid.function",
            )

    def test_valid_scipy_functions(self) -> None:
        """Test that valid scipy functions are accepted."""
        valid_functions = [
            'stats.linregress',
            'stats.pearsonr',
            'spearmanr',
            'kendalltau',
            'optimize.minimize',
            'signal.find_peaks',
            'interpolate.interp1d',
        ]

        for func in valid_functions:
            config = SciPyModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="scipy",
                scipy_function=func,
            )
            assert config.scipy_function == func

    def test_optional_fields(self) -> None:
        """Test optional fields."""
        config = SciPyModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="scipy",
            scipy_function="stats.linregress",
        )

        assert config.function_params == {}
        assert config.optimization_method is None

    def test_function_params_dict(self) -> None:
        """Test function parameters dictionary."""
        params = {
            'alpha': 0.05,
            'method': 'pearson',
            'alternative': 'greater',
        }

        config = SciPyModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="scipy",
            scipy_function="stats.pearsonr",
            function_params=params,
        )

        assert config.function_params == params


class TestPyTorchModelConfig:
    """Test cases for PyTorchModelConfig class."""

    def test_valid_pytorch_config(self) -> None:
        """Test valid pytorch model configuration."""
        config = PyTorchModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="pytorch",
            model_architecture={'layers': []},
            training_config={'epochs': 50},
            device="cuda:0",
            input_shape=(10, 1),
        )

        assert config.framework == "pytorch"
        assert config.model_architecture == {'layers': []}
        assert config.training_config == {'epochs': 50}
        assert config.device == "cuda:0"
        assert config.input_shape == (10, 1)

    def test_invalid_device(self) -> None:
        """Test validation of device."""
        with pytest.raises(ValidationError, match="Unsupported device"):
            PyTorchModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="pytorch",
                device="invalid_device",
            )

    def test_valid_devices(self) -> None:
        """Test that valid devices are accepted."""
        valid_devices = ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps']

        for device in valid_devices:
            config = PyTorchModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="pytorch",
                device=device,
            )
            assert config.device == device

    def test_invalid_input_shape(self) -> None:
        """Test validation of input_shape."""
        # Test with single dimension
        with pytest.raises(ValidationError, match="Input shape must have at least 2 dimensions"):
            PyTorchModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="pytorch",
                input_shape=(10,),
            )

        # Test with negative dimension
        with pytest.raises(ValidationError, match="All input shape dimensions must be positive"):
            PyTorchModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="pytorch",
                input_shape=(10, -1),
            )

    def test_default_pytorch_config(self) -> None:
        """Test default pytorch configuration."""
        config = PyTorchModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="pytorch",
        )

        assert config.device == "cpu"
        assert config.training_config == {}
        assert config.input_shape is None
        assert config.model_architecture == {}

    def test_model_architecture_dict(self) -> None:
        """Test model architecture dictionary handling."""
        architecture = {
            'layers': [
                {'type': 'Linear', 'input_dim': 10, 'output_dim': 64},
                {'type': 'ReLU'},
                {'type': 'Linear', 'output_dim': 1},
            ]
        }

        config = PyTorchModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="pytorch",
            model_architecture=architecture,
        )

        assert config.model_architecture == architecture


class TestConfigInheritance:
    """Test cases for configuration class inheritance."""

    def test_sklearn_inherits_base_config(self) -> None:
        """Test that SklearnModelConfig inherits from ModelConfig."""
        config = SklearnModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            model_class="LinearRegression",
            lookback_period=50,
            signal_threshold=0.7,
        )

        assert isinstance(config, ModelConfig)
        assert config.lookback_period == 50
        assert config.signal_threshold == 0.7

    def test_framework_override(self) -> None:
        """Test that framework field is overridden in subclasses."""
        sklearn_config = SklearnModelConfig(
            model_name="test_model",
            model_type="regression",
            framework="sklearn",
            model_class="LinearRegression",
        )

        assert sklearn_config.framework == "sklearn"

    def test_config_validation_chain(self) -> None:
        """Test that validation works through the inheritance chain."""
        # Test base config validation
        with pytest.raises(ValidationError):
            ModelConfig(
                model_name="test_model",
                model_type="invalid_type",
                framework="sklearn",
            )

        # Test subclass validation with invalid base field
        with pytest.raises(ValidationError):
            SklearnModelConfig(
                model_name="test_model",
                model_type="invalid_type",  # Invalid base field
                framework="sklearn",
                model_class="LinearRegression",
            )

        # Test subclass validation with invalid subclass field
        with pytest.raises(ValidationError):
            SklearnModelConfig(
                model_name="test_model",
                model_type="regression",
                framework="sklearn",
                model_class="InvalidModel",  # Invalid subclass field
            )
