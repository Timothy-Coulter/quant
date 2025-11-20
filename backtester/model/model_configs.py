"""Model configuration system for the backtester.

This module provides standardized configuration classes for all machine learning models,
following the established pydantic patterns from the core configuration system.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for model parameters using pydantic BaseModel.

    This class defines all the parameters needed to configure any machine learning model,
    with validation and defaults that ensure proper operation across all model types.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # Core model settings
    model_name: str = Field(description="Name of the model")
    model_type: str = Field(description="Type/category of model")
    framework: str = Field(description="ML framework: sklearn, tensorflow, scipy, pytorch")

    # Data parameters
    lookback_period: int = Field(default=30, description="Historical periods for features")
    prediction_horizon: int = Field(default=1, description="Future periods to predict")

    # Training parameters
    train_test_split: float = Field(default=0.8, description="Train/test split ratio")
    validation_split: float = Field(default=0.2, description="Validation split for training")

    # Signal generation
    signal_threshold: float = Field(default=0.5, description="Threshold for signal generation")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence for signals")

    # Performance monitoring
    enable_monitoring: bool = Field(default=True, description="Enable model performance tracking")
    retrain_threshold: float = Field(
        default=0.7, description="Retrain when accuracy falls below this"
    )

    # Feature engineering
    feature_columns: list[str] | None = Field(
        default=None, description="Specific columns to use as features"
    )
    target_column: str = Field(default="close", description="Target column for prediction")
    normalize_features: bool = Field(default=True, description="Whether to normalize features")

    # Cross-validation
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    random_state: int | None = Field(default=None, description="Random state for reproducibility")

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type is one of the allowed values."""
        valid_types = ['regression', 'classification', 'forecasting', 'clustering']
        if v not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        return v

    @field_validator('framework')
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate framework is one of the supported values."""
        valid_frameworks = ['sklearn', 'tensorflow', 'scipy', 'pytorch']
        if v not in valid_frameworks:
            raise ValueError(f"framework must be one of {valid_frameworks}")
        return v

    @field_validator('lookback_period', 'prediction_horizon', 'cv_folds')
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Validate that integer parameters are positive."""
        if v <= 0:
            raise ValueError("Period must be positive")
        return v

    @field_validator(
        'train_test_split',
        'validation_split',
        'signal_threshold',
        'confidence_threshold',
        'retrain_threshold',
    )
    @classmethod
    def validate_ratios_and_thresholds(cls, v: float, info: Any) -> float:
        """Validate ratio and threshold parameters are in valid ranges."""
        field_name = info.field_name
        if field_name in ['train_test_split', 'validation_split']:
            if not 0.0 <= v <= 1.0:
                raise ValueError("Split ratio must be between 0.0 and 1.0")
        else:
            if not 0.0 <= v <= 1.0:
                raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


class SklearnModelConfig(ModelConfig):
    """Configuration for scikit-learn models.

    This class extends ModelConfig with parameters specific to scikit-learn models.
    """

    framework: str = "sklearn"
    model_class: str = Field(description="Sklearn model class name")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    preprocessing_steps: list[str] = Field(
        default_factory=list, description="Preprocessing pipeline steps"
    )

    @field_validator('model_class')
    @classmethod
    def validate_sklearn_model(cls, v: str) -> str:
        """Validate that the model class is a known sklearn model."""
        sklearn_models = [
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
        if v not in sklearn_models:
            raise ValueError(f"Unknown sklearn model class: {v}. Valid options: {sklearn_models}")
        return v


class SciPyModelConfig(ModelConfig):
    """Configuration for SciPy models.

    This class extends ModelConfig with parameters specific to SciPy statistical models.
    """

    framework: str = "scipy"
    scipy_function: str = Field(description="SciPy function to use")
    function_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the SciPy function"
    )
    optimization_method: str | None = Field(default=None, description="Optimization method to use")

    @field_validator('scipy_function')
    @classmethod
    def validate_scipy_function(cls, v: str) -> str:
        """Validate that the function is a known scipy function."""
        scipy_functions = [
            'stats.linregress',
            'stats.pearsonr',
            'spearmanr',
            'kendalltau',
            'optimize.minimize',
            'signal.find_peaks',
            'interpolate.interp1d',
        ]
        if v not in scipy_functions:
            raise ValueError(f"Unknown scipy function: {v}. Valid options: {scipy_functions}")
        return v


class PyTorchModelConfig(ModelConfig):
    """Configuration for PyTorch models.

    This class extends ModelConfig with parameters specific to PyTorch models.
    """

    framework: str = "pytorch"
    model_architecture: dict[str, Any] = Field(
        default_factory=dict, description="PyTorch model architecture definition"
    )
    training_config: dict[str, Any] = Field(
        default_factory=dict, description="PyTorch training parameters"
    )
    device: str = Field(default="cpu", description="Device to use for training (cpu, cuda, etc.)")
    input_shape: tuple[int, ...] | None = Field(
        default=None, description="Input shape for the model"
    )

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate that the device is supported."""
        valid_devices = ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps']
        if v not in valid_devices:
            raise ValueError(f"Unsupported device: {v}. Valid options: {valid_devices}")
        return v

    @field_validator('input_shape')
    @classmethod
    def validate_input_shape(cls, v: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """Validate input shape is valid for neural networks."""
        if v is not None:
            if len(v) < 2:
                raise ValueError("Input shape must have at least 2 dimensions")
            if any(dim <= 0 for dim in v):
                raise ValueError("All input shape dimensions must be positive")
        return v
