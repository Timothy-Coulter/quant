"""PyTorch adapter for ML model integration.

This module provides an adapter for integrating PyTorch models with the backtester model system.
"""

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from backtester.model.base_model import ModelFrameworkAdapter
from backtester.model.model_configs import PyTorchModelConfig


class PyTorchAdapter(ModelFrameworkAdapter[nn.Module]):
    """Adapter for PyTorch models.

    This adapter provides a consistent interface for PyTorch models while
    handling framework-specific implementation details.
    """

    def __init__(self, config: PyTorchModelConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the PyTorch adapter.

        Args:
            config: Configuration for the PyTorch model
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._model: nn.Module | None = None

    def initialize_model(self, config: PyTorchModelConfig) -> nn.Module:
        """Initialize a PyTorch model based on configuration.

        Args:
            config: PyTorch model configuration

        Returns:
            Initialized PyTorch model instance
        """
        # Create model based on architecture configuration
        if config.model_architecture:
            model = self._build_model_from_config(config.model_architecture, config.input_shape)
        else:
            # Default simple model
            model = self._build_default_model(config.input_shape)

        # Move model to specified device
        device = torch.device(config.device)
        model = model.to(device)

        self._model = model
        self.logger.info(f"Initialized PyTorch model on device: {config.device}")
        return model

    def _build_model_from_config(
        self, architecture: dict[str, Any], input_shape: tuple[int, ...] | None
    ) -> nn.Module:
        """Build model from architecture configuration.

        Args:
            architecture: Model architecture definition
            input_shape: Input shape for the model

        Returns:
            PyTorch model instance
        """

        class ConfigurableModel(nn.Module):
            def __init__(self, config: dict[str, Any], input_dim: int) -> None:
                super().__init__()
                self.layers = nn.ModuleList()

                # Add layers based on configuration
                for layer_config in config.get('layers', []):
                    layer_type = layer_config.get('type', 'Linear')

                    if layer_type == 'Linear':
                        self.layers.append(
                            nn.Linear(
                                (
                                    input_dim
                                    if len(self.layers) == 0
                                    else int(layer_config.get('input_dim', 64))
                                ),
                                layer_config.get('output_dim', 64),
                            )
                        )
                    elif layer_type == 'ReLU':
                        self.layers.append(nn.ReLU())
                    elif layer_type == 'Dropout':
                        self.layers.append(nn.Dropout(layer_config.get('p', 0.2)))
                    elif layer_type == 'BatchNorm1d':
                        self.layers.append(nn.BatchNorm1d(layer_config.get('num_features', 64)))

                    # Update input_dim for next layer
                    input_dim = int(layer_config.get('output_dim', 64))

                # Add output layer
                output_dim = config.get('output_dim', 1)
                self.layers.append(nn.Linear(input_dim, output_dim))

            def forward(self, x: Any) -> Any:
                for layer in self.layers:
                    x = layer(x)
                return x

        # Calculate input dimension
        input_dim = int(np.prod(input_shape)) if input_shape else 1

        model = ConfigurableModel(architecture, input_dim)
        return model

    def _build_default_model(self, input_shape: tuple[int, ...] | None) -> nn.Module:
        """Build a default model.

        Args:
            input_shape: Input shape for the model

        Returns:
            Default PyTorch model instance
        """

        class DefaultModel(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                )

            def forward(self, x: Any) -> Any:
                return self.layers(x)

        # Calculate input dimension
        input_dim = int(np.prod(input_shape)) if input_shape else 1
        model = DefaultModel(input_dim)
        return model

    def train_model(self, model: nn.Module, features: pd.DataFrame, target: pd.Series) -> nn.Module:
        """Train the PyTorch model.

        Args:
            model: PyTorch model instance
            features: Training features
            target: Training targets

        Returns:
            Trained model instance
        """
        # Convert to tensors
        x_train = torch.tensor(features.values, dtype=torch.float32)
        y_train = torch.tensor(target.values, dtype=torch.float32).reshape(-1, 1)

        # Get device
        device = torch.device(self.config.device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # Get training configuration
        training_config = self.config.training_config
        epochs = training_config.get('epochs', 50)
        lr = training_config.get('learning_rate', 0.001)
        batch_size = training_config.get('batch_size', 32)

        # Setup training
        import torch.nn as nn

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for _epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(x_train), batch_size):
                batch_x = x_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.logger.info(f"Trained PyTorch model for {epochs} epochs")
        return model

    def predict(self, model: nn.Module, features: pd.DataFrame) -> NDArray[Any]:
        """Generate predictions using PyTorch model.

        Args:
            model: PyTorch model instance
            features: Features for prediction

        Returns:
            Model predictions
        """
        # Convert to tensor
        x_test = torch.tensor(features.values, dtype=torch.float32)

        # Get device
        device = torch.device(self.config.device)
        x_test = x_test.to(device)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions = model(x_test)

        # Convert back to numpy
        predictions_array = cast(NDArray[Any], predictions.cpu().numpy())

        # Flatten if needed
        if predictions_array.ndim > 1 and predictions_array.shape[1] == 1:
            predictions_array = predictions_array.flatten()

        return predictions_array

    def save_model(self, model: nn.Module, filepath: str) -> None:
        """Save PyTorch model to file.

        Args:
            model: PyTorch model instance
            filepath: Path to save the model
        """
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'config': self.config.model_dump(),
                'model_architecture': self.config.model_architecture,
            },
            filepath,
        )

        self.logger.info(f"Saved PyTorch model to: {filepath}")

    def load_model(self, filepath: str) -> nn.Module:
        """Load PyTorch model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Reconstructed PyTorch model instance
        """
        checkpoint = self._load_checkpoint(filepath)
        model = self._reconstruct_model_from_checkpoint(checkpoint)
        self.logger.info(f"Loaded PyTorch model from: {filepath}")
        return model

    def load_and_reconstruct_model(self, filepath: str) -> nn.Module:
        """Load and reconstruct PyTorch model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Reconstructed PyTorch model instance
        """
        checkpoint = self._load_checkpoint(filepath)
        return self._reconstruct_model_from_checkpoint(checkpoint)

    def _load_checkpoint(self, filepath: str) -> dict[str, Any]:
        """Load the raw checkpoint dictionary from disk."""
        checkpoint = torch.load(filepath, map_location=torch.device(self.config.device))
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        return checkpoint

    def _reconstruct_model_from_checkpoint(self, checkpoint: dict[str, Any]) -> nn.Module:
        """Rebuild a model instance from a checkpoint."""
        architecture = checkpoint.get('model_architecture')
        input_shape = self.config.input_shape
        if input_shape is None:
            checkpoint_config = checkpoint.get('config')
            if isinstance(checkpoint_config, dict):
                shape_from_checkpoint = checkpoint_config.get('input_shape')
                if isinstance(shape_from_checkpoint, (list, tuple)):
                    input_shape = tuple(int(dim) for dim in shape_from_checkpoint) or None

        if architecture:
            model = self._build_model_from_config(architecture, input_shape)
        else:
            model = self._build_default_model(input_shape)

        state_dict = checkpoint.get('model_state_dict')
        if state_dict is None:
            raise ValueError("Checkpoint is missing 'model_state_dict'")
        model.load_state_dict(state_dict)
        model.to(torch.device(self.config.device))
        return model

    def evaluate_model(
        self, model: nn.Module, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            model: PyTorch model instance
            features: Test features
            target: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to tensors
        x_test = torch.tensor(features.values, dtype=torch.float32)
        y_test = torch.tensor(target.values, dtype=torch.float32).reshape(-1, 1)

        # Get device
        device = torch.device(self.config.device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            import torch.nn as nn

            predictions = model(x_test)
            loss = nn.MSELoss()(predictions, y_test)

        # Convert to numpy for additional metrics
        pred_np = predictions.cpu().numpy().flatten()
        target_np = target.values

        # Calculate additional metrics
        mae = np.mean(np.abs(target_np - pred_np))

        # R-squared
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {'mse': float(loss.item()), 'mae': float(mae), 'r2': float(r2)}

    def get_model_parameters(self, model: nn.Module) -> dict[str, NDArray[Any]]:
        """Get model parameters as numpy arrays.

        Args:
            model: PyTorch model instance

        Returns:
            Dictionary of parameter names and values
        """
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.data.cpu().numpy()

        return params

    @classmethod
    def create_model(cls, model_name: str, config: PyTorchModelConfig) -> 'PyTorchAdapter':
        """Create a PyTorch model adapter with the specified model.

        Args:
            model_name: Name of the model to create
            config: Configuration for the model

        Returns:
            PyTorchAdapter instance
        """
        adapter = cls(config)
        adapter.initialize_model(config)
        return adapter
