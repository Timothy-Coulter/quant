# Model configs

Pydantic ML configs for strategy/model adapters.

- `sklearn_random_forest.yaml`: sklearn classification baseline for directional signals.
- `pytorch_mlp.yaml`: simple PyTorch MLP forecaster with GPU-friendly defaults.
- `scipy_stats.yaml`: SciPy stats utility setup for correlation/edge tests.

These map to `SklearnModelConfig`, `PyTorchModelConfig`, and `SciPyModelConfig` respectively.
