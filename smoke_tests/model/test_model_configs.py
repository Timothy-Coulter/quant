"""Smoke tests for model configuration YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.model.model_configs import (
    ModelConfig,
    PyTorchModelConfig,
    SciPyModelConfig,
    SklearnModelConfig,
)
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class

MODEL_CLASS_MAP: dict[str, type[ModelConfig]] = {
    "SklearnModelConfig": SklearnModelConfig,
    "PyTorchModelConfig": PyTorchModelConfig,
    "SciPyModelConfig": SciPyModelConfig,
    "ModelConfig": ModelConfig,
}


def _resolve_model_class(payload: dict[str, Any]) -> type[ModelConfig]:
    class_name = payload.get("__config_class__")
    if isinstance(class_name, str) and class_name in MODEL_CLASS_MAP:
        return MODEL_CLASS_MAP[class_name]
    return ModelConfig


def _load_model_payloads() -> list[tuple[Path, dict[str, Any], type[ModelConfig]]]:
    model_dir = COMPONENT_CONFIGS / "model"
    payloads: list[tuple[Path, dict[str, Any], type[ModelConfig]]] = []
    for path in sorted(model_dir.glob("*.yaml")):
        raw = load_yaml(path)
        model_cls = _resolve_model_class(raw)
        payloads.append((path, strip_config_class(raw), model_cls))
    return payloads


def test_model_configs_load() -> None:
    """Ensure model config payloads instantiate the expected model class."""
    for path, payload, model_cls in _load_model_payloads():
        model = model_cls(**payload)
        assert isinstance(model, model_cls), f"Failed for {path}"
