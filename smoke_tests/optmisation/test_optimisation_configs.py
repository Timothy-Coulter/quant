"""Smoke tests for optimisation YAMLs and parameter spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backtester.optmisation.parameter_space import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
)
from smoke_tests.conftest import COMPONENT_CONFIGS, load_yaml, strip_config_class


def _load_optimisation_payloads() -> list[tuple[Path, dict[str, Any]]]:
    opt_dir = COMPONENT_CONFIGS / "optmisation"
    return [(path, strip_config_class(load_yaml(path))) for path in sorted(opt_dir.glob("*.yaml"))]


def _build_parameter_def(param: dict[str, Any]) -> ParameterDefinition:
    name = str(param["name"])
    param_type = str(param["param_type"])
    return ParameterDefinition(
        name=name,
        param_type=param_type,
        low=param.get("low"),
        high=param.get("high"),
        step=param.get("step"),
        log=bool(param.get("log", False)),
        choices=param.get("choices"),
        q=param.get("q"),
    )


def test_optimisation_configs_load() -> None:
    """Ensure optimisation payloads initialise configs and parameter definitions."""
    for path, payload in _load_optimisation_payloads():
        config = OptimizationConfig()
        for key, value in payload.items():
            if key == "parameter_space":
                continue
            setattr(config, key, value)

        param_space = ParameterSpace()
        for param in payload.get("parameter_space", []):
            param_def = _build_parameter_def(param)
            param_space.add_parameter(param_def)

        assert config.n_trials >= 0, f"Invalid trials in {path}"
        assert isinstance(param_space, ParameterSpace), f"Failed to build ParameterSpace for {path}"
