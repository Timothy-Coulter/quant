"""Integration test ensuring model configs attach to signal strategies."""

from __future__ import annotations

from pathlib import Path

import yaml

from backtester.strategy.signal.signal_strategy_config import SignalStrategyConfig


def test_model_configs_load_into_signal_strategy() -> None:
    """Load ML strategy YAML and ensure model definitions are preserved."""
    yaml_path = Path("component_configs/strategy/signal/ml_directional.yaml")
    payload = yaml.safe_load(yaml_path.read_text())
    assert isinstance(payload, dict)

    config = SignalStrategyConfig(**payload)
    assert config.strategy_type.value == "ml_model"
    assert config.models, "Expected model definitions"
    model_names = {model.model_name for model in config.models}
    assert "rf_classifier" in model_names
    assert "lstm_forecaster" in model_names
