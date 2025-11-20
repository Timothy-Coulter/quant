"""Tests for ConfigProcessor."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backtester.core.config import BacktesterConfig, StrategyConfig
from backtester.core.config_processor import (
    ConfigProcessor,
    ConfigSourceError,
    ConfigValidationError,
)


def test_apply_returns_base_config_copy() -> None:
    """Calling apply() should return a new BacktesterConfig copy."""
    processor = ConfigProcessor()
    config = processor.apply()
    assert isinstance(config, BacktesterConfig)
    assert config is not processor._base_config  # pylint: disable=protected-access


def test_apply_merges_mapping_override() -> None:
    """Mapping overrides should be merged into the resulting config."""
    processor = ConfigProcessor()
    config = processor.apply(source={"data": {"tickers": ["TSLA"]}})
    assert isinstance(config, BacktesterConfig)
    assert config.data is not None
    assert config.data.tickers == ["TSLA"]


def test_component_override_with_model_instance() -> None:
    """Model instance overrides should be applied when provided."""
    processor = ConfigProcessor()
    override = StrategyConfig(ma_short=99)
    config = processor.apply(component_overrides={"strategy": override})
    assert isinstance(config, BacktesterConfig)
    assert config.strategy is not None
    assert config.strategy.ma_short == 99


def test_apply_component_returns_component_instance() -> None:
    """apply_component() should return typed component instances."""
    processor = ConfigProcessor()
    strategy = processor.apply_component("strategy", {"ma_short": 7})
    assert isinstance(strategy, StrategyConfig)
    assert strategy.ma_short == 7


def test_component_override_from_yaml(tmp_path: Path) -> None:
    """YAML payloads should be accepted as component overrides."""
    processor = ConfigProcessor()
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(yaml.safe_dump({"ma_short": 3}), encoding="utf-8")

    config = processor.apply(component_overrides={"strategy": yaml_path})
    assert isinstance(config, BacktesterConfig)
    assert config.strategy is not None
    assert config.strategy.ma_short == 3


def test_validation_failure_raises_config_validation_error() -> None:
    """Invalid overrides should raise ConfigValidationError."""
    processor = ConfigProcessor()
    with pytest.raises(ConfigValidationError):
        processor.apply(component_overrides={"data": {"tickers": []}})


def test_unknown_component_raises_source_error() -> None:
    """Unknown components should raise ConfigSourceError."""
    processor = ConfigProcessor()
    with pytest.raises(ConfigSourceError):
        processor.apply(component_overrides={"unknown": {"foo": "bar"}})


def test_diff_with_defaults_highlights_changes() -> None:
    """diff_with_defaults should include mutated component paths."""
    processor = ConfigProcessor()
    config = processor.apply(component_overrides={"strategy": {"ma_short": 11}})
    assert isinstance(config, BacktesterConfig)
    deltas = processor.diff_with_defaults(config)
    paths = {delta.path for delta in deltas}
    assert "strategy.ma_short" in paths
