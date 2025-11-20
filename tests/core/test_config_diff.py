"""Tests for configuration diff helpers."""

from backtester.core.config import BacktesterConfig
from backtester.core.config_diff import diff_configs, format_config_diff


def test_diff_configs_detects_nested_updates() -> None:
    """Changing nested fields should produce a delta entry."""
    base = BacktesterConfig()
    updated = base.model_copy(deep=True)
    assert updated.data is not None
    updated.data.tickers = ["MSFT"]
    assert updated.strategy is not None
    updated.strategy.ma_short = 42

    deltas = diff_configs(base, updated)
    paths = {delta.path for delta in deltas}
    assert "data.tickers" in paths
    assert "strategy.ma_short" in paths


def test_format_config_diff_renders_human_readable_lines() -> None:
    """Formatted diff should include both path and value information."""
    base = BacktesterConfig()
    updated = base.model_copy(deep=True)
    assert updated.data is not None
    updated.data.freq = "1h"

    formatted = format_config_diff(diff_configs(base, updated))
    assert "data.freq" in formatted
    assert "daily" in formatted
    assert "1h" in formatted
