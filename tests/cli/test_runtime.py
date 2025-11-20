"""Unit tests for CLI runtime helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path

from backtester.cli import build_run_config_from_cli, collect_overrides, parse_runtime_args
from backtester.core.config_processor import ConfigProcessor


def test_collect_overrides_produces_typed_models() -> None:
    """CLIOverrides should contain typed models when overrides are supplied."""
    args = parse_runtime_args(
        [
            "--ticker",
            "MSFT",
            "--start-date",
            "2022-01-01",
            "--strategy-name",
            "alt_strategy",
            "--strategy-ma-short",
            "8",
        ]
    )
    processor = ConfigProcessor()
    overrides = collect_overrides(args, env={}, processor=processor)

    assert overrides.primary_ticker == "MSFT"
    assert overrides.data is not None
    assert overrides.data.start_date == "2022-01-01"
    assert overrides.strategy is not None
    assert overrides.strategy.strategy_name == "alt_strategy"
    assert overrides.strategy.ma_short == 8

    raw_strategy = overrides.get_override("strategy")
    assert raw_strategy == {'strategy_name': 'alt_strategy', 'ma_short': 8}


def test_build_run_config_from_cli_applies_yaml_source(tmp_path: Path) -> None:
    """YAML config sources should be merged before CLI overrides."""
    config_payload = textwrap.dedent(
        """
        data:
          tickers: ["QQQ", "SPY"]
          freq: weekly
        strategy:
          strategy_name: yaml_strategy
        """
    ).strip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_payload, encoding="utf-8")

    args = parse_runtime_args(
        ["--ticker", "AAPL", "--start-date", "2021-05-10", "--config", str(config_path)]
    )
    processor = ConfigProcessor()
    overrides = collect_overrides(
        args,
        env={},
        processor=processor,
        config_source=config_path,
    )

    config = build_run_config_from_cli(
        overrides,
        processor=processor,
        config_source=config_path,
    )

    assert config.data is not None
    assert config.data.tickers == ["AAPL"]
    assert config.data.start_date == "2021-05-10"
    assert config.data.freq == "weekly"
    assert config.strategy is not None
    assert config.strategy.strategy_name == "yaml_strategy"
