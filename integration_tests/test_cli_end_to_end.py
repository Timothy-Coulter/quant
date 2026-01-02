"""Integration test for CLI bootstrap path."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

import main


def test_cli_dry_run_initialises_engine(
    patch_data_retrieval: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure main() completes with a validated config in dry-run mode."""
    args: Sequence[str] = [
        "--config",
        "component_configs/core/momentum_daily.yaml",
        "--dry-run",
        "--ticker",
        "SPY",
    ]
    exit_code = main.main(args)
    assert exit_code == 0
