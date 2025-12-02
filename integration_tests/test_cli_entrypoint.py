"""Integration tests for the CLI entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from backtester import __path__  # noqa: F401
from main import main as cli_main


def test_cli_entrypoint_runs_in_dry_mode(tmp_path: Path) -> None:
    """Running the CLI with a ticker should exit cleanly in dry mode."""
    env = os.environ.copy()
    env['BACKTEST_DRY_RUN'] = '1'
    result = subprocess.run(
        [sys.executable, "main.py", "--ticker", "AAPL"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0
    assert "validated configuration snapshot" in result.stdout


def test_cli_rejects_conflicting_date_inputs() -> None:
    """Mutually exclusive arguments should return a meaningful error."""
    exit_code = cli_main(
        ["--ticker", "MSFT", "--start-date", "2023-01-01", "--date-preset", "year"]
    )
    assert exit_code == 2
