#!/usr/bin/env python3
"""Execute a backtest directly from the CLI with optional overrides."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from backtester.cli import build_run_config_from_cli, collect_overrides, parse_runtime_args
from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import get_config
from backtester.core.config_processor import ConfigProcessor, ConfigProcessorError


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point for the convenience backtest runner."""
    args = parse_runtime_args(argv)
    env = os.environ.copy()
    if env.get("BACKTEST_DRY_RUN", "").lower() in {"1", "true", "yes"}:
        args.dry_run = True

    config_source = args.config or env.get("BACKTEST_CONFIG_PATH")
    base_config = get_config()
    processor = ConfigProcessor(base=base_config)

    try:
        overrides = collect_overrides(
            args,
            env=env,
            processor=processor,
            config_source=config_source,
        )
        run_config = build_run_config_from_cli(
            overrides,
            processor=processor,
            config_source=config_source,
            base_config=base_config,
        )
    except (ValueError, ConfigProcessorError) as exc:
        print(f"[run_backtest] {exc}", file=sys.stderr)
        return 2

    engine = BacktestEngine(config=run_config)

    if args.dry_run:
        print("[run_backtest] Engine initialised (dry run).")
        return 0

    data_override_payload = overrides.get_override("data") or {}
    strategy_override_payload = overrides.get_override("strategy")
    portfolio_override_payload = overrides.get_override("portfolio")

    engine.run_backtest(
        ticker=overrides.primary_ticker,
        start_date=data_override_payload.get('start_date'),
        end_date=data_override_payload.get('finish_date'),
        interval=data_override_payload.get('freq'),
        strategy_params=dict(strategy_override_payload) if strategy_override_payload else None,
        portfolio_params=dict(portfolio_override_payload) if portfolio_override_payload else None,
    )
    print("[run_backtest] Backtest completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
