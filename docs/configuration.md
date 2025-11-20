# Configuration Ownership and Overrides

This document explains how runtime configuration is managed inside the backtester,
which component owns each section, and how to apply safe per-run overrides.

## Runtime Snapshots

- The module-level `backtester.core.config.get_config()` still returns the mutable global defaults.
- `BacktestRunConfig` wraps those defaults and produces immutable snapshots via `build()`.
- Each snapshot is deep-copied, so modifying one run does not bleed into others.
- `BacktestEngine` and the CLI (`main.py`) now always consume a snapshot to avoid mutating the global singleton.

### Applying Overrides

```python
from backtester.core.config import BacktestRunConfig, get_config

run_config = (
    BacktestRunConfig(get_config())
    .with_data_overrides(tickers=["MSFT"], start_date="2022-01-01")
    .with_strategy_overrides(strategy_name="custom_momentum", ma_short=10, ma_long=40)
    .build()
)
```

Available helpers: `with_data_overrides`, `with_strategy_overrides`, `with_portfolio_overrides`,
`with_execution_overrides`, `with_risk_overrides`, and `with_performance_overrides`. Each helper
accepts either keyword pairs or an existing component model and returns the builder for chaining.

The CLI entrypoint (`python main.py`) and the convenience script (`python scripts/run_backtest.py`)
use the same builder under the hood. Flags/environment variables (`--ticker`, `--start-date`,
`--date-preset`, `--freq`, `--strategy-name`, `--strategy-ma-short`, `BACKTEST_TICKERS`,
`BACKTEST_START_DATE`, etc.) are applied through `BacktestRunConfig` so every run receives a
validated snapshot. Set `BACKTEST_DRY_RUN=1` (or `--dry-run`) when you only want to materialise the
configuration without streaming data.

## Ownership

- **Data Retrieval**: `DataRetrievalConfig` describes symbols, date ranges, frequencies, API keys,
  and cache parameters. Only `backtester.data.data_retrieval.DataRetrieval` consumes this block.
- **Strategy**: `StrategyConfig` values are consumed by strategy orchestrators and should be treated as read-only outside the strategy module.
- **Portfolio**: `PortfolioConfig` feeds portfolio creation (`backtester.portfolio.*`) and should not be mutated by strategies or data loaders.
- **Execution & Risk**: Execution-specific knobs belong to the broker, while risk configuration is read exclusively by `RiskControlManager`.
- **Performance**: Only `PerformanceAnalyzer` should consume `PerformanceConfig`.

To preserve separation of concerns, components receive only the relevant slice of the snapshot and should never rewrite it.

### Read-only Component Views

Downstream components no longer receive the mutable `BacktesterConfig` object. Instead, helper
functions in `backtester.core.config` build frozen dataclasses:

- `build_data_config_view`, `build_portfolio_config_view`, and `build_risk_config_view` yield
  immutable snapshots that mirror the active run.
- `GeneralPortfolio`, `RiskControlManager`, and the strategy orchestrator only consume these
  views, so attempts to mutate configuration at runtime now raise `dataclasses.FrozenInstanceError`.
- Execution components receive the typed `SimulatedBrokerConfig` directly, allowing richer
  validation (throttling, latency controls, etc.) without the extra view layer.

This guarantees that once a backtest starts, every component sees the exact same inputs that were
logged via the configuration diff helper.

### Config Diff Logging

`BacktestRunConfig` snapshots are compared using `backtester.core.config_diff`. On engine
initialisation the diff between the global defaults and the CLI/programmatic overrides is logged,
and each call to `BacktestEngine.run_backtest` logs an INFO-level summary whenever per-run overrides
are applied. Combine this with the `run_id` logging context for reproducible audits.

## Validation

`validate_run_config()` performs early checks before any heavy component spins up:

- Ensures `data.tickers` is populated.
- Guards against invalid date ranges (`finish_date < start_date`).
- Blocks negative leverage inputs at the portfolio level.

The validator runs automatically when `BacktestRunConfig.build()` is invoked, but it is also exposed for ad-hoc use in tests or custom pipelines.

## Data Retrieval Overrides & Caching

- `DataRetrieval.get_data(config_override=...)` accepts a `DataRetrievalConfig` snapshot so
  callers can temporarily change tickers, dates, or frequencies without mutating the base handler.
- An in-memory cache is keyed by `(data_source, tickers, start, end, interval)` to avoid re-fetching
  identical frames during iterative workflows.
- TTL (`DataRetrievalConfig.cache_ttl_seconds`) and max-size (`DataRetrievalConfig.cache_max_entries`)
  controls feed into the shared cache (`backtester.utils.cache_utils.FrameCache`). Tune them per-run
  or use the defaults (5 minutes, 256 entries) for most workflows.
- The helper `clear_data_retrieval_cache()` exists for tests or scenarios where a hard refresh is required.
