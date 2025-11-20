# BacktestRunConfig Walkthrough

This tutorial shows how to create reproducible configuration snapshots using both CLI overrides and
pure Python helpers. The goal is to ensure every run is traceable and easy to parameterise.

## 1. CLI-Driven Overrides

Run the entrypoint with the symbols, dates, and strategy knobs you care about:

```bash
uv run python main.py \
    --ticker AAPL \
    --start-date 2023-01-01 \
    --finish-date 2023-06-30 \
    --freq 1d \
    --strategy-name custom_momo \
    --strategy-ma-short 10 \
    --strategy-ma-long 40 \
    --dry-run
```

Key environment variables mirror the CLI flags:

| Variable | Purpose |
| --- | --- |
| `BACKTEST_TICKERS` | Comma-separated tickers (`AAPL,MSFT`) |
| `BACKTEST_START_DATE` / `BACKTEST_FINISH_DATE` | ISO timestamps |
| `BACKTEST_DATE_PRESET` | Relative keywords (`year`, `month`, `ytd`, `max`) |
| `BACKTEST_FREQ` | Frequency override (`daily`, `1h`, etc.) |
| `BACKTEST_STRATEGY_NAME`, `BACKTEST_STRATEGY_MA_SHORT`, `BACKTEST_STRATEGY_MA_LONG` | Strategy knobs |
| `BACKTEST_DRY_RUN` | Force dry-run mode for CI or scripting |

Both `main.py` and `scripts/run_backtest.py` honour these inputs via `BacktestRunConfig`.

## 2. Programmatic Sweeps

When you need a sweep across several parameters, build snapshots explicitly:

```python
from backtester.core.config import BacktestRunConfig, get_config
from backtester.core.backtest_engine import BacktestEngine

base_builder = BacktestRunConfig(get_config())

for ticker in ["AAPL", "MSFT", "NVDA"]:
    for window in (5, 10, 20):
        run_config = (
            base_builder
            .with_data_overrides(tickers=[ticker])
            .with_strategy_overrides(ma_short=window, ma_long=window * 4)
            .build()
        )
        engine = BacktestEngine(config=run_config)
        engine.run_backtest()
```

Every iteration emits a configuration diff so you can audit exactly what changed.

## 3. Automation / CI Integration

- In CI pipelines, call `python scripts/run_backtest.py --ticker SPY --date-preset year`.
- Add `BACKTEST_DRY_RUN=1` for smoke-tests that only validate configuration wiring.
- Combine with `python scripts/check_docs.py` to ensure accompanying documentation stays healthy.

## 4. Inspecting Diffs

`backtester.core.config_diff.diff_configs` returns a structured list of changes:

```python
from backtester.core.config_diff import diff_configs, format_config_diff

base = BacktestRunConfig(get_config()).build()
updated = BacktestRunConfig(base).with_data_overrides(tickers=["QQQ"]).build()
print(format_config_diff(diff_configs(base, updated)))
```

Sample output:

```
- data.tickers: ['SPY'] -> ['QQQ']
```

Store this alongside your backtest artefacts to keep an immutable audit trail.
