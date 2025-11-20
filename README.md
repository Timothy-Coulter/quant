# quant-bench

A quantitative backtesting framework for financial analysis.

## Features

- Market data handling and validation
- Portfolio management and backtesting
- Strategy development and optimization
- Performance analysis and reporting

## Installation

```bash
uv sync
```

## Usage

```python
from backtester.main import run_backtest

# Run a basic backtest
results = run_backtest()
```

### Command-Line Overrides

The entrypoint now accepts runtime overrides so you can iterate without touching Python code:

```bash
uv run python main.py --ticker AAPL --start-date 2023-01-01 --freq 1h --dry-run
```

Key flags:

- `--ticker`/`--tickers` – specify one or more symbols (repeatable)
- `--start-date`, `--finish-date`, `--date-preset` – explicit or relative date windows
- `--freq` – override the data frequency (e.g. `daily`, `1h`)
- `--strategy-name`, `--strategy-ma-short`, `--strategy-ma-long` – momentum strategy knobs
- `--dry-run` – validate the configuration without invoking the backtest loop

Environment variables mirror the flags (`BACKTEST_TICKERS`, `BACKTEST_START_DATE`,
`BACKTEST_DATE_PRESET`, `BACKTEST_FREQ`, `BACKTEST_STRATEGY_NAME`, etc.) and the helper
`BACKTEST_DRY_RUN=1` forces dry-run mode for automation. Power users can also call
`python scripts/run_backtest.py ...` to execute the same workflow inside scripted
pipelines or CI.

## Architecture Overview

- **Event-driven core:** The `EventBus` fan-outs market, signal, order, and risk
  events so each subsystem stays decoupled. See `docs/engine_workflow.md` for a
  diagram of the canonical loop.
- **Configuration layering:** Global defaults live in `backtester.core.config`,
  but every run clones those defaults via `BacktestRunConfig` so overrides never
  leak between experiments. Details live in `docs/configuration.md`.
- **Extension points:** Strategies, portfolio allocators, brokers, and risk
  modules are all thin interfaces. Register new implementations through the
  orchestrator or swap entire modules by instantiating the concrete classes and
  wiring them into `BacktestEngine`.

## Examples

### Run a quick backtest with an isolated configuration snapshot

```python
from backtester.core.backtest_engine import BacktestEngine
from backtester.core.config import BacktestRunConfig, get_config

run_config = (
    BacktestRunConfig(get_config())
    .with_data_overrides(tickers=["MSFT"], start_date="2023-01-01", finish_date="2023-06-30")
    .with_strategy_overrides(ma_short=10, ma_long=40)
    .build()
)

engine = BacktestEngine(config=run_config)
results = engine.run_backtest()
print(results["performance"]["total_return"])
```

### Register an additional strategy with the orchestrator

```python
from backtester.strategy.signal.base_signal_strategy import BaseSignalStrategy
from backtester.strategy.signal.signal_strategy_config import SignalStrategyConfig

class AlwaysOnStrategy(BaseSignalStrategy):
    def generate_signals(self, data, symbol):  # noqa: D401 - demo only
        return [{"signal_type": "BUY", "confidence": 0.5, "symbol": symbol}]

strategy = AlwaysOnStrategy(
    SignalStrategyConfig(strategy_name="always_on", symbols=["SPY"]), engine.event_bus
)
engine.strategy_orchestrator.register_strategy(
    identifier="always_on",
    strategy=strategy,
    kind=strategy.type,
    priority=1,
)
```

### Configure comprehensive risk controls

```python
from backtester.risk_management import ComprehensiveRiskConfig, RiskControlManager

risk_config = ComprehensiveRiskConfig(
    max_position_size=0.15,
    stop_loss_pct=0.02,
    take_profit_pct=0.08,
)
risk_manager = RiskControlManager(config=risk_config, logger=engine.logger, event_bus=engine.event_bus)
engine.current_risk_manager = risk_manager
```

For a walkthrough on building multiple `BacktestRunConfig` snapshots (CLI and code-driven),
see `docs/run_config_tutorial.md`.

## Event Payload Contract

All components communicate through the event bus using a shared metadata contract. The keys
below are guaranteed to exist (or be explicitly empty) so subscribers can build filters without
having to reverse‑engineer publisher internals.

- **MarketDataEvent**
  - `metadata.symbol` and `metadata.symbols` (universe slice for the payload)
  - `metadata.bar` containing `open`, `high`, `low`, `close`, `volume`, `timestamp`
  - `metadata.provenance.source` and `metadata.provenance.ingested_at`
  - `metadata.data_frame` when the upstream producer supplied a pandas window
- **SignalEvent**
  - `metadata.symbol`, `metadata.symbols`, and `metadata.signal_type`
  - `metadata.source_strategy` and any raw indicator payload under `metadata.raw_signal`
  - Priority defaults to HIGH so downstream order routers can rely on delivery order
- **OrderEvent**
  - `metadata.symbol`, `metadata.side`, `metadata.order_type`
  - Execution context (`fill_quantity`, `fill_price`, `commission`) plus `metadata.message`
    when an order is rejected or cancelled
- **PortfolioUpdateEvent**
  - `metadata.portfolio_id`, `metadata.positions`, and `metadata.position_updates`
  - Mirrors the monetary snapshot via `metadata.total_value`, `metadata.cash_balance`,
    and `metadata.positions_value`
- **RiskAlertEvent**
  - `metadata.component`, `metadata.portfolio_id` (when applicable), and `metadata.violations`
  - `metadata.recommendations` summarising the suggested remediation steps

When subscribing, prefer the metadata keys (`symbol`/`symbols`) rather than bespoke attributes,
as they are populated uniformly regardless of which module produced the event.

## Engine Workflow

The canonical simulation loop—from market data ingestion through strategy signals,
risk evaluation, order routing, broker fills, and portfolio/performance updates—is
documented in detail (with a Mermaid diagram) in `docs/engine_workflow.md`.
Reference it when implementing new components or investigating event ordering.

## Observability & Diagnostics

- Every component should obtain its logger via `get_backtester_logger(__name__)`.
  The helper automatically attaches a `run_id` (per engine instance) and an
  optional `symbol`, so log aggregators can filter by context with no extra work.
- `bind_logger_context(logger, symbol="MSFT")` can be used when the same logger
  needs to emit records for different instruments over time.
- `PerformanceAnalyzer` now records `operational_metrics` (latency, queue depth,
  throughput) which are surfaced both in `engine.backtest_results['performance']`
  and the text report. See `docs/observability.md` for the full reference.

## Development Commands

### Code Formatting and Linting

**Windows one-liners:**
```cmd
rem Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

rem Lint code (ruff check)
uv run ruff check .

rem Lint and fix (ruff check --fix)
uv run ruff check --fix .

rem Type checking (mypy)
uv run mypy .

rem combined command
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check --fix . && uv run mypy .

```

**Linux/macOS one-liners:**
```bash
# Format code (ruff format, black, isort)
uv run ruff format . && uv run black . && uv run isort .

# Lint code (ruff check)
uv run ruff check .

# Lint and fix (ruff check --fix)
uv run ruff check --fix .

# Type checking (mypy)
uv run mypy .

# Validate docs/ links
uv run python scripts/check_docs.py
```

### Testing

```bash
# Run tests (uses pyproject addopts: -n auto --reruns 3)
uv run pytest

# Run tests with coverage
uv run pytest --cov=backtester --cov-report=term-missing --cov-report=html
```

### Complete Development Workflow

**Windows:**
```cmd
rem Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```

**Linux/macOS:**
```bash
# Format, lint, and typecheck
uv run ruff format . && uv run black . && uv run isort . && uv run ruff check . && uv run mypy .
```

### CI Helper Script

Prefer a single command? Run `python scripts/run_ci_checks.py` to execute the
formatters, `ruff check --fix`, `mypy`, and `pytest` with consistent ordering
and friendly status output. Pass `--skip-tests` when you only need the fast
format/lint cycle.
