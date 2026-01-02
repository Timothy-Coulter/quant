# Integration Test Plan

Goal: add high-value integration coverage that exercises interactions across components and at least one end-to-end (E2E) backtest flow, without duplicating unit/smoke tests.

## Existing Coverage (to avoid duplication)
- Unit tests in `tests/**` already validate individual classes (e.g., `BacktestEngine`, `DataRetrieval`, strategy configs, performance metrics).
- Smoke tests in `smoke_tests/**` load all YAML configs from `component_configs/**`.
- No integration suites currently exist in `integration_tests/`.

## Proposed Suites

### 1) Config → Engine Wiring
- **Path:** `integration_tests/test_config_processor_integration.py`
- **Scope:** Load `component_configs/core/*.yaml` via `ConfigProcessor`, instantiate `BacktestEngine`, and verify components (data handler, strategy, portfolio, broker, risk manager) are wired with the expected config values.
- **Fixtures:** Stub `findatapy`/network calls; reuse cached sample market data frames.

### 2) Data + Indicators + Signal Pipeline
- **Path:** `integration_tests/test_data_indicator_signal_pipeline.py`
- **Scope:** Run `DataRetrieval` → indicator factory → `SignalStrategy` end-to-end on a synthetic OHLCV DataFrame; assert signals carry indicator-derived metadata and respect symbol universe.
- **Checks:** Indicator columns present; `SignalStrategy` emits `SignalEvent` with confidence thresholds; multiple symbols routed via `EventBus`.

### 3) Strategy ↔ Portfolio ↔ Broker Loop
- **Path:** `integration_tests/test_strategy_portfolio_broker_flow.py`
- **Scope:** Simulate a trading loop: strategy emits signals → orders routed through `SimulatedBroker` → portfolio updates → performance snapshot.
- **Checks:** Order fills respect slippage/commission from `SimulatedBrokerConfig`; portfolio positions/cash update; trade log captured; performance analyzer ingests portfolio values.
- **Fixtures:** Use `component_configs/execution/retail.yaml` and `component_configs/portfolio/general.yaml`; stub fills for determinism.

### 4) Risk Management Integration
- **Path:** `integration_tests/test_risk_management_integration.py`
- **Scope:** Combine `ComprehensiveRiskConfig` with active strategy and broker; verify stop-loss/take-profit/position limits gate orders and emit `RiskAlertEvent`.
- **Checks:** Breach scenarios trigger halts or sizing adjustments; risk monitor thresholds produce alerts; limits reflected in portfolio updates.
- **Configs:** Use `component_configs/risk_management/balanced.yaml`.

### 5) Multi-Strategy Orchestration
- **Path:** `integration_tests/test_orchestration_multi_strategy.py`
- **Scope:** Instantiate an `OrchestrationConfig` (e.g., `ensemble_blend.yaml`) with two `SignalStrategy` instances and ensure coordination rules (threshold, fallback) and conflict resolution (weighted merge) behave as configured.
- **Checks:** Strategy registration order, gating/priority respected, blended outputs recorded.

### 6) Model-Backed Strategy Integration
- **Path:** `integration_tests/test_model_strategy_integration.py`
- **Scope:** Plug `ModelConfig` (e.g., `model/pytorch_mlp.yaml`) into `SignalStrategy` and confirm model predictions propagate to signals and risk filters; use lightweight stub model to avoid heavy deps.
- **Checks:** Model invoked with expected feature shape; ensemble weights applied; signals honor confidence thresholds.

### 7) CLI + Backtester End-to-End
- **Path:** `integration_tests/test_cli_end_to_end.py`
- **Scope:** Invoke `python main.py` (or `scripts/run_backtest.py`) via `subprocess` using `component_configs/cli/*` presets, ensuring the run completes and outputs performance metrics JSON/printouts.
- **Checks:** Non-zero exit, run log includes `total_return`, output artifacts created (e.g., run report), and config overrides respected.
- **Fixtures:** Use minimal date ranges and small symbol sets to keep runtime low; mock network/data retrieval.

## Cross-Cutting Fixtures & Harness
- Central `conftest.py` under `integration_tests/` to:
  - Provide deterministic OHLCV frames and stubbed `findatapy`/network calls.
  - Seed `BacktesterConfig` from `component_configs/core/*.yaml` with in-memory data injection.
  - Patch model/broker latency to keep tests fast.
- Use markers (e.g., `@pytest.mark.integration`) and skip/xfail for GPU-dependent paths if any.

## Success Criteria
- New integration tests cover cross-component flows and E2E CLI execution.
- No duplication of existing unit/smoke checks; focus on interaction, orchestration, and full-loop behavior.
- Suites run within CI time budget (use synthetic data and stubs).
