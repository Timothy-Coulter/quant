# Smoke test plan

Objective: ensure every YAML config under `component_configs/` can be loaded via ConfigProcessor and exercised in a minimal workflow. The directory layout mirrors `backtester/` so each component has targeted smoke tests.

```
smoke_tests/
├── cli/
├── core/
├── data/
├── execution/
├── indicators/
├── model/
├── optmisation/
├── portfolio/
├── risk_management/
├── signal/
├── strategy/
└── utils/
```

## Test patterns by module
- **data**: load each `DataRetrievalConfig` YAML, instantiate `DataRetrieval`, and stub `Market.fetch_market` to confirm cache keys + API key resolution work.
- **execution**: instantiate `SimulatedBroker` with YAML configs, submit mock orders, and assert commission/slippage defaults match the config.
- **portfolio**: create portfolios from YAML, feed synthetic price ticks, ensure positions respect limits from config.
- **risk_management**: instantiate `RiskControlManager` with each comprehensive config, simulate portfolio events, assert alerts/triggers fire per thresholds.
- **strategy/signal**: load strategy configs, feed fixture data frames, and confirm signals are produced with schema validated by `SignalSchema`.
- **strategy/orchestration**: load orchestrator YAML (sequential/ensemble) and ensure strategies can be registered + cycle executed with dummy market data.
- **core**: full BacktestEngine initialization from `component_configs/core/*.yaml`, run a single-day backtest to ensure no component requests additional kwargs.
- **optmisation**: run a single-trial Optuna study using YAML-defined parameter spaces to verify ConfigProcessor integration.
- **cli**: simulate CLI argument parsing + YAML path flags, assert resulting config matches expectation using config diffs.
- **utils**: ensure support configs (cache/db/formatting) can be loaded and applied to helper classes.

## Automation
1. Provide a pytest marker `@pytest.mark.smoke_config` and run these tests on a lightweight dataset (or stubbed data) to keep runtime low.
2. Each module-level smoke test should iterate over YAML files in its matching `component_configs/<module>` directory.
3. Failures must include the path to the problematic YAML plus ConfigProcessor validation errors for easy debugging.
