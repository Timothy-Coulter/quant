# Component config coverage plan

Goal: provide ready-to-run YAML configs for every major component so smoke tests can instantiate the entire stack without ad hoc kwargs. Directory layout mirrors `backtester/`.

```
component_configs/
├── cli/                # CLI presets (argument bundles, env overrides)
├── core/               # full BacktesterConfig snapshots
├── data/               # DataRetrievalConfig variants (yahoo_daily, intraday, crypto)
├── execution/          # SimulatedBrokerConfig presets (retail, institutional)
├── indicators/         # IndicatorConfig templates (ema_fast, macd_default, bollinger_wide)
├── model/              # ModelConfig definitions per framework
├── optmisation/        # OptimizationConfig + ParameterSpace YAMLs
├── portfolio/          # PortfolioConfig + specialized dual-pool configs
├── risk_management/    # ComprehensiveRiskConfig bundles + per-component overrides
├── signal/             # Signal schema templates / canned signals
├── strategy/
│   ├── signal/         # Momentum, mean reversion, ML strategy configs
│   ├── portfolio/      # Kelly, risk parity, equal weight configs
│   └── orchestration/  # OrchestrationConfig files for sequential/ensemble flows
└── utils/              # Cache/db/formatting/global settings
```

## Authoring guidelines
1. **One YAML per scenario**: each file defines a single config instance (e.g., `portfolio/general.yaml` -> `PortfolioConfig`).
2. **Type metadata**: include a `__config_class__` field so ConfigProcessor knows which Pydantic model to instantiate when loading from disk.
3. **Cross references**: core/backtester-level configs reference component YAMLs via relative paths to avoid duplication. Example snippet:
   ```yaml
   __config_class__: BacktesterConfig
   data: !include ../data/yahoo_daily.yaml
   strategy: !include ../strategy/signal/momentum_default.yaml
   portfolio: !include ../portfolio/general.yaml
   ```
4. **Validation**: run ConfigProcessor on every file inside `component_configs/` via a dedicated smoke test to ensure schema drift is detected in CI.
5. **Documentation**: add README files in subdirectories that describe each preset (when to use it, expected markets, invariants).
