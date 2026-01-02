# Core backtester configs

Complete `BacktesterConfig` snapshots that wire data, strategy, portfolio, execution, risk, and performance settings together. Use these as turn-key configs for CLI runs or smoke tests.

- `momentum_daily.yaml`: SPY/QQQ daily momentum stack with conservative risk and retail execution.
- `intraday_mean_reversion.yaml`: 1h mean reversion setup with volatility-aware risk and faster broker settings.

Values are fully expanded (no custom YAML tags) so `ConfigProcessor` can load them directly.
