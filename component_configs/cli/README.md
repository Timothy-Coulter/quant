# CLI presets

Preset argument/env bundles for running the backtester without hand-typing flags.

- `momentum_daily.yaml`: daily equity preset for SPY/QQQ with overrideable MA windows.
- `intraday_mean_reversion.yaml`: 1h mean-reversion setup with tighter stop/TP defaults.

Each file carries `args` (CLI flags), optional `env`, and an optional `config_path` that points to a core YAML snapshot for quick runs.
