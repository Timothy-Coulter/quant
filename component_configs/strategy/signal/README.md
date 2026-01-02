# Signal strategy configs

`SignalStrategyConfig` presets wrapping specific strategy types.

- `momentum_default.yaml`: multi-timeframe momentum with EMA/RSI indicators.
- `mean_reversion_pairs.yaml`: pairs/BB-based mean reversion with correlation filter.
- `ml_directional.yaml`: ML ensemble with indicator features and confidence gating.

Each file embeds indicators, filters, risk/execution params, and the strategy-specific `strategy_config`.
