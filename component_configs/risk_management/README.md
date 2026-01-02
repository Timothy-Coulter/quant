# Risk management presets

`ComprehensiveRiskConfig` bundles with nested stop-loss, take-profit, position sizing, and monitoring settings.

- `conservative.yaml`: tight stops, low max drawdown, small position sizes.
- `balanced.yaml`: moderate stops and limits for general-purpose testing.
- `aggressive.yaml`: higher risk budget with dynamic hedging enabled.

Each file expands nested configs (`stop_loss_config`, `take_profit_config`, `position_sizing_config`, `risk_limits_config`, `risk_monitoring_config`) to avoid surprises at runtime.
