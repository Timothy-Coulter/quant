# Execution presets

`SimulatedBrokerConfig` presets for different trading environments.

- `retail.yaml`: higher spreads/slippage and modest rate limits suited to retail flows.
- `institutional.yaml`: tighter spreads, lower latency, and higher throughput for institutional routing.

Use these in `BacktesterConfig.execution` to mirror the execution venue you want to simulate.
