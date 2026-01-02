# Optimisation presets

`OptimizationConfig` + parameter space YAMLs for Optuna-style studies (note repository spelling `optmisation`).

- `grid_search.yaml`: bounded grid with deterministic combinations for CI-safe smoke.
- `bayesian_tpe.yaml`: lightweight TPE search with time-boxed trials.

Parameter spaces are described with `name`, `param_type`, bounds/choices, and optional `step`/`log`/`q`.
