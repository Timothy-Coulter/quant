# Orchestration configs

`OrchestrationConfig` files that describe how multiple strategies coordinate.

- `sequential_primary.yaml`: simple sequential runner with priority ordering.
- `ensemble_blend.yaml`: ensemble orchestrator with weighted merge conflict resolution.

Strategies are referenced by identifier/kind/role with coordination rules for gating and fallbacks.
