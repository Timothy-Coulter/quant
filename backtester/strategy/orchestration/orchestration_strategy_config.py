"""Configuration models for the strategy orchestration layer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StrategyKind(str, Enum):
    """Supported strategy categories for orchestration."""

    SIGNAL = "signal"
    PORTFOLIO = "portfolio"


class StrategyRole(str, Enum):
    """High-level role a strategy can play inside an orchestrator."""

    PRIMARY = "primary"
    MASTER = "master"
    SLAVE = "slave"
    AUXILIARY = "auxiliary"


class OrchestratorType(str, Enum):
    """Available orchestrator implementations."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MASTER_SLAVE = "master_slave"
    ENSEMBLE = "ensemble"
    CONDITIONAL = "conditional"


class CoordinationRuleType(str, Enum):
    """Types of coordination rules that can be applied."""

    SEQUENCE = "sequence"
    TRIGGER = "trigger"
    FALLBACK = "fallback"
    THRESHOLD = "threshold"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving competing signals."""

    FIRST_SIGNAL = "first_signal"
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_MERGE = "weighted_merge"
    CONSENSUS = "consensus"
    LATEST = "latest"


class PerformanceAggregationMethod(str, Enum):
    """Approach used to aggregate performance metrics across strategies."""

    AVERAGE = "average"
    WEIGHTED = "weighted"
    SUM = "sum"
    COMPOSITE = "composite"


class StrategyReference(BaseModel):
    """Reference metadata for strategies managed by an orchestrator."""

    identifier: str = Field(..., description="Unique identifier for the strategy instance.")
    kind: StrategyKind = Field(default=StrategyKind.SIGNAL)
    role: StrategyRole = Field(default=StrategyRole.PRIMARY)
    weight: float = Field(default=1.0, ge=0.0)
    priority: int = Field(default=0)
    depends_on: list[str] = Field(default_factory=list)
    enabled: bool = Field(default=True)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("identifier")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Strategy identifier cannot be empty.")
        return value

    @field_validator("depends_on")
    @classmethod
    def _validate_dependencies(cls, values: list[str], info: Any) -> list[str]:
        identifier: str = info.data.get("identifier", "")
        if identifier and identifier in values:
            raise ValueError("A strategy cannot depend on itself.")
        return values


class CoordinationRule(BaseModel):
    """Defines how strategies interact or gate each other."""

    rule_type: CoordinationRuleType = Field(default=CoordinationRuleType.SEQUENCE)
    primary: str = Field(..., description="Primary strategy identifier.")
    secondary: list[str] = Field(default_factory=list, description="Affected strategy identifiers.")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("secondary")
    @classmethod
    def _ensure_non_empty(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("Coordination rules must target at least one secondary strategy.")
        return values


class OrchestrationConfig(BaseModel):
    """Top-level configuration for a strategy orchestrator instance."""

    orchestrator_type: OrchestratorType = Field(default=OrchestratorType.SEQUENTIAL)
    strategies: list[StrategyReference] = Field(default_factory=list)
    coordination_rules: list[CoordinationRule] = Field(default_factory=list)
    conflict_resolution: ConflictResolutionStrategy = Field(
        default=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    )
    performance_aggregation: PerformanceAggregationMethod = Field(
        default=PerformanceAggregationMethod.WEIGHTED
    )
    evaluation_window: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_strategy(self, identifier: str) -> StrategyReference | None:
        """Return the reference for the given strategy identifier."""
        for reference in self.strategies:
            if reference.identifier == identifier:
                return reference
        return None

    def is_registered(self, identifier: str) -> bool:
        """Check whether a strategy identifier exists in the configuration."""
        return self.get_strategy(identifier) is not None

    @classmethod
    def default_config(cls) -> OrchestrationConfig:
        """Return the default orchestration configuration."""
        return cls(
            orchestrator_type=OrchestratorType.SEQUENTIAL,
            strategies=[
                StrategyReference(
                    identifier="primary_strategy",
                    kind=StrategyKind.SIGNAL,
                    priority=0,
                )
            ],
            conflict_resolution=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            performance_aggregation=PerformanceAggregationMethod.WEIGHTED,
        )
