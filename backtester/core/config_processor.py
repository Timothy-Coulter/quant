"""ConfigProcessor centralises loading, merging, and validating configs."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ValidationError

from backtester.core.config import (
    BacktesterConfig,
    ComprehensiveRiskConfig,
    DataRetrievalConfig,
    ExecutionConfig,
    PerformanceConfig,
    PortfolioConfig,
    StrategyConfig,
    validate_run_config,
)
from backtester.core.config_diff import ConfigDelta, diff_configs

type ConfigInput = BacktesterConfig | BaseModel | Mapping[str, Any] | str | bytes | Path | None


class ConfigProcessorError(RuntimeError):
    """Base exception for config processor failures."""


class ConfigSourceError(ConfigProcessorError):
    """Raised when a config source cannot be resolved."""


class ConfigValidationError(ConfigProcessorError):
    """Raised when a merged configuration fails validation."""

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        source: str | None = None,
        errors: Sequence[Any] | None = None,
    ) -> None:
        """Capture rich validation metadata for downstream error reporting."""
        context = []
        if component:
            context.append(f"component={component}")
        if source:
            context.append(f"source={source}")
        if errors:
            context.append(f"errors={errors}")
        suffix = f" ({', '.join(context)})" if context else ""
        super().__init__(f"{message}{suffix}")
        self.component = component
        self.source = source
        self.errors = errors


class ConfigProcessor:
    """Utility that normalises BacktesterConfig loading and merging."""

    _COMPONENT_MODELS: dict[str, type[BaseModel]] = {
        "data": DataRetrievalConfig,
        "strategy": StrategyConfig,
        "portfolio": PortfolioConfig,
        "execution": ExecutionConfig,
        "risk": ComprehensiveRiskConfig,
        "performance": PerformanceConfig,
    }

    def __init__(self, base: BacktesterConfig | None = None) -> None:
        """Initialise the processor with a snapshot of the provided base config."""
        self._base_config = (base or BacktesterConfig()).model_copy(deep=True)
        self._defaults = self._base_config.model_copy(deep=True)

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def apply(
        self,
        source: ConfigInput = None,
        *,
        component_overrides: Mapping[str, ConfigInput] | None = None,
        component: str | None = None,
        validate: bool = True,
    ) -> BacktesterConfig | BaseModel | None:
        """Merge the provided sources and return a BacktesterConfig snapshot."""
        payload = self._base_config.model_dump(mode="python")

        if source is not None:
            update_payload = self._coerce_config_payload(source)
            payload = self._deep_merge(payload, update_payload)

        if component_overrides:
            for name, override_source in component_overrides.items():
                normalized = self._normalize_component_name(name)
                override_payload = self._coerce_component_payload(normalized, override_source)
                if override_payload is None:
                    payload[normalized] = None
                else:
                    base_value = payload.get(normalized)
                    if isinstance(base_value, Mapping):
                        merged = self._deep_merge(dict(base_value), override_payload)
                    else:
                        merged = override_payload
                    payload[normalized] = merged

        resolved = self._build_config(payload)
        if validate:
            resolved = self.validate(resolved)

        if component:
            normalized_component = self._normalize_component_name(component)
            value = getattr(resolved, normalized_component)
            if isinstance(value, BaseModel):
                return value.model_copy(deep=True)
            return cast(BaseModel | None, value)
        return resolved

    def apply_component(
        self,
        component: str,
        overrides: ConfigInput = None,
        *,
        base: ConfigInput = None,
        validate: bool = True,
    ) -> BaseModel | None:
        """Return a single component by applying overrides to the base."""
        overrides_map = {component: overrides} if overrides is not None else None
        config = self.apply(
            source=base,
            component_overrides=overrides_map,
            validate=validate,
        )
        if not isinstance(config, BacktesterConfig):
            raise ConfigSourceError(f"Expected BacktesterConfig, got {type(config).__name__}")
        normalized = self._normalize_component_name(component)
        value = getattr(config, normalized)
        if isinstance(value, BaseModel):
            return value.model_copy(deep=True)
        return cast(BaseModel | None, value)

    def resolve_component(
        self,
        component: str,
        *,
        base_model: BaseModel | None = None,
        source: ConfigInput = None,
        overrides: ConfigInput = None,
    ) -> BaseModel | None:
        """Resolve a standalone component model without materialising BacktesterConfig."""
        normalized = self._normalize_component_name(component)
        model_cls = self._COMPONENT_MODELS.get(normalized)
        if model_cls is None:
            raise ConfigSourceError(f"Unknown component: {component}")

        payload: dict[str, Any] | None = None
        if base_model is not None:
            payload = base_model.model_dump(mode="python")
        else:
            defaults = getattr(self._defaults, normalized, None)
            if isinstance(defaults, BaseModel):
                payload = defaults.model_dump(mode="python")

        if source is not None:
            source_payload = self.load_component_payload(normalized, source)
            if source_payload is not None:
                payload = self._deep_merge(payload or {}, source_payload)

        if overrides is not None:
            override_payload = self.load_component_payload(normalized, overrides)
            if override_payload is not None:
                payload = self._deep_merge(payload or {}, override_payload)

        if payload is None:
            return None

        try:
            return model_cls(**payload)
        except ValidationError as exc:
            raise ConfigValidationError(
                f"Unable to resolve component '{normalized}'",
                component=normalized,
                errors=exc.errors(),
            ) from exc

    def load_yaml(self, path: str | Path) -> Mapping[str, Any]:
        """Read a YAML file and return its mapping payload."""
        resolved_path = Path(path).expanduser()
        if not resolved_path.is_file():
            raise ConfigSourceError(f"Config file not found: {resolved_path}")

        with resolved_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, MutableMapping):
            raise ConfigSourceError(f"YAML root must be a mapping: {resolved_path}")

        return dict(data)

    def merge_models(
        self,
        model: BaseModel,
        overrides: ConfigInput,
    ) -> BaseModel:
        """Return a new model instance with overrides applied."""
        base_payload = model.model_dump(mode="python")
        override_payload = self._coerce_mapping(
            overrides,
            context=f"{model.__class__.__name__} overrides",
        )
        merged = self._deep_merge(base_payload, override_payload)
        try:
            return model.__class__(**merged)
        except ValidationError as exc:
            raise ConfigValidationError(
                "Invalid overrides for model",
                component=model.__class__.__name__,
                errors=exc.errors(),
            ) from exc

    def validate(self, config: BacktesterConfig) -> BacktesterConfig:
        """Validate a BacktesterConfig snapshot."""
        try:
            return validate_run_config(config)
        except ValueError as exc:
            raise ConfigValidationError(str(exc), source="validation") from exc

    def load_component_payload(
        self,
        component: str,
        source: ConfigInput,
    ) -> dict[str, Any] | None:
        """Return a mutable mapping for a component source without instantiating a model."""
        normalized = self._normalize_component_name(component)
        return self._coerce_component_payload(normalized, source)

    def diff_with_defaults(self, config: BacktesterConfig | None = None) -> list[ConfigDelta]:
        """Compare a resolved config with the processor defaults."""
        target = config or self.apply(validate=False)
        if not isinstance(target, BacktesterConfig):
            raise ConfigSourceError(f"Expected BacktesterConfig, got {type(target).__name__}")
        return diff_configs(self._defaults, target)

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _build_config(self, payload: Mapping[str, Any]) -> BacktesterConfig:
        try:
            return BacktesterConfig(**payload)
        except ValidationError as exc:
            raise ConfigValidationError(
                "Unable to build BacktesterConfig",
                source="payload",
                errors=exc.errors(),
            ) from exc

    def _coerce_config_payload(self, source: ConfigInput) -> Mapping[str, Any]:
        if isinstance(source, BacktesterConfig):
            return source.model_dump(mode="python")
        if isinstance(source, BaseModel):
            if isinstance(source, BacktesterConfig):
                return source.model_dump(mode="python")
            raise ConfigSourceError(
                f"Expected BacktesterConfig, received {source.__class__.__name__}",
            )
        if isinstance(source, Mapping):
            return dict(source)
        path = self._path_from_source(source)
        if path:
            return dict(self.load_yaml(path))
        raise ConfigSourceError("Unsupported config source for BacktesterConfig")

    def _coerce_component_payload(
        self,
        component: str,
        source: ConfigInput,
    ) -> dict[str, Any] | None:
        if source is None:
            return None

        model_cls = self._COMPONENT_MODELS.get(component)
        if model_cls is None:
            raise ConfigSourceError(f"Unknown component: {component}")

        if isinstance(source, model_cls):
            return source.model_dump(mode="python")
        if isinstance(source, BaseModel):
            raise ConfigSourceError(
                f"Unexpected model for component '{component}': {source.__class__.__name__}",
            )
        if isinstance(source, Mapping):
            return dict(source)
        path = self._path_from_source(source)
        if path:
            data = self.load_yaml(path)
            return dict(data)
        raise ConfigSourceError(f"Unsupported source for component '{component}'")

    def _coerce_mapping(self, source: ConfigInput, *, context: str) -> Mapping[str, Any]:
        if isinstance(source, BaseModel):
            return source.model_dump(mode="python")
        if isinstance(source, Mapping):
            return dict(source)
        path = self._path_from_source(source)
        if path:
            return dict(self.load_yaml(path))
        raise ConfigSourceError(f"Unsupported mapping source for {context}")

    @staticmethod
    def _normalize_component_name(name: str) -> str:
        normalized = name.strip().lower()
        if normalized not in ConfigProcessor._COMPONENT_MODELS:
            raise ConfigSourceError(f"Unknown component: {name}")
        return normalized

    @staticmethod
    def _deep_merge(
        base: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(base)
        for key, value in update.items():
            if (
                isinstance(value, Mapping)
                and value is not None
                and isinstance(merged.get(key), Mapping)
            ):
                nested_base = cast(Mapping[str, Any], merged[key])
                merged[key] = ConfigProcessor._deep_merge(
                    nested_base,
                    cast(Mapping[str, Any], value),
                )
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _path_from_source(source: Any) -> Path | None:
        if isinstance(source, (str, bytes)):
            candidate = Path(os.fsdecode(source)).expanduser()
        elif isinstance(source, Path):
            candidate = source.expanduser()
        else:
            return None

        if candidate.is_file():
            return candidate
        if candidate.suffix.lower() in {".yml", ".yaml"}:
            return candidate
        return None
