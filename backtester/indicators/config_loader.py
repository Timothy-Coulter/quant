"""Indicator configuration loading helpers.

This module centralises loading IndicatorConfig instances from dictionaries,
YAML files under ``component_configs/indicators``, or shorthand preset names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backtester.core.config_processor import ConfigProcessor, ConfigSourceError
from backtester.indicators.indicator_configs import IndicatorConfig


class IndicatorConfigResolver:
    """Resolve indicator configuration definitions into IndicatorConfig instances."""

    def __init__(
        self,
        *,
        processor: ConfigProcessor | None = None,
        search_paths: Sequence[Path] | None = None,
    ) -> None:
        """Initialise the resolver with optional custom processor and search paths."""
        self._processor = processor or ConfigProcessor()
        default_root = Path(__file__).resolve().parents[2] / "component_configs" / "indicators"
        self._search_paths = list(search_paths or (default_root,))

    def resolve(self, definition: Any) -> IndicatorConfig:
        """Resolve any supported definition into an IndicatorConfig."""
        if isinstance(definition, IndicatorConfig):
            return definition.model_copy(deep=True)

        payload: Mapping[str, Any]
        if isinstance(definition, Mapping):
            payload = self._resolve_mapping(definition)
        elif isinstance(definition, (str, Path)):
            payload = self._load_from_identifier(definition)
        else:
            raise TypeError(
                "Indicator definitions must be IndicatorConfig, mapping, or preset string",
            )

        return IndicatorConfig(**dict(payload))

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _resolve_mapping(self, definition: Mapping[str, Any]) -> Mapping[str, Any]:
        config_path = definition.get('config_path')
        preset = definition.get('preset') or definition.get('preset_name')
        overrides_section = definition.get('overrides') or {}

        inline_overrides = {
            key: value
            for key, value in definition.items()
            if key
            not in {
                'config_path',
                'preset',
                'preset_name',
                'overrides',
            }
        }

        payload: dict[str, Any]
        identifier = config_path or preset
        payload = dict(self._load_from_identifier(identifier)) if identifier else {}

        if inline_overrides:
            payload = ConfigProcessor._deep_merge(payload, inline_overrides)
        if isinstance(overrides_section, Mapping) and overrides_section:
            payload = ConfigProcessor._deep_merge(payload, dict(overrides_section))

        return payload

    def _load_from_identifier(self, identifier: str | Path) -> Mapping[str, Any]:
        path = self._resolve_path(identifier)
        data = dict(self._processor.load_yaml(path))
        meta = data.pop('__config_class__', None)
        if meta is not None and meta != 'IndicatorConfig':
            raise ConfigSourceError(
                f"Expected IndicatorConfig payload, received {meta} (file={path})",
            )
        return data

    def _resolve_path(self, identifier: str | Path) -> Path:
        raw = Path(identifier)
        candidates: list[Path] = []

        def _append(path: Path) -> None:
            if path not in candidates:
                candidates.append(path)

        if raw.suffix.lower() in {'.yml', '.yaml'}:
            _append(raw)
        else:
            _append(raw.with_suffix('.yaml'))
            _append(raw.with_suffix('.yml'))

        for candidate in list(candidates):
            if candidate.is_absolute():
                continue
            _append(Path.cwd() / candidate)
            for root in self._search_paths:
                _append(root / candidate)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise ConfigSourceError(f"Indicator config file not found: {identifier}")
