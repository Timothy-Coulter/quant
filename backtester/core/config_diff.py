"""Utilities for comparing configuration snapshots."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from backtester.core.config import BacktesterConfig

type ConfigPayload = BacktesterConfig | BaseModel | Mapping[str, Any]

ConfigDelta = namedtuple("ConfigDelta", ["path", "before", "after"])


def diff_configs(base: ConfigPayload, updated: ConfigPayload) -> list[ConfigDelta]:
    """Return a flat list of differences between two configuration snapshots."""
    baseline = _as_mapping(base)
    candidate = _as_mapping(updated)
    deltas: list[ConfigDelta] = []
    _collect_diffs(path="", left=baseline, right=candidate, deltas=deltas)
    return deltas


def format_config_diff(deltas: Sequence[ConfigDelta]) -> str:
    """Render a configuration diff in a log-friendly format."""
    return "\n".join(f"- {delta.path}: {delta.before!r} -> {delta.after!r}" for delta in deltas)


def _collect_diffs(
    *,
    path: str,
    left: Any,
    right: Any,
    deltas: list[ConfigDelta],
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        all_keys = set(left.keys()) | set(right.keys())
        for key in sorted(all_keys):
            next_path = f"{path}.{key}" if path else key
            if key not in left:
                deltas.append(ConfigDelta(path=next_path, before=None, after=right[key]))
                continue
            if key not in right:
                deltas.append(ConfigDelta(path=next_path, before=left[key], after=None))
                continue
            _collect_diffs(path=next_path, left=left[key], right=right[key], deltas=deltas)
        return

    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            deltas.append(ConfigDelta(path=path or "[root]", before=left, after=right))
        return

    if left != right:
        deltas.append(ConfigDelta(path=path or "[root]", before=left, after=right))


def _as_mapping(config: ConfigPayload) -> Mapping[str, Any]:
    if isinstance(config, BacktesterConfig):
        return config.model_dump(mode="python")
    if isinstance(config, BaseModel):
        return config.model_dump(mode="python")
    if isinstance(config, Mapping):
        return config
    raise TypeError(f"Unsupported config payload type: {type(config)!r}")
