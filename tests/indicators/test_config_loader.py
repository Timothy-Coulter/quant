"""Tests for indicator configuration resolver."""

from pathlib import Path

import pytest

from backtester.indicators.config_loader import IndicatorConfigResolver


class TestIndicatorConfigResolver:
    """Verify indicator YAML/preset resolution."""

    def test_resolve_preset_name(self) -> None:
        """Preset names should resolve to bundled YAML files."""
        resolver = IndicatorConfigResolver()
        config = resolver.resolve({'preset': 'rsi_momentum'})
        assert config.indicator_name == 'rsi'
        assert config.factory_name == 'rsi'
        assert config.indicator_type == 'momentum'

    def test_resolve_inline_definition(self) -> None:
        """Inline dictionaries should hydrate into IndicatorConfig objects."""
        resolver = IndicatorConfigResolver()
        config = resolver.resolve(
            {
                'indicator_name': 'ema',
                'indicator_type': 'trend',
                'period': 25,
                'ma_type': 'exponential',
            }
        )
        assert config.period == 25
        assert config.factory_name == 'ema'

    def test_resolve_custom_yaml_path(self, tmp_path: Path) -> None:
        """Custom search paths should allow ad-hoc YAML config loading."""
        resolver = IndicatorConfigResolver(search_paths=[tmp_path])
        yaml_path = tmp_path / 'custom_indicator.yaml'
        yaml_path.write_text(
            """__config_class__: IndicatorConfig
indicator_name: custom
indicator_type: trend
period: 10
""",
            encoding='utf-8',
        )
        config = resolver.resolve(
            {'config_path': 'custom_indicator.yaml', 'overrides': {'period': 15}}
        )
        assert config.indicator_name == 'custom'
        assert config.period == 15

    def test_invalid_source_raises(self) -> None:
        """Unsupported definition types should raise a TypeError."""
        resolver = IndicatorConfigResolver()
        with pytest.raises(TypeError):
            resolver.resolve(42)
