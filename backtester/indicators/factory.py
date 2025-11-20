"""Factory module for the indicator system.

This module provides the factory pattern for creating indicator instances,
following the established modular component architecture.
"""

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig

# Re-export the factory for easy access
__all__ = ["IndicatorFactory", "BaseIndicator", "IndicatorConfig"]
