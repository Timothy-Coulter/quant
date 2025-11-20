"""Indicator system module for quant-bench backtesting framework.

This module provides comprehensive technical indicators for trading strategy development
and backtesting. It follows the established modular component architecture pattern
with proper typing and integration with existing backtester components.
"""

from backtester.signal.signal_types import SignalGenerator, SignalType

from .atr import ATRIndicator
from .base_indicator import BaseIndicator
from .bollinger_bands import BollingerBandsIndicator
from .cci import CCIIndicator
from .ema import EMAIndicator
from .factory import IndicatorFactory
from .indicator_configs import IndicatorConfig
from .macd import MACDIndicator
from .obv import OBVIndicator
from .rsi import RSIIndicator

# Import all indicator implementations
from .sma import SMAIndicator
from .stochastic import StochasticOscillator
from .williams_r import WilliamsROscillator

__all__ = [
    "BaseIndicator",
    "IndicatorConfig",
    "IndicatorFactory",
    "SignalType",
    "SignalGenerator",
    # Indicator classes
    "SMAIndicator",
    "EMAIndicator",
    "RSIIndicator",
    "MACDIndicator",
    "StochasticOscillator",
    "WilliamsROscillator",
    "BollingerBandsIndicator",
    "ATRIndicator",
    "CCIIndicator",
    "OBVIndicator",
]
