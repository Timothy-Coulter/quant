"""Commodity Channel Index (CCI) indicator implementation.

This module provides the Commodity Channel Index indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import numpy as np
import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("cci")
class CCIIndicator(BaseIndicator):
    """Commodity Channel Index (CCI) momentum indicator.

    The Commodity Channel Index is a momentum-based oscillator used to help
    determine when an investment vehicle has been overbought and oversold.
    CCI measures the deviation of the price from the statistical mean of the price.

    CCI = (Typical Price - Simple Moving Average of Typical Price) / (0.015 × Mean Deviation)

    Where:
    - Typical Price (TP) = (High + Low + Close) / 3
    - Mean Deviation = Mean of absolute differences from SMA
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the baseline configuration for the CCI indicator."""
        return IndicatorConfig(
            indicator_name="cci",
            factory_name="cci",
            indicator_type="momentum",
            cci_period=20,
            cci_constant=0.015,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index (CCI) values.

        The CCI calculation involves:
        1. Calculate Typical Price: (High + Low + Close) / 3
        2. Calculate Simple Moving Average of Typical Price
        3. Calculate Mean Deviation
        4. Calculate CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with CCI values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=self.config.cci_period).mean()

        # Calculate Mean Deviation
        mean_deviation = typical_price.rolling(window=self.config.cci_period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )

        # Calculate CCI
        cci_column = f"{self.name.lower()}_cci"
        cci = (typical_price - sma_tp) / (self.config.cci_constant * mean_deviation)
        result[cci_column] = cci

        self.logger.debug(
            f"Calculated CCI with period {self.config.cci_period}, constant {self.config.cci_constant}"
        )
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on CCI extreme values and crossovers.

        This implementation generates signals for:
        1. Extreme CCI readings (overbought/oversold)
        2. CCI zero-line crossovers
        3. Divergence signals
        4. Trend momentum signals

        Args:
            data: DataFrame with market data and calculated CCI values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if CCI was calculated
        cci_column = f"{self.name.lower()}_cci"
        if cci_column not in data.columns:
            self.logger.warning(f"CCI column {cci_column} not found in data")
            return []

        # Get CCI data, removing NaN values
        cci_data = data[cci_column].dropna()

        if len(cci_data) < 2:
            return []

        signals = []
        current_cci = cci_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(self._generate_extreme_reading_signals(current_cci, current_timestamp))
        signals.extend(self._generate_moderate_level_signals(current_cci, current_timestamp))
        signals.extend(self._generate_crossover_signals(cci_data, current_cci, current_timestamp))
        signals.extend(self._generate_trend_signals(current_cci, current_timestamp))
        signals.extend(
            self._generate_divergence_signals(cci_data, data['close'], current_timestamp)
        )
        signals.extend(self._generate_momentum_signals(cci_data, current_cci, current_timestamp))
        signals.extend(self._generate_very_extreme_signals(current_cci, current_timestamp))

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_extreme_reading_signals(
        self, current_cci: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate extreme CCI reading signals (±100 levels)."""
        signals = []
        if current_cci > 100:
            extreme_action = f"Extreme overbought: CCI {current_cci:.1f} above +100"
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=extreme_action,
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_cci,
                extreme_level=100,
                condition="extreme_overbought",
            )
            signals.append(extreme_signal)
        elif current_cci < -100:
            extreme_action = f"Extreme oversold: CCI {current_cci:.1f} below -100"
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=extreme_action,
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_cci,
                extreme_level=-100,
                condition="extreme_oversold",
            )
            signals.append(extreme_signal)

        return signals

    def _generate_moderate_level_signals(
        self, current_cci: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate moderate overbought/oversold signals (±50 levels)."""
        signals = []
        if current_cci > 50:
            moderate_action = f"Overbought: CCI {current_cci:.1f} above +50"
            moderate_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=moderate_action,
                confidence=0.6,
                timestamp=timestamp,
                indicator_value=current_cci,
                moderate_level=50,
                condition="overbought",
            )
            signals.append(moderate_signal)
        elif current_cci < -50:
            moderate_action = f"Oversold: CCI {current_cci:.1f} below -50"
            moderate_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=moderate_action,
                confidence=0.6,
                timestamp=timestamp,
                indicator_value=current_cci,
                moderate_level=-50,
                condition="oversold",
            )
            signals.append(moderate_signal)

        return signals

    def _generate_crossover_signals(
        self, cci_data: pd.Series, current_cci: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate zero-line crossover signals."""
        signals = []
        if len(cci_data) > 1:
            prev_cci = cci_data.iloc[-2]

            # CCI crosses above zero (bullish)
            if prev_cci <= 0 and current_cci > 0:
                crossover_action = f"CCI bullish crossover: {current_cci:.1f} above zero"
                crossover_signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_cci,
                    crossover_type="above_zero",
                )
                signals.append(crossover_signal)

            # CCI crosses below zero (bearish)
            elif prev_cci >= 0 and current_cci < 0:
                crossover_action = f"CCI bearish crossover: {current_cci:.1f} below zero"
                crossover_signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_cci,
                    crossover_type="below_zero",
                )
                signals.append(crossover_signal)

        return signals

    def _generate_trend_signals(self, current_cci: float, timestamp: Any) -> list[dict[str, Any]]:
        """Generate trend signals based on CCI level."""
        signals = []
        if current_cci > 0:
            trend_action = f"Positive CCI: {current_cci:.1f} (bullish territory)"
            trend_confidence = min(0.6, abs(current_cci) / 200)
            trend_signal_type = SignalType.BUY
        else:
            trend_action = f"Negative CCI: {current_cci:.1f} (bearish territory)"
            trend_confidence = min(0.6, abs(current_cci) / 200)
            trend_signal_type = SignalType.SELL

        trend_signal = self._create_standard_signal(
            signal_type=trend_signal_type,
            action=trend_action,
            confidence=trend_confidence,
            timestamp=timestamp,
            indicator_value=current_cci,
            cci_level="positive" if current_cci > 0 else "negative",
        )
        signals.append(trend_signal)
        return signals

    def _generate_divergence_signals(
        self, cci_data: pd.Series, price_data: pd.Series, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate divergence signals."""
        signals = []
        if len(cci_data) >= 10:
            divergence_signal = self._check_divergence(cci_data, price_data, timestamp)
            if divergence_signal:
                signals.append(divergence_signal)

        return signals

    def _generate_momentum_signals(
        self, cci_data: pd.Series, current_cci: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate momentum signals based on CCI slope."""
        signals = []
        if len(cci_data) > 1:
            cci_slope = current_cci - cci_data.iloc[-2]
            if cci_slope > 0:
                momentum_action = f"CCI momentum increasing: {cci_slope:.1f} points"
                momentum_signal_type = SignalType.BUY
            elif cci_slope < 0:
                momentum_action = f"CCI momentum decreasing: {abs(cci_slope):.1f} points"
                momentum_signal_type = SignalType.SELL
            else:
                momentum_action = "CCI momentum flat"
                momentum_signal_type = SignalType.HOLD

            momentum_signal = self._create_standard_signal(
                signal_type=momentum_signal_type,
                action=momentum_action,
                confidence=min(0.6, abs(cci_slope) / 20),
                timestamp=timestamp,
                indicator_value=current_cci,
                cci_slope=cci_slope,
                momentum_direction=(
                    "increasing" if cci_slope > 0 else "decreasing" if cci_slope < 0 else "flat"
                ),
            )
            signals.append(momentum_signal)

        return signals

    def _generate_very_extreme_signals(
        self, current_cci: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate very extreme reading signals (±200 levels)."""
        signals = []
        if current_cci > 200:
            very_extreme_action = f"Very extreme overbought: CCI {current_cci:.1f} above +200"
            very_extreme_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=very_extreme_action,
                confidence=0.9,
                timestamp=timestamp,
                indicator_value=current_cci,
                extreme_condition="very_extreme_overbought",
            )
            signals.append(very_extreme_signal)
        elif current_cci < -200:
            very_extreme_action = f"Very extreme oversold: CCI {current_cci:.1f} below -200"
            very_extreme_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=very_extreme_action,
                confidence=0.9,
                timestamp=timestamp,
                indicator_value=current_cci,
                extreme_condition="very_extreme_oversold",
            )
            signals.append(very_extreme_signal)

        return signals

    def _check_divergence(
        self, cci_data: pd.Series, price_data: pd.Series, current_timestamp: Any
    ) -> dict[str, Any] | None:
        """Check for CCI-price divergence.

        Args:
            cci_data: CCI time series
            price_data: Price time series
            current_timestamp: Current timestamp

        Returns:
            Divergence signal if found, None otherwise
        """
        if len(cci_data) < 10 or len(price_data) < 10:
            return None

        # Look for divergence in the last 5 periods
        recent_periods = 5
        cci_recent = cci_data.tail(recent_periods)
        price_recent = price_data.tail(recent_periods)

        # Check for bullish divergence: price making lower lows, CCI making higher lows
        price_lower_lows = price_recent.iloc[-1] < price_recent.iloc[0]
        cci_higher_lows = cci_recent.iloc[-1] > cci_recent.iloc[0]

        if price_lower_lows and cci_higher_lows and cci_recent.iloc[-1] < 50:
            return self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Bullish divergence: price lower lows, CCI higher lows",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=cci_data.iloc[-1],
                divergence_type="bullish",
            )

        # Check for bearish divergence: price making higher highs, CCI making lower highs
        price_higher_highs = price_recent.iloc[-1] > price_recent.iloc[0]
        cci_lower_highs = cci_recent.iloc[-1] < cci_recent.iloc[0]

        if price_higher_highs and cci_lower_highs and cci_recent.iloc[-1] > 50:
            return self._create_standard_signal(
                signal_type=SignalType.SELL,
                action="Bearish divergence: price higher highs, CCI lower highs",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=cci_data.iloc[-1],
                divergence_type="bearish",
            )

        return None

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_cci"]

    def calculate_typical_price(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Typical Price values.

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            Series containing Typical Price values
        """
        return (data['high'] + data['low'] + data['close']) / 3

    def get_cci_levels(self) -> dict[str, float]:
        """Get standard CCI interpretation levels.

        Returns:
            Dictionary with standard CCI levels
        """
        return {
            "extreme_overbought": 200,
            "overbought": 100,
            "moderate_overbought": 50,
            "neutral": 0,
            "moderate_oversold": -50,
            "oversold": -100,
            "extreme_oversold": -200,
        }
