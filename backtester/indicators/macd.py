"""Moving Average Convergence Divergence (MACD) indicator implementation.

This module provides the MACD momentum indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("macd")
class MACDIndicator(BaseIndicator):
    """Moving Average Convergence Divergence momentum indicator.

    MACD is a trend-following momentum indicator that shows the relationship between
    two moving averages of a security's price. The MACD is calculated by subtracting
    the 26-period EMA from the 12-period EMA. The result of this calculation is
    the MACD line. A nine-period EMA of the MACD (called the "signal line") is
    then plotted on top of the MACD line, which can function as a trigger for
    buy and sell signals.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the standard MACD configuration."""
        return IndicatorConfig(
            indicator_name="macd",
            factory_name="macd",
            indicator_type="momentum",
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD values (MACD line, signal line, and histogram).

        The MACD calculation involves:
        1. Calculate 12-period EMA (fast EMA)
        2. Calculate 26-period EMA (slow EMA)
        3. MACD line = fast EMA - slow EMA
        4. Signal line = 9-period EMA of MACD line
        5. Histogram = MACD line - signal line

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with MACD values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Get the price series to calculate MACD on
        price_series = data[self.config.price_column]

        # Calculate EMAs
        ema_fast = price_series.ewm(span=self.config.fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=self.config.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=self.config.signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Add to result DataFrame
        macd_column = f"{self.name.lower()}_macd"
        signal_column = f"{self.name.lower()}_signal"
        histogram_column = f"{self.name.lower()}_histogram"

        result[macd_column] = macd_line
        result[signal_column] = signal_line
        result[histogram_column] = histogram

        self.logger.debug(
            f"Calculated MACD with fast={self.config.fast_period}, slow={self.config.slow_period}, signal={self.config.signal_period}"
        )
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on MACD crossovers and zero-line crosses.

        This implementation generates signals for:
        1. MACD line crossing above/below signal line
        2. MACD line crossing above/below zero line
        3. Histogram momentum changes
        4. Multiple timeframe analysis

        Args:
            data: DataFrame with market data and calculated MACD values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if MACD was calculated
        macd_column = f"{self.name.lower()}_macd"
        signal_column = f"{self.name.lower()}_signal"
        histogram_column = f"{self.name.lower()}_histogram"

        if not all(col in data.columns for col in [macd_column, signal_column, histogram_column]):
            self.logger.warning("MACD columns not found in data")
            return []

        # Get MACD data, removing NaN values
        macd_data = data[macd_column].dropna()
        signal_data = data[signal_column].dropna()
        histogram_data = data[histogram_column].dropna()

        if len(macd_data) < 2:
            return []

        signals = []
        current_macd = macd_data.iloc[-1]
        current_signal = signal_data.iloc[-1]
        current_histogram = histogram_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(
            self._generate_crossover_signals(
                macd_data, signal_data, current_macd, current_signal, current_timestamp
            )
        )
        signals.extend(self._generate_zero_line_signals(macd_data, current_macd, current_timestamp))
        signals.extend(
            self._generate_trend_signals(
                current_macd, current_signal, current_timestamp, macd_data, signal_data
            )
        )
        signals.extend(
            self._generate_histogram_signals(histogram_data, current_histogram, current_timestamp)
        )
        signals.extend(self._generate_extreme_signals(current_macd, current_timestamp))

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_crossover_signals(
        self,
        macd_data: pd.Series,
        signal_data: pd.Series,
        current_macd: float,
        current_signal: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate MACD vs signal line crossover signals."""
        signals = []
        if len(macd_data) > 1:
            prev_macd = macd_data.iloc[-2]
            prev_signal = signal_data.iloc[-2]

            # MACD crosses above signal line (bullish)
            if prev_macd <= prev_signal and current_macd > current_signal:
                crossover_action = f"MACD bullish crossover: MACD {current_macd:.4f} above signal {current_signal:.4f}"
                signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_macd,
                    signal_line_value=current_signal,
                    crossover_type="macd_above_signal",
                )
                signals.append(signal)

            # MACD crosses below signal line (bearish)
            elif prev_macd >= prev_signal and current_macd < current_signal:
                crossover_action = f"MACD bearish crossover: MACD {current_macd:.4f} below signal {current_signal:.4f}"
                signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_macd,
                    signal_line_value=current_signal,
                    crossover_type="macd_below_signal",
                )
                signals.append(signal)

        return signals

    def _generate_zero_line_signals(
        self, macd_data: pd.Series, current_macd: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate zero line crossover signals."""
        signals = []
        if len(macd_data) > 1:
            prev_macd = macd_data.iloc[-2]

            # MACD crosses above zero (bullish)
            if prev_macd <= 0 and current_macd > 0:
                zero_action = f"MACD bullish zero crossover: {current_macd:.4f} above zero"
                signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=zero_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_macd,
                    crossover_type="above_zero",
                )
                signals.append(signal)

            # MACD crosses below zero (bearish)
            elif prev_macd >= 0 and current_macd < 0:
                zero_action = f"MACD bearish zero crossover: {current_macd:.4f} below zero"
                signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=zero_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_macd,
                    crossover_type="below_zero",
                )
                signals.append(signal)

        return signals

    def _generate_trend_signals(
        self,
        current_macd: float,
        current_signal: float,
        timestamp: Any,
        macd_data: pd.Series,
        signal_data: pd.Series,
    ) -> list[dict[str, Any]]:
        """Generate trend signals based on MACD and signal line positions."""
        signals = []
        if current_macd > current_signal:
            trend_action = (
                f"Bullish trend: MACD {current_macd:.4f} above signal {current_signal:.4f}"
            )
            trend_confidence = min(0.6, abs(current_macd - current_signal) * 1000)
            trend_signal_type = SignalType.BUY
        else:
            trend_action = (
                f"Bearish trend: MACD {current_macd:.4f} below signal {current_signal:.4f}"
            )
            trend_confidence = min(0.6, abs(current_macd - current_signal) * 1000)
            trend_signal_type = SignalType.SELL

        # Create trend signal
        trend_signal = self._create_standard_signal(
            signal_type=trend_signal_type,
            action=trend_action,
            confidence=trend_confidence,
            timestamp=timestamp,
            indicator_value=current_macd,
            signal_line_value=current_signal,
            macd_signal_distance=current_macd - current_signal,
            trend_strength=self._calculate_trend_strength(macd_data, signal_data),
        )
        signals.append(trend_signal)
        return signals

    def _generate_histogram_signals(
        self, histogram_data: pd.Series, current_histogram: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate histogram momentum signals."""
        signals = []
        if len(histogram_data) > 1:
            prev_histogram = histogram_data.iloc[-2]
            histogram_change = current_histogram - prev_histogram

            if abs(histogram_change) > 0.001:  # Significant change threshold
                momentum_direction = "increasing" if histogram_change > 0 else "decreasing"
                momentum_action = f"MACD histogram {momentum_direction}: {current_histogram:.4f}"
                momentum_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=momentum_action,
                    confidence=min(0.6, abs(histogram_change) * 1000),
                    timestamp=timestamp,
                    indicator_value=current_histogram,
                    histogram_change=histogram_change,
                    momentum_direction=momentum_direction,
                )
                signals.append(momentum_signal)

        return signals

    def _generate_extreme_signals(
        self, current_macd: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate extreme MACD reading signals."""
        signals = []
        if abs(current_macd) > 0.1:  # Arbitrary threshold for extreme readings
            extreme_action = f"Extreme MACD reading: {current_macd:.4f}"
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=extreme_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_macd,
                extreme_condition=True,
            )
            signals.append(extreme_signal)

        return signals

    def _calculate_trend_strength(self, macd_data: pd.Series, signal_data: pd.Series) -> str:
        """Calculate the strength of the current MACD trend.

        Args:
            macd_data: MACD line data
            signal_data: Signal line data

        Returns:
            Trend strength description
        """
        if len(macd_data) < 5:
            return "unknown"

        # Calculate recent MACD momentum
        recent_macd = macd_data.tail(5)
        recent_signal = signal_data.tail(5)

        # Check if MACD is consistently above/below signal
        macd_above_signal = (recent_macd > recent_signal).sum()
        macd_below_signal = (recent_macd < recent_signal).sum()

        if macd_above_signal >= 4:
            return "strong_bullish"
        elif macd_below_signal >= 4:
            return "strong_bearish"
        elif macd_above_signal > macd_below_signal:
            return "weak_bullish"
        elif macd_below_signal > macd_above_signal:
            return "weak_bearish"
        else:
            return "neutral"

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        base_name = self.name.lower()
        return [f"{base_name}_macd", f"{base_name}_signal", f"{base_name}_histogram"]
