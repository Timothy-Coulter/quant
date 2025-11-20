"""Stochastic Oscillator indicator implementation.

This module provides the Stochastic Oscillator momentum indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("stochastic")
class StochasticOscillator(BaseIndicator):
    """Stochastic Oscillator momentum indicator.

    The Stochastic Oscillator is a momentum indicator that compares a security's
    closing price to its price range over a given time period. The oscillator
    ranges from 0 to 100. Readings above 80 are considered overbought, while
    readings below 20 are considered oversold. The %K line is the main signal
    line, while %D is a moving average of %K.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the baseline Stochastic Oscillator configuration."""
        return IndicatorConfig(
            indicator_name="stochastic",
            factory_name="stochastic",
            indicator_type="momentum",
            k_period=14,
            d_period=3,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator values (%K and %D).

        The Stochastic calculation involves:
        1. Calculate %K: ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        2. Calculate %D: Simple Moving Average of %K over d_period

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with Stochastic values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate rolling min and max for the k_period
        low_min = data['low'].rolling(window=self.config.k_period).min()
        high_max = data['high'].rolling(window=self.config.k_period).max()

        # Calculate %K
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))

        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=self.config.d_period).mean()

        # Add to result DataFrame
        k_column = f"{self.name.lower()}_k"
        d_column = f"{self.name.lower()}_d"

        result[k_column] = k_percent
        result[d_column] = d_percent

        self.logger.debug(
            f"Calculated Stochastic with k_period={self.config.k_period}, d_period={self.config.d_period}"
        )
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on Stochastic overbought/oversold conditions.

        This implementation generates signals for:
        1. %K and %D crossover signals
        2. Overbought/oversold level signals
        3. Divergence signals
        4. Extreme readings

        Args:
            data: DataFrame with market data and calculated Stochastic values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if Stochastic was calculated
        k_column = f"{self.name.lower()}_k"
        d_column = f"{self.name.lower()}_d"

        if not all(col in data.columns for col in [k_column, d_column]):
            self.logger.warning("Stochastic columns not found in data")
            return []

        # Get Stochastic data, removing NaN values
        k_data = data[k_column].dropna()
        d_data = data[d_column].dropna()

        if len(k_data) < 2:
            return []

        signals = []
        current_k = k_data.iloc[-1]
        current_d = d_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(
            self._generate_crossover_signals(
                k_data, d_data, current_k, current_d, current_timestamp
            )
        )
        signals.extend(self._generate_overbought_oversold_signals(current_k, current_timestamp))
        signals.extend(self._generate_trend_signals(current_k, current_d, current_timestamp))
        signals.extend(self._generate_divergence_signals(k_data, data['close'], current_timestamp))
        signals.extend(self._generate_extreme_signals(current_k, current_timestamp))
        signals.extend(self._generate_momentum_signals(k_data, current_k, current_timestamp))

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_crossover_signals(
        self,
        k_data: pd.Series,
        d_data: pd.Series,
        current_k: float,
        current_d: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate %K and %D crossover signals."""
        signals = []
        if len(k_data) > 1:
            prev_k = k_data.iloc[-2]
            prev_d = d_data.iloc[-2]

            # %K crosses above %D (bullish)
            if prev_k <= prev_d and current_k > current_d:
                crossover_action = (
                    f"Stochastic bullish crossover: %K {current_k:.1f} above %D {current_d:.1f}"
                )
                signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_k,
                    signal_line_value=current_d,
                    crossover_type="k_above_d",
                )
                signals.append(signal)

            # %K crosses below %D (bearish)
            elif prev_k >= prev_d and current_k < current_d:
                crossover_action = (
                    f"Stochastic bearish crossover: %K {current_k:.1f} below %D {current_d:.1f}"
                )
                signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=crossover_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_k,
                    signal_line_value=current_d,
                    crossover_type="k_below_d",
                )
                signals.append(signal)

        return signals

    def _generate_overbought_oversold_signals(
        self, current_k: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate overbought/oversold level signals."""
        signals = []
        if current_k >= self.config.overbought_threshold:
            ob_action = (
                f"Overbought: %K {current_k:.1f} above {self.config.overbought_threshold:.0f}"
            )
            ob_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=ob_action,
                confidence=min(0.8, (current_k - self.config.overbought_threshold) / 20 + 0.5),
                timestamp=timestamp,
                indicator_value=current_k,
                overbought_threshold=self.config.overbought_threshold,
                condition="overbought",
            )
            signals.append(ob_signal)
        elif current_k <= self.config.oversold_threshold:
            os_action = f"Oversold: %K {current_k:.1f} below {self.config.oversold_threshold:.0f}"
            os_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=os_action,
                confidence=min(0.8, (self.config.oversold_threshold - current_k) / 20 + 0.5),
                timestamp=timestamp,
                indicator_value=current_k,
                oversold_threshold=self.config.oversold_threshold,
                condition="oversold",
            )
            signals.append(os_signal)

        return signals

    def _generate_trend_signals(
        self, current_k: float, current_d: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate trend signals based on %K and %D positions."""
        signals = []
        if current_k > current_d:
            trend_action = f"Bullish: %K {current_k:.1f} above %D {current_d:.1f}"
            trend_confidence = min(0.6, abs(current_k - current_d) / 10)
            trend_signal_type = SignalType.BUY
        else:
            trend_action = f"Bearish: %K {current_k:.1f} below %D {current_d:.1f}"
            trend_confidence = min(0.6, abs(current_k - current_d) / 10)
            trend_signal_type = SignalType.SELL

        trend_signal = self._create_standard_signal(
            signal_type=trend_signal_type,
            action=trend_action,
            confidence=trend_confidence,
            timestamp=timestamp,
            indicator_value=current_k,
            signal_line_value=current_d,
            k_d_distance=current_k - current_d,
        )
        signals.append(trend_signal)
        return signals

    def _generate_divergence_signals(
        self, k_data: pd.Series, price_data: pd.Series, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate divergence signals."""
        signals = []
        if len(k_data) >= 10:
            divergence_signal = self._check_divergence(k_data, price_data, timestamp)
            if divergence_signal:
                signals.append(divergence_signal)

        return signals

    def _generate_extreme_signals(self, current_k: float, timestamp: Any) -> list[dict[str, Any]]:
        """Generate extreme reading signals."""
        signals = []
        if current_k >= 95:  # Extreme overbought
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=f"Extreme overbought: %K {current_k:.1f} indicates potential reversal",
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_k,
                extreme_condition="extreme_overbought",
            )
            signals.append(extreme_signal)
        elif current_k <= 5:  # Extreme oversold
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=f"Extreme oversold: %K {current_k:.1f} indicates potential reversal",
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_k,
                extreme_condition="extreme_oversold",
            )
            signals.append(extreme_signal)

        return signals

    def _generate_momentum_signals(
        self, k_data: pd.Series, current_k: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate momentum signals based on %K slope."""
        signals = []
        if len(k_data) > 1:
            k_slope = current_k - k_data.iloc[-2]
            momentum_action = f"Stochastic momentum: {k_slope:.1f} points"
            momentum_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=momentum_action,
                confidence=min(0.6, abs(k_slope) / 5),
                timestamp=timestamp,
                indicator_value=current_k,
                k_slope=k_slope,
                momentum_direction=(
                    "bullish" if k_slope > 0 else "bearish" if k_slope < 0 else "neutral"
                ),
            )
            signals.append(momentum_signal)

        return signals

    def _check_divergence(
        self, k_data: pd.Series, price_data: pd.Series, current_timestamp: Any
    ) -> dict[str, Any] | None:
        """Check for Stochastic-price divergence.

        Args:
            k_data: %K time series
            price_data: Price time series
            current_timestamp: Current timestamp

        Returns:
            Divergence signal if found, None otherwise
        """
        if len(k_data) < 10 or len(price_data) < 10:
            return None

        # Look for divergence in the last 5 periods
        recent_periods = 5
        k_recent = k_data.tail(recent_periods)
        price_recent = price_data.tail(recent_periods)

        # Check for bullish divergence: price making lower lows, %K making higher lows
        price_lower_lows = price_recent.iloc[-1] < price_recent.iloc[0]
        k_higher_lows = k_recent.iloc[-1] > k_recent.iloc[0]

        if price_lower_lows and k_higher_lows and k_recent.iloc[-1] < 50:
            return self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Bullish divergence: price lower lows, Stochastic higher lows",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=k_data.iloc[-1],
                divergence_type="bullish",
            )

        # Check for bearish divergence: price making higher highs, %K making lower highs
        price_higher_highs = price_recent.iloc[-1] > price_recent.iloc[0]
        k_lower_highs = k_recent.iloc[-1] < k_recent.iloc[0]

        if price_higher_highs and k_lower_highs and k_recent.iloc[-1] > 50:
            return self._create_standard_signal(
                signal_type=SignalType.SELL,
                action="Bearish divergence: price higher highs, Stochastic lower highs",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=k_data.iloc[-1],
                divergence_type="bearish",
            )

        return None

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        base_name = self.name.lower()
        return [f"{base_name}_k", f"{base_name}_d"]
