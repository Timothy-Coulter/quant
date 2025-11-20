"""Relative Strength Index (RSI) indicator implementation.

This module provides the Relative Strength Index momentum indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("rsi")
class RSIIndicator(BaseIndicator):
    """Relative Strength Index momentum indicator.

    The RSI is a momentum oscillator that measures the speed and change of price movements,
    typically used to identify overbought or oversold conditions in a market. It oscillates
    between 0 and 100, with readings above 70 generally considered overbought and
    readings below 30 considered oversold.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the canonical RSI configuration."""
        return IndicatorConfig(
            indicator_name="rsi",
            factory_name="rsi",
            indicator_type="momentum",
            period=14,
            overbought_threshold=70.0,
            oversold_threshold=30.0,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI values.

        The RSI is calculated using the following steps:
        1. Calculate price changes (gains and losses)
        2. Calculate exponential moving averages of gains and losses
        3. Calculate the Relative Strength (RS) = avg_gain / avg_loss
        4. Calculate RSI = 100 - (100 / (1 + RS))

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with RSI values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate price changes
        delta = data['close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving averages
        avg_gains = gains.ewm(span=self.config.period, adjust=False).mean()
        avg_losses = losses.ewm(span=self.config.period, adjust=False).mean()

        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Add RSI to result DataFrame
        rsi_column = f"{self.name.lower()}_rsi"
        result[rsi_column] = rsi

        self.logger.debug(f"Calculated RSI with period {self.config.period}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on RSI overbought/oversold conditions.

        This implementation generates signals when RSI crosses above the oversold threshold
        (potential buy signal) or below the overbought threshold (potential sell signal).
        It also considers RSI divergence and extreme readings.

        Args:
            data: DataFrame with market data and calculated RSI values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if RSI was calculated
        rsi_column = f"{self.name.lower()}_rsi"
        if rsi_column not in data.columns:
            self.logger.warning(f"RSI column {rsi_column} not found in data")
            return []

        # Get RSI data, removing NaN values
        rsi_data = data[rsi_column].dropna()
        price_data = data['close'].dropna()

        if len(rsi_data) < 2:
            return []

        signals = []
        current_rsi = rsi_data.iloc[-1]
        current_price = price_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Determine current market condition based on RSI
        if current_rsi >= self.config.overbought_threshold:
            primary_signal = SignalType.SELL
            action = (
                f"Overbought: RSI {current_rsi:.1f} above {self.config.overbought_threshold:.0f}"
            )
            base_confidence = min(0.9, (current_rsi - self.config.overbought_threshold) / 20 + 0.5)
        elif current_rsi <= self.config.oversold_threshold:
            primary_signal = SignalType.BUY
            action = f"Oversold: RSI {current_rsi:.1f} below {self.config.oversold_threshold:.0f}"
            base_confidence = min(0.9, (self.config.oversold_threshold - current_rsi) / 20 + 0.5)
        else:
            primary_signal = SignalType.HOLD
            action = f"Neutral: RSI {current_rsi:.1f} in normal range"
            base_confidence = 0.3

        # Create the main RSI signal
        signal = self._create_standard_signal(
            signal_type=primary_signal,
            action=action,
            confidence=base_confidence,
            timestamp=current_timestamp,
            indicator_value=current_rsi,
            price_value=current_price,
            rsi_level=self._get_rsi_level(current_rsi),
            overbought_threshold=self.config.overbought_threshold,
            oversold_threshold=self.config.oversold_threshold,
            distance_from_neutral=abs(current_rsi - 50) / 50,
        )

        signals.append(signal)

        # Check for RSI divergences (simplified version)
        if len(rsi_data) >= 10:
            divergence_signal = self._check_divergence(rsi_data, price_data, current_timestamp)
            if divergence_signal:
                signals.append(divergence_signal)

        # Generate momentum signal based on RSI slope
        if len(rsi_data) > 1:
            rsi_slope = (current_rsi - rsi_data.iloc[-2]) if len(rsi_data) > 1 else 0
            momentum_action = f"RSI momentum: {rsi_slope:.1f} points"
            momentum_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=momentum_action,
                confidence=min(0.7, abs(rsi_slope) / 10),
                timestamp=current_timestamp,
                indicator_value=current_rsi,
                rsi_slope=rsi_slope,
                momentum_direction=(
                    "bullish" if rsi_slope > 0 else "bearish" if rsi_slope < 0 else "neutral"
                ),
            )
            signals.append(momentum_signal)

        # Generate signal for extreme RSI readings (below 10 or above 90)
        if current_rsi < 10:
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,  # Strong buy signal in extreme oversold
                action=f"Extreme oversold: RSI {current_rsi:.1f} indicates potential reversal",
                confidence=0.8,
                timestamp=current_timestamp,
                indicator_value=current_rsi,
                extreme_condition="extreme_oversold",
            )
            signals.append(extreme_signal)
        elif current_rsi > 90:
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,  # Strong sell signal in extreme overbought
                action=f"Extreme overbought: RSI {current_rsi:.1f} indicates potential reversal",
                confidence=0.8,
                timestamp=current_timestamp,
                indicator_value=current_rsi,
                extreme_condition="extreme_overbought",
            )
            signals.append(extreme_signal)

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _get_rsi_level(self, rsi_value: float) -> str:
        """Get the RSI level classification.

        Args:
            rsi_value: RSI value to classify

        Returns:
            RSI level description
        """
        if rsi_value >= 80:
            return "extremely_overbought"
        elif rsi_value >= 70:
            return "overbought"
        elif rsi_value <= 20:
            return "extremely_oversold"
        elif rsi_value <= 30:
            return "oversold"
        else:
            return "neutral"

    def _check_divergence(
        self, rsi_data: pd.Series, price_data: pd.Series, current_timestamp: Any
    ) -> dict[str, Any] | None:
        """Check for RSI-price divergence (simplified implementation).

        Args:
            rsi_data: RSI time series
            price_data: Price time series
            current_timestamp: Current timestamp

        Returns:
            Divergence signal if found, None otherwise
        """
        if len(rsi_data) < 10 or len(price_data) < 10:
            return None

        # Look for divergence in the last 5 periods
        recent_periods = 5
        rsi_recent = rsi_data.tail(recent_periods)
        price_recent = price_data.tail(recent_periods)

        # Check for bullish divergence: price making lower lows, RSI making higher lows
        price_lower_lows = price_recent.iloc[-1] < price_recent.iloc[0]
        rsi_higher_lows = rsi_recent.iloc[-1] > rsi_recent.iloc[0]

        if price_lower_lows and rsi_higher_lows and rsi_recent.iloc[-1] < 50:
            return self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Bullish divergence detected: price lower lows, RSI higher lows",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=rsi_data.iloc[-1],
                divergence_type="bullish",
            )

        # Check for bearish divergence: price making higher highs, RSI making lower highs
        price_higher_highs = price_recent.iloc[-1] > price_recent.iloc[0]
        rsi_lower_highs = rsi_recent.iloc[-1] < rsi_recent.iloc[0]

        if price_higher_highs and rsi_lower_highs and rsi_recent.iloc[-1] > 50:
            return self._create_standard_signal(
                signal_type=SignalType.SELL,
                action="Bearish divergence detected: price higher highs, RSI lower highs",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=rsi_data.iloc[-1],
                divergence_type="bearish",
            )

        return None

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_rsi"]
