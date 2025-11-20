"""Williams %R (Williams Percent Range) indicator implementation.

This module provides the Williams %R momentum indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.indicators.base_indicator import BaseIndicator, IndicatorFactory
from backtester.indicators.indicator_configs import IndicatorConfig
from backtester.signal.signal_types import SignalType


@IndicatorFactory.register("williams_r")
class WilliamsROscillator(BaseIndicator):
    """Williams %R (Williams Percent Range) momentum indicator.

    Williams %R is a momentum indicator that measures overbought and oversold levels.
    It is similar to the Stochastic Oscillator except that %R is plotted on a negative
    scale from -100 to 0. Readings below -80 are considered oversold, while readings
    above -20 are considered overbought.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the canonical Williams %R configuration."""
        return IndicatorConfig(
            indicator_name="williams_r",
            factory_name="williams_r",
            indicator_type="momentum",
            williams_r_period=14,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R values.

        The Williams %R calculation involves:
        %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

        Where:
        - Highest High is the highest price during the lookback period
        - Lowest Low is the lowest price during the lookback period
        - Close is the current closing price

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with Williams %R values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate rolling min and max for the williams_r_period
        low_min = data['low'].rolling(window=self.config.williams_r_period).min()
        high_max = data['high'].rolling(window=self.config.williams_r_period).max()

        # Calculate Williams %R
        # %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100
        williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))

        # Add to result DataFrame
        wr_column = f"{self.name.lower()}_wr"
        result[wr_column] = williams_r

        self.logger.debug(f"Calculated Williams %R with period {self.config.williams_r_period}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on Williams %R overbought/oversold conditions.

        This implementation generates signals for:
        1. Overbought/oversold level signals (above -20 / below -80)
        2. Signal line crossovers (simplified for single-line indicator)
        3. Divergence signals
        4. Extreme readings

        Args:
            data: DataFrame with market data and calculated Williams %R values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if Williams %R was calculated
        wr_column = f"{self.name.lower()}_wr"
        if wr_column not in data.columns:
            self.logger.warning(f"Williams %R column {wr_column} not found in data")
            return []

        # Get Williams %R data, removing NaN values
        wr_data = data[wr_column].dropna()

        if len(wr_data) < 2:
            return []

        signals = []
        current_wr = wr_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(self._generate_overbought_oversold_signals(current_wr, current_timestamp))
        signals.extend(self._generate_extreme_condition_signals(current_wr, current_timestamp))
        signals.extend(self._generate_momentum_signals(wr_data, current_wr, current_timestamp))
        signals.extend(self._generate_divergence_signals(wr_data, data['close'], current_timestamp))
        signals.extend(
            self._generate_slope_momentum_signals(wr_data, current_wr, current_timestamp)
        )

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_overbought_oversold_signals(
        self, current_wr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate overbought/oversold level signals."""
        signals = []
        if current_wr >= -20:  # Overbought (less negative)
            ob_action = f"Overbought: %R {current_wr:.1f} above -20"
            ob_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=ob_action,
                confidence=min(0.8, (-20 - current_wr) / 20 + 0.5),
                timestamp=timestamp,
                indicator_value=current_wr,
                overbought_threshold=-20,
                condition="overbought",
            )
            signals.append(ob_signal)
        elif current_wr <= -80:  # Oversold (more negative)
            os_action = f"Oversold: %R {current_wr:.1f} below -80"
            os_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=os_action,
                confidence=min(0.8, (-80 - current_wr) / -20 + 0.5),
                timestamp=timestamp,
                indicator_value=current_wr,
                oversold_threshold=-80,
                condition="oversold",
            )
            signals.append(os_signal)

        return signals

    def _generate_extreme_condition_signals(
        self, current_wr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate extreme condition signals."""
        signals = []
        if current_wr > -10:  # Extremely overbought
            extreme_action = f"Extremely overbought: %R {current_wr:.1f}"
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=extreme_action,
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_wr,
                extreme_condition="extreme_overbought",
            )
            signals.append(extreme_signal)
        elif current_wr < -90:  # Extremely oversold
            extreme_action = f"Extremely oversold: %R {current_wr:.1f}"
            extreme_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=extreme_action,
                confidence=0.8,
                timestamp=timestamp,
                indicator_value=current_wr,
                extreme_condition="extreme_oversold",
            )
            signals.append(extreme_signal)

        return signals

    def _generate_momentum_signals(
        self, wr_data: pd.Series, current_wr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate momentum signals based on neutral zone."""
        signals = []
        if len(wr_data) > 1:
            # Signal at -50 (neutral zone)
            neutral_zone = -50
            distance_from_neutral = abs(current_wr - neutral_zone)

            if current_wr > neutral_zone:  # Above neutral
                if current_wr > -20:  # Overbought - potential sell
                    action = f"Overbought zone: %R {current_wr:.1f}"
                    signal_type = SignalType.SELL
                else:
                    action = f"Above neutral: %R {current_wr:.1f}"
                    signal_type = SignalType.BUY  # Slight bullish bias
            else:  # Below neutral
                if current_wr < -80:  # Oversold - potential buy
                    action = f"Oversold zone: %R {current_wr:.1f}"
                    signal_type = SignalType.BUY
                else:
                    action = f"Below neutral: %R {current_wr:.1f}"
                    signal_type = SignalType.SELL  # Slight bearish bias

            # Adjust confidence based on distance from neutral
            base_confidence = min(0.6, distance_from_neutral / 30)

            momentum_signal = self._create_standard_signal(
                signal_type=signal_type,
                action=action,
                confidence=base_confidence,
                timestamp=timestamp,
                indicator_value=current_wr,
                distance_from_neutral=distance_from_neutral,
                neutral_zone=neutral_zone,
            )
            signals.append(momentum_signal)

        return signals

    def _generate_divergence_signals(
        self, wr_data: pd.Series, price_data: pd.Series, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate divergence signals."""
        signals = []
        if len(wr_data) >= 10:
            divergence_signal = self._check_divergence(wr_data, price_data, timestamp)
            if divergence_signal:
                signals.append(divergence_signal)

        return signals

    def _generate_slope_momentum_signals(
        self, wr_data: pd.Series, current_wr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate momentum signals based on %R slope."""
        signals = []
        if len(wr_data) > 1:
            wr_slope = current_wr - wr_data.iloc[-2]
            # Since %R is negative, positive slope means moving toward neutral (bullish)
            # and negative slope means moving away from neutral (bearish)
            if wr_slope > 0:  # Moving up (toward zero/overbought)
                momentum_action = f"Bullish momentum: %R rising {wr_slope:.1f} points"
                momentum_signal_type = SignalType.BUY
            elif wr_slope < 0:  # Moving down (toward -100/oversold)
                momentum_action = f"Bearish momentum: %R falling {abs(wr_slope):.1f} points"
                momentum_signal_type = SignalType.SELL
            else:
                momentum_action = "No momentum: %R flat"
                momentum_signal_type = SignalType.HOLD

            momentum_signal = self._create_standard_signal(
                signal_type=momentum_signal_type,
                action=momentum_action,
                confidence=min(0.6, abs(wr_slope) / 10),
                timestamp=timestamp,
                indicator_value=current_wr,
                wr_slope=wr_slope,
                momentum_direction=(
                    "bullish" if wr_slope > 0 else "bearish" if wr_slope < 0 else "neutral"
                ),
            )
            signals.append(momentum_signal)

        return signals

    def _check_divergence(
        self, wr_data: pd.Series, price_data: pd.Series, current_timestamp: Any
    ) -> dict[str, Any] | None:
        """Check for Williams %R-price divergence.

        Args:
            wr_data: Williams %R time series
            price_data: Price time series
            current_timestamp: Current timestamp

        Returns:
            Divergence signal if found, None otherwise
        """
        if len(wr_data) < 10 or len(price_data) < 10:
            return None

        # Look for divergence in the last 5 periods
        recent_periods = 5
        wr_recent = wr_data.tail(recent_periods)
        price_recent = price_data.tail(recent_periods)

        # Check for bullish divergence: price making lower lows, %R making higher lows (less negative)
        price_lower_lows = price_recent.iloc[-1] < price_recent.iloc[0]
        wr_higher_lows = wr_recent.iloc[-1] > wr_recent.iloc[0]  # Higher means less negative

        if price_lower_lows and wr_higher_lows and wr_recent.iloc[-1] < -50:
            return self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Bullish divergence: price lower lows, Williams %R higher lows",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=wr_data.iloc[-1],
                divergence_type="bullish",
            )

        # Check for bearish divergence: price making higher highs, %R making lower highs (more negative)
        price_higher_highs = price_recent.iloc[-1] > price_recent.iloc[0]
        wr_lower_highs = wr_recent.iloc[-1] < wr_recent.iloc[0]  # Lower means more negative

        if price_higher_highs and wr_lower_highs and wr_recent.iloc[-1] > -50:
            return self._create_standard_signal(
                signal_type=SignalType.SELL,
                action="Bearish divergence: price higher highs, Williams %R lower highs",
                confidence=0.7,
                timestamp=current_timestamp,
                indicator_value=wr_data.iloc[-1],
                divergence_type="bearish",
            )

        return None

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_wr"]
