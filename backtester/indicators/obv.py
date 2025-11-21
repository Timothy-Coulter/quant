"""On-Balance Volume (OBV) indicator implementation.

This module provides the On-Balance Volume volume indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("obv")
class OBVIndicator(BaseIndicator):
    """On-Balance Volume (OBV) volume indicator.

    On-Balance Volume is a momentum technical trading indicator that uses
    volume flow to predict changes in stock price. Joseph Granville developed
    this simple yet calculation-intensive indicator in the 1960s.

    OBV calculation:
    - If today's close > yesterday's close: OBV = yesterday's OBV + today's volume
    - If today's close < yesterday's close: OBV = yesterday's OBV - today's volume
    - If today's close = yesterday's close: OBV = yesterday's OBV

    OBV is used to detect divergences between volume and price action.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the default OBV configuration."""
        return IndicatorConfig(
            indicator_name="obv",
            factory_name="obv",
            indicator_type="volume",
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume (OBV) values.

        The OBV calculation involves:
        1. Compare current close to previous close
        2. Add or subtract volume based on price direction
        3. Create cumulative total

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with OBV values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate price changes
        price_change = data['close'].diff()

        # Initialize OBV series
        obv_column = f"{self.name.lower()}_obv"
        obv = pd.Series(index=data.index, dtype=float)

        # Calculate OBV
        obv.iloc[0] = data['volume'].iloc[0]  # First value is first volume

        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                # Close increased: add volume
                obv.iloc[i] = obv.iloc[i - 1] + data['volume'].iloc[i]
            elif price_change.iloc[i] < 0:
                # Close decreased: subtract volume
                obv.iloc[i] = obv.iloc[i - 1] - data['volume'].iloc[i]
            else:
                # Close unchanged: no change
                obv.iloc[i] = obv.iloc[i - 1]

        result[obv_column] = obv

        self.logger.debug(f"Calculated OBV with volume column {self.config.volume_column}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on OBV divergences and trends.

        This implementation generates signals for:
        1. OBV price divergences
        2. OBV trend analysis
        3. Volume-price confirmation signals
        4. OBV breakout signals

        Args:
            data: DataFrame with market data and calculated OBV values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if OBV was calculated
        obv_column = f"{self.name.lower()}_obv"
        if obv_column not in data.columns:
            self.logger.warning(f"OBV column {obv_column} not found in data")
            return []

        # Get OBV and price data, removing NaN values
        obv_data = data[obv_column].dropna()
        price_data = data['close'].dropna()
        volume_data = data['volume'].dropna()

        if len(obv_data) < 2:
            return []

        signals = []
        current_obv = obv_data.iloc[-1]
        current_price = price_data.iloc[-1]
        current_volume = volume_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(self._generate_divergence_signals(obv_data, price_data, current_timestamp))
        signals.extend(
            self._generate_trend_confirmation_signals(
                obv_data, price_data, current_obv, current_timestamp
            )
        )
        signals.extend(
            self._generate_volume_price_signals(
                obv_data, price_data, current_obv, current_price, current_timestamp
            )
        )
        signals.extend(
            self._generate_high_volume_signals(
                volume_data, current_volume, current_obv, current_timestamp
            )
        )
        signals.extend(self._generate_momentum_signals(obv_data, current_obv, current_timestamp))
        signals.extend(self._generate_level_signals(obv_data, current_obv, current_timestamp))

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_divergence_signals(
        self, obv_data: pd.Series, price_data: pd.Series, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate divergence analysis signals."""
        signals = []
        divergence_signal = self._check_divergence(obv_data, price_data, timestamp)
        if divergence_signal:
            signals.append(divergence_signal)
        return signals

    def _generate_trend_confirmation_signals(
        self, obv_data: pd.Series, price_data: pd.Series, current_obv: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate OBV trend confirmation signals."""
        signals = []
        if len(obv_data) > 10:
            # Calculate recent OBV slope
            recent_obv = obv_data.tail(10)
            recent_price = price_data.tail(10)

            # Linear regression slope for OBV
            obv_slope = self._calculate_slope(recent_obv.index, recent_obv.values)
            price_slope = self._calculate_slope(recent_price.index, recent_price.values)

            # Trend confirmation signals
            if obv_slope > 0 and price_slope > 0:
                trend_action = "OBV confirms uptrend: OBV rising, price rising"
                trend_signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=trend_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    obv_slope=obv_slope,
                    price_slope=price_slope,
                    trend_confirmation="bullish",
                )
                signals.append(trend_signal)
            elif obv_slope < 0 and price_slope < 0:
                trend_action = "OBV confirms downtrend: OBV falling, price falling"
                trend_signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=trend_action,
                    confidence=0.7,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    obv_slope=obv_slope,
                    price_slope=price_slope,
                    trend_confirmation="bearish",
                )
                signals.append(trend_signal)

        return signals

    def _generate_volume_price_signals(
        self,
        obv_data: pd.Series,
        price_data: pd.Series,
        current_obv: float,
        current_price: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate volume-price confirmation signals."""
        signals = []
        if len(obv_data) > 1:
            prev_obv = obv_data.iloc[-2]
            prev_price = price_data.iloc[-2]

            # Check if volume supports price movement
            price_increase = current_price > prev_price
            obv_increase = current_obv > prev_obv

            if price_increase and obv_increase:
                volume_action = (
                    f"Volume confirms price increase: ${current_price:.2f} with higher OBV"
                )
                volume_signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=volume_action,
                    confidence=0.6,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    price_increase=price_increase,
                    obv_increase=obv_increase,
                    volume_confirmation=True,
                )
                signals.append(volume_signal)
            elif not price_increase and not obv_increase:
                volume_action = (
                    f"Volume confirms price decrease: ${current_price:.2f} with lower OBV"
                )
                volume_signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=volume_action,
                    confidence=0.6,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    price_increase=price_increase,
                    obv_increase=obv_increase,
                    volume_confirmation=True,
                )
                signals.append(volume_signal)

        return signals

    def _generate_high_volume_signals(
        self, volume_data: pd.Series, current_volume: float, current_obv: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate high volume signals."""
        signals = []
        if len(volume_data) > 20:
            avg_volume = volume_data.tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio > 2.0:  # 2x average volume
                high_vol_action = f"High volume spike: {current_volume:,} vs avg {avg_volume:,.0f} (ratio: {volume_ratio:.1f})"
                high_vol_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=high_vol_action,
                    confidence=0.5,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    volume_ratio=volume_ratio,
                    current_volume=current_volume,
                    average_volume=avg_volume,
                )
                signals.append(high_vol_signal)

        return signals

    def _generate_momentum_signals(
        self, obv_data: pd.Series, current_obv: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate OBV momentum signals."""
        signals = []
        if len(obv_data) > 5:
            recent_obv_change = current_obv - obv_data.iloc[-6]  # 5 periods ago
            obv_momentum = "increasing" if recent_obv_change > 0 else "decreasing"
            momentum_action = (
                f"OBV momentum {obv_momentum}: {recent_obv_change:,.0f} change over 5 periods"
            )
            momentum_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=momentum_action,
                confidence=0.4,
                timestamp=timestamp,
                indicator_value=current_obv,
                obv_momentum=obv_momentum,
                obv_change=recent_obv_change,
            )
            signals.append(momentum_signal)

        return signals

    def _generate_level_signals(
        self, obv_data: pd.Series, current_obv: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate OBV level signals."""
        signals = []
        if len(obv_data) > 50:
            obv_percentile = (obv_data < current_obv).sum() / len(obv_data) * 100

            if obv_percentile > 80:  # Near recent high
                level_action = f"OBV at {obv_percentile:.0f}th percentile: near recent highs"
                level_signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=level_action,
                    confidence=0.5,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    obv_percentile=obv_percentile,
                    level="near_highs",
                )
                signals.append(level_signal)
            elif obv_percentile < 20:  # Near recent low
                level_action = f"OBV at {obv_percentile:.0f}th percentile: near recent lows"
                level_signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=level_action,
                    confidence=0.5,
                    timestamp=timestamp,
                    indicator_value=current_obv,
                    obv_percentile=obv_percentile,
                    level="near_lows",
                )
                signals.append(level_signal)

        return signals

    def _check_divergence(
        self, obv_data: pd.Series, price_data: pd.Series, current_timestamp: Any
    ) -> dict[str, Any] | None:
        """Check for OBV-price divergence.

        Args:
            obv_data: OBV time series
            price_data: Price time series
            current_timestamp: Current timestamp

        Returns:
            Divergence signal if found, None otherwise
        """
        if len(obv_data) < 10 or len(price_data) < 10:
            return None

        # Look for divergence in the last 5 periods
        recent_periods = 5
        obv_recent = obv_data.tail(recent_periods)
        price_recent = price_data.tail(recent_periods)

        # Check for bullish divergence: price making lower lows, OBV making higher lows
        price_lower_lows = price_recent.iloc[-1] < price_recent.iloc[0]
        obv_higher_lows = obv_recent.iloc[-1] > obv_recent.iloc[0]

        if price_lower_lows and obv_higher_lows:
            return self._create_standard_signal(
                signal_type=SignalType.BUY,
                action="Bullish divergence: price lower lows, OBV higher lows",
                confidence=0.8,
                timestamp=current_timestamp,
                indicator_value=obv_data.iloc[-1],
                divergence_type="bullish",
            )

        # Check for bearish divergence: price making higher highs, OBV making lower highs
        price_higher_highs = price_recent.iloc[-1] > price_recent.iloc[0]
        obv_lower_highs = obv_recent.iloc[-1] < obv_recent.iloc[0]

        if price_higher_highs and obv_lower_highs:
            return self._create_standard_signal(
                signal_type=SignalType.SELL,
                action="Bearish divergence: price higher highs, OBV lower highs",
                confidence=0.8,
                timestamp=current_timestamp,
                indicator_value=obv_data.iloc[-1],
                divergence_type="bearish",
            )

        return None

    def _calculate_slope(self, x_values: pd.Index, y_values: pd.Series | NDArray[Any]) -> float:
        """Calculate the slope using simple linear regression.

        Args:
            x_values: X values (typically timestamp indices)
            y_values: Y values to calculate slope for (can be Series or array)

        Returns:
            Slope value
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        # Convert to numeric indices for calculation
        x_numeric = np.arange(len(x_values))
        # Handle both Series and numpy array inputs
        y_numeric = y_values.values if hasattr(y_values, 'values') else y_values

        # Simple linear regression slope calculation
        n = len(x_numeric)
        sum_x = np.sum(x_numeric)
        sum_y = np.sum(y_numeric)
        sum_xy = np.sum(x_numeric * y_numeric)
        sum_x2 = np.sum(x_numeric * x_numeric)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_obv"]

    def calculate_obv_change(self, data: pd.DataFrame, periods: int = 1) -> float:
        """Calculate the change in OBV over a specified number of periods.

        Args:
            data: DataFrame with calculated OBV values
            periods: Number of periods to look back

        Returns:
            OBV change value
        """
        obv_column = f"{self.name.lower()}_obv"
        if obv_column not in data.columns or len(data) <= periods:
            return 0.0

        current_obv = data[obv_column].iloc[-1]
        previous_obv = data[obv_column].iloc[-1 - periods]

        return float(current_obv - previous_obv)
