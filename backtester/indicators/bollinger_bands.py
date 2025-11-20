"""Bollinger Bands indicator implementation.

This module provides the Bollinger Bands volatility indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("bollinger_bands")
class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands volatility indicator.

    Bollinger Bands are a type of statistical chart characterizing the prices and
    volatility over time of a financial instrument or commodity, using a formula
    proposed by John Bollinger in the 1980s. The bands consist of:
    - Middle Band: Simple Moving Average (typically 20-period)
    - Upper Band: Middle Band + (Standard Deviation × 2)
    - Lower Band: Middle Band - (Standard Deviation × 2)

    They are used to identify overbought and oversold conditions and to measure
    market volatility.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the canonical Bollinger Bands configuration."""
        return IndicatorConfig(
            indicator_name="bollinger_bands",
            factory_name="bollinger_bands",
            indicator_type="volatility",
            period=20,
            standard_deviations=2.0,
            ma_type="simple",
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands values (upper, middle, lower bands).

        The Bollinger Bands calculation involves:
        1. Calculate middle band: Simple Moving Average
        2. Calculate standard deviation over the same period
        3. Upper band = middle band + (standard deviation × multiplier)
        4. Lower band = middle band - (standard deviation × multiplier)

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with Bollinger Bands values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Get the price series to calculate Bollinger Bands on
        price_series = data[self.config.price_column]

        # Calculate middle band (SMA)
        if self.config.ma_type == "simple":
            middle_band = price_series.rolling(window=self.config.period).mean()
        elif self.config.ma_type == "exponential":
            middle_band = price_series.ewm(span=self.config.period, adjust=False).mean()
        else:
            # Default to SMA for unsupported types
            middle_band = price_series.rolling(window=self.config.period).mean()

        # Calculate standard deviation
        std_dev = price_series.rolling(window=self.config.period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * self.config.standard_deviations)
        lower_band = middle_band - (std_dev * self.config.standard_deviations)

        # Add to result DataFrame
        base_name = self.name.lower()
        upper_column = f"{base_name}_upper"
        middle_column = f"{base_name}_middle"
        lower_column = f"{base_name}_lower"

        result[upper_column] = upper_band
        result[middle_column] = middle_band
        result[lower_column] = lower_band

        self.logger.debug(
            f"Calculated Bollinger Bands with period {self.config.period}, std_dev {self.config.standard_deviations}, MA type {self.config.ma_type}"
        )
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on Bollinger Bands price interactions.

        This implementation generates signals for:
        1. Price touching upper/lower bands
        2. Band width expansion/contraction
        3. Squeeze patterns (low volatility)
        4. Breakout signals

        Args:
            data: DataFrame with market data and calculated Bollinger Bands values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if Bollinger Bands were calculated
        base_name = self.name.lower()
        upper_column = f"{base_name}_upper"
        middle_column = f"{base_name}_middle"
        lower_column = f"{base_name}_lower"

        if not all(col in data.columns for col in [upper_column, middle_column, lower_column]):
            self.logger.warning("Bollinger Bands columns not found in data")
            return []

        # Get Bollinger Bands data, removing NaN values
        price_data = data[self.config.price_column].dropna()
        upper_data = data[upper_column].dropna()
        middle_data = data[middle_column].dropna()
        lower_data = data[lower_column].dropna()

        if len(upper_data) < 2:
            return []

        signals = []
        current_price = price_data.iloc[-1]
        current_upper = upper_data.iloc[-1]
        current_middle = middle_data.iloc[-1]
        current_lower = lower_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Generate different types of signals
        signals.extend(
            self._generate_band_touch_signals(
                current_price, current_upper, current_lower, current_timestamp
            )
        )
        signals.extend(
            self._generate_band_width_signals(
                upper_data, lower_data, current_upper, current_lower, current_timestamp
            )
        )
        signals.extend(
            self._generate_trend_signals(current_price, current_middle, current_timestamp)
        )
        signals.extend(
            self._generate_squeeze_signals(
                upper_data, lower_data, current_upper, current_lower, current_timestamp
            )
        )
        signals.extend(
            self._generate_breakout_signals(
                price_data,
                current_middle,
                current_upper,
                current_lower,
                current_price,
                current_timestamp,
            )
        )
        signals.extend(
            self._generate_position_signals(
                current_price, current_upper, current_lower, current_timestamp
            )
        )

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_band_touch_signals(
        self, current_price: float, current_upper: float, current_lower: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate price interaction with bands signals."""
        signals = []
        if current_price >= current_upper:
            # Price touches or exceeds upper band (overbought)
            touch_action = f"Price touches upper band: ${current_price:.2f} >= ${current_upper:.2f}"
            signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=touch_action,
                confidence=0.7,
                timestamp=timestamp,
                indicator_value=current_price,
                band_value=current_upper,
                interaction_type="upper_band_touch",
                band_position="upper",
            )
            signals.append(signal)
        elif current_price <= current_lower:
            # Price touches or goes below lower band (oversold)
            touch_action = f"Price touches lower band: ${current_price:.2f} <= ${current_lower:.2f}"
            signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=touch_action,
                confidence=0.7,
                timestamp=timestamp,
                indicator_value=current_price,
                band_value=current_lower,
                interaction_type="lower_band_touch",
                band_position="lower",
            )
            signals.append(signal)

        return signals

    def _generate_band_width_signals(
        self,
        upper_data: pd.Series,
        lower_data: pd.Series,
        current_upper: float,
        current_lower: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate band width analysis signals."""
        signals = []
        if len(upper_data) > 1:
            band_width = current_upper - current_lower
            prev_upper = upper_data.iloc[-2]
            prev_lower = lower_data.iloc[-2]
            prev_band_width = prev_upper - prev_lower

            if band_width > prev_band_width * 1.1:  # 10% increase in band width
                expansion_action = f"Band expansion: width {band_width:.2f} (volatility increasing)"
                expansion_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=expansion_action,
                    confidence=0.6,
                    timestamp=timestamp,
                    indicator_value=band_width,
                    previous_band_width=prev_band_width,
                    volatility_change="expanding",
                )
                signals.append(expansion_signal)
            elif band_width < prev_band_width * 0.9:  # 10% decrease in band width
                contraction_action = (
                    f"Band contraction: width {band_width:.2f} (volatility decreasing)"
                )
                contraction_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=contraction_action,
                    confidence=0.6,
                    timestamp=timestamp,
                    indicator_value=band_width,
                    previous_band_width=prev_band_width,
                    volatility_change="contracting",
                )
                signals.append(contraction_signal)

        return signals

    def _generate_trend_signals(
        self, current_price: float, current_middle: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate trend signals based on price position relative to middle band."""
        signals = []
        if current_price > current_middle:
            trend_action = f"Above middle band: ${current_price:.2f} > ${current_middle:.2f}"
            trend_confidence = min(0.6, (current_price - current_middle) / current_middle)
            trend_signal_type = SignalType.BUY
        else:
            trend_action = f"Below middle band: ${current_price:.2f} < ${current_middle:.2f}"
            trend_confidence = min(0.6, (current_middle - current_price) / current_middle)
            trend_signal_type = SignalType.SELL

        trend_signal = self._create_standard_signal(
            signal_type=trend_signal_type,
            action=trend_action,
            confidence=trend_confidence,
            timestamp=timestamp,
            indicator_value=current_price,
            band_value=current_middle,
            distance_from_middle=current_price - current_middle,
            band_position="above_middle" if current_price > current_middle else "below_middle",
        )
        signals.append(trend_signal)
        return signals

    def _generate_squeeze_signals(
        self,
        upper_data: pd.Series,
        lower_data: pd.Series,
        current_upper: float,
        current_lower: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate squeeze detection signals."""
        signals = []
        if len(upper_data) >= 10:
            recent_widths = upper_data.tail(10) - lower_data.tail(10)
            avg_width = recent_widths.mean()
            current_width = current_upper - current_lower

            if current_width < avg_width * 0.8:  # 20% below average
                squeeze_action = f"Volatility squeeze detected: width {current_width:.2f} below average {avg_width:.2f}"
                squeeze_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=squeeze_action,
                    confidence=0.5,
                    timestamp=timestamp,
                    indicator_value=current_width,
                    average_width=avg_width,
                    squeeze_condition=True,
                )
                signals.append(squeeze_signal)

        return signals

    def _generate_breakout_signals(
        self,
        price_data: pd.Series,
        current_middle: float,
        current_upper: float,
        current_lower: float,
        current_price: float,
        timestamp: Any,
    ) -> list[dict[str, Any]]:
        """Generate breakout signals."""
        signals = []
        if len(price_data) > 1:
            prev_price = price_data.iloc[-2]

            # Upward breakout
            if prev_price <= current_middle and current_price > current_upper:
                breakout_action = f"Upward breakout: ${current_price:.2f} breaks above upper band ${current_upper:.2f}"
                breakout_signal = self._create_standard_signal(
                    signal_type=SignalType.BUY,
                    action=breakout_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_price,
                    breakout_level=current_upper,
                    breakout_type="upward",
                )
                signals.append(breakout_signal)

            # Downward breakout
            elif prev_price >= current_middle and current_price < current_lower:
                breakout_action = f"Downward breakout: ${current_price:.2f} breaks below lower band ${current_lower:.2f}"
                breakout_signal = self._create_standard_signal(
                    signal_type=SignalType.SELL,
                    action=breakout_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_price,
                    breakout_level=current_lower,
                    breakout_type="downward",
                )
                signals.append(breakout_signal)

        return signals

    def _generate_position_signals(
        self, current_price: float, current_upper: float, current_lower: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate band position analysis signals."""
        signals = []
        price_position = (current_price - current_lower) / (current_upper - current_lower)
        if price_position > 0.9:  # Near upper band
            position_action = f"Price near upper band: {price_position:.1%} of band range"
            position_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,
                action=position_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_price,
                price_position=price_position,
                band_position="near_upper",
            )
            signals.append(position_signal)
        elif price_position < 0.1:  # Near lower band
            position_action = f"Price near lower band: {price_position:.1%} of band range"
            position_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,
                action=position_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_price,
                price_position=price_position,
                band_position="near_lower",
            )
            signals.append(position_signal)

        return signals

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        base_name = self.name.lower()
        return [f"{base_name}_upper", f"{base_name}_middle", f"{base_name}_lower"]

    def calculate_band_width(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the width of the Bollinger Bands.

        Args:
            data: DataFrame with calculated Bollinger Bands

        Returns:
            Series containing band widths
        """
        base_name = self.name.lower()
        upper_column = f"{base_name}_upper"
        lower_column = f"{base_name}_lower"

        if upper_column in data.columns and lower_column in data.columns:
            return data[upper_column] - data[lower_column]
        else:
            return pd.Series(dtype=float)

    def calculate_percent_b(self, data: pd.DataFrame) -> pd.Series:
        """Calculate %B indicator (price position within bands).

        %B = (Price - Lower Band) / (Upper Band - Lower Band)

        Args:
            data: DataFrame with calculated Bollinger Bands

        Returns:
            Series containing %B values
        """
        base_name = self.name.lower()
        upper_column = f"{base_name}_upper"
        lower_column = f"{base_name}_lower"
        price_column = self.config.price_column

        if all(col in data.columns for col in [upper_column, lower_column, price_column]):
            return (data[price_column] - data[lower_column]) / (
                data[upper_column] - data[lower_column]
            )
        else:
            return pd.Series(dtype=float)
