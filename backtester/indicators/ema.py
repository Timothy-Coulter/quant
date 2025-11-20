"""Exponential Moving Average (EMA) indicator implementation.

This module provides the Exponential Moving Average indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("ema")
class EMAIndicator(BaseIndicator):
    """Exponential Moving Average trend indicator.

    The Exponential Moving Average is a weighted moving average that gives
    more importance to recent price data. It responds more quickly to price
    changes than the Simple Moving Average, making it more sensitive to
    short-term price movements and trend changes.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the canonical EMA configuration."""
        return IndicatorConfig(
            indicator_name="ema",
            factory_name="ema",
            indicator_type="trend",
            period=12,
            ma_type="exponential",
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Average values.

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with EMA values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Get the price series to calculate EMA on
        price_series = data[self.config.price_column]

        # Calculate EMA using pandas ewm (exponentially weighted moving) function
        ema_column = f"{self.name.lower()}_ema"
        result[ema_column] = price_series.ewm(span=self.config.period, adjust=False).mean()

        self.logger.debug(f"Calculated EMA with period {self.config.period}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on EMA crossovers.

        This implementation generates signals when price crosses above or below
        the EMA, indicating potential trend changes. EMA signals are typically
        more responsive than SMA signals.

        Args:
            data: DataFrame with market data and calculated EMA values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if EMA was calculated
        ema_column = f"{self.name.lower()}_ema"
        if ema_column not in data.columns:
            self.logger.warning(f"EMA column {ema_column} not found in data")
            return []

        # Get price and EMA data, ensuring they have the same index
        price_data = data[self.config.price_column]
        ema_data = data[ema_column]

        # Create aligned DataFrame to handle NaN values properly
        aligned_data = pd.DataFrame({'price': price_data, 'ema': ema_data}).dropna()

        if len(aligned_data) < 2:
            return []

        signals = []
        price_series = aligned_data['price']
        ema_series = aligned_data['ema']

        # Find crossover points using proper boolean logic
        price_above_ema_prev = price_series.shift(1) > ema_series.shift(1)
        price_above_ema_curr = price_series > ema_series

        # Identify crossovers
        crossover_up = (~price_above_ema_prev) & price_above_ema_curr
        crossover_down = price_above_ema_prev & (~price_above_ema_curr)

        # Generate current trend signal
        current_price = price_data.iloc[-1]
        current_ema = ema_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Price-EMA relationship for current signal
        if current_price > current_ema:
            primary_signal = SignalType.BUY
            action = f"Bullish trend: Price ${current_price:.2f} above EMA ${current_ema:.2f}"
        else:
            primary_signal = SignalType.SELL
            action = f"Bearish trend: Price ${current_price:.2f} below EMA ${current_ema:.2f}"

        # Calculate confidence based on distance from EMA and EMA momentum
        distance_ratio = abs(current_price - current_ema) / current_ema
        confidence = min(0.9, max(0.3, distance_ratio * 10))

        # Add EMA momentum information
        if len(ema_data) > 1:
            ema_change = current_ema - ema_data.iloc[-2]
            ema_momentum = "increasing" if ema_change > 0 else "decreasing"
            momentum_strength = abs(ema_change) / ema_data.iloc[-2] if ema_data.iloc[-2] != 0 else 0
        else:
            ema_momentum = "neutral"
            momentum_strength = 0

        # Adjust confidence based on EMA momentum alignment
        if (primary_signal == SignalType.BUY and ema_momentum == "increasing") or (
            primary_signal == SignalType.SELL and ema_momentum == "decreasing"
        ):
            confidence = min(0.95, confidence * 1.2)  # Boost confidence for aligned signals
        else:
            confidence = max(0.2, confidence * 0.8)  # Reduce confidence for conflicting signals

        # Create the main signal
        signal = self._create_standard_signal(
            signal_type=primary_signal,
            action=action,
            confidence=confidence,
            timestamp=current_timestamp,
            indicator_value=current_ema,
            price_value=current_price,
            price_ema_distance=current_price - current_ema,
            price_ema_ratio=current_price / current_ema,
            ema_momentum=ema_momentum,
            momentum_strength=momentum_strength,
            crossover_signals={
                'crossover_up_detected': crossover_up.iloc[-1] if not crossover_up.empty else False,
                'crossover_down_detected': (
                    crossover_down.iloc[-1] if not crossover_down.empty else False
                ),
            },
        )

        signals.append(signal)

        # Generate additional signals for recent crossovers (last 5 periods)
        recent_crossovers = crossover_up.tail(5) | crossover_down.tail(5)
        if recent_crossovers.any():
            last_crossover_idx = recent_crossovers[recent_crossovers].index[-1]
            last_crossover_type = "UP" if crossover_up.loc[last_crossover_idx] else "DOWN"

            crossover_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=f"Recent {last_crossover_type} crossover detected at {last_crossover_idx}",
                confidence=0.6,
                timestamp=current_timestamp,
                indicator_value=current_ema,
                price_value=current_price,
                crossover_type=last_crossover_type,
                crossover_timestamp=last_crossover_idx,
            )
            signals.append(crossover_signal)

        # Generate momentum signal if EMA momentum is strong
        if momentum_strength > 0.005:  # 0.5% change threshold
            momentum_action = f"Strong {ema_momentum} momentum detected"
            momentum_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=momentum_action,
                confidence=min(0.8, momentum_strength * 100),
                timestamp=current_timestamp,
                indicator_value=current_ema,
                momentum_strength=momentum_strength,
                ema_momentum=ema_momentum,
            )
            signals.append(momentum_signal)

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_ema"]

    def calculate_ema_slope(self, data: pd.DataFrame, periods: int = 5) -> float:
        """Calculate the slope of the EMA over a specified number of periods.

        Args:
            data: DataFrame with calculated EMA values
            periods: Number of periods to calculate slope over

        Returns:
            EMA slope (rate of change per period)
        """
        ema_column = f"{self.name.lower()}_ema"
        if ema_column not in data.columns:
            return 0.0

        ema_data = data[ema_column].dropna()
        if len(ema_data) < periods:
            return 0.0

        recent_ema = ema_data.tail(periods)
        return float((recent_ema.iloc[-1] - recent_ema.iloc[0]) / periods)
