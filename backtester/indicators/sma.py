"""Simple Moving Average (SMA) indicator implementation.

This module provides the Simple Moving Average indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("sma")
class SMAIndicator(BaseIndicator):
    """Simple Moving Average trend indicator.

    The Simple Moving Average is the unweighted mean of the previous n data points.
    It's one of the most basic and widely used technical indicators for identifying
    trend direction and potential support/resistance levels.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the reference SMA configuration."""
        return IndicatorConfig(
            indicator_name="sma",
            factory_name="sma",
            indicator_type="trend",
            period=20,
            ma_type="simple",
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple Moving Average values.

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with SMA values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Get the price series to calculate SMA on
        price_series = data[self.config.price_column]

        # Calculate SMA
        sma_column = f"{self.name.lower()}_sma"
        result[sma_column] = price_series.rolling(window=self.config.period).mean()

        self.logger.debug(f"Calculated SMA with period {self.config.period}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on SMA crossovers.

        This implementation generates signals when price crosses above or below
        the SMA, indicating potential trend changes.

        Args:
            data: DataFrame with market data and calculated SMA values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if SMA was calculated
        sma_column = f"{self.name.lower()}_sma"
        if sma_column not in data.columns:
            self.logger.warning(f"SMA column {sma_column} not found in data")
            return []

        # Get price and SMA data, ensuring they have the same index
        price_data = data[self.config.price_column]
        sma_data = data[sma_column]

        # Create aligned DataFrame to handle NaN values properly
        aligned_data = pd.DataFrame({'price': price_data, 'sma': sma_data}).dropna()

        if len(aligned_data) < 2:
            return []

        signals = []
        price_series = aligned_data['price']
        sma_series = aligned_data['sma']

        # Find crossover points using proper boolean logic
        price_above_sma_prev = price_series.shift(1) > sma_series.shift(1)
        price_above_sma_curr = price_series > sma_series

        # Identify crossovers
        crossover_up = (~price_above_sma_prev) & price_above_sma_curr
        crossover_down = price_above_sma_prev & (~price_above_sma_curr)

        # Generate signals
        current_price = price_data.iloc[-1]
        current_sma = sma_data.iloc[-1]
        current_timestamp = data.index[-1]

        # Price-SMA relationship for current signal
        if current_price > current_sma:
            primary_signal = SignalType.BUY
            action = f"Bullish trend: Price ${current_price:.2f} above SMA ${current_sma:.2f}"
        else:
            primary_signal = SignalType.SELL
            action = f"Bearish trend: Price ${current_price:.2f} below SMA ${current_sma:.2f}"

        # Calculate confidence based on distance from SMA
        distance_ratio = abs(current_price - current_sma) / current_sma
        confidence = min(0.9, max(0.3, distance_ratio * 10))

        # Add trend strength metadata
        sma_slope = (
            (current_sma - sma_data.iloc[-2]) / sma_data.iloc[-2] if len(sma_data) > 1 else 0
        )
        trend_strength = "strong" if abs(sma_slope) > 0.001 else "weak"

        # Create the signal
        signal = self._create_standard_signal(
            signal_type=primary_signal,
            action=action,
            confidence=confidence,
            timestamp=current_timestamp,
            indicator_value=current_sma,
            price_value=current_price,
            price_sma_distance=current_price - current_sma,
            price_sma_ratio=current_price / current_sma,
            sma_slope=sma_slope,
            trend_strength=trend_strength,
            crossover_signals={
                'crossover_up_detected': crossover_up.iloc[-1] if not crossover_up.empty else False,
                'crossover_down_detected': (
                    crossover_down.iloc[-1] if not crossover_down.empty else False
                ),
            },
        )

        signals.append(signal)

        # Check for recent crossover signals (last 5 periods)
        recent_crossovers = crossover_up.tail(5) | crossover_down.tail(5)
        if recent_crossovers.any():
            last_crossover_idx = recent_crossovers[recent_crossovers].index[-1]
            last_crossover_type = "UP" if crossover_up.loc[last_crossover_idx] else "DOWN"

            crossover_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=f"Recent {last_crossover_type} crossover detected at {last_crossover_idx}",
                confidence=0.6,
                timestamp=current_timestamp,
                indicator_value=current_sma,
                price_value=current_price,
                crossover_type=last_crossover_type,
                crossover_timestamp=last_crossover_idx,
            )
            signals.append(crossover_signal)

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_sma"]
