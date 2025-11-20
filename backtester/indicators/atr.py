"""Average True Range (ATR) indicator implementation.

This module provides the Average True Range volatility indicator implementation,
following the established patterns from the base indicator class.
"""

from typing import Any

import pandas as pd

from backtester.signal.signal_types import SignalType

from .base_indicator import BaseIndicator, IndicatorFactory
from .indicator_configs import IndicatorConfig


@IndicatorFactory.register("atr")
class ATRIndicator(BaseIndicator):
    """Average True Range (ATR) volatility indicator.

    The Average True Range (ATR) is a market volatility indicator. The average
    true range is an N-period smoothed moving average of the true range values.
    The true range is the greatest of:
    1. Current high minus current low
    2. Absolute value of current high minus previous close
    3. Absolute value of current low minus previous close

    ATR measures market volatility and is used to determine optimal position
    sizing and stop-loss levels.
    """

    @classmethod
    def default_config(cls) -> IndicatorConfig:
        """Return the reference configuration for the ATR indicator."""
        return IndicatorConfig(
            indicator_name="atr",
            factory_name="atr",
            indicator_type="volatility",
            period=14,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR) values.

        The ATR calculation involves:
        1. Calculate True Range: max(high-low, |high-prev_close|, |low-prev_close|)
        2. Calculate ATR as exponential moving average of True Range

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            DataFrame with ATR values added as new columns

        Raises:
            ValueError: If data format is invalid or insufficient
            KeyError: If required data columns are missing
        """
        self.validate_data(data)
        result = data.copy()

        # Calculate True Range for each period
        tr1 = data['high'] - data['low']  # High - Low
        tr2 = abs(data['high'] - data['close'].shift(1))  # |High - Previous Close|
        tr3 = abs(data['low'] - data['close'].shift(1))  # |Low - Previous Close|

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR using exponential moving average
        atr_column = f"{self.name.lower()}_atr"
        result[atr_column] = true_range.ewm(span=self.config.period, adjust=False).mean()

        self.logger.debug(f"Calculated ATR with period {self.config.period}")
        return result

    def generate_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate trading signals based on ATR volatility changes.

        This implementation generates signals for:
        1. High volatility periods
        2. Low volatility periods
        3. ATR trend changes
        4. Volatility breakouts
        5. Risk assessment signals

        Args:
            data: DataFrame with market data and calculated ATR values

        Returns:
            List of signal dictionaries with required fields:
            - 'signal_type': str ('BUY', 'SELL', 'HOLD')
            - 'action': str (detailed action description)
            - 'confidence': float (0.0 to 1.0)
            - 'metadata': dict (additional signal information)

        Raises:
            ValueError: If data is missing required columns
        """
        # Check if ATR was calculated
        atr_column = f"{self.name.lower()}_atr"
        if atr_column not in data.columns:
            self.logger.warning(f"ATR column {atr_column} not found in data")
            return []

        # Get ATR data, removing NaN values
        atr_data = data[atr_column].dropna()

        if len(atr_data) < 2:
            return []

        signals = []
        current_atr = atr_data.iloc[-1]
        current_timestamp = data.index[-1]
        current_price = data['close'].iloc[-1]

        # Generate different types of signals
        signals.extend(
            self._generate_volatility_signals(current_atr, current_price, current_timestamp)
        )
        signals.extend(self._generate_trend_signals(atr_data, current_atr, current_timestamp))
        signals.extend(self._generate_breakout_signals(atr_data, current_atr, current_timestamp))
        signals.extend(
            self._generate_risk_signals(current_atr, current_price, current_timestamp, atr_data)
        )
        signals.extend(
            self._generate_position_sizing_signals(
                current_atr, current_price, current_timestamp, atr_data
            )
        )
        signals.extend(
            self._generate_stop_loss_signals(current_atr, current_price, current_timestamp)
        )

        self.logger.debug(f"Generated {len(signals)} signals for {self.name}")
        return signals

    def _generate_volatility_signals(
        self, current_atr: float, current_price: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate volatility level signals."""
        signals = []
        relative_atr = (current_atr / current_price) * 100

        if relative_atr > 3.0:  # High volatility (> 3%)
            high_vol_action = (
                f"High volatility: ATR {current_atr:.2f} ({relative_atr:.1f}% of price)"
            )
            high_vol_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=high_vol_action,
                confidence=0.7,
                timestamp=timestamp,
                indicator_value=current_atr,
                relative_atr=relative_atr,
                volatility_level="high",
            )
            signals.append(high_vol_signal)
        elif relative_atr < 0.5:  # Low volatility (< 0.5%)
            low_vol_action = f"Low volatility: ATR {current_atr:.2f} ({relative_atr:.1f}% of price)"
            low_vol_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=low_vol_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_atr,
                relative_atr=relative_atr,
                volatility_level="low",
            )
            signals.append(low_vol_signal)

        return signals

    def _generate_trend_signals(
        self, atr_data: pd.Series, current_atr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate ATR trend analysis signals."""
        signals = []
        if len(atr_data) > 1:
            prev_atr = atr_data.iloc[-2]
            atr_change = current_atr - prev_atr
            atr_change_pct = (atr_change / prev_atr) * 100 if prev_atr != 0 else 0

            if atr_change > 0:
                trend_action = f"ATR increasing: {current_atr:.2f} (+{atr_change_pct:.1f}%)"
                trend_confidence = min(0.6, abs(atr_change_pct) / 10)
            else:
                trend_action = f"ATR decreasing: {current_atr:.2f} ({atr_change_pct:.1f}%)"
                trend_confidence = min(0.6, abs(atr_change_pct) / 10)

            trend_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=trend_action,
                confidence=trend_confidence,
                timestamp=timestamp,
                indicator_value=current_atr,
                atr_change=atr_change,
                atr_change_percent=atr_change_pct,
                trend_direction="increasing" if atr_change > 0 else "decreasing",
            )
            signals.append(trend_signal)

        return signals

    def _generate_breakout_signals(
        self, atr_data: pd.Series, current_atr: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate volatility breakout signals."""
        signals = []
        if len(atr_data) >= 10:
            recent_atr = atr_data.tail(10)
            avg_atr = recent_atr.mean()
            atr_std = recent_atr.std()

            # High volatility breakout
            if current_atr > avg_atr + (2 * atr_std):
                breakout_action = f"Volatility breakout: ATR {current_atr:.2f} above 2σ threshold"
                breakout_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=breakout_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_atr,
                    average_atr=avg_atr,
                    volatility_breakout=True,
                    breakout_direction="high",
                )
                signals.append(breakout_signal)

            # Low volatility breakout
            elif current_atr < avg_atr - (2 * atr_std):
                breakout_action = f"Low volatility: ATR {current_atr:.2f} below 2σ threshold"
                breakout_signal = self._create_standard_signal(
                    signal_type=SignalType.HOLD,
                    action=breakout_action,
                    confidence=0.8,
                    timestamp=timestamp,
                    indicator_value=current_atr,
                    average_atr=avg_atr,
                    volatility_breakout=True,
                    breakout_direction="low",
                )
                signals.append(breakout_signal)

        return signals

    def _generate_risk_signals(
        self, current_atr: float, current_price: float, timestamp: Any, atr_data: pd.Series
    ) -> list[dict[str, Any]]:
        """Generate risk management signals."""
        signals = []
        if current_atr > current_price * 0.05:  # ATR > 5% of price
            risk_action = f"High risk: ATR {current_atr:.2f} suggests wider stops"
            risk_signal = self._create_standard_signal(
                signal_type=SignalType.HOLD,
                action=risk_action,
                confidence=0.6,
                timestamp=timestamp,
                indicator_value=current_atr,
                current_price=current_price,
                risk_level="high",
                suggested_stop_loss=current_atr * self.config.atr_multiplier,
            )
            signals.append(risk_signal)

        return signals

    def _generate_position_sizing_signals(
        self, current_atr: float, current_price: float, timestamp: Any, atr_data: pd.Series
    ) -> list[dict[str, Any]]:
        """Generate position sizing signals."""
        signals = []
        relative_atr = (current_atr / current_price) * 100
        position_size_factor = min(2.0, max(0.5, 1.0 / relative_atr)) if relative_atr > 0 else 1.0

        if position_size_factor < 0.8:  # Reduce position size
            sizing_action = (
                f"Reduce position size: high volatility (factor {position_size_factor:.2f})"
            )
            sizing_signal = self._create_standard_signal(
                signal_type=SignalType.SELL,  # Reduce exposure
                action=sizing_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_atr,
                position_size_factor=position_size_factor,
                relative_atr=relative_atr,
            )
            signals.append(sizing_signal)
        elif position_size_factor > 1.5:  # Increase position size
            sizing_action = (
                f"Increase position size: low volatility (factor {position_size_factor:.2f})"
            )
            sizing_signal = self._create_standard_signal(
                signal_type=SignalType.BUY,  # Increase exposure
                action=sizing_action,
                confidence=0.5,
                timestamp=timestamp,
                indicator_value=current_atr,
                position_size_factor=position_size_factor,
                relative_atr=relative_atr,
            )
            signals.append(sizing_signal)

        return signals

    def _generate_stop_loss_signals(
        self, current_atr: float, current_price: float, timestamp: Any
    ) -> list[dict[str, Any]]:
        """Generate ATR-based stop loss signals."""
        signals = []
        stop_loss_level = current_price - (current_atr * self.config.atr_multiplier)
        stop_loss_action = (
            f"ATR-based stop loss: ${stop_loss_level:.2f} ({self.config.atr_multiplier}x ATR)"
        )
        stop_loss_signal = self._create_standard_signal(
            signal_type=SignalType.HOLD,
            action=stop_loss_action,
            confidence=0.4,
            timestamp=timestamp,
            indicator_value=current_atr,
            suggested_stop_loss=stop_loss_level,
            current_price=current_price,
            atr_multiplier=self.config.atr_multiplier,
        )
        signals.append(stop_loss_signal)

        return signals

    def get_indicator_columns(self) -> list[str]:
        """Get the column names that this indicator adds to the DataFrame.

        Returns:
            List of column names that will be added
        """
        return [f"{self.name.lower()}_atr"]

    def calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range values.

        Args:
            data: Market data in OLHV format with datetime index

        Returns:
            Series containing True Range values
        """
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def get_volatility_percentile(self, data: pd.DataFrame, lookback: int = 50) -> float:
        """Get the percentile of current ATR relative to historical values.

        Args:
            data: DataFrame with calculated ATR
            lookback: Number of periods to look back

        Returns:
            Percentile value (0-100)
        """
        atr_column = f"{self.name.lower()}_atr"
        if atr_column not in data.columns:
            return 50.0

        atr_data = data[atr_column].dropna()
        if len(atr_data) < lookback:
            return 50.0

        current_atr = atr_data.iloc[-1]
        historical_atr = atr_data.tail(lookback)

        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        return float(percentile)
