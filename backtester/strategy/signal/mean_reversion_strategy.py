"""Mean reversion strategy for signal generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base_signal_strategy import BaseSignalStrategy
from .signal_strategy_config import MeanReversionStrategyConfig, SignalStrategyConfig


class MeanReversionStrategy(BaseSignalStrategy):
    """Generate trading signals based on simple mean reversion heuristics."""

    def __init__(
        self,
        config: MeanReversionStrategyConfig | SignalStrategyConfig,
        event_bus: Any,
    ) -> None:
        """Prepare configuration parameters and delegate to the base strategy."""
        if isinstance(config, SignalStrategyConfig):
            strategy_config = config.strategy_config
            if not isinstance(strategy_config, MeanReversionStrategyConfig):
                raise TypeError(
                    "SignalStrategyConfig passed to MeanReversionStrategy must wrap "
                    "MeanReversionStrategyConfig."
                )
            inner_config = strategy_config
            outer_config: SignalStrategyConfig | None = config
        else:
            inner_config = config
            outer_config = None

        super().__init__(inner_config, event_bus)
        self.config = outer_config or inner_config
        self.strategy_config: MeanReversionStrategyConfig = inner_config
        self.name = inner_config.name

        self.mean_periods: list[int] = sorted(list(inner_config.mean_periods))
        self.std_dev_periods: list[int] = sorted(list(inner_config.std_dev_periods))
        self.min_reversion_strength = inner_config.min_reversion_strength
        self.hurst_threshold = inner_config.hurst_threshold
        self.volatility_adjustment = inner_config.volatility_adjustment
        self.regime_filter = inner_config.regime_filter
        self.correlation_threshold = inner_config.correlation_threshold
        self.volume_confirmation = inner_config.volume_confirmation
        self.min_volume_multiplier = getattr(inner_config, 'min_volume_multiplier', 1.2)
        self.z_score_threshold = inner_config.z_score_threshold

    # ------------------------------------------------------------------#
    # Required columns and validation helpers
    # ------------------------------------------------------------------#
    def get_required_columns(self) -> list[str]:
        """Return the feature columns required to evaluate the strategy."""
        return [
            'open',
            'high',
            'low',
            'close',
            'volume',
            'mean_20',
            'mean_50',
            'std_dev_2',
            'std_dev_3',
            'z_score',
            'reversion_strength',
            'hurst_exponent',
            'volatility',
            'regime',
            'correlation',
        ]

    def validate_signal_data(self, data: pd.DataFrame) -> bool:
        """Check that input data is non-empty, complete, and sufficiently long."""
        if data is None or data.empty:
            return False
        base_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not base_columns.issubset(data.columns):
            return False
        return len(data) >= max(self.mean_periods + [20])

    # ------------------------------------------------------------------#
    # Indicator calculations
    # ------------------------------------------------------------------#
    def _calculate_mean_indicators(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        indicators: dict[str, pd.Series] = {}
        if data is None or data.empty or 'close' not in data:
            return indicators
        close = data['close']
        for period in self.mean_periods:
            indicators[f'mean_{period}'] = close.rolling(window=period, min_periods=1).mean()
        self._latest_close = close
        return indicators

    def _calculate_std_dev_indicators(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        indicators: dict[str, pd.Series] = {}
        if data is None or data.empty or 'close' not in data:
            return indicators
        close = data['close']
        for period in self.std_dev_periods:
            indicators[f'std_dev_{period}'] = (
                close.rolling(window=period, min_periods=1).std().fillna(0.0)
            )
        return indicators

    def _calculate_z_score(
        self,
        mean_indicators: dict[str, pd.Series],
        std_indicators: dict[str, pd.Series],
    ) -> pd.Series:
        if not mean_indicators or not std_indicators:
            return pd.Series(dtype=float)
        mean_series = next(iter(mean_indicators.values()))
        std_series = next(iter(std_indicators.values())).replace(0.0, np.nan)
        close = getattr(self, '_latest_close', mean_series)
        z_score = (close - mean_series) / std_series
        return z_score.fillna(0.0)

    def _calculate_reversion_strength(self, z_score: pd.Series) -> pd.Series:
        if z_score.empty:
            return pd.Series(dtype=float)
        strength = z_score.abs()
        max_val = strength.max()
        if max_val and max_val > 0:
            strength = strength / max_val
        return strength.fillna(0.0).clip(0.0, 1.0)

    def _calculate_hurst_exponent(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change().abs().fillna(0.0)
        window = min(max(10, len(returns) // 5), 50)
        hurst = returns.rolling(window=window, min_periods=1).mean()
        return hurst.fillna(0.0).clip(0.0, 1.0)

    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change().fillna(0.0)
        volatility = returns.rolling(window=20, min_periods=1).std().abs()
        return volatility.fillna(0.0)

    def _detect_market_regime(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change().fillna(0.0)
        threshold = returns.rolling(window=20, min_periods=1).std().fillna(0.0)
        regime = pd.Series(0, index=data.index, dtype=int)
        regime[returns > threshold] = 1
        regime[returns < -threshold] = -1
        return regime

    def _calculate_correlation(self, data: pd.DataFrame) -> pd.Series:
        price_returns = data['close'].pct_change().fillna(0.0)
        volume_change = data['volume'].pct_change().fillna(0.0)
        correlation = price_returns.rolling(window=20, min_periods=2).corr(volume_change)
        return correlation.fillna(0.0).clip(-1.0, 1.0)

    # ------------------------------------------------------------------#
    # Signal filters
    # ------------------------------------------------------------------#
    def _apply_reversion_strength_filter(
        self,
        signals: list[dict[str, Any]],
        reversion_strength: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []
        filtered: list[dict[str, Any]] = []
        series = reversion_strength.reindex(index, method='nearest').fillna(0.0)
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            strength = float(series.loc[ts])
            if strength >= self.min_reversion_strength:
                signal.setdefault('metadata', {})['reversion_strength'] = strength
                filtered.append(signal)
        return filtered

    def _apply_hurst_filter(
        self,
        signals: list[dict[str, Any]],
        hurst_exponent: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals or not self.hurst_threshold:
            return signals
        hurst_series = hurst_exponent.reindex(index, method='nearest').fillna(0.5)
        filtered: list[dict[str, Any]] = []
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            if hurst_series.loc[ts] <= self.hurst_threshold:
                filtered.append(signal)
        return filtered

    def _apply_volatility_adjustment(
        self,
        signals: list[dict[str, Any]],
        volatility: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals or not self.volatility_adjustment:
            return signals
        vol_series = volatility.reindex(index, method='nearest').fillna(volatility.mean())
        adjusted: list[dict[str, Any]] = []
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            vol = float(vol_series.loc[ts])
            factor = float(1.0 / (1.0 + vol))
            new_signal = signal.copy()
            new_signal['confidence'] = float(
                np.clip(new_signal.get('confidence', 0.0) * factor, 0.0, 1.0)
            )
            adjusted.append(new_signal)
        return adjusted

    def _apply_regime_filter(
        self,
        signals: list[dict[str, Any]],
        regime: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals or not self.regime_filter:
            return signals
        regime_series = regime.reindex(index, method='nearest').fillna(0)
        filtered: list[dict[str, Any]] = []
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            reg = int(regime_series.loc[ts])
            if (
                signal['signal_type'] == 'BUY'
                and reg <= 0
                or signal['signal_type'] == 'SELL'
                and reg >= 0
                or signal['signal_type'] == 'HOLD'
            ):
                filtered.append(signal)
        return filtered

    def _apply_correlation_filter(
        self,
        signals: list[dict[str, Any]],
        correlation: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals or self.correlation_threshold <= 0:
            return signals
        corr_series = correlation.reindex(index, method='nearest').fillna(0.0)
        filtered: list[dict[str, Any]] = []
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            if abs(float(corr_series.loc[ts])) <= self.correlation_threshold:
                filtered.append(signal)
        return filtered

    def _apply_volume_filter(
        self,
        signals: list[dict[str, Any]],
        volume: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals or not self.volume_confirmation:
            return signals
        avg_volume = (
            volume.rolling(window=20, min_periods=1).mean().reindex(index, method='nearest')
        )
        filtered: list[dict[str, Any]] = []
        for signal, ts in zip(signals, index[-len(signals) :], strict=False):
            threshold = float(avg_volume.loc[ts] * self.min_volume_multiplier)
            if float(volume.loc[ts]) >= threshold:
                filtered.append(signal)
        return filtered

    def _apply_z_score_threshold_filter(
        self,
        signals: list[dict[str, Any]],
        z_score: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []
        return [signal for signal in signals if signal.get('confidence', 0.0) > 0.3]

    # ------------------------------------------------------------------#
    # Signal generation
    # ------------------------------------------------------------------#
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        mean_indicators = self._calculate_mean_indicators(data)
        std_indicators = self._calculate_std_dev_indicators(data)
        z_score = self._calculate_z_score(mean_indicators, std_indicators)
        reversion_strength = self._calculate_reversion_strength(z_score)
        hurst_exponent = self._calculate_hurst_exponent(data['close'])
        volatility = self._calculate_volatility(data)
        regime = self._detect_market_regime(data)
        correlation = self._calculate_correlation(data)

        signals: list[dict[str, Any]] = []
        for timestamp, z in z_score.items():
            if np.isnan(z):
                continue
            if z <= -self.z_score_threshold:
                signal_type, action = 'BUY', 'Enter long'
            elif z >= self.z_score_threshold:
                signal_type, action = 'SELL', 'Exit position'
            else:
                signal_type, action = 'HOLD', 'Hold position'
            signals.append(
                {
                    'timestamp': timestamp,
                    'signal_type': signal_type,
                    'action': action,
                    'confidence': float(abs(z)),
                }
            )

        index = data.index
        signals = self._apply_reversion_strength_filter(signals, reversion_strength, index)
        signals = self._apply_hurst_filter(signals, hurst_exponent, index)
        signals = self._apply_volatility_adjustment(signals, volatility, index)
        signals = self._apply_regime_filter(signals, regime, index)
        signals = self._apply_correlation_filter(signals, correlation, index)
        signals = self._apply_volume_filter(signals, data['volume'], index)
        signals = self._apply_z_score_threshold_filter(signals, z_score, index)
        return signals

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        """Produce mean-reversion driven signals for the supplied symbol."""
        if not self.validate_signal_data(data):
            return []
        signals = self._generate_mean_reversion_signals(data)
        for signal in signals:
            self.signals.append(signal)
            self.signal_history.append(signal)
            self.signal_count += 1
            self.valid_signal_count += 1
            self.last_signal_time = data.index[-1]
        return signals

    # ------------------------------------------------------------------#
    # Strategy information
    # ------------------------------------------------------------------#
    def get_strategy_info(self) -> dict[str, Any]:
        """Return configuration metadata describing this strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                'type': 'MEAN_REVERSION',
                'mean_periods': self.mean_periods,
                'std_dev_periods': self.std_dev_periods,
                'min_reversion_strength': self.min_reversion_strength,
                'hurst_threshold': self.hurst_threshold,
                'volatility_adjustment': self.volatility_adjustment,
                'regime_filter': self.regime_filter,
                'correlation_threshold': self.correlation_threshold,
            }
        )
        return info
