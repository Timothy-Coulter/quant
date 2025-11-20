"""Momentum based signal strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base_signal_strategy import BaseSignalStrategy
from .signal_strategy_config import MomentumStrategyConfig


class MomentumStrategy(BaseSignalStrategy):
    """Generate signals using simple momentum style heuristics."""

    def __init__(self, config: MomentumStrategyConfig, event_bus: Any) -> None:
        """Store configuration and initialise shared base state."""
        super().__init__(config, event_bus)
        self.config = config
        self.name = config.name

        self.momentum_periods = sorted(config.momentum_periods)
        self.momentum_weighting = config.momentum_weighting
        self.momentum_threshold = config.momentum_threshold
        self.trend_confirmation = config.trend_confirmation
        self.volatility_filter = config.volatility_filter
        self.min_volume_threshold = config.min_volume_threshold

        self.use_breakout_signals = config.use_breakout_signals
        self.use_divergence_signals = config.use_divergence_signals
        self.use_trend_following = config.use_trend_following

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def get_required_columns(self) -> list[str]:
        """Return the columns required for momentum calculations."""
        return [
            'open',
            'high',
            'low',
            'close',
            'volume',
            *(f'momentum_{period}' for period in self.momentum_periods),
            'momentum_score',
            'trend_strength',
            'volatility',
        ]

    def validate_signal_data(self, data: pd.DataFrame) -> bool:
        """Check that input data contains core columns and enough history."""
        if data is None or data.empty:
            return False
        base_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not base_columns.issubset(data.columns):
            return False
        return len(data) >= max(self.momentum_periods) + 3

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        indicators: dict[str, pd.Series] = {}
        for period in self.momentum_periods:
            series = data['close'].pct_change(period).fillna(0.0)
            indicators[f'momentum_{period}'] = series
        return indicators

    def _weight_momentum_periods(self, indicators: dict[str, pd.Series], method: str) -> pd.Series:
        if not indicators:
            return pd.Series(0.0, index=pd.RangeIndex(0))

        df = pd.DataFrame(indicators).fillna(0.0)
        n_periods = df.shape[1]

        if method == 'linear':
            weights = np.linspace(1, n_periods, n_periods)
        elif method == 'exponential':
            weights = np.array([0.5 ** (n_periods - i - 1) for i in range(n_periods)])
        else:  # equal weighting and fallback
            weights = np.ones(n_periods)

        weights = weights / weights.sum()
        weighted = df.dot(weights)

        # Normalize to 0-1 range for easier interpretation
        min_val, max_val = weighted.min(), weighted.max()
        if max_val - min_val > 1e-12:
            weighted = (weighted - min_val) / (max_val - min_val)
        else:
            weighted = pd.Series(0.0, index=df.index)
        return weighted

    def _calculate_momentum_score(self, indicators: dict[str, pd.Series]) -> pd.Series:
        if not indicators:
            return pd.Series(dtype=float)
        return self._weight_momentum_periods(indicators, self.momentum_weighting)

    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        trend = close.ewm(span=20, adjust=False).mean()
        strength = (close - trend).abs()
        if strength.max() == 0:
            return pd.Series(0.0, index=data.index)
        return (strength / strength.max()).fillna(0.0)

    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change().fillna(0.0)
        vol = returns.rolling(window=20).std().fillna(0.0)
        return vol.replace([np.inf, -np.inf], 0.0)

    def _apply_momentum_threshold_filter(
        self,
        signals: list[dict[str, Any]],
        momentum_score: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []
        filtered: list[dict[str, Any]] = []
        relevant_index = index[-len(signals) :]
        for signal, ts in zip(signals, relevant_index, strict=False):
            score_value = float(momentum_score.reindex(index, method='nearest').get(ts, 0.0))
            if signal.get('confidence', 0.0) >= self.momentum_threshold:
                signal.setdefault('metadata', {})
                signal['metadata']['momentum_score'] = score_value
                filtered.append(signal)
        return filtered

    def _apply_trend_confirmation(
        self,
        signals: list[dict[str, Any]],
        momentum_score: pd.Series,
        trend_strength: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        confirmed: list[dict[str, Any]] = []
        relevant_index = index[-len(signals) :]
        for signal, ts in zip(signals, relevant_index, strict=False):
            strength = float(trend_strength.reindex(index, method='nearest').get(ts, 0.0))
            if strength >= 0.2:
                signal['confidence'] = float(
                    np.clip(signal['confidence'] + strength * 0.2, 0.0, 1.0)
                )
                confirmed.append(signal)
        return confirmed

    def _apply_volatility_filter(
        self,
        signals: list[dict[str, Any]],
        volatility: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []
        filtered: list[dict[str, Any]] = []
        relevant_index = index[-len(signals) :]
        for signal, ts in zip(signals, relevant_index, strict=False):
            vol_value = float(volatility.reindex(index, method='nearest').get(ts, 0.0))
            if vol_value < 0.05 or signal['signal_type'] == 'HOLD':
                filtered.append(signal)
        return filtered

    def _apply_volume_filter(
        self,
        signals: list[dict[str, Any]],
        volume: pd.Series,
        index: pd.Index,
    ) -> list[dict[str, Any]]:
        if not signals:
            return []
        filtered: list[dict[str, Any]] = []
        avg_volume = volume.rolling(window=20).mean().bfill()
        relevant_index = index[-len(signals) :]
        for signal, ts in zip(signals, relevant_index, strict=False):
            current_volume = float(volume.loc[ts])
            threshold = float(avg_volume.loc[ts]) if ts in avg_volume else self.min_volume_threshold
            if current_volume >= max(self.min_volume_threshold, threshold):
                filtered.append(signal)
        return filtered

    def _generate_momentum_signals(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        indicators = self._calculate_momentum_indicators(data)
        momentum_score = self._calculate_momentum_score(indicators)
        if momentum_score.empty:
            return []

        latest_score = float(momentum_score.iloc[-1])
        primary_period = self.momentum_periods[0]
        primary_momentum = indicators[f'momentum_{primary_period}'].iloc[-1]

        if latest_score >= self.momentum_threshold and primary_momentum >= 0:
            signal_type = 'BUY'
        elif latest_score >= self.momentum_threshold and primary_momentum < 0:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'

        confidence = float(np.clip(latest_score, 0.0, 1.0))
        signal = {
            'signal_type': signal_type,
            'confidence': confidence,
            'action': 'Momentum signal',
            'metadata': {
                'timestamp': data.index[-1],
                'primary_momentum': float(primary_momentum),
            },
        }
        return [signal]

    # ------------------------------------------------------------------#
    # Strategy entry point
    # ------------------------------------------------------------------#
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        """Generate momentum-based signals for the supplied symbol."""
        if not self.validate_signal_data(data):
            return []

        indicators = self._calculate_momentum_indicators(data)
        momentum_score = self._calculate_momentum_score(indicators)
        trend_strength = self._calculate_trend_strength(data)
        volatility = self._calculate_volatility(data)

        signals = self._generate_momentum_signals(data)
        signals = self._apply_momentum_threshold_filter(signals, momentum_score, data.index)

        if self.trend_confirmation:
            signals = self._apply_trend_confirmation(
                signals, momentum_score, trend_strength, data.index
            )
        if self.volatility_filter:
            signals = self._apply_volatility_filter(signals, volatility, data.index)

        signals = self._apply_volume_filter(signals, data['volume'], data.index)

        for signal in signals:
            self.signals.append(signal)
            self.signal_history.append(signal)
            self.signal_count += 1
            self.valid_signal_count += 1
            self.last_signal_time = data.index[-1]

        return signals

    def get_strategy_info(self) -> dict[str, Any]:
        """Expose strategy configuration for reporting."""
        info = super().get_strategy_info()
        info.update(
            {
                'momentum_periods': self.momentum_periods,
                'momentum_weighting': self.momentum_weighting,
                'momentum_threshold': self.momentum_threshold,
                'trend_confirmation': self.trend_confirmation,
                'volatility_filter': self.volatility_filter,
                'min_volume_threshold': self.min_volume_threshold,
            }
        )
        return info
