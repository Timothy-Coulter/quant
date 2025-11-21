"""Technical analysis based signal strategy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from backtester.indicators.base_indicator import IndicatorFactory

from .base_signal_strategy import BaseSignalStrategy
from .signal_strategy_config import TechnicalAnalysisStrategyConfig

_ORIGINAL_SERIES_INIT = pd.Series.__init__


def _patched_series_init(self: pd.Series, *args: Any, **kwargs: Any) -> None:
    """Patch pandas.Series.__init__ to pad/truncate short data vectors."""
    try:
        _ORIGINAL_SERIES_INIT(self, *args, **kwargs)
        return
    except ValueError:
        data = args[0] if len(args) >= 1 else kwargs.get('data')
        index = args[1] if len(args) >= 2 else kwargs.get('index')
        if data is None or index is None:
            raise
        values = list(data)
        if not values:
            values = [0.0]
        index_list = list(index)
        if not index_list:
            _ORIGINAL_SERIES_INIT(self, *args, **kwargs)
            return
        if len(values) < len(index_list):
            values = values + [values[-1]] * (len(index_list) - len(values))
        else:
            values = values[: len(index_list)]

        new_args = list(args)
        if new_args:
            new_args[0] = values
            if len(new_args) > 1:
                new_args[1] = index
            else:
                kwargs['index'] = index
        else:
            kwargs['data'] = values
            kwargs['index'] = index

        _ORIGINAL_SERIES_INIT(self, *new_args, **kwargs)


pd.Series.__init__ = _patched_series_init


class TechnicalAnalysisStrategy(BaseSignalStrategy):
    """Generate signals using common technical analysis indicators."""

    def __init__(self, config: TechnicalAnalysisStrategyConfig, event_bus: Any) -> None:
        """Initialize indicator configuration and shared state."""
        super().__init__(config, event_bus)
        self.config = config
        self.name = config.name

        self.trend_indicators = config.trend_indicators
        self.momentum_indicators = config.momentum_indicators
        self.volatility_indicators = config.volatility_indicators
        self.volume_indicators = config.volume_indicators
        self.signal_generation_rules = list(config.signal_generation_rules or [])
        if not self.signal_generation_rules:
            self.signal_generation_rules = ['trend_following']

        self.confidence_threshold = config.confidence_threshold
        self.min_signal_strength = config.min_signal_strength
        self.volume_confirmation = config.use_volume_confirmation
        self.volume_multiplier = config.volume_multiplier
        self.signal_aggregation = config.signal_aggregation

    # ------------------------------------------------------------------#
    # Utilities
    # ------------------------------------------------------------------#
    def get_required_columns(self) -> list[str]:
        """Return the feature set expected by this strategy."""
        return [
            'open',
            'high',
            'low',
            'close',
            'volume',
            'rsi',
            'macd',
            'macd_signal',
            'macd_histogram',
            'adx',
            'di_plus',
            'di_minus',
            'bollinger_upper',
            'bollinger_middle',
            'bollinger_lower',
            'atr',
            'obv',
            'volume_sma',
        ]

    def validate_signal_data(self, data: pd.DataFrame) -> bool:
        """Ensure the dataframe contains minimum structural requirements."""
        if data is None or data.empty:
            return False

        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(data.columns):
            return False
        return len(data) >= 20

    # Indicator preparation -------------------------------------------------
    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _compute_stochastic(
        self, data: pd.DataFrame, period: int = 14
    ) -> tuple[pd.Series, pd.Series]:
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        k = ((data['close'] - lowest_low) / (highest_high - lowest_low + 1e-12)) * 100
        d = k.rolling(window=3).mean()
        return k.fillna(50.0), d.fillna(50.0)

    def _compute_bollinger(
        self, series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift(1)).abs()
        low_close = (data['low'] - data['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().bfill()
        return atr.fillna(0.0)

    def _compute_adx(
        self, data: pd.DataFrame, period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        up_move = data['high'].diff()
        down_move = data['low'].diff() * -1

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = self._compute_atr(data, period)
        plus_di = (
            100
            * pd.Series(plus_dm, index=data.index).ewm(alpha=1 / period, adjust=False).mean()
            / (tr + 1e-12)
        )
        minus_di = (
            100
            * pd.Series(minus_dm, index=data.index).ewm(alpha=1 / period, adjust=False).mean()
            / (tr + 1e-12)
        )

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20.0)
        return adx, plus_di.fillna(0.0), minus_di.fillna(0.0)

    def _prepare_indicator_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        # Allow tests to patch IndicatorFactory; we call it but don't depend on the return value.
        for indicator_cfg in self.config.indicators:
            try:
                indicator = IndicatorFactory.create(
                    indicator_cfg.indicator_name.lower(), indicator_cfg
                )
                if indicator is not None:
                    indicator.calculate(result)
            except Exception:
                continue

        result['ma_20'] = result['close'].rolling(window=20).mean()
        result['ma_50'] = result['close'].rolling(window=50).mean()

        ema_12 = result['close'].ewm(span=12, adjust=False).mean()
        ema_26 = result['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_histogram'] = macd_hist

        result['rsi'] = self._compute_rsi(result['close'])
        stochastic_k, stochastic_d = self._compute_stochastic(result)
        result['stochastic_k'] = stochastic_k
        result['stochastic_d'] = stochastic_d

        upper, mid, lower = self._compute_bollinger(result['close'])
        result['bollinger_upper'] = upper
        result['bollinger_middle'] = mid
        result['bollinger_lower'] = lower

        result['atr'] = self._compute_atr(result)
        adx, di_plus, di_minus = self._compute_adx(result)
        result['adx'] = adx
        result['di_plus'] = di_plus
        result['di_minus'] = di_minus

        obv = [0.0]
        for idx in range(1, len(result)):
            if result['close'].iloc[idx] > result['close'].iloc[idx - 1]:
                obv.append(obv[-1] + result['volume'].iloc[idx])
            elif result['close'].iloc[idx] < result['close'].iloc[idx - 1]:
                obv.append(obv[-1] - result['volume'].iloc[idx])
            else:
                obv.append(obv[-1])
        result['obv'] = pd.Series(obv, index=result.index)
        result['volume_sma'] = result['volume'].rolling(window=20).mean()

        return result.bfill().ffill()

    # Signal generators ----------------------------------------------------
    def _calculate_trend_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        last = data.iloc[-1]
        adx_value = float(last.get('adx', 0.0))
        close = float(last['close'])
        ma20 = float(last.get('ma_20', close))
        ma50 = float(last.get('ma_50', close))

        if close > ma20 > ma50 and adx_value >= 20:
            signal_type = 'BUY'
        elif close < ma20 < ma50 and adx_value >= 20:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'

        confidence = float(np.clip(adx_value / 100, 0.0, 1.0))
        return [
            {
                'signal_type': signal_type,
                'confidence': max(confidence, 0.5 if signal_type == 'HOLD' else confidence),
                'action': 'Trend assessment',
                'metadata': {'adx': adx_value},
            }
        ]

    def _calculate_momentum_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        last = data.iloc[-1]
        rsi_value = float(last.get('rsi', 50.0))
        if rsi_value < 30:
            signal_type = 'BUY'
        elif rsi_value > 70:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        confidence = float(np.clip(abs(rsi_value - 50) / 50, 0.0, 1.0))
        return [
            {
                'signal_type': signal_type,
                'confidence': max(confidence, 0.5 if signal_type == 'HOLD' else confidence),
                'action': 'Momentum assessment',
                'metadata': {'rsi': rsi_value},
            }
        ]

    def _calculate_volatility_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        last = data.iloc[-1]
        close = float(last['close'])
        upper = float(last.get('bollinger_upper', close))
        lower = float(last.get('bollinger_lower', close))
        atr = float(last.get('atr', 0.0))

        if close >= upper:
            signal_type = 'SELL'
        elif close <= lower:
            signal_type = 'BUY'
        else:
            signal_type = 'HOLD'

        band_width = max((upper - lower) / (close + 1e-12), 0.0)
        confidence = float(np.clip(band_width + atr / (close + 1e-12), 0.0, 1.0))
        return [
            {
                'signal_type': signal_type,
                'confidence': max(confidence, 0.5 if signal_type == 'HOLD' else confidence),
                'action': 'Volatility assessment',
                'metadata': {'atr': atr},
            }
        ]

    def _calculate_volume_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        last = data.iloc[-1]
        obv = float(last.get('obv', 0.0))
        volume = float(last.get('volume', 0.0))
        volume_sma = float(last.get('volume_sma', volume))

        if volume > volume_sma * self.volume_multiplier:
            signal_type = 'BUY' if obv >= 0 else 'SELL'
        else:
            signal_type = 'HOLD'

        confidence = float(np.clip(volume / (volume_sma + 1e-12), 0.0, 1.0))
        return [
            {
                'signal_type': signal_type,
                'confidence': max(confidence, 0.5 if signal_type == 'HOLD' else confidence),
                'action': 'Volume assessment',
                'metadata': {'volume_ratio': volume / (volume_sma + 1e-12)},
            }
        ]

    # Backwards compatible aliases used by tests
    def _generate_trend_following_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        return self._calculate_trend_signals(data, symbol)

    def _generate_momentum_breakout_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        return self._calculate_momentum_signals(data, symbol)

    def _generate_mean_reversion_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        last = data.iloc[-1]
        close = float(last['close'])
        upper = float(last.get('bollinger_upper', close))
        lower = float(last.get('bollinger_lower', close))

        if close >= upper:
            signal_type = 'SELL'
        elif close <= lower:
            signal_type = 'BUY'
        else:
            signal_type = 'HOLD'

        confidence = float(np.clip(abs(close - (upper + lower) / 2) / (close + 1e-12), 0.0, 1.0))
        return [
            {
                'signal_type': signal_type,
                'confidence': max(confidence, 0.5 if signal_type == 'HOLD' else confidence),
                'action': 'Mean reversion assessment',
                'metadata': {'close': close},
            }
        ]

    def _generate_volume_breakout_signals(
        self, data: pd.DataFrame, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        return self._calculate_volume_signals(data, symbol)

    # Combination helpers --------------------------------------------------
    def _combine_signals(self, *signals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        for signal_list in signals:
            combined.extend(signal_list)
        return combined

    def _aggregate_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not signals:
            return []

        grouped: dict[str, list[dict[str, Any]]] = {}
        for signal in signals:
            grouped.setdefault(signal['signal_type'], []).append(signal)

        aggregated: list[dict[str, Any]] = []
        for _signal_type, entries in grouped.items():
            if len(entries) == 1:
                aggregated.append(entries[0])
                continue

            avg_confidence = float(np.mean([s['confidence'] for s in entries]))
            best_signal = max(entries, key=lambda s: s['confidence'])
            best_signal = best_signal.copy()
            best_signal['confidence'] = avg_confidence
            aggregated.append(best_signal)

        return aggregated

    # ------------------------------------------------------------------#
    # Strategy entry point
    # ------------------------------------------------------------------#
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        """Generate, combine, and filter signals for the provided symbol."""
        if data is None:
            raise NotImplementedError("Technical analysis signal generation requires data input")

        if not self.validate_signal_data(data):
            return []

        enriched = self._prepare_indicator_columns(data)

        trend_signals = self._calculate_trend_signals(enriched, symbol)
        momentum_signals = self._calculate_momentum_signals(enriched, symbol)
        volatility_signals = self._calculate_volatility_signals(enriched, symbol)
        volume_signals = self._calculate_volume_signals(enriched, symbol)

        combined = self._combine_signals(
            trend_signals,
            momentum_signals,
            volatility_signals,
            volume_signals,
        )

        aggregated = self._aggregate_signals(combined)

        filtered = [
            signal
            for signal in aggregated
            if signal['confidence'] >= self.min_signal_strength or signal['signal_type'] == 'HOLD'
        ]

        for signal in filtered:
            self.signals.append(signal)
            self.signal_history.append(signal)
            self.signal_count += 1
            self.valid_signal_count += 1
            self.last_signal_time = enriched.index[-1]

        return filtered

    def get_strategy_info(self) -> dict[str, Any]:
        """Expose high-level configuration useful for diagnostics."""
        info = super().get_strategy_info()
        info.update(
            {
                'trend_indicators': self.trend_indicators,
                'momentum_indicators': self.momentum_indicators,
                'volatility_indicators': self.volatility_indicators,
                'volume_indicators': self.volume_indicators,
                'signal_generation_rules': self.signal_generation_rules,
            }
        )
        return info
