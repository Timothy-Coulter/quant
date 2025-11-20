"""Mathematical and statistical utility functions."""

import numpy as np
import pandas as pd


class MathUtils:
    """Utility class for mathematical operations."""

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            prices: Series of prices
            period: Period for moving average

        Returns:
            Series with SMA values
        """
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            prices: Series of prices
            period: Period for EMA

        Returns:
            Series with EMA values
        """
        return prices.ewm(span=period).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.

        Args:
            prices: Series of prices
            period: Period for RSI calculation

        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            prices: Series of prices
            period: Period for moving average
            std_dev: Number of standard deviations

        Returns:
            DataFrame with upper, middle, and lower bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

    @staticmethod
    def calculate_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD, signal, and histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal).mean()
        histogram = macd - signal

        return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})

    @staticmethod
    def calculate_skewness(data: pd.Series) -> float:
        """Calculate skewness of data.

        Args:
            data: Series of data

        Returns:
            Skewness value
        """
        return data.skew()  # type: ignore[no-any-return]

    @staticmethod
    def calculate_kurtosis(data: pd.Series) -> float:
        """Calculate kurtosis of data.

        Args:
            data: Series of data

        Returns:
            Kurtosis value
        """
        return data.kurtosis()  # type: ignore[no-any-return]

    @staticmethod
    def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)  # type: ignore[no-any-return]

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division that handles zero and infinite denominators.

        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value to return if division is not possible

        Returns:
            Division result or default value
        """
        if denominator == 0 or np.isinf(denominator):
            return default
        if np.isinf(numerator):
            return numerator
        return numerator / denominator

    @staticmethod
    def rolling_window(data: pd.Series, window: int, operation: str = 'mean') -> pd.Series:
        """Apply rolling window operation.

        Args:
            data: Series of data
            window: Window size
            operation: Operation to apply ('mean', 'sum', 'std', 'var')

        Returns:
            Series with rolling window results
        """
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def interpolate_missing(data: pd.Series, method: str = 'linear') -> pd.Series:
        """Interpolate missing values.

        Args:
            data: Series with missing values
            method: Interpolation method ('linear', 'ffill', 'bfill')

        Returns:
            Series with interpolated values
        """
        if method == 'linear':
            return data.interpolate(method='linear')
        elif method == 'ffill':
            return data.ffill()
        elif method == 'bfill':
            return data.bfill()
        else:
            raise ValueError(f"Unknown interpolation method: {method}")


# Standalone functions for backward compatibility
def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return MathUtils.calculate_sma(prices, period)


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return MathUtils.calculate_ema(prices, period)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    return MathUtils.calculate_rsi(prices, period)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division function."""
    return MathUtils.safe_divide(numerator, denominator, default)


def rolling_window(data: pd.Series, window: int, operation: str = 'mean') -> pd.Series:
    """Apply rolling window operation."""
    return MathUtils.rolling_window(data, window, operation)


def interpolate_missing(data: pd.Series, method: str = 'linear') -> pd.Series:
    """Interpolate missing values."""
    return MathUtils.interpolate_missing(data, method)
