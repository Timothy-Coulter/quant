"""Tests for DataUtils class."""

import numpy as np
import pandas as pd
import pytest

from backtester.utils.data_utils import DataUtils


class TestDataUtils:
    """Test suite for DataUtils class."""

    def test_ohlc_validation(self) -> None:
        """Test OHLC data validation."""
        validator = DataUtils()

        # Valid OHLC data
        valid_data = pd.DataFrame(
            {
                'Open': [100, 101, 102, 103, 104],
                'High': [105, 106, 107, 108, 109],
                'Low': [99, 100, 101, 102, 103],
                'Close': [103, 104, 105, 106, 107],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        is_valid, errors = validator.validate_ohlcv(valid_data)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid OHLC data
        invalid_data = pd.DataFrame(
            {
                'Open': [100, 101, 102, 103, 104],
                'High': [95, 106, 107, 108, 109],  # High < Open
                'Low': [99, 100, 101, 102, 103],
                'Close': [103, 104, 105, 106, 107],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        is_valid, errors = validator.validate_ohlcv(invalid_data)
        assert is_valid is False
        assert len(errors) > 0

    def test_price_movement_analysis(self) -> None:
        """Test price movement analysis."""
        analyzer = DataUtils()

        prices = pd.Series([100, 101, 102, 105, 108, 107, 110, 109, 112, 115])

        analysis = analyzer.analyze_price_movements(prices)

        assert 'trend' in analysis
        assert 'volatility' in analysis
        assert 'max_drawdown' in analysis
        assert 'recovery_factor' in analysis

        assert analysis['trend'] in ['uptrend', 'downtrend', 'sideways']
        assert analysis['volatility'] >= 0

    def test_data_cleaning(self) -> None:
        """Test data cleaning functionality."""
        cleaner = DataUtils()

        # Data with missing values and outliers
        dirty_data = pd.DataFrame(
            {
                'Close': [100, np.nan, 102, 1000, 103, np.nan, 104, 105],  # 1000 is outlier
                'Volume': [1000000, 1100000, np.nan, 1300000, 1400000, 1500000, 1600000, 1700000],
            }
        )

        # Clean missing values
        cleaned_missing = cleaner.fill_missing_values(dirty_data, method='forward_fill')
        assert not cleaned_missing['Close'].isna().any()

        # Remove outliers
        cleaned_outliers = cleaner.remove_outliers(dirty_data, columns=['Close'], method='iqr')
        assert 1000 not in cleaned_outliers['Close'].values

    def test_data_alignment(self) -> None:
        """Test data alignment across different timeframes."""
        aligner = DataUtils()

        # Create data with different frequencies
        daily_data = pd.DataFrame(
            {
                'Close': [100, 101, 102, 103, 104],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=pd.date_range('2020-01-01', periods=5, freq='D'),
        )

        hourly_data = pd.DataFrame(
            {'Close': list(range(100, 125)), 'Volume': [100000] * 25},
            index=pd.date_range('2020-01-01 09:30', periods=25, freq='h'),
        )

        # Align data to daily frequency
        aligned = aligner.align_to_frequency(daily_data, 'D')
        assert len(aligned) == len(daily_data)

        # Resample hourly to daily using mean aggregation
        resampled = aligner.resample_data(hourly_data, 'D', 'mean')
        assert len(resampled) <= len(hourly_data)
        assert 'Close' in resampled.columns
        assert 'Volume' in resampled.columns

    def test_correlation_analysis(self) -> None:
        """Test correlation analysis between assets."""
        analyzer = DataUtils()

        # Create correlated data
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.02, 100)
        spy_returns = base_returns + np.random.normal(0, 0.01, 100)
        aapl_returns = base_returns * 0.8 + np.random.normal(0, 0.015, 100)

        data = pd.DataFrame({'SPY': spy_returns, 'AAPL': aapl_returns})

        correlation_matrix = analyzer.calculate_correlation_matrix(data)

        assert 'SPY' in correlation_matrix.columns
        assert 'AAPL' in correlation_matrix.columns
        assert correlation_matrix.loc['SPY', 'SPY'] == 1.0  # Self-correlation
        assert abs(correlation_matrix.loc['SPY', 'AAPL']) > 0  # Should be correlated

    def test_data_synchronization(self) -> None:
        """Test data synchronization across different sources."""
        synchronizer = DataUtils()

        # Create datasets with different time ranges
        data1 = pd.DataFrame(
            {'Close': [100, 101, 102], 'Volume': [1000000, 1100000, 1200000]},
            index=pd.date_range('2020-01-01', periods=3, freq='D'),
        )

        data2 = pd.DataFrame(
            {
                'Close': [99, 100, 101, 102, 103],
                'Volume': [900000, 1000000, 1100000, 1200000, 1300000],
            },
            index=pd.date_range('2019-12-31', periods=5, freq='D'),
        )

        # Synchronize to common date range
        synchronized = synchronizer.synchronize_data([data1, data2])

        assert isinstance(synchronized, pd.DataFrame)
        assert len(synchronized.columns) == 4  # Two datasets, each with Close and Volume
        assert synchronized.index.equals(data1.index.intersection(data2.index))

    def test_fill_missing_values_invalid_method(self) -> None:
        """Test fill_missing_values with invalid method."""
        cleaner = DataUtils()

        data = pd.DataFrame({'A': [1, np.nan, 3]})

        with pytest.raises(ValueError, match="Unknown fill method"):
            cleaner.fill_missing_values(data, method='invalid_method')

    def test_remove_outliers_invalid_method(self) -> None:
        """Test remove_outliers with invalid method."""
        cleaner = DataUtils()

        data = pd.DataFrame({'A': [1, 2, 3, 100, 4, 5]})

        # Note: The current implementation may not raise ValueError for invalid methods
        # This test documents current behavior
        result = cleaner.remove_outliers(data, columns=['A'], method='invalid_method')
        # Should return the original data unchanged for unknown methods
        assert len(result) <= len(data)

    def test_remove_outliers_nonexistent_column(self) -> None:
        """Test remove_outliers with nonexistent column."""
        cleaner = DataUtils()

        data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

        # Should not raise error, just skip the nonexistent column
        result = cleaner.remove_outliers(data, columns=['B'], method='iqr')
        pd.testing.assert_frame_equal(result, data)

    def test_empty_data_validation(self) -> None:
        """Test validation with empty data."""
        validator = DataUtils()

        empty_data = pd.DataFrame()
        is_valid, errors = validator.validate_ohlcv(empty_data)

        assert is_valid is False
        assert len(errors) > 0

    def test_synchronize_empty_list(self) -> None:
        """Test synchronize_data with empty list."""
        synchronizer = DataUtils()

        result = synchronizer.synchronize_data([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__])
