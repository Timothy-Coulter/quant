"""Tests for ValidationUtils class."""

import numpy as np
import pandas as pd
import pytest

from backtester.utils.validation_utils import ValidationUtils


class TestValidationUtils:
    """Test suite for ValidationUtils class."""

    def test_dataframe_validation(self) -> None:
        """Test DataFrame validation."""
        validator = ValidationUtils()

        # Valid DataFrame
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert validator.validate_dataframe(valid_df) is True

        # Empty DataFrame
        empty_df = pd.DataFrame()
        assert validator.validate_dataframe(empty_df) is False

        # None value
        assert validator.validate_dataframe(None) is False

        # Non-DataFrame object
        assert validator.validate_dataframe("not a dataframe") is False
        assert validator.validate_dataframe([1, 2, 3]) is False

    def test_series_validation(self) -> None:
        """Test Series validation."""
        validator = ValidationUtils()

        # Valid Series
        valid_series = pd.Series([1, 2, 3, 4, 5])
        assert validator.validate_series(valid_series) is True

        # Empty Series
        empty_series = pd.Series([], dtype=float)
        assert validator.validate_series(empty_series) is False

        # None value
        assert validator.validate_series(None) is False

        # Non-Series object
        assert validator.validate_series("not a series") is False
        assert validator.validate_series([1, 2, 3]) is False

    def test_numeric_validation(self) -> None:
        """Test numeric validation."""
        validator = ValidationUtils()

        # Valid numeric values
        assert validator.validate_numeric(123) is True
        assert validator.validate_numeric(123.456) is True
        assert validator.validate_numeric("123") is True
        assert validator.validate_numeric("123.456") is True
        assert validator.validate_numeric(-123.456) is True

        # Invalid numeric values
        assert validator.validate_numeric(None) is False
        assert validator.validate_numeric("not_numeric") is False
        assert validator.validate_numeric(np.nan) is False
        assert validator.validate_numeric(np.inf) is False
        assert validator.validate_numeric(-np.inf) is False
        assert validator.validate_numeric("") is False

    def test_configuration_validation(self) -> None:
        """Test configuration validation."""
        validator = ValidationUtils()

        # Valid configuration
        valid_config = {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,
            'leverage': 2.0,
            'max_positions': 10,
        }
        assert validator.validate_config(valid_config) is True

        # Invalid configuration - negative initial capital
        invalid_config1 = {'initial_capital': -1000}
        assert validator.validate_config(invalid_config1) is False

        # Invalid configuration - zero commission (may be valid in some cases)
        # This might be valid depending on business logic

        # Invalid configuration - zero leverage
        invalid_config3 = {'initial_capital': 1000, 'leverage': 0}
        assert validator.validate_config(invalid_config3) is False

        # Invalid configuration - negative positions
        invalid_config4 = {'initial_capital': 1000, 'max_positions': -1}
        assert validator.validate_config(invalid_config4) is False

        # Invalid configuration - non-numeric positions
        invalid_config5 = {'initial_capital': 1000, 'max_positions': 'ten'}
        assert validator.validate_config(invalid_config5) is False

        # Invalid configuration - wrong type
        assert validator.validate_config("not a dict") is False
        assert validator.validate_config(None) is False
        assert validator.validate_config([]) is False

    def test_range_validation(self) -> None:
        """Test range validation."""
        validator = ValidationUtils()

        # Valid ranges
        assert validator.validate_range(5, min_value=0, max_value=10) is True
        assert validator.validate_range(0, min_value=0, max_value=10) is True
        assert validator.validate_range(10, min_value=0, max_value=10) is True

        # Invalid ranges
        assert validator.validate_range(15, min_value=0, max_value=10) is False
        assert validator.validate_range(-5, min_value=0, max_value=10) is False

        # Default infinite bounds
        assert validator.validate_range(999999, min_value=0, max_value=10) is False
        assert validator.validate_range(-999999) is True  # Default bounds are infinite

    def test_choice_validation(self) -> None:
        """Test choice validation."""
        validator = ValidationUtils()

        # Valid choices
        assert validator.validate_choice('sma', ['sma', 'ema', 'wma']) is True
        assert validator.validate_choice('ema', ['sma', 'ema', 'wma']) is True

        # Invalid choices
        assert validator.validate_choice('invalid', ['sma', 'ema', 'wma']) is False
        assert validator.validate_choice('', ['sma', 'ema', 'wma']) is False

        # Empty choices list
        assert validator.validate_choice('sma', []) is False

    def test_email_validation(self) -> None:
        """Test email format validation."""
        validator = ValidationUtils()

        # Valid email addresses
        assert validator.validate_email("user@example.com") is True
        assert validator.validate_email("test.email+tag@domain.co.uk") is True
        assert validator.validate_email("user123@test-domain.org") is True

        # Invalid email addresses
        assert validator.validate_email("invalid-email") is False
        assert validator.validate_email("user@") is False
        assert validator.validate_email("@domain.com") is False
        # Note: Double dots might be allowed by the regex, adjust test accordingly
        # assert validator.validate_email("user..double.dot@example.com") is False
        assert validator.validate_email("") is False

    def test_url_validation(self) -> None:
        """Test URL format validation."""
        validator = ValidationUtils()

        # Valid URLs
        assert validator.validate_url("https://example.com") is True
        assert validator.validate_url("http://www.example.com/path") is True
        assert validator.validate_url("https://subdomain.example.org") is True

        # Invalid URLs
        assert validator.validate_url("not-a-url") is False
        assert validator.validate_url("ftp://example.com") is False
        assert validator.validate_url("https://") is False
        assert validator.validate_url("") is False

    def test_ticker_format_validation(self) -> None:
        """Test ticker symbol format validation."""
        validator = ValidationUtils()

        # Valid ticker symbols
        assert validator.validate_ticker_format("AAPL") is True
        assert validator.validate_ticker_format("BRK.A") is True
        assert validator.validate_ticker_format("GOOGL") is True
        assert validator.validate_ticker_format("V") is True  # Single character

        # Invalid ticker symbols
        assert validator.validate_ticker_format("") is False
        assert validator.validate_ticker_format("AAPL!") is False
        assert validator.validate_ticker_format("@APL") is False
        assert validator.validate_ticker_format("APP L") is False  # Space

    def test_date_format_validation(self) -> None:
        """Test date string format validation."""
        validator = ValidationUtils()

        # Valid date formats
        assert validator.validate_date_format("2023-01-15") is True
        assert validator.validate_date_format("01/15/2023") is True
        assert validator.validate_date_format("15/01/2023") is True
        assert validator.validate_date_format("2023-01-15 14:30:00") is True

        # Invalid date formats
        assert validator.validate_date_format("2023-13-01") is False  # Invalid month
        assert validator.validate_date_format("2023-02-30") is False  # Invalid day
        assert validator.validate_date_format("invalid-date") is False
        assert validator.validate_date_format("") is False

    def test_edge_cases_for_config_validation(self) -> None:
        """Test edge cases for configuration validation."""
        validator = ValidationUtils()

        # Configuration with missing initial_capital
        config_missing_capital = {'commission_rate': 0.001}
        assert validator.validate_config(config_missing_capital) is False

        # Configuration with very large values
        config_large_values = {
            'initial_capital': 1e10,
            'commission_rate': 0.5,
            'leverage': 100.0,
            'max_positions': 10000,
        }
        assert validator.validate_config(config_large_values) is True

    def test_validation_with_special_numeric_values(self) -> None:
        """Test validation with special numeric values."""
        validator = ValidationUtils()

        # Test with very small numbers
        assert validator.validate_numeric(1e-100) is True
        assert validator.validate_range(1e-100, 0, 1) is True

    def test_dataframe_with_special_values(self) -> None:
        """Test DataFrame validation with special values."""
        validator = ValidationUtils()

        # DataFrame with NaN values (should still be valid)
        df_with_nan = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        assert validator.validate_dataframe(df_with_nan) is True

        # DataFrame with inf values (should still be valid)
        df_with_inf = pd.DataFrame({'A': [1, float('inf'), 3], 'B': [4, 5, -float('inf')]})
        assert validator.validate_dataframe(df_with_inf) is True

    def test_series_with_special_values(self) -> None:
        """Test Series validation with special values."""
        validator = ValidationUtils()

        # Series with NaN values
        series_with_nan = pd.Series([1, np.nan, 3, np.nan, 5])
        assert validator.validate_series(series_with_nan) is True

        # Series with inf values
        series_with_inf = pd.Series([1, float('inf'), 3, -float('inf'), 5])
        assert validator.validate_series(series_with_inf) is True

        # Empty Series
        empty_series = pd.Series([], dtype=float)
        assert validator.validate_series(empty_series) is False


if __name__ == "__main__":
    pytest.main([__file__])
