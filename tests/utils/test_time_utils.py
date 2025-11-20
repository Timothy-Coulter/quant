"""Tests for TimeUtils class."""

from datetime import datetime

import pytest

from backtester.utils.time_utils import TimeUtils


class TestTimeUtils:
    """Test suite for TimeUtils class."""

    def test_date_parsing(self) -> None:
        """Test date parsing functionality."""
        parser = TimeUtils()

        # Test various date formats
        date_str = "2023-01-15"
        parsed_date = parser.parse_date(date_str)
        assert isinstance(parsed_date, datetime)
        assert parsed_date.year == 2023
        assert parsed_date.month == 1
        assert parsed_date.day == 15

        # Test invalid date
        assert parser.parse_date("invalid-date") is None
        assert parser.parse_date("") is None

    def test_date_validation(self) -> None:
        """Test date validation."""
        validator = TimeUtils()

        # Valid dates
        assert validator.validate_date("2023-01-01") is True
        assert validator.validate_date("2023-12-31") is True
        assert validator.validate_date("01/15/2023") is True

        # Invalid dates
        assert validator.validate_date("2023-13-01") is False  # Invalid month
        assert validator.validate_date("2023-02-30") is False  # Invalid day
        assert validator.validate_date("invalid-date") is False
        assert validator.validate_date("") is False

    def test_business_day_calculations(self) -> None:
        """Test business day calculations."""
        calculator = TimeUtils()

        # Test adding business days
        start_date = datetime(2023, 1, 2)  # Monday
        result = calculator.add_business_days(start_date, 5)
        expected_date = datetime(2023, 1, 9)  # Next Monday (5 business days)
        assert result == expected_date

        # Test weekend adjustment
        weekend_date = datetime(2023, 1, 7)  # Saturday
        adjusted = calculator.adjust_to_business_day(weekend_date)
        assert adjusted.weekday() < 5  # Should be weekday (0-4)
        assert adjusted >= weekend_date  # Should be same or after original date

    def test_frequency_conversion(self) -> None:
        """Test frequency conversion utilities."""
        converter = TimeUtils()

        # Test frequency parsing
        daily_freq = converter.parse_frequency("1d")
        monthly_freq = converter.parse_frequency("1mo")
        hourly_freq = converter.parse_frequency("1h")

        assert daily_freq == 'D'
        assert monthly_freq == 'M'
        assert hourly_freq == 'H'

    def test_market_hours_check(self) -> None:
        """Test market hours checking."""
        checker = TimeUtils()

        # Test market hours (assuming 9:30 AM to 4:00 PM ET)
        market_open = datetime(2023, 1, 3, 9, 30)  # Tuesday 9:30 AM
        market_close = datetime(2023, 1, 3, 16, 0)  # Tuesday 4:00 PM
        after_hours = datetime(2023, 1, 3, 20, 0)  # Tuesday 8:00 PM
        weekend = datetime(2023, 1, 7, 10, 0)  # Saturday 10:00 AM

        assert checker.is_market_hours(market_open) is True
        assert checker.is_market_hours(market_close) is True
        assert checker.is_market_hours(after_hours) is False
        assert checker.is_market_hours(weekend) is False


if __name__ == "__main__":
    pytest.main([__file__])
