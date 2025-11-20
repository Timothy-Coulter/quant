"""Tests for FormatUtils class."""

from datetime import datetime

import pytest

from backtester.utils.format_utils import FormatUtils


class TestFormatUtils:
    """Test suite for FormatUtils class."""

    def test_currency_formatting(self) -> None:
        """Test currency formatting."""
        formatter = FormatUtils()

        # Test basic currency formatting
        assert formatter.currency(1234.56) == "$1,234.56"
        assert formatter.currency(1234.56, symbol="€", decimal=".", thousands=",") == "€1,234.56"

        # Test custom decimal places
        assert formatter.currency(1234.56, decimals=2) == "$1,234.56"
        assert formatter.currency(1234.56, decimals=5) == "$1,234.56000"

        # Test zero value
        assert formatter.currency(0) == "$0.00"

        # Test negative value
        assert formatter.currency(-1234.56) == "$-1,234.56"

    def test_percentage_formatting(self) -> None:
        """Test percentage formatting."""
        formatter = FormatUtils()

        # Test basic percentage
        assert formatter.percentage(0.1234) == "12.34%"
        assert formatter.percentage(0.05) == "5.00%"

        # Test different decimal places
        assert formatter.percentage(0.05, decimals=1) == "5.0%"
        assert formatter.percentage(0.05, decimals=4) == "5.0000%"

        # Test zero
        assert formatter.percentage(0) == "0.00%"

        # Test negative percentage
        assert formatter.percentage(-0.1234) == "-12.34%"

    def test_number_formatting(self) -> None:
        """Test number formatting."""
        formatter = FormatUtils()

        # Test basic number formatting
        assert formatter.number(1234567) == "1,234,567"
        assert formatter.number(1234.567, decimals=2) == "1,234.57"

        # Test zero decimals
        assert formatter.number(1234.567, decimals=0) == "1,235"

        # Test negative numbers
        assert formatter.number(-1234567) == "-1,234,567"

    def test_compact_number_formatting(self) -> None:
        """Test compact number formatting."""
        formatter = FormatUtils()

        # Test millions
        assert formatter.compact_number(1234567) == "1.23M"
        assert formatter.compact_number(-1234567) == "-1.23M"

        # Test thousands
        assert formatter.compact_number(1234) == "1.23K"
        assert formatter.compact_number(-1234) == "-1.23K"

        # Test billions
        assert formatter.compact_number(1234567890) == "1.23B"

        # Test small numbers
        assert formatter.compact_number(1.23) == "1.23"
        assert formatter.compact_number(0) == "0.00"

    def test_date_formatting(self) -> None:
        """Test date formatting."""
        formatter = FormatUtils()

        test_date = datetime(2023, 1, 15, 14, 30)

        # Test different date formats
        assert formatter.date(test_date, format="yyyy-MM-dd") == "2023-01-15"
        assert formatter.date(test_date, format="MM/dd/yyyy") == "01/15/2023"
        assert formatter.date(test_date, format="dd/MM/yyyy") == "15/01/2023"
        assert formatter.date(test_date, format="yyyy-MM-dd HH:mm:ss") == "2023-01-15 14:30:00"

        # Test custom format
        assert formatter.date(test_date, format="%b %d, %Y") == "Jan 15, 2023"

    def test_datetime_formatting(self) -> None:
        """Test datetime formatting."""
        formatter = FormatUtils()

        test_datetime = datetime(2023, 1, 15, 14, 30, 45)

        # Test datetime formatting
        assert formatter.format_datetime(test_datetime) == "2023-01-15 14:30:45"

        # Test with None
        assert formatter.format_datetime(None) == ""

    def test_time_formatting(self) -> None:
        """Test time formatting."""
        formatter = FormatUtils()

        test_datetime = datetime(2023, 1, 15, 14, 30, 45)

        # Test time formatting
        assert formatter.time_format(test_datetime) == "14:30:45"

        # Test with None
        assert formatter.time_format(None) == ""

    def test_return_percentage_formatting(self) -> None:
        """Test return percentage formatting with sign."""
        formatter = FormatUtils()

        # Test positive returns
        assert formatter.return_pct(0.1234) == "+12.34%"
        assert formatter.return_pct(0.05) == "+5.00%"

        # Test negative returns
        assert formatter.return_pct(-0.0567) == "-5.67%"

        # Test zero
        assert formatter.return_pct(0) == "+0.00%"

    def test_ratio_formatting(self) -> None:
        """Test ratio formatting."""
        formatter = FormatUtils()

        # Test basic ratio (default is 3 decimals)
        assert formatter.ratio(1.234) == "1.234"
        assert formatter.ratio(2.0) == "2.000"

        # Test custom decimals
        assert formatter.ratio(1.234, decimals=4) == "1.2340"

    def test_leverage_ratio_formatting(self) -> None:
        """Test leverage ratio formatting."""
        formatter = FormatUtils()

        # Test leverage ratio formatting
        assert formatter.leverage_ratio(2.0) == "2.00x"
        assert formatter.leverage_ratio(1.5) == "1.50x"

        # Test custom decimals
        assert formatter.leverage_ratio(2.0, decimals=3) == "2.000x"

    def test_drawdown_formatting(self) -> None:
        """Test drawdown formatting (always negative)."""
        formatter = FormatUtils()

        # Test drawdown formatting
        assert formatter.drawdown(-0.1567) == "-15.67%"
        assert formatter.drawdown(-0.05) == "-5.00%"

        # Test with positive input (should format as is)
        assert formatter.drawdown(0.1567) == "15.67%"

    def test_none_values(self) -> None:
        """Test handling of None values."""
        formatter = FormatUtils()

        # Test None in currency (should handle gracefully)
        assert formatter.currency(None) == "$0.00"

        # Test None in number
        assert formatter.number(None) == "0"

        # Test None in compact number
        assert formatter.compact_number(None) == "0.00"


if __name__ == "__main__":
    pytest.main([__file__])
