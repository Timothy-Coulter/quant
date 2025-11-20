"""Formatting utilities for numbers, dates, and performance metrics."""

from datetime import datetime

import pandas as pd


class FormatUtils:
    """Utility class for formatting various data types."""

    def currency(
        self,
        amount: float | None,
        symbol: str = "$",
        decimal: str = ".",
        thousands: str = ",",
        decimals: int = 2,
    ) -> str:
        """Format amount as currency.

        Args:
            amount: Amount to format
            symbol: Currency symbol
            decimal: Decimal separator
            thousands: Thousands separator
            decimals: Number of decimal places

        Returns:
            Formatted currency string
        """
        # Handle non-numeric types gracefully
        if amount is None:
            amount = 0.0

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            amount = 0.0

        # Format with thousands separator
        formatted = f"{amount:,.{decimals}f}"

        return f"{symbol}{formatted}"

    def percentage(self, value: float, decimals: int = 2) -> str:
        """Format value as percentage.

        Args:
            value: Decimal value (e.g., 0.1234 for 12.34%)
            decimals: Number of decimal places

        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.{decimals}f}%"

    def number(self, value: int | float | None, decimals: int = 0) -> str:
        """Format number with thousands separator.

        Args:
            value: Number to format
            decimals: Number of decimal places

        Returns:
            Formatted number string
        """
        # Handle non-numeric types gracefully
        try:
            value = 0.0 if value is None else float(value)
        except (ValueError, TypeError):
            value = 0.0

        if decimals == 0:
            return f"{value:,.0f}"
        else:
            return f"{value:,.{decimals}f}"

    def compact_number(self, value: int | float | None) -> str:
        """Format number in compact form (e.g., 1.23M, 4.56K).

        Args:
            value: Number to format

        Returns:
            Compact formatted number string
        """
        # Handle non-numeric types gracefully
        try:
            value = 0.0 if value is None else float(value)
        except (ValueError, TypeError):
            value = 0.0

        abs_value = abs(value)

        if abs_value >= 1e9:
            return f"{value / 1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"{value / 1e6:.2f}M"
        elif abs_value >= 1e3:
            return f"{value / 1e3:.2f}K"
        else:
            return f"{value:.2f}"

    def date(self, date_obj: datetime | pd.Timestamp, format: str = "yyyy-MM-dd") -> str:
        """Format date object.

        Args:
            date_obj: Date object
            format: Date format string

        Returns:
            Formatted date string
        """
        if isinstance(date_obj, pd.Timestamp):
            date_obj = date_obj.to_pydatetime()

        if format == "yyyy-MM-dd":
            return date_obj.strftime("%Y-%m-%d")
        elif format == "MM/dd/yyyy":
            return date_obj.strftime("%m/%d/%Y")
        elif format == "dd/MM/yyyy":
            return date_obj.strftime("%d/%m/%Y")
        elif format == "yyyy-MM-dd HH:mm:ss":
            return date_obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return date_obj.strftime(format)

    def format_datetime(self, datetime_obj: datetime | pd.Timestamp) -> str:
        """Format datetime object.

        Args:
            datetime_obj: Datetime object

        Returns:
            Formatted datetime string
        """
        if datetime_obj is None:
            return ""

        if isinstance(datetime_obj, pd.Timestamp):
            datetime_obj = datetime_obj.to_pydatetime()

        return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

    def time_format(self, datetime_obj: datetime | pd.Timestamp) -> str:
        """Format time part of datetime object.

        Args:
            datetime_obj: Datetime object

        Returns:
            Formatted time string
        """
        if datetime_obj is None:
            return ""

        if isinstance(datetime_obj, pd.Timestamp):
            datetime_obj = datetime_obj.to_pydatetime()

        return datetime_obj.strftime("%H:%M:%S")

    def return_pct(self, return_value: float, decimals: int = 2) -> str:
        """Format return percentage with sign.

        Args:
            return_value: Return value as decimal
            decimals: Number of decimal places

        Returns:
            Formatted return percentage string
        """
        sign = "+" if return_value >= 0 else ""
        return f"{sign}{return_value * 100:.{decimals}f}%"

    def ratio(self, value: float, decimals: int = 3) -> str:
        """Format ratio value.

        Args:
            value: Ratio value
            decimals: Number of decimal places

        Returns:
            Formatted ratio string
        """
        return f"{value:.{decimals}f}"

    def leverage_ratio(self, leverage: float, decimals: int = 2) -> str:
        """Format leverage ratio.

        Args:
            leverage: Leverage value
            decimals: Number of decimal places

        Returns:
            Formatted leverage ratio string
        """
        return f"{leverage:.{decimals}f}x"

    def drawdown(self, drawdown_value: float, decimals: int = 2) -> str:
        """Format drawdown value (always negative).

        Args:
            drawdown_value: Drawdown value as decimal
            decimals: Number of decimal places

        Returns:
            Formatted drawdown string
        """
        return f"{drawdown_value * 100:.{decimals}f}%"


# Standalone formatting functions for backward compatibility
def format_currency(
    amount: float | None,
    symbol: str = "$",
    decimal: str = ".",
    thousands: str = ",",
    decimals: int = 2,
) -> str:
    """Format amount as currency."""
    formatter = FormatUtils()
    return formatter.currency(amount, symbol, decimal, thousands, decimals)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    formatter = FormatUtils()
    return formatter.percentage(value, decimals)
