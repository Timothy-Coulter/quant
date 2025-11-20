"""Time and date utility functions."""

from datetime import datetime, timedelta

import pandas as pd


class TimeUtils:
    """Utility class for time and date operations."""

    def parse_date(self, date_string: str) -> datetime | None:
        """Parse date string to datetime object.

        Args:
            date_string: Date string in various formats

        Returns:
            Parsed datetime object or None if parsing fails
        """
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        return None

    def parse_datetime(self, datetime_string: str) -> datetime | None:
        """Parse datetime string to datetime object.

        Args:
            datetime_string: Datetime string

        Returns:
            Parsed datetime object or None if parsing fails
        """
        return self.parse_date(datetime_string)

    def validate_date(self, date_string: str) -> bool:
        """Validate if date string is valid.

        Args:
            date_string: Date string to validate

        Returns:
            True if valid, False otherwise
        """
        return self.parse_date(date_string) is not None

    def add_business_days(self, start_date: datetime, num_days: int) -> datetime:
        """Add business days to a date.

        Args:
            start_date: Starting date
            num_days: Number of business days to add

        Returns:
            Resulting date
        """
        current_date = start_date
        days_added = 0

        while days_added < num_days:
            current_date += timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                days_added += 1

        return current_date

    def adjust_to_business_day(self, date: datetime) -> datetime:
        """Adjust date to next business day if it's weekend.

        Args:
            date: Input date

        Returns:
            Adjusted business day
        """
        if date.weekday() >= 5:  # Weekend
            # Add days to get to Monday
            days_to_add = 7 - date.weekday()
            date += timedelta(days=days_to_add)

        return date

    def parse_frequency(self, frequency_str: str) -> str | None:
        """Parse frequency string to pandas frequency.

        Args:
            frequency_str: Frequency string

        Returns:
            Pandas frequency string or None if invalid
        """
        frequency_map = {
            '1d': 'D',
            'daily': 'D',
            '1h': 'H',
            'hourly': 'H',
            '1w': 'W',
            'weekly': 'W',
            '1mo': 'M',
            'monthly': 'M',
            '1y': 'Y',
            'yearly': 'Y',
        }

        return frequency_map.get(frequency_str.lower())

    def calculate_periods(self, start_date: datetime, end_date: datetime, frequency: str) -> int:
        """Calculate number of periods between two dates.

        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency string

        Returns:
            Number of periods
        """
        try:
            freq = self.parse_frequency(frequency)
            if freq is None:
                return 0

            return len(pd.date_range(start=start_date, end=end_date, freq=freq))
        except Exception:
            return 0

    def is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within market hours.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if within market hours, False otherwise
        """
        # Check if it's a weekday
        if timestamp.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if it's within 9:30 AM to 4:00 PM ET (simplified)
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= timestamp <= market_close


def validate_date_string(date_string: str) -> bool:
    """Validate date string.

    Args:
        date_string: Date string to validate

    Returns:
        True if valid, False otherwise
    """
    utils = TimeUtils()
    return utils.validate_date(date_string)
