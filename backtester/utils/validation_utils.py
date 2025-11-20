"""Data validation utilities."""

import re
from typing import Any

import numpy as np
import pandas as pd


class ValidationUtils:
    """Utility class for data validation."""

    def validate_dataframe(self, data: Any) -> bool:
        """Validate if input is a valid DataFrame.

        Args:
            data: Object to validate

        Returns:
            True if valid DataFrame, False otherwise
        """
        return isinstance(data, pd.DataFrame) and not data.empty

    def validate_series(self, data: Any) -> bool:
        """Validate if input is a valid Series.

        Args:
            data: Object to validate

        Returns:
            True if valid Series, False otherwise
        """
        return isinstance(data, pd.Series) and not data.empty

    def validate_numeric(self, value: Any) -> bool:
        """Validate if input is a valid numeric value.

        Args:
            value: Value to validate

        Returns:
            True if valid numeric, False otherwise
        """
        if value is None:
            return False

        try:
            num = float(value)
            return not (np.isnan(num) or np.isinf(num))
        except (ValueError, TypeError):
            return False

    def validate_config(self, config: Any) -> bool:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        # Runtime validation despite type annotation for defensive programming
        if not isinstance(config, dict):
            return False

        # Check for required keys and valid values
        required_keys = ['initial_capital']
        for key in required_keys:
            if key not in config:
                return False

        # Validate initial_capital
        if 'initial_capital' in config and (
            not self.validate_numeric(config['initial_capital']) or config['initial_capital'] <= 0
        ):
            return False

        # Validate other numeric fields if present
        numeric_fields = ['commission_rate', 'leverage', 'max_positions']
        return self._validate_numeric_fields(config, numeric_fields)

    def _validate_numeric_fields(self, config: dict[str, Any], numeric_fields: list[str]) -> bool:
        """Validate numeric fields in configuration."""
        for field in numeric_fields:
            if field in config and config[field] is not None:
                if not self.validate_numeric(config[field]):
                    return False

                # Additional validation rules
                if field == 'commission_rate' and (config[field] < 0 or config[field] > 1):
                    return False
                if field == 'leverage' and config[field] <= 0:
                    return False
                if field == 'max_positions' and (
                    config[field] < 0 or not isinstance(config[field], int)
                ):
                    return False

        return True

    def validate_range(
        self,
        value: int | float,
        min_value: float = float('-inf'),
        max_value: float = float('inf'),
    ) -> bool:
        """Validate if value is within specified range.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            True if within range, False otherwise
        """
        if not self.validate_numeric(value):
            return False

        return min_value <= float(value) <= max_value

    def validate_choice(self, value: str, choices: list[str]) -> bool:
        """Validate if value is in list of choices.

        Args:
            value: Value to validate
            choices: List of valid choices

        Returns:
            True if valid choice, False otherwise
        """
        return value in choices

    def validate_email(self, email: str) -> bool:
        """Validate email format.

        Args:
            email: Email address to validate

        Returns:
            True if valid email format, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def validate_url(self, url: str) -> bool:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if valid URL format, False otherwise
        """
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))

    def validate_ticker_format(self, ticker: str) -> bool:
        """Validate ticker symbol format.

        Args:
            ticker: Ticker symbol

        Returns:
            True if valid format, False otherwise
        """
        if not ticker or len(ticker) == 0:
            return False

        # Basic validation: alphanumeric with optional dots and hyphens
        pattern = r'^[A-Z0-9][A-Z0-9.\-]*[A-Z0-9]$|^[A-Z0-9]$'
        return bool(re.match(pattern, ticker.upper()))

    def validate_date_format(self, date_str: str) -> bool:
        """Validate date string format.

        Args:
            date_str: Date string

        Returns:
            True if valid format, False otherwise
        """
        if not date_str:
            return False

        # Try common date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']

        for fmt in formats:
            try:
                from datetime import datetime

                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue

        return False
