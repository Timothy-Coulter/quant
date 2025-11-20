"""String processing and validation utilities."""

import os
import re


class StringUtils:
    """Utility class for string operations."""

    def validate_ticker_symbol(self, symbol: str) -> bool:
        """Validate ticker symbol format.

        Args:
            symbol: Ticker symbol to validate

        Returns:
            True if valid, False otherwise
        """
        if not symbol or len(symbol) == 0:
            return False

        # Basic ticker symbol validation (alphanumeric with optional dots/hyphens)
        pattern = r'^[A-Z0-9][A-Z0-9.\-]*[A-Z0-9]$|^[A-Z0-9]$'
        return bool(re.match(pattern, symbol.upper()))

    def validate_file_path(self, file_path: str) -> bool:
        """Validate file path format.

        Args:
            file_path: File path to validate

        Returns:
            True if valid, False otherwise
        """
        if not file_path or len(file_path) == 0:
            return False

        # Basic path validation (no null characters or control characters)
        if any(ord(char) < 32 for char in file_path):
            return False

        # Check for invalid path characters (Windows and Unix)
        invalid_chars = '<>:"|?*'
        return not any(char in file_path for char in invalid_chars)

    def normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol.

        Args:
            ticker: Raw ticker symbol

        Returns:
            Normalized ticker symbol
        """
        if not ticker:
            return ""

        # Remove extra whitespace and convert to uppercase
        normalized = ticker.strip().upper()

        # Keep dots and hyphens as they are valid in ticker symbols
        return normalized

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path.

        Args:
            file_path: Raw file path

        Returns:
            Normalized file path
        """
        if not file_path:
            return ""

        # Normalize path separators
        normalized = file_path.replace('\\', '/')

        # Remove redundant separators and resolve relative components
        normalized = os.path.normpath(normalized)

        # Ensure relative paths start with ./
        if not normalized.startswith(('/', './', '../')):
            normalized = './' + normalized

        return normalized

    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format value as percentage.

        Args:
            value: Decimal value (e.g., 0.1234 for 12.34%)
            decimals: Number of decimal places

        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.{decimals}f}%"

    def format_currency(
        self,
        amount: float,
        symbol: str = "$",
        decimal: str = ".",
        thousands: str = ",",
        locale_code: str = "en_US",
    ) -> str:
        """Format amount as currency.

        Args:
            amount: Amount to format
            symbol: Currency symbol
            decimal: Decimal separator
            thousands: Thousands separator
            locale_code: Locale for formatting

        Returns:
            Formatted currency string
        """
        # Format number with thousands separator
        formatted_amount = f"{amount:,.2f}"

        # Replace separators based on locale
        if locale_code == "de_DE":
            # German format: 1.234,56 €
            formatted_amount = (
                formatted_amount.replace(',', 'X').replace('.', ',').replace('X', '.')
            )
            symbol = symbol or "€"
        else:
            # Default US format: $1,234.56
            symbol = symbol or "$"

        return f"{symbol}{formatted_amount}"

    def format_number(self, number: int | float, decimals: int = 0) -> str:
        """Format number with thousands separator.

        Args:
            number: Number to format
            decimals: Number of decimal places

        Returns:
            Formatted number string
        """
        if decimals == 0:
            return f"{number:,.0f}"
        else:
            return f"{number:,.{decimals}f}"

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        cleaned = ' '.join(text.split())

        return cleaned

    def split_text(self, text: str, max_length: int = 80) -> list[str]:
        """Split text into chunks of specified maximum length.

        Args:
            text: Text to split
            max_length: Maximum length per chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) <= max_length:
            return [text] if text else []

        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk + " " + word) <= max_length:
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
