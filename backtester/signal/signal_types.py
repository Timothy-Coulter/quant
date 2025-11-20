"""Signal types and signal generation utilities for the indicator system.

This module defines the standard signal types and provides utilities for
generating consistent trading signals across all indicators.
"""

from enum import Enum
from typing import Any


class SignalType(Enum):
    """Enumeration of possible signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalGenerator:
    """Utility class for generating consistent trading signals."""

    @staticmethod
    def create_signal(
        signal_type: SignalType,
        action: str,
        confidence: float,
        timestamp: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized trading signal.

        Args:
            signal_type: Type of signal (BUY, SELL, HOLD)
            action: Detailed action description
            confidence: Confidence level (0.0 to 1.0)
            timestamp: Signal timestamp
            metadata: Additional signal information

        Returns:
            Standardized signal dictionary
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        return {
            'timestamp': timestamp,
            'signal_type': signal_type.value,
            'action': action,
            'confidence': float(confidence),
            'metadata': metadata or {},
        }

    @staticmethod
    def validate_signal(signal: dict[str, Any]) -> bool:
        """Validate a signal dictionary structure.

        Args:
            signal: Signal dictionary to validate

        Returns:
            True if signal is valid, raises exception otherwise
        """
        required_fields = ['signal_type', 'action', 'confidence', 'metadata', 'timestamp']

        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")

        if signal['signal_type'] not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError("Invalid signal_type")

        if not isinstance(signal['confidence'], (int, float)) or not (
            0.0 <= signal['confidence'] <= 1.0
        ):
            raise ValueError("Confidence must be a number between 0.0 and 1.0")

        if not isinstance(signal['metadata'], dict):
            raise ValueError("Metadata must be a dictionary")

        return True


# Type aliases for convenience
SignalDict = dict[str, Any]
SignalList = list[SignalDict]
