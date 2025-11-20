"""Logging Configuration System.

This module provides comprehensive logging functionality for the backtester
with support for file rotation, different log levels, and structured logging.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Self


class LogLevel(StrEnum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __lt__(self, other: object) -> bool:
        """Enable comparison between log levels."""
        if isinstance(other, LogLevel):
            order = [self.DEBUG, self.INFO, self.WARNING, self.ERROR, self.CRITICAL]
            return order.index(self) < order.index(other)
        return NotImplemented


class LogFormat(str, Enum):
    """Log format enumeration."""

    SIMPLE = "SIMPLE"
    STANDARD = "STANDARD"
    DETAILED = "DETAILED"
    JSON = "JSON"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                'name',
                'msg',
                'args',
                'levelname',
                'levelno',
                'pathname',
                'filename',
                'module',
                'lineno',
                'funcName',
                'created',
                'msecs',
                'relativeCreated',
                'thread',
                'threadName',
                'processName',
                'process',
                'getMessage',
                'exc_info',
                'exc_text',
                'stack_info',
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class LoggerContextFilter(logging.Filter):
    """Ensure every record has consistent contextual metadata."""

    def __init__(
        self,
        run_id: str | None = None,
        symbol: str | None = None,
        *,
        is_default: bool = False,
    ) -> None:
        """Initialize the filter with optional context overrides."""
        super().__init__()
        self.run_id = run_id
        self.symbol = symbol
        self.is_default = is_default

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject contextual attributes if they are missing."""
        if self.run_id is not None:
            record.run_id = self.run_id
        elif getattr(record, 'run_id', None) is None:
            record.run_id = '-'

        if self.symbol is not None:
            record.symbol = self.symbol
        elif getattr(record, 'symbol', None) is None:
            record.symbol = '-'
        return True


ROOT_LOGGER_NAME = "backtester"


def _normalized_logger_name(name: str) -> str:
    if not name or name == ROOT_LOGGER_NAME:
        return ROOT_LOGGER_NAME
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return name
    return f"{ROOT_LOGGER_NAME}.{name}"


def _ensure_default_context(filterable: logging.Filterer) -> None:
    """Attach a default context filter so templates can reference run_id/symbol safely."""
    for existing in filterable.filters:
        if isinstance(existing, LoggerContextFilter) and existing.is_default:
            return
    filterable.addFilter(LoggerContextFilter(is_default=True))


def _replace_context_filter(logger: logging.Logger, context_filter: LoggerContextFilter) -> None:
    """Ensure only a single non-default context filter is attached to the logger."""
    for existing in list(logger.filters):
        if isinstance(existing, LoggerContextFilter) and not existing.is_default:
            logger.removeFilter(existing)
    logger.addFilter(context_filter)


def bind_logger_context(
    logger: logging.Logger,
    *,
    run_id: str | None = None,
    symbol: str | None = None,
) -> logging.Logger:
    """Bind contextual metadata to an existing logger instance."""
    current_run_id: str | None = None
    current_symbol: str | None = None
    for existing in logger.filters:
        if isinstance(existing, LoggerContextFilter) and not existing.is_default:
            current_run_id = existing.run_id
            current_symbol = existing.symbol
            break

    new_run_id = run_id if run_id is not None else current_run_id
    new_symbol = symbol if symbol is not None else current_symbol

    if new_run_id is None and new_symbol is None:
        return logger

    _replace_context_filter(logger, LoggerContextFilter(run_id=new_run_id, symbol=new_symbol))
    return logger


# Utility functions for easy access
def get_backtester_logger(
    name: str = ROOT_LOGGER_NAME,
    *,
    run_id: str | None = None,
    symbol: str | None = None,
    **kwargs: Any,
) -> logging.Logger:
    """Get a backtester logger with default settings and contextual metadata."""
    root_logger = BacktesterLogger.get_logger(ROOT_LOGGER_NAME, **kwargs)
    normalized_name = _normalized_logger_name(name)
    if normalized_name == root_logger.name:
        target_logger = root_logger
    else:
        relative_name = normalized_name.split(f"{ROOT_LOGGER_NAME}.", 1)[1]
        target_logger = root_logger.getChild(relative_name)
        _ensure_default_context(target_logger)
    if run_id is not None or symbol is not None:
        return bind_logger_context(target_logger, run_id=run_id, symbol=symbol)
    return target_logger


class BacktesterLogger:
    """Backtester-specific logger class for compatibility with tests."""

    _loggers: dict[str, logging.Logger] = {}

    def __init__(
        self,
        name: str = "backtester",
        level: str = "INFO",
        format: str = "STANDARD",
        log_file: str | None = None,
    ) -> None:
        """Initialize the backtester logger.

        Args:
            name: Logger name
            level: Log level
            format: Log format
            log_file: Optional log file path
        """
        self.name = name
        self.level = str(level)
        self.format = str(format)
        self.log_file = log_file
        self._logger: logging.Logger | None = None

    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        if not isinstance(level, LogLevel):
            raise ValueError("Invalid log level")
        self.level = level
        if self._logger:
            self._logger.setLevel(getattr(logging, level.value))

    def set_format(self, format: LogFormat) -> None:
        """Set the log format."""
        self.format = format

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = "INFO",
        file_path: str | None = None,
        max_file_size: int = 10485760,  # 10MB
        backup_count: int = 5,
        console: bool = True,
        structured: bool = False,
    ) -> logging.Logger:
        """Get or create a logger with the specified configuration."""
        if name in cls._loggers:
            return cls._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        if structured:
            formatter: logging.Formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s symbol=%(symbol)s | '
                '%(module)s:%(funcName)s:%(lineno)d | %(message)s'
            )

        # Add console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            _ensure_default_context(console_handler)

        # Add file handler if specified
        if file_path:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            _ensure_default_context(file_handler)

        _ensure_default_context(logger)

        # Store logger
        cls._loggers[name] = logger
        return logger

    def _get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        if self._logger is None:
            self._logger = self._create_logger_instance()
        # Type cast to satisfy MyPy - logger is guaranteed to be non-None
        return self._logger

    def _create_logger_instance(self) -> logging.Logger:
        """Create a new logger instance (for test compatibility)."""
        return BacktesterLogger.get_logger(self.name)

    def info(self, message: str, extra_data: dict[str, Any] | None = None) -> None:
        """Log info message."""
        logger = self._get_logger()
        if extra_data:
            logger.info(f"{message} | Extra: {extra_data}")
        else:
            logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._get_logger().warning(message)

    def error(self, message: str, exception: Exception | None = None) -> None:
        """Log error message."""
        logger = self._get_logger()
        if exception:
            logger.error(f"{message} | Exception: {exception}")
        else:
            logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._get_logger().debug(message)

    def log_performance(self, performance_data: dict[str, Any]) -> None:
        """Log performance metrics."""
        self.info(f"Performance: {performance_data}")

    def log_trade(self, trade_data: dict[str, Any]) -> None:
        """Log trade execution."""
        self.info(f"Trade: {trade_data}")

    def log_config(self, config_data: dict[str, Any]) -> None:
        """Log configuration data."""
        self.info(f"Config: {config_data}")

    def log_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Log performance metrics."""
        self.info(f"Performance Metrics: {metrics}")

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Context manager exit."""
        pass
