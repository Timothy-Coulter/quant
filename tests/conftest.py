"""Pytest configuration and shared fixtures for the backtester test suite.

This module provides pytest configuration, fixtures, and shared test utilities
that are used across all test modules.
"""

import warnings
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from backtester.core.config import (
    BacktesterConfig,
    DataRetrievalConfig,
    ExecutionConfig,
    PerformanceConfig,
    PortfolioConfig,
    StrategyConfig,
)

# Pytest configuration
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"findatapy\.market\.marketdatarequest",
)


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance suites")


@pytest.fixture(scope="session")
def test_data() -> pd.DataFrame:
    """Generate sample test data for use in tests."""
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)  # For reproducible tests

    # Generate realistic OHLCV data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = initial_price * np.cumprod(1 + returns)

    # Create OHLCV data
    data = pd.DataFrame(
        {
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


@pytest.fixture
def mock_config() -> BacktesterConfig:
    """Concrete configuration object for testing."""
    data_config = DataRetrievalConfig(
        data_source="yahoo",
        tickers=["SPY"],
        start_date="2020-01-01",
        finish_date="2024-01-01",
        freq="daily",
    )
    strategy_config = StrategyConfig(
        strategy_name="momentum_strategy",
        ma_short=20,
        ma_long=50,
        leverage_base=2.0,
        leverage_alpha=3.0,
        base_to_alpha_split=0.2,
        alpha_to_base_split=0.2,
        stop_loss_base=0.025,
        stop_loss_alpha=0.025,
        take_profit_target=0.10,
    )
    portfolio_config = PortfolioConfig(
        portfolio_strategy_name="kelly_criterion",
        initial_capital=1000.0,
        commission_rate=0.001,
        maintenance_margin=0.25,
        interest_rate_daily=0.0001,
        spread_rate=0.0005,
        slippage_std=0.001,
        funding_enabled=True,
        tax_rate=0.15,
        max_positions=5,
    )
    execution_config = ExecutionConfig(
        commission_rate=0.001,
        min_commission=1.0,
        spread=0.0001,
        slippage_model="normal",
        slippage_std=0.0005,
        latency_ms=0.0,
    )
    performance_config = PerformanceConfig(
        risk_free_rate=0.02,
        benchmark_enabled=False,
        benchmark_symbol="SPY",
    )

    return BacktesterConfig(
        data=data_config,
        strategy=strategy_config,
        portfolio=portfolio_config,
        execution=execution_config,
        performance=performance_config,
    )


@pytest.fixture
def sample_portfolio_state() -> Mock:
    """Sample portfolio state for testing."""
    state = Mock()
    state.total_value = 1100.0
    state.cash = 100.0
    state.positions = {}
    state.leverage_base = 2.0
    state.leverage_alpha = 3.0
    state.base_pool_value = 600.0
    state.alpha_pool_value = 400.0
    return state


@pytest.fixture
def mock_broker() -> Mock:
    """Mock broker for testing order execution."""
    broker = Mock()
    broker.execute_order = Mock(
        return_value={
            'success': True,
            'order_id': 'test_order_123',
            'fill_price': 100.0,
            'fill_quantity': 10,
        }
    )
    broker.get_account_info = Mock(
        return_value={'cash': 1000.0, 'buying_power': 2000.0, 'positions': {}}
    )
    return broker


@pytest.fixture
def mock_logger() -> Mock:
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def performance_metrics() -> dict[str, Any]:
    """Sample performance metrics for testing."""
    return {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.25,
        'max_drawdown': -0.08,
        'volatility': 0.18,
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'total_trades': 50,
        'winning_trades': 32,
        'losing_trades': 18,
    }


@pytest.fixture
def strategy_signals() -> pd.DataFrame:
    """Sample strategy signals for testing."""
    return pd.DataFrame(
        {
            'signal': [0, 1, -1, 0, 1, 0, -1, 1],
            'confidence': [0.5, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6, 0.85],
            'strength': [0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0],
        },
        index=pd.date_range('2020-01-01', periods=8, freq='D'),
    )


# Utility functions for tests
def assert_valid_dataframe(df: pd.DataFrame, expected_columns: list[str] | None = None) -> None:
    """Assert that a DataFrame is valid and optionally check columns."""
    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    if expected_columns:
        assert (
            list(df.columns) == expected_columns
        ), f"Expected columns {expected_columns}, got {list(df.columns)}"
    assert df.index.is_monotonic_increasing, "DataFrame index should be sorted"


def create_test_order(
    ticker: str = 'TEST', side: str = 'buy', quantity: int = 10, order_type: str = 'market'
) -> dict[str, Any]:
    """Create a test order for use in tests."""
    return {
        'ticker': ticker,
        'side': side,
        'quantity': quantity,
        'order_type': order_type,
        'timestamp': datetime.now(),
        'status': 'pending',
    }


# Mark slow tests
pytestmark = pytest.mark.slow
