"""Comprehensive unit tests for the RiskControlManager class.

This module contains tests for the risk control manager functionality including
position initialization, risk control updates, position sizing, portfolio risk checking,
and all edge cases.
"""

import logging
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from backtester.risk_management.component_configs.comprehensive_risk_config import (
    ComprehensiveRiskConfig,
)
from backtester.risk_management.component_configs.position_sizing_config import (
    PositionSizingConfig,
)
from backtester.risk_management.component_configs.risk_limit_config import (
    RiskLimitConfig,
)
from backtester.risk_management.component_configs.risk_monitoring_config import (
    RiskMonitoringConfig,
)
from backtester.risk_management.component_configs.stop_loss_config import (
    StopLossConfig,
    StopLossType,
)
from backtester.risk_management.component_configs.take_profit_config import (
    TakeProfitConfig,
    TakeProfitType,
)
from backtester.risk_management.risk_control_manager import RiskControlManager


class TestComprehensiveRiskConfig:
    """Test suite for ComprehensiveRiskConfig validation and behavior."""

    def test_init_default_config(self) -> None:
        """Test ComprehensiveRiskConfig initialization with defaults."""
        config = ComprehensiveRiskConfig()

        # Check default legacy fields
        assert config.max_portfolio_risk == 0.02
        assert config.max_position_size == 0.10
        assert config.max_leverage == 5.0
        assert config.max_drawdown == 0.20
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.06
        assert config.max_daily_loss == 0.05
        assert config.volatility_threshold == 0.03
        assert config.correlation_limit == 0.7

        # Check default advanced settings
        assert config.enable_dynamic_hedging is False
        assert config.stress_test_frequency == "monthly"
        assert config.rebalance_frequency == "weekly"
        assert config.risk_attribution_enabled is True
        assert config.factor_analysis_enabled is False

        # Check initialized component configs
        assert config.stop_loss_config is not None
        assert config.take_profit_config is not None
        assert config.position_sizing_config is not None
        assert config.risk_limits_config is not None
        assert config.risk_monitoring_config is not None

    def test_init_custom_config(self) -> None:
        """Test ComprehensiveRiskConfig initialization with custom values."""
        custom_stop_loss = StopLossConfig(stop_loss_value=0.03)
        custom_take_profit = TakeProfitConfig(take_profit_value=0.08)

        config = ComprehensiveRiskConfig(
            max_position_size=0.15,
            max_leverage=3.0,
            enable_dynamic_hedging=True,
            stress_test_frequency="weekly",
            stop_loss_config=custom_stop_loss,
            take_profit_config=custom_take_profit,
        )

        assert config.max_position_size == 0.15
        assert config.max_leverage == 3.0
        assert config.enable_dynamic_hedging is True
        assert config.stress_test_frequency == "weekly"
        assert config.stop_loss_config == custom_stop_loss
        assert config.take_profit_config == custom_take_profit

    def test_get_effective_risk_limits_with_config(self) -> None:
        """Test get_effective_risk_limits with risk limits config."""
        risk_limits_config = RiskLimitConfig(
            max_drawdown=0.15,
            max_leverage=4.0,
            max_single_position=0.08,
        )
        config = ComprehensiveRiskConfig(risk_limits_config=risk_limits_config)

        limits = config.get_effective_risk_limits()

        assert limits['max_drawdown'] == 0.15
        assert limits['max_leverage'] == 4.0
        assert limits['max_single_position'] == 0.08
        assert 'max_portfolio_var' in limits
        assert 'max_daily_loss' in limits
        assert 'max_correlation' in limits

    def test_get_effective_risk_limits_fallback(self) -> None:
        """Test get_effective_risk_limits fallback to legacy fields."""
        config = ComprehensiveRiskConfig(
            max_drawdown=0.18,
            max_leverage=6.0,
            max_position_size=0.12,
            max_daily_loss=0.04,
            correlation_limit=0.8,
        )

        limits = config.get_effective_risk_limits()

        assert limits['max_drawdown'] == 0.18
        assert limits['max_leverage'] == 6.0
        assert limits['max_single_position'] == 0.12
        assert limits['max_daily_loss'] == 0.04
        assert limits['max_correlation'] == 0.8

    def test_get_position_sizing_params_with_config(self) -> None:
        """Test get_position_sizing_params with position sizing config."""
        position_config = PositionSizingConfig(
            max_position_size=0.12,
            min_position_size=0.02,
            risk_per_trade=0.03,
        )
        config = ComprehensiveRiskConfig(position_sizing_config=position_config)

        params = config.get_position_sizing_params()

        assert params['max_position_size'] == 0.12
        assert params['min_position_size'] == 0.02
        assert params['risk_per_trade'] == 0.03
        assert 'sizing_method' in params
        assert 'volatility_adjustment' in params

    def test_get_position_sizing_params_fallback(self) -> None:
        """Test get_position_sizing_params fallback to legacy fields."""
        config = ComprehensiveRiskConfig(max_position_size=0.15)

        params = config.get_position_sizing_params()

        assert params['max_position_size'] == 0.15
        assert params['min_position_size'] == 0.01
        assert params['risk_per_trade'] == 0.02
        assert params['sizing_method'] == 'fixed_percentage'
        assert params['volatility_adjustment'] is True

    def test_get_stop_loss_params_with_config(self) -> None:
        """Test get_stop_loss_params with stop loss config."""
        stop_loss_config = StopLossConfig(
            stop_loss_type=StopLossType.TRAILING,
            stop_loss_value=0.025,
        )
        config = ComprehensiveRiskConfig(stop_loss_config=stop_loss_config)

        params = config.get_stop_loss_params()

        assert params['stop_loss_type'] == "TRAILING"
        assert params['stop_loss_value'] == 0.025
        assert 'trail_distance' in params
        assert 'trail_step' in params

    def test_get_stop_loss_params_fallback(self) -> None:
        """Test get_stop_loss_params fallback to legacy fields."""
        config = ComprehensiveRiskConfig(stop_loss_pct=0.025)

        params = config.get_stop_loss_params()

        assert params['stop_loss_type'] == "PERCENTAGE"
        assert params['stop_loss_value'] == 0.025
        assert 'trail_distance' in params
        assert 'trail_step' in params

    def test_get_take_profit_params_with_config(self) -> None:
        """Test get_take_profit_params with take profit config."""
        take_profit_config = TakeProfitConfig(
            take_profit_type=TakeProfitType.FIXED,
            take_profit_value=0.08,
        )
        config = ComprehensiveRiskConfig(take_profit_config=take_profit_config)

        params = config.get_take_profit_params()

        assert params['take_profit_type'] == "FIXED"
        assert params['take_profit_value'] == 0.08
        assert 'trail_distance' in params
        assert 'trail_step' in params

    def test_get_take_profit_params_fallback(self) -> None:
        """Test get_take_profit_params fallback to legacy fields."""
        config = ComprehensiveRiskConfig(take_profit_pct=0.08)

        params = config.get_take_profit_params()

        assert params['take_profit_type'] == "PERCENTAGE"
        assert params['take_profit_value'] == 0.08
        assert 'trail_distance' in params
        assert 'trail_step' in params

    def test_get_monitoring_params_with_config(self) -> None:
        """Test get_monitoring_params with monitoring config."""
        monitoring_config = RiskMonitoringConfig(
            check_interval=120,
            enable_real_time_alerts=False,
        )
        config = ComprehensiveRiskConfig(risk_monitoring_config=monitoring_config)

        params = config.get_monitoring_params()

        assert params['check_interval'] == 120
        assert params['enable_real_time_alerts'] is False
        assert 'volatility_threshold' in params
        assert 'drawdown_threshold' in params

    def test_get_monitoring_params_fallback(self) -> None:
        """Test get_monitoring_params fallback to defaults."""
        config = ComprehensiveRiskConfig(volatility_threshold=0.04)

        params = config.get_monitoring_params()

        assert params['check_interval'] == 60
        assert params['enable_real_time_alerts'] is True
        assert params['volatility_threshold'] == 0.04
        assert 'drawdown_threshold' in params


class TestRiskControlManager:
    """Test suite for the RiskControlManager class."""

    @pytest.fixture
    def default_config(self) -> ComprehensiveRiskConfig:
        """Create default ComprehensiveRiskConfig for testing."""
        return ComprehensiveRiskConfig()

    @pytest.fixture
    def custom_config(self) -> ComprehensiveRiskConfig:
        """Create custom ComprehensiveRiskConfig for testing."""
        return ComprehensiveRiskConfig(
            max_position_size=0.15,
            max_leverage=3.0,
            enable_dynamic_hedging=True,
        )

    @pytest.fixture
    def minimal_config(self) -> ComprehensiveRiskConfig:
        """Create minimal ComprehensiveRiskConfig for testing."""
        return ComprehensiveRiskConfig(
            stop_loss_config=None,
            take_profit_config=None,
            position_sizing_config=None,
            risk_limits_config=None,
            risk_monitoring_config=None,
        )

    @pytest.fixture
    def mock_logger(self) -> Mock:
        """Create mock logger for testing."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def sample_timestamp(self) -> pd.Timestamp:
        """Create sample timestamp for testing."""
        return pd.Timestamp("2023-01-01 12:00:00")

    def test_init_default(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test RiskControlManager initialization with default config."""
        manager = RiskControlManager()

        assert manager.config == default_config
        assert manager.logger is not None
        assert isinstance(manager.logger, logging.Logger)
        assert manager.current_positions == {}
        assert manager.risk_signals_history == []

        # Check that all components are initialized
        assert manager.stop_loss is not None
        assert manager.take_profit is not None
        assert manager.position_sizer is not None
        assert manager.risk_limits is not None
        assert manager.risk_monitor is not None

    def test_init_custom_config(
        self, custom_config: ComprehensiveRiskConfig, mock_logger: Mock
    ) -> None:
        """Test RiskControlManager initialization with custom config and logger."""
        manager = RiskControlManager(config=custom_config, logger=mock_logger)

        assert manager.config == custom_config
        assert manager.logger == mock_logger
        assert manager.config.max_leverage == 3.0
        assert manager.config.enable_dynamic_hedging is True

    def test_init_minimal_config(self, minimal_config: ComprehensiveRiskConfig) -> None:
        """Test RiskControlManager initialization with minimal config."""
        manager = RiskControlManager(config=minimal_config)

        # Components should be None when not configured
        assert manager.stop_loss is None
        assert manager.take_profit is None
        assert manager.position_sizer is None
        assert manager.risk_limits is None
        assert manager.risk_monitor is None

    def test_initialize_position(
        self, default_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test position initialization."""
        manager = RiskControlManager(config=default_config)

        symbol = "AAPL"
        entry_price = 150.0
        quantity = 100.0

        manager.initialize_position(symbol, entry_price, quantity, sample_timestamp)

        assert symbol in manager.current_positions
        position = manager.current_positions[symbol]
        assert position['symbol'] == symbol
        assert position['entry_price'] == entry_price
        assert position['quantity'] == quantity
        assert position['entry_timestamp'] == sample_timestamp

    def test_update_position_not_found(
        self, default_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_position when position not found."""
        manager = RiskControlManager(config=default_config)

        result = manager.update_position("NONEXISTENT", 155.0, sample_timestamp)

        assert result['triggered'] is False
        assert result['action'] == 'NONE'
        assert result['reason'] == 'Position not found'
        assert result['exit_price'] is None

    def test_update_position_no_components(
        self, minimal_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test update_position with minimal config (no components)."""
        manager = RiskControlManager(config=minimal_config)

        # Initialize position first
        manager.initialize_position("AAPL", 150.0, 100.0, sample_timestamp)

        # Update with no risk control components
        result = manager.update_position("AAPL", 155.0, sample_timestamp)

        assert result['triggered'] is False
        assert result['action'] == 'NONE'
        assert result['reason'] == 'No risk controls triggered'
        assert result['stop_loss'] is None
        assert result['take_profit'] is None

    def test_calculate_position_size_with_position_sizer(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test calculate_position_size with position sizer."""
        manager = RiskControlManager(config=default_config)

        symbol = "AAPL"
        account_value = 100000.0
        entry_price = 150.0
        stop_price = 145.0
        volatility = 0.02
        conviction = 1.2

        position_size, risk_amount = manager.calculate_position_size(
            symbol, account_value, entry_price, stop_price, volatility, conviction
        )

        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)
        assert position_size > 0
        assert risk_amount >= 0

    def test_calculate_position_size_without_stop_price(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test calculate_position_size without stop price."""
        manager = RiskControlManager(config=default_config)

        symbol = "AAPL"
        account_value = 100000.0
        entry_price = 150.0
        volatility = 0.02
        conviction = 1.2

        position_size, risk_amount = manager.calculate_position_size(
            symbol, account_value, entry_price, volatility=volatility, conviction=conviction
        )

        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)
        assert position_size >= 0
        assert risk_amount >= 0

    def test_calculate_position_size_fallback(
        self, minimal_config: ComprehensiveRiskConfig
    ) -> None:
        """Test calculate_position_size fallback when no position sizer."""
        manager = RiskControlManager(config=minimal_config)

        symbol = "AAPL"
        account_value = 100000.0
        entry_price = 150.0

        position_size, risk_amount = manager.calculate_position_size(
            symbol, account_value, entry_price
        )

        # Should use fallback calculation
        expected_position_size = (account_value * minimal_config.max_position_size) / entry_price
        assert abs(position_size - expected_position_size) < 0.01
        assert risk_amount == 0.0

    def test_can_open_position_respects_limits(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Risk manager should block trades that exceed max position size."""
        manager = RiskControlManager(config=default_config)
        assert manager.can_open_position("AAPL", 500.0, 10000.0) is True

        # Default max position size is 10% so $2,000 on $10k should be rejected.
        assert manager.can_open_position("AAPL", 2000.0, 10000.0) is False
        assert manager.risk_signals_history, "Risk signal should be recorded on violation"

    def test_record_order_and_fill_updates_state(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """record_order and record_fill should keep internal tracking in sync."""
        manager = RiskControlManager(config=default_config)

        manager.record_order("AAPL", "BUY", 10.0, 100.0, {'source': 'test'})
        assert len(manager.order_history) == 1

        manager.record_fill("AAPL", "BUY", 10.0, 100.0, portfolio_value=10000.0)
        assert "AAPL" in manager.current_positions
        assert manager.current_leverage > 0

        manager.record_fill("AAPL", "SELL", 10.0, 110.0, portfolio_value=10000.0)
        assert "AAPL" not in manager.current_positions

    def test_check_portfolio_risk_empty_positions(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test check_portfolio_risk with empty positions."""
        manager = RiskControlManager(config=default_config)

        portfolio_value = 100000.0
        positions: dict[str, Any] = {}

        result = manager.check_portfolio_risk(portfolio_value, positions)

        assert result['portfolio_value'] == portfolio_value
        assert result['risk_level'] == 'LOW'
        assert result['violations'] == []
        assert result['recommendations'] == ['Risk levels within acceptable range']

    def test_check_portfolio_risk_with_violations(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test check_portfolio_risk with position size violations."""
        manager = RiskControlManager(config=default_config)

        portfolio_value = 100000.0
        positions = {
            "AAPL": {
                'market_value': 20000.0,  # 20% of portfolio, exceeds 10% limit
                'active': True,
            }
        }

        result = manager.check_portfolio_risk(portfolio_value, positions)

        assert result['risk_level'] == 'HIGH'
        assert len(result['violations']) > 0
        assert "Max position size exceeded" in result['violations'][0]
        assert len(result['recommendations']) > 0

    def test_check_portfolio_risk_with_leverage_violation(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test check_portfolio_risk with leverage violation."""
        manager = RiskControlManager(config=default_config)
        # Set a low leverage value to test the violation
        manager.config.max_leverage = 2.0

        portfolio_value = 100000.0
        # Create positions that exceed leverage (600k exposure / 100k portfolio = 6x > 2x max)
        positions: dict[str, Any] = {
            "AAPL": {
                'market_value': 300000.0,  # 3x leverage
                'active': True,
            },
            "MSFT": {
                'market_value': 300000.0,  # 3x leverage
                'active': True,
            },
        }

        result = manager.check_portfolio_risk(portfolio_value, positions)

        assert result['risk_level'] == 'HIGH'
        assert any("Max leverage exceeded" in violation for violation in result['violations'])

    def test_get_position_risk_status(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test get_position_risk_status."""
        manager = RiskControlManager(config=default_config)

        status = manager.get_position_risk_status("AAPL")

        assert status['symbol'] == "AAPL"
        assert 'has_stop_loss' in status
        assert 'has_take_profit' in status
        assert 'stop_loss_status' in status
        assert 'take_profit_status' in status

    def test_get_portfolio_risk_summary(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test get_portfolio_risk_summary."""
        manager = RiskControlManager(config=default_config)

        # Add some positions
        manager.current_positions = {
            "AAPL": {'symbol': 'AAPL'},
            "MSFT": {'symbol': 'MSFT'},
        }

        summary = manager.get_portfolio_risk_summary()

        assert summary['total_positions'] == 2
        assert 'positions_with_stop_loss' in summary
        assert 'positions_with_take_profit' in summary
        assert 'risk_monitor_active' in summary
        assert 'risk_limits_configured' in summary
        assert summary['risk_monitor_active'] is True
        assert summary['risk_limits_configured'] is True

    def test_get_portfolio_risk_summary_minimal_config(
        self, minimal_config: ComprehensiveRiskConfig
    ) -> None:
        """Test get_portfolio_risk_summary with minimal config."""
        manager = RiskControlManager(config=minimal_config)

        manager.current_positions = {"AAPL": {'symbol': 'AAPL'}}

        summary = manager.get_portfolio_risk_summary()

        assert summary['total_positions'] == 1
        assert summary['risk_monitor_active'] is False
        assert summary['risk_limits_configured'] is False

    def test_reset_position(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test reset_position."""
        manager = RiskControlManager(config=default_config)

        # Add a position
        manager.current_positions["AAPL"] = {'symbol': 'AAPL'}

        # Reset the position
        manager.reset_position("AAPL")

        assert "AAPL" not in manager.current_positions

    def test_reset_position_not_found(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test reset_position when position not found."""
        manager = RiskControlManager(config=default_config)

        # Should not raise an error
        manager.reset_position("NONEXISTENT")

        assert len(manager.current_positions) == 0

    def test_reset_all_positions(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test reset_all_positions."""
        manager = RiskControlManager(config=default_config)

        # Add some positions
        manager.current_positions = {
            "AAPL": {'symbol': 'AAPL'},
            "MSFT": {'symbol': 'MSFT'},
        }

        # Reset all positions
        manager.reset_all_positions()

        assert len(manager.current_positions) == 0

    def test_update_risk_limits(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test update_risk_limits."""
        manager = RiskControlManager(config=default_config)

        positions: dict[str, Any] = {"AAPL": {'symbol': 'AAPL'}}
        sector_mapping = {"AAPL": "Technology"}

        # Should not raise an error even if risk_limits is None or mocked
        manager.update_risk_limits(positions, sector_mapping)

    def test_update_risk_limits_minimal_config(
        self, minimal_config: ComprehensiveRiskConfig
    ) -> None:
        """Test update_risk_limits with minimal config."""
        manager = RiskControlManager(config=minimal_config)

        positions: dict[str, Any] = {"AAPL": {'symbol': 'AAPL'}}

        # Should handle None risk_limits gracefully
        manager.update_risk_limits(positions)

    def test_process_portfolio_update(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test process_portfolio_update."""
        manager = RiskControlManager(config=default_config)

        update = {'portfolio_value': 100000.0, 'positions': []}

        result = manager.process_portfolio_update(update)

        # Should handle the update gracefully (may return None if no monitor)
        assert result is None or isinstance(result, dict)

    def test_process_portfolio_update_minimal_config(
        self, minimal_config: ComprehensiveRiskConfig
    ) -> None:
        """Test process_portfolio_update with minimal config."""
        manager = RiskControlManager(config=minimal_config)

        update = {'portfolio_value': 100000.0, 'positions': []}

        result = manager.process_portfolio_update(update)

        # Should return None when no risk monitor
        assert result is None

    def test_add_risk_signal(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test add_risk_signal."""
        manager = RiskControlManager(config=default_config)

        initial_count = len(manager.risk_signals_history)

        signal = {
            'action': 'STOP_LOSS',
            'reason': 'Price dropped below stop level',
            'symbol': 'AAPL',
        }

        manager.add_risk_signal(signal)

        assert len(manager.risk_signals_history) == initial_count + 1
        last_signal = manager.risk_signals_history[-1]
        assert 'timestamp' in last_signal
        assert last_signal['signal'] == signal

    def test_add_risk_signal_multiple(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test add_risk_signal with multiple signals."""
        manager = RiskControlManager(config=default_config)

        signals = [
            {'action': 'STOP_LOSS', 'reason': 'Test 1'},
            {'action': 'TAKE_PROFIT', 'reason': 'Test 2'},
            {'action': 'RISK_LIMIT', 'reason': 'Test 3'},
        ]

        for signal in signals:
            manager.add_risk_signal(signal)

        assert len(manager.risk_signals_history) == 3
        for i, signal in enumerate(signals):
            assert manager.risk_signals_history[i]['signal'] == signal

    def test_get_risk_signals_summary_empty(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test get_risk_signals_summary with empty history."""
        manager = RiskControlManager(config=default_config)

        summary = manager.get_risk_signals_summary()

        assert summary['total_signals'] == 0
        assert summary['recent_signals'] == []

    def test_get_risk_signals_summary_with_signals(
        self, default_config: ComprehensiveRiskConfig
    ) -> None:
        """Test get_risk_signals_summary with signals in history."""
        manager = RiskControlManager(config=default_config)

        # Add multiple signals
        for i in range(15):
            manager.add_risk_signal({'action': f'TEST_{i}', 'reason': f'Reason {i}'})

        summary = manager.get_risk_signals_summary()

        assert summary['total_signals'] == 15
        assert len(summary['recent_signals']) == 10  # Should return last 10
        assert all('timestamp' in signal for signal in summary['recent_signals'])
        assert all('signal' in signal for signal in summary['recent_signals'])

    def test_edge_case_zero_account_value(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test calculate_position_size with zero account value."""
        manager = RiskControlManager(config=default_config)

        position_size, risk_amount = manager.calculate_position_size("AAPL", 0.0, 150.0)

        # Should handle zero account value gracefully
        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)

    def test_edge_case_negative_values(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test with negative input values."""
        manager = RiskControlManager(config=default_config)

        # Test negative account value
        position_size, risk_amount = manager.calculate_position_size("AAPL", -1000.0, 150.0)

        # Should handle negative values gracefully
        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)

    def test_edge_case_extreme_values(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test with extreme input values."""
        manager = RiskControlManager(config=default_config)

        # Test very large values
        position_size, risk_amount = manager.calculate_position_size("AAPL", 1e12, 1e6)

        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)
        assert position_size >= 0

        # Test very small values
        position_size_small, risk_amount_small = manager.calculate_position_size(
            "AAPL", 1e-6, 1e-10
        )

        assert isinstance(position_size_small, float)
        assert isinstance(risk_amount_small, float)

    def test_multiple_positions_scenario(
        self, default_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test scenario with multiple positions."""
        manager = RiskControlManager(config=default_config)

        # Initialize multiple positions
        positions = [
            ("AAPL", 150.0, 100.0),
            ("MSFT", 300.0, 50.0),
            ("GOOGL", 2500.0, 10.0),
        ]

        for symbol, price, quantity in positions:
            manager.initialize_position(symbol, price, quantity, sample_timestamp)

        assert len(manager.current_positions) == 3
        assert all(symbol in manager.current_positions for symbol, _, _ in positions)

        # Test portfolio risk with multiple positions
        portfolio_value = 200000.0
        positions_dict = {
            symbol: {'market_value': price * quantity, 'active': True}
            for symbol, price, quantity in positions
        }

        result = manager.check_portfolio_risk(portfolio_value, positions_dict)
        assert 'risk_level' in result
        assert 'violations' in result

    def test_integration_scenario_complete_workflow(
        self, default_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test complete workflow from initialization to risk monitoring."""
        manager = RiskControlManager(config=default_config)

        # 1. Initialize position
        symbol = "AAPL"
        entry_price = 150.0
        quantity = 100.0
        manager.initialize_position(symbol, entry_price, quantity, sample_timestamp)

        assert symbol in manager.current_positions

        # 2. Calculate position size
        account_value = 100000.0
        stop_price = 145.0
        position_size, risk_amount = manager.calculate_position_size(
            symbol, account_value, entry_price, stop_price
        )

        assert isinstance(position_size, float)
        assert isinstance(risk_amount, float)

        # 3. Update position (mock scenario)
        current_price = 155.0
        update_result = manager.update_position(symbol, current_price, sample_timestamp)

        assert 'triggered' in update_result
        assert 'action' in update_result

        # 4. Check position risk status
        risk_status = manager.get_position_risk_status(symbol)
        assert 'has_stop_loss' in risk_status
        assert 'has_take_profit' in risk_status

        # 5. Add risk signal
        signal = {'action': 'POSITION_UPDATE', 'reason': 'Price update'}
        manager.add_risk_signal(signal)
        assert len(manager.risk_signals_history) == 1

        # 6. Get portfolio risk summary
        portfolio_summary = manager.get_portfolio_risk_summary()
        assert portfolio_summary['total_positions'] == 1

        # 7. Get risk signals summary
        signals_summary = manager.get_risk_signals_summary()
        assert signals_summary['total_signals'] == 1

    def test_stress_test_scenario(
        self, default_config: ComprehensiveRiskConfig, sample_timestamp: pd.Timestamp
    ) -> None:
        """Test stress scenario with many operations."""
        manager = RiskControlManager(config=default_config)

        # Add many positions
        num_positions = 50
        for i in range(num_positions):
            symbol = f"TEST_{i:03d}"
            price = 100.0 + i
            quantity = 10.0 + i
            manager.initialize_position(symbol, price, quantity, sample_timestamp)

        assert len(manager.current_positions) == num_positions

        # Add many risk signals
        num_signals = 100
        for i in range(num_signals):
            signal = {'action': f'SIGNAL_{i}', 'reason': f'Reason {i}'}
            manager.add_risk_signal(signal)

        assert len(manager.risk_signals_history) == num_signals

        # Check portfolio risk with many positions
        portfolio_value = 1000000.0
        positions_dict = {
            f"TEST_{i:03d}": {
                'market_value': (100.0 + i) * (10.0 + i),
                'active': True,
            }
            for i in range(num_positions)
        }

        result = manager.check_portfolio_risk(portfolio_value, positions_dict)
        assert 'risk_level' in result

        # Get summaries
        portfolio_summary = manager.get_portfolio_risk_summary()
        assert portfolio_summary['total_positions'] == num_positions

        signals_summary = manager.get_risk_signals_summary()
        assert signals_summary['total_signals'] == num_signals

    def test_configuration_edge_cases(self) -> None:
        """Test edge cases in configuration handling."""
        # Test with None config
        manager1 = RiskControlManager(config=None)
        assert manager1.config is not None  # Should create default config

    def test_logger_integration(self, mock_logger: Mock) -> None:
        """Test logger integration and logging calls."""
        config = ComprehensiveRiskConfig()
        manager = RiskControlManager(config=config, logger=mock_logger)

        # Test that methods that should log are called
        # Note: This would require more sophisticated mocking to verify actual log calls
        symbol = "AAPL"
        entry_price = 150.0
        quantity = 100.0
        timestamp = pd.Timestamp("2023-01-01")

        manager.initialize_position(symbol, entry_price, quantity, timestamp)
        manager.reset_position(symbol)

    def test_backwards_compatibility(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test that legacy field access still works."""
        manager = RiskControlManager(config=default_config)

        # Test accessing legacy fields
        assert hasattr(manager.config, 'max_position_size')
        assert hasattr(manager.config, 'max_leverage')
        assert hasattr(manager.config, 'max_drawdown')
        assert manager.config.max_position_size == 0.10
        assert manager.config.max_leverage == 5.0

    def test_state_consistency(self, default_config: ComprehensiveRiskConfig) -> None:
        """Test that manager maintains consistent state."""
        manager = RiskControlManager(config=default_config)

        # Initialize and then reset
        symbol = "AAPL"
        manager.initialize_position(symbol, 150.0, 100.0, pd.Timestamp.now())
        assert symbol in manager.current_positions

        manager.reset_position(symbol)
        assert symbol not in manager.current_positions

        # Add signal and check history
        signal = {'action': 'TEST', 'reason': 'Test signal'}
        manager.add_risk_signal(signal)
        assert len(manager.risk_signals_history) == 1

        # Reset all and check
        manager.reset_all_positions()
        assert len(manager.current_positions) == 0
        # History should remain intact
        assert len(manager.risk_signals_history) == 1


if __name__ == "__main__":
    pytest.main([__file__])
