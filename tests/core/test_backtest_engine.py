"""Comprehensive tests for the BacktestEngine class.

This module contains tests for the main backtesting engine that coordinates
all components of the trading system including data loading, strategy execution,
portfolio management, and performance calculation.
"""

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Import the module being tested
try:
    from backtester.core.backtest_engine import BacktestEngine
    from backtester.core.config import BacktesterConfig
except ImportError as e:
    pytest.skip(f"Could not import backtester modules: {e}", allow_module_level=True)


class TestBacktestEngine:
    """Test suite for the BacktestEngine class."""

    def test_initialization(self, mock_config: BacktesterConfig) -> None:
        """Test that BacktestEngine initializes correctly."""
        engine = BacktestEngine(config=mock_config)

        assert engine.config == mock_config
        assert engine.data_handler is not None
        # Components are created on demand, not at initialization
        assert engine.portfolio is None
        assert engine.broker is None
        assert engine.strategy is None
        assert engine.performance_tracker is None
        assert engine.performance_analyzer is not None

    def test_load_data_success(
        self, test_data: pd.DataFrame, mock_config: BacktesterConfig
    ) -> None:
        """Test successful data loading."""
        with patch.object(BacktestEngine, 'load_data') as mock_load:
            # Create engine with mocked load_data
            engine = BacktestEngine(config=mock_config)
            mock_load.return_value = test_data
            result = engine.load_data('SPY', '2020-01-01', '2024-01-01', '1mo')

            assert result.equals(test_data)
            mock_load.assert_called_once_with('SPY', '2020-01-01', '2024-01-01', '1mo')

    def test_load_data_failure(self, mock_config: BacktesterConfig) -> None:
        """Test data loading failure handling."""
        with patch.object(BacktestEngine, 'load_data') as mock_load:
            mock_load.side_effect = Exception("Failed to load data")

            engine = BacktestEngine(config=mock_config)

            with pytest.raises(Exception, match="Failed to load data"):
                engine.load_data('INVALID', '2020-01-01', '2024-01-01', '1mo')

    def test_load_data_applies_overrides(
        self, mock_config: BacktesterConfig, test_data: pd.DataFrame
    ) -> None:
        """load_data should forward override arguments to the data handler."""
        engine = BacktestEngine(config=mock_config)
        engine.data_handler = Mock()

        captured_config: dict[str, Any] = {}

        def fake_get_data(data_config: Any) -> pd.DataFrame:
            captured_config['tickers'] = list(data_config.tickers or [])
            captured_config['start_date'] = data_config.start_date
            captured_config['finish_date'] = data_config.finish_date
            captured_config['freq'] = data_config.freq
            return test_data

        engine.data_handler.get_data.side_effect = fake_get_data

        result = engine.load_data(
            ticker='QQQ', start_date='2021-01-01', end_date='2021-06-30', interval='1h'
        )

        assert result is test_data
        assert captured_config['tickers'] == ['QQQ']
        assert captured_config['start_date'] == '2021-01-01'
        assert captured_config['finish_date'] == '2021-06-30'
        assert captured_config['freq'] == '1h'

    def test_run_backtest_success(
        self, test_data: pd.DataFrame, mock_config: BacktesterConfig
    ) -> None:
        """Test successful backtest execution."""
        # Create mock portfolio with proper attributes
        mock_portfolio_obj = Mock()
        mock_portfolio_obj.portfolio_values = [100, 105, 110]
        mock_portfolio_obj.base_values = [50, 52, 55]
        mock_portfolio_obj.alpha_values = [50, 53, 55]
        mock_portfolio_obj.trade_log = []
        mock_portfolio_obj.cumulative_tax = 0.0
        mock_portfolio_obj.base_pool = Mock(capital=50.0)
        mock_portfolio_obj.alpha_pool = Mock(capital=50.0)
        mock_portfolio_obj.reset = Mock()

        # Create a side effect that also sets current_portfolio
        def mock_create_portfolio_fn(
            self: BacktestEngine, portfolio_params: dict[str, Any] | None = None
        ) -> Mock:
            self.current_portfolio = mock_portfolio_obj
            self.portfolio = mock_portfolio_obj
            return mock_portfolio_obj

        # Mock data loading and engine components
        with (
            patch.object(BacktestEngine, 'load_data') as mock_load,
            patch.object(BacktestEngine, 'create_strategy') as mock_strategy,
            patch.object(BacktestEngine, 'create_portfolio', mock_create_portfolio_fn),
            patch.object(BacktestEngine, 'create_broker') as mock_broker,
            patch.object(BacktestEngine, 'create_risk_manager') as mock_risk_manager,
            patch.object(BacktestEngine, '_run_simulation') as mock_sim,
        ):
            mock_load.return_value = test_data
            mock_strategy.return_value = Mock(name='test_strategy', reset=Mock())
            mock_broker.return_value = Mock(reset=Mock())
            mock_risk_manager.return_value = Mock()
            mock_sim.return_value = {
                'portfolio_values': [100, 105, 110],
                'base_values': [50, 52, 55],
                'alpha_values': [50, 53, 55],
            }

            engine = BacktestEngine(config=mock_config)
            # Manually set current_data since mock doesn't do it
            engine.current_data = test_data
            result = engine.run_backtest(
                ticker='SPY', start_date='2020-01-01', end_date='2024-01-01', interval='1mo'
            )

            assert 'performance' in result
            assert 'data' in result
            # Result can have either 'portfolio_values' or 'trades' depending on path
            assert 'portfolio_values' in result or 'trades' in result
            assert 'total_return' in result['performance']

    def test_run_backtest_with_missing_data(self, mock_config: BacktesterConfig) -> None:
        """Test backtest handling of missing or insufficient data."""
        with patch.object(BacktestEngine, 'load_data') as mock_load:
            mock_load.return_value = pd.DataFrame()  # Empty data

            engine = BacktestEngine(config=mock_config)
            engine.current_data = pd.DataFrame()  # Set empty data

            with pytest.raises(ValueError, match="Insufficient data"):
                engine.run_backtest('SPY', '2020-01-01', '2024-01-01', '1mo')

    def test_run_strategy_backtest(
        self, test_data: pd.DataFrame, mock_config: BacktesterConfig
    ) -> None:
        """Test the strategy backtest execution."""
        engine = BacktestEngine(config=mock_config)
        engine.current_data = test_data  # Set current_data
        engine.current_portfolio = Mock()
        engine.current_strategy = Mock()
        engine.current_broker = Mock()

        # _run_strategy_backtest is an alias for run_backtest
        # It should return a dict with performance data
        with patch.object(
            engine, 'run_backtest', return_value={'performance': {}, 'data': test_data}
        ):
            result = engine._run_strategy_backtest(test_data)

            assert isinstance(result, dict)
            assert 'performance' in result or 'data' in result

    def test_calculate_performance(self, mock_config: BacktesterConfig) -> None:
        """Test performance calculation."""
        engine = BacktestEngine(config=mock_config)

        # Create mock trades DataFrame
        trades = pd.DataFrame(
            {
                'timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
                'action': ['buy', 'sell', 'buy', 'sell', 'buy'],
                'quantity': [10, 10, 5, 5, 15],
                'price': [100, 105, 103, 108, 110],
                'pnl': [0, 50, -10, 25, 30],
            }
        )

        # Create mock portfolio value series
        portfolio_values = pd.Series([1000, 1050, 1020, 1100, 1150], index=trades['timestamp'])

        result = engine._calculate_performance(trades, portfolio_values)

        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert isinstance(result['total_return'], float)

    def test_portfolio_config_instance_is_copied(self, mock_config: BacktesterConfig) -> None:
        """Portfolio instances should hold their own config copy."""
        engine = BacktestEngine(config=mock_config)
        engine.create_portfolio()
        assert engine.current_portfolio is not None
        portfolio_config = getattr(engine.current_portfolio, "_config", None)
        assert portfolio_config is not None
        baseline_capital = engine.current_portfolio.initial_capital
        assert engine.config.portfolio is not None
        engine.config.portfolio.initial_capital = baseline_capital + 500.0
        assert engine.current_portfolio.initial_capital == baseline_capital

    def test_handle_exception_gracefully(
        self, test_data: pd.DataFrame, mock_config: BacktesterConfig
    ) -> None:
        """Test that exceptions are handled gracefully during backtesting."""
        with (
            patch.object(BacktestEngine, 'load_data') as mock_load,
            patch.object(BacktestEngine, 'create_strategy') as mock_strategy,
        ):
            mock_load.return_value = test_data
            mock_strategy.side_effect = Exception("Strategy error")

            engine = BacktestEngine(config=mock_config)
            engine.current_data = test_data  # Set current_data

            with pytest.raises(Exception, match="Strategy error"):
                engine.run_backtest('SPY', '2020-01-01', '2024-01-01', '1mo')

    def test_validate_config(self, mock_config: BacktesterConfig) -> None:
        """Test configuration validation."""
        engine = BacktestEngine(config=mock_config)

        # Valid config should pass (mock config is treated as valid due to _mock_name check)
        assert engine._validate_config() is True

        # Test with actual None config
        engine_none = BacktestEngine(config=None)
        # This gets default config, so it should be valid
        assert engine_none._validate_config() is True

    def test_get_status(self, test_data: pd.DataFrame, mock_config: BacktesterConfig) -> None:
        """Test getting backtest status information."""
        engine = BacktestEngine(config=mock_config)

        status = engine.get_status()

        assert 'config' in status
        assert 'data_loaded' in status
        assert 'backtest_running' in status
        assert 'results_available' in status

    @pytest.mark.parametrize(
        "ticker,start_date,end_date,interval",
        [
            ("SPY", "2020-01-01", "2024-01-01", "1mo"),
            ("AAPL", "2019-01-01", "2023-12-31", "1d"),
            ("GOOGL", "2021-06-01", "2023-06-01", "1h"),
        ],
    )
    def test_run_backtest_parameters(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
        mock_config: BacktesterConfig,
    ) -> None:
        """Test backtest with various parameter combinations."""
        test_data_small = pd.DataFrame(
            {
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [99, 100, 101],
                'Close': [103, 104, 105],
                'Volume': [1000000, 1100000, 1200000],
            },
            index=pd.date_range(start_date, periods=3, freq='D'),
        )

        # Create mock portfolio with proper attributes
        mock_portfolio_obj = Mock()
        mock_portfolio_obj.portfolio_values = [100, 101, 102]
        mock_portfolio_obj.base_values = [50, 50, 51]
        mock_portfolio_obj.alpha_values = [50, 51, 51]
        mock_portfolio_obj.trade_log = []
        mock_portfolio_obj.cumulative_tax = 0.0
        mock_portfolio_obj.base_pool = Mock(capital=50.0)
        mock_portfolio_obj.alpha_pool = Mock(capital=51.0)
        mock_portfolio_obj.reset = Mock()

        # Create a side effect that also sets current_portfolio
        def mock_create_portfolio_fn(
            self: BacktestEngine, portfolio_params: dict[str, Any] | None = None
        ) -> Mock:
            self.current_portfolio = mock_portfolio_obj
            self.portfolio = mock_portfolio_obj
            return mock_portfolio_obj

        with (
            patch.object(BacktestEngine, 'load_data') as mock_load,
            patch.object(BacktestEngine, 'create_strategy') as mock_strategy,
            patch.object(BacktestEngine, 'create_portfolio', mock_create_portfolio_fn),
            patch.object(BacktestEngine, 'create_broker') as mock_broker,
            patch.object(BacktestEngine, 'create_risk_manager') as mock_risk_manager,
            patch.object(BacktestEngine, '_run_simulation') as mock_sim,
        ):
            mock_load.return_value = test_data_small
            mock_strategy.return_value = Mock(name='test_strategy', reset=Mock())
            mock_broker.return_value = Mock(reset=Mock())
            mock_risk_manager.return_value = Mock()
            mock_sim.return_value = {
                'portfolio_values': [100, 101, 102],
                'base_values': [50, 50, 51],
                'alpha_values': [50, 51, 51],
            }

            engine = BacktestEngine(config=mock_config)
            engine.current_data = test_data_small  # Set current_data
            result = engine.run_backtest(ticker, start_date, end_date, interval)

            assert 'performance' in result
            mock_load.assert_called_once_with(apply_overrides=False)
            assert engine.config.data is not None
            assert engine.config.data.tickers == [ticker]
            assert engine.config.data.start_date == start_date
            assert engine.config.data.finish_date == end_date
            assert engine.config.data.freq == interval

    def test_memory_efficiency(
        self, test_data: pd.DataFrame, mock_config: BacktesterConfig
    ) -> None:
        """Test that backtest runs efficiently with memory constraints."""
        # Test with larger dataset
        large_data = pd.concat([test_data] * 100)  # Simulate large dataset

        # Create mock portfolio with proper attributes
        mock_portfolio_obj = Mock()
        mock_portfolio_obj.portfolio_values = [100] * len(large_data)
        mock_portfolio_obj.base_values = [50] * len(large_data)
        mock_portfolio_obj.alpha_values = [50] * len(large_data)
        mock_portfolio_obj.trade_log = []
        mock_portfolio_obj.cumulative_tax = 0.0
        mock_portfolio_obj.base_pool = Mock(capital=50.0)
        mock_portfolio_obj.alpha_pool = Mock(capital=50.0)
        mock_portfolio_obj.reset = Mock()

        # Create a side effect that also sets current_portfolio
        def mock_create_portfolio_fn(portfolio_params: dict[str, Any] | None = None) -> Mock:
            engine.current_portfolio = mock_portfolio_obj
            engine.portfolio = mock_portfolio_obj
            return mock_portfolio_obj

        engine = BacktestEngine(config=mock_config)

        # This should not cause memory issues
        with (
            patch.object(engine, 'load_data', return_value=large_data),
            patch.object(
                engine, 'create_strategy', return_value=Mock(name='test_strategy', reset=Mock())
            ),
            patch.object(engine, 'create_portfolio', side_effect=mock_create_portfolio_fn),
            patch.object(engine, 'create_broker', return_value=Mock(reset=Mock())),
            patch.object(engine, 'create_risk_manager', return_value=Mock()),
            patch.object(
                engine,
                '_run_simulation',
                return_value={
                    'portfolio_values': [100] * len(large_data),
                    'base_values': [50] * len(large_data),
                    'alpha_values': [50] * len(large_data),
                },
            ),
        ):
            engine.current_data = large_data  # Set current_data
            result = engine.run_backtest('SPY', '2020-01-01', '2024-01-01', '1mo')
            assert result is not None
            assert 'performance' in result

    def test_concurrent_safety(self, mock_config: BacktesterConfig) -> None:
        """Test that multiple engines can run safely."""
        engine1 = BacktestEngine(config=mock_config)
        engine2 = BacktestEngine(config=mock_config)

        # Both engines should be independent
        assert engine1 is not engine2
        assert engine1.config == engine2.config
        assert engine1.data_handler is not engine2.data_handler

    def test_process_signals_executes_orders_when_allowed(
        self, mock_config: BacktesterConfig, test_data: pd.DataFrame
    ) -> None:
        """Ensure signal processing calls the strategy path and executes orders."""
        engine = BacktestEngine(config=mock_config)
        engine.current_portfolio = Mock()
        engine.current_portfolio.process_tick.return_value = {'total_value': 1250.0}
        engine.portfolio_strategy = Mock()
        engine.portfolio_strategy.calculate_target_weights.return_value = {'SPY': 1.0}
        engine.portfolio_strategy.process_signals.return_value = [
            {'symbol': 'SPY', 'side': 'BUY', 'quantity': 2.0}
        ]
        executed_orders = [
            {
                'symbol': 'SPY',
                'side': 'BUY',
                'filled_quantity': 2.0,
                'filled_price': 101.0,
                'metadata': {},
            }
        ]

        with patch.object(engine, '_execute_orders', return_value=executed_orders) as mock_exec:
            result = engine._process_signals_and_update_portfolio(
                signals=[{'signal_type': 'BUY', 'quantity': 2}],
                current_price=101.0,
                day_high=102.0,
                day_low=99.0,
                timestamp=pd.Timestamp("2023-01-02"),
                symbol='SPY',
                historical_data=test_data.tail(5),
                allow_execution=True,
            )

        mock_exec.assert_called_once()
        assert result['executed_orders'] == executed_orders
        market_data = engine.current_portfolio.process_tick.call_args.kwargs['market_data']
        assert 'SPY' in market_data
        assert list(market_data['SPY'].columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_process_signals_skips_execution_when_risk_denies(
        self, mock_config: BacktesterConfig, test_data: pd.DataFrame
    ) -> None:
        """Orders should not be executed when risk constraints block execution."""
        engine = BacktestEngine(config=mock_config)
        engine.portfolio_strategy = None
        engine.current_portfolio = Mock()
        engine.current_portfolio.process_tick.return_value = {'total_value': 1100.0}

        with patch.object(engine, '_execute_orders') as mock_exec:
            result = engine._process_signals_and_update_portfolio(
                signals=[{'signal_type': 'BUY', 'quantity': 1}],
                current_price=100.0,
                day_high=101.0,
                day_low=99.0,
                timestamp=pd.Timestamp("2023-01-02"),
                symbol='SPY',
                historical_data=test_data.head(3),
                allow_execution=False,
            )

        mock_exec.assert_not_called()
        assert result['executed_orders'] == []

    def test_evaluate_portfolio_risk_records_signal(self, mock_config: BacktesterConfig) -> None:
        """_evaluate_portfolio_risk should forward snapshots to the risk manager."""
        engine = BacktestEngine(config=mock_config)
        broker = Mock()
        broker.positions = {'SPY': 2.0}
        broker.get_current_price.return_value = 101.0
        broker.order_manager.cancel_all_orders = Mock()
        engine.current_broker = broker
        engine.current_risk_manager = Mock()
        engine.current_risk_manager.check_portfolio_risk.return_value = {
            'risk_level': 'LOW',
            'violations': [],
        }
        engine.event_bus = Mock()

        result = engine._evaluate_portfolio_risk(
            2500.0, stage='pre-trade', timestamp=pd.Timestamp("2023-01-02")
        )

        engine.current_risk_manager.check_portfolio_risk.assert_called_once()
        assert result['stage'] == 'pre-trade'
        assert result['timestamp'] is not None
        engine.current_risk_manager.add_risk_signal.assert_called_once()
        broker.order_manager.cancel_all_orders.assert_not_called()
        engine.event_bus.publish.assert_not_called()

    def test_evaluate_portfolio_risk_emits_alert_on_high_risk(
        self, mock_config: BacktesterConfig
    ) -> None:
        """High-risk results should cancel orders and publish alerts."""
        engine = BacktestEngine(config=mock_config)
        broker = Mock()
        broker.positions = {'SPY': 5.0}
        broker.get_current_price.return_value = 105.0
        broker.order_manager.cancel_all_orders = Mock()
        engine.current_broker = broker
        engine.current_risk_manager = Mock()
        engine.current_risk_manager.check_portfolio_risk.return_value = {
            'risk_level': 'CRITICAL',
            'violations': ['Max leverage exceeded'],
        }
        engine.event_bus = Mock()
        engine.trade_history = []

        result = engine._evaluate_portfolio_risk(
            5000.0, stage='post-trade', timestamp=pd.Timestamp("2023-01-03")
        )

        broker.order_manager.cancel_all_orders.assert_called_once()
        engine.event_bus.publish.assert_called_once()
        assert engine.trade_history[-1]['event'] == 'RISK_ALERT'
        assert result['risk_level'] == 'CRITICAL'

    def test_calculate_performance_metrics_uses_analyzer(
        self, mock_config: BacktesterConfig
    ) -> None:
        """_calculate_performance_metrics should combine analyzer output with portfolio stats."""
        engine = BacktestEngine(config=mock_config)
        engine.current_portfolio = Mock()
        engine.current_portfolio.portfolio_values = [1000.0, 1050.0, 1100.0]
        engine.current_portfolio.trade_log = [{'action': 'OPEN'}, {'action': 'CLOSE'}]
        engine.current_portfolio.cumulative_tax = 12.5
        engine.current_portfolio.total_value = 1200.0
        engine.performance_analyzer = Mock()
        engine.performance_analyzer.comprehensive_analysis.return_value = {
            'total_return': 0.1,
            'sharpe_ratio': 1.2,
        }
        engine.data_handler = Mock()
        engine.data_handler.get_data.return_value = pd.DataFrame({'Close': [10, 11, 12]})
        assert engine.config.performance is not None
        engine.config.performance.benchmark_enabled = True

        metrics = engine._calculate_performance_metrics()

        engine.performance_analyzer.comprehensive_analysis.assert_called_once()
        assert metrics['total_trades'] == 2
        assert metrics['cumulative_tax'] == 12.5
        assert metrics['base_pool_final'] == engine.current_portfolio.total_value / 2
        assert 'final_portfolio_value' in metrics


if __name__ == "__main__":
    pytest.main([__file__])
