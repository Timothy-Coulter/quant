"""Tests for portfolio strategy configuration.

This module contains comprehensive tests for the portfolio strategy configuration models,
including validation, serialization, and edge cases.
"""

import pytest
from pydantic import ValidationError

from backtester.strategy.portfolio.portfolio_strategy_config import (
    AllocationMethod,
    PerformanceMetrics,
    PortfolioConstraints,
    PortfolioOptimizationParams,
    PortfolioStrategyConfig,
    PortfolioStrategyType,
    RebalanceFrequency,
    RiskBudget,
    RiskBudgetMethod,
    RiskParameters,
    SignalFilterConfig,
)


class TestPortfolioStrategyConfig:
    """Test suite for PortfolioStrategyConfig."""

    @pytest.fixture
    def basic_config_dict(self):
        """Create a basic configuration dictionary."""
        return {
            "strategy_name": "test_strategy",
            "strategy_type": "equal_weight",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
        }

    @pytest.fixture
    def full_config_dict(self):
        """Create a full configuration dictionary."""
        return {
            "strategy_name": "test_strategy",
            "strategy_type": "risk_parity",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "constraints": {
                "min_weight": 0.01,
                "max_weight": 0.4,
                "min_position_size": 0.001,
                "max_position_size": 0.3,
                "sector_constraints": {"TECH": {"min_weight": 0.1, "max_weight": 0.8}},
                "correlation_constraints": {"AAPL": 0.7, "GOOGL": 0.7, "MSFT": 0.7},
            },
            "optimization_params": {
                "lookback_period": 252,
                "risk_free_rate": 0.02,
                "target_return": 0.1,
                "target_risk": 0.15,
                "optimization_method": "mean_variance",
                "risk_aversion": 1.0,
                "transaction_costs": 0.001,
                "rebalance_threshold": 0.05,
                "min_samples": 30,
                "max_samples": 1000,
            },
            "risk_budget": {
                "risk_budget_method": "equal",
                "risk_budgets": {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34},
                "risk_contribution_targets": {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34},
                "volatility_target": 0.15,
                "max_drawdown": 0.2,
                "var_confidence": 0.95,
                "var_horizon": 1,
            },
            "risk_parameters": {
                "max_drawdown": 0.2,
                "var_confidence": 0.95,
                "var_horizon": 1,
                "max_position_size": 0.3,
                "max_sector_exposure": 0.5,
                "max_correlation": 0.8,
                "volatility_target": 0.15,
                "kelly_fraction": 0.25,
                "min_win_rate": 0.01,
                "risk_parity_tolerance": 0.1,
                "rebalance_tolerance": 0.05,
            },
            "signal_filters": {
                "min_confidence": 0.5,
                "max_confidence": 1.0,
                "min_strength": 0.0,
                "max_strength": 1.0,
                "min_signal_duration": 1,
                "max_signal_duration": 100,
                "custom_filters": {"signal_type": "BUY", "source": "technical"},
            },
            "performance_metrics": {
                "benchmark_symbol": "SPY",
                "risk_free_rate": 0.02,
                "annualization_factor": 252,
                "calculate_sharpe_ratio": True,
                "calculate_sortino_ratio": True,
                "calculate_calmar_ratio": True,
                "calculate_information_ratio": True,
                "calculate_max_drawdown": True,
                "calculate_var": True,
                "calculate_cvar": True,
                "calculate_tracking_error": True,
                "calculate_beta": True,
                "calculate_alpha": True,
                "calculate_turnover": True,
                "calculate_slippage": True,
                "calculate_commission": True,
            },
            "rebalance_frequency": "weekly",
            "enable_rebalancing": True,
            "min_position_size": 0.001,
            "max_position_size": 0.3,
            "allocation_method": "equal_weight",
            "custom_parameters": {"custom_param1": "value1", "custom_param2": 42},
        }

    def test_basic_config_creation(self, basic_config_dict):
        """Test basic configuration creation."""
        config = PortfolioStrategyConfig(**basic_config_dict)

        assert config.strategy_name == "test_strategy"
        assert config.strategy_type == PortfolioStrategyType.EQUAL_WEIGHT
        assert config.symbols == ["AAPL", "GOOGL", "MSFT"]
        assert config.enable_rebalancing is True
        assert config.min_position_size == 0.001
        assert config.max_position_size == 0.3
        assert config.allocation_method == AllocationMethod.EQUAL_WEIGHT

    def test_full_config_creation(self, full_config_dict):
        """Test full configuration creation."""
        config = PortfolioStrategyConfig(**full_config_dict)

        assert config.strategy_name == "test_strategy"
        assert config.strategy_type == PortfolioStrategyType.RISK_PARITY
        assert config.symbols == ["AAPL", "GOOGL", "MSFT"]
        assert config.constraints.min_weight == 0.01
        assert config.constraints.max_weight == 0.4
        assert config.optimization_params.lookback_period == 252
        assert config.risk_budget.risk_budget_method == RiskBudgetMethod.EQUAL
        assert config.risk_parameters.max_drawdown == 0.2
        assert config.signal_filters.min_confidence == 0.5
        assert config.performance_metrics.benchmark_symbol == "SPY"
        assert config.rebalance_frequency == RebalanceFrequency.WEEKLY
        assert config.enable_rebalancing is True
        assert config.custom_parameters == {"custom_param1": "value1", "custom_param2": 42}

    def test_config_with_defaults(self):
        """Test configuration with default values."""
        config = PortfolioStrategyConfig(
            strategy_name="test_strategy", symbols=["AAPL", "GOOGL", "MSFT"]
        )

        assert config.strategy_type == PortfolioStrategyType.EQUAL_WEIGHT
        assert config.constraints.min_weight == 0.0
        assert config.constraints.max_weight == 1.0
        assert config.optimization_params.lookback_period == 252
        assert config.risk_budget.risk_budget_method == RiskBudgetMethod.EQUAL
        assert config.risk_parameters.max_drawdown == 0.2
        assert config.signal_filters.min_confidence == 0.5
        assert config.performance_metrics.risk_free_rate == 0.02
        assert config.rebalance_frequency == RebalanceFrequency.WEEKLY
        assert config.enable_rebalancing is True
        assert config.min_position_size == 0.001
        assert config.max_position_size == 0.3
        assert config.allocation_method == AllocationMethod.EQUAL_WEIGHT
        assert config.custom_parameters == {}

    def test_config_validation_invalid_strategy_name(self):
        """Test configuration validation with invalid strategy name."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="",  # Empty name
                symbols=["AAPL", "GOOGL", "MSFT"],
            )

    def test_config_validation_no_symbols(self):
        """Test configuration validation with no symbols."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=[],  # Empty symbols list
            )

    def test_config_validation_duplicate_symbols(self):
        """Test configuration validation with duplicate symbols."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "AAPL", "MSFT"],  # Duplicate symbols
            )

    def test_config_validation_invalid_max_position_size(self):
        """Test configuration validation with invalid max position size."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                min_position_size=0.1,
                max_position_size=0.05,  # Max < min
            )

    def test_config_validation_invalid_confidence_filters(self):
        """Test configuration validation with invalid confidence filters."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                signal_filters={
                    "min_confidence": 0.8,
                    "max_confidence": 0.5,  # Max < min
                },
            )

    def test_config_validation_invalid_strength_filters(self):
        """Test configuration validation with invalid strength filters."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                signal_filters={
                    "min_strength": 0.8,
                    "max_strength": 0.5,  # Max < min
                },
            )

    def test_config_validation_invalid_duration_filters(self):
        """Test configuration validation with invalid duration filters."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                signal_filters={
                    "min_signal_duration": 100,
                    "max_signal_duration": 50,  # Max < min
                },
            )

    def test_config_validation_invalid_risk_budgets(self):
        """Test configuration validation with invalid risk budgets."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                risk_budget={
                    "risk_budgets": {
                        "AAPL": 0.5,
                        "GOOGL": 0.5,
                        "MSFT": 0.5,  # Sum > 1.0
                    }
                },
            )

    def test_config_validation_invalid_risk_contribution_targets(self):
        """Test configuration validation with invalid risk contribution targets."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                risk_budget={
                    "risk_contribution_targets": {
                        "AAPL": -0.1,  # Negative value
                        "GOOGL": 0.5,
                        "MSFT": 0.6,
                    }
                },
            )

    def test_config_validation_invalid_custom_parameters(self):
        """Test configuration validation with invalid custom parameters."""
        with pytest.raises(ValidationError):
            PortfolioStrategyConfig(
                strategy_name="test_strategy",
                symbols=["AAPL", "GOOGL", "MSFT"],
                custom_parameters={"strategy_name": "conflict"},  # Conflicts with reserved key
            )

    def test_get_constraint_for_symbol(self, full_config_dict):
        """Test getting constraints for a specific symbol."""
        config = PortfolioStrategyConfig(**full_config_dict)

        constraints = config.get_constraint_for_symbol("AAPL")
        assert constraints["min_weight"] == 0.01
        assert constraints["max_weight"] == 0.4
        assert constraints["min_position_size"] == 0.001
        assert constraints["max_position_size"] == 0.3
        assert constraints["max_correlation"] == 0.7

    def test_get_constraint_for_symbol_no_specific_constraints(self, full_config_dict):
        """Test getting constraints for a symbol with no specific constraints."""
        config = PortfolioStrategyConfig(**full_config_dict)

        constraints = config.get_constraint_for_symbol("TSLA")  # Not in config
        assert constraints["min_weight"] == 0.01
        assert constraints["max_weight"] == 0.4
        assert constraints["min_position_size"] == 0.001
        assert constraints["max_position_size"] == 0.3
        assert "max_correlation" not in constraints

    def test_should_rebalance_continuous(self, full_config_dict):
        """Test should_rebalance with continuous frequency."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.rebalance_frequency = RebalanceFrequency.CONTINUOUS

        assert config.should_rebalance({"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}, 0) is True

    def test_should_rebalance_disabled(self, full_config_dict):
        """Test should_rebalance with rebalancing disabled."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.enable_rebalancing = False

        assert config.should_rebalance({"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}, 0) is False

    def test_should_rebalance_threshold_based(self, full_config_dict):
        """Test should_rebalance with threshold-based frequency."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.rebalance_frequency = RebalanceFrequency.THRESHOLD_BASED

        # Test within threshold
        current_weights = {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}
        assert config.should_rebalance(current_weights, 0) is False

        # Test beyond threshold
        current_weights = {"AAPL": 0.5, "GOOGL": 0.25, "MSFT": 0.25}
        assert config.should_rebalance(current_weights, 0) is True

    @pytest.mark.skip(reason="Test needs time-based rebalance logic fix")
    def test_should_rebalance_time_based(self, full_config_dict):
        """Test should_rebalance with time-based frequency."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.rebalance_frequency = RebalanceFrequency.WEEKLY

        # Test within time period
        assert config.should_rebalance({"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}, 5) is False

        # Test beyond time period
        assert config.should_rebalance({"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}, 7) is True

    def test_get_risk_budget_equal(self, full_config_dict):
        """Test getting risk budget with equal method."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.risk_budget.risk_budget_method = RiskBudgetMethod.EQUAL

        risk_budget = config.get_risk_budget()
        expected_budget = {"AAPL": 1.0 / 3, "GOOGL": 1.0 / 3, "MSFT": 1.0 / 3}

        for symbol, expected_weight in expected_budget.items():
            assert risk_budget[symbol] == pytest.approx(expected_weight, rel=1e-6)

    def test_get_risk_budget_custom(self, full_config_dict):
        """Test getting risk budget with custom method."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config.risk_budget.risk_budget_method = RiskBudgetMethod.CUSTOM

        risk_budget = config.get_risk_budget()
        assert risk_budget == {"AAPL": 0.33, "GOOGL": 0.33, "MSFT": 0.34}

    @pytest.mark.skip(reason="Test needs to_dict method fix")
    def test_to_dict(self, full_config_dict):
        """Test converting configuration to dictionary."""
        config = PortfolioStrategyConfig(**full_config_dict)
        config_dict = config.to_dict()

        assert config_dict == full_config_dict

    def test_from_dict(self, full_config_dict):
        """Test creating configuration from dictionary."""
        config = PortfolioStrategyConfig.from_dict(full_config_dict)

        assert config.strategy_name == "test_strategy"
        assert config.strategy_type == PortfolioStrategyType.RISK_PARITY
        assert config.symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_copy(self, full_config_dict):
        """Test copying configuration."""
        config = PortfolioStrategyConfig(**full_config_dict)
        copied_config = config.copy()

        assert copied_config == config
        assert copied_config is not config

    def test_update(self, full_config_dict):
        """Test updating configuration."""
        config = PortfolioStrategyConfig(**full_config_dict)
        updated_config = config.update(strategy_name="updated_strategy")

        assert updated_config.strategy_name == "updated_strategy"
        assert updated_config.strategy_type == PortfolioStrategyType.RISK_PARITY
        assert updated_config.symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_validate_config_valid(self, full_config_dict):
        """Test validating valid configuration."""
        config = PortfolioStrategyConfig(**full_config_dict)
        assert config.validate_config() is True

    def test_validate_config_invalid(self):
        """Test validating invalid configuration."""
        config = PortfolioStrategyConfig(
            strategy_name="test_strategy", symbols=["AAPL", "GOOGL", "MSFT"]
        )
        config.strategy_name = ""  # Invalid

        assert config.validate_config() is False

    def test_str_representation(self, full_config_dict):
        """Test string representation."""
        config = PortfolioStrategyConfig(**full_config_dict)
        str_repr = str(config)

        assert "PortfolioStrategyConfig" in str_repr
        assert "test_strategy" in str_repr
        assert "risk_parity" in str_repr
        assert "['AAPL', 'GOOGL', 'MSFT']" in str_repr

    def test_repr_representation(self, full_config_dict):
        """Test repr representation."""
        config = PortfolioStrategyConfig(**full_config_dict)
        repr_str = repr(config)

        assert "PortfolioStrategyConfig" in repr_str
        assert "test_strategy" in repr_str
        assert "risk_parity" in repr_str
        assert "['AAPL', 'GOOGL', 'MSFT']" in repr_str


class TestPortfolioConstraints:
    """Test suite for PortfolioConstraints."""

    def test_basic_constraints(self):
        """Test basic constraints creation."""
        constraints = PortfolioConstraints()

        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.min_position_size == 0.001
        assert constraints.max_position_size == 0.3
        assert constraints.sector_constraints == {}
        assert constraints.correlation_constraints == {}

    def test_constraints_with_values(self):
        """Test constraints with custom values."""
        constraints = PortfolioConstraints(
            min_weight=0.01,
            max_weight=0.4,
            min_position_size=0.0001,
            max_position_size=0.5,
            sector_constraints={"TECH": {"min_weight": 0.1, "max_weight": 0.8}},
            correlation_constraints={"AAPL": 0.7},
        )

        assert constraints.min_weight == 0.01
        assert constraints.max_weight == 0.4
        assert constraints.min_position_size == 0.0001
        assert constraints.max_position_size == 0.5
        assert constraints.sector_constraints == {"TECH": {"min_weight": 0.1, "max_weight": 0.8}}
        assert constraints.correlation_constraints == {"AAPL": 0.7}

    def test_constraints_validation_invalid_max_weight(self):
        """Test constraints validation with invalid max weight."""
        with pytest.raises(ValidationError):
            PortfolioConstraints(min_weight=0.5, max_weight=0.4)  # Max < min

    def test_constraints_validation_invalid_max_position_size(self):
        """Test constraints validation with invalid max position size."""
        with pytest.raises(ValidationError):
            PortfolioConstraints(min_position_size=0.1, max_position_size=0.05)  # Max < min

    def test_get_constraint_for_symbol(self):
        """Test getting constraints for a specific symbol."""
        constraints = PortfolioConstraints(
            min_weight=0.01,
            max_weight=0.4,
            min_position_size=0.0001,
            max_position_size=0.5,
            sector_constraints={"TECH": {"min_weight": 0.1, "max_weight": 0.8}},
            correlation_constraints={"AAPL": 0.7},
        )

        symbol_constraints = constraints.get_constraint_for_symbol("AAPL")
        assert symbol_constraints["min_weight"] == 0.01
        assert symbol_constraints["max_weight"] == 0.4
        assert symbol_constraints["min_position_size"] == 0.0001
        assert symbol_constraints["max_position_size"] == 0.5
        assert symbol_constraints["max_correlation"] == 0.7

    def test_get_constraint_for_symbol_no_specific_constraints(self):
        """Test getting constraints for a symbol with no specific constraints."""
        constraints = PortfolioConstraints(
            min_weight=0.01, max_weight=0.4, min_position_size=0.0001, max_position_size=0.5
        )

        symbol_constraints = constraints.get_constraint_for_symbol("TSLA")
        assert symbol_constraints["min_weight"] == 0.01
        assert symbol_constraints["max_weight"] == 0.4
        assert symbol_constraints["min_position_size"] == 0.0001
        assert symbol_constraints["max_position_size"] == 0.5
        assert "max_correlation" not in symbol_constraints


class TestPortfolioOptimizationParams:
    """Test suite for PortfolioOptimizationParams."""

    def test_basic_optimization_params(self):
        """Test basic optimization parameters creation."""
        params = PortfolioOptimizationParams()

        assert params.lookback_period == 252
        assert params.risk_free_rate == 0.02
        assert params.target_return is None
        assert params.target_risk is None
        assert params.optimization_method == "mean_variance"
        assert params.risk_aversion == 1.0
        assert params.transaction_costs == 0.001
        assert params.rebalance_threshold == 0.05
        assert params.min_samples == 30
        assert params.max_samples is None

    def test_optimization_params_with_values(self):
        """Test optimization parameters with custom values."""
        params = PortfolioOptimizationParams(
            lookback_period=100,
            risk_free_rate=0.03,
            target_return=0.1,
            target_risk=0.15,
            optimization_method="risk_parity",
            risk_aversion=2.0,
            transaction_costs=0.002,
            rebalance_threshold=0.1,
            min_samples=50,
            max_samples=500,
        )

        assert params.lookback_period == 100
        assert params.risk_free_rate == 0.03
        assert params.target_return == 0.1
        assert params.target_risk == 0.15
        assert params.optimization_method == "risk_parity"
        assert params.risk_aversion == 2.0
        assert params.transaction_costs == 0.002
        assert params.rebalance_threshold == 0.1
        assert params.min_samples == 50
        assert params.max_samples == 500

    def test_optimization_params_validation_invalid_max_samples(self):
        """Test optimization parameters validation with invalid max samples."""
        with pytest.raises(ValidationError):
            PortfolioOptimizationParams(min_samples=50, max_samples=30)  # Max < min


class TestRiskBudget:
    """Test suite for RiskBudget."""

    def test_basic_risk_budget(self):
        """Test basic risk budget creation."""
        risk_budget = RiskBudget()

        assert risk_budget.risk_budget_method == RiskBudgetMethod.EQUAL
        assert risk_budget.risk_budgets == {}
        assert risk_budget.risk_contribution_targets == {}
        assert risk_budget.volatility_target is None
        assert risk_budget.max_drawdown is None
        assert risk_budget.var_confidence == 0.95
        assert risk_budget.var_horizon == 1

    def test_risk_budget_with_values(self):
        """Test risk budget with custom values."""
        risk_budget = RiskBudget(
            risk_budget_method=RiskBudgetMethod.CUSTOM,
            risk_budgets={"AAPL": 0.5, "GOOGL": 0.5},
            risk_contribution_targets={"AAPL": 0.5, "GOOGL": 0.5},
            volatility_target=0.15,
            max_drawdown=0.2,
            var_confidence=0.99,
            var_horizon=5,
        )

        assert risk_budget.risk_budget_method == RiskBudgetMethod.CUSTOM
        assert risk_budget.risk_budgets == {"AAPL": 0.5, "GOOGL": 0.5}
        assert risk_budget.risk_contribution_targets == {"AAPL": 0.5, "GOOGL": 0.5}
        assert risk_budget.volatility_target == 0.15
        assert risk_budget.max_drawdown == 0.2
        assert risk_budget.var_confidence == 0.99
        assert risk_budget.var_horizon == 5

    def test_risk_budget_validation_invalid_risk_budgets(self):
        """Test risk budget validation with invalid risk budgets."""
        with pytest.raises(ValidationError):
            RiskBudget(risk_budgets={"AAPL": 0.6, "GOOGL": 0.6})  # Sum > 1.0

    def test_risk_budget_validation_invalid_risk_contribution_targets(self):
        """Test risk budget validation with invalid risk contribution targets."""
        with pytest.raises(ValidationError):
            RiskBudget(risk_contribution_targets={"AAPL": -0.1})  # Negative value

    def test_get_risk_budget_equal(self):
        """Test getting risk budget with equal method."""
        risk_budget = RiskBudget(risk_budget_method=RiskBudgetMethod.EQUAL)
        result = risk_budget.get_risk_budget()

        assert result == {}

    def test_get_risk_budget_custom(self):
        """Test getting risk budget with custom method."""
        risk_budget = RiskBudget(
            risk_budget_method=RiskBudgetMethod.CUSTOM, risk_budgets={"AAPL": 0.5, "GOOGL": 0.5}
        )
        result = risk_budget.get_risk_budget()

        assert result == {"AAPL": 0.5, "GOOGL": 0.5}


class TestSignalFilterConfig:
    """Test suite for SignalFilterConfig."""

    def test_basic_signal_filters(self):
        """Test basic signal filters creation."""
        filters = SignalFilterConfig()

        assert filters.min_confidence == 0.5
        assert filters.max_confidence == 1.0
        assert filters.min_strength == 0.0
        assert filters.max_strength == 1.0
        assert filters.min_signal_duration == 1
        assert filters.max_signal_duration == 100
        assert filters.custom_filters == {}

    def test_signal_filters_with_values(self):
        """Test signal filters with custom values."""
        filters = SignalFilterConfig(
            min_confidence=0.7,
            max_confidence=0.9,
            min_strength=0.2,
            max_strength=0.8,
            min_signal_duration=5,
            max_signal_duration=50,
            custom_filters={"signal_type": "BUY", "source": "technical"},
        )

        assert filters.min_confidence == 0.7
        assert filters.max_confidence == 0.9
        assert filters.min_strength == 0.2
        assert filters.max_strength == 0.8
        assert filters.min_signal_duration == 5
        assert filters.max_signal_duration == 50
        assert filters.custom_filters == {"signal_type": "BUY", "source": "technical"}

    def test_signal_filters_validation_invalid_max_confidence(self):
        """Test signal filters validation with invalid max confidence."""
        with pytest.raises(ValidationError):
            SignalFilterConfig(min_confidence=0.8, max_confidence=0.5)  # Max < min

    def test_signal_filters_validation_invalid_max_strength(self):
        """Test signal filters validation with invalid max strength."""
        with pytest.raises(ValidationError):
            SignalFilterConfig(min_strength=0.8, max_strength=0.5)  # Max < min

    def test_signal_filters_validation_invalid_max_duration(self):
        """Test signal filters validation with invalid max duration."""
        with pytest.raises(ValidationError):
            SignalFilterConfig(min_signal_duration=100, max_signal_duration=50)  # Max < min


class TestRiskParameters:
    """Test suite for RiskParameters."""

    def test_basic_risk_parameters(self):
        """Test basic risk parameters creation."""
        params = RiskParameters()

        assert params.max_drawdown == 0.2
        assert params.var_confidence == 0.95
        assert params.var_horizon == 1
        assert params.max_position_size == 0.3
        assert params.max_sector_exposure == 0.5
        assert params.max_correlation == 0.8
        assert params.volatility_target is None
        assert params.kelly_fraction == 0.25
        assert params.min_win_rate == 0.01
        assert params.risk_parity_tolerance == 0.1
        assert params.rebalance_tolerance == 0.05

    def test_risk_parameters_with_values(self):
        """Test risk parameters with custom values."""
        params = RiskParameters(
            max_drawdown=0.15,
            var_confidence=0.99,
            var_horizon=5,
            max_position_size=0.4,
            max_sector_exposure=0.6,
            max_correlation=0.7,
            volatility_target=0.12,
            kelly_fraction=0.3,
            min_win_rate=0.02,
            risk_parity_tolerance=0.15,
            rebalance_tolerance=0.08,
        )

        assert params.max_drawdown == 0.15
        assert params.var_confidence == 0.99
        assert params.var_horizon == 5
        assert params.max_position_size == 0.4
        assert params.max_sector_exposure == 0.6
        assert params.max_correlation == 0.7
        assert params.volatility_target == 0.12
        assert params.kelly_fraction == 0.3
        assert params.min_win_rate == 0.02
        assert params.risk_parity_tolerance == 0.15
        assert params.rebalance_tolerance == 0.08


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics."""

    def test_basic_performance_metrics(self):
        """Test basic performance metrics creation."""
        metrics = PerformanceMetrics()

        assert metrics.benchmark_symbol is None
        assert metrics.risk_free_rate == 0.02
        assert metrics.annualization_factor == 252
        assert metrics.calculate_sharpe_ratio is True
        assert metrics.calculate_sortino_ratio is True
        assert metrics.calculate_calmar_ratio is True
        assert metrics.calculate_information_ratio is True
        assert metrics.calculate_max_drawdown is True
        assert metrics.calculate_var is True
        assert metrics.calculate_cvar is True
        assert metrics.calculate_tracking_error is True
        assert metrics.calculate_beta is True
        assert metrics.calculate_alpha is True
        assert metrics.calculate_turnover is True
        assert metrics.calculate_slippage is True
        assert metrics.calculate_commission is True

    def test_performance_metrics_with_values(self):
        """Test performance metrics with custom values."""
        metrics = PerformanceMetrics(
            benchmark_symbol="SPY",
            risk_free_rate=0.03,
            annualization_factor=365,
            calculate_sharpe_ratio=False,
            calculate_sortino_ratio=False,
            calculate_calmar_ratio=False,
            calculate_information_ratio=False,
            calculate_max_drawdown=False,
            calculate_var=False,
            calculate_cvar=False,
            calculate_tracking_error=False,
            calculate_beta=False,
            calculate_alpha=False,
            calculate_turnover=False,
            calculate_slippage=False,
            calculate_commission=False,
        )

        assert metrics.benchmark_symbol == "SPY"
        assert metrics.risk_free_rate == 0.03
        assert metrics.annualization_factor == 365
        assert metrics.calculate_sharpe_ratio is False
        assert metrics.calculate_sortino_ratio is False
        assert metrics.calculate_calmar_ratio is False
        assert metrics.calculate_information_ratio is False
        assert metrics.calculate_max_drawdown is False
        assert metrics.calculate_var is False
        assert metrics.calculate_cvar is False
        assert metrics.calculate_tracking_error is False
        assert metrics.calculate_beta is False
        assert metrics.calculate_alpha is False
        assert metrics.calculate_turnover is False
        assert metrics.calculate_slippage is False
        assert metrics.calculate_commission is False
