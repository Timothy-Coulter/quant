"""Integration test for risk management controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from backtester.core.event_bus import EventBus
from backtester.risk_management.component_configs.comprehensive_risk_config import (
    ComprehensiveRiskConfig,
)
from backtester.risk_management.risk_control_manager import RiskControlManager


def test_risk_manager_flags_excess_concentration(event_bus: EventBus) -> None:
    """RiskControlManager should raise HIGH risk when position sizing breaches limits."""
    risk_yaml = Path("component_configs/risk_management/balanced.yaml")
    payload = yaml.safe_load(risk_yaml.read_text())
    assert isinstance(payload, dict)
    config = ComprehensiveRiskConfig(**payload)

    manager = RiskControlManager(config=config, event_bus=event_bus)
    positions: dict[str, dict[str, Any]] = {
        "SPY": {"active": True, "market_value": 30000.0},
        "QQQ": {"active": True, "market_value": 15000.0},
    }

    result = manager.check_portfolio_risk(portfolio_value=100000.0, positions=positions)

    assert result["risk_level"] == "HIGH"
    assert any("Max position size exceeded" in violation for violation in result["violations"])
