"""Risk Limits Management System.

This module provides comprehensive risk limits functionality with support for
portfolio-level limits, position concentration limits, and dynamic risk limit management.
"""

import logging
from typing import Any

from backtester.risk_management.component_configs.risk_limit_config import RiskLimitConfig
from backtester.risk_management.risk_profile import RiskProfile


class RiskLimits:
    """Risk limits management class."""

    def __init__(
        self,
        config: RiskLimitConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize risk limits.

        Args:
            config: RiskLimitConfig with risk limits parameters
            logger: Optional logger instance
        """
        self.config: RiskLimitConfig = config or RiskLimitConfig()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

        # State tracking
        self.current_profile = self.config.risk_profile
        self.position_weights: dict[str, float] = {}
        self.sector_exposures: dict[str, float] = {}
        self.concentration_alerts: list[dict[str, Any]] = []

    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Check if drawdown is within limits.

        Args:
            current_drawdown: Current portfolio drawdown

        Returns:
            True if within limit
        """
        return abs(current_drawdown) <= self.config.max_drawdown

    def check_leverage_limit(self, current_leverage: float) -> bool:
        """Check if leverage is within limits.

        Args:
            current_leverage: Current leverage factor

        Returns:
            True if within limit
        """
        return current_leverage <= self.config.max_leverage

    def check_position_size_limit(self, position_size: float) -> bool:
        """Check if position size is within limits.

        Args:
            position_size: Position size as fraction of portfolio

        Returns:
            True if within limit
        """
        return position_size <= self.config.max_single_position

    def check_sector_exposure_limit(self, sector: str, exposure: float) -> bool:
        """Check if sector exposure is within limits.

        Args:
            sector: Sector name
            exposure: Sector exposure

        Returns:
            True if within limit
        """
        return exposure <= self.config.max_sector_exposure

    def check_correlation_limit(self, correlation: float) -> bool:
        """Check if correlation is within limits.

        Args:
            correlation: Correlation value

        Returns:
            True if within limit
        """
        return correlation <= self.config.max_correlation

    def check_volatility_limit(self, volatility: float) -> bool:
        """Check if volatility is within limits.

        Args:
            volatility: Volatility value

        Returns:
            True if within limit
        """
        return volatility <= self.config.max_volatility

    def check_limits(self, portfolio: Any) -> list[str]:
        """Check risk limits and return list of violations.

        Args:
            portfolio: Portfolio object with positions and metrics

        Returns:
            List of violation descriptions
        """
        violations = []

        # Check if portfolio has positions
        if hasattr(portfolio, 'positions'):
            # Check position sizes
            for symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                allocation = abs(position.get('allocation', 0))
                if allocation > self.config.max_single_position:
                    violations.append(
                        f"Position {symbol} exceeds max size: "
                        f"{allocation:.2%} > {self.config.max_single_position:.2%}"
                    )

        # Check drawdown if available
        if hasattr(portfolio, 'get_current_drawdown'):
            current_dd = portfolio.get_current_drawdown()
            if abs(current_dd) > self.config.max_drawdown:
                violations.append(f"Max drawdown exceeded: {current_dd:.2%}")

        # Check leverage if available
        if hasattr(portfolio, 'get_total_leverage'):
            leverage = portfolio.get_total_leverage()
            if leverage > self.config.max_leverage:
                violations.append(f"Leverage exceeds limit: {leverage:.2f}x")

        return violations

    def check_all_limits(self, portfolio_state: dict[str, Any]) -> dict[str, Any]:
        """Check all risk limits.

        Args:
            portfolio_state: Dictionary with portfolio metrics

        Returns:
            Dictionary with check results
        """
        breached_limits = []
        all_passed = True

        # Check drawdown
        if 'current_drawdown' in portfolio_state and not self.check_drawdown_limit(
            portfolio_state['current_drawdown']
        ):
            breached_limits.append('drawdown')
            all_passed = False

        # Check leverage
        if 'leverage' in portfolio_state and not self.check_leverage_limit(
            portfolio_state['leverage']
        ):
            breached_limits.append('leverage')
            all_passed = False

        # Check position sizes
        if 'largest_position' in portfolio_state and not self.check_position_size_limit(
            portfolio_state['largest_position']
        ):
            breached_limits.append('position_size')
            all_passed = False

        # Check VaR
        if 'current_var' in portfolio_state:
            current_var = abs(portfolio_state['current_var'])
            if current_var > self.config.max_portfolio_var:
                breached_limits.append('var')
                all_passed = False

        # Calculate risk score (simplified)
        risk_score = len(breached_limits) * 0.25

        return {
            'all_limits_passed': all_passed,
            'breached_limits': breached_limits,
            'risk_score': risk_score,
            'recommendations': self._generate_recommendations(breached_limits),
        }

    def set_risk_profile(self, profile_name: str) -> None:
        """Set risk profile with different limits.

        Args:
            profile_name: Risk profile name (conservative, moderate, aggressive)
        """
        try:
            # Create the appropriate risk profile
            if profile_name.lower() == 'conservative':
                profile = RiskProfile.conservative()
            elif profile_name.lower() == 'aggressive':
                profile = RiskProfile.aggressive()
            else:  # moderate or any other value
                profile = RiskProfile.moderate()

            self.config.risk_profile = profile
            self.current_profile = profile

            # Apply profile limits
            profile_limits = self.config.get_profile_limits()
            for limit_name, limit_value in profile_limits.items():
                if hasattr(self.config, limit_name):
                    setattr(self.config, limit_name, limit_value)

            self.logger.info(f"Risk profile set to {profile_name}")
        except ValueError:
            self.logger.warning(f"Invalid risk profile: {profile_name}")

    def handle_limit_breach(self, breaches: list[dict[str, Any]]) -> dict[str, Any]:
        """Handle limit breach escalation.

        Args:
            breaches: List of limit breach information

        Returns:
            Dictionary with escalation information
        """
        severity_level = 'medium'
        if len(breaches) >= 3:
            severity_level = 'critical'
        elif len(breaches) >= 2:
            severity_level = 'high'

        required_actions = ['reduce_position_sizes']
        if severity_level in ['high', 'critical']:
            required_actions.extend(['increase_cash_position', 'review_strategies'])

        timeline = 'immediate' if severity_level == 'critical' else 'within_hour'

        return {
            'severity_level': severity_level,
            'required_actions': required_actions,
            'timeline': timeline,
        }

    def update_exposures(
        self, positions: dict[str, dict[str, Any]], sector_mapping: dict[str, str] | None = None
    ) -> None:
        """Update current exposures based on positions.

        Args:
            positions: Dictionary of position information
            sector_mapping: Optional mapping of symbols to sectors
        """
        if not positions:
            self.position_weights.clear()
            self.sector_exposures.clear()
            return

        total_value = sum(pos.get('market_value', 0) for pos in positions.values())

        # Update position weights
        self.position_weights.clear()
        for symbol, position in positions.items():
            weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
            self.position_weights[symbol] = weight

        # Update sector exposures
        self.sector_exposures.clear()
        if sector_mapping:
            for symbol, weight in self.position_weights.items():
                sector = sector_mapping.get(symbol, 'OTHER')
                self.sector_exposures[sector] = self.sector_exposures.get(sector, 0.0) + weight

    def get_exposure_summary(self) -> dict[str, Any]:
        """Get current exposure summary.

        Returns:
            Dictionary with exposure information
        """
        concentration_score = self._calculate_concentration_risk_score()

        violations = self._check_exposure_limits()

        return {
            'position_weights': self.position_weights.copy(),
            'sector_exposures': self.sector_exposures.copy(),
            'concentration_score': concentration_score,
            'violations': violations,
            'total_positions': len(self.position_weights),
        }

    def _calculate_concentration_risk_score(self) -> float:
        """Calculate concentration risk score.

        Returns:
            Concentration risk score (0.0 to 1.0)
        """
        if not self.position_weights:
            return 0.0

        # Herfindahl-Hirschman Index for concentration
        hhi = sum(w * w for w in self.position_weights.values())

        # Normalize to 0-1 scale
        concentration_score = min(1.0, hhi * 4)  # Scale factor for typical portfolios

        return concentration_score

    def _check_exposure_limits(self) -> list[dict[str, Any]]:
        """Check exposure limits.

        Returns:
            List of exposure violations
        """
        violations = []

        # Check position concentration
        for symbol, weight in self.position_weights.items():
            if weight > self.config.concentration_limit:
                violations.append(
                    {
                        'type': 'CONCENTRATION_LIMIT',
                        'symbol': symbol,
                        'weight': weight,
                        'limit': self.config.concentration_limit,
                        'severity': (
                            'HIGH' if weight > self.config.concentration_limit * 1.5 else 'MEDIUM'
                        ),
                    }
                )

        # Check sector exposures
        for sector, exposure in self.sector_exposures.items():
            if exposure > self.config.max_sector_exposure:
                violations.append(
                    {
                        'type': 'SECTOR_LIMIT',
                        'sector': sector,
                        'exposure': exposure,
                        'limit': self.config.max_sector_exposure,
                        'severity': (
                            'HIGH' if exposure > self.config.max_sector_exposure * 1.5 else 'MEDIUM'
                        ),
                    }
                )

        return violations

    def calculate_sector_exposure(self, positions: dict[str, Any]) -> dict[str, float]:
        """Calculate sector exposure for portfolio positions.

        Args:
            positions: Dictionary mapping symbols to position info with sector

        Returns:
            Dictionary mapping sectors to their total exposure
        """
        sector_exposure: dict[str, float] = {}
        total_value = 0.0

        # Calculate total portfolio value
        for _symbol, position in positions.items():
            if isinstance(position, dict):
                total_value += position.get('market_value', 0)
            else:
                total_value += position

        # Calculate sector exposures
        for _symbol, position in positions.items():
            if isinstance(position, dict):
                weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
                sector = position.get('sector', 'OTHER')
            else:
                # Position is a weight
                weight = position / total_value if total_value > 0 else 0
                sector = 'OTHER'  # Default sector

            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        return sector_exposure

    def _generate_recommendations(self, breached_limits: list[str]) -> list[str]:
        """Generate recommendations based on breached limits.

        Args:
            breached_limits: List of breached limit types

        Returns:
            List of recommendations
        """
        recommendations = []

        if 'drawdown' in breached_limits:
            recommendations.append("Reduce overall portfolio risk")
        if 'leverage' in breached_limits:
            recommendations.append("Lower leverage ratio")
        if 'position_size' in breached_limits:
            recommendations.append("Reduce individual position sizes")
        if 'var' in breached_limits:
            recommendations.append("Implement additional risk controls")
        if 'volatility' in breached_limits:
            recommendations.append("Increase diversification")

        return recommendations

    def get_limits_summary(self) -> dict[str, Any]:
        """Get current risk limits summary.

        Returns:
            Dictionary with limits information
        """
        return {
            'max_drawdown': self.config.max_drawdown,
            'max_leverage': self.config.max_leverage,
            'max_single_position': self.config.max_single_position,
            'max_sector_exposure': self.config.max_sector_exposure,
            'max_portfolio_var': self.config.max_portfolio_var,
            'max_daily_loss': self.config.max_daily_loss,
            'concentration_limit': self.config.concentration_limit,
            'current_profile': self.current_profile,
        }
