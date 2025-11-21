"""Abstract base portfolio class.

This module defines the base portfolio interface that all portfolio implementations
should inherit from, providing common functionality and ensuring consistent API.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from backtester.core.event_bus import EventBus
from backtester.core.events import create_portfolio_update_event


class PortfolioPerformance(Protocol):
    """Protocol for portfolio performance metrics."""

    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    cumulative_tax: float
    total_trades: int
    winning_trades: int
    win_rate: float
    final_portfolio_value: float
    portfolio_values: list[float] | NDArray[np.float64]


class BasePortfolio(ABC):
    """Abstract base class for all portfolio implementations.

    This class defines the common interface and shared functionality that all
    portfolio implementations should provide, including:
    - Position management
    - Performance tracking
    - Risk management
    - Capital allocation
    - Tax handling
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        commission_rate: float = 0.001,
        interest_rate_daily: float = 0.00025,
        spread_rate: float = 0.0002,
        slippage_std: float = 0.0005,
        funding_enabled: bool = True,
        tax_rate: float = 0.45,
        logger: logging.Logger | None = None,
        event_bus: EventBus | None = None,
        portfolio_id: str | None = None,
    ) -> None:
        """Initialize base portfolio.

        Args:
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate for trades
            interest_rate_daily: Daily interest rate for borrowed funds
            spread_rate: Spread cost per trade
            slippage_std: Standard deviation for slippage simulation
            funding_enabled: Whether to charge interest on borrowed funds
            tax_rate: Tax rate on capital gains
            logger: Optional logger instance
            event_bus: Optional event bus for publishing portfolio updates
            portfolio_id: Identifier used in portfolio update events
        """
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.event_bus = event_bus
        self.portfolio_id = portfolio_id or self.__class__.__name__.lower()

        # Core portfolio parameters
        self.initial_capital: float = initial_capital
        self.commission_rate: float = commission_rate
        self.interest_rate_daily: float = interest_rate_daily
        self.spread_rate: float = spread_rate
        self.slippage_std: float = slippage_std
        self.funding_enabled: bool = funding_enabled
        self.tax_rate: float = tax_rate

        # Portfolio tracking
        self.portfolio_values: list[float] = [initial_capital]
        self.trade_log: list[dict[str, Any]] = []
        self.cash: float = initial_capital
        self.cumulative_tax: float = 0.0
        self.current_year: int | None = None
        self.yearly_gains: dict[str, float] = {}

        self.logger.info(
            f"Initialized {self.__class__.__name__} with ${initial_capital:.2f} capital"
        )

    def _publish_portfolio_update(
        self,
        total_value: float,
        cash_balance: float,
        positions_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish a portfolio update event to the event bus."""
        if self.event_bus is None:
            return

        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("portfolio_id", self.portfolio_id)

        event = create_portfolio_update_event(
            portfolio_id=self.portfolio_id,
            total_value=total_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            metadata=metadata_payload,
        )
        self.event_bus.publish(event, immediate=True)

    @property
    def total_value(self) -> float:
        """Get total portfolio value.

        Returns:
            Total portfolio value
        """
        return self.portfolio_values[-1] if self.portfolio_values else self.initial_capital

    @abstractmethod
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Any,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        leverage: float = 1.0,
        pool_type: str = 'base',
        **kwargs: Any,
    ) -> bool:
        """Add a new position to the portfolio.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            leverage: Leverage factor (1.0 = no leverage)
            pool_type: Pool type for dual pool portfolios
            **kwargs: Additional position-specific parameters

        Returns:
            True if position was added successfully
        """
        ...

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: Any,
        quantity: float | None = None,
    ) -> bool:
        """Close a position.

        Args:
            symbol: Symbol to close
            price: Exit price
            timestamp: Exit timestamp
            quantity: Quantity to close (None for full position)

        Returns:
            True if position was closed successfully
        """
        ...

    @abstractmethod
    def process_tick(
        self,
        timestamp: Any,
        market_data: dict[str, Any] | None = None,
        current_price: float | None = None,
        day_high: float | None = None,
        day_low: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a market tick and update portfolio state.

        Args:
            timestamp: Current timestamp
            market_data: Optional market data dictionary
            current_price: Optional current price
            day_high: Optional day high price
            day_low: Optional day low price
            **kwargs: Additional market data parameters

        Returns:
            Dictionary with updated portfolio information
        """
        ...

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)

        # Basic metrics
        total_return = ((portfolio_values[-1] / self.initial_capital) - 1) * 100
        if np.isnan(total_return) or np.isinf(total_return):
            total_return = 0.0

        # Calculate returns for risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        # Risk metrics
        sharpe_ratio = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Win rate
        winning_trades = len(
            [
                t
                for t in self.trade_log
                if t.get('action') == 'CLOSE' and t.get('realized_pnl', 0) > 0
            ]
        )
        closed_trades = len([t for t in self.trade_log if t.get('action') == 'CLOSE'])
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'cumulative_tax': self.cumulative_tax,
            'total_trades': len(self.trade_log),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'portfolio_values': portfolio_values,
            'trade_log': self.trade_log,
        }

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.portfolio_values = [self.initial_capital]
        self.trade_log.clear()
        self.cumulative_tax = 0.0
        self.current_year = None
        self.yearly_gains.clear()

        self.logger.info(f"{self.__class__.__name__} reset to initial state")

    def _handle_tax_calculation(self, timestamp: Any) -> None:
        """Handle tax calculations at year end.

        Args:
            timestamp: Current timestamp
        """
        if self.current_year is None:
            self.current_year = timestamp.year
            return

        current_year = timestamp.year

        # If new year, process previous year's taxes
        if current_year != self.current_year:
            yearly_total = sum(self.yearly_gains.values())

            # Apply tax if positive
            if yearly_total > 0:
                tax = yearly_total * self.tax_rate
                self.cumulative_tax += tax
                self.cash -= tax

                self.logger.info(f"Applied yearly taxes: ${tax:.2f}")

            # Reset for new year
            self.yearly_gains.clear()
            self.current_year = current_year

    def _calculate_costs(
        self,
        position_value: float,
        quantity: float,
        price: float,
    ) -> tuple[float, float, float]:
        """Calculate trading costs (commission, spread, slippage).

        Args:
            position_value: Total position value
            quantity: Position quantity
            price: Position price

        Returns:
            Tuple of (commission, spread_cost, slippage)
        """
        commission = position_value * self.commission_rate
        spread_cost = position_value * self.spread_rate
        slippage = np.random.normal(0, self.slippage_std) * position_value

        return commission, spread_cost, slippage

    def _log_trade(
        self,
        timestamp: Any,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        **extra_data: Any,
    ) -> None:
        """Log a trade to the trade log.

        Args:
            timestamp: Trade timestamp
            action: Trade action (OPEN, CLOSE, UPDATE, etc.)
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            **extra_data: Additional trade data to log
        """
        trade_record = {
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            **extra_data,
        }
        self.trade_log.append(trade_record)

        self.logger.debug(f"Trade logged: {trade_record}")

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio summary
        """
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        return {
            'total_value': current_value,
            'total_return': ((current_value / self.initial_capital) - 1) * 100,
            'total_trades': len(self.trade_log),
            'cumulative_tax': self.cumulative_tax,
        }
