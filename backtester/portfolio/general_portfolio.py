"""General Portfolio Implementation.

This module implements the GeneralPortfolio class that handles multi-asset
portfolios with support for 1-N tickers and general position management.
"""

import logging
from typing import Any

import pandas as pd

from backtester.core.config import PortfolioConfig
from backtester.core.event_bus import EventBus
from backtester.core.interfaces import RiskManagerProtocol
from backtester.portfolio.base_portfolio import BasePortfolio
from backtester.portfolio.position import Position


class GeneralPortfolio(BasePortfolio):
    """Multi-asset portfolio for handling 1-N tickers.

    This portfolio class supports multiple positions across different assets
    with comprehensive position management, risk controls, and performance tracking.
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
        max_positions: int = 10,
        logger: logging.Logger | None = None,
        event_bus: EventBus | None = None,
        portfolio_id: str | None = None,
        risk_manager: RiskManagerProtocol | None = None,
        config: PortfolioConfig | None = None,
    ) -> None:
        """Initialize the general portfolio.

        Args:
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate for trades
            interest_rate_daily: Daily interest rate for borrowed funds
            spread_rate: Spread cost per trade
            slippage_std: Standard deviation for slippage simulation
            funding_enabled: Whether to charge interest on borrowed funds
            tax_rate: Tax rate on capital gains
            max_positions: Maximum number of concurrent positions
            logger: Optional logger instance
            event_bus: Optional event bus used to broadcast updates
            portfolio_id: Identifier for emitted portfolio events
            risk_manager: Optional risk manager that centralizes risk checks
            config: PortfolioConfig applied to the portfolio
        """
        resolved_config = (
            config.model_copy(deep=True)
            if config
            else self._config_from_params(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                interest_rate_daily=interest_rate_daily,
                spread_rate=spread_rate,
                slippage_std=slippage_std,
                funding_enabled=funding_enabled,
                tax_rate=tax_rate,
                max_positions=max_positions,
            )
        )

        super().__init__(
            initial_capital=resolved_config.initial_capital,
            commission_rate=resolved_config.commission_rate,
            interest_rate_daily=resolved_config.interest_rate_daily,
            spread_rate=resolved_config.spread_rate,
            slippage_std=resolved_config.slippage_std,
            funding_enabled=resolved_config.funding_enabled,
            tax_rate=resolved_config.tax_rate,
            logger=logger,
            event_bus=event_bus,
            portfolio_id=portfolio_id,
        )

        # Portfolio-specific parameters
        self.max_positions: int = resolved_config.max_positions
        self.cash: float = resolved_config.initial_capital

        # Portfolio tracking (additional to base class)
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.positions: dict[str, Position] = {}
        self.risk_manager = risk_manager
        self._config = resolved_config

        self.logger.info(
            "Initialized GeneralPortfolio with $%.2f capital, max %s positions",
            resolved_config.initial_capital,
            resolved_config.max_positions,
        )

    @classmethod
    def default_config(cls) -> PortfolioConfig:
        """Return the default PortfolioConfig instance."""
        return PortfolioConfig()

    @staticmethod
    def _config_from_params(
        *,
        initial_capital: float,
        commission_rate: float,
        interest_rate_daily: float,
        spread_rate: float,
        slippage_std: float,
        funding_enabled: bool,
        tax_rate: float,
        max_positions: int,
    ) -> PortfolioConfig:
        """Create a PortfolioConfig from primitive parameters."""
        return PortfolioConfig(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            interest_rate_daily=interest_rate_daily,
            spread_rate=spread_rate,
            slippage_std=slippage_std,
            funding_enabled=funding_enabled,
            tax_rate=tax_rate,
            max_positions=max_positions,
        )

    @property
    def total_value(self) -> float:
        """Get total portfolio value including cash and positions."""
        total = self.cash
        for position in self.positions.values():
            total += position.quantity * position.current_price
        return total

    # ------------------------------------------------------------------#
    # Lifecycle hooks
    # ------------------------------------------------------------------#
    def before_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked before the simulation loop begins."""
        return None

    def before_tick(self, context: dict[str, Any]) -> None:
        """Hook invoked before processing each tick."""
        return None

    def after_tick(self, context: dict[str, Any], results: dict[str, Any]) -> None:
        """Hook invoked after finishing each tick."""
        return None

    def after_run(self, metadata: dict[str, Any] | None = None) -> None:
        """Hook invoked after the simulation loop completes."""
        return None

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
        # Check if we can open another position
        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"Cannot open position for {symbol}: max positions reached")
            return False

        # Check if we already have a position in this symbol
        if symbol in self.positions:
            self.logger.warning(f"Cannot open position for {symbol}: already have position")
            return False

        # Calculate costs - leverage reduces required cash
        total_position_value = quantity * price
        margin_required = total_position_value / leverage
        commission, spread_cost, slippage = self._calculate_costs(margin_required, quantity, price)

        # Total required cash - only costs, not margin (margin gets returned on close)
        total_required = commission + spread_cost + abs(slippage)

        # Check if we have enough cash
        if total_required > self.cash:
            self.logger.warning(
                f"Insufficient cash for {symbol} position: need {total_required:.2f}, have {self.cash:.2f}"
            )
            return False

        # Risk manager approval before committing capital
        if self.risk_manager:
            portfolio_value = self.total_value or self.initial_capital
            if not self.risk_manager.can_open_position(
                symbol,
                abs(total_position_value),
                portfolio_value,
                {'source': 'portfolio_add'},
            ):
                self.logger.warning("Risk manager rejected opening %s", symbol)
                return False

        # Deduct from cash
        self.cash -= total_required

        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            current_price=price,
            entry_timestamp=timestamp,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            commission_paid=commission + spread_cost + slippage,
            total_cost=margin_required,
            total_commission=commission,
        )

        self.positions[symbol] = position

        # Log the trade
        self._log_trade(
            timestamp=timestamp,
            action='OPEN',
            symbol=symbol,
            quantity=quantity,
            price=price,
            leverage=leverage,
            margin_required=margin_required,
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )

        self.total_commission += commission
        self.total_slippage += slippage

        self.logger.info(
            f"Opened {symbol} position: {quantity}@{price:.4f}, leverage: {leverage:.1f}x, margin: {margin_required:.2f}"
        )
        return True

    def apply_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Broadcast an executed fill to the portfolio so holdings stay in sync."""
        side_normalized = side.upper()
        metadata = metadata or {}

        if side_normalized == 'BUY':
            if symbol in self.positions:
                self.update_position(symbol, quantity, price, timestamp)
            else:
                self.add_position(symbol, quantity, price, timestamp)
        elif side_normalized == 'SELL':
            if symbol in self.positions:
                self.close_position(symbol, price, timestamp, quantity=quantity)
        else:
            self.logger.debug("Ignoring fill with unsupported side '%s'", side)
            return

        if self.risk_manager:
            self.risk_manager.record_fill(
                symbol,
                side_normalized,
                quantity,
                price,
                self.total_value,
                metadata,
            )

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
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        close_quantity = quantity or position.quantity

        if close_quantity > position.quantity:
            return False

        # Calculate realized P&L
        realized_pnl = (price - position.avg_price) * close_quantity

        # Calculate costs
        commission, spread_cost, slippage = self._calculate_costs(
            close_quantity * price, close_quantity, price
        )

        # Update position or remove it
        if close_quantity == position.quantity:
            # Close entire position
            position.realized_pnl += realized_pnl - commission - spread_cost - slippage
            self.cash += close_quantity * price - commission - spread_cost - slippage
            del self.positions[symbol]
        else:
            # Partial close
            position.quantity -= close_quantity
            position.realized_pnl += realized_pnl - commission - spread_cost - slippage
            self.cash += close_quantity * price - commission - spread_cost - slippage

        # Update tracking
        self.total_commission += commission
        self.total_slippage += slippage

        # Log the trade
        self._log_trade(
            timestamp=timestamp,
            action='CLOSE',
            symbol=symbol,
            quantity=close_quantity,
            price=price,
            realized_pnl=realized_pnl,
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
        )

        return True

    def update_position(self, symbol: str, quantity: float, price: float, timestamp: Any) -> bool:
        """Update an existing position by adding to it.

        Args:
            symbol: Symbol to update
            quantity: Additional quantity
            price: Price for the additional quantity
            timestamp: Update timestamp

        Returns:
            True if position was updated successfully
        """
        if symbol not in self.positions:
            return False

        old_position = self.positions[symbol]

        # Calculate new average price
        total_quantity = old_position.quantity + quantity
        total_cost = (old_position.quantity * old_position.avg_price) + (quantity * price)
        new_avg_price = total_cost / total_quantity

        # Update position
        old_position.quantity = total_quantity
        old_position.avg_price = new_avg_price
        old_position.current_price = price

        # Calculate and update costs
        commission, _, _ = self._calculate_costs(quantity * price, quantity, price)
        old_position.total_commission += commission
        self.total_commission += commission

        # Deduct additional cash for the new quantity
        additional_cost = quantity * price + commission
        self.cash -= additional_cost

        # Log the trade
        self._log_trade(
            timestamp=timestamp,
            action='UPDATE',
            symbol=symbol,
            quantity=quantity,
            price=price,
            new_total_quantity=total_quantity,
            new_avg_price=new_avg_price,
            commission=commission,
        )

        return True

    def process_tick(
        self,
        timestamp: Any,
        market_data: dict[str, pd.DataFrame] | None = None,
        current_price: float | None = None,
        day_high: float | None = None,
        day_low: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a market tick and update portfolio state.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary with symbol -> DataFrame mapping
            current_price: Optional current price
            day_high: Optional day high price
            day_low: Optional day low price
            **kwargs: Additional market data parameters

        Returns:
            Dictionary with updated portfolio information
        """
        total_portfolio_value = self.cash
        position_updates: list[dict[str, Any]] = []

        # Update all positions
        for symbol, position in list(self.positions.items()):
            if market_data is not None and symbol in market_data:
                current_data = market_data[symbol]
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    day_high = current_data['High'].iloc[-1]
                    day_low = current_data['Low'].iloc[-1]

                    # Update position
                    update_result = position.update_market_data(current_price, day_high, day_low)
                    position_updates.append(update_result)

                    # Check if position should be closed
                    if update_result.get('should_close', False):
                        self._close_position(
                            symbol,
                            update_result['close_reason'],
                            update_result['exit_price'],
                            timestamp,
                        )
                        continue

                    # Add position value to portfolio
                    total_portfolio_value += position.quantity * current_price

        # Calculate financing costs
        financing_cost = self._calculate_financing_costs()
        self.cash -= financing_cost

        # Update portfolio value
        self.portfolio_values.append(total_portfolio_value)

        # Handle tax calculations
        self._handle_tax_calculation(timestamp)
        positions_value = max(total_portfolio_value - self.cash, 0.0)
        positions_snapshot = {
            sym: {
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
            }
            for sym, pos in self.positions.items()
        }
        event_metadata = {
            'positions': positions_snapshot,
            'position_updates': position_updates,
            'timestamp': timestamp,
        }
        self._publish_portfolio_update(
            total_value=total_portfolio_value,
            cash_balance=self.cash,
            positions_value=positions_value,
            metadata=event_metadata,
        )

        return {
            'timestamp': timestamp,
            'total_value': total_portfolio_value,
            'cash': self.cash,
            'position_count': len(self.positions),
            'position_updates': position_updates,
            'financing_cost': financing_cost,
            'cumulative_tax': self.cumulative_tax,
        }

    def _close_position(self, symbol: str, reason: str, exit_price: float, timestamp: Any) -> None:
        """Close a position.

        Args:
            symbol: Symbol to close
            reason: Reason for closing
            exit_price: Exit price
            timestamp: Exit timestamp
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate realized P&L
        realized_pnl = (exit_price - position.avg_price) * position.quantity

        # Calculate total costs
        commission, spread_cost, slippage = self._calculate_costs(
            position.quantity * exit_price, position.quantity, exit_price
        )

        # Update cash
        position.realized_pnl += realized_pnl - commission - spread_cost - slippage
        self.cash += position.quantity * exit_price - commission - spread_cost - slippage

        # Update tracking
        self.total_commission += commission
        self.total_slippage += slippage

        # Log the trade
        self._log_trade(
            timestamp=timestamp,
            action='CLOSE',
            symbol=symbol,
            quantity=position.quantity,
            price=exit_price,
            entry_price=position.avg_price,
            realized_pnl=realized_pnl,
            commission=commission,
            spread_cost=spread_cost,
            slippage=slippage,
            reason=reason,
        )

        # Remove position
        del self.positions[symbol]

        self.logger.debug(f"Closed {symbol} position: {reason}, P&L: {realized_pnl:.2f}")

    def _calculate_financing_costs(self) -> float:
        """Calculate financing costs for all positions.

        Returns:
            Total financing cost for the period
        """
        if not self.funding_enabled:
            return 0.0

        total_cost = 0.0
        for position in self.positions.values():
            # Assume long positions, so financing cost is on full position value
            position_value = position.quantity * position.current_price
            daily_cost = position_value * self.interest_rate_daily
            total_cost += daily_cost

        return total_cost

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        # Get base metrics from parent class
        base_metrics = super().get_performance_metrics()

        # Additional metrics
        additional_metrics = {
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'final_cash': self.cash,
            'current_positions': len(self.positions),
        }

        return {**base_metrics, **additional_metrics}

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        super().reset()
        self.cash = self.initial_capital
        self.positions.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0

        self.logger.info("GeneralPortfolio reset to initial state")

    def get_position_value_current(self, symbol: str) -> float:
        """Get current market value of position for the given symbol.

        Args:
            symbol: The ticker symbol to get position value for

        Returns:
            Current market value of the position
        """
        if symbol not in self.positions:
            return 0.0

        position = self.positions[symbol]
        return position.quantity * position.current_price

    def get_position_allocation(self, symbol: str) -> float:
        """Get position as percentage of total portfolio value.

        Args:
            symbol: The ticker symbol to get allocation for

        Returns:
            Position value as percentage of total portfolio
        """
        total_value = self.total_value
        if total_value == 0:
            return 0.0

        position_value = self.get_position_value_current(symbol)
        return position_value / total_value

    def can_add_position(self, symbol: str) -> bool:
        """Check if we can add a position for the given symbol.

        Args:
            symbol: The ticker symbol

        Returns:
            True if we can add the position
        """
        return len(self.positions) < self.max_positions and symbol not in self.positions

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio summary
        """
        base_summary = super().get_summary()

        additional_summary = {
            'cash': self.cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'market_value': pos.quantity * pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            'current_positions': len(self.positions),
            'max_positions': self.max_positions,
        }

        return {**base_summary, **additional_summary}
