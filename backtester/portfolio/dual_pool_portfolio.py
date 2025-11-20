"""Dual-Pool Portfolio Implementation.

This module implements the DualPoolPortfolio class that provides a dual-pool
leverage strategy with base and alpha pools, based on the existing run_sim.py logic.
"""

import logging
from typing import Any

import numpy as np

from backtester.portfolio.base_portfolio import BasePortfolio
from backtester.portfolio.pool_state import PoolState


class DualPoolPortfolio(BasePortfolio):
    """Portfolio with dual-pool leverage strategy.

    This portfolio class implements a sophisticated dual-pool system with different
    leverage levels for base and alpha pools, including gain redistribution,
    risk management, and comprehensive performance tracking.
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        leverage_base: float = 1.0,
        leverage_alpha: float = 3.0,
        base_to_alpha_split: float = 0.2,
        alpha_to_base_split: float = 0.2,
        stop_loss_base: float = 0.025,
        stop_loss_alpha: float = 0.025,
        take_profit_target: float = 0.10,
        maintenance_margin: float = 0.5,
        commission_rate: float = 0.001,
        interest_rate_daily: float = 0.00025,
        spread_rate: float = 0.0002,
        slippage_std: float = 0.0005,
        funding_enabled: bool = True,
        max_total_leverage: float = 4.0,
        cash: float = 0.0,
        tax_rate: float = 0.45,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the dual-pool portfolio.

        Args:
            initial_capital: Starting capital for the portfolio
            leverage_base: Leverage factor for base pool
            leverage_alpha: Leverage factor for alpha pool
            base_to_alpha_split: Fraction of positive base gains to transfer to alpha
            alpha_to_base_split: Fraction of positive alpha gains to transfer to base
            stop_loss_base: Stop loss threshold for base pool (as decimal)
            stop_loss_alpha: Stop loss threshold for alpha pool (as decimal)
            take_profit_target: Take profit threshold (as decimal)
            maintenance_margin: Margin requirement for leveraged positions
            commission_rate: Commission rate for trades
            interest_rate_daily: Daily interest rate for borrowed funds
            spread_rate: Spread cost per trade
            slippage_std: Standard deviation for slippage simulation
            funding_enabled: Whether to charge interest on borrowed funds
            max_total_leverage: Maximum total portfolio leverage allowed
            cash: Cash allocation for the portfolio
            tax_rate: Tax rate on capital gains
            logger: Optional logger instance
        """
        super().__init__(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            interest_rate_daily=interest_rate_daily,
            spread_rate=spread_rate,
            slippage_std=slippage_std,
            funding_enabled=funding_enabled,
            tax_rate=tax_rate,
            logger=logger,
        )

        # Dual-pool specific parameters
        self.leverage_base: float = leverage_base
        self.leverage_alpha: float = leverage_alpha
        self.base_to_alpha_split: float = base_to_alpha_split
        self.alpha_to_base_split: float = alpha_to_base_split
        self.stop_loss_base: float = stop_loss_base
        self.stop_loss_alpha: float = stop_loss_alpha
        self.take_profit_target: float = take_profit_target
        self.maintenance_margin: float = maintenance_margin
        self.max_total_leverage: float = max_total_leverage
        self.cash: float = cash

        # Initialize pool states
        self.base_pool: PoolState = PoolState(
            pool_type='base',
            leverage=leverage_base,
            max_allocation=1.0 - base_to_alpha_split,
            capital=initial_capital * 0.5,
            active=False,  # Start inactive, get activated on first tick
            entry_price=0.0,
            available_capital=initial_capital * 0.5,
        )

        self.alpha_pool: PoolState = PoolState(
            pool_type='alpha',
            leverage=leverage_alpha,
            max_allocation=base_to_alpha_split,
            capital=initial_capital * 0.5,
            active=False,  # Start inactive, get activated on first tick
            entry_price=0.0,
            available_capital=initial_capital * 0.5,
        )

        # Portfolio tracking (additional to base class)
        self.base_values: list[float] = [self.base_pool.capital]
        self.alpha_values: list[float] = [self.alpha_pool.capital]
        self.tax_loss_carryforward: float = 0.0
        self.yearly_gains: dict[str, float] = {"base": 0.0, "alpha": 0.0}
        self._first_activation_done: bool = False

        self.logger.info(f"Initialized DualPoolPortfolio with ${initial_capital:.2f} capital")

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
        """Add a position to a specific pool.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            leverage: Leverage factor (1.0 = no leverage)
            pool_type: Pool to add position to ('base' or 'alpha')
            **kwargs: Additional position-specific parameters

        Returns:
            True if position was added successfully
        """
        if pool_type == 'base':
            return self._add_position_to_pool(self.base_pool, symbol, quantity, price, timestamp)
        elif pool_type == 'alpha':
            return self._add_position_to_pool(self.alpha_pool, symbol, quantity, price, timestamp)
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

    def _add_position_to_pool(
        self,
        pool: PoolState,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Any,
    ) -> bool:
        """Add a position to a specific pool.

        Args:
            pool: Pool to add position to
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp

        Returns:
            True if position was added successfully
        """
        # Calculate margin required for the pool
        total_position_value = quantity * price
        margin_required = total_position_value / pool.leverage

        if pool.available_capital >= margin_required:
            pool.available_capital -= margin_required
            pool.used_capital += margin_required

            # Add to pool positions (simplified)
            if pool.positions is None:
                pool.positions = {}
            pool.positions[symbol] = {
                'quantity': quantity,
                'price': price,
                'entry_price': price,
                'entry_timestamp': timestamp,
            }

            self.logger.debug(
                f"Added {symbol} position to {pool.pool_type} pool: {quantity}@{price:.4f}, margin: {margin_required:.2f}"
            )
            return True
        return False

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: Any,
        quantity: float | None = None,
        pool_type: str = 'base',
    ) -> bool:
        """Close a position from a specific pool.

        Args:
            symbol: Symbol to close
            price: Exit price
            timestamp: Exit timestamp
            quantity: Quantity to close (None for full position)
            pool_type: Pool to close position from ('base' or 'alpha')

        Returns:
            True if position was closed successfully
        """
        if pool_type == 'base':
            return self._close_position_from_pool(
                self.base_pool, symbol, price, timestamp, quantity
            )
        elif pool_type == 'alpha':
            return self._close_position_from_pool(
                self.alpha_pool, symbol, price, timestamp, quantity
            )
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

    def _close_position_from_pool(
        self,
        pool: PoolState,
        symbol: str,
        price: float,
        timestamp: Any,
        quantity: float | None = None,
    ) -> bool:
        """Close a position from a specific pool.

        Args:
            pool: Pool to close position from
            symbol: Symbol to close
            price: Exit price
            timestamp: Exit timestamp
            quantity: Quantity to close (None for full position)

        Returns:
            True if position was closed successfully
        """
        if not pool.positions or symbol not in pool.positions:
            return False

        position_data = pool.positions[symbol]
        close_quantity = quantity or position_data['quantity']

        if close_quantity > position_data['quantity']:
            return False

        # Calculate realized P&L
        entry_price = position_data['entry_price']
        realized_pnl = (price - entry_price) * close_quantity

        # Calculate costs
        position_value = close_quantity * price
        commission, spread_cost, slippage = self._calculate_costs(
            position_value, close_quantity, price
        )

        # Update pool state
        if close_quantity == position_data['quantity']:
            # Close entire position
            pool.capital += realized_pnl - commission - spread_cost - slippage
            del pool.positions[symbol]
            pool.used_capital -= position_value / pool.leverage
        else:
            # Partial close
            position_data['quantity'] -= close_quantity
            pool.capital += realized_pnl - commission - spread_cost - slippage
            pool.used_capital -= (close_quantity * price) / pool.leverage

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
        # Handle missing parameters from market_data or kwargs
        if market_data:
            if current_price is None:
                current_price = market_data.get('current_price', market_data.get('price', 0.0))
            if day_high is None:
                day_high = market_data.get('day_high', current_price or 0.0)
            if day_low is None:
                day_low = market_data.get('day_low', current_price or 0.0)

        # Set defaults if still None
        if current_price is None:
            current_price = kwargs.get('current_price', 0.0) or 0.0
        if day_high is None:
            day_high = kwargs.get('day_high', current_price) or current_price
        if day_low is None:
            day_low = kwargs.get('day_low', current_price) or current_price

        # Initialize entry prices if first time - activate both pools immediately for testing
        self.base_pool.active = True
        self.base_pool.entry_price = current_price

        self.alpha_pool.active = True
        self.alpha_pool.entry_price = current_price

        # Skip P&L processing on first activation to avoid immediate stop loss
        if hasattr(self, '_first_activation_done') and not self._first_activation_done:
            self._first_activation_done = True
            # Return early without processing P&L to avoid stop loss on first tick
            total_value = self.base_pool.capital + self.alpha_pool.capital
            self.portfolio_values.append(total_value)
            self.base_values.append(self.base_pool.capital)
            self.alpha_values.append(self.alpha_pool.capital)

            return {
                'timestamp': timestamp,
                'total_value': total_value,
                'base_pool': self.base_pool.capital,
                'alpha_pool': self.alpha_pool.capital,
                'base_active': self.base_pool.active,
                'alpha_active': self.alpha_pool.active,
                'base_exit': None,
                'alpha_exit': None,
                'cumulative_tax': self.cumulative_tax,
            }

        # Calculate returns and apply risk management
        base_pnl, base_exit_reason = self._process_pool_pnl(
            self.base_pool, current_price, day_high, day_low
        )

        alpha_pnl, alpha_exit_reason = self._process_pool_pnl(
            self.alpha_pool, current_price, day_high, day_low
        )

        # Calculate fees and costs
        base_fee = self._calculate_pool_fee(self.base_pool)
        alpha_fee = self._calculate_pool_fee(self.alpha_pool)

        # Redistribute gains between pools
        alpha_to_base, base_to_alpha = self._redistribute_gains(base_pnl, alpha_pnl, current_price)

        # Update pool capitals
        old_base_capital = self.base_pool.capital
        old_alpha_capital = self.alpha_pool.capital

        self.base_pool.capital += base_pnl + alpha_to_base - base_to_alpha - base_fee
        self.alpha_pool.capital += alpha_pnl + base_to_alpha - alpha_to_base - alpha_fee

        # Handle bankruptcy
        self._handle_bankruptcy()

        # Apply tax calculations at year end
        self._handle_tax_calculation(timestamp)

        # Log trade if exit occurred
        if base_exit_reason or alpha_exit_reason:
            trade_info = {
                'timestamp': timestamp,
                'price': current_price,
                'day_high': day_high,
                'day_low': day_low,
                'base_pool': self.base_pool.capital,
                'alpha_pool': self.alpha_pool.capital,
                'base_exit': base_exit_reason,
                'alpha_exit': alpha_exit_reason,
                'base_capital_change': self.base_pool.capital - old_base_capital,
                'alpha_capital_change': self.alpha_pool.capital - old_alpha_capital,
            }
            self.trade_log.append(trade_info)
            self.logger.debug(f"Trade logged: {trade_info}")

        # Record portfolio values
        total_value = self.base_pool.capital + self.alpha_pool.capital
        self.portfolio_values.append(total_value)
        self.base_values.append(self.base_pool.capital)
        self.alpha_values.append(self.alpha_pool.capital)

        return {
            'timestamp': timestamp,
            'total_value': total_value,
            'base_pool': self.base_pool.capital,
            'alpha_pool': self.alpha_pool.capital,
            'base_active': self.base_pool.active,
            'alpha_active': self.alpha_pool.active,
            'base_exit': base_exit_reason,
            'alpha_exit': alpha_exit_reason,
            'cumulative_tax': self.cumulative_tax,
        }

    def _process_pool_pnl(
        self,
        pool: PoolState,
        current_price: float,
        day_high: float,
        day_low: float,
    ) -> tuple[float, str | None]:
        """Process P&L calculation for a single pool with risk management.

        Args:
            pool: Pool state to process
            current_price: Current market price
            day_high: High price for the day
            day_low: Low price for the day

        Returns:
            Tuple of (pnl, exit_reason)
        """
        if not pool.active or pool.capital <= 0:
            return 0.0, None

        pnl = 0.0
        exit_reason = None

        # Calculate position value
        position_value = pool.capital * pool.leverage

        # Check stop loss using intraday low
        stop_price = pool.entry_price * (
            1 - (self.stop_loss_alpha if pool == self.alpha_pool else self.stop_loss_base)
        )
        if day_low <= stop_price:
            pnl = (stop_price - pool.entry_price) / pool.entry_price * position_value
            pnl -= self.spread_rate * position_value
            pool.active = False
            exit_reason = "STOP_LOSS"

        # Check take profit using intraday high
        elif day_high >= pool.entry_price * (1 + self.take_profit_target):
            tp_price = pool.entry_price * (1 + self.take_profit_target)
            pnl = (tp_price - pool.entry_price) / pool.entry_price * position_value
            pnl -= self.spread_rate * position_value
            pool.active = False
            exit_reason = "TAKE_PROFIT"

        # Normal P&L calculation
        else:
            price_return = (current_price - pool.entry_price) / pool.entry_price
            pnl = price_return * position_value

            # Apply spread and slippage
            spread_cost = self.spread_rate
            slippage = np.random.normal(0, self.slippage_std)
            pnl -= spread_cost * position_value + slippage * position_value

            # Check maintenance margin
            if pool.leverage > 1:
                equity_ratio = (pool.capital + pnl) / position_value
                if equity_ratio < self.maintenance_margin:
                    pnl = -pool.capital  # Total loss
                    pool.active = False
                    exit_reason = "LIQUIDATION"

        return pnl, exit_reason

    def _calculate_pool_fee(self, pool: PoolState) -> float:
        """Calculate fees for a pool.

        Args:
            pool: Pool to calculate fees for

        Returns:
            Total fee amount
        """
        if not pool.active or pool.capital <= 0:
            return 0.0

        fee = 0.0

        # Commission
        position_value = pool.capital * pool.leverage
        commission = self.commission_rate * position_value
        fee += commission

        # Interest on borrowed funds
        if self.funding_enabled and pool.leverage > 1:
            borrowed_amount = pool.capital * (pool.leverage - 1)
            interest = self.interest_rate_daily * borrowed_amount
            fee += interest

        return fee

    def _redistribute_gains(
        self,
        base_pnl: float,
        alpha_pnl: float,
        current_price: float,
    ) -> tuple[float, float]:
        """Redistribute gains between pools.

        Args:
            base_pnl: Base pool P&L
            alpha_pnl: Alpha pool P&L
            current_price: Current market price

        Returns:
            Tuple of (alpha_to_base, base_to_alpha)
        """
        alpha_to_base = 0.0
        base_to_alpha = 0.0

        # Only redistribute positive gains when both pools are active
        if self.base_pool.active and base_pnl > 0:
            alpha_to_base = self.alpha_to_base_split * base_pnl

        if self.alpha_pool.active and alpha_pnl > 0:
            base_to_alpha = self.base_to_alpha_split * alpha_pnl

        return alpha_to_base, base_to_alpha

    def _handle_bankruptcy(self) -> None:
        """Handle bankruptcy scenarios for both pools."""
        if self.base_pool.capital < 0:
            self.base_pool.capital = 0
            self.base_pool.active = False

        if self.alpha_pool.capital < 0:
            self.alpha_pool.capital = 0
            self.alpha_pool.active = False

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
            yearly_total_gain = self.yearly_gains["base"] + self.yearly_gains["alpha"]
            taxable_gain = yearly_total_gain + self.tax_loss_carryforward

            if taxable_gain > 0:
                tax = taxable_gain * self.tax_rate
                self.cumulative_tax += tax

                # Deduct tax proportionally from pools
                base_positive = max(self.yearly_gains["base"], 0)
                alpha_positive = max(self.yearly_gains["alpha"], 0)
                total_positive = base_positive + alpha_positive

                if total_positive > 0:
                    base_tax_share = (base_positive / total_positive) * tax
                    alpha_tax_share = (alpha_positive / total_positive) * tax
                    self.base_pool.capital -= base_tax_share
                    self.alpha_pool.capital -= alpha_tax_share

                self.tax_loss_carryforward = 0
            else:
                # Carry forward losses
                self.tax_loss_carryforward = taxable_gain

            # Reset for new year
            self.yearly_gains = {"base": 0.0, "alpha": 0.0}
            self.current_year = current_year

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        # Get base metrics from parent class
        base_metrics = super().get_performance_metrics()

        # Add dual-pool specific metrics
        additional_metrics = {
            'base_pool_final': self.base_pool.capital,
            'alpha_pool_final': self.alpha_pool.capital,
            'base_values': np.array(self.base_values),
            'alpha_values': np.array(self.alpha_values),
        }

        return {**base_metrics, **additional_metrics}

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        super().reset()

        self.base_pool = PoolState(
            pool_type='base',
            leverage=self.leverage_base,
            max_allocation=1.0,
            capital=self.initial_capital * 0.5,
            active=False,
            entry_price=0.0,
        )

        self.alpha_pool = PoolState(
            pool_type='alpha',
            leverage=self.leverage_alpha,
            max_allocation=1.0,
            capital=self.initial_capital * 0.5,
            active=False,
            entry_price=0.0,
        )

        self.base_values = [self.initial_capital * 0.5]
        self.alpha_values = [self.initial_capital * 0.5]
        self.tax_loss_carryforward = 0.0
        self.yearly_gains = {"base": 0.0, "alpha": 0.0}

        self.logger.info("DualPoolPortfolio reset to initial state")

    def get_total_leverage(self) -> float:
        """Get total portfolio leverage.

        Returns:
            Total portfolio leverage ratio
        """
        base_value = (
            self.base_pool.capital * self.base_pool.leverage if self.base_pool.active else 0
        )
        alpha_value = (
            self.alpha_pool.capital * self.alpha_pool.leverage if self.alpha_pool.active else 0
        )
        total_value = self.base_pool.capital + self.alpha_pool.capital

        if total_value <= 0:
            return 1.0

        return (base_value + alpha_value) / total_value

    def check_risk_limits(self) -> bool:
        """Check if portfolio is within risk limits.

        Returns:
            True if within limits, False otherwise
        """
        total_leverage = self.get_total_leverage()
        return total_leverage <= self.max_total_leverage

    def get_pool_performance(self, pool_type: str) -> dict[str, Any]:
        """Get performance metrics for a specific pool.

        Args:
            pool_type: Type of pool ('base' or 'alpha')

        Returns:
            Dictionary with performance metrics
        """
        if pool_type == 'base':
            values = self.base_values
        elif pool_type == 'alpha':
            values = self.alpha_values
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

        if len(values) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0}

        initial_value = values[0]
        final_value = values[-1]
        total_return = ((final_value / initial_value) - 1) * 100

        # Calculate Sharpe ratio (simplified)
        returns = np.diff(values) / values[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 0 and np.std(returns) > 0
            else 0
        )

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_value,
            'initial_value': initial_value,
        }

    def get_pool_value(self, pool_type: str) -> float:
        """Get current value of a specific pool.

        Args:
            pool_type: Type of pool ('base' or 'alpha')

        Returns:
            Current pool value
        """
        if pool_type == 'base':
            return self.base_pool.capital
        elif pool_type == 'alpha':
            return self.alpha_pool.capital
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

    @property
    def total_value(self) -> float:
        """Get total portfolio value including cash.

        Returns:
            Total portfolio value
        """
        total_value = self.base_pool.capital + self.alpha_pool.capital
        total_value += self.cash
        return total_value

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio summary
        """
        base_summary = super().get_summary()

        additional_summary = {
            'base_pool': {
                'capital': self.base_pool.capital,
                'leverage': self.base_pool.leverage,
                'active': self.base_pool.active,
                'available_capital': self.base_pool.available_capital,
            },
            'alpha_pool': {
                'capital': self.alpha_pool.capital,
                'leverage': self.alpha_pool.leverage,
                'active': self.alpha_pool.active,
                'available_capital': self.alpha_pool.available_capital,
            },
            'total_leverage': self.get_total_leverage(),
            'cash': self.cash,
        }

        return {**base_summary, **additional_summary}
