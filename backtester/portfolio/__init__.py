"""Portfolio layer for portfolio management and risk handling."""

from backtester.portfolio.base_portfolio import BasePortfolio
from backtester.portfolio.dual_pool_portfolio import DualPoolPortfolio
from backtester.portfolio.general_portfolio import GeneralPortfolio
from backtester.portfolio.pool_state import PoolState
from backtester.portfolio.position import Position

__all__ = [
    'BasePortfolio',
    'Position',
    'PoolState',
    'GeneralPortfolio',
    'DualPoolPortfolio',
]

# For backward compatibility, also provide the original class names
DualPoolPortfolio.__name__ = 'DualPoolPortfolio'
GeneralPortfolio.__name__ = 'GeneralPortfolio'

# Type aliases for convenience
PortfolioType = BasePortfolio
