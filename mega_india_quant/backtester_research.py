"""
Research-Grade Backtester: Proper PnL, Sharpe, Sortino, max drawdown, win rate calculations.
Based on institutional quant backtesting standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_cost: float
    exit_proceeds: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    holding_days: int
    win: bool


class PerformanceCalculator:
    """Calculate institutional-grade performance metrics."""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Simple daily returns."""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Log returns for better properties."""
        return np.diff(np.log(prices))
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio = (avg_return - risk_free_rate) / std_dev
        Annualized for 252 trading days.
        """
        if len(returns) < 2:
            return 0.0
        
        daily_excess_return = np.mean(returns) - risk_free_rate / 252
        daily_std = np.std(returns)
        
        if daily_std == 0:
            return 0.0
        
        sharpe = daily_excess_return / daily_std * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Sortino Ratio = (avg_return - risk_free) / downside_deviation
        Only penalizes downside volatility.
        """
        if len(returns) < 2:
            return 0.0
        
        daily_excess = np.mean(returns) - risk_free_rate / 252
        
        # Downside deviation (negative returns only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            downside_std = 0.0
        else:
            downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return float('inf') if daily_excess > 0 else 0.0
        
        sortino = daily_excess / downside_std * np.sqrt(252)
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Maximum Drawdown = (peak - trough) / peak
        Measures largest percentage loss from peak.
        """
        if len(equity_curve) < 2:
            return 0.0
        
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """
        Calmar Ratio = Annual Return / Max Drawdown
        Measures return per unit of drawdown risk.
        """
        annual_return = np.sum(returns) * 252
        max_dd = PerformanceCalculator.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0 or max_dd > 0:
            return 0.0
        
        calmar = annual_return / abs(max_dd)
        return calmar
    
    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float:
        """Percentage of winning trades."""
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.win)
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades: List[Trade]) -> float:
        """Gross profit / Gross loss."""
        if len(trades) == 0:
            return 0.0
        
        gross_profit = sum(max(0, trade.net_pnl) for trade in trades)
        gross_loss = abs(sum(min(0, trade.net_pnl) for trade in trades))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_avg_trade_return(trades: List[Trade]) -> float:
        """Average return per trade."""
        if len(trades) == 0:
            return 0.0
        
        total_return = sum(trade.return_pct for trade in trades)
        return total_return / len(trades)


class ResearchBacktester:
    """Professional backtester with proper metrics."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades: List[Trade] = []
        self.positions: Dict[str, Tuple[int, float]] = {}  # {symbol: (qty, entry_price)}
        self.entry_dates: Dict[str, datetime] = {}
        
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.002  # 0.2%
        
        self.calculator = PerformanceCalculator()
    
    def execute_buy(self,
                   symbol: str,
                   entry_date: datetime,
                   price: float,
                   quantity: int) -> bool:
        """Execute a BUY trade."""
        if quantity <= 0 or price <= 0:
            return False
        
        # Apply slippage
        slipped_price = price * (1 + self.slippage_rate)
        
        # Apply commission
        cost = quantity * slipped_price * (1 + self.commission_rate)
        
        if cost <= self.current_capital:
            self.positions[symbol] = (quantity, slipped_price)
            self.entry_dates[symbol] = entry_date
            self.current_capital -= cost
            
            logger.info(f"BUY {quantity} {symbol} @ ${slipped_price:.2f}, cost: ${cost:.2f}")
            return True
        else:
            logger.warning(f"Insufficient capital to buy {quantity} {symbol}")
            return False
    
    def execute_sell(self,
                    symbol: str,
                    exit_date: datetime,
                    price: float) -> Optional[Trade]:
        """Execute a SELL trade and record the trade."""
        if symbol not in self.positions:
            return None
        
        quantity, entry_price = self.positions[symbol]
        entry_date = self.entry_dates[symbol]
        
        # Apply slippage
        slipped_price = price * (1 - self.slippage_rate)
        
        # Calculate P&L
        entry_cost = quantity * entry_price * (1 + self.commission_rate)
        exit_proceeds = quantity * slipped_price * (1 - self.commission_rate)
        
        gross_pnl = (slipped_price - entry_price) * quantity
        fees = quantity * (entry_price * self.commission_rate + slipped_price * self.commission_rate)
        net_pnl = exit_proceeds - entry_cost
        
        return_pct = (net_pnl / entry_cost) * 100 if entry_cost > 0 else 0
        holding_days = (exit_date - entry_date).days + 1
        
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            symbol=symbol,
            entry_price=entry_price,
            exit_price=slipped_price,
            quantity=quantity,
            entry_cost=entry_cost,
            exit_proceeds=exit_proceeds,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            return_pct=return_pct,
            holding_days=holding_days,
            win=net_pnl > 0
        )
        
        self.trades.append(trade)
        self.current_capital += exit_proceeds
        
        del self.positions[symbol]
        del self.entry_dates[symbol]
        
        logger.info(f"SELL {quantity} {symbol} @ ${slipped_price:.2f}, "
                   f"P&L: ${net_pnl:.2f} ({return_pct:.2f}%)")
        
        return trade
    
    def update_equity(self, mark_to_market_prices: Dict[str, float]):
        """Update equity curve with current market values."""
        portfolio_value = self.current_capital
        
        # Add unrealized P&L
        for symbol, (qty, entry_price) in self.positions.items():
            if symbol in mark_to_market_prices:
                current_price = mark_to_market_prices[symbol]
                unrealized = qty * (current_price - entry_price)
                portfolio_value += unrealized
        
        self.equity_curve.append(portfolio_value)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        if len(self.equity_curve) == 0:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns = self.calculator.calculate_returns(equity_array)
        
        # Total metrics
        total_return = (equity_array[-1] - self.initial_capital) / self.initial_capital
        annual_return = (equity_array[-1] / self.initial_capital) ** (252 / len(self.equity_curve)) - 1
        
        # Risk metrics
        sharpe = self.calculator.calculate_sharpe_ratio(returns)
        sortino = self.calculator.calculate_sortino_ratio(returns)
        max_dd = self.calculator.calculate_max_drawdown(equity_array)
        calmar = self.calculator.calculate_calmar_ratio(returns, equity_array)
        
        # Trade metrics
        win_rate = self.calculator.calculate_win_rate(self.trades)
        profit_factor = self.calculator.calculate_profit_factor(self.trades)
        avg_trade = self.calculator.calculate_avg_trade_return(self.trades)
        
        total_pnl = sum(trade.net_pnl for trade in self.trades)
        total_fees = sum(trade.fees for trade in self.trades)
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': equity_array[-1],
                'total_return': total_return,
                'annual_return': annual_return,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar,
                'volatility': np.std(returns) * np.sqrt(252),
            },
            'trade_metrics': {
                'total_trades': len(self.trades),
                'winning_trades': sum(1 for t in self.trades if t.win),
                'losing_trades': sum(1 for t in self.trades if not t.win),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_return': avg_trade,
                'avg_winner': np.mean([t.return_pct for t in self.trades if t.win]) if any(t.win for t in self.trades) else 0,
                'avg_loser': np.mean([t.return_pct for t in self.trades if not t.win]) if any(not t.win for t in self.trades) else 0,
            },
            'equity_curve': self.equity_curve,
        }
        
        return report
