"""
Backtesting engine: simulates trading strategy and records all metrics.
Includes commission, slippage, position sizing, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_volume: float
    exit_price: float
    exit_volume: float
    pnl: float
    commission: float
    slippage: float
    return_pct: float


class PositionSizer:
    """
    Dynamic position sizing based on volatility and account size.
    Reference: Kelly criterion, fixed fractional sizing
    """
    
    def __init__(self, initial_capital: float = 1000000, max_position_pct: float = 0.02):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.current_capital = initial_capital
    
    def compute_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute Kelly criterion optimal position size.
        Kelly % = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        """
        if avg_win <= 0 or avg_loss <= 0:
            return 0.02  # Default to 2%
        
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Limit to max position
        return np.clip(kelly_pct, 0.01, self.max_position_pct)
    
    def compute_volatility_adjusted_size(self, price: float, volatility: float) -> float:
        """
        Adjust position size inversely with volatility.
        Higher volatility = smaller position
        """
        if volatility <= 0:
            return self.current_capital * self.max_position_pct / price
        
        vol_adjustment = 1.0 / (1.0 + volatility)
        position_size = self.current_capital * self.max_position_pct * vol_adjustment / price
        
        return position_size


class RiskManager:
    """Risk management: stops, limits, portfolio hedging."""
    
    def __init__(self, max_drawdown: float = 0.15, stop_loss_pct: float = 0.02):
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
    
    def check_stop_loss(self, entry_price: float, current_price: float) -> bool:
        """Check if stop loss is hit."""
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= self.stop_loss_pct
    
    def check_max_drawdown(self, current_equity: float) -> bool:
        """Check if max drawdown limit is exceeded."""
        if current_equity < self.peak_equity:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        
        return self.current_drawdown > self.max_drawdown
    
    def compute_portfolio_heat(self, positions: List[Tuple[str, float, float]]) -> float:
        """
        Compute total portfolio risk (sum of position risks).
        positions: list of (symbol, position_size, volatility)
        """
        total_heat = 0.0
        for symbol, pos_size, vol in positions:
            position_heat = abs(pos_size) * vol
            total_heat += position_heat
        
        return total_heat


class BacktestingEngine:
    """Main backtesting engine."""
    
    def __init__(self, 
                initial_capital: float = 1000000,
                commission: float = 0.001,  # 10 bps
                slippage: float = 0.0005):  # 5 bps
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.position_sizer = PositionSizer(initial_capital=initial_capital)
        self.risk_manager = RiskManager(max_drawdown=0.15, stop_loss_pct=0.02)
        
        self.trades: List[Trade] = []
        self.equity_curve = [initial_capital]
        self.daily_returns = []
        self.positions: Dict[str, Tuple[float, float]] = {}  # symbol -> (shares, entry_price)
        
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
    
    def open_position(self, 
                     timestamp: datetime,
                     symbol: str,
                     direction: str,
                     entry_price: float,
                     signal_strength: float = 1.0) -> None:
        """Open a new position."""
        
        # Compute position size
        position_size = self.position_sizer.compute_volatility_adjusted_size(
            entry_price, 0.15)  # Assume 15% volatility
        
        position_size *= signal_strength  # Scale by signal strength
        
        # Apply slippage
        slipped_price = entry_price * (1 + self.slippage)
        
        # Cost including commission
        trade_cost = position_size * slipped_price * (1 + self.commission)
        
        if trade_cost <= self.current_capital:
            self.positions[symbol] = (position_size, slipped_price)
            self.current_capital -= trade_cost
            
            logger.info(f"Opened {direction} {symbol} at {slipped_price:.2f}, "
                       f"size={position_size:.0f}")
        else:
            logger.warning(f"Insufficient capital to open position in {symbol}")
    
    def close_position(self,
                      timestamp: datetime,
                      symbol: str,
                      exit_price: float) -> Optional[Trade]:
        """Close an open position."""
        
        if symbol not in self.positions:
            return None
        
        position_size, entry_price = self.positions[symbol]
        
        # Apply slippage
        slipped_exit_price = exit_price * (1 - self.slippage)
        
        # Compute P&L
        gross_pnl = (slipped_exit_price - entry_price) * position_size
        commission_cost = position_size * slipped_exit_price * self.commission
        net_pnl = gross_pnl - commission_cost
        
        # Update capital
        self.current_capital += position_size * slipped_exit_price
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            direction='long',
            entry_price=entry_price,
            entry_volume=position_size,
            exit_price=slipped_exit_price,
            exit_volume=position_size,
            pnl=net_pnl,
            commission=commission_cost,
            slippage=position_size * (exit_price - slipped_exit_price),
            return_pct=net_pnl / (entry_price * position_size) * 100
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        logger.info(f"Closed {symbol} position, PnL={net_pnl:.2f}, return={trade.return_pct:.2f}%")
        
        return trade
    
    def mark_to_market(self, current_prices: Dict[str, float]) -> None:
        """Update position values based on current prices."""
        
        for symbol, (shares, entry_price) in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                unrealized_pnl = (current_price - entry_price) * shares
                
                # Update equity
                self.current_capital += unrealized_pnl
        
        # Update equity curve and returns
        self.equity_curve.append(self.current_capital)
        
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
        
        # Update drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        
        current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Compute backtest performance metrics."""
        
        if len(self.daily_returns) == 0:
            return {}
        
        returns = np.array(self.daily_returns)
        
        # Returns metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        annual_return = total_return * 252 / len(returns) if len(returns) > 0 else 0.0
        
        # Volatility metrics
        daily_volatility = np.std(returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - 0.05) / annual_volatility if annual_volatility > 0 else 0.0
        
        # Win rate
        win_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        win_rate = win_trades / len(self.trades) if len(self.trades) > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average trade
        avg_trade = np.mean([trade.pnl for trade in self.trades]) if len(self.trades) > 0 else 0.0
        
        # Recovery factor
        total_pnl = sum(trade.pnl for trade in self.trades)
        max_loss_trade = min([trade.pnl for trade in self.trades]) if len(self.trades) > 0 else 0.0
        recovery_factor = total_pnl / abs(max_loss_trade) if max_loss_trade < 0 else np.inf
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'average_trade': avg_trade,
            'num_trades': len(self.trades),
            'recovery_factor': recovery_factor,
        }
    
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive backtest report."""
        
        metrics = self.get_performance_metrics()
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                **metrics
            },
            'trades': [trade.__dict__ for trade in self.trades],
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
        }
        
        return report


class WalkForwardTester:
    """Walk-forward testing for robust backtesting."""
    
    def __init__(self, train_period: int = 252, test_period: int = 63):
        self.train_period = train_period  # 1 year
        self.test_period = test_period    # ~3 months
    
    def split_data(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into overlapping train/test windows.
        """
        n = len(data)
        splits = []
        
        for i in range(0, n - self.train_period - self.test_period, self.test_period):
            train_data = data.iloc[i:i + self.train_period]
            test_data = data.iloc[i + self.train_period:i + self.train_period + self.test_period]
            
            if len(train_data) == self.train_period and len(test_data) == self.test_period:
                splits.append((train_data, test_data))
        
        return splits
    
    def run_walk_forward_test(self, 
                             data: pd.DataFrame,
                             strategy_func: callable) -> Dict[str, any]:
        """
        Run walk-forward test on historical data.
        """
        
        splits = self.split_data(data)
        all_results = []
        
        for train_data, test_data in splits:
            # Train on historical data
            model = strategy_func(train_data)
            
            # Test on forward data
            backtester = BacktestingEngine()
            
            for idx, row in test_data.iterrows():
                prediction = model.predict(row)
                
                if prediction > 0.5:  # Long signal
                    backtester.open_position(
                        idx, 'NIFTY50', 'long',
                        row['close'], prediction - 0.5)
                
                backtester.mark_to_market({'NIFTY50': row['close']})
            
            # Close remaining positions
            if 'NIFTY50' in backtester.positions:
                backtester.close_position(test_data.index[-1], 'NIFTY50', 
                                         test_data['close'].iloc[-1])
            
            metrics = backtester.get_performance_metrics()
            all_results.append(metrics)
        
        # Aggregate results
        avg_metrics = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if key in r]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics
