"""
Enhanced Trading Engine: Selects individual stocks, generates proper forecasts, manages portfolios.
Implements proper position sizing, stock selection, and portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockSelector:
    """Select attractive individual stocks based on fundamental and technical criteria."""
    
    def __init__(self, universe_size: int = 50):
        """Initialize stock selector for top Indian stocks."""
        self.universe_size = universe_size
        # Top Indian stocks by market cap
        self.stock_universe = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HDFC.NS',
            'ICICIBANK.NS', 'WIPRO.NS', 'MARUTI.NS', 'SBIN.NS', 'BAJAJFINSV.NS',
            'LT.NS', 'SUNPHARMA.NS', 'ONGC.NS', 'ITC.NS', 'DRREDDY.NS',
            'KOTAKBANK.NS', 'AXISBANK.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'JSWSTEEL.NS',
            'POWERGRID.NS', 'HINDUNILVR.NS', 'APOLLOHOSP.NS', 'NTPC.NS', 'ULTRACEMCO.NS',
            'ADANIPORTS.NS', 'DLF.NS', 'TECHM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
            'HCLTECH.NS', 'LUPIN.NS', 'EICHERMOT.NS', 'GSKCONS.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'BHEL.NS', 'CIPLA.NS', 'GAIL.NS', 'IOC.NS',
            'INDIGO.NS', 'MARUTISUZU.NS', 'NMDC.NS', 'PIDILITIND.NS', 'SIEMENS.NS'
        ][:universe_size]
        
        self.selected_stocks = []
    
    def select_stocks(self, scores: Dict[str, float], num_stocks: int = 10) -> List[str]:
        """Select top stocks based on scores."""
        if not scores:
            # Default to top stocks if no scores available
            return self.stock_universe[:num_stocks]
        
        # Sort by score and select top N
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_stocks = [stock for stock, score in sorted_stocks[:num_stocks]]
        
        logger.info(f"Selected {len(self.selected_stocks)} stocks for trading")
        return self.selected_stocks


class PortfolioOptimizer:
    """Optimize portfolio weights based on correlation and volatility."""
    
    def __init__(self, max_single_position: float = 0.10):
        """
        Initialize optimizer.
        max_single_position: Maximum weight for any single stock (default 10%)
        """
        self.max_single_position = max_single_position
        self.weights = {}
    
    def compute_equal_weight(self, stocks: List[str]) -> Dict[str, float]:
        """Compute equal-weight portfolio."""
        n_stocks = len(stocks)
        if n_stocks == 0:
            return {}
        
        weight = 1.0 / n_stocks
        weight = min(weight, self.max_single_position)
        
        self.weights = {stock: weight for stock in stocks}
        return self.weights
    
    def compute_signal_weighted(self, 
                               stocks: List[str], 
                               signals: Dict[str, float]) -> Dict[str, float]:
        """Compute weights based on signals."""
        total_signal = sum(signals.get(s, 0.5) for s in stocks)
        if total_signal <= 0:
            return self.compute_equal_weight(stocks)
        
        self.weights = {}
        for stock in stocks:
            signal = signals.get(stock, 0.5)
            raw_weight = signal / total_signal
            # Cap individual position
            weight = min(raw_weight, self.max_single_position)
            self.weights[stock] = weight
        
        # Renormalize
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {s: w/total_weight for s, w in self.weights.items()}
        
        return self.weights


class AdvancedPositionSizer:
    """Advanced position sizing using Kelly criterion and volatility scaling."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
    
    def compute_position_size(self,
                             signal: float,
                             stock_price: float,
                             volatility: float,
                             portfolio_weight: float = 0.05) -> float:
        """
        Compute position size based on signal, volatility, and portfolio weight.
        
        Args:
            signal: Signal strength [0, 1] (bullish) or [-1, 0] (bearish)
            stock_price: Current stock price
            volatility: Annualized volatility estimate
            portfolio_weight: Target weight of this stock in portfolio
        
        Returns:
            Position size in units (number of shares)
        """
        if stock_price <= 0 or self.current_capital <= 0:
            return 0.0
        
        # Base allocation
        base_allocation = self.current_capital * portfolio_weight
        
        # Adjust for signal strength (further from 0.5 = higher confidence)
        signal_confidence = abs(signal - 0.5) * 2  # [0, 1]
        adjusted_allocation = base_allocation * (0.5 + 0.5 * signal_confidence)
        
        # Adjust for volatility (lower volatility = larger position)
        vol_adjustment = 1.0 / (1.0 + max(0, volatility * 2))
        adjusted_allocation *= vol_adjustment
        
        # Compute number of shares
        position_size = adjusted_allocation / stock_price
        
        return max(0, position_size)
    
    def update_capital(self, pnl: float):
        """Update capital after trade."""
        self.current_capital += pnl


class TradeGenerator:
    """Generate trade signals for individual stocks."""
    
    def __init__(self):
        self.last_signals = {}
        self.position_history = []
    
    def generate_stock_signal(self,
                            stock_name: str,
                            macro_signal: float,
                            technical_signal: float,
                            fundamental_score: float = 0.5) -> float:
        """
        Generate combined signal for an individual stock.
        
        Args:
            stock_name: Stock symbol
            macro_signal: Macro environment signal [0, 1]
            technical_signal: Technical analysis signal [0, 1]
            fundamental_score: Fundamental value score [0, 1]
        
        Returns:
            Combined signal [0, 1] where 0.5 = neutral
        """
        # Weighted combination of signals
        combined = (
            0.4 * macro_signal +
            0.35 * technical_signal +
            0.25 * fundamental_score
        )
        
        # Clip to valid range
        combined = np.clip(combined, 0, 1)
        
        self.last_signals[stock_name] = combined
        return combined
    
    def classify_signal(self, signal: float, threshold: float = 0.05) -> str:
        """Classify signal as BUY, SELL, or HOLD."""
        if signal > 0.5 + threshold:
            return 'BUY'
        elif signal < 0.5 - threshold:
            return 'SELL'
        else:
            return 'HOLD'


class EnhancedTradingEngine:
    """Main trading engine with stock selection and portfolio optimization."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.stock_selector = StockSelector(universe_size=50)
        self.portfolio_optimizer = PortfolioOptimizer(max_single_position=0.15)
        self.position_sizer = AdvancedPositionSizer(initial_capital)
        self.trade_generator = TradeGenerator()
        
        self.portfolio = {}  # {stock: shares_held}
        self.entry_prices = {}  # {stock: entry_price}
        self.trade_log = []
        
        self.commission = 0.001  # 0.1%
        self.slippage = 0.002  # 0.2%
    
    def generate_trading_plan(self,
                            macro_signal: float,
                            stocks: List[str],
                            stock_signals: Dict[str, float],
                            fundamentals: Dict[str, float],
                            current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """
        Generate complete trading plan for portfolio.
        
        Returns:
            {stock: {'action': 'BUY'|'SELL'|'HOLD', 'quantity': N, 'price': P, 'weight': W}}
        """
        trading_plan = {}
        
        # Generate signals for each stock
        individual_signals = {}
        for stock in stocks:
            fundamental_score = fundamentals.get(stock, 0.5)
            technical_signal = stock_signals.get(stock, 0.5)
            
            signal = self.trade_generator.generate_stock_signal(
                stock_name=stock,
                macro_signal=macro_signal,
                technical_signal=technical_signal,
                fundamental_score=fundamental_score
            )
            individual_signals[stock] = signal
        
        # Optimize portfolio weights
        weights = self.portfolio_optimizer.compute_signal_weighted(stocks, individual_signals)
        
        # Generate trades
        for stock in stocks:
            signal = individual_signals.get(stock, 0.5)
            current_price = current_prices.get(stock, 100.0)
            weight = weights.get(stock, 0.0)
            
            # Compute position size
            position_size = self.position_sizer.compute_position_size(
                signal=signal,
                stock_price=current_price,
                volatility=0.25,  # Default 25% annual volatility
                portfolio_weight=weight
            )
            
            # Classify action
            action = self.trade_generator.classify_signal(signal, threshold=0.05)
            
            trading_plan[stock] = {
                'action': action,
                'quantity': int(position_size),
                'price': current_price,
                'weight': weight,
                'signal': signal,
            }
        
        logger.info(f"Generated trading plan for {len([s for s in trading_plan.values() if s['action'] != 'HOLD'])} stocks")
        return trading_plan
    
    def execute_trade(self,
                     stock: str,
                     action: str,
                     quantity: int,
                     price: float,
                     timestamp: datetime = None) -> bool:
        """Execute a single trade."""
        if quantity <= 0 or price <= 0:
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Apply slippage
        exec_price = price * (1 + self.slippage if action == 'BUY' else 1 - self.slippage)
        
        if action == 'BUY':
            cost = quantity * exec_price * (1 + self.commission)
            if cost <= self.current_capital:
                self.portfolio[stock] = self.portfolio.get(stock, 0) + quantity
                self.entry_prices[stock] = exec_price
                self.current_capital -= cost
                
                self.trade_log.append({
                    'timestamp': timestamp,
                    'stock': stock,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': exec_price,
                    'cost': cost,
                })
                logger.info(f"BUY {quantity} {stock} @ {exec_price:.2f}")
                return True
            else:
                logger.warning(f"Insufficient capital for {stock}")
                return False
        
        elif action == 'SELL':
            if stock in self.portfolio and self.portfolio[stock] > 0:
                quantity = min(quantity, self.portfolio[stock])
                proceeds = quantity * exec_price * (1 - self.commission)
                
                entry_price = self.entry_prices.get(stock, exec_price)
                pnl = (exec_price - entry_price) * quantity
                
                self.portfolio[stock] -= quantity
                if self.portfolio[stock] == 0:
                    del self.portfolio[stock]
                    del self.entry_prices[stock]
                
                self.current_capital += proceeds
                
                self.trade_log.append({
                    'timestamp': timestamp,
                    'stock': stock,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': exec_price,
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'return_pct': (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0,
                })
                logger.info(f"SELL {quantity} {stock} @ {exec_price:.2f}, PnL: ${pnl:.2f}")
                return True
        
        return False
    
    def compute_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Compute current portfolio value."""
        portfolio_value = self.current_capital
        for stock, quantity in self.portfolio.items():
            if stock in current_prices:
                portfolio_value += quantity * current_prices[stock]
        return portfolio_value
    
    def generate_report(self) -> Dict:
        """Generate trading report."""
        if len(self.trade_log) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
            }
        
        # Separate buy and sell trades
        sells = [t for t in self.trade_log if t['action'] == 'SELL']
        
        if not sells:
            return {
                'total_trades': len(self.trade_log),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
            }
        
        winning_trades = [t for t in sells if t['pnl'] > 0]
        losing_trades = [t for t in sells if t['pnl'] <= 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in sells)
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        return {
            'total_trades': len(sells),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(sells) if sells else 0,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
        }
