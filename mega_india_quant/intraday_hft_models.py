"""
Intraday HFT Models: Microprice forecasting, OFI, Hawkes process, queue reaction, execution.
References: Almgren-Chriss (2000), Kyle (1985), Hasbrouck (2007), Hawkes processes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicropriceForecaster:
    """
    Forecast microprice (fair value between bid/ask spread).
    Based on order flow and quote dynamics.
    """
    
    def __init__(self, window: int = 60):
        self.window = window
        self.bid_history = []
        self.ask_history = []
        self.volume_history = []
    
    def compute_microprice(self, bid: float, ask: float, 
                          bid_volume: float, ask_volume: float) -> float:
        """Weighted mid-price based on volumes (Kyle 1985)."""
        if bid_volume + ask_volume == 0:
            return (bid + ask) / 2
        
        weight = ask_volume / (bid_volume + ask_volume)
        microprice = bid * weight + ask * (1 - weight)
        return microprice
    
    def forecast_next_microprice(self, 
                                current_bid: float,
                                current_ask: float,
                                bid_vol: float,
                                ask_vol: float,
                                recent_trades: List[float]) -> float:
        """Forecast next microprice using order imbalance."""
        current_micro = self.compute_microprice(current_bid, current_ask, bid_vol, ask_vol)
        
        # Order flow imbalance (aggressiveness)
        spread = current_ask - current_bid
        imbalance = (ask_vol - bid_vol) / (ask_vol + bid_vol + 1e-8)
        
        # Momentum from recent trades
        if len(recent_trades) > 0:
            momentum = np.mean(np.diff(recent_trades[-10:]))
        else:
            momentum = 0
        
        # Forecast: microprice + momentum + spread effect
        forecast = current_micro + 0.5 * momentum + 0.3 * imbalance * spread
        
        return forecast


class OrderFlowIndicator:
    """Order Flow Imbalance (OFI) model for intraday prediction."""
    
    def __init__(self, window: int = 30):
        self.window = window
        self.ofi_history = []
    
    def compute_ofi(self, 
                   buy_volume: float,
                   sell_volume: float) -> float:
        """Order Flow Imbalance = (Buy Vol - Sell Vol) / Total Vol."""
        total = buy_volume + sell_volume
        if total == 0:
            return 0
        
        ofi = (buy_volume - sell_volume) / total
        self.ofi_history.append(ofi)
        
        if len(self.ofi_history) > self.window:
            self.ofi_history.pop(0)
        
        return ofi
    
    def predict_direction(self, ofi: float) -> Tuple[float, str]:
        """Predict price direction from OFI."""
        if len(self.ofi_history) < 5:
            return 0.5, 'NEUTRAL'
        
        # Mean reversion tendency in OFI
        ofi_mean = np.mean(self.ofi_history)
        ofi_std = np.std(self.ofi_history)
        
        if ofi_std > 0:
            z_score = (ofi - ofi_mean) / ofi_std
        else:
            z_score = 0
        
        # Positive OFI → likely up
        signal = 0.5 + 0.3 * np.tanh(z_score)
        signal = np.clip(signal, 0, 1)
        
        direction = 'UP' if signal > 0.6 else ('DOWN' if signal < 0.4 else 'NEUTRAL')
        return signal, direction


class HawkesProcessVolatilityBursts:
    """Detect volatility bursts using Hawkes point process."""
    
    def __init__(self, base_intensity: float = 0.1, decay: float = 0.9):
        self.base_intensity = base_intensity
        self.decay = decay
        self.trade_times = []
        self.volatility_bursts = []
    
    def detect_burst(self, current_volatility: float, 
                    historical_vol: float,
                    recent_large_trades: int) -> Tuple[float, bool]:
        """Detect if current volatility is burst (self-exciting process)."""
        
        # Hawkes intensity = base + sum of decayed impacts
        intensity = self.base_intensity
        
        # Recent large trades increase intensity (self-exciting)
        for _ in range(recent_large_trades):
            intensity += 0.3 * (self.decay ** np.arange(1, 11)).sum()
        
        # Compare to historical
        burst_threshold = historical_vol * 1.5
        is_burst = current_volatility > burst_threshold
        
        burst_probability = min(intensity, 1.0)
        
        if is_burst:
            self.volatility_bursts.append(burst_probability)
        
        return burst_probability, is_burst


class QueueReactionModel:
    """Model price reaction to order queue depth changes."""
    
    def __init__(self):
        self.bid_queue_history = []
        self.ask_queue_history = []
        self.price_history = []
    
    def compute_queue_imbalance(self, bid_queue: int, ask_queue: int) -> float:
        """Queue imbalance signal."""
        total = bid_queue + ask_queue
        if total == 0:
            return 0
        return (bid_queue - ask_queue) / total
    
    def predict_price_reaction(self,
                              current_price: float,
                              bid_queue: int,
                              ask_queue: int,
                              queue_change: int) -> float:
        """Predict price reaction to queue changes."""
        
        imbalance = self.compute_queue_imbalance(bid_queue, ask_queue)
        
        # Larger ask queue relative to bid = downward pressure
        price_pressure = -imbalance * 0.01  # basis points
        
        # Queue depth increase suggests persistence
        queue_momentum = np.sign(queue_change) * min(abs(queue_change) / 1000, 0.5)
        
        forecast_price = current_price * (1 + price_pressure + queue_momentum)
        
        return forecast_price


class AlmgrenChrissExecution:
    """
    Optimal execution model: minimize market impact + volatility cost.
    Reference: Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions"
    """
    
    def __init__(self, initial_quantity: int, execution_window: int = 60):
        self.initial_quantity = initial_quantity
        self.execution_window = execution_window
        self.remaining = initial_quantity
        self.scheduled_trades = []
    
    def compute_optimal_schedule(self,
                                volatility: float,
                                price_impact_coefficient: float = 0.1) -> List[int]:
        """
        Compute optimal execution schedule.
        Trade-off: permanent impact (price moves away) vs temporary impact (spread).
        """
        
        # Risk aversion parameter
        lambda_param = price_impact_coefficient / (volatility ** 2)
        
        # Optimal schedule is proportional to hyperbolic sine
        times = np.linspace(0, self.execution_window, self.execution_window)
        
        # ALMGREN-CHRISS FORMULA: v(t) = V / T * cosh(lambda * (t - T/2)) / sinh(lambda * T/2)
        T = self.execution_window
        if lambda_param * T < 20:  # Avoid overflow
            denominator = np.sinh(lambda_param * T / 2)
            if abs(denominator) > 1e-8:
                schedule = self.initial_quantity / T * np.cosh(lambda_param * (times - T/2)) / denominator
            else:
                schedule = np.full_like(times, self.initial_quantity / self.execution_window, dtype=float)
        else:
            schedule = np.full_like(times, self.initial_quantity / self.execution_window, dtype=float)
        
        # Ensure positive trades
        schedule = np.maximum(schedule, 0)
        
        self.scheduled_trades = np.cumsum(schedule).astype(int)
        return self.scheduled_trades.tolist()
    
    def estimate_market_impact(self, quantity: int,
                              daily_volume: int) -> float:
        """Estimate permanent price impact (basis points)."""
        if daily_volume == 0:
            return 0
        
        # Market impact increases with order size relative to volume
        participation_rate = quantity / daily_volume
        
        # Empirical formula: impact ≈ sqrt(participation_rate)
        impact_bps = 10 * np.sqrt(participation_rate)  # 10 bps per sqrt(participation)
        
        return impact_bps


class IntradayPredictionEngine:
    """Complete intraday ML system combining all HFT models."""
    
    def __init__(self):
        self.microprice = MicropriceForecaster()
        self.ofi = OrderFlowIndicator()
        self.hawkes = HawkesProcessVolatilityBursts()
        self.queue_model = QueueReactionModel()
        self.execution = AlmgrenChrissExecution(10000)
    
    def generate_intraday_signal(self,
                               bid: float,
                               ask: float,
                               bid_vol: int,
                               ask_vol: int,
                               bid_queue: int,
                               ask_queue: int,
                               recent_vol: float,
                               historical_vol: float,
                               recent_trades: List[float]) -> Dict[str, float]:
        """Generate comprehensive intraday signal."""
        
        signals = {}
        
        # Microprice forecast
        microprice = self.microprice.compute_microprice(bid, ask, bid_vol, ask_vol)
        forecast_micro = self.microprice.forecast_next_microprice(
            bid, ask, bid_vol, ask_vol, recent_trades
        )
        signals['microprice_signal'] = 0.5 + 0.4 * np.tanh((forecast_micro - microprice) / (ask - bid))
        
        # OFI signal
        ofi = self.ofi.compute_ofi(bid_vol, ask_vol)
        ofi_signal, direction = self.ofi.predict_direction(ofi)
        signals['ofi_signal'] = ofi_signal
        
        # Hawkes volatility burst
        large_trades = 5  # dummy
        burst_prob, is_burst = self.hawkes.detect_burst(recent_vol, historical_vol, large_trades)
        signals['burst_probability'] = burst_prob
        
        # Queue reaction
        queue_imbalance = self.queue_model.compute_queue_imbalance(bid_queue, ask_queue)
        signals['queue_signal'] = 0.5 + 0.4 * queue_imbalance
        
        # Ensemble intraday signal
        signals['intraday_ensemble'] = np.mean([
            signals['microprice_signal'],
            signals['ofi_signal'],
            0.5 if is_burst else 0.5 + signals['queue_signal'] - 0.5,
        ])
        
        return signals
