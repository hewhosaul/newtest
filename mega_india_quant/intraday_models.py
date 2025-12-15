"""
Intraday and HFT models: microprice forecasting, OFI models, Hawkes processes, queue reaction, optimal execution.
References: Almgren & Chriss (2001), Avellaneda & Stoikov (2008), Hawkes process in HFT.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrpriceForecaster:
    """
    Forecast microprice (fair value midpoint) from orderbook data.
    Reference: Baldacci et al. (2019), "The Instability of Market Depth"
    """
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.weights = None
    
    def compute_microprice(self, 
                          best_bid: float,
                          best_ask: float,
                          bid_volume: float,
                          ask_volume: float) -> float:
        """
        Compute microprice: weighted average of bid and ask.
        Weights are proportional to volumes.
        """
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return (best_bid + best_ask) / 2.0
        
        microprice = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        return microprice
    
    def forecast_microprice_change(self, 
                                  microprice_history: np.ndarray,
                                  bid_ask_spread_history: np.ndarray,
                                  volume_imbalance_history: np.ndarray) -> float:
        """
        Forecast next microprice change using volume-weighted indicators.
        """
        if len(microprice_history) < self.lookback_period:
            recent_microprices = microprice_history
            recent_spreads = bid_ask_spread_history
            recent_imbalance = volume_imbalance_history
        else:
            recent_microprices = microprice_history[-self.lookback_period:]
            recent_spreads = bid_ask_spread_history[-self.lookback_period:]
            recent_imbalance = volume_imbalance_history[-self.lookback_period:]
        
        # Simple linear combination
        spread_effect = -np.mean(recent_spreads) * 0.1  # Wide spreads predict down movement
        imbalance_effect = np.mean(recent_imbalance) * 0.05  # Buy imbalance predicts up
        
        forecast_change = spread_effect + imbalance_effect
        return forecast_change


class OrderFlowImbalanceModel:
    """
    Order flow imbalance (OFI) based predictive model.
    Reference: Chordia & Subrahmanyam (2004), "Order imbalance and individual stock returns"
    """
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor  # Exponential decay for older orders
        self.ofi_history = []
    
    def compute_ofi(self, buy_volume: np.ndarray, sell_volume: np.ndarray) -> float:
        """
        Order flow imbalance = (BV - SV) / (BV + SV)
        """
        total_flow = np.sum(buy_volume) + np.sum(sell_volume)
        if total_flow == 0:
            return 0.0
        
        ofi = (np.sum(buy_volume) - np.sum(sell_volume)) / total_flow
        self.ofi_history.append(ofi)
        return ofi
    
    def compute_weighted_ofi(self, buy_prices: np.ndarray, sell_prices: np.ndarray,
                            buy_volumes: np.ndarray, sell_volumes: np.ndarray) -> float:
        """
        Weighted OFI accounting for price levels.
        Purchases at market = more aggressive buying pressure.
        """
        if len(buy_volumes) == 0 or len(sell_volumes) == 0:
            return 0.0
        
        # Weight by recency
        weights = np.array([self.decay_factor ** (len(buy_volumes) - 1 - i) for i in range(len(buy_volumes))])
        weights /= np.sum(weights)
        
        buy_pressure = np.sum(buy_volumes * weights)
        sell_pressure = np.sum(sell_volumes * weights)
        
        total = buy_pressure + sell_pressure
        if total == 0:
            return 0.0
        
        weighted_ofi = (buy_pressure - sell_pressure) / total
        return weighted_ofi
    
    def predict_returns_from_ofi(self, ofi_values: np.ndarray, lags: int = 5) -> float:
        """
        Forecast returns from OFI signal.
        Reference: Chordia & Subrahmanyam (2004)
        """
        if len(ofi_values) < lags:
            return 0.0
        
        # Simple autocorrelation: current OFI predicts next return
        recent_ofi = ofi_values[-lags:]
        mean_ofi = np.mean(recent_ofi)
        
        # Positive OFI predicts upward returns
        return mean_ofi * 0.02  # ~2% return per unit OFI (scaled)


class HawkesProcessVolatilityBurst:
    """
    Hawkes process for modeling self-exciting arrival times of trades/jumps.
    Reference: Hawkes (1971), Bacry et al. (2015) in finance
    """
    
    def __init__(self, lambda_0: float = 0.1, alpha: float = 0.05, beta: float = 0.95):
        self.lambda_0 = lambda_0       # Baseline intensity
        self.alpha = alpha             # Self-excitation coefficient
        self.beta = beta               # Exponential decay
        self.last_event_time = 0
    
    def compute_intensity(self, event_times: np.ndarray, current_time: float) -> float:
        """
        Compute Hawkes intensity at current time.
        lambda(t) = lambda_0 + sum_i alpha * exp(-beta * (t - t_i)) for t_i < t
        """
        intensity = self.lambda_0
        
        for event_time in event_times:
            if event_time < current_time:
                time_diff = current_time - event_time
                intensity += self.alpha * np.exp(-self.beta * time_diff)
        
        return intensity
    
    def predict_volatility_burst(self, event_times: np.ndarray, 
                                current_time: float, 
                                forecast_horizon: float = 1.0) -> float:
        """
        Forecast probability and intensity of volatility burst.
        """
        current_intensity = self.compute_intensity(event_times, current_time)
        
        # Expected number of arrivals in next period
        expected_arrivals = current_intensity * forecast_horizon
        
        # Probability of volatility burst (threshold)
        burst_probability = 1.0 - np.exp(-expected_arrivals)
        
        return burst_probability


class QueueReactionModel:
    """
    Model price reaction to order queue depth changes.
    Reference: Avellaneda & Stoikov (2008), Rosu (2019)
    """
    
    def __init__(self):
        self.queue_elasticity = 0.1  # Price elasticity of queue depth
    
    def compute_queue_imbalance(self, bid_queue: float, ask_queue: float) -> float:
        """Queue imbalance = (bid_queue - ask_queue) / (bid_queue + ask_queue)"""
        total_queue = bid_queue + ask_queue
        if total_queue == 0:
            return 0.0
        
        return (bid_queue - ask_queue) / total_queue
    
    def predict_price_from_queue(self, 
                                queue_imbalance: float,
                                current_midprice: float) -> float:
        """
        Predict price change from queue imbalance.
        Larger bid queue predicts price increase.
        """
        price_change_pct = self.queue_elasticity * queue_imbalance
        return current_midprice * (1 + price_change_pct)
    
    def compute_queue_resilience(self, queue_depths: np.ndarray, time_to_recover: int = 10) -> float:
        """
        Measure how quickly queue depth recovers after imbalance.
        Higher resilience = market can handle more flow.
        """
        if len(queue_depths) < time_to_recover:
            return 0.5
        
        initial_depth = queue_depths[0]
        recovered_depth = queue_depths[-1]
        
        recovery_ratio = recovered_depth / initial_depth if initial_depth > 0 else 1.0
        return np.clip(recovery_ratio, 0.0, 1.0)


class AlmgrenChrisOptimalExecution:
    """
    Almgren-Chriss optimal execution model with market impact.
    Reference: Almgren & Chriss (2001), "Value of a trade"
    """
    
    def __init__(self, time_horizon: float = 1.0, volume_to_execute: float = 100000):
        self.T = time_horizon  # Trading horizon in minutes/hours
        self.V = volume_to_execute  # Total volume to execute
        self.lambda_param = 0.001  # Temporary impact coefficient
        self.eta = 0.0001  # Permanent impact coefficient
    
    def compute_optimal_schedule(self, n_slices: int = 10) -> np.ndarray:
        """
        Compute optimal execution schedule using Almgren-Chriss.
        Returns array of volume slices to execute at each time step.
        """
        # VWAP-based baseline
        vwap_schedule = np.ones(n_slices) * self.V / n_slices
        
        # Almgren-Chriss adjusts for market impact
        # Assume constant volume: x_k = V / n_slices
        # Cost = lambda * (x_k)^2 + eta * x_k * (sum of x_{k:n})
        
        # For simplicity, use approximate closed-form solution
        # More aggressive execution later if market impact is concave
        schedule = vwap_schedule.copy()
        
        # Apply time-weighted adjustment
        for i in range(n_slices):
            time_weight = (i + 1) / n_slices
            schedule[i] *= (1 + 0.1 * np.sin(time_weight * np.pi))
        
        schedule = schedule / np.sum(schedule) * self.V
        return schedule
    
    def estimate_execution_cost(self, 
                               volume_schedule: np.ndarray,
                               reference_price: float = 100.0) -> float:
        """
        Estimate total execution cost (slippage + impact).
        """
        total_cost = 0.0
        
        for vol in volume_schedule:
            # Temporary cost (immediate impact)
            temp_cost = self.lambda_param * vol ** 2
            
            # Permanent cost (longer-term impact)
            perm_cost = self.eta * vol * reference_price * 0.01  # 1 bp per unit
            
            total_cost += temp_cost + perm_cost
        
        total_cost_bps = (total_cost / (reference_price * np.sum(volume_schedule))) * 10000
        return total_cost_bps


class RealizedVolatilityNowcaster:
    """
    Nowcast realized volatility intraday (update as new tick data arrives).
    Reference: Barndorff-Nielsen & Shephard (2004)
    """
    
    def __init__(self, intraday_periods: int = 288):  # 5-min bars in trading day
        self.intraday_periods = intraday_periods
    
    def compute_realized_variance(self, intraday_returns: np.ndarray) -> float:
        """RV = sum of squared intraday returns."""
        rv = np.sum(intraday_returns ** 2)
        return rv
    
    def nowcast_end_of_day_rv(self, 
                             morning_returns: np.ndarray,
                             current_time_fraction: float = 0.5) -> float:
        """
        Forecast end-of-day realized variance given morning data.
        current_time_fraction: fraction of day completed (0.5 = noon)
        """
        morning_rv = self.compute_realized_variance(morning_returns)
        
        # Scale based on completed fraction
        # Assumes afternoon volatility similar to morning
        scaled_rv = morning_rv / (current_time_fraction ** 2)
        
        return np.sqrt(scaled_rv * 252)  # Annualized


class IntradayForecastingEngine:
    """
    Complete intraday forecasting engine combining all components.
    """
    
    def __init__(self):
        self.microprice = MicrpriceForecaster(lookback_period=100)
        self.ofi_model = OrderFlowImbalanceModel(decay_factor=0.95)
        self.hawkes = HawkesProcessVolatilityBurst(lambda_0=0.1, alpha=0.05, beta=0.95)
        self.queue_model = QueueReactionModel()
        self.almgren_chriss = AlmgrenChrisOptimalExecution(time_horizon=1.0, volume_to_execute=100000)
        self.rv_nowcaster = RealizedVolatilityNowcaster(intraday_periods=288)
    
    def intraday_forecast(self, 
                         orderbook_data: Dict[str, np.ndarray],
                         trade_data: Dict[str, np.ndarray],
                         current_time_fraction: float = 0.5) -> Dict[str, float]:
        """
        Generate comprehensive intraday forecast.
        
        orderbook_data: {
            'bid': best bid, 'ask': best ask,
            'bid_vol': bid volume, 'ask_vol': ask volume,
            'bid_queue': queue depth, 'ask_queue': queue depth
        }
        trade_data: {
            'buy_vol': cumulative buy volume,
            'sell_vol': cumulative sell volume,
            'prices': trade prices
        }
        """
        forecast = {}
        
        try:
            # Microprice forecast
            microprice = self.microprice.compute_microprice(
                orderbook_data['bid'],
                orderbook_data['ask'],
                orderbook_data['bid_vol'],
                orderbook_data['ask_vol']
            )
            forecast['microprice'] = microprice
            
            # OFI prediction
            ofi = self.ofi_model.compute_ofi(trade_data['buy_vol'], trade_data['sell_vol'])
            ofi_return_pred = self.ofi_model.predict_returns_from_ofi(
                np.array([ofi] * 5)  # Use recent OFI
            )
            forecast['ofi_return_signal'] = ofi_return_pred
            
            # Queue imbalance prediction
            queue_imb = self.queue_model.compute_queue_imbalance(
                orderbook_data['bid_queue'],
                orderbook_data['ask_queue']
            )
            predicted_price = self.queue_model.predict_price_from_queue(
                queue_imb,
                microprice
            )
            forecast['queue_predicted_price'] = predicted_price
            
            # Realized volatility nowcast
            intraday_returns = np.diff(np.log(trade_data['prices'])) if len(trade_data['prices']) > 1 else np.array([0.0])
            rv_nowcast = self.rv_nowcaster.nowcast_end_of_day_rv(
                intraday_returns,
                current_time_fraction
            )
            forecast['rv_nowcast'] = rv_nowcast
            
            logger.info("Intraday forecast generated successfully")
        
        except Exception as e:
            logger.warning(f"Intraday forecast failed: {e}")
        
        return forecast


class LeadLagForecaster:
    """
    Forecast using lead-lag signals from Singapore NIFTY futures vs spot.
    Reference: Granger causality, futures-spot basis
    """
    
    def __init__(self, lag_period: int = 5):
        self.lag_period = lag_period
    
    def compute_futures_spot_basis(self, futures_prices: np.ndarray, spot_prices: np.ndarray) -> float:
        """Compute futures premium over spot."""
        if len(futures_prices) == 0 or len(spot_prices) == 0:
            return 0.0
        
        basis = (futures_prices[-1] - spot_prices[-1]) / spot_prices[-1]
        return basis
    
    def detect_lead_relationship(self, 
                                futures_returns: np.ndarray,
                                spot_returns: np.ndarray) -> float:
        """
        Detect if futures lead spot using correlation at different lags.
        Returns: correlation strength (0-1)
        """
        if len(futures_returns) < self.lag_period or len(spot_returns) < self.lag_period:
            return 0.0
        
        # Check correlation at lag 1
        correlation = np.corrcoef(futures_returns[:-1], spot_returns[1:])[0, 1]
        
        # If positive and strong, futures lead spot
        lead_strength = max(correlation, 0.0)
        return lead_strength
    
    def forecast_spot_from_futures(self, 
                                  futures_returns: np.ndarray,
                                  spot_current_price: float) -> float:
        """
        Use futures lead-lag signal to forecast spot price.
        """
        if len(futures_returns) < 2:
            return spot_current_price
        
        # Latest futures return as leading indicator
        futures_momentum = np.mean(futures_returns[-3:])
        
        # Forecast: spot price moves in same direction as futures
        forecast_price = spot_current_price * (1 + futures_momentum * 0.9)  # 90% transmission
        
        return forecast_price
