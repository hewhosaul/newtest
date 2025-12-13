"""
Research-paper-driven factor models: Carhart 4-factor, Fama-French 5-factor, Momentum, Value, Quality.
References: Jegadeesh & Titman (1993), Carhart (1997), Fama-French (2015), Asness et al.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumFactor:
    """
    Momentum factor based on Jegadeesh & Titman (1993).
    Long past winners, short past losers.
    """
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period  # Trading days in 1 year
    
    def compute_momentum(self, prices: np.ndarray) -> float:
        """Compute 1-year momentum (excluding most recent 1 month)."""
        if len(prices) < self.lookback_period + 21:
            return 0.0
        
        # Exclude most recent month
        historical_prices = prices[-(self.lookback_period + 21):-21]
        return np.log(historical_prices[-1] / historical_prices[0])


class ValueFactor:
    """
    Value factor: High book-to-market, low P/E, low P/B ratios.
    Reference: Fama & French (1992), Asness et al. (2013)
    """
    
    def __init__(self):
        self.value_scores = {}
    
    def compute_value_score(self, price: float, book_value: float, earnings: float) -> float:
        """
        Composite value score combining P/B and P/E metrics.
        """
        pb_ratio = price / book_value if book_value > 0 else 1.0
        pe_ratio = price / earnings if earnings > 0 else 1.0
        
        # Lower ratios = higher value
        value_score = (1.0 / (1.0 + pb_ratio)) + (1.0 / (1.0 + pe_ratio))
        return value_score


class QualityFactor:
    """
    Quality factor: High profitability, low leverage, stable earnings.
    Reference: Asness, Frazzini, Pedersen (2018)
    """
    
    def __init__(self):
        pass
    
    def compute_quality_score(self, roe: float, debt_to_equity: float, earnings_growth: float) -> float:
        """
        Composite quality score.
        roe: Return on Equity (%)
        debt_to_equity: Leverage ratio
        earnings_growth: Earnings growth rate (%)
        """
        # Higher ROE is better
        roe_component = roe / 100.0 if roe > 0 else 0.0
        
        # Lower leverage is better (inverse)
        leverage_component = 1.0 / (1.0 + debt_to_equity)
        
        # Higher and stable growth is better
        growth_component = min(earnings_growth / 100.0, 0.5)
        
        quality_score = 0.4 * roe_component + 0.3 * leverage_component + 0.3 * growth_component
        return quality_score


class VolatilityFactor:
    """
    Low volatility anomaly (betting against beta).
    Reference: Frazzini & Pedersen (2014), "Betting Against Beta"
    """
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
    
    def compute_realized_volatility(self, returns: np.ndarray) -> float:
        """Compute realized volatility (annualized)."""
        if len(returns) < self.lookback_period:
            returns_subset = returns
        else:
            returns_subset = returns[-self.lookback_period:]
        
        realized_vol = np.std(returns_subset) * np.sqrt(252)
        return realized_vol
    
    def compute_low_vol_score(self, realized_vol: float, market_vol: float = 0.15) -> float:
        """
        Low volatility score (higher volatility = lower score).
        market_vol: Typical market volatility (fallback: 0.15)
        """
        vol_ratio = realized_vol / market_vol if market_vol > 0 else 1.0
        low_vol_score = 1.0 / (1.0 + vol_ratio)
        return low_vol_score


class CarrartFourFactorModel:
    """
    Carhart 4-Factor Model (Momentum + Fama-French 3-factor).
    Reference: Carhart (1997), "Mutual Fund Performance"
    
    Factors:
    1. Market excess return (MKT-RF)
    2. Size (SMB: Small Minus Big)
    3. Value (HML: High Minus Low)
    4. Momentum (WML: Winners Minus Losers)
    """
    
    def __init__(self):
        self.momentum = MomentumFactor()
        self.value = ValueFactor()
        self.quality = QualityFactor()
        self.volatility = VolatilityFactor()
        self.factor_returns = {}
    
    def compute_factor_exposures(self, 
                                stock_data: Dict[str, float],
                                market_excess_return: float) -> Dict[str, float]:
        """
        Compute exposures to the 4 Carhart factors.
        
        stock_data: {
            'price': current price,
            'book_value': book value per share,
            'earnings': earnings per share,
            'momentum_return': 1-year return,
            'market_cap': market cap,
            'returns': price returns array,
        }
        """
        exposures = {}
        
        try:
            # Market factor (beta)
            if 'returns' in stock_data:
                stock_returns = np.array(stock_data['returns'])
                if len(stock_returns) > 20:
                    market_returns = np.array([market_excess_return / 252] * len(stock_returns))
                    beta = np.cov(stock_returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 1.0
                    exposures['beta'] = beta
            else:
                exposures['beta'] = 1.0
            
            # Value factor
            if all(k in stock_data for k in ['price', 'book_value', 'earnings']):
                value_score = self.value.compute_value_score(
                    stock_data['price'],
                    stock_data['book_value'],
                    stock_data['earnings']
                )
                exposures['value'] = value_score
            
            # Quality factor
            if all(k in stock_data for k in ['roe', 'debt_to_equity', 'earnings_growth']):
                quality_score = self.quality.compute_quality_score(
                    stock_data['roe'],
                    stock_data['debt_to_equity'],
                    stock_data['earnings_growth']
                )
                exposures['quality'] = quality_score
            
            # Volatility factor
            if 'returns' in stock_data:
                realized_vol = self.volatility.compute_realized_volatility(np.array(stock_data['returns']))
                low_vol_score = self.volatility.compute_low_vol_score(realized_vol)
                exposures['low_vol'] = low_vol_score
            
            # Momentum factor
            if 'price_history' in stock_data:
                momentum = self.momentum.compute_momentum(np.array(stock_data['price_history']))
                exposures['momentum'] = momentum
            
        except Exception as e:
            logger.warning(f"Factor exposure computation failed: {e}")
        
        return exposures


class FamaFrenchFiveFactorModel:
    """
    Fama-French 5-Factor Model (adds profitability and investment factors).
    Reference: Fama & French (2015), "A five-factor asset pricing model"
    
    Factors:
    1. Market excess return (MKT-RF)
    2. Size (SMB)
    3. Value (HML)
    4. Profitability (RMW: Robust Minus Weak)
    5. Investment (CMA: Conservative Minus Aggressive)
    """
    
    def __init__(self):
        self.carhart = CarrartFourFactorModel()
    
    def compute_profitability_score(self, roe: float, roa: float) -> float:
        """Compute profitability score (RMW factor)."""
        prof_score = 0.6 * (roe / 100.0) + 0.4 * (roa / 100.0)
        return np.clip(prof_score, 0.0, 1.0)
    
    def compute_investment_score(self, capex_to_assets: float, inventory_change: float) -> float:
        """Compute investment score (CMA factor)."""
        # Lower capex intensity = more conservative
        conservative_score = 1.0 / (1.0 + capex_to_assets)
        return np.clip(conservative_score, 0.0, 1.0)
    
    def compute_ff5_exposures(self, 
                             carhart_exposures: Dict[str, float],
                             roe: float = 10.0,
                             roa: float = 5.0,
                             capex_to_assets: float = 0.05,
                             inventory_change: float = 0.02) -> Dict[str, float]:
        """Compute 5-factor exposures."""
        exposures = carhart_exposures.copy()
        
        try:
            prof_score = self.compute_profitability_score(roe, roa)
            exposures['profitability'] = prof_score
            
            inv_score = self.compute_investment_score(capex_to_assets, inventory_change)
            exposures['investment'] = inv_score
        except Exception as e:
            logger.warning(f"FF5 exposure computation failed: {e}")
        
        return exposures


class BettingAgainstBetaModel:
    """
    Betting Against Beta (BAB) factor construction.
    Reference: Frazzini & Pedersen (2014)
    
    Long low-beta stocks, short high-beta stocks, while maintaining market neutrality.
    """
    
    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        self.betas = {}
    
    def compute_beta(self, stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Compute beta from historical returns."""
        if len(stock_returns) < 20 or len(market_returns) < 20:
            return 1.0
        
        cov = np.cov(stock_returns, market_returns)[0, 1]
        market_var = np.var(market_returns)
        
        if market_var > 0:
            return cov / market_var
        return 1.0
    
    def construct_bab_signal(self, 
                            stock_beta: float,
                            market_beta: float = 1.0) -> float:
        """
        Construct BAB signal: go long low-beta, short high-beta.
        Signal ranges from -1 (high beta) to +1 (low beta).
        """
        beta_ratio = stock_beta / market_beta if market_beta > 0 else 1.0
        
        # Logistic transformation to [-1, 1]
        bab_signal = 2.0 / (1.0 + np.exp(beta_ratio)) - 1.0
        return bab_signal


class MultiFactorCombiner:
    """
    Combine multiple factor signals into composite score.
    Uses cross-sectional regression and factor weighting.
    """
    
    def __init__(self):
        self.ff5_model = FamaFrenchFiveFactorModel()
        self.bab_model = BettingAgainstBetaModel()
        self.factor_weights = {
            'beta': 0.1,
            'value': 0.2,
            'quality': 0.25,
            'low_vol': 0.15,
            'momentum': 0.15,
            'profitability': 0.1,
            'investment': 0.05,
            'bab': 0.0
        }
    
    def compute_composite_score(self, 
                               stock_data: Dict[str, float],
                               market_excess_return: float = 0.08,
                               use_ff5: bool = True) -> float:
        """
        Compute composite alpha/long score from multiple factors.
        """
        try:
            # Get Carhart 4-factor exposures
            c4_exposures = self.ff5_model.carhart.compute_factor_exposures(stock_data, market_excess_return)
            
            # Extend to FF5 if requested
            if use_ff5:
                exposures = self.ff5_model.compute_ff5_exposures(
                    c4_exposures,
                    roe=stock_data.get('roe', 10.0),
                    roa=stock_data.get('roa', 5.0),
                )
            else:
                exposures = c4_exposures
            
            # Add BAB signal
            if 'beta' in exposures:
                bab_signal = self.bab_model.construct_bab_signal(exposures['beta'])
                exposures['bab'] = bab_signal
            
            # Normalize exposures to [0, 1]
            for key in exposures:
                if key in ['momentum']:
                    # Momentum is unbounded, apply tanh
                    exposures[key] = (np.tanh(exposures[key]) + 1.0) / 2.0
                else:
                    # Clip to [0, 1]
                    exposures[key] = np.clip(exposures[key], 0.0, 1.0)
            
            # Weighted combination
            composite_score = 0.0
            total_weight = 0.0
            
            for factor, weight in self.factor_weights.items():
                if factor in exposures:
                    composite_score += weight * exposures[factor]
                    total_weight += weight
            
            if total_weight > 0:
                composite_score = composite_score / total_weight
            
            return composite_score
        
        except Exception as e:
            logger.warning(f"Composite score computation failed: {e}")
            return 0.5  # Neutral score


class MicrostructureFactors:
    """
    Microstructure-based factors: Kyle lambda, price impact, orderflow.
    Reference: Kyle (1985), Hasbrouck (1991)
    """
    
    def __init__(self):
        self.kyle_lambda_history = []
    
    def compute_kyle_lambda(self, 
                          bid_ask_spread: float,
                          volume: float,
                          price_change: float) -> float:
        """
        Estimate Kyle's lambda (price impact coefficient).
        lambda = |price_change| / volume
        """
        if volume > 0:
            lambda_est = abs(price_change) / volume
            self.kyle_lambda_history.append(lambda_est)
            return lambda_est
        return 0.0
    
    def compute_hasbrouck_lambda(self, returns: np.ndarray, volume: np.ndarray) -> float:
        """
        Hasbrouck's information share / realized spread.
        Reference: Hasbrouck (1991)
        """
        if len(returns) < 2 or len(volume) < 2:
            return 0.0
        
        # Simple linear regression of returns on volume
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(volume[-100:], returns[-100:])
            return np.abs(slope)
        except Exception as e:
            logger.warning(f"Hasbrouck lambda computation failed: {e}")
            return 0.0
    
    def compute_orderflow_imbalance(self, buy_volume: np.ndarray, sell_volume: np.ndarray) -> float:
        """
        Order flow imbalance indicator.
        OFI = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        """
        if len(buy_volume) == 0 or len(sell_volume) == 0:
            return 0.0
        
        total_flow = np.sum(buy_volume) + np.sum(sell_volume)
        if total_flow > 0:
            ofi = (np.sum(buy_volume) - np.sum(sell_volume)) / total_flow
            return ofi
        return 0.0
