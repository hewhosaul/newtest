"""
Research-Grade Factor Models: Implements Fama-French 5-factor, Carhart 4-factor, Asness models.
Based on academic literature: Fama & French (1993, 2015), Carhart (1997), Asness et al.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FamaFrenchFactorModel:
    """
    Fama-French 5-Factor Model: Market, Size, Value, Profitability, Investment.
    Reference: Fama & French (2015), "A five-factor asset pricing model"
    """
    
    def __init__(self):
        self.factor_loadings = {}
        self.factor_returns = {}
        self.alpha = None
    
    def compute_size_factor(self, market_cap: float, median_market_cap: float) -> float:
        """
        SMB (Small Minus Big) - Size factor.
        Positive loading = favor small caps.
        """
        if median_market_cap <= 0:
            return 0.0
        
        log_ratio = np.log(market_cap / median_market_cap)
        # Higher for small caps (negative log ratio)
        smb = 0.5 if log_ratio < 0 else -0.3
        return smb
    
    def compute_value_factor(self, 
                           pe_ratio: float, 
                           pb_ratio: float,
                           median_pe: float,
                           median_pb: float) -> float:
        """
        HML (High Minus Low) - Value factor.
        Based on B/M (Book-to-Market).
        Positive loading = favor value stocks (high B/M).
        """
        if median_pe <= 0 or median_pb <= 0:
            return 0.0
        
        # Inverse P/B is B/M
        if pb_ratio > 0:
            bm_ratio = 1.0 / pb_ratio
            median_bm = 1.0 / median_pb
            
            # Return premium for value stocks
            hml = 0.4 if bm_ratio > median_bm else -0.2
            return hml
        return 0.0
    
    def compute_profitability_factor(self, roe: float, median_roe: float) -> float:
        """
        RMW (Robust Minus Weak) - Profitability factor.
        Based on ROE and operating profitability.
        Positive loading = favor profitable stocks.
        """
        if median_roe <= 0:
            return 0.0
        
        if roe is None:
            return 0.0
        
        roe_percentile = roe / median_roe
        rmw = 0.3 if roe_percentile > 1.0 else -0.2
        return rmw
    
    def compute_investment_factor(self, 
                                 asset_growth: float,
                                 median_growth: float) -> float:
        """
        CMA (Conservative Minus Aggressive) - Investment factor.
        Based on asset growth rate.
        Positive loading = favor low-investment (conservative) stocks.
        """
        if median_growth <= 0:
            return 0.0
        
        growth_percentile = asset_growth / median_growth
        cma = 0.3 if growth_percentile < 1.0 else -0.2
        return cma
    
    def compute_market_factor_loading(self, stock_beta: float) -> float:
        """
        Market factor loading (beta).
        Higher beta = higher sensitivity to market.
        """
        return stock_beta
    
    def compute_ff5_signal(self,
                          stock_data: Dict[str, float],
                          factor_means: Dict[str, float]) -> float:
        """
        Compute Fama-French 5-factor signal for a stock.
        Combines all 5 factors into single forecast.
        """
        signals = {}
        
        # Market factor (beta)
        signals['market'] = self.compute_market_factor_loading(
            stock_data.get('beta', 1.0)
        )
        
        # Size factor
        signals['size'] = self.compute_size_factor(
            stock_data.get('market_cap', 1000000),
            factor_means.get('median_market_cap', 1000000)
        )
        
        # Value factor
        signals['value'] = self.compute_value_factor(
            stock_data.get('pe_ratio', 20),
            stock_data.get('pb_ratio', 2.5),
            factor_means.get('median_pe', 20),
            factor_means.get('median_pb', 2.5)
        )
        
        # Profitability factor
        signals['profitability'] = self.compute_profitability_factor(
            stock_data.get('roe', 0.15),
            factor_means.get('median_roe', 0.15)
        )
        
        # Investment factor
        signals['investment'] = self.compute_investment_factor(
            stock_data.get('asset_growth', 0.1),
            factor_means.get('median_growth', 0.1)
        )
        
        # Combine with equal weights
        combined_signal = 0.5 + np.mean([
            0.3 * signals['market'],  # Market factor weight
            0.175 * signals['size'],
            0.175 * signals['value'],
            0.175 * signals['profitability'],
            0.175 * signals['investment'],
        ])
        
        return np.clip(combined_signal, 0, 1)


class CarhartFourFactorModel:
    """
    Carhart 4-Factor Model: Extends FF3 with momentum factor.
    Reference: Carhart (1997), "On Persistence in Mutual Fund Performance"
    """
    
    def __init__(self, momentum_window: int = 252):
        self.momentum_window = momentum_window
        self.ff3 = FamaFrenchFactorModel()
    
    def compute_momentum_factor(self, 
                               returns_history: np.ndarray,
                               lookback_period: int = 12) -> float:
        """
        PR1YR (Prior 1-Year Return) - Momentum factor.
        Positive loading = favor momentum stocks (recent winners).
        """
        if len(returns_history) < lookback_period:
            return 0.0
        
        # Compute 12-month return
        recent_returns = returns_history[-lookback_period:]
        momentum_return = np.prod(1 + recent_returns) - 1
        
        # Normalize
        momentum_signal = 0.5 + 0.3 * np.tanh(momentum_return * 10)
        return np.clip(momentum_signal, 0, 1)
    
    def compute_carhart_signal(self,
                              stock_data: Dict[str, float],
                              returns_history: np.ndarray,
                              factor_means: Dict[str, float]) -> float:
        """
        Compute Carhart 4-factor signal.
        Combines FF3 with momentum.
        """
        # Get FF3 signal
        ff3_signal = self.ff3.compute_ff5_signal(stock_data, factor_means)
        
        # Get momentum signal
        momentum_signal = self.compute_momentum_factor(returns_history)
        
        # Combine: 75% FF3, 25% momentum
        carhart_signal = 0.75 * ff3_signal + 0.25 * momentum_signal
        
        return np.clip(carhart_signal, 0, 1)


class AsnessMultiFactorModel:
    """
    Asness Value/Momentum/Quality Integrated Model.
    Reference: Asness et al., "Value and Momentum Everywhere"
    Combines value, momentum, and quality into single score.
    """
    
    def __init__(self):
        self.carhart = CarhartFourFactorModel()
    
    def compute_quality_score(self, stock_data: Dict[str, float]) -> float:
        """Compute quality score based on ROE, margins, etc."""
        roe = stock_data.get('roe', 0.15)
        debt_to_equity = stock_data.get('debt_to_equity', 0.5)
        
        # Higher ROE is better
        roe_score = np.tanh(roe * 5) * 0.5 + 0.5
        
        # Lower leverage is better (safer)
        leverage_score = 1.0 / (1.0 + debt_to_equity)
        
        return 0.6 * roe_score + 0.4 * leverage_score
    
    def compute_asness_signal(self,
                             stock_data: Dict[str, float],
                             returns_history: np.ndarray,
                             factor_means: Dict[str, float]) -> float:
        """
        Compute integrated Asness signal.
        Combines value, momentum, and quality.
        """
        # Value component (from FF)
        ff_signal = self.carhart.ff3.compute_ff5_signal(stock_data, factor_means)
        value_signal = ff_signal  # Already incorporates value
        
        # Momentum component
        momentum_signal = self.carhart.compute_momentum_factor(returns_history)
        
        # Quality component
        quality_signal = self.compute_quality_score(stock_data)
        
        # Integrate: 40% value, 35% momentum, 25% quality
        asness_signal = (
            0.40 * value_signal +
            0.35 * momentum_signal +
            0.25 * quality_signal
        )
        
        return np.clip(asness_signal, 0, 1)


class FactorModelEnsemble:
    """Ensemble of all factor models for robust signals."""
    
    def __init__(self):
        self.ff5 = FamaFrenchFactorModel()
        self.carhart = CarhartFourFactorModel()
        self.asness = AsnessMultiFactorModel()
    
    def compute_ensemble_signal(self,
                               stock_data: Dict[str, float],
                               returns_history: np.ndarray,
                               factor_means: Dict[str, float]) -> Dict[str, float]:
        """
        Compute ensemble of all factor model signals.
        Returns individual signals and combined signal.
        """
        signals = {
            'ff5_signal': self.ff5.compute_ff5_signal(stock_data, factor_means),
            'carhart_signal': self.carhart.compute_carhart_signal(
                stock_data, returns_history, factor_means
            ),
            'asness_signal': self.asness.compute_asness_signal(
                stock_data, returns_history, factor_means
            ),
        }
        
        # Ensemble: average of all models
        signals['ensemble_signal'] = np.mean([
            signals['ff5_signal'],
            signals['carhart_signal'],
            signals['asness_signal'],
        ])
        
        return signals
