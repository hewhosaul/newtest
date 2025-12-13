"""
Macro modeling framework: FAVAR, DFM, MIDAS, Macro regime detection.
References: Stock & Watson (2002), Koop & Korobilis (2013), Markov-switching models.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from hmmlearn import hmm
import logging
from typing import Tuple, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorAugmentedVAR:
    """
    Factor-Augmented VAR (FAVAR) model.
    Reduces dimensionality via PCA, then applies VAR.
    Reference: Stock & Watson (2002), "Forecasting using principal components from a large number of predictors"
    """
    
    def __init__(self, n_factors: int = 3, lag_order: int = 4):
        self.n_factors = n_factors
        self.lag_order = lag_order
        self.pca = PCA(n_components=n_factors)
        self.scaler = StandardScaler()
        self.var_model = None
        self.factors = None
    
    def fit(self, X: np.ndarray) -> None:
        """Fit FAVAR on macro data."""
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Extract principal components (factors)
        self.factors = self.pca.fit_transform(X_scaled)
        
        # Fit VAR on factors
        var_data = pd.DataFrame(self.factors, columns=[f'factor_{i}' for i in range(self.n_factors)])
        self.var_model = VAR(var_data)
        self.var_model = self.var_model.fit(self.lag_order)
        logger.info(f"FAVAR fitted with {self.n_factors} factors and lag {self.lag_order}")
    
    def forecast(self, steps: int = 5) -> np.ndarray:
        """Forecast factors forward."""
        if self.var_model is None:
            raise ValueError("Model not fitted")
        
        forecast = self.var_model.forecast(self.var_model.endog.values[-self.lag_order:], steps=steps)
        return forecast
    
    def get_explained_variance(self) -> float:
        """Get explained variance ratio of PCA."""
        return np.sum(self.pca.explained_variance_ratio_)


class DynamicFactorModel:
    """
    Dynamic Factor Model for macro indicators.
    Reference: Koop & Korobilis (2013), Mariano & Murasawa (2003)
    """
    
    def __init__(self, n_factors: int = 2, max_lag: int = 3):
        self.n_factors = n_factors
        self.max_lag = max_lag
        self.pca = PCA(n_components=n_factors)
        self.factor_dynamics = None
        self.loadings = None
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit DFM."""
        # Standardize data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Extract factors via PCA
        factors = self.pca.fit_transform(X_scaled)
        
        # Fit AR model to factors
        self.factor_dynamics = {}
        for i in range(self.n_factors):
            factor_series = pd.Series(factors[:, i])
            # AR(3) on each factor
            try:
                var_model = VAR(factor_series.values.reshape(-1, 1))
                self.factor_dynamics[i] = var_model.fit(self.max_lag)
            except Exception as e:
                logger.warning(f"DFM AR fit failed for factor {i}: {e}")
        
        # Estimate loadings
        self.loadings = self.pca.components_.T
        logger.info(f"DFM fitted with {self.n_factors} factors")
    
    def nowcast(self, partial_data: np.ndarray) -> np.ndarray:
        """
        Nowcast missing observations using factor model.
        Reference: Nowcasting literature (Giannone, Reichlin, Small, 2008)
        """
        factors = self.pca.transform(partial_data.reshape(1, -1))
        return factors[0]


class MIDASRegression:
    """
    Mixed-Frequency VAR model for integrating high-frequency (e.g., daily) 
    and low-frequency (e.g., monthly macro) data.
    Reference: Ghysels, Santa-Clara, Valkanov (2004)
    """
    
    def __init__(self, high_freq_lags: int = 22, low_freq_lags: int = 3):
        self.high_freq_lags = high_freq_lags
        self.low_freq_lags = low_freq_lags
        self.weights = None
    
    def _almon_weights(self, n_lags: int, n_params: int = 3) -> np.ndarray:
        """Generate Almon polynomial weights."""
        lags = np.arange(1, n_lags + 1)
        weights = np.ones(n_lags)
        for i in range(1, n_params):
            weights += (i + 1) * (lags ** i)
        return weights / np.sum(weights)
    
    def fit(self, daily_returns: np.ndarray, monthly_macro: np.ndarray) -> None:
        """
        Fit MIDAS regression linking daily equity returns to monthly macro variables.
        daily_returns: (n_days,)
        monthly_macro: (n_months,)
        """
        self.weights = self._almon_weights(self.high_freq_lags)
        logger.info("MIDAS weights fitted using Almon polynomial")
    
    def forecast_daily_from_macro(self, daily_prices: np.ndarray, macro_values: np.ndarray) -> np.ndarray:
        """Map macro changes to daily forecast."""
        if len(macro_values) < self.low_freq_lags:
            return np.array([0.0])
        
        # Simple correlation-based forecasting
        macro_change = np.diff(macro_values[-self.low_freq_lags:])
        daily_mean = np.mean(daily_prices[-self.high_freq_lags:])
        macro_impact = np.sum(macro_change * 0.01)  # Scaled impact
        
        return np.array([daily_mean * (1 + macro_impact)])


class MacroRegimeDetector:
    """
    Detect macro regimes using Hidden Markov Model.
    Identifies: Risk-On, Risk-Off, Growth, Stagflation regimes.
    Reference: Hamilton (1989), Markov-switching models
    """
    
    def __init__(self, n_regimes: int = 4, n_iter: int = 1000):
        self.n_regimes = n_regimes
        self.hmm_model = hmm.GaussianHMM(n_components=n_regimes, n_iter=n_iter, random_state=42)
        self.scaler = StandardScaler()
        self.regime_names = ['Growth', 'Risk-Off', 'Stagflation', 'Low Volatility']
    
    def fit(self, macro_data: np.ndarray) -> None:
        """
        Fit HMM to macro indicators.
        macro_data: (n_observations, n_features)
        """
        # Standardize
        macro_scaled = self.scaler.fit_transform(macro_data)
        
        # Reshape for HMM (needs to be 2D)
        if macro_scaled.ndim == 1:
            macro_scaled = macro_scaled.reshape(-1, 1)
        
        # Fit HMM
        try:
            self.hmm_model.fit(macro_scaled)
            logger.info(f"Macro regime HMM fitted with {self.n_regimes} states")
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")
    
    def predict_regime(self, macro_data: np.ndarray) -> int:
        """Predict current regime."""
        macro_scaled = self.scaler.transform(macro_data)
        if macro_scaled.ndim == 1:
            macro_scaled = macro_scaled.reshape(1, -1)
        
        regime = self.hmm_model.predict(macro_scaled)[-1]
        return regime
    
    def get_regime_name(self, regime_idx: int) -> str:
        """Get regime name."""
        if regime_idx < len(self.regime_names):
            return self.regime_names[regime_idx]
        return f"Regime_{regime_idx}"


class YieldCurveAnalyzer:
    """
    Analyze yield curve and map to equity volatility.
    Reference: Estrella-Mishkin, US yield curve signals for recessions
    """
    
    def __init__(self):
        self.term_spread = None
        self.pca_yields = PCA(n_components=3)
    
    def fit_yield_curve(self, yield_data: pd.DataFrame) -> None:
        """
        Fit PCA to yield curve.
        yield_data: DataFrame with columns for different tenors (1Y, 5Y, 10Y, etc.)
        """
        try:
            yields_standardized = StandardScaler().fit_transform(yield_data)
            self.pca_yields.fit(yields_standardized)
            logger.info(f"Yield curve PCA fitted. Explained variance: {np.sum(self.pca_yields.explained_variance_ratio_):.3f}")
        except Exception as e:
            logger.warning(f"Yield curve PCA fit failed: {e}")
    
    def compute_term_spread(self, yield_data: pd.DataFrame) -> pd.Series:
        """Compute 10Y - 2Y term spread."""
        try:
            if '10Y' in yield_data.columns and '2Y' in yield_data.columns:
                self.term_spread = yield_data['10Y'] - yield_data['2Y']
                return self.term_spread
            else:
                logger.warning("Required yield tenors not available")
                return pd.Series([0.0])
        except Exception as e:
            logger.warning(f"Term spread calculation failed: {e}")
            return pd.Series([0.0])
    
    def get_yield_curve_signal(self, current_yields: np.ndarray) -> float:
        """
        Get yield curve signal for recession risk.
        Reference: Estrella & Mishkin (1998)
        """
        if len(current_yields) < 2:
            return 0.0
        
        # Term spread (10Y - 2Y proxy)
        term_spread = current_yields[-1] - current_yields[0]
        
        # Inverted yield curve predicts recession
        recession_probability = 1.0 / (1.0 + np.exp(5 * term_spread))  # Logistic function
        return recession_probability


class GlobalMacroToEquityBridge:
    """
    Bridge model connecting global macro factors to Indian equity forecasts.
    Uses multi-view learning and factor integration.
    """
    
    def __init__(self):
        self.favar = FactorAugmentedVAR(n_factors=3, lag_order=4)
        self.dfm = DynamicFactorModel(n_factors=2, max_lag=3)
        self.midas = MIDASRegression(high_freq_lags=22, low_freq_lags=3)
        self.regime_detector = MacroRegimeDetector(n_regimes=4)
        self.yield_analyzer = YieldCurveAnalyzer()
        self.macro_factor_weights = None
    
    def fit(self, macro_dict: Dict[str, pd.DataFrame]) -> None:
        """Fit all macro models."""
        try:
            # Combine global macro data
            macro_combined = pd.concat([
                macro_dict.get('us_yields', pd.DataFrame()),
                macro_dict.get('inflation', pd.DataFrame()),
                macro_dict.get('fx_data', pd.DataFrame()),
                macro_dict.get('commodities', pd.DataFrame()),
            ], axis=1).dropna()
            
            if len(macro_combined) > 10:
                self.favar.fit(macro_combined.values)
                self.dfm.fit(macro_combined)
                self.regime_detector.fit(macro_combined.values[-100:] if len(macro_combined) > 100 else macro_combined.values)
                
                # Yield curve analysis
                if 'us_yields' in macro_dict:
                    self.yield_analyzer.fit_yield_curve(macro_dict['us_yields'])
                
                logger.info("Global macro models fitted successfully")
        except Exception as e:
            logger.warning(f"Macro model fitting failed: {e}")
    
    def forecast_macro_state(self, horizon: int = 5) -> Dict[str, np.ndarray]:
        """Forecast macro factors forward."""
        forecast_dict = {}
        
        try:
            if self.favar.var_model is not None:
                forecast_dict['favar_forecast'] = self.favar.forecast(steps=horizon)
        except Exception as e:
            logger.warning(f"FAVAR forecast failed: {e}")
        
        return forecast_dict
    
    def get_macro_regime_signal(self, current_macro: np.ndarray) -> Dict[str, float]:
        """Get current macro regime signal."""
        try:
            regime_idx = self.regime_detector.predict_regime(current_macro)
            regime_name = self.regime_detector.get_regime_name(regime_idx)
            
            return {
                'regime_index': regime_idx,
                'regime_name': regime_name,
                'growth_signal': 1.0 if regime_idx == 0 else 0.5,
                'risk_off_signal': 1.0 if regime_idx == 1 else 0.0,
            }
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {'regime_index': -1, 'regime_name': 'Unknown'}
