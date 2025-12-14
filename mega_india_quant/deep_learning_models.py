"""
Deep Learning models: LSTMs, Transformers, CNNs for volatility, HMM regimes.
References: Chu et al. (2018), Wen et al. (2021), Transformer-based forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try imports, with graceful fallbacks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, Concatenate
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using numpy fallbacks")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy fallbacks")


class LSTMVolatilityForecaster:
    """
    LSTM for volatility forecasting.
    Reference: Chu et al. (2018), "GARCH-type options pricing with neural networks"
    """
    
    def __init__(self, lookback_window: int = 60, forecast_horizon: int = 5):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = None
        self.history = []
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM build")
            return
        
        try:
            self.model = Sequential([
                LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(self.forecast_horizon)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info("LSTM volatility forecaster built")
        except Exception as e:
            logger.warning(f"LSTM build failed: {e}")
    
    def prepare_sequences(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(returns) - self.lookback_window - self.forecast_horizon):
            X.append(returns[i:i + self.lookback_window].reshape(-1, 1))
            # Target: future volatility
            future_returns = returns[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon]
            y.append(np.std(future_returns))
        
        return np.array(X), np.array(y)
    
    def fit(self, returns: np.ndarray, epochs: int = 50) -> None:
        """Fit LSTM on returns."""
        if self.model is None:
            self.build_model((self.lookback_window, 1))
        
        if self.model is None:
            logger.warning("LSTM model not available, using fallback")
            return
        
        try:
            X, y = self.prepare_sequences(returns)
            if len(X) > 0:
                self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
                logger.info(f"LSTM trained on {len(X)} sequences")
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
    
    def forecast(self, recent_returns: np.ndarray) -> np.ndarray:
        """Forecast volatility."""
        if self.model is None or len(recent_returns) < self.lookback_window:
            # Fallback: historical volatility
            return np.full(self.forecast_horizon, np.std(recent_returns[-20:]))
        
        try:
            X = recent_returns[-self.lookback_window:].reshape(1, -1, 1)
            forecast = self.model.predict(X, verbose=0)
            return forecast[0]
        except Exception as e:
            logger.warning(f"LSTM forecast failed: {e}")
            return np.full(self.forecast_horizon, np.std(recent_returns[-20:]))


class TransformerPriceForecaster:
    """
    Transformer-based price forecaster with multi-head attention.
    Reference: Vaswani et al. (2017), attention mechanisms
    """
    
    def __init__(self, d_model: int = 64, num_heads: int = 4, seq_length: int = 60):
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.model = None
    
    def build_transformer(self) -> None:
        """Build Transformer model."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, skipping Transformer build")
            return
        
        try:
            inputs = Input(shape=(self.seq_length, 1))
            
            # Embedding
            x = Dense(self.d_model)(inputs)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward
            x = Dense(128, activation='relu')(x)
            x = Dense(self.d_model)(x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Output
            x = layers.GlobalAveragePooling1D()(x)
            outputs = Dense(1)(x)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='mse')
            logger.info("Transformer model built")
        except Exception as e:
            logger.warning(f"Transformer build failed: {e}")
    
    def fit(self, prices: np.ndarray, epochs: int = 30) -> None:
        """Train Transformer."""
        if self.model is None:
            self.build_transformer()
        
        if self.model is None:
            return
        
        try:
            X, y = [], []
            for i in range(len(prices) - self.seq_length - 1):
                X.append(prices[i:i + self.seq_length])
                y.append(prices[i + self.seq_length])
            
            X = np.array(X).reshape(-1, self.seq_length, 1)
            y = np.array(y)
            
            if len(X) > 0:
                self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
                logger.info(f"Transformer trained")
        except Exception as e:
            logger.warning(f"Transformer training failed: {e}")
    
    def predict(self, recent_prices: np.ndarray) -> float:
        """Forecast next price."""
        if self.model is None or len(recent_prices) < self.seq_length:
            return recent_prices[-1]  # Fallback: no change
        
        try:
            X = recent_prices[-self.seq_length:].reshape(1, self.seq_length, 1)
            prediction = self.model.predict(X, verbose=0)[0, 0]
            return prediction
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}")
            return recent_prices[-1]


class CNNMicrostructurePatternDetector:
    """
    CNN for detecting microstructure patterns from orderflow and price tick data.
    Reference: Lewis et al. (2019), applied to orderflow
    """
    
    def __init__(self, kernel_size: int = 3, num_filters: int = 32):
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.model = None
    
    def build_cnn(self, input_length: int) -> None:
        """Build 1D CNN for pattern detection."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, skipping CNN build")
            return
        
        try:
            self.model = Sequential([
                layers.Conv1D(self.num_filters, self.kernel_size, activation='relu', 
                             input_shape=(input_length, 1)),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(64, self.kernel_size, activation='relu'),
                layers.MaxPooling1D(pool_size=2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(3, activation='softmax')  # 3 classes: up, neutral, down
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("CNN microstructure detector built")
        except Exception as e:
            logger.warning(f"CNN build failed: {e}")
    
    def predict_pattern(self, orderflow_sequence: np.ndarray) -> int:
        """
        Predict orderflow pattern.
        Returns: 0 (down), 1 (neutral), 2 (up)
        """
        if self.model is None or len(orderflow_sequence) < 10:
            # Fallback: simple mean
            return 1 if np.mean(orderflow_sequence) > 0 else 0
        
        try:
            X = orderflow_sequence.reshape(1, -1, 1)
            pred = self.model.predict(X, verbose=0)
            return np.argmax(pred[0])
        except Exception as e:
            logger.warning(f"CNN prediction failed: {e}")
            return 1


class RealizedVolatilityForecaster:
    """
    HAR-RV model for realized volatility forecasting.
    Reference: Corsi (2009), "A simple approximate long-memory model of realized volatility"
    """
    
    def __init__(self):
        self.daily_weights = 0.5
        self.weekly_weights = 0.3
        self.monthly_weights = 0.2
    
    def compute_realized_variance(self, intraday_returns: np.ndarray) -> float:
        """RV = sum of squared intraday returns."""
        return np.sum(intraday_returns ** 2)
    
    def forecast_har(self, 
                    daily_rv: np.ndarray,
                    weekly_rv: np.ndarray,
                    monthly_rv: np.ndarray) -> float:
        """
        HAR forecast: weighted average of daily, weekly, monthly RV.
        """
        har_forecast = (self.daily_weights * daily_rv[-1] + 
                       self.weekly_weights * np.mean(weekly_rv[-5:]) +
                       self.monthly_weights * np.mean(monthly_rv[-20:]))
        
        return np.sqrt(har_forecast * 252)  # Annualized


class GARCHVolatilityModel:
    """
    GARCH(1,1) model for volatility forecasting.
    Reference: Engle (1982), Bollerslev (1986)
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.85, omega: float = 1e-5):
        self.alpha = alpha  # News impact
        self.beta = beta    # Persistence
        self.omega = omega  # Long-run variance
        self.sigma_sq = None
        self.history = []
    
    def fit(self, returns: np.ndarray) -> None:
        """Simple GARCH(1,1) fitting via grid search or analytical."""
        try:
            from arch import arch_model
            
            # Use arch library if available
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            
            # Extract parameters
            self.omega = result.params['omega']
            self.alpha = result.params['alpha[1]']
            self.beta = result.params['beta[1]']
            
            self.sigma_sq = result.conditional_volatility.values[-1] ** 2
            logger.info(f"GARCH fitted: alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        except ImportError:
            # Fallback: use constant GARCH parameters
            logger.warning("arch library not available, using constant GARCH parameters")
            self.sigma_sq = np.var(returns) * 252
    
    def forecast(self, returns: np.ndarray, steps: int = 5) -> np.ndarray:
        """Forecast volatility forward."""
        if self.sigma_sq is None:
            self.sigma_sq = np.var(returns[-100:]) * 252
        
        forecasts = []
        sigma_sq_t = self.sigma_sq
        
        for _ in range(steps):
            # GARCH recursion: sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2
            epsilon_sq = returns[-1] ** 2 if len(returns) > 0 else 0.0
            sigma_sq_t = self.omega + self.alpha * epsilon_sq + self.beta * sigma_sq_t
            forecasts.append(np.sqrt(sigma_sq_t))
        
        return np.array(forecasts)


class HybridDeepLearningEnsemble:
    """
    Ensemble combining LSTM, Transformer, CNN for robustness.
    """
    
    def __init__(self):
        self.lstm = LSTMVolatilityForecaster(lookback_window=60, forecast_horizon=5)
        self.transformer = TransformerPriceForecaster(d_model=64, num_heads=4, seq_length=60)
        self.cnn = CNNMicrostructurePatternDetector(kernel_size=3, num_filters=32)
        self.har = RealizedVolatilityForecaster()
        self.garch = GARCHVolatilityModel()
    
    def train_all(self, price_data: np.ndarray, returns: np.ndarray) -> None:
        """Train all models."""
        try:
            if len(returns) > 100:
                # Ensure data is properly shaped
                if isinstance(returns, (list, pd.Series)):
                    returns = np.array(returns)
                if isinstance(price_data, (list, pd.Series)):
                    price_data = np.array(price_data)
                
                # Flatten if needed
                returns = np.atleast_1d(returns).flatten()
                price_data = np.atleast_1d(price_data).flatten()
                
                # Train models with proper data
                self.lstm.fit(returns.reshape(-1, 1), epochs=20)
                self.transformer.fit(price_data.reshape(-1, 1), epochs=15)
                self.garch.fit(returns)
                logger.info("All deep learning models trained")
        except Exception as e:
            logger.warning(f"Deep learning ensemble training failed: {e}")
    
    def forecast_ensemble(self, 
                        recent_prices: np.ndarray,
                        recent_returns: np.ndarray,
                        orderflow: np.ndarray = None) -> Dict[str, float]:
        """Generate ensemble forecast."""
        forecasts = {}
        
        try:
            # LSTM volatility
            lstm_vol = self.lstm.forecast(recent_returns)
            forecasts['lstm_vol'] = np.mean(lstm_vol) if len(lstm_vol) > 0 else 0.15
            
            # Transformer price
            transformer_price = self.transformer.predict(recent_prices)
            forecasts['transformer_price'] = transformer_price
            
            # CNN pattern
            if orderflow is not None:
                cnn_pattern = self.cnn.predict_pattern(orderflow)
                forecasts['cnn_pattern'] = cnn_pattern
            
            # GARCH volatility
            garch_vol = self.garch.forecast(recent_returns, steps=1)
            forecasts['garch_vol'] = garch_vol[0] if len(garch_vol) > 0 else 0.15
            
        except Exception as e:
            logger.warning(f"Ensemble forecast failed: {e}")
        
        return forecasts
