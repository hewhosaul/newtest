"""
Main orchestration engine: coordinates all modules and produces final forecasts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json

from . import data_feeds
from . import macro_models
from . import factor_models
from . import deep_learning_models
from . import computer_vision
from . import intraday_models
from . import advanced_models
from . import fusion_engine
from . import self_learning_engine
from . import backtester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MegaIndiaQuantSystem:
    """
    Main quant system orchestrating all modules.
    """
    
    def __init__(self, mode: str = 'backtest'):
        """
        Initialize the mega quant system.
        mode: 'backtest' or 'live'
        """
        self.mode = mode
        self.data_collector = data_feeds.MacroDataCollector()
        
        # Initialize all models
        self.macro_bridge = macro_models.GlobalMacroToEquityBridge()
        self.factor_model = factor_models.MultiFactorCombiner()
        self.deep_ensemble = deep_learning_models.HybridDeepLearningEnsemble()
        self.cv_pipeline = computer_vision.CVAnalysisPipeline()
        self.intraday_engine = intraday_models.IntradayForecastingEngine()
        self.regime_detector = advanced_models.RegimeSwitchingVAR(n_regimes=4, lag_order=2)
        self.particle_filter = advanced_models.ParticleFilter(n_particles=1000)
        self.kalman_filter = advanced_models.EnsembleKalmanFilter(n_ensemble=100)
        self.copula_model = advanced_models.GaussianCopula()
        self.change_detector = advanced_models.ChangePointDetector()
        self.granger_analyzer = advanced_models.GrangerCausalityAnalyzer(max_lag=5)
        
        # Fusion
        self.fusion_pipeline = fusion_engine.CompleteFusionPipeline()
        
        # Self-learning
        self.learning_engine = self_learning_engine.SelfLearningPipeline()
        
        # Backtester
        self.backtester = backtester.BacktestingEngine()
        
        # Data cache
        self.macro_data_cache = {}
        self.equity_data_cache = {}
        
        # Results
        self.latest_forecast = {}
        self.model_ensemble_weights = {}
        self.trade_log = []
        
        logger.info(f"Mega India Quant System initialized in {mode} mode")
    
    def fetch_all_data(self, end_date: str = None) -> bool:
        """Fetch all required data."""
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data up to {end_date}")
            
            # Fetch macro data
            self.macro_data_cache = self.data_collector.collect_all_macro_data(end_date)
            
            # Validate data was collected
            if not self.macro_data_cache:
                logger.error("No data collected from sources")
                return False
            
            # Check for at least some data
            valid_sources = 0
            for key, value in self.macro_data_cache.items():
                if isinstance(value, pd.DataFrame) and len(value) > 0:
                    valid_sources += 1
                    logger.info(f"  ✓ {key}: {len(value)} rows")
                else:
                    logger.warning(f"  ✗ {key}: No data")
            
            if valid_sources == 0:
                logger.error("No valid data sources collected")
                return False
            
            logger.info(f"Data fetch completed: {valid_sources} valid sources")
            return True
        
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def train_all_models(self) -> bool:
        """Train all models on collected data."""
        try:
            logger.info("Starting model training...")
            
            # Prepare data - safely combine what's available
            if 'us_yields' not in self.macro_data_cache or len(self.macro_data_cache['us_yields']) == 0:
                logger.warning("Insufficient data for macro training")
                return False
            
            # Combine macro data - handle different indices
            try:
                data_to_combine = []
                for key in ['us_yields', 'inflation', 'commodities', 'fx_data']:
                    if key in self.macro_data_cache and len(self.macro_data_cache[key]) > 0:
                        df = self.macro_data_cache[key]
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            data_to_combine.append(df)
                
                if data_to_combine:
                    # Reset indices and concat to avoid misalignment
                    macro_combined = pd.concat(data_to_combine, axis=1, join='inner')
                    if len(macro_combined) > 0:
                        macro_combined = macro_combined.dropna(how='all')
                    logger.info(f"Combined macro data: {macro_combined.shape}")
                else:
                    macro_combined = pd.DataFrame()
                    logger.warning("No valid macro data to combine")
            except Exception as e:
                logger.warning(f"Failed to combine macro data: {e}")
                macro_combined = pd.DataFrame()
            
            # Train macro models
            if len(macro_combined) > 10:
                try:
                    self.macro_bridge.fit(self.macro_data_cache)
                    logger.info("Macro models trained")
                except Exception as e:
                    logger.warning(f"Macro model training failed: {e}")
            
            # Train DL models on equity data
            try:
                if 'indian_equities' in self.macro_data_cache:
                    equity_df = self.macro_data_cache['indian_equities']
                    if isinstance(equity_df, pd.DataFrame) and len(equity_df) > 100:
                        prices = equity_df['NIFTY50'].values if 'NIFTY50' in equity_df.columns else None
                        
                        if prices is not None and len(prices) > 50:
                            # Ensure numeric data
                            prices = pd.to_numeric(prices, errors='coerce')
                            prices = prices.dropna().values
                            
                            if len(prices) > 50:
                                returns = np.diff(np.log(prices))
                                self.deep_ensemble.train_all(prices, returns)
                                logger.info("Deep learning models trained")
            except Exception as e:
                logger.warning(f"DL model training failed: {e}")
            
            # Train regime switching model
            try:
                if len(macro_combined) > 20:
                    macro_values = macro_combined.values
                    if macro_values.ndim > 1:
                        self.regime_detector.fit(macro_values[-100:] 
                                                if len(macro_values) > 100 else macro_values)
                        logger.info("Regime switching model trained")
            except Exception as e:
                logger.warning(f"Regime switching model training failed: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_macro_signal(self) -> Dict[str, float]:
        """Generate macro-based trading signal."""
        try:
            signals = {}
            
            # Extract macro factors
            if 'us_yields' in self.macro_data_cache:
                yields_df = self.macro_data_cache['us_yields']
                if len(yields_df) > 0:
                    yields_latest = yields_df.iloc[-1].values
                    recession_prob = self.macro_bridge.yield_analyzer.get_yield_curve_signal(yields_latest)
                    signals['recession_probability'] = recession_prob
                    signals['yield_signal'] = 1.0 - recession_prob  # Bullish if no recession
            
            # Macro regime signal
            if 'inflation' in self.macro_data_cache:
                inflation_df = self.macro_data_cache['inflation']
                if len(inflation_df) > 0:
                    inflation_latest = inflation_df.iloc[-1].values
                    regime_signal = self.macro_bridge.get_macro_regime_signal(inflation_latest)
                    signals['regime'] = regime_signal.get('regime_index', -1)
                    signals['growth_signal'] = regime_signal.get('growth_signal', 0.5)
            
            return signals
        
        except Exception as e:
            logger.error(f"Macro signal generation failed: {e}")
            return {}
    
    def generate_factor_signal(self, stock_data: Dict[str, float]) -> float:
        """Generate factor-based trading signal."""
        try:
            # Compute composite factor score
            composite_score = self.factor_model.compute_composite_score(stock_data, use_ff5=True)
            return composite_score
        
        except Exception as e:
            logger.error(f"Factor signal generation failed: {e}")
            return 0.5
    
    def generate_technical_signal(self) -> float:
        """Generate technical analysis signal."""
        try:
            signal = 0.5  # Neutral default
            
            if 'indian_equities' in self.macro_data_cache:
                equity_df = self.macro_data_cache['indian_equities']
                
                if 'NIFTY50' in equity_df.columns and len(equity_df) > 50:
                    prices = equity_df['NIFTY50'].values
                    
                    # Compute technical indicators
                    # Simple momentum
                    momentum = (prices[-1] - prices[-50]) / prices[-50]
                    
                    # Volatility
                    returns = np.diff(np.log(prices[-20:]))
                    volatility = np.std(returns) * np.sqrt(252)
                    
                    # RSI-like indicator
                    ups = np.sum(np.diff(prices[-14:]) > 0)
                    rsi_like = ups / 14.0
                    
                    signal = 0.4 * rsi_like + 0.3 * (1.0 / (1.0 + abs(momentum))) + 0.3 * (1.0 - np.clip(volatility, 0, 1))
            
            return signal
        
        except Exception as e:
            logger.error(f"Technical signal generation failed: {e}")
            return 0.5
    
    def generate_dl_signal(self) -> float:
        """Generate deep learning signal."""
        try:
            if 'indian_equities' in self.macro_data_cache:
                equity_df = self.macro_data_cache['indian_equities']
                
                if 'NIFTY50' in equity_df.columns and len(equity_df) > 60:
                    prices = equity_df['NIFTY50'].values
                    returns = np.diff(np.log(prices))
                    
                    # Get DL ensemble forecast
                    dl_forecast = self.deep_ensemble.forecast_ensemble(prices, returns)
                    
                    # Aggregate signals
                    vol_forecast = dl_forecast.get('lstm_vol', 0.15)
                    price_forecast = dl_forecast.get('transformer_price', prices[-1])
                    
                    # Signal: bullish if price forecast > current
                    signal = 0.5 + 0.5 * (price_forecast - prices[-1]) / prices[-1]
                    signal = np.clip(signal, 0, 1)
                    
                    return signal
            
            return 0.5
        
        except Exception as e:
            logger.error(f"DL signal generation failed: {e}")
            return 0.5
    
    def generate_cv_signal(self) -> Dict[str, any]:
        """Generate CV-based signal."""
        try:
            if 'indian_equities' in self.macro_data_cache:
                equity_df = self.macro_data_cache['indian_equities']
                
                if 'NIFTY50' in equity_df.columns and len(equity_df) > 20:
                    prices = np.array(equity_df['NIFTY50'].values, dtype=float)
                    
                    # Get volumes if available, otherwise use constant
                    if 'volume' in equity_df.columns:
                        volumes = np.array(equity_df['volume'].values, dtype=float)
                    else:
                        volumes = np.ones_like(prices) * 1000000
                    
                    # Create OHLCV data structure (use close for simplicity)
                    ohlcv_data = {
                        'open': prices,
                        'high': prices * 1.01,
                        'low': prices * 0.99,
                        'close': prices,
                        'volume': volumes,
                    }
                    
                    # Run CV pipeline
                    cv_results = self.cv_pipeline.analyze_chart(ohlcv_data)
                    
                    # Extract signal from patterns
                    patterns = cv_results.get('patterns', {})
                    pattern_signal = np.mean(list(patterns.values())) if len(patterns) > 0 else 0.5
                    
                    return {
                        'cv_signal': pattern_signal,
                        'patterns': patterns,
                        'regime_embedding': cv_results.get('regime_embedding', np.zeros(32)),
                    }
            
            return {'cv_signal': 0.5, 'patterns': {}, 'regime_embedding': np.zeros(32)}
        
        except Exception as e:
            logger.error(f"CV signal generation failed: {e}")
            return {'cv_signal': 0.5, 'patterns': {}, 'regime_embedding': np.zeros(32)}
    
    def generate_intraday_signal(self) -> Dict[str, float]:
        """Generate intraday/HFT signal."""
        try:
            # Simulate orderbook data for demonstration
            current_price = 17000.0  # Approximate NIFTY50
            
            orderbook_data = {
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'bid_vol': 1000000,
                'ask_vol': 800000,
                'bid_queue': 1000,
                'ask_queue': 800,
            }
            
            trade_data = {
                'buy_vol': np.array([100000, 150000, 120000, 180000]),
                'sell_vol': np.array([80000, 100000, 110000, 90000]),
                'prices': np.array([current_price * (1 - 0.001 * i) for i in range(4)]),
            }
            
            # Get intraday forecast
            intraday_forecast = self.intraday_engine.intraday_forecast(
                orderbook_data, trade_data, current_time_fraction=0.5)
            
            # Extract signal
            ofi_signal = intraday_forecast.get('ofi_return_signal', 0.0)
            
            return {
                'ofi_signal': ofi_signal,
                'microprice': intraday_forecast.get('microprice', current_price),
                'rv_nowcast': intraday_forecast.get('rv_nowcast', 0.15),
            }
        
        except Exception as e:
            logger.error(f"Intraday signal generation failed: {e}")
            return {'ofi_signal': 0.0, 'microprice': 17000.0, 'rv_nowcast': 0.15}
    
    def generate_microstructure_signal(self) -> float:
        """Generate microstructure-based signal."""
        try:
            # Simulate order flow
            bid_queue = np.array([500, 520, 510, 530, 540])
            ask_queue = np.array([480, 500, 490, 520, 510])
            
            # Queue imbalance
            queue_imb = (np.mean(bid_queue) - np.mean(ask_queue)) / (np.mean(bid_queue) + np.mean(ask_queue))
            
            # Signal: positive imbalance = bullish
            signal = 0.5 + 0.5 * np.tanh(queue_imb * 2.0)
            
            return np.clip(signal, 0, 1)
        
        except Exception as e:
            logger.error(f"Microstructure signal generation failed: {e}")
            return 0.5
    
    def generate_unified_forecast(self) -> Dict[str, any]:
        """Generate final unified forecast combining all signals."""
        try:
            logger.info("Generating unified forecast...")
            
            # Generate all signals
            macro_signals = self.generate_macro_signal()
            macro_signal = macro_signals.get('growth_signal', 0.5)
            
            stock_data = {
                'price': 17000.0,
                'book_value': 8500.0,
                'earnings': 850.0,
                'roe': 12.0,
                'roa': 6.0,
                'debt_to_equity': 0.5,
                'earnings_growth': 15.0,
                'returns': np.random.normal(0, 0.01, 252),
                'price_history': 17000 * np.cumprod(1 + np.random.normal(0, 0.01, 252)),
            }
            
            factor_signal = self.generate_factor_signal(stock_data)
            technical_signal = self.generate_technical_signal()
            dl_signal = self.generate_dl_signal()
            cv_result = self.generate_cv_signal()
            cv_signal = cv_result.get('cv_signal', 0.5)
            intraday_signals = self.generate_intraday_signal()
            microstructure_signal = self.generate_microstructure_signal()
            
            # Get current regime
            regime = macro_signals.get('regime', 0)
            
            # Fuse all signals
            fused_forecast = self.fusion_pipeline.fuse_all_signals(
                macro_signal=macro_signal,
                technical_signal=technical_signal,
                ml_signal=factor_signal,
                dl_signal=dl_signal,
                cv_signal=cv_signal,
                microstructure_signal=microstructure_signal,
                regime=int(regime)
            )
            
            # Build complete forecast (ensure all values are scalars)
            self.latest_forecast = {
                'timestamp': str(datetime.now()),
                'macro_signal': float(macro_signal),
                'factor_signal': float(factor_signal),
                'technical_signal': float(technical_signal),
                'dl_signal': float(dl_signal),
                'cv_signal': float(cv_signal),
                'intraday_signals': {k: float(v) if isinstance(v, (int, float)) else v 
                                    for k, v in intraday_signals.items()},
                'microstructure_signal': float(microstructure_signal),
                'fused_forecast': {k: float(v) if isinstance(v, (int, float)) else v 
                                  for k, v in fused_forecast.items()},
                'final_signal': float(fused_forecast.get('final_signal', 0.5)),
                'regime': int(regime) if isinstance(regime, (int, float, np.integer)) else -1,
                'confidence': float(self._compute_confidence(fused_forecast)),
            }
            
            logger.info(f"Forecast generated. Final signal: {self.latest_forecast['final_signal']:.3f}")
            
            return self.latest_forecast
        
        except Exception as e:
            logger.error(f"Unified forecast generation failed: {e}")
            return {}
    
    def _compute_confidence(self, fused_forecast: Dict) -> float:
        """Compute confidence level of forecast."""
        weights = fused_forecast.get('attention_weights', {})
        
        if len(weights) == 0:
            return 0.5
        
        weight_values = [v for k, v in weights.items() if isinstance(v, float)]
        if len(weight_values) == 0:
            return 0.5
        
        # Entropy-based confidence (lower entropy = higher confidence)
        entropy = -np.sum([w * np.log(w + 1e-8) for w in weight_values])
        max_entropy = np.log(len(weight_values))
        
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        return np.clip(confidence, 0, 1)
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, any]:
        """Run backtest on historical data."""
        try:
            logger.info("Running backtest...")
            
            for position, (idx, row) in enumerate(data.iterrows()):
                # Generate forecast for this date
                # Simplified: use returns to generate signals
                
                # Simulated signal
                if 'NIFTY50' in data.columns:
                    price = float(row['NIFTY50'])
                    # Use position counter instead of timestamp for signal generation
                    signal = 0.5 + 0.3 * np.sin(position / 50.0)  # Oscillating signal
                    
                    if signal > 0.6:
                        self.backtester.open_position(str(idx), 'NIFTY50', 'long', price, signal - 0.5)
                    
                    if signal < 0.4:
                        if 'NIFTY50' in self.backtester.positions:
                            self.backtester.close_position(str(idx), 'NIFTY50', price)
                    
                    # Mark to market
                    self.backtester.mark_to_market({'NIFTY50': price})
            
            # Close remaining positions
            if len(data) > 0:
                final_price = float(data.iloc[-1]['NIFTY50']) if 'NIFTY50' in data.columns else 17000.0
                if 'NIFTY50' in self.backtester.positions:
                    self.backtester.close_position(str(data.index[-1]), 'NIFTY50', final_price)
            
            # Get results
            results = self.backtester.generate_report()
            
            logger.info(f"Backtest complete. Final capital: {results['summary'].get('final_capital', 0):.2f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            return {}
    
    def generate_full_report(self) -> Dict[str, any]:
        """Generate comprehensive system report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'latest_forecast': self.latest_forecast,
            'model_ensemble_weights': self.model_ensemble_weights,
            'system_status': 'operational',
        }
        
        return report
    
    def save_results(self, filepath: str) -> None:
        """Save results to file."""
        try:
            report = self.generate_full_report()
            
            # Convert non-serializable objects
            report_clean = self._make_serializable(report)
            
            with open(filepath, 'w') as f:
                json.dump(report_clean, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable form."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        else:
            return obj
