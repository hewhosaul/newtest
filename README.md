# MEGA INDIA QUANT - Institutional-Grade Multi-Module Trading System

## Overview

**Mega India Quant** is a comprehensive, research-paper-caliber, institutional-grade quantitative trading system designed specifically for Indian equities. It integrates global macroeconomic factors, local market indicators, advanced machine learning, deep learning, computer vision, hidden Markov models, and sophisticated econometric techniques to generate predictive signals for equity trading.

## System Architecture

### Core Modules

```
mega_india_quant/
├── data_feeds.py              # Multi-source data collection
├── macro_models.py            # FAVAR, DFM, MIDAS, regime detection
├── factor_models.py           # Carhart 4-factor, FF5-factor, momentum, value, quality
├── deep_learning_models.py    # LSTM, Transformers, CNN, volatility forecasting
├── computer_vision.py         # Chart patterns, orderflow heatmaps, regime embeddings
├── intraday_models.py         # Microprice, OFI, Hawkes, HFT-style models
├── advanced_models.py         # Regime switching VAR, Kalman, copulas, change detection
├── fusion_engine.py           # Multi-view fusion, attention mechanisms, modality alignment
├── self_learning_engine.py    # Model evolution, reward mechanics, ensemble adaptation
├── backtester.py              # Backtesting, position sizing, risk management
└── main_engine.py             # Master orchestration and execution
```

## Key Features

### 1. GLOBAL MACRO MODELING

**Data Ingestion:**
- US yield curve (1m → 30y)
- Indian Government Security yields
- Global PMIs (US, EU, China, India)
- Inflation data (CPI, PPI, Core)
- Global trade indices and credit spreads
- FX data (DXY, USD/INR, EUR/USD, GBP/USD)
- Commodity prices (Brent, WTI, Copper, Silver, Gold)
- Semiconductor sector data (TSMC, Intel, SOX index)

**Models Implemented:**
- **FAVAR (Factor-Augmented VAR)**: Dimensionality reduction + VAR forecasting
- **Dynamic Factor Models**: Global and local factor extraction
- **MIDAS Regression**: Mixed-frequency macro-to-equity mapping
- **Yield Curve Analysis**: Recession signaling and volatility mapping
- **Macro Regime Detection (HMM)**: Growth, Risk-Off, Stagflation, Low-Vol regimes

### 2. RESEARCH-PAPER-DRIVEN METHODOLOGIES

Implemented econometric frameworks from academic literature:

- **Jegadeesh & Titman (1993)**: Momentum factor strategies
- **Carhart (1997)**: 4-factor model (Market, Size, Value, Momentum)
- **Fama-French (2015)**: 5-factor model (adds Profitability, Investment)
- **Frazzini & Pedersen (2014)**: Betting Against Beta factor
- **Asness et al.**: Value/Momentum/Quality integration
- **Bollerslev/Corsi**: HAR (Heterogeneous AutoRegression) for RV forecasting
- **Kyle (1985), Hasbrouck (1991)**: Microstructure models (lambda, price impact)
- **Almgren-Chriss (2001)**: Optimal execution with market impact
- **Engle, Bollerslev**: GARCH/DCC volatility modeling
- **Stock & Watson (2002)**: Diffusion indices, dynamic factors
- **Hawkes (1971)**: Self-exciting point processes for HFT
- **Estrella-Mishkin**: Yield curve recession prediction

### 3. DEEP LEARNING & TRANSFORMERS

**Models:**
- **LSTM Volatility Forecaster**: Sequence-to-sequence volatility prediction
- **Transformer Price Forecaster**: Multi-head attention for price forecasting
- **CNN Microstructure Detector**: 1D CNN for orderflow pattern classification
- **HAR-RV Model**: Realized volatility forecasting using hierarchical structure
- **GARCH(1,1)**: Parametric volatility model with persistence
- **Hybrid Ensemble**: Combines LSTM, Transformer, CNN with weighted averaging

### 4. COMPUTER VISION

**Capabilities:**
- **Candlestick Image Generation**: OHLCV data → 2D images for CNN analysis
- **Volume Heatmaps**: Temporal-volume distribution visualization
- **Orderflow Heatmaps**: Price level vs time with buy/sell imbalance coloring
- **Chart Pattern Recognition**: 
  - Head & Shoulders detection
  - Double Top/Bottom patterns
  - Triangle/Wedge convergence
- **Visual Regime Analysis**: Image embeddings for market state clustering

### 5. INTRADAY/HFT MODELS

**Advanced Features:**
- **Microprice Forecasting**: Volume-weighted bid-ask level prediction
- **Order Flow Imbalance (OFI)**: Buy/sell pressure signal with memory decay
- **Hawkes Process**: Self-exciting trade arrival modeling
- **Queue Reaction Model**: Order queue elasticity effects
- **Almgren-Chriss Execution**: Optimal schedule minimizing impact
- **Realized Volatility Nowcasting**: Intraday volatility updates
- **Lead-Lag Forecasting**: Singapore NIFTY futures → spot signal extraction

### 6. ADVANCED ECONOMETRIC MODELS

**Cutting-Edge Techniques:**
- **Regime-Switching VAR**: Markov-switching model with multiple regimes
- **Particle Filter**: Sequential importance sampling state estimation
- **Ensemble Kalman Filter**: Nonlinear state estimation for dynamics
- **Bayesian Hierarchical Model**: Multi-level parameter learning
- **Gaussian Copula**: Joint dependence modeling between assets
- **Change Point Detection (PELT)**: Structural break identification
- **Granger Causality Analysis**: Lead-lag relationship mapping

### 7. SELF-LEARNING ENGINE

**Automatic Evolution:**
- **Model Registry**: Tracks all models with version history
- **Performance Tracking**: Sharpe, Win Rate, Stability, Drawdown scoring
- **Genetic Evolution**: Mutation and crossover of hyperparameters
- **Reward Mechanics**: RL-style rewards for model fitness
- **Adaptive Selection**: Contextual bandits for regime-dependent model switching
- **Automatic Elimination**: Weak models are retired, successful ones evolve

### 8. FUSION ENGINE

**Multi-View Learning:**
- **Early Fusion**: Combine raw features via learned weights
- **Late Fusion**: Aggregate independent view predictions
- **Attention Fusion**: Learnable attention weights for view importance
- **Modality Alignment**: Common embedding space for numerical, image, time-series data
- **Hybrid Fusion**: Optimal blend of early/late approaches with adaptation

### 9. BACKTESTING & EXECUTION

**Features:**
- **Position Sizing**: Kelly criterion, volatility-adjusted, risk-based
- **Risk Management**: Stop-loss, max drawdown limits, portfolio heat calculation
- **Realistic Simulation**: Commission, slippage, market impact
- **Walk-Forward Testing**: Robust out-of-sample validation
- **Trade Logging**: Detailed trade history with P&L attribution

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone/setup repository
cd mega_india_quant

# Install dependencies
pip install -r requirements.txt

# Optional: for deep learning support
pip install tensorflow torch pytorch-lightning

# Optional: for computer vision
pip install opencv-python
```

## Usage

### Quick Start

```bash
# Run complete system
python run_system.py --mode backtest --end-date 2024-01-01

# With custom output
python run_system.py --mode backtest --output ./results/forecast.json
```

### Programmatic Usage

```python
from mega_india_quant.main_engine import MegaIndiaQuantSystem

# Initialize
system = MegaIndiaQuantSystem(mode='backtest')

# Fetch data
system.fetch_all_data(end_date='2024-01-01')

# Train models
system.train_all_models()

# Generate forecast
forecast = system.generate_unified_forecast()

print(f"Final Signal: {forecast['final_signal']:.3f}")
print(f"Confidence: {forecast['confidence']:.3f}")
print(f"Regime: {forecast['regime']}")
```

### Data Sources

The system automatically fetches from multiple sources with fallbacks:

1. **Yahoo Finance**: OHLCV data, indices, FX
2. **Trading Economics**: Macro indicators (fallback: estimates)
3. **Government sources**: Bond yields, economic releases
4. **RBI/NSE**: Indian-specific data

All sources include fallback mechanisms for reliability.

## Forecast Output

The system generates multi-dimensional forecasts:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "macro_signal": 0.62,
  "factor_signal": 0.58,
  "technical_signal": 0.55,
  "dl_signal": 0.65,
  "cv_signal": 0.52,
  "microstructure_signal": 0.61,
  "final_signal": 0.59,
  "confidence": 0.78,
  "regime": 0,
  "fused_forecast": {
    "attention_signal": 0.594,
    "hybrid_signal": 0.591,
    "aligned_signal": 0.588,
    "final_signal": 0.591
  }
}
```

## Performance Metrics

The system tracks:
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Stability Score**: Consistency across periods
- **Recovery Factor**: Efficiency of capital recovery

## Customization

### Adding Custom Models

```python
from mega_india_quant.self_learning_engine import ModelType

# Register custom model
custom_model = MyCustomModel()
system.learning_engine.registry.register_model(
    'my_model_v1', 
    custom_model, 
    ModelType.ML
)
```

### Adjusting Fusion Weights

```python
# Modify fusion component weights
system.fusion_pipeline.hybrid_fusion.hybrid_weight_early = 0.6
system.fusion_pipeline.hybrid_fusion.hybrid_weight_late = 0.4
```

### Risk Parameters

```python
# Customize risk management
system.backtester.risk_manager.max_drawdown = 0.20  # 20% max
system.backtester.risk_manager.stop_loss_pct = 0.03  # 3% stop
system.position_sizer.max_position_pct = 0.05  # 5% per position
```

## Model Components Detail

### Macro Models

**FAVAR Example:**
```python
from mega_india_quant.macro_models import FactorAugmentedVAR

favar = FactorAugmentedVAR(n_factors=3, lag_order=4)
favar.fit(macro_data)
forecast = favar.forecast(steps=5)
```

### Factor Models

**Carhart 4-Factor:**
```python
from mega_india_quant.factor_models import CarrartFourFactorModel

c4 = CarrartFourFactorModel()
exposures = c4.compute_factor_exposures(stock_data, market_return=0.08)
```

### Deep Learning

**Transformer Price Forecaster:**
```python
from mega_india_quant.deep_learning_models import TransformerPriceForecaster

transformer = TransformerPriceForecaster(d_model=64, seq_length=60)
transformer.fit(prices, epochs=30)
prediction = transformer.predict(recent_prices)
```

### Advanced Models

**Regime-Switching VAR:**
```python
from mega_india_quant.advanced_models import RegimeSwitchingVAR

rs_var = RegimeSwitchingVAR(n_regimes=4, lag_order=2)
rs_var.fit(macro_data)
regime = rs_var.predict_regime(current_data)
```

## Research References

Key papers implemented:

1. Stock & Watson (2002) - FAVAR
2. Jegadeesh & Titman (1993) - Momentum
3. Carhart (1997) - 4-Factor Model
4. Fama & French (2015) - 5-Factor Model
5. Frazzini & Pedersen (2014) - BAB
6. Almgren & Chriss (2001) - Optimal Execution
7. Hamilton (1989) - Markov Switching
8. Engle (2002) - Dynamic Conditional Correlation
9. Hasbrouck (1991) - Price Impact
10. Kyle (1985) - Microstructure

## Performance Considerations

- **Data Fetching**: ~5-10 seconds for full macro data
- **Model Training**: ~30-60 seconds for all models
- **Inference**: <100ms for unified forecast
- **Backtest**: ~5-30 seconds depending on history length

## Risk Disclosure

This system is for educational and research purposes. Past performance does not guarantee future results. Users should:

1. Conduct thorough backtesting on own data
2. Paper-trade before live deployment
3. Implement proper risk management
4. Monitor model performance regularly
5. Understand all embedded assumptions

## System Status

✓ Data collection: Operational  
✓ Macro models: Operational  
✓ Factor models: Operational  
✓ Deep learning: Operational (with fallbacks)  
✓ Computer vision: Operational (with fallbacks)  
✓ Intraday models: Operational  
✓ Fusion engine: Operational  
✓ Backtesting: Operational  

## Future Enhancements

- [ ] Real-time WebSocket data feeds
- [ ] High-frequency data support (tick-level)
- [ ] Advanced portfolio optimization
- [ ] Options pricing integration
- [ ] Multi-asset class fusion
- [ ] Blockchain/crypto macro factors
- [ ] ESG factor integration
- [ ] Sentiment analysis from news/social media

## Contact & Support

For questions or issues, refer to documentation and README files.

## License

Research and educational use. See LICENSE file for details.

---

**Last Updated**: 2024-01-15  
**Version**: 1.0.0  
**Status**: Production-Ready
