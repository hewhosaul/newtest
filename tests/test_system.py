"""
Unit tests for Mega India Quant System.
Tests core functionality without requiring live data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import system modules
from mega_india_quant.macro_models import (
    FactorAugmentedVAR, DynamicFactorModel, MacroRegimeDetector,
    YieldCurveAnalyzer, MIDASRegression
)
from mega_india_quant.factor_models import (
    MomentumFactor, ValueFactor, QualityFactor, CarrartFourFactorModel,
    FamaFrenchFiveFactorModel, BettingAgainstBetaModel, MultiFactorCombiner
)
from mega_india_quant.advanced_models import (
    RegimeSwitchingVAR, ParticleFilter, GaussianCopula,
    ChangePointDetector, GrangerCausalityAnalyzer
)
from mega_india_quant.fusion_engine import (
    AttentionFusionModule, EarlyFusionModel, LateFusionModel,
    HybridFusionEngine, CompleteFusionPipeline
)
from mega_india_quant.self_learning_engine import (
    ModelPerformance, ModelRegistry, ModelType, ModelEvolutionEngine,
    RewardMechanism, SelfLearningPipeline
)
from mega_india_quant.backtester import BacktestingEngine, PositionSizer


class TestMacroModels:
    """Test macro modeling components."""
    
    def test_favar(self):
        """Test FAVAR model."""
        # Generate synthetic data
        X = np.random.randn(100, 5)
        
        favar = FactorAugmentedVAR(n_factors=2, lag_order=2)
        favar.fit(X)
        
        forecast = favar.forecast(steps=5)
        assert forecast.shape[0] == 5
        assert favar.get_explained_variance() > 0
    
    def test_macro_regime_detector(self):
        """Test HMM regime detection."""
        X = np.random.randn(100, 3)
        
        detector = MacroRegimeDetector(n_regimes=3)
        detector.fit(X)
        
        regime = detector.predict_regime(np.random.randn(3))
        assert 0 <= regime < 3
        
        regime_name = detector.get_regime_name(0)
        assert isinstance(regime_name, str)
    
    def test_yield_curve_analyzer(self):
        """Test yield curve analysis."""
        yields_df = pd.DataFrame({
            '1Y': np.linspace(1, 2, 50),
            '5Y': np.linspace(2, 3, 50),
            '10Y': np.linspace(3, 4, 50),
            '30Y': np.linspace(4, 5, 50),
        })
        
        analyzer = YieldCurveAnalyzer()
        analyzer.fit_yield_curve(yields_df)
        
        spread = analyzer.compute_term_spread(yields_df)
        assert len(spread) == len(yields_df)
        
        signal = analyzer.get_yield_curve_signal(np.array([1.0, 2.0, 3.0, 4.0]))
        assert 0 <= signal <= 1


class TestFactorModels:
    """Test factor models."""
    
    def test_momentum_factor(self):
        """Test momentum computation."""
        momentum_factor = MomentumFactor(lookback_period=252)
        
        prices = np.linspace(100, 120, 300)
        momentum = momentum_factor.compute_momentum(prices)
        
        assert isinstance(momentum, float)
    
    def test_value_factor(self):
        """Test value score."""
        value_factor = ValueFactor()
        
        score = value_factor.compute_value_score(price=100, book_value=50, earnings=10)
        assert 0 <= score <= 2
    
    def test_quality_factor(self):
        """Test quality score."""
        quality_factor = QualityFactor()
        
        score = quality_factor.compute_quality_score(roe=15, debt_to_equity=0.5, 
                                                    earnings_growth=10)
        assert 0 <= score <= 1
    
    def test_carhart_4factor(self):
        """Test Carhart 4-factor model."""
        c4_model = CarrartFourFactorModel()
        
        stock_data = {
            'price': 100,
            'book_value': 50,
            'earnings': 10,
            'roe': 15,
            'debt_to_equity': 0.5,
            'earnings_growth': 10,
            'returns': np.random.normal(0, 0.01, 252),
            'price_history': 100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)),
        }
        
        exposures = c4_model.compute_factor_exposures(stock_data, market_excess_return=0.08)
        assert 'beta' in exposures or 'value' in exposures


class TestAdvancedModels:
    """Test advanced econometric models."""
    
    def test_regime_switching_var(self):
        """Test regime-switching VAR."""
        X = np.random.randn(100, 2)
        
        rs_var = RegimeSwitchingVAR(n_regimes=2, lag_order=1)
        rs_var.fit(X)
        
        regime = rs_var.predict_regime(np.random.randn(2))
        assert 0 <= regime < 2
    
    def test_gaussian_copula(self):
        """Test Gaussian copula."""
        data = np.random.randn(100, 2)
        
        copula = GaussianCopula()
        copula.fit(data)
        
        samples = copula.simulate(n_samples=10, n_vars=2)
        assert samples.shape == (10, 2)
        assert np.all((samples >= 0) & (samples <= 1))
    
    def test_change_point_detector(self):
        """Test change point detection."""
        # Create data with change point
        X = np.concatenate([np.random.normal(0, 1, 50),
                           np.random.normal(5, 1, 50)])
        
        detector = ChangePointDetector(penalty='bic')
        changepoints = detector.fit(X)
        
        assert isinstance(changepoints, list)
    
    def test_granger_causality(self):
        """Test Granger causality."""
        X = np.random.randn(100)
        Y = np.random.randn(100)
        
        analyzer = GrangerCausalityAnalyzer(max_lag=3)
        result = analyzer.test_causality(X, Y)
        
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'granger_causes' in result


class TestFusionEngine:
    """Test fusion components."""
    
    def test_attention_fusion(self):
        """Test attention-based fusion."""
        from mega_india_quant.fusion_engine import FeatureView
        
        views = [
            FeatureView('view1', np.array([0.5])),
            FeatureView('view2', np.array([0.6])),
        ]
        
        fusion = AttentionFusionModule(n_views=2)
        weights = fusion.compute_attention_scores(views)
        
        assert len(weights) == 2
        assert np.abs(np.sum(weights) - 1.0) < 1e-6
    
    def test_hybrid_fusion(self):
        """Test hybrid fusion."""
        hybrid = HybridFusionEngine()
        
        macro = np.array([0.5])
        technical = np.array([0.6])
        ml = np.array([0.55])
        
        view_preds = {'macro': 0.5, 'technical': 0.6, 'ml': 0.55}
        
        pred = hybrid.predict(macro, technical, ml, view_preds)
        assert 0 <= pred <= 1


class TestSelfLearning:
    """Test self-learning engine."""
    
    def test_model_performance(self):
        """Test model performance metrics."""
        perf = ModelPerformance(
            model_name='test_model',
            model_type=ModelType.ML,
            sharpe_ratio=1.5,
            win_rate=0.55,
            total_return=0.10,
            max_drawdown=0.15,
            stability_score=0.8,
            training_time=1.0,
            inference_time=10.0,
            prediction_accuracy=0.55
        )
        
        fitness = perf.compute_fitness()
        assert 0 <= fitness <= 1
    
    def test_model_registry(self):
        """Test model registry."""
        registry = ModelRegistry()
        
        mock_model = type('MockModel', (), {})()
        registry.register_model('test1', mock_model, ModelType.ML)
        
        assert registry.get_model('test1') is not None
        
        perf = ModelPerformance(
            model_name='test1',
            model_type=ModelType.ML,
            sharpe_ratio=1.5,
            win_rate=0.55,
            total_return=0.10,
            max_drawdown=0.15,
            stability_score=0.8,
            training_time=1.0,
            inference_time=10.0,
            prediction_accuracy=0.55
        )
        
        registry.record_performance('test1', perf)
        assert len(registry.performance_history['test1']) > 0
    
    def test_reward_mechanism(self):
        """Test reward mechanism."""
        reward = RewardMechanism()
        
        returns = np.random.normal(0.001, 0.01, 252)
        
        sharpe = reward.compute_sharpe_reward(returns)
        assert isinstance(sharpe, float)
        
        win_rate = reward.compute_win_rate_reward(returns)
        assert 0 <= win_rate <= 1
        
        stability = reward.compute_stability_reward(returns)
        assert 0 <= stability <= 1


class TestBacktester:
    """Test backtesting engine."""
    
    def test_position_sizer(self):
        """Test position sizing."""
        sizer = PositionSizer(initial_capital=1000000, max_position_pct=0.02)
        
        kelly_size = sizer.compute_kelly_position(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        assert 0 <= kelly_size <= 0.02
        
        vol_adjusted = sizer.compute_volatility_adjusted_size(price=100, volatility=0.15)
        assert vol_adjusted > 0
    
    def test_backtesting_engine(self):
        """Test backtesting engine."""
        backtester = BacktestingEngine(initial_capital=1000000)
        
        # Simulate trades
        backtester.open_position(datetime.now(), 'TEST', 'long', 100, 1.0)
        backtester.close_position(datetime.now(), 'TEST', 105)
        
        metrics = backtester.get_performance_metrics()
        assert 'sharpe_ratio' in metrics
        assert metrics['num_trades'] == 1


def test_system_integration():
    """Integration test of complete system."""
    from mega_india_quant.main_engine import MegaIndiaQuantSystem
    
    system = MegaIndiaQuantSystem(mode='backtest')
    
    # Test that system initializes without error
    assert system is not None
    assert system.mode == 'backtest'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
