"""
Self-learning engine: automatic model selection, evolution, and reward mechanics.
Uses RL-like reward signals (Sharpe ratio, win rate) to evolve models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model categories."""
    MACRO = "macro"
    FACTOR = "factor"
    TECHNICAL = "technical"
    ML = "ml"
    DL = "dl"
    CV = "cv"
    MICROSTRUCTURE = "microstructure"
    ADVANCED = "advanced"


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    model_type: ModelType
    sharpe_ratio: float
    win_rate: float
    total_return: float
    max_drawdown: float
    stability_score: float  # Consistency across different time periods
    training_time: float
    inference_time: float
    prediction_accuracy: float  # For classification models
    
    def compute_fitness(self) -> float:
        """Compute overall fitness score for model selection."""
        # Weighted combination of metrics
        fitness = (
            0.35 * np.clip(self.sharpe_ratio / 2.0, 0, 1) +  # Sharpe normalized
            0.25 * np.clip(self.win_rate, 0, 1) +
            0.20 * np.clip(self.stability_score, 0, 1) +
            0.15 * np.clip(1.0 - self.max_drawdown, 0, 1) +  # Lower drawdown is better
            0.05 * np.clip(1.0 / (1.0 + self.inference_time), 0, 1)  # Speed bonus
        )
        return fitness
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'stability_score': self.stability_score,
            'fitness': self.compute_fitness(),
        }


class ModelRegistry:
    """Registry of all models in the system."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
    
    def register_model(self, name: str, model: Any, model_type: ModelType) -> None:
        """Register a model."""
        self.models[name] = {
            'instance': model,
            'type': model_type,
            'created_at': pd.Timestamp.now(),
            'status': 'active'
        }
        self.performance_history[name] = []
        logger.info(f"Registered model: {name} ({model_type.value})")
    
    def record_performance(self, model_name: str, performance: ModelPerformance) -> None:
        """Record performance metrics for a model."""
        if model_name in self.performance_history:
            self.performance_history[model_name].append(performance)
    
    def get_best_models(self, model_type: Optional[ModelType] = None, top_k: int = 5) -> List[str]:
        """Get top-k models by fitness."""
        candidates = []
        
        for model_name, perf_list in self.performance_history.items():
            if len(perf_list) == 0:
                continue
            
            if model_type is not None and self.models[model_name]['type'] != model_type:
                continue
            
            # Use latest performance
            latest_perf = perf_list[-1]
            fitness = latest_perf.compute_fitness()
            candidates.append((model_name, fitness))
        
        # Sort by fitness
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in candidates[:top_k]]
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get model instance."""
        if name in self.models:
            return self.models[name]['instance']
        return None
    
    def retire_model(self, name: str) -> None:
        """Mark model as inactive."""
        if name in self.models:
            self.models[name]['status'] = 'retired'
            logger.info(f"Retired model: {name}")


class ModelEvolutionEngine:
    """
    Evolutionary algorithm for generating new models from successful ones.
    Implements mutation and crossover of hyperparameters.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.mutation_rate = 0.3
        self.population_size = 20
        self.generation_count = 0
    
    def mutate_hyperparameters(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """
        Generate mutated hyperparameters from base model.
        Implements Gaussian mutation.
        """
        mutated = {}
        
        for key, value in base_params.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation with std = 10% of current value
                mutation_std = abs(value * 0.1) if value != 0 else 0.1
                mutation = np.random.normal(0, mutation_std)
                
                # Keep within reasonable bounds
                mutated[key] = np.clip(value + mutation, value * 0.5, value * 2.0)
            else:
                mutated[key] = value
        
        return mutated
    
    def crossover_hyperparameters(self, 
                                 parent1_params: Dict[str, float],
                                 parent2_params: Dict[str, float]) -> Dict[str, float]:
        """
        Crossover (blend) two parent parameter sets.
        """
        offspring = {}
        
        for key in parent1_params.keys():
            if key in parent2_params:
                # Uniform crossover + blend
                if random.random() < 0.5:
                    # Take from parent 1, then mutate slightly
                    offspring[key] = parent1_params[key]
                else:
                    # Blend both parents
                    alpha = random.random()
                    offspring[key] = alpha * parent1_params[key] + (1 - alpha) * parent2_params[key]
                
                # Apply small mutation
                if random.random() < self.mutation_rate:
                    mutation = np.random.normal(0, offspring[key] * 0.05)
                    offspring[key] = offspring[key] + mutation
            else:
                offspring[key] = parent1_params[key]
        
        return offspring
    
    def evolve_generation(self) -> List[Dict[str, Any]]:
        """
        Generate new model variations from best existing models.
        Returns: list of new model configurations to train
        """
        self.generation_count += 1
        
        # Get best models by type
        new_configs = []
        
        for model_type in ModelType:
            best_models = self.registry.get_best_models(model_type=model_type, top_k=3)
            
            if len(best_models) == 0:
                continue
            
            # Create mutations
            for best_model_name in best_models:
                model = self.registry.get_model(best_model_name)
                
                if hasattr(model, 'hyperparameters'):
                    base_params = model.hyperparameters
                else:
                    # Fallback: generic hyperparameters
                    base_params = {'learning_rate': 0.01, 'batch_size': 32}
                
                # Generate 2 mutations per best model
                for _ in range(2):
                    mutated = self.mutate_hyperparameters(base_params)
                    new_configs.append({
                        'type': model_type,
                        'parent': best_model_name,
                        'hyperparameters': mutated,
                        'generation': self.generation_count,
                    })
        
        # Generate 2-parent crossovers
        if len(self.registry.models) >= 2:
            best_overall = self.registry.get_best_models(top_k=5)
            
            for i in range(min(3, len(best_overall) - 1)):
                parent1 = self.registry.get_model(best_overall[i])
                parent2 = self.registry.get_model(best_overall[i + 1])
                
                if (hasattr(parent1, 'hyperparameters') and 
                    hasattr(parent2, 'hyperparameters')):
                    offspring_params = self.crossover_hyperparameters(
                        parent1.hyperparameters,
                        parent2.hyperparameters
                    )
                    
                    new_configs.append({
                        'type': parent1.__class__.__name__,
                        'parent1': best_overall[i],
                        'parent2': best_overall[i + 1],
                        'hyperparameters': offspring_params,
                        'generation': self.generation_count,
                    })
        
        logger.info(f"Evolution generation {self.generation_count}: generated {len(new_configs)} new configs")
        return new_configs


class RewardMechanism:
    """
    Reward mechanics for model evaluation and selection.
    Uses RL-inspired rewards combining financial and statistical metrics.
    """
    
    def __init__(self):
        self.reward_components = {
            'sharpe': 0.35,
            'win_rate': 0.25,
            'stability': 0.20,
            'drawdown': 0.15,
            'speed': 0.05,
        }
    
    def compute_sharpe_reward(self, returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Compute Sharpe ratio reward (unbounded, typically 1-3)."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe
    
    def compute_win_rate_reward(self, returns: np.ndarray) -> float:
        """Compute win rate reward (0-1)."""
        if len(returns) == 0:
            return 0.5
        
        wins = np.sum(returns > 0)
        return wins / len(returns)
    
    def compute_stability_reward(self, returns: np.ndarray, window: int = 50) -> float:
        """
        Compute stability score (consistency across subperiods).
        Returns 0-1, where 1 is perfect consistency.
        """
        if len(returns) < window:
            return 0.5
        
        # Compute rolling Sharpe ratios
        rolling_sharpes = []
        for i in range(len(returns) - window):
            period_returns = returns[i:i + window]
            if np.std(period_returns) > 0:
                sharpe = np.mean(period_returns) / np.std(period_returns)
                rolling_sharpes.append(sharpe)
        
        if len(rolling_sharpes) == 0:
            return 0.5
        
        # Stability = inverse of coefficient of variation
        mean_sharpe = np.mean(rolling_sharpes)
        if mean_sharpe == 0:
            return 0.0
        
        cv = np.std(rolling_sharpes) / abs(mean_sharpe)
        stability = 1.0 / (1.0 + cv)
        return np.clip(stability, 0.0, 1.0)
    
    def compute_drawdown_reward(self, returns: np.ndarray) -> float:
        """
        Compute drawdown reward (0-1, lower drawdown = higher reward).
        """
        if len(returns) == 0:
            return 0.5
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)
        
        # Convert to reward (1 - drawdown, clipped)
        return np.clip(1.0 - max_drawdown, 0.0, 1.0)
    
    def compute_speed_reward(self, inference_time_ms: float) -> float:
        """Speed reward (faster = higher reward)."""
        # Reward times < 100ms heavily
        if inference_time_ms < 100:
            return 1.0
        elif inference_time_ms < 1000:
            return 0.5
        else:
            return 0.1
    
    def compute_total_reward(self, 
                            returns: np.ndarray,
                            inference_time_ms: float = 10.0) -> float:
        """Compute total reward combining all components."""
        sharpe = self.compute_sharpe_reward(returns)
        win_rate = self.compute_win_rate_reward(returns)
        stability = self.compute_stability_reward(returns)
        drawdown = self.compute_drawdown_reward(returns)
        speed = self.compute_speed_reward(inference_time_ms)
        
        # Normalize components
        sharpe_norm = np.clip(sharpe / 2.0, 0, 1)
        
        total_reward = (
            self.reward_components['sharpe'] * sharpe_norm +
            self.reward_components['win_rate'] * win_rate +
            self.reward_components['stability'] * stability +
            self.reward_components['drawdown'] * drawdown +
            self.reward_components['speed'] * speed
        )
        
        return total_reward


class AdaptiveModelSelector:
    """
    Selects best model for current market regime.
    Uses contextual bandits approach.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.regime_model_mapping: Dict[int, List[str]] = {}
        self.selection_counts: Dict[str, int] = {}
        self.selection_rewards: Dict[str, float] = {}
    
    def map_models_to_regimes(self, regime_labels: np.ndarray, model_returns: Dict[str, np.ndarray]) -> None:
        """
        Map models to regimes based on historical performance.
        regime_labels: array of regime indices
        model_returns: dict of {model_name: returns array}
        """
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            best_models_regime = []
            
            for model_name, returns in model_returns.items():
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_sharpe = np.mean(regime_returns) / (np.std(regime_returns) + 1e-6)
                    best_models_regime.append((model_name, regime_sharpe))
            
            # Keep top 3 models for this regime
            best_models_regime.sort(key=lambda x: x[1], reverse=True)
            self.regime_model_mapping[int(regime)] = [m[0] for m in best_models_regime[:3]]
    
    def select_model(self, current_regime: int, epsilon: float = 0.1) -> Optional[str]:
        """
        Select model for current regime using epsilon-greedy strategy.
        """
        if current_regime not in self.regime_model_mapping:
            # Fallback: select best overall model
            best_models = self.registry.get_best_models(top_k=1)
            return best_models[0] if len(best_models) > 0 else None
        
        candidates = self.regime_model_mapping[current_regime]
        
        if len(candidates) == 0:
            return None
        
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < epsilon:
            selected = random.choice(candidates)
        else:
            # Exploit: select model with highest cumulative reward
            selected = max(candidates, 
                         key=lambda m: self.selection_rewards.get(m, 0.0))
        
        # Update selection count
        self.selection_counts[selected] = self.selection_counts.get(selected, 0) + 1
        
        return selected
    
    def update_model_reward(self, model_name: str, reward: float) -> None:
        """Update model's cumulative reward."""
        if model_name not in self.selection_rewards:
            self.selection_rewards[model_name] = 0.0
        
        # Exponential moving average
        alpha = 0.1
        self.selection_rewards[model_name] = (
            alpha * reward + (1 - alpha) * self.selection_rewards[model_name]
        )


class SelfLearningPipeline:
    """
    Main self-learning pipeline that ties everything together.
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.evolution_engine = ModelEvolutionEngine(self.registry)
        self.reward_mechanism = RewardMechanism()
        self.model_selector = AdaptiveModelSelector(self.registry)
        self.training_history = []
    
    def register_all_models(self, models_dict: Dict[str, Tuple[Any, ModelType]]) -> None:
        """Register all models from a dictionary."""
        for model_name, (model_instance, model_type) in models_dict.items():
            self.registry.register_model(model_name, model_instance, model_type)
    
    def evaluate_model(self, model_name: str, returns: np.ndarray, 
                      inference_time_ms: float = 10.0) -> ModelPerformance:
        """
        Evaluate a model and record performance.
        """
        # Compute metrics
        sharpe = self.reward_mechanism.compute_sharpe_reward(returns)
        win_rate = self.reward_mechanism.compute_win_rate_reward(returns)
        stability = self.reward_mechanism.compute_stability_reward(returns)
        max_drawdown = self._compute_max_drawdown(returns)
        total_return = np.sum(returns) if len(returns) > 0 else 0.0
        
        performance = ModelPerformance(
            model_name=model_name,
            model_type=self.registry.models[model_name]['type'],
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            stability_score=stability,
            training_time=0.0,  # Placeholder
            inference_time=inference_time_ms,
            prediction_accuracy=win_rate,  # Fallback
        )
        
        self.registry.record_performance(model_name, performance)
        
        # Update adaptive selector
        reward = self.reward_mechanism.compute_total_reward(returns, inference_time_ms)
        self.model_selector.update_model_reward(model_name, reward)
        
        return performance
    
    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)
    
    def generate_new_models(self) -> List[Dict[str, Any]]:
        """Generate new model configurations via evolution."""
        return self.evolution_engine.evolve_generation()
    
    def get_ensemble_weights(self, top_k: int = 5) -> Dict[str, float]:
        """
        Get ensemble weights based on model performance.
        Returns dictionary of {model_name: weight}
        """
        best_models = self.registry.get_best_models(top_k=top_k)
        
        weights = {}
        total_fitness = 0.0
        
        for model_name in best_models:
            perf_list = self.registry.performance_history.get(model_name, [])
            if len(perf_list) > 0:
                fitness = perf_list[-1].compute_fitness()
                weights[model_name] = fitness
                total_fitness += fitness
        
        # Normalize
        if total_fitness > 0:
            for model_name in weights:
                weights[model_name] /= total_fitness
        
        return weights
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances."""
        summary = {
            'total_models': len(self.registry.models),
            'models': {}
        }
        
        for model_name, perf_list in self.registry.performance_history.items():
            if len(perf_list) > 0:
                latest_perf = perf_list[-1]
                summary['models'][model_name] = latest_perf.to_dict()
        
        return summary
