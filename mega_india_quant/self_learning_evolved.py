"""
Self-Learning & Evolution Engine: Automated model mutation, fitness scoring, model selection.
Uses genetic algorithm + reinforcement learning concepts for continuous improvement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
import logging
from dataclasses import dataclass
from copy import deepcopy
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Model:
    """Model genome with parameters and fitness."""
    name: str
    model_type: str  # 'factor', 'macro', 'ml', 'cv', 'hmm', 'intraday', 'supplychain'
    parameters: Dict[str, float]
    fitness_score: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    stability: float = 0.0
    max_drawdown: float = 0.0
    trades: int = 0
    generation: int = 0


class ModelGenome:
    """Defines hyperparameter ranges for each model type."""
    
    GENOMES = {
        'factor': {
            'ff_weight': (0.1, 0.9),
            'momentum_lookback': (10, 252),
            'momentum_weight': (0.0, 0.5),
            'value_weight': (0.0, 0.5),
            'quality_weight': (0.0, 0.5),
        },
        'macro': {
            'macro_lookback': (5, 60),
            'regime_smoothing': (0.1, 0.9),
            'yield_curve_weight': (0.0, 1.0),
            'inflation_weight': (0.0, 1.0),
        },
        'ml': {
            'tree_depth': (3, 15),
            'n_estimators': (10, 500),
            'learning_rate': (0.001, 0.3),
            'regularization': (0.0, 0.5),
        },
        'intraday': {
            'ofi_weight': (0.0, 1.0),
            'microprice_weight': (0.0, 1.0),
            'hawkes_sensitivity': (0.1, 0.9),
            'queue_depth_threshold': (100, 10000),
        },
        'supplychain': {
            'tsmc_exposure': (0.0, 1.0),
            'commodity_sensitivity': (0.0, 1.0),
            'inflation_passthrough': (0.1, 0.9),
        },
    }


class GeneticAlgorithm:
    """Genetic algorithm for model evolution."""
    
    def __init__(self, population_size: int = 50, elite_fraction: float = 0.2):
        self.population: List[Model] = []
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.generation = 0
        self.fitness_history = []
    
    def initialize_population(self, model_types: List[str]) -> List[Model]:
        """Create initial population with random hyperparameters."""
        self.population = []
        
        for i in range(self.population_size):
            model_type = random.choice(model_types)
            genome = ModelGenome.GENOMES.get(model_type, {})
            
            parameters = {}
            for param, (min_val, max_val) in genome.items():
                if isinstance(min_val, int):
                    parameters[param] = random.randint(int(min_val), int(max_val))
                else:
                    parameters[param] = random.uniform(min_val, max_val)
            
            model = Model(
                name=f"{model_type}_gen0_{i}",
                model_type=model_type,
                parameters=parameters,
                generation=0
            )
            
            self.population.append(model)
        
        logger.info(f"Initialized population of {self.population_size} models")
        return self.population
    
    def evaluate_fitness(self, model: Model,
                        sharpe: float,
                        win_rate: float,
                        max_dd: float,
                        num_trades: int) -> float:
        """
        Compute fitness score combining multiple metrics.
        Fitness = 0.4 * Sharpe + 0.3 * Win_Rate + 0.2 * (1 - |Max_DD|) + 0.1 * Trade_Efficiency
        """
        
        # Normalize metrics to [0, 1]
        sharpe_score = np.clip(sharpe / 2.0, 0, 1)  # Sharpe target: 2.0
        win_rate_score = np.clip(win_rate, 0, 1)
        dd_score = 1 - np.clip(abs(max_dd), 0, 1)  # Lower is better
        
        # Trade efficiency: prefer models with good metrics from reasonable trade count
        trade_efficiency = 1.0 if 20 < num_trades < 200 else 0.7
        
        # Composite fitness
        fitness = (0.4 * sharpe_score +
                  0.3 * win_rate_score +
                  0.2 * dd_score +
                  0.1 * trade_efficiency)
        
        # Stability bonus: higher fitness if consistent
        model.sharpe = sharpe
        model.win_rate = win_rate
        model.max_drawdown = max_dd
        model.trades = num_trades
        model.fitness_score = fitness
        
        return fitness
    
    def select_parents(self, k: int = 2) -> Tuple[Model, Model]:
        """Tournament selection: select k best from random sample."""
        sample = random.sample(self.population, min(k + 5, len(self.population)))
        sample.sort(key=lambda m: m.fitness_score, reverse=True)
        return sample[0], sample[1]
    
    def mutate(self, model: Model, mutation_rate: float = 0.3) -> Model:
        """Mutate model parameters."""
        mutated = deepcopy(model)
        genome = ModelGenome.GENOMES.get(model.model_type, {})
        
        for param in mutated.parameters:
            if random.random() < mutation_rate and param in genome:
                min_val, max_val = genome[param]
                
                # 80% small mutation, 20% large mutation
                if random.random() < 0.8:
                    # Small mutation: ±10%
                    current = mutated.parameters[param]
                    delta = (max_val - min_val) * 0.1
                    mutated.parameters[param] = np.clip(
                        current + random.uniform(-delta, delta),
                        min_val, max_val
                    )
                else:
                    # Large mutation: random
                    if isinstance(min_val, int):
                        mutated.parameters[param] = random.randint(int(min_val), int(max_val))
                    else:
                        mutated.parameters[param] = random.uniform(min_val, max_val)
        
        return mutated
    
    def crossover(self, parent1: Model, parent2: Model) -> Model:
        """Crossover: combine best genes from both parents."""
        child = deepcopy(parent1)
        child.name = f"{parent1.model_type}_gen{self.generation}_cross"
        
        for param in child.parameters:
            if random.random() < 0.5:
                child.parameters[param] = parent2.parameters.get(param,
                                                                parent1.parameters[param])
        
        return child
    
    def evolve_generation(self) -> List[Model]:
        """Evolve population for one generation."""
        self.generation += 1
        
        # Sort by fitness
        self.population.sort(key=lambda m: m.fitness_score, reverse=True)
        
        # Keep elite
        elite_count = max(2, int(self.population_size * self.elite_fraction))
        elite = self.population[:elite_count]
        
        # Create new generation
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Selection
            parent1, parent2 = self.select_parents()
            
            # Crossover
            if random.random() < 0.7:
                child = self.crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            child = self.mutate(child)
            child.generation = self.generation
            child.name = f"{child.model_type}_gen{self.generation}_{len(new_population)}"
            child.fitness_score = 0  # Reset until evaluated
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        
        # Track best fitness
        best_fitness = max([m.fitness_score for m in self.population])
        self.fitness_history.append(best_fitness)
        
        logger.info(f"Generation {self.generation}: Best fitness = {best_fitness:.4f}")
        
        return self.population
    
    def get_best_models(self, top_k: int = 5) -> List[Model]:
        """Get top K models."""
        sorted_pop = sorted(self.population, key=lambda m: m.fitness_score, reverse=True)
        return sorted_pop[:min(top_k, len(sorted_pop))]


class ReinforcementLearningEngine:
    """Reinforcement learning: reward successful model combinations."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # Q(model_combo) → expected return
        self.reward_history = []
    
    def compute_reward(self,
                      pnl: float,
                      num_trades: int,
                      max_dd: float,
                      sharpe: float) -> float:
        """Compute reward signal."""
        
        # Reward = P&L adjusted for risk and trade efficiency
        pnl_reward = np.sign(pnl) * np.log1p(abs(pnl) / 10000)  # Log scale
        
        # Penalty for drawdown
        dd_penalty = max_dd * 10
        
        # Bonus for efficiency (Sharpe > 1)
        sharpe_bonus = max(0, sharpe - 1.0)
        
        # Trade count penalty (too many = overfitting)
        trade_penalty = np.log1p(abs(num_trades - 50) / 50) if num_trades != 50 else 0
        
        reward = pnl_reward - dd_penalty + sharpe_bonus * 0.5 - trade_penalty
        
        return reward
    
    def update_q_value(self, model_combo: str, reward: float, next_value: float = 0):
        """Update Q-value: Q(s) ← Q(s) + α[r + γQ(s') - Q(s)]"""
        
        current_q = self.q_values.get(model_combo, 0)
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_value - current_q)
        
        self.q_values[model_combo] = new_q
        self.reward_history.append(new_q)
        
        return new_q
    
    def get_best_combo(self) -> str:
        """Get model combination with highest Q-value."""
        if not self.q_values:
            return 'default'
        
        return max(self.q_values.items(), key=lambda x: x[1])[0]


class SelfLearningPipeline:
    """Complete self-learning system orchestration."""
    
    def __init__(self):
        self.ga = GeneticAlgorithm(population_size=50)
        self.rl = ReinforcementLearningEngine()
        self.best_models_history = []
    
    def train_generation(self,
                         model_performance_dict: Dict[str, Dict[str, float]],
                         model_types: List[str]) -> List[Model]:
        """
        Train one generation:
        1. Evaluate current population
        2. Compute fitness
        3. Evolve to next generation
        """
        
        # Evaluate models
        for model in self.ga.population:
            perf = model_performance_dict.get(model.name, {})
            
            sharpe = perf.get('sharpe', 0.5)
            win_rate = perf.get('win_rate', 0.5)
            max_dd = perf.get('max_dd', -0.2)
            num_trades = perf.get('num_trades', 30)
            pnl = perf.get('pnl', 0)
            
            # Evaluate fitness
            self.ga.evaluate_fitness(model, sharpe, win_rate, max_dd, num_trades)
            
            # Update RL engine
            reward = self.rl.compute_reward(pnl, num_trades, max_dd, sharpe)
            self.rl.update_q_value(model.name, reward)
        
        # Evolve next generation
        next_population = self.ga.evolve_generation()
        
        # Track best
        best = self.ga.get_best_models(1)[0]
        self.best_models_history.append({
            'generation': self.ga.generation,
            'model': best.name,
            'fitness': best.fitness_score,
            'sharpe': best.sharpe,
            'win_rate': best.win_rate,
        })
        
        logger.info(f"Best model: {best.name} (Fitness: {best.fitness_score:.4f})")
        
        return next_population
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get model weights for ensemble based on fitness."""
        best_models = self.ga.get_best_models(top_k=10)
        
        total_fitness = sum([m.fitness_score for m in best_models])
        
        weights = {}
        for model in best_models:
            weight = model.fitness_score / total_fitness if total_fitness > 0 else 1/len(best_models)
            weights[model.name] = weight
        
        return weights
