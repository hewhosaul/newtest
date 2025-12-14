"""
Advanced econometric models: regime-switching VAR, particle filtering, Kalman filter, 
Bayesian models, copulas, change point detection.
References: Hamilton (1989), Kalman (1960), PELT (Killick et al., 2012), 
Change point detection literature.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeSwitchingVAR:
    """
    Markov-switching VAR model.
    Reference: Hamilton (1989), "A new approach to the economic analysis of nonstationary time series"
    """
    
    def __init__(self, n_regimes: int = 2, lag_order: int = 1):
        self.n_regimes = n_regimes
        self.lag_order = lag_order
        self.transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        self.regime_means = None
        self.regime_covs = None
        self.filtered_regimes = None
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        """
        Fit regime-switching VAR using EM algorithm (simplified).
        X: (n_observations, n_variables)
        """
        n_obs, n_vars = X.shape
        
        # Initialize regime assignments randomly
        regime_assignment = np.random.randint(0, self.n_regimes, n_obs)
        
        for iteration in range(max_iter):
            # E-step: compute regime probabilities
            regime_probs = np.zeros((n_obs, self.n_regimes))
            
            for regime in range(self.n_regimes):
                mask = regime_assignment == regime
                if np.sum(mask) > 0:
                    regime_data = X[mask]
                    mu = np.mean(regime_data, axis=0)
                    sigma = np.cov(regime_data.T)
                    
                    # Multivariate normal likelihood
                    diff = X - mu
                    try:
                        sigma_inv = np.linalg.inv(sigma)
                        log_det = np.linalg.slogdet(sigma)[1]
                        
                        mahal = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)
                        regime_probs[:, regime] = mahal - 0.5 * log_det
                    except np.linalg.LinAlgError:
                        regime_probs[:, regime] = -np.inf
            
            # Normalize using softmax
            regime_probs = softmax(regime_probs, axis=1)
            
            # M-step: update assignments
            new_assignment = np.argmax(regime_probs, axis=1)
            
            if np.array_equal(new_assignment, regime_assignment):
                break
            
            regime_assignment = new_assignment
        
        # Store regime parameters
        self.regime_means = []
        self.regime_covs = []
        
        for regime in range(self.n_regimes):
            mask = regime_assignment == regime
            if np.sum(mask) > 0:
                regime_data = X[mask]
                self.regime_means.append(np.mean(regime_data, axis=0))
                self.regime_covs.append(np.cov(regime_data.T))
            else:
                self.regime_means.append(np.zeros(n_vars))
                self.regime_covs.append(np.eye(n_vars))
        
        logger.info(f"Regime-switching VAR fitted with {self.n_regimes} regimes")
    
    def predict_regime(self, X: np.ndarray) -> int:
        """Predict current regime."""
        if self.regime_means is None:
            return 0
        
        best_regime = 0
        best_likelihood = -np.inf
        
        for regime in range(self.n_regimes):
            try:
                sigma_inv = np.linalg.inv(self.regime_covs[regime])
                diff = X[-1] - self.regime_means[regime]
                likelihood = -0.5 * diff @ sigma_inv @ diff
                
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_regime = regime
            except np.linalg.LinAlgError:
                continue
        
        return best_regime


class ParticleFilter:
    """
    Particle filter for sequential state estimation.
    Reference: Gordon et al. (1993), sequential importance resampling
    """
    
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
    
    def initialize_particles(self, initial_state: np.ndarray, initial_cov: np.ndarray) -> None:
        """Initialize particles from Gaussian distribution."""
        n_states = len(initial_state)
        self.particles = np.random.multivariate_normal(initial_state, initial_cov, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def update(self, observation: np.ndarray, 
               likelihood_fn, transition_fn) -> None:
        """
        Update particles using importance sampling and resampling.
        """
        # Prediction: apply state transition
        for i in range(self.n_particles):
            # Add process noise
            noise = np.random.multivariate_normal(np.zeros(self.particles.shape[1]), 
                                                 0.01 * np.eye(self.particles.shape[1]))
            self.particles[i] = transition_fn(self.particles[i]) + noise
        
        # Update: evaluate likelihood
        likelihoods = np.array([likelihood_fn(self.particles[i], observation) 
                               for i in range(self.n_particles)])
        
        # Update weights
        self.weights = self.weights * likelihoods
        self.weights /= np.sum(self.weights)
        
        # Resample if weights become too skewed
        effective_n = 1.0 / np.sum(self.weights ** 2)
        
        if effective_n < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, size=self.n_particles, 
                                      p=self.weights, replace=True)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_estimate(self) -> np.ndarray:
        """Get weighted average of particles."""
        return np.average(self.particles, axis=0, weights=self.weights)


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter for nonlinear state estimation.
    Reference: Evensen (1994)
    """
    
    def __init__(self, n_ensemble: int = 100, process_noise: float = 0.01, obs_noise: float = 0.1):
        self.n_ensemble = n_ensemble
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.ensemble = None
    
    def initialize(self, initial_state: np.ndarray, initial_spread: float = 0.1) -> None:
        """Initialize ensemble around initial state."""
        n_states = len(initial_state)
        perturbations = np.random.normal(0, initial_spread, (self.n_ensemble, n_states))
        self.ensemble = initial_state[np.newaxis, :] + perturbations
    
    def forecast_step(self, transition_fn) -> None:
        """Apply forecast model to each ensemble member."""
        for i in range(self.n_ensemble):
            self.ensemble[i] = transition_fn(self.ensemble[i])
            # Add model error
            self.ensemble[i] += np.random.normal(0, self.process_noise, self.ensemble[i].shape)
    
    def analysis_step(self, observation: np.ndarray, obs_operator) -> None:
        """Update ensemble with observations (EnKF update)."""
        # Get predicted observations
        predicted_obs = np.array([obs_operator(self.ensemble[i]) for i in range(self.n_ensemble)])
        
        # Ensemble mean and anomalies
        mean_obs = np.mean(predicted_obs, axis=0)
        obs_anomalies = predicted_obs - mean_obs
        
        # Kalman gain
        cov_obs_state = np.cov(obs_anomalies.T, self.ensemble.T - np.mean(self.ensemble, axis=0))
        cov_obs = np.cov(obs_anomalies.T) + self.obs_noise * np.eye(len(observation))
        
        try:
            K = cov_obs_state @ np.linalg.inv(cov_obs)
            
            # Update ensemble
            for i in range(self.n_ensemble):
                innovation = observation - predicted_obs[i] + np.random.normal(0, self.obs_noise, len(observation))
                self.ensemble[i] += K @ innovation
        except np.linalg.LinAlgError:
            logger.warning("EnKF update failed due to singular matrix")
    
    def get_state_estimate(self) -> np.ndarray:
        """Get ensemble mean as state estimate."""
        return np.mean(self.ensemble, axis=0)


class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for parameter estimation.
    Simple implementation with conjugate priors.
    """
    
    def __init__(self, n_groups: int = 5):
        self.n_groups = n_groups
        self.group_means = None
        self.group_precisions = None
        self.global_mean = None
        self.global_precision = None
    
    def fit(self, data: List[np.ndarray], n_iter: int = 100) -> None:
        """
        Fit hierarchical model.
        data: list of arrays, one per group
        """
        # Initialize
        self.group_means = np.array([np.mean(d) for d in data])
        self.group_precisions = np.array([1.0 / (np.var(d) + 1e-6) for d in data])
        self.global_mean = np.mean(self.group_means)
        self.global_precision = 1.0
        
        # Gibbs sampling loop
        for iteration in range(n_iter):
            # Sample global parameters
            self.global_mean = np.mean(self.group_means)
            self.global_precision = np.mean(self.group_precisions)
            
            # Update group parameters
            for i, d in enumerate(data):
                n = len(d)
                data_mean = np.mean(d)
                data_precision = n / (np.var(d) + 1e-6)
                
                # Posterior mean
                posterior_precision = self.global_precision + data_precision
                self.group_means[i] = (self.global_precision * self.global_mean + 
                                      data_precision * data_mean) / posterior_precision
        
        logger.info("Bayesian hierarchical model fitted")
    
    def predict_group(self, group_idx: int) -> float:
        """Predict for a group."""
        if self.group_means is None:
            return 0.0
        return self.group_means[min(group_idx, len(self.group_means) - 1)]


class GaussianCopula:
    """
    Gaussian copula for modeling joint dependence structure.
    Reference: Embrechts et al. (1999), copula methods
    """
    
    def __init__(self):
        self.correlation_matrix = None
    
    def fit(self, data: np.ndarray) -> None:
        """
        Fit Gaussian copula to data.
        data: (n_observations, n_variables)
        """
        # Transform to uniform via empirical CDF
        n_obs, n_vars = data.shape
        uniform_data = np.zeros_like(data)
        
        for j in range(n_vars):
            uniform_data[:, j] = stats.rankdata(data[:, j]) / (n_obs + 1)
        
        # Transform to standard normal (inverse Gaussian CDF)
        normal_data = stats.norm.ppf(uniform_data)
        
        # Estimate correlation matrix
        self.correlation_matrix = np.corrcoef(normal_data.T)
        
        logger.info("Gaussian copula fitted")
    
    def simulate(self, n_samples: int = 1000, n_vars: int = 2) -> np.ndarray:
        """Simulate from fitted copula."""
        if self.correlation_matrix is None:
            self.correlation_matrix = np.eye(n_vars)
        
        # Generate correlated normals
        normal_samples = np.random.multivariate_normal(
            np.zeros(n_vars),
            self.correlation_matrix,
            n_samples
        )
        
        # Transform back to uniform
        uniform_samples = stats.norm.cdf(normal_samples)
        
        return uniform_samples


class ChangePointDetector:
    """
    Change point detection using PELT (Pruned Exact Linear Time).
    Reference: Killick et al. (2012)
    """
    
    def __init__(self, penalty: str = 'bic', min_segment_length: int = 10):
        self.penalty = penalty
        self.min_segment_length = min_segment_length
    
    def fit(self, X: np.ndarray) -> List[int]:
        """
        Detect change points in time series.
        X: (n_observations,) or (n_observations, n_features)
        Returns: list of change point indices
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_obs = X.shape[0]
        
        # Cost function: sum of squared residuals
        def cost(segment):
            if len(segment) < self.min_segment_length:
                return np.inf
            mean = np.mean(segment, axis=0)
            return np.sum((segment - mean) ** 2)
        
        # Dynamic programming
        costs = np.zeros(n_obs + 1)
        changepoints = np.zeros(n_obs + 1, dtype=int)
        
        for t in range(1, n_obs + 1):
            cost_t = np.inf
            best_cp = 0
            
            for s in range(max(0, t - 100), t - self.min_segment_length + 1):
                segment = X[s:t]
                c = cost(segment)
                
                # Add penalty for new segment
                if self.penalty == 'bic':
                    penalty_term = np.log(t - s) * X.shape[1]
                else:
                    penalty_term = 0
                
                total_cost = costs[s] + c + penalty_term
                
                if total_cost < cost_t:
                    cost_t = total_cost
                    best_cp = s
            
            costs[t] = cost_t
            changepoints[t] = best_cp
        
        # Backtrack to find changepoints
        detected_cps = []
        t = n_obs
        while t > 0:
            s = changepoints[t]
            if s > 0:
                detected_cps.append(s)
            t = s
        
        detected_cps.reverse()
        
        logger.info(f"Detected {len(detected_cps)} change points")
        return detected_cps


class GrangerCausalityAnalyzer:
    """
    Granger causality for detecting lead-lag relationships.
    Reference: Granger (1969), causality in econometrics
    """
    
    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level
    
    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        Test if X Granger-causes Y.
        Returns: F-statistic, p-value, causality indicator
        """
        if len(X) != len(Y):
            raise ValueError("X and Y must have same length")
        
        n = len(X)
        
        # Model 1: Y depends only on its own lagged values
        y_lags = np.column_stack([np.roll(Y, i, axis=0) for i in range(1, self.max_lag + 1)])
        y_lags = y_lags[self.max_lag:]
        Y_trimmed = Y[self.max_lag:]
        
        try:
            from numpy.linalg import lstsq
            
            # Fit restricted model (Y only depends on itself)
            y_const = np.ones((len(Y_trimmed), 1))
            X_restricted = np.hstack([y_const, y_lags])
            beta_r = lstsq(X_restricted, Y_trimmed, rcond=None)[0]
            rss_r = np.sum((Y_trimmed - X_restricted @ beta_r) ** 2)
            
            # Fit unrestricted model (Y depends on X and Y)
            x_lags = np.column_stack([np.roll(X, i, axis=0) for i in range(1, self.max_lag + 1)])
            x_lags = x_lags[self.max_lag:]
            X_unrestricted = np.hstack([y_const, y_lags, x_lags])
            beta_u = lstsq(X_unrestricted, Y_trimmed, rcond=None)[0]
            rss_u = np.sum((Y_trimmed - X_unrestricted @ beta_u) ** 2)
            
            # F-statistic
            f_stat = ((rss_r - rss_u) / self.max_lag) / (rss_u / (len(Y_trimmed) - 2 * self.max_lag - 1))
            p_value = 1 - stats.f.cdf(f_stat, self.max_lag, len(Y_trimmed) - 2 * self.max_lag - 1)
            
            granger_causes = p_value < self.significance_level
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'granger_causes': granger_causes,
                'strength': 1.0 - p_value if p_value < 1.0 else 0.0
            }
        
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}")
            return {'f_statistic': 0.0, 'p_value': 1.0, 'granger_causes': False, 'strength': 0.0}
