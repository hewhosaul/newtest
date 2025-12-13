"""
Fusion engine: combines macro factors, technical/microstructure features, 
ML/DL predictions, and CV embeddings into unified forecasts.
Uses multi-view learning and attention-based fusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureView:
    """Represents one view of the data (e.g., macro, technical, CV)."""
    
    def __init__(self, name: str, features: np.ndarray, importance: float = 1.0):
        self.name = name
        self.features = features  # Shape: (n_features,) or (n_time_periods, n_features)
        self.importance = importance  # View-level importance weight
    
    def normalize(self) -> None:
        """Normalize features to [0, 1]."""
        if self.features.ndim == 1:
            feature_min = np.min(self.features)
            feature_max = np.max(self.features)
            feature_range = feature_max - feature_min if feature_max > feature_min else 1.0
            self.features = (self.features - feature_min) / feature_range
        else:
            for i in range(self.features.shape[1]):
                col = self.features[:, i]
                col_min = np.min(col)
                col_max = np.max(col)
                col_range = col_max - col_min if col_max > col_min else 1.0
                self.features[:, i] = (col - col_min) / col_range


class AttentionFusionModule:
    """
    Attention-based fusion of multiple views.
    Learns to weight different data sources based on their relevance.
    """
    
    def __init__(self, n_views: int, hidden_dim: int = 32, n_heads: int = 4):
        self.n_views = n_views
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention_weights = None
        self.query_vectors = np.random.normal(0, 1, (n_heads, hidden_dim))
    
    def compute_attention_scores(self, views: List[FeatureView]) -> np.ndarray:
        """
        Compute attention scores for each view.
        Returns: (n_views,) array of normalized attention weights
        """
        if len(views) == 0:
            return np.array([])
        
        scores = []
        
        for i, view in enumerate(views):
            # Feature importance score
            feature_importance = np.mean(np.abs(view.features)) if view.features.size > 0 else 0.0
            
            # Variance as signal strength indicator
            feature_variance = np.var(view.features) if view.features.size > 0 else 0.0
            
            # Combined score
            score = (feature_importance * 0.6 + feature_variance * 0.4) * view.importance
            scores.append(score)
        
        scores = np.array(scores)
        
        # Normalize to sum to 1 (softmax-like)
        if np.sum(scores) > 0:
            scores = scores / np.sum(scores)
        else:
            scores = np.ones(len(views)) / len(views)
        
        self.attention_weights = scores
        return scores
    
    def fuse_views(self, views: List[FeatureView], fusion_type: str = 'weighted_mean') -> np.ndarray:
        """
        Fuse multiple views into single representation.
        
        fusion_type:
            - 'weighted_mean': attention-weighted average
            - 'concatenate': stack all features
            - 'gating': learned gating mechanism
        """
        if len(views) == 0:
            return np.array([])
        
        if fusion_type == 'weighted_mean':
            attention_scores = self.compute_attention_scores(views)
            
            # Flatten all views to 1D for averaging
            fused = np.zeros(1)  # Placeholder
            total_weight = 0.0
            
            for view, weight in zip(views, attention_scores):
                if view.features.size > 0:
                    feature_signal = np.mean(view.features)  # Aggregate features
                    fused = fused + weight * feature_signal if fused.size > 0 else weight * feature_signal
                    total_weight += weight
            
            return fused / total_weight if total_weight > 0 else np.array([0.0])
        
        elif fusion_type == 'concatenate':
            flat_views = []
            for view in views:
                if view.features.size > 0:
                    if view.features.ndim == 1:
                        flat_views.append(view.features)
                    else:
                        flat_views.append(view.features.flatten())
            
            if len(flat_views) > 0:
                return np.concatenate(flat_views)
            else:
                return np.array([])
        
        else:  # gating
            # Simple learned gating (approximation without explicit training)
            attention_scores = self.compute_attention_scores(views)
            
            gated_features = []
            for view, gate in zip(views, attention_scores):
                if view.features.size > 0:
                    gated = view.features * (gate ** 0.5)  # Apply gating
                    gated_features.append(gated)
            
            if len(gated_features) > 0:
                # Aggregate gated features
                agg_shape = max([f.size for f in gated_features])
                aggregated = np.zeros(agg_shape)
                
                for gf in gated_features:
                    if gf.size <= agg_shape:
                        aggregated[:gf.size] += gf
                
                return aggregated
            else:
                return np.array([])


class EarlyFusionModel:
    """
    Early fusion: combine raw features from all views before processing.
    """
    
    def __init__(self):
        self.fusion_matrix = None
        self.bias = None
    
    def fit(self, macro_features: np.ndarray, 
            technical_features: np.ndarray,
            ml_predictions: np.ndarray,
            targets: np.ndarray) -> None:
        """
        Fit linear fusion weights.
        """
        # Stack all features horizontally
        if macro_features.size > 0 and technical_features.size > 0:
            if macro_features.ndim == 1:
                macro_features = macro_features.reshape(-1, 1)
            if technical_features.ndim == 1:
                technical_features = technical_features.reshape(-1, 1)
            if ml_predictions.ndim == 1:
                ml_predictions = ml_predictions.reshape(-1, 1)
            
            all_features = np.hstack([macro_features, technical_features, ml_predictions])
            
            # Simple linear regression
            try:
                self.fusion_matrix = np.linalg.lstsq(all_features, targets, rcond=None)[0]
                self.bias = np.mean(targets) - np.mean(all_features @ self.fusion_matrix)
                logger.info("Early fusion model fitted")
            except Exception as e:
                logger.warning(f"Early fusion fitting failed: {e}")
    
    def predict(self, macro_features: np.ndarray,
               technical_features: np.ndarray,
               ml_predictions: np.ndarray) -> float:
        """Generate early-fused prediction."""
        if self.fusion_matrix is None:
            # Fallback: simple average
            valid_features = [f for f in [macro_features, technical_features, ml_predictions] 
                            if f is not None and np.isfinite(f)]
            return np.mean(valid_features) if len(valid_features) > 0 else 0.0
        
        try:
            if macro_features.ndim == 0:
                macro_features = np.array([macro_features])
            if technical_features.ndim == 0:
                technical_features = np.array([technical_features])
            if ml_predictions.ndim == 0:
                ml_predictions = np.array([ml_predictions])
            
            all_features = np.concatenate([macro_features, technical_features, ml_predictions])
            prediction = all_features @ self.fusion_matrix + self.bias
            return float(prediction)
        except Exception as e:
            logger.warning(f"Early fusion prediction failed: {e}")
            return 0.0


class LateFusionModel:
    """
    Late fusion: generate separate predictions from each view, then combine.
    More robust to individual view failures.
    """
    
    def __init__(self):
        self.view_weights = {}
        self.view_biases = {}
    
    def fit(self, view_predictions: Dict[str, np.ndarray],
           targets: np.ndarray) -> None:
        """
        Learn weights for each view's predictions.
        view_predictions: {view_name: predictions array}
        """
        for view_name, preds in view_predictions.items():
            if len(preds) > 0 and len(targets) > 0:
                # Simple correlation-based weighting
                if np.std(preds) > 0 and np.std(targets) > 0:
                    correlation = np.corrcoef(preds, targets)[0, 1]
                    self.view_weights[view_name] = max(0, correlation)
                else:
                    self.view_weights[view_name] = 0.5
                
                self.view_biases[view_name] = np.mean(targets) - np.mean(preds)
        
        # Normalize weights
        if len(self.view_weights) > 0:
            total_weight = sum(self.view_weights.values())
            if total_weight > 0:
                for view_name in self.view_weights:
                    self.view_weights[view_name] /= total_weight
        
        logger.info(f"Late fusion fitted with weights: {self.view_weights}")
    
    def predict(self, view_predictions: Dict[str, float]) -> float:
        """
        Combine predictions from multiple views.
        """
        if len(self.view_weights) == 0:
            # Fallback: simple average
            valid_preds = [p for p in view_predictions.values() if np.isfinite(p)]
            return np.mean(valid_preds) if len(valid_preds) > 0 else 0.0
        
        combined = 0.0
        total_weight = 0.0
        
        for view_name, weight in self.view_weights.items():
            if view_name in view_predictions:
                pred = view_predictions[view_name]
                bias = self.view_biases.get(view_name, 0.0)
                
                if np.isfinite(pred):
                    combined += weight * (pred + bias)
                    total_weight += weight
        
        if total_weight > 0:
            return combined / total_weight
        else:
            return 0.0


class HybridFusionEngine:
    """
    Hybrid fusion combining early and late fusion approaches.
    """
    
    def __init__(self):
        self.early_fusion = EarlyFusionModel()
        self.late_fusion = LateFusionModel()
        self.attention_fusion = AttentionFusionModule(n_views=4, hidden_dim=32)
        self.hybrid_weight_early = 0.5  # 50-50 blend initially
        self.hybrid_weight_late = 0.5
    
    def fit(self, macro_features: np.ndarray,
           technical_features: np.ndarray,
           ml_predictions: np.ndarray,
           view_predictions: Dict[str, np.ndarray],
           targets: np.ndarray) -> None:
        """Fit both fusion models."""
        # Fit early fusion
        self.early_fusion.fit(macro_features, technical_features, 
                             ml_predictions, targets)
        
        # Fit late fusion
        self.late_fusion.fit(view_predictions, targets)
        
        logger.info("Hybrid fusion engine fitted")
    
    def predict(self, macro_features: np.ndarray,
               technical_features: np.ndarray,
               ml_predictions: np.ndarray,
               view_predictions: Dict[str, float]) -> float:
        """
        Generate hybrid prediction.
        """
        # Early fusion prediction
        early_pred = self.early_fusion.predict(macro_features, 
                                              technical_features, 
                                              ml_predictions)
        
        # Late fusion prediction
        late_pred = self.late_fusion.predict(view_predictions)
        
        # Blend predictions
        hybrid_pred = self.hybrid_weight_early * early_pred + self.hybrid_weight_late * late_pred
        
        return hybrid_pred
    
    def adapt_weights(self, early_performance: float, late_performance: float) -> None:
        """
        Adapt fusion weights based on recent performance.
        Simulates online learning of optimal fusion strategy.
        """
        total_perf = early_performance + late_performance
        if total_perf > 0:
            self.hybrid_weight_early = early_performance / total_perf
            self.hybrid_weight_late = late_performance / total_perf
            
            logger.info(f"Adapted fusion weights: early={self.hybrid_weight_early:.3f}, "
                       f"late={self.hybrid_weight_late:.3f}")


class ModalityAlignmentLayer:
    """
    Align different modalities (numerical features, images, time series)
    into common representation space.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.embeddings = {}
    
    def embed_numerical(self, features: np.ndarray) -> np.ndarray:
        """Embed numerical features."""
        if features.size == 0:
            return np.zeros(self.embedding_dim)
        
        # Normalize and project
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Random projection to embedding space (fallback without NN)
        projection = np.random.RandomState(42).normal(0, 1, (features_normalized.shape[1], self.embedding_dim))
        embedding = features_normalized @ projection
        
        return np.mean(embedding, axis=0)  # Aggregate to embedding_dim
    
    def embed_image(self, image_features: np.ndarray) -> np.ndarray:
        """Embed image/CV features."""
        if image_features.size == 0:
            return np.zeros(self.embedding_dim)
        
        # Direct projection if already extracted features
        if image_features.shape[-1] <= self.embedding_dim:
            # Pad if necessary
            padded = np.zeros(self.embedding_dim)
            padded[:image_features.shape[-1]] = image_features.flatten()[:self.embedding_dim]
            return padded
        else:
            # Reduce dimensionality via averaging
            return np.average(image_features.reshape(-1, self.embedding_dim), axis=0)
    
    def embed_timeseries(self, timeseries: np.ndarray) -> np.ndarray:
        """Embed time series features."""
        if len(timeseries) == 0:
            return np.zeros(self.embedding_dim)
        
        # Statistical features
        stats_features = np.array([
            np.mean(timeseries),
            np.std(timeseries),
            np.min(timeseries),
            np.max(timeseries),
            np.percentile(timeseries, 25),
            np.percentile(timeseries, 75),
        ])
        
        # Pad to embedding dimension
        padded = np.zeros(self.embedding_dim)
        padded[:len(stats_features)] = stats_features
        
        return padded
    
    def align(self, numerical: np.ndarray = None, 
             images: np.ndarray = None,
             timeseries: np.ndarray = None) -> np.ndarray:
        """
        Align all modalities to common space.
        Returns concatenated embedding.
        """
        embeddings = []
        
        if numerical is not None and numerical.size > 0:
            embeddings.append(self.embed_numerical(numerical))
        
        if images is not None and images.size > 0:
            embeddings.append(self.embed_image(images))
        
        if timeseries is not None and timeseries.size > 0:
            embeddings.append(self.embed_timeseries(timeseries))
        
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)
        
        return np.concatenate(embeddings)


class CompleteFusionPipeline:
    """
    Complete fusion pipeline combining all components.
    """
    
    def __init__(self):
        self.attention_fusion = AttentionFusionModule(n_views=5)
        self.hybrid_fusion = HybridFusionEngine()
        self.modality_alignment = ModalityAlignmentLayer(embedding_dim=64)
    
    def fuse_all_signals(self, 
                        macro_signal: float = 0.5,
                        technical_signal: float = 0.5,
                        ml_signal: float = 0.5,
                        dl_signal: float = 0.5,
                        cv_signal: float = 0.5,
                        microstructure_signal: float = 0.5,
                        regime: int = 0) -> Dict[str, float]:
        """
        Main fusion function combining all signals.
        All signals should be normalized to [0, 1] or [-1, 1].
        """
        
        # Create feature views
        views = [
            FeatureView('macro', np.array([macro_signal]), importance=0.2),
            FeatureView('technical', np.array([technical_signal]), importance=0.15),
            FeatureView('ml', np.array([ml_signal]), importance=0.2),
            FeatureView('dl', np.array([dl_signal]), importance=0.2),
            FeatureView('cv', np.array([cv_signal]), importance=0.15),
            FeatureView('microstructure', np.array([microstructure_signal]), importance=0.1),
        ]
        
        # Normalize views
        for view in views:
            view.normalize()
        
        # Attention-based fusion
        attention_weights = self.attention_fusion.compute_attention_scores(views)
        attention_signal = self.attention_fusion.fuse_views(views, fusion_type='weighted_mean')
        
        # Hybrid fusion
        hybrid_signals = {
            'macro': macro_signal,
            'technical': technical_signal,
            'ml': ml_signal,
            'dl': dl_signal,
            'cv': cv_signal,
            'microstructure': microstructure_signal,
        }
        hybrid_signal = self.hybrid_fusion.predict(
            macro_signal, technical_signal, ml_signal, hybrid_signals
        )
        
        # Modality alignment
        numerical_features = np.array([macro_signal, technical_signal, ml_signal])
        aligned_embedding = self.modality_alignment.align(numerical=numerical_features)
        aligned_signal = np.mean(aligned_embedding)
        
        # Final ensemble
        final_signal = (
            0.4 * attention_signal +
            0.3 * hybrid_signal +
            0.3 * aligned_signal
        )
        
        return {
            'attention_signal': attention_signal,
            'hybrid_signal': hybrid_signal,
            'aligned_signal': aligned_signal,
            'final_signal': final_signal,
            'attention_weights': {f'weight_{i}': float(w) for i, w in enumerate(attention_weights)},
            'regime': regime,
        }
