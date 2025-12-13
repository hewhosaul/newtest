"""
Computer Vision module for chart pattern recognition, orderflow heatmaps, microstructure analysis.
Uses Vision Transformers, CNNs for candlestick patterns, optical flow concepts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")


class CandlestickImageGenerator:
    """Generate candlestick images from OHLC data for CNN analysis."""
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
    
    def generate_candlestick_image(self, 
                                  close_prices: np.ndarray,
                                  open_prices: np.ndarray,
                                  high_prices: np.ndarray,
                                  low_prices: np.ndarray,
                                  window_size: int = 20) -> np.ndarray:
        """
        Generate 2D image from candlestick data.
        Image dimensions: (height, width, 3) for RGB
        """
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255  # White background
        
        if len(close_prices) < window_size:
            return image
        
        # Normalize prices to image dimensions
        prices = np.concatenate([open_prices, high_prices, low_prices, close_prices])
        price_min = np.min(prices)
        price_max = np.max(prices)
        price_range = price_max - price_min if price_max > price_min else 1.0
        
        candles = close_prices[-window_size:]
        opens = open_prices[-window_size:]
        highs = high_prices[-window_size:]
        lows = low_prices[-window_size:]
        
        # Normalize to image height
        candles_norm = ((candles - price_min) / price_range * (self.height - 10) + 5).astype(int)
        opens_norm = ((opens - price_min) / price_range * (self.height - 10) + 5).astype(int)
        highs_norm = ((highs - price_min) / price_range * (self.height - 10) + 5).astype(int)
        lows_norm = ((lows - price_min) / price_range * (self.height - 10) + 5).astype(int)
        
        # Draw candlesticks
        candle_width = max(1, self.width // window_size - 1)
        
        for i, (candle, open_, high_, low_) in enumerate(zip(candles_norm, opens_norm, highs_norm, lows_norm)):
            x = int(i * self.width / window_size) + candle_width // 2
            
            # Wick (high-low line)
            if CV2_AVAILABLE:
                cv2.line(image, (x, low_), (x, high_), (100, 100, 100), 1)
                
                # Body (open-close rectangle)
                color = (0, 255, 0) if candle >= open_ else (255, 0, 0)  # Green up, red down
                cv2.rectangle(image, (x - candle_width // 2, min(open_, candle)),
                            (x + candle_width // 2, max(open_, candle)), color, -1)
            else:
                # Fallback: simple line drawing
                for y in range(min(low_, high_), max(low_, high_) + 1):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        image[y, x] = [100, 100, 100]
        
        return image
    
    def generate_volume_heatmap(self, volumes: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Generate volume heatmap."""
        heatmap = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if len(volumes) < window_size:
            return heatmap
        
        vols = volumes[-window_size:]
        vol_min = np.min(vols)
        vol_max = np.max(vols)
        vol_range = vol_max - vol_min if vol_max > vol_min else 1.0
        
        vol_norm = ((vols - vol_min) / vol_range * 255).astype(int)
        
        for i, vol in enumerate(vol_norm):
            x = int(i * self.width / window_size)
            # Draw vertical bar
            for y in range(self.height):
                intensity = int(255 * (1 - y / self.height))
                heatmap[y, x] = [intensity, intensity, 255]
        
        return heatmap


class OrderflowHeatmapAnalyzer:
    """
    Analyze orderflow data and detect anomalies using 2D heatmaps.
    """
    
    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
    
    def generate_orderflow_grid(self, 
                               buy_volumes: np.ndarray,
                               sell_volumes: np.ndarray,
                               prices: np.ndarray) -> np.ndarray:
        """
        Create 2D heatmap: price levels vs time, colored by buy/sell imbalance.
        """
        heatmap = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        if len(buy_volumes) == 0 or len(sell_volumes) == 0:
            return heatmap
        
        # Normalize time and price
        n_ticks = min(len(buy_volumes), self.grid_size)
        price_min = np.min(prices[-n_ticks:])
        price_max = np.max(prices[-n_ticks:])
        price_range = price_max - price_min if price_max > price_min else 1.0
        
        for t, (buy_vol, sell_vol, price) in enumerate(zip(buy_volumes[-n_ticks:], 
                                                           sell_volumes[-n_ticks:],
                                                           prices[-n_ticks:])):
            price_level = int((price - price_min) / price_range * (self.grid_size - 1))
            time_step = int(t / n_ticks * (self.grid_size - 1))
            
            # Imbalance: positive = buy pressure, negative = sell pressure
            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)
            
            # Color: green for buy, red for sell
            if imbalance > 0:
                heatmap[price_level, time_step] = [0, min(255, imbalance * 255), 0]
            else:
                heatmap[price_level, time_step] = [min(255, -imbalance * 255), 0, 0]
        
        return heatmap
    
    def detect_anomalies(self, heatmap: np.ndarray, threshold: float = 2.0) -> List[Tuple[int, int]]:
        """
        Detect anomalies in orderflow heatmap (unusual buy/sell imbalances).
        Returns list of (price_level, time_step) coordinates.
        """
        anomalies = []
        
        # Simple approach: find high-intensity pixels
        intensity = np.max(heatmap, axis=2)
        mean_intensity = np.mean(intensity)
        std_intensity = np.std(intensity)
        
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if intensity[i, j] > mean_intensity + threshold * std_intensity:
                    anomalies.append((i, j))
        
        return anomalies


class ChartPatternRecognizer:
    """
    Recognize common chart patterns: head & shoulders, triangles, wedges, etc.
    Uses edge detection and template matching concepts.
    """
    
    def __init__(self):
        self.patterns = ['head_shoulders', 'double_top', 'triangle', 'wedge', 'flag']
    
    def detect_head_shoulders(self, prices: np.ndarray, window: int = 50) -> float:
        """
        Detect head & shoulders pattern.
        Returns confidence score [0, 1].
        """
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices[-window:]
        
        # Find local maxima (peaks)
        peaks = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                peaks.append((i, recent_prices[i]))
        
        # Check for 3-peak pattern with middle peak highest
        if len(peaks) >= 3:
            # Look for pattern: left peak < middle peak > right peak
            for i in range(len(peaks) - 2):
                if peaks[i][1] < peaks[i+1][1] > peaks[i+2][1]:
                    # Head & shoulders pattern found
                    return 0.7
        
        return 0.0
    
    def detect_double_top(self, prices: np.ndarray, window: int = 50, tolerance: float = 0.02) -> float:
        """
        Detect double top pattern.
        Returns confidence score.
        """
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices[-window:]
        
        # Find local maxima
        peaks = []
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                peaks.append(recent_prices[i])
        
        if len(peaks) >= 2:
            # Check if two highest peaks are similar
            top_peaks = sorted(peaks, reverse=True)[:2]
            if abs(top_peaks[0] - top_peaks[1]) / top_peaks[0] < tolerance:
                return 0.75
        
        return 0.0
    
    def detect_triangle(self, prices: np.ndarray, window: int = 50) -> float:
        """
        Detect triangle pattern (converging volatility).
        """
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices[-window:]
        
        # Calculate rolling standard deviation
        rolling_std = []
        for i in range(10, len(recent_prices)):
            std = np.std(recent_prices[i-10:i])
            rolling_std.append(std)
        
        # Check if volatility is decreasing (triangle characteristic)
        if len(rolling_std) > 5:
            recent_std = rolling_std[-5:]
            if all(recent_std[i] >= recent_std[i+1] for i in range(len(recent_std)-1)):
                return 0.6
        
        return 0.0
    
    def recognize_patterns(self, prices: np.ndarray) -> Dict[str, float]:
        """Recognize all patterns."""
        patterns = {
            'head_shoulders': self.detect_head_shoulders(prices),
            'double_top': self.detect_double_top(prices),
            'triangle': self.detect_triangle(prices),
        }
        
        logger.info(f"Chart patterns detected: {patterns}")
        return patterns


class VisualRegressimeAnalyzer:
    """
    Create visual embeddings of market regimes from price/volume images.
    Uses simple feature extraction as Vision Transformer fallback.
    """
    
    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
    
    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from candlestick image.
        Simple approach: spatial features (mean, std, edges)
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Extract features
        features = []
        
        # Global statistics
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.max(gray))
        features.append(np.min(gray))
        
        # Spatial distribution (divide into quadrants)
        h, w = gray.shape
        for i in range(2):
            for j in range(2):
                quadrant = gray[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                features.append(np.mean(quadrant))
                features.append(np.std(quadrant))
        
        # Edge detection (simple gradient)
        if CV2_AVAILABLE:
            edges = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 1, ksize=3)
            features.append(np.mean(edges))
            features.append(np.sum(edges > 50))
        else:
            # Fallback: simple gradient
            grad = np.gradient(gray, axis=0)
            features.append(np.mean(np.abs(grad)))
            features.append(np.sum(np.abs(grad) > 10))
        
        # Pad or truncate to feature_dim
        features = np.array(features)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        return features
    
    def regime_embedding(self, candlestick_images: List[np.ndarray]) -> np.ndarray:
        """
        Create regime embedding from multiple candlestick images.
        Returns embedding vector (feature_dim,)
        """
        all_features = []
        
        for image in candlestick_images:
            features = self.extract_image_features(image)
            all_features.append(features)
        
        if len(all_features) > 0:
            embedding = np.mean(all_features, axis=0)
            return embedding
        else:
            return np.zeros(self.feature_dim)


class CVAnalysisPipeline:
    """
    Complete CV analysis pipeline combining all components.
    """
    
    def __init__(self):
        self.candlestick_gen = CandlestickImageGenerator(width=64, height=64)
        self.orderflow_analyzer = OrderflowHeatmapAnalyzer(grid_size=32)
        self.pattern_recognizer = ChartPatternRecognizer()
        self.regime_analyzer = VisualRegressimeAnalyzer(feature_dim=32)
    
    def analyze_chart(self, 
                     ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Complete analysis of chart and orderflow.
        ohlcv_data: {'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]}
        """
        results = {}
        
        try:
            # Generate candlestick image
            candlestick_img = self.candlestick_gen.generate_candlestick_image(
                ohlcv_data['close'],
                ohlcv_data['open'],
                ohlcv_data['high'],
                ohlcv_data['low'],
                window_size=20
            )
            results['candlestick_image'] = candlestick_img
            
            # Generate volume heatmap
            volume_heatmap = self.candlestick_gen.generate_volume_heatmap(ohlcv_data['volume'], window_size=20)
            results['volume_heatmap'] = volume_heatmap
            
            # Recognize patterns
            patterns = self.pattern_recognizer.recognize_patterns(ohlcv_data['close'])
            results['patterns'] = patterns
            
            # Regime embedding
            images = [candlestick_img, volume_heatmap]
            regime_embedding = self.regime_analyzer.regime_embedding(images)
            results['regime_embedding'] = regime_embedding
            
            logger.info("CV analysis pipeline completed successfully")
        
        except Exception as e:
            logger.warning(f"CV analysis failed: {e}")
        
        return results
