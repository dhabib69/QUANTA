"""
QUANTA v11: SHAP Explainability Module

Provides TreeExplainer-based feature importance for CatBoost specialist predictions.
Based on: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" (NeurIPS)

Usage:
    explainer = SHAPExplainer(catboost_model, feature_names)
    explanation = explainer.explain(features_row)  
    # → "RSI_5m oversold (-0.45), Volume_spike (+0.38), BTC_corr (-0.22)"
"""

import numpy as np
import logging
import time

from quanta_config import Config as _Cfg
_EXPL = _Cfg.explainer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Run: pip install shap")


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for CatBoost specialist models.
    
    Features:
    - Uses shap.TreeExplainer (exact, not approximate) for CatBoost
    - Caches explainer objects per model to avoid re-initialization
    - Returns human-readable top-N feature importance strings
    - Thread-safe: each call creates independent SHAP values
    
    Parameters:
        top_n (int): Number of top features to show (default 3)
    """
    
    def __init__(self, top_n=None):
        self.top_n = top_n if top_n is not None else _EXPL.top_n
        self._explainer_cache = {}  # model_id -> shap.TreeExplainer
        self._last_init_time = {}
        self._feature_names = None
    
    def _get_explainer(self, model, model_key="default"):
        """Get or create cached TreeExplainer for a CatBoost model."""
        if not SHAP_AVAILABLE:
            return None
        
        if model_key not in self._explainer_cache:
            try:
                self._explainer_cache[model_key] = shap.TreeExplainer(model)
                self._last_init_time[model_key] = time.time()
                logging.info(f"SHAP TreeExplainer initialized for {model_key}")
            except Exception as e:
                logging.error(f"SHAP init failed for {model_key}: {e}")
                return None
        
        return self._explainer_cache[model_key]
    
    def _get_feature_names(self, model, n_features):
        """Extract feature names from CatBoost model or generate defaults."""
        try:
            names = model.feature_names_
            if names and len(names) == n_features:
                return names
        except (AttributeError, TypeError):
            pass
        
        # Auto-generate readable feature names based on QUANTA's feature vector layout
        names = []
        # Core timeframe features (per timeframe: ~30 features each)
        tf_labels = ['5m', '15m', '30m', '1h', '4h', '1d', '1w']
        per_tf_features = [
            'close', 'rsi', 'macd', 'macd_sig', 'macd_hist',
            'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_pct',
            'atr', 'ema_9', 'ema_21', 'ema_50', 'sma_200',
            'adx', 'obv', 'mfi', 'stoch_k', 'stoch_d',
            'vwap', 'volume_ratio', 'return_1', 'return_3', 'return_7',
            'volatility', 'momentum', 'roc', 'cci', 'willr'
        ]
        
        for tf in tf_labels:
            for feat in per_tf_features:
                names.append(f"{feat}_{tf}")
                if len(names) >= n_features:
                    break
            if len(names) >= n_features:
                break
        
        # Fill remaining with generic names
        while len(names) < n_features:
            idx = len(names)
            # Known special feature slots
            special = {
                210: 'btc_correlation', 211: 'transfer_entropy',
                212: 'funding_rate', 213: 'open_interest',
                214: 'oi_change', 215: 'long_short_ratio',
                216: 'lob_imbalance', 217: 'lob_depth_ratio',
                218: 'frac_diff', 219: 'sentiment_score',
                220: 'sentiment_magnitude', 221: 'fear_greed',
                222: 'meta_label', 223: 'odin_lstm_prob'
            }
            names.append(special.get(idx, f'feat_{idx}'))
        
        return names[:n_features]
    
    def explain(self, model, features, model_key="default", class_idx=1):
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            model: CatBoost model
            features: Feature array (1D or 2D with single row)
            model_key: Cache key for the explainer
            class_idx: Class to explain (1=bullish for binary)
            
        Returns:
            dict with:
                'summary': Human-readable string "RSI_5m (-0.45), Volume (+0.38), ..."
                'top_features': List of (name, shap_value) tuples
                'shap_values': Raw SHAP values array (or None)
        """
        if not SHAP_AVAILABLE:
            return {'summary': 'SHAP unavailable', 'top_features': [], 'shap_values': None}
        
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        n_features = features.shape[1]
        feature_names = self._get_feature_names(model, n_features)
        
        explainer = self._get_explainer(model, model_key)
        if explainer is None:
            return {'summary': 'SHAP init failed', 'top_features': [], 'shap_values': None}
        
        try:
            shap_values = explainer.shap_values(features)
            
            # For binary classification, shap_values is [class_0_vals, class_1_vals]
            # Each is shape (n_samples, n_features)
            if isinstance(shap_values, list) and len(shap_values) > class_idx:
                sv = shap_values[class_idx][0]  # First (only) sample, target class
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    sv = shap_values[0, :, class_idx]
                else:
                    sv = shap_values[0]
            else:
                sv = np.zeros(n_features)
            
            # Get top N by absolute value
            abs_sv = np.abs(sv)
            top_indices = np.argsort(abs_sv)[::-1][:self.top_n]
            
            top_features = []
            parts = []
            for idx in top_indices:
                name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                val = float(sv[idx])
                top_features.append((name, val))
                sign = '+' if val > 0 else ''
                parts.append(f"{name} ({sign}{val:.3f})")
            
            summary = ', '.join(parts)
            
            return {
                'summary': summary,
                'top_features': top_features,
                'shap_values': sv
            }
            
        except Exception as e:
            logging.error(f"SHAP explain error: {e}")
            return {'summary': f'SHAP error: {str(e)[:50]}', 'top_features': [], 'shap_values': None}
    
    def explain_batch(self, model, features_batch, model_key="default", class_idx=1):
        """
        Generate SHAP explanations for a batch of predictions.
        More efficient than calling explain() in a loop.
        
        Returns list of explanation dicts.
        """
        if not SHAP_AVAILABLE:
            return [{'summary': 'SHAP unavailable', 'top_features': [], 'shap_values': None}] * len(features_batch)
        
        features_batch = np.asarray(features_batch)
        if features_batch.ndim == 1:
            features_batch = features_batch.reshape(1, -1)
        
        n_features = features_batch.shape[1]
        feature_names = self._get_feature_names(model, n_features)
        
        explainer = self._get_explainer(model, model_key)
        if explainer is None:
            return [{'summary': 'SHAP init failed', 'top_features': [], 'shap_values': None}] * len(features_batch)
        
        try:
            shap_values = explainer.shap_values(features_batch)
            
            results = []
            for i in range(len(features_batch)):
                if isinstance(shap_values, list) and len(shap_values) > class_idx:
                    sv = shap_values[class_idx][i]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        sv = shap_values[i, :, class_idx]
                    else:
                        sv = shap_values[i]
                else:
                    sv = np.zeros(n_features)
                
                abs_sv = np.abs(sv)
                top_indices = np.argsort(abs_sv)[::-1][:self.top_n]
                
                top_features = []
                parts = []
                for idx in top_indices:
                    name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                    val = float(sv[idx])
                    top_features.append((name, val))
                    sign = '+' if val > 0 else ''
                    parts.append(f"{name} ({sign}{val:.3f})")
                
                results.append({
                    'summary': ', '.join(parts),
                    'top_features': top_features,
                    'shap_values': sv
                })
            
            return results
            
        except Exception as e:
            logging.error(f"SHAP batch explain error: {e}")
            return [{'summary': f'SHAP error', 'top_features': [], 'shap_values': None}] * len(features_batch)
    
    def invalidate_cache(self, model_key=None):
        """Clear cached explainers (call after retraining)."""
        if model_key:
            self._explainer_cache.pop(model_key, None)
            self._last_init_time.pop(model_key, None)
        else:
            self._explainer_cache.clear()
            self._last_init_time.clear()
