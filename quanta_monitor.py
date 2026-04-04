"""
QUANTA v11: Real-Time Model Monitoring

Tracks rolling prediction accuracy, calibration error, and feature distribution shifts.
Alerts via Telegram when accuracy drops below threshold or calibration error is high.

Based on:
- HackerNoon (2024): "AI Observability: Detecting Silent Model Degradation"
- Bayram et al. (2023): "Concept Drift Detection Methods Survey" (MDPI Sensors)

Usage:
    monitor = ModelMonitor(telegram_send_fn=tg.send)
    monitor.log_prediction(predicted_class=1, predicted_prob=0.82, actual_class=1)
    monitor.log_features(feature_vector)
    stats = monitor.get_stats()
"""

import numpy as np
import logging
import time
import os
from collections import deque

from quanta_config import Config as _Cfg
_MON = _Cfg.monitor

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ModelMonitor:
    """
    Real-time model monitoring for QUANTA trading predictions.
    
    Tracks:
    1. Rolling 100-prediction accuracy
    2. Calibration error (|predicted_prob - actual_accuracy_at_that_prob|)
    3. Feature distribution drift (mean/std shifts)
    4. Prediction latency
    
    Parameters:
        window_size (int): Rolling window for accuracy tracking (default 100)
        accuracy_alert_threshold (float): Alert when accuracy drops below this (default 0.55)
        calibration_alert_threshold (float): Alert when calibration error exceeds this (default 0.15)
        alert_cooldown (int): Seconds between alerts to prevent spam (default 3600)
        metrics_dir (str): Directory to save metrics (default 'quanta_data')
        telegram_send_fn: Optional function to send Telegram alerts
    """
    
    def __init__(self, window_size=None, accuracy_alert_threshold=None,
                 calibration_alert_threshold=None, alert_cooldown=None,
                 metrics_dir='quanta_data', telegram_send_fn=None):
        self.window_size = window_size or _MON.window_size
        self.accuracy_alert_threshold = accuracy_alert_threshold if accuracy_alert_threshold is not None else _MON.accuracy_alert_threshold
        self.calibration_alert_threshold = calibration_alert_threshold if calibration_alert_threshold is not None else _MON.calibration_alert_threshold
        self.alert_cooldown = alert_cooldown or _MON.alert_cooldown
        self.metrics_dir = metrics_dir
        self.telegram_send = telegram_send_fn

        # Rolling prediction outcomes
        self._outcomes = deque(maxlen=self.window_size)
        self._predicted_probs = deque(maxlen=self.window_size)
        self._predicted_classes = deque(maxlen=self.window_size)
        self._actual_classes = deque(maxlen=self.window_size)

        # Feature distribution tracking
        self._feature_means_baseline = None
        self._feature_stds_baseline = None
        self._recent_features = deque(maxlen=_MON.feature_buffer_size)
        self._feature_drift_score = 0.0
        
        # Counters
        self._total_predictions = 0
        self._total_correct = 0
        self._last_alert_time = 0
        self._last_metrics_save = 0
        
        # Metrics log for persistence
        self._metrics_log = []
        
        os.makedirs(metrics_dir, exist_ok=True)
    
    def log_prediction(self, predicted_class, predicted_prob, actual_class):
        """
        Log a single prediction outcome.
        
        Args:
            predicted_class: 0 or 1 (BEARISH/BULLISH)
            predicted_prob: Probability of the predicted class
            actual_class: 0 or 1 (actual outcome)
        """
        self._predicted_probs.append(float(predicted_prob))
        self._predicted_classes.append(int(predicted_class))

        if actual_class is not None:
            correct = float(predicted_class == actual_class)
            self._outcomes.append(correct)
            self._actual_classes.append(int(actual_class))
            self._total_correct += int(correct)

        self._total_predictions += 1
        
        # Check for alerts periodically
        if self._total_predictions % _MON.alert_check_interval == 0 and len(self._outcomes) >= _MON.alert_min_outcomes:
            self._check_alerts()

        # Save metrics periodically
        if time.time() - self._last_metrics_save > _MON.metrics_save_interval:
            self._save_metrics()
    
    def log_features(self, features):
        """Log a feature vector for distribution drift tracking."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim > 1:
            features = features.flatten()
        
        self._recent_features.append(features)
        
        # Set baseline after enough observations
        if self._feature_means_baseline is None and len(self._recent_features) >= _MON.baseline_min_samples:
            all_feats = np.array(list(self._recent_features))
            self._feature_means_baseline = np.mean(all_feats, axis=0)
            self._feature_stds_baseline = np.std(all_feats, axis=0) + 1e-8

        # Calculate drift score periodically
        if self._feature_means_baseline is not None and len(self._recent_features) % _MON.drift_check_interval == 0:
            recent = np.array(list(self._recent_features)[-_MON.drift_recent_window:])
            recent_means = np.mean(recent, axis=0)
            
            # Z-score drift: how many stddevs have the means shifted?
            z_scores = np.abs((recent_means - self._feature_means_baseline) / self._feature_stds_baseline)
            self._feature_drift_score = float(np.mean(z_scores))
    
    @property
    def rolling_accuracy(self):
        """Current rolling accuracy over the window."""
        if not self._outcomes:
            return None
        return float(np.mean(self._outcomes))
    
    @property
    def calibration_error(self):
        """
        Expected Calibration Error (ECE).
        Bins predictions into 10 buckets and measures |avg_predicted_prob - avg_accuracy| per bin.
        """
        if len(self._actual_classes) < 20:
            return None

        # Only use entries where actual outcome is known (may be fewer than total predictions)
        n = len(self._actual_classes)
        actuals = np.array(self._actual_classes)
        predicted = np.array(list(self._predicted_classes)[-n:])
        probs = np.array(list(self._predicted_probs)[-n:])

        # Correct = (predicted == actual)
        corrects = (predicted == actuals).astype(float)
        
        # Bin predictions by probability
        n_bins = _MON.calibration_bins
        bin_edges = np.linspace(_MON.calibration_prob_min, _MON.calibration_prob_max, n_bins + 1)
        ece = 0.0
        total = len(probs)
        
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_confidence = np.mean(probs[mask])
                avg_accuracy = np.mean(corrects[mask])
                ece += (mask.sum() / total) * abs(avg_confidence - avg_accuracy)
        
        return float(ece)
    
    def get_stats(self):
        """Return comprehensive monitoring statistics."""
        stats = {
            'total_predictions': self._total_predictions,
            'rolling_accuracy': self.rolling_accuracy,
            'calibration_error': self.calibration_error,
            'feature_drift_score': self._feature_drift_score,
            'window_size': len(self._outcomes),
            'lifetime_accuracy': (self._total_correct / self._total_predictions * 100) if self._total_predictions > 0 else 0
        }
        return stats
    
    def _check_alerts(self):
        """Check if any monitoring thresholds are breached."""
        now = time.time()
        if now - self._last_alert_time < self.alert_cooldown:
            return
        
        alerts = []
        
        # 1. Accuracy alert
        acc = self.rolling_accuracy
        if acc is not None and acc < self.accuracy_alert_threshold:
            alerts.append(f"📉 Rolling accuracy: {acc:.1%} (threshold: {self.accuracy_alert_threshold:.0%})")
        
        # 2. Calibration alert
        ece = self.calibration_error
        if ece is not None and ece > self.calibration_alert_threshold:
            alerts.append(f"📊 Calibration error: {ece:.3f} (threshold: {self.calibration_alert_threshold:.2f})")
        
        # 3. Feature drift alert
        if self._feature_drift_score > _MON.drift_zscore_threshold:
            alerts.append(f"📐 Feature drift: z-score {self._feature_drift_score:.2f} (high)")
        
        if alerts:
            self._last_alert_time = now
            msg = "⚠️ *MODEL HEALTH ALERT*\n\n" + "\n".join(alerts)
            msg += f"\n\n_Based on last {len(self._outcomes)} predictions_"
            
            print(f"\n{msg}")
            
            if self.telegram_send:
                try:
                    self.telegram_send(msg)
                except Exception as e:
                    logging.error(f"ModelMonitor telegram alert failed: {e}")
    
    def _save_metrics(self):
        """Save metrics snapshot to feather file."""
        self._last_metrics_save = time.time()
        
        snapshot = {
            'timestamp': time.time(),
            'rolling_accuracy': self.rolling_accuracy or 0,
            'calibration_error': self.calibration_error or 0,
            'feature_drift': self._feature_drift_score,
            'total_predictions': self._total_predictions,
            'window_size': len(self._outcomes)
        }
        self._metrics_log.append(snapshot)
        
        # Save to feather every 50 snapshots
        if PANDAS_AVAILABLE and len(self._metrics_log) % _MON.metrics_batch_save == 0:
            try:
                df = pd.DataFrame(self._metrics_log)
                path = os.path.join(self.metrics_dir, 'model_metrics.feather')
                df.to_feather(path)
                logging.info(f"ModelMonitor: saved {len(self._metrics_log)} snapshots to {path}")
            except Exception as e:
                logging.error(f"ModelMonitor save error: {e}")
    
    def reset_baseline(self):
        """Reset feature drift baseline (call after retraining)."""
        self._feature_means_baseline = None
        self._feature_stds_baseline = None
        self._feature_drift_score = 0.0
        self._recent_features.clear()
