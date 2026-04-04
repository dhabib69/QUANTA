"""
QUANTA Model Registry — Versioning, Lineage Tracking & Deploy Gate (v11.5b)

Every trained model gets a JSON sidecar ("birth certificate") that records:
  1. What features it was trained on (count, mask, indices)
  2. What data it saw (symbols, sample count, class balance, date range)
  3. How well it performed (val AUC, accuracy, Brier, Sharpe on OOS)
  4. Its lineage (parent generation, warm-start or fresh)

The Deploy Gate compares a newly trained model against the previous generation's
recorded metrics. Only deploys if the new model meets quality thresholds.
If 3 consecutive generations show declining OOS performance, forces a fresh
retrain (no warm-start) to escape catastrophic forgetting.

References:
    - Pardo (2008) "Design, Testing, Optimization of Trading Systems" — walk-forward validation
    - López de Prado (2018) AFML Ch.12 — backtesting through cross-validation
    - Goodfellow et al. (2013) — catastrophic forgetting in neural networks (applies to ensemble drift)
"""

import os
import json
import time
import hashlib
import logging
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelMetadata:
    """Birth certificate for a trained model generation."""
    # Identity
    agent_name: str = ""
    generation: int = 0
    parent_generation: int = 0          # Which gen was warm-started from (0 = fresh)
    timestamp: str = ""
    training_mode: str = "warm_start"   # "warm_start" or "fresh"

    # Feature configuration
    feature_count: int = 268
    feature_mask: str = ""              # e.g. "domain_trend"
    feature_indices_hash: str = ""      # SHA256 of the actual index array

    # Training data
    training_symbols: list = field(default_factory=list)
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    class_ratio: float = 1.0
    cross_event_samples: int = 0        # v11.5 cross-event negatives added

    # Validation metrics (computed on held-out val split during training)
    val_auc: float = 0.0
    val_accuracy: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_brier: float = 0.25            # 0.25 = random baseline
    val_log_loss: float = 0.693        # ln(2) = random baseline

    # CPCV robustness metrics (Lopez de Prado AFML Ch.12)
    cpcv_mean: float = 0.0
    cpcv_std: float = 0.0

    # OOS metrics (computed by deploy gate on untouched OOS window)
    oos_accuracy: float = 0.0
    oos_auc: float = 0.0
    oos_sharpe: float = 0.0
    oos_profit_factor: float = 0.0
    oos_n_trades: int = 0

    # Deploy decision
    deployed: bool = False
    deploy_reason: str = ""             # "improvement", "first_gen", "forced", "declined"

    # Config hash (detect if config changed between generations)
    config_hash: str = ""


class ModelRegistry:
    """
    Persistent registry of all model generations with deploy gate logic.

    Usage:
        registry = ModelRegistry(model_dir)

        # At save time:
        meta = registry.create_metadata(agent_name, gen, ...)
        should_deploy, reason = registry.should_deploy(agent_name, meta)
        if should_deploy:
            save_model(...)
            registry.record_deployment(agent_name, meta)

        # At load time:
        meta = registry.get_latest_metadata(agent_name)
        if meta and meta.feature_count != current_feature_count:
            print("WARNING: feature dimension mismatch!")
    """

    # Deploy gate thresholds
    MIN_AUC_IMPROVEMENT = -0.02         # Allow up to 2% AUC drop (warm-start noise)
    MIN_ACCURACY = 0.50                 # Must beat random
    MAX_BRIER = 0.30                    # Must be somewhat calibrated
    DECLINE_STREAK_RESET = 3            # Force fresh retrain after 3 consecutive declines
    MIN_OOS_TRADES = 10                 # Need at least 10 OOS trades to evaluate

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self.registry_file = os.path.join(self.model_dir, 'model_registry.json')
        self._lock = threading.RLock()  # Protect all registry file I/O from concurrent access
        self._registry = self._load_registry()

    def _load_registry(self):
        """Load the persistent registry from disk (caller must hold self._lock or be in __init__)."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"ModelRegistry: Failed to load registry: {e}", exc_info=True)
        return {}

    def _save_registry(self):
        """Persist registry to disk atomically via a temp file (caller must hold self._lock)."""
        tmp_path = self.registry_file + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self._registry, f, indent=2, default=str)
            os.replace(tmp_path, self.registry_file)  # atomic on POSIX, best-effort on Windows
        except Exception as e:
            logging.error(f"ModelRegistry: Failed to save registry: {e}", exc_info=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def create_metadata(self, agent_name, generation, feature_count, feature_mask,
                        feature_indices=None, training_symbols=None,
                        n_samples=0, n_positive=0, n_negative=0,
                        cross_event_samples=0, parent_generation=0,
                        training_mode="warm_start", config_hash=""):
        """Create a metadata record for a newly trained model."""
        # Hash the feature indices for reproducibility tracking
        idx_hash = ""
        if feature_indices is not None:
            idx_bytes = np.array(feature_indices, dtype=np.int64).tobytes()
            idx_hash = hashlib.sha256(idx_bytes).hexdigest()[:16]

        meta = ModelMetadata(
            agent_name=agent_name,
            generation=generation,
            parent_generation=parent_generation,
            timestamp=datetime.utcnow().isoformat(),
            training_mode=training_mode,
            feature_count=feature_count,
            feature_mask=feature_mask or "none",
            feature_indices_hash=idx_hash,
            training_symbols=list(training_symbols or []),
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
            class_ratio=max(n_positive, n_negative) / max(min(n_positive, n_negative), 1),
            cross_event_samples=cross_event_samples,
            config_hash=config_hash,
        )
        return meta

    def set_val_metrics(self, meta, y_true, y_pred_proba):
        """Compute and set validation metrics on the metadata object."""
        try:
            from sklearn.metrics import (roc_auc_score, accuracy_score,
                                         precision_score, recall_score,
                                         f1_score, log_loss, brier_score_loss)

            y_pred = (y_pred_proba >= 0.5).astype(int)
            meta.val_accuracy = float(accuracy_score(y_true, y_pred))
            meta.val_precision = float(precision_score(y_true, y_pred, zero_division=0))
            meta.val_recall = float(recall_score(y_true, y_pred, zero_division=0))
            meta.val_f1 = float(f1_score(y_true, y_pred, zero_division=0))
            meta.val_brier = float(brier_score_loss(y_true, y_pred_proba))
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            meta.val_log_loss = float(log_loss(y_true, y_pred_proba_clipped))

            if len(np.unique(y_true)) > 1:
                meta.val_auc = float(roc_auc_score(y_true, y_pred_proba))
            else:
                meta.val_auc = 0.5
        except Exception as e:
            logging.warning(f"ModelRegistry: Failed to compute val metrics: {e}")

    def set_oos_metrics(self, meta, oos_pnls, oos_accuracy=0.0, oos_auc=0.0):
        """Set OOS (out-of-sample) evaluation metrics."""
        meta.oos_accuracy = oos_accuracy
        meta.oos_auc = oos_auc
        meta.oos_n_trades = len(oos_pnls) if oos_pnls is not None else 0

        if oos_pnls is not None and len(oos_pnls) > 1:
            oos_arr = np.array(oos_pnls, dtype=np.float64)
            # Sharpe ratio (annualized, assuming 5m bars)
            if np.std(oos_arr) > 1e-8:
                meta.oos_sharpe = float(np.mean(oos_arr) / np.std(oos_arr) * np.sqrt(252))
            # Profit factor
            gross_profit = float(np.sum(oos_arr[oos_arr > 0]))
            gross_loss = float(abs(np.sum(oos_arr[oos_arr < 0])))
            meta.oos_profit_factor = gross_profit / max(gross_loss, 1e-8)

    def should_deploy(self, agent_name, new_meta, oos_sharpe=None):
        """
        Deploy gate: decide whether the newly trained model should replace the current one.

        Args:
            agent_name: Specialist name (e.g. 'athena')
            new_meta:   ModelMetadata for the newly trained model
            oos_sharpe: Optional OOS Sharpe ratio from auto-backtest.
                        If provided and previous gen has OOS Sharpe, new must beat it.

        Returns:
            (should_deploy: bool, reason: str)
        """
        history = self._get_agent_history(agent_name)

        # First generation — always deploy
        if not history:
            return True, "first_generation"

        prev = history[-1]

        # ── CHECK 1: Absolute quality floor ──
        # Skip AUC/accuracy gates when val set is too small (<20 events) — unreliable metrics
        skip_auc_gate = getattr(new_meta, '_skip_auc_gate', False)

        if not skip_auc_gate:
            if new_meta.val_accuracy < self.MIN_ACCURACY:
                return False, f"val_accuracy {new_meta.val_accuracy:.3f} < {self.MIN_ACCURACY} (worse than random)"

            if new_meta.val_brier > self.MAX_BRIER:
                return False, f"val_brier {new_meta.val_brier:.3f} > {self.MAX_BRIER} (poorly calibrated)"
        else:
            print(f"   ⚠️  AUC/accuracy gate skipped (val set < 20 events — metrics unreliable)")

        # ── CHECK 2: Feature dimension compatibility ──
        if new_meta.feature_count != prev.get('feature_count', 0):
            # Feature dimension changed — must deploy (old model incompatible)
            return True, f"feature_dim_changed ({prev.get('feature_count', '?')} -> {new_meta.feature_count})"

        # ── CHECK 3: OOS Sharpe gate (auto-backtest) ──
        if oos_sharpe is not None:
            new_meta.oos_sharpe = oos_sharpe
            prev_sharpe = prev.get('oos_sharpe', 0.0)
            if prev_sharpe > 0 and oos_sharpe < prev_sharpe:
                return False, (f"oos_sharpe_regression: {prev_sharpe:.3f} -> {oos_sharpe:.3f} "
                               f"(new model underperforms on OOS backtest)")
            logging.info(f"ModelRegistry: {agent_name} OOS Sharpe {prev_sharpe:.3f} -> {oos_sharpe:.3f}")

        # ── CHECK 4: Relative improvement ──
        prev_auc = prev.get('val_auc', 0.5)
        auc_delta = new_meta.val_auc - prev_auc

        prev_brier = prev.get('val_brier', 0.25)
        brier_improved = new_meta.val_brier <= prev_brier

        prev_accuracy = prev.get('val_accuracy', 0.5)
        accuracy_delta = new_meta.val_accuracy - prev_accuracy

        # Accept if AUC improved OR (AUC within tolerance AND Brier improved)
        if auc_delta >= 0:
            return True, f"auc_improved ({prev_auc:.4f} -> {new_meta.val_auc:.4f}, +{auc_delta:.4f})"

        if auc_delta >= self.MIN_AUC_IMPROVEMENT and brier_improved:
            return True, f"auc_within_tolerance ({auc_delta:+.4f}) and brier_improved ({prev_brier:.4f} -> {new_meta.val_brier:.4f})"

        if accuracy_delta > 0.01 and brier_improved:
            return True, f"accuracy_improved ({accuracy_delta:+.4f}) and brier_improved"

        # ── DECLINED ──
        return False, (f"no_improvement: AUC {prev_auc:.4f}->{new_meta.val_auc:.4f} ({auc_delta:+.4f}), "
                       f"Brier {prev_brier:.4f}->{new_meta.val_brier:.4f}")

    def should_fresh_retrain(self, agent_name):
        """
        Check if this agent has declined for DECLINE_STREAK_RESET consecutive
        generations. If so, warm-starting is likely causing catastrophic forgetting
        and we should do a fresh retrain.

        Returns:
            (should_reset: bool, streak: int)
        """
        history = self._get_agent_history(agent_name)
        if len(history) < self.DECLINE_STREAK_RESET + 1:
            return False, 0

        # Count consecutive AUC declines from the latest generation backward
        streak = 0
        for i in range(len(history) - 1, 0, -1):
            current_auc = history[i].get('val_auc', 0.5)
            previous_auc = history[i - 1].get('val_auc', 0.5)
            if current_auc < previous_auc:
                streak += 1
            else:
                break

        return streak >= self.DECLINE_STREAK_RESET, streak

    def record_deployment(self, agent_name, meta, deployed=True, reason=""):
        """Record a model deployment (or rejection) to the registry (thread-safe)."""
        meta.deployed = deployed
        meta.deploy_reason = reason or meta.deploy_reason

        with self._lock:
            if agent_name not in self._registry:
                self._registry[agent_name] = []
            self._registry[agent_name].append(asdict(meta))
            self._save_registry()

        status = "DEPLOYED" if deployed else "REJECTED"
        logging.info(f"ModelRegistry: {agent_name} gen{meta.generation} {status}: {reason}")

    def get_latest_metadata(self, agent_name):
        """Get the most recent deployed metadata for an agent (thread-safe)."""
        with self._lock:
            history = self._get_agent_history(agent_name)
            if not history:
                return None
            for entry in reversed(history):
                if entry.get('deployed', False):
                    return entry
            return history[-1] if history else None

    def get_generation_history(self, agent_name):
        """Get full generation history for dashboard display."""
        return self._get_agent_history(agent_name)

    def get_lineage_summary(self, agent_name):
        """Get a compact summary of model lineage for logging."""
        history = self._get_agent_history(agent_name)
        if not history:
            return f"{agent_name}: no history"

        lines = []
        for h in history[-5:]:  # Last 5 generations
            gen = h.get('generation', '?')
            auc = h.get('val_auc', 0)
            brier = h.get('val_brier', 0.25)
            deployed = "✅" if h.get('deployed') else "❌"
            mode = h.get('training_mode', '?')[:5]
            reason = h.get('deploy_reason', '')[:30]
            lines.append(f"  gen{gen} [{mode}] AUC={auc:.4f} Brier={brier:.4f} {deployed} {reason}")

        return f"{agent_name} lineage (last {len(lines)} gens):\n" + "\n".join(lines)

    def validate_feature_compatibility(self, agent_name, current_feature_count, current_mask_hash=""):
        """
        Check if the currently loaded model is compatible with the current feature configuration.

        Returns:
            (compatible: bool, warning: str)
        """
        meta = self.get_latest_metadata(agent_name)
        if meta is None:
            return True, ""

        warnings = []
        stored_count = meta.get('feature_count', 0)
        stored_hash = meta.get('feature_indices_hash', '')

        if stored_count > 0 and stored_count != current_feature_count:
            warnings.append(
                f"Feature count mismatch: model trained on {stored_count}, "
                f"current config has {current_feature_count}"
            )

        if stored_hash and current_mask_hash and stored_hash != current_mask_hash:
            warnings.append(
                f"Feature mask changed since training "
                f"(old hash: {stored_hash[:8]}..., new: {current_mask_hash[:8]}...)"
            )

        if warnings:
            return False, "; ".join(warnings)
        return True, ""

    def _get_agent_history(self, agent_name):
        """Get the history list for an agent, or empty list."""
        return self._registry.get(agent_name, [])

    def get_all_agents_summary(self):
        """Dashboard-friendly summary of all agents."""
        summary = {}
        for agent_name in self._registry:
            history = self._registry[agent_name]
            if not history:
                continue
            latest = history[-1]
            deployed_count = sum(1 for h in history if h.get('deployed'))
            rejected_count = len(history) - deployed_count

            summary[agent_name] = {
                'total_generations': len(history),
                'deployed': deployed_count,
                'rejected': rejected_count,
                'latest_gen': latest.get('generation', 0),
                'latest_auc': latest.get('val_auc', 0),
                'latest_brier': latest.get('val_brier', 0.25),
                'latest_deployed': latest.get('deployed', False),
                'latest_reason': latest.get('deploy_reason', ''),
            }
        return summary
