"""
norse_tuner/learned_filter.py
==============================
Walk-forward logistic-regression entry filter that replaces blunt hour gates.

Trains 8 classifiers (one per SL candidate) strictly causally: fold-k's
test rows are predicted by a model trained only on rows 0..fold_k_start.
This means the filter will refuse to overfit to artefacts like "hour 14 UTC
looks good" unless those patterns genuinely generalise across all folds.

Public API
----------
    from norse_tuner.learned_filter import LearnedFilter

    lf = LearnedFilter(df)          # df = norse_pump_mae_rows.csv DataFrame
    lf.fit(fold_bounds)             # ~20 s for 8 SLs × 5 folds
    # proba arrays now in lf.proba_by_sl — attach to CachedTrialEvaluator:
    evaluator.learned_proba = lf.proba_by_sl
    # last-fold models pickled to norse_tuner/learned_filter_{sl}.pkl for live use
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

_FEATURE_COLS = [
    "nike_score",
    "hour_of_day_utc",
    "hour_of_week",
    "regime_state",
    "weighted_trend",
    "bs_prob",
    "wave_strength_score_at_entry",
    "top_risk_score_at_entry",
    "participation_score_at_entry",
    "volume_ratio_at_entry",
    "flow_exhaustion_score_at_entry",
    "vpin_at_entry",
    "atr_rank_at_entry",
    "upper_wick_ratio_at_entry",
    "close_pos_at_entry",
    "impulse_body_eff_at_entry",
    "impulse_taker_persist_at_entry",
    "pre_impulse_r2_at_entry",
    "chain_pump_flag",
]

_SL_TAGS = ["0p8", "1p0", "1p2", "1p5", "1p8", "2p0", "2p4", "3p0"]
_SL_KEYS = [0.8,   1.0,   1.2,   1.5,   1.8,   2.0,   2.4,   3.0]


def _extract_X(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    X = X.fillna(0.0)
    # chain_pump_flag is bool — cast to float
    if "chain_pump_flag" in X.columns:
        X["chain_pump_flag"] = X["chain_pump_flag"].astype(float)
    return X.to_numpy(dtype=np.float64)


class LearnedFilter:
    """
    Walk-forward entry profitability classifier.

    Attributes
    ----------
    proba_by_sl : dict[float, np.ndarray]
        Per-row P(profitable) for each SL candidate (NaN for burn-in rows).
    models : dict[float, tuple[LogisticRegression, StandardScaler]]
        Last-fold models for live inference.
    auc_by_sl : dict[float, list[float]]
        Per-fold validation AUC for quality monitoring.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.sort_values("entry_ts").reset_index(drop=True)
        self._X  = _extract_X(self._df)
        self.proba_by_sl: dict[float, np.ndarray] = {}
        self.models: dict[float, tuple] = {}
        self.auc_by_sl: dict[float, list[float]] = {}

    def fit(
        self,
        fold_bounds: list[tuple[int, int]],
        verbose: bool = True,
    ) -> None:
        """
        Train walk-forward classifiers for all 8 SL candidates.

        fold_bounds : list[(lo, hi)] — same as CachedTrialEvaluator._fold_bounds.
                      Fold-k trains on rows 0..lo, tests on rows lo..hi.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        n = len(self._df)
        X_all = self._X

        for sl_key, sl_tag in zip(_SL_KEYS, _SL_TAGS):
            target_col = f"realized_atr_sl_{sl_tag}"
            if target_col not in self._df.columns:
                continue

            y_all = (self._df[target_col] > 0).astype(int).to_numpy()
            proba = np.full(n, np.nan, dtype=np.float64)
            fold_aucs: list[float] = []
            last_model = last_scaler = None

            for lo, hi in fold_bounds:
                if lo < 50:
                    # Not enough training data yet
                    continue
                Xtr, ytr = X_all[:lo], y_all[:lo]
                # Drop NaN rows
                valid = np.isfinite(Xtr).all(axis=1)
                Xtr, ytr = Xtr[valid], ytr[valid]
                if len(np.unique(ytr)) < 2 or len(ytr) < 50:
                    continue

                scaler = StandardScaler()
                Xtr_s  = scaler.fit_transform(Xtr)
                model  = LogisticRegression(
                    penalty="l2", C=0.5, class_weight="balanced",
                    max_iter=500, random_state=42, n_jobs=1,
                )
                model.fit(Xtr_s, ytr)

                Xte_s = scaler.transform(X_all[lo:hi])
                p     = model.predict_proba(Xte_s)[:, 1]
                proba[lo:hi] = p

                # AUC for quality tracking
                yte = y_all[lo:hi]
                if len(np.unique(yte)) == 2:
                    try:
                        auc = float(roc_auc_score(yte, p))
                        fold_aucs.append(auc)
                    except Exception:
                        pass

                last_model, last_scaler = model, scaler

            self.proba_by_sl[sl_key] = proba
            self.auc_by_sl[sl_key]   = fold_aucs

            if last_model is not None:
                self.models[sl_key] = (last_model, last_scaler)
                pkl_path = _HERE / f"learned_filter_{sl_tag}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump({"model": last_model, "scaler": last_scaler,
                                 "feature_cols": [c for c in _FEATURE_COLS if c in self._df.columns]}, f)

            if verbose:
                avg_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
                print(
                    f"  [learned_filter] sl={sl_key:.1f}  "
                    f"folds_trained={len(fold_aucs)}  "
                    f"avg_auc={avg_auc:.4f}",
                    flush=True,
                )
                if avg_auc < 0.55 and not np.isnan(avg_auc):
                    print(
                        f"    WARNING: AUC {avg_auc:.4f} < 0.55 — filter for sl={sl_key:.1f} "
                        "may not add value; threshold will still be tuned by Optuna.",
                        flush=True,
                    )

    def top_features(self, sl_key: float, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n features by |coefficient| for the last-fold model."""
        if sl_key not in self.models:
            return []
        model, _ = self.models[sl_key]
        cols = [c for c in _FEATURE_COLS if c in self._df.columns]
        coefs = model.coef_[0]
        pairs = sorted(zip(cols, coefs.tolist()), key=lambda x: abs(x[1]), reverse=True)
        return pairs[:n]
