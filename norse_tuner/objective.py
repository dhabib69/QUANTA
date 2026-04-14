"""
norse_tuner/objective.py
========================
Multi-objective scoring functions used by every stage of the Turbo Tuner.

Key design:
- Calmar ratio (growth / max_dd) as the primary metric — avoids the
  current `final_capital * (1-dd_penalty)` that treats 3000%/70%DD
  the same as 800%/20%DD.
- median-of-folds aggregation (not mean, not max) — structural defence
  against the hour-14 overfitting problem: a param that looks great on
  one fold but collapses on others gets a bad median.
- stability penalty on fold variance — further penalises brittle params.
- bootstrap gate: rejects candidates whose 5th-percentile PF < 1.1.
"""

from __future__ import annotations

import numpy as np


MIN_TRADES_PER_FOLD = 4
TARGET_TRADES_PER_FOLD = 12
MIN_TRADE_CONFIDENCE = 0.25
LOW_TRADE_PENALTY = 5.0
STABILITY_WEIGHT    = 0.5   # penalty multiplier on fold stdev
WORST_DD_PENALTY    = 10.0  # applied per unit above 50% DD
WORST_DD_THRESHOLD  = 50.0  # relaxed cap to prioritize capital growth
TRADE_BONUS_CAP     = 2.0   # maximum bonus for high trade count
TRADE_BONUS_TARGET  = 120   # lower target so low-frequency Thor still gets some credit
ASYMMETRY_PENALTY_WEIGHT = 5.0  # penalty scale for avg_loss > avg_win
BOOTSTRAP_MIN_PF    = 1.1   # 5th-pct PF threshold for bootstrap gate


def _trade_confidence(trades: int) -> float:
    if trades <= 0:
        return 0.0
    if trades >= TARGET_TRADES_PER_FOLD:
        return 1.0
    frac = max(0.0, float(trades) / float(TARGET_TRADES_PER_FOLD))
    return max(MIN_TRADE_CONFIDENCE, float(np.sqrt(frac)))


def score_single(metrics: dict) -> float:
    """
    Score one fold / one full-sim result.

    metrics keys: growth_pct, max_drawdown_pct, trades, pf, expectancy, calmar
    """
    trades = int(metrics["trades"])
    if trades <= 0:
        return -100.0
    base_score = float(metrics["calmar"])
    
    avg_win = float(metrics.get("avg_win", 0.0))
    avg_loss = float(metrics.get("avg_loss", 0.0))
    if avg_loss > avg_win and avg_win > 0.0:
        ratio = avg_loss / avg_win
        base_score -= (ratio - 1.0) * ASYMMETRY_PENALTY_WEIGHT

    confidence = _trade_confidence(trades)

    # Low-trade folds should have less positive influence, but they should not
    # collapse the entire objective for a naturally low-frequency strategy.
    if base_score >= 0.0:
        score = base_score * confidence
    else:
        score = base_score / max(confidence, MIN_TRADE_CONFIDENCE)

    if trades < MIN_TRADES_PER_FOLD:
        underfill = float(MIN_TRADES_PER_FOLD - trades) / float(MIN_TRADES_PER_FOLD)
        score -= LOW_TRADE_PENALTY * underfill
    return float(score)


def score_multifold(fold_metrics: list[dict]) -> float:
    """
    Aggregate per-fold Calmar scores into a single trial objective.

    Returns higher = better.
    """
    calmars = [score_single(m) for m in fold_metrics]
    dds     = [m["max_drawdown_pct"] for m in fold_metrics]
    trades  = [m["trades"]          for m in fold_metrics]

    med_calmar = float(np.median(calmars))
    std_calmar = float(np.std(calmars))
    worst_dd   = float(np.max(dds))
    total_tr   = int(np.sum(trades))

    stability_penalty = STABILITY_WEIGHT * std_calmar
    dd_penalty        = max(0.0, (worst_dd - WORST_DD_THRESHOLD) / 100.0) * WORST_DD_PENALTY
    trade_bonus       = min(TRADE_BONUS_CAP, total_tr / TRADE_BONUS_TARGET * TRADE_BONUS_CAP)

    return med_calmar - stability_penalty - dd_penalty + trade_bonus


def bootstrap_pf_lower(realized_atr: np.ndarray, n_boot: int = 1000) -> float:
    """
    Return the 5th-percentile PF from n_boot bootstrap resamples.
    Used to reject lucky candidates.
    """
    if len(realized_atr) < 10:
        return 0.0
    rng = np.random.default_rng(42)
    n   = len(realized_atr)
    pfs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(realized_atr, size=n, replace=True)
        wins   = float(sample[sample > 0].sum())
        losses = float(-sample[sample < 0].sum())
        pfs[i] = wins / losses if losses > 0 else 10.0
    return float(np.percentile(pfs, 5))


def bootstrap_passed(realized_atr: np.ndarray) -> bool:
    """Return True if the candidate passes the bootstrap gate."""
    return bootstrap_pf_lower(realized_atr) >= BOOTSTRAP_MIN_PF


def sensitivity_drop(base_score: float, perturbed_score: float) -> float:
    """Fractional Calmar drop under a ±20% perturbation."""
    if base_score <= 0:
        return 0.0
    return max(0.0, (base_score - perturbed_score) / base_score)
