"""
norse_tuner/optuna_search.py
=============================
Optuna TPE search over the cacheable parameter space.

Pattern cloned from QUANTA_ml_engine.py:1029-1104 (persistent pickle study,
TPESampler seed=42, multivariate=True for cross-param correlation).

Public API
----------
    from norse_tuner.optuna_search import run_optuna

    study = run_optuna(evaluator, base_params, n_trials=800)
    best  = study.best_params
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_DIR  = Path(__file__).resolve().parent.parent / "optuna_studies"
_STUDY_PATH = _STUDY_DIR / "norse_tuner.pkl"
_STUDY_CONTRACT_VERSION = 7   # bumped: reverted three-phase, wide multi-day runner capture

# Search space — 10 dimensions covering all cacheable params.
# "" for thor_entry_hour_utc means no hour gate (Optuna can learn to avoid it).
_HOUR_CHOICES  = ["", "13-15", "12-16", "11-17", "14", "13,14,15", "14,15", "12-18"]
_TIER_CHOICES  = ["A", "B", "C", "A,B", "A,B,C", "B,C"]
# Restoring full parameter sweep (except computationally toxic 0.8-1.0) so the ML hones naturally
_SL_CHOICES    = [1.2, 1.5, 1.8, 2.0, 2.4, 3.0, 3.5, 4.0]


def _build_trial_params(trial: optuna.Trial, base_params: dict, use_learned_filter: bool) -> dict:
    p = dict(base_params)
    p["thor_entry_hour_utc"]    = trial.suggest_categorical("thor_entry_hour_utc",    _HOUR_CHOICES)
    p["thor_trade_tiers"]       = trial.suggest_categorical("thor_trade_tiers",       _TIER_CHOICES)
    p["thor_min_score_trade"]   = trial.suggest_float("thor_min_score_trade",   60.0, 95.0, step=2.0)
    p["thor_wave_strength_min"] = trial.suggest_float("thor_wave_strength_min", 25.0, 80.0, step=5.0)
    p["thor_top_risk_max"]      = trial.suggest_float("thor_top_risk_max",      30.0, 80.0, step=5.0)
    p["thor_mae_veto_atr"]      = trial.suggest_float("thor_mae_veto_atr",       1.0,  8.0, step=0.2)
    p["thor_sl_atr"]            = trial.suggest_categorical("thor_sl_atr",         _SL_CHOICES)
    p["thor_trade_cooldown_bars"] = trial.suggest_int("thor_trade_cooldown_bars", 12, 144, step=12)
    p["baldur_warning_exit_score"] = trial.suggest_float("baldur_warning_exit_score", 55.0, 90.0, step=2.5)
    if use_learned_filter:
        p["learned_filter_threshold"] = trial.suggest_float("learned_filter_threshold", 0.10, 0.70, step=0.02)
    else:
        p["learned_filter_threshold"] = 0.0
    return p


def run_optuna(
    evaluator,
    base_params: dict,
    n_trials: int = 800,
    use_learned_filter: bool = True,
    resume: bool = True,
) -> optuna.Study:
    """
    Run Optuna TPE over cacheable params.

    Parameters
    ----------
    evaluator        : CachedTrialEvaluator
    base_params      : dict — full year-sim param dict (non-cacheable values
                       are passed through to every trial unchanged)
    n_trials         : number of Optuna trials
    use_learned_filter: include learned_filter_threshold in search space
    resume           : if True, load a previously saved study from disk

    Returns
    -------
    optuna.Study with best_params populated.
    """
    from .objective import score_multifold

    _STUDY_DIR.mkdir(exist_ok=True)

    # ── Load or create study ─────────────────────────────────────────────────
    study: Optional[optuna.Study] = None
    if resume and _STUDY_PATH.exists():
        try:
            with open(_STUDY_PATH, "rb") as f:
                study = pickle.load(f)
            if study.user_attrs.get("contract_version") != _STUDY_CONTRACT_VERSION:
                print(
                    "  [optuna] ignoring persisted study with incompatible contract "
                    f"(found={study.user_attrs.get('contract_version')} expected={_STUDY_CONTRACT_VERSION})",
                    flush=True,
                )
                study = None
            elif study.user_attrs.get("score_column") != "thor_feature_score":
                print(
                    "  [optuna] ignoring persisted study with incompatible score column "
                    f"(found={study.user_attrs.get('score_column')!r})",
                    flush=True,
                )
                study = None
        except Exception:
            study = None

    if study is None:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=40,
                multivariate=True,
                group=True,
            ),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        )
        study.set_user_attr("contract_version", _STUDY_CONTRACT_VERSION)
        study.set_user_attr("score_column", "thor_feature_score")
    else:
        print(
            f"  [optuna] loaded existing study ({len(study.trials)} prior trials)",
            flush=True,
        )
        study.set_user_attr("contract_version", _STUDY_CONTRACT_VERSION)
        study.set_user_attr("score_column", "thor_feature_score")

    def objective(trial: optuna.Trial) -> float:
        params = _build_trial_params(trial, base_params, use_learned_filter)
        fold_metrics = evaluator.evaluate_walkforward(params)

        # Report intermediate values for pruner
        for k, m in enumerate(fold_metrics):
            from .objective import score_single
            trial.report(score_single(m), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score_multifold(fold_metrics)

    print(
        f"  [optuna] running {n_trials} trials  "
        f"(learned_filter={'yes' if use_learned_filter else 'no'})",
        flush=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Persist study ────────────────────────────────────────────────────────
    try:
        with open(_STUDY_PATH, "wb") as f:
            pickle.dump(study, f)
        print(f"  [optuna] study saved → {_STUDY_PATH}", flush=True)
    except Exception as e:
        print(f"  [optuna] WARNING: study save failed: {e}", flush=True)

    best = study.best_trial
    print(
        f"  [optuna] best score={best.value:.4f}  "
        f"trial #{best.number}  params={best.params}",
        flush=True,
    )
    return study
