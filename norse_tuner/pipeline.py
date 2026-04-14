"""
norse_tuner/pipeline.py
=======================
Top-level orchestrator for the 3-stage Turbo Tuner.

Replaces _tune_params in quanta_norse_year_sim.py with the same call signature.
Stage 0 (event cache build) is run externally via norse_event_cache_builder.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MAE_CSV = _ROOT / "norse_pump_mae_rows.csv"

_T0 = time.time()
_BOOTSTRAP_REVIEW_LIMIT = 12
_BOOTSTRAP_PASS_KEEP = 4
_FULLSIM_RERANK_TOP_K = 3
_FULLSIM_DD_SOFT_CAP = 35.0


def _log(msg: str) -> None:
    print(f"[tuner {time.time() - _T0:6.0f}s] {msg}", flush=True)


def _trial_signature(params: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), repr(v)) for k, v in params.items()))


def _fullsim_rank_key(summary: dict) -> tuple[float, float, float, int]:
    growth = float(summary.get("growth_pct", -100.0))
    dd = float(summary.get("max_drawdown_pct", 100.0))
    trades = int(summary.get("trades", 0))
    calmar = growth / max(dd, 10.0)
    dd_over = max(0.0, dd - _FULLSIM_DD_SOFT_CAP)
    trade_credit = min(0.5, trades / 120.0)
    score = calmar + 0.01 * growth - 0.03 * dd_over + trade_credit
    return float(score), growth, -dd, trades


def run_tuning_pipeline(
    prepared_symbols: list,
    base_params: dict,
    mae_csv_path: Optional[str | Path] = None,
    n_optuna_trials: int = 800,
    skip_learned_filter: bool = False,
    skip_fine_tune: bool = False,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Drop-in replacement for _tune_params.

    Returns
    -------
    (best_params, best_result, tuning_df)
      best_result: dict from _evaluate_params (full pump outputs included)
      tuning_df: history DataFrame compatible with existing report code
    """
    from quanta_norse_year_sim import _evaluate_params

    mae_path = Path(mae_csv_path) if mae_csv_path else _MAE_CSV
    history_rows: list[dict] = []

    _log("Stage 1: loading CachedTrialEvaluator ...")
    from .cached_evaluator import CachedTrialEvaluator

    evaluator = CachedTrialEvaluator(mae_path)
    _log(f"  loaded {evaluator.n_rows} rows  folds={evaluator.n_folds}")

    use_learned_filter = (not skip_learned_filter) and mae_path.exists()
    if use_learned_filter:
        _log("Stage 1: training walk-forward entry filter ...")
        from .learned_filter import LearnedFilter

        df_mae = pd.read_csv(mae_path).sort_values("entry_ts").reset_index(drop=True)
        lf = LearnedFilter(df_mae)
        lf.fit(evaluator._fold_bounds, verbose=True)
        evaluator.learned_proba = lf.proba_by_sl
        learned_filter_obj = lf
    else:
        learned_filter_obj = None

    _log(f"Stage 1: Optuna TPE ({n_optuna_trials} trials) ...")
    from .optuna_search import run_optuna

    study = run_optuna(
        evaluator,
        base_params,
        n_trials=n_optuna_trials,
        use_learned_filter=use_learned_filter,
    )

    all_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )
    unique_trials = []
    seen_signatures: set[tuple[tuple[str, str], ...]] = set()
    for trial in all_trials:
        sig = _trial_signature(trial.params)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        unique_trials.append(trial)

    _log("Stage 1: bootstrap CI gate on top unique candidates ...")
    from .objective import bootstrap_passed

    stage1_candidates: list[dict] = []
    for trial in unique_trials[:_BOOTSTRAP_REVIEW_LIMIT]:
        candidate = dict(base_params)
        candidate.update(trial.params)
        stage1_estimate = evaluator.evaluate_full(candidate)
        realized = stage1_estimate.get("realized_atr", np.array([]))
        if len(realized) > 0 and bootstrap_passed(realized):
            _log(f"  bootstrap PASSED: trial #{trial.number}  score={trial.value:.4f}")
            stage1_candidates.append(
                {
                    "trial": trial,
                    "cacheable_params": candidate,
                    "stage1_estimate": stage1_estimate,
                    "bootstrap_passed": True,
                }
            )
            if len(stage1_candidates) >= _BOOTSTRAP_PASS_KEEP:
                break
        else:
            _log(f"  bootstrap REJECTED: trial #{trial.number}  score={trial.value:.4f}")

    if not stage1_candidates:
        _log("  WARNING: no candidate passed bootstrap; falling back to top unique trials")
        for trial in unique_trials[:_FULLSIM_RERANK_TOP_K]:
            candidate = dict(base_params)
            candidate.update(trial.params)
            stage1_candidates.append(
                {
                    "trial": trial,
                    "cacheable_params": candidate,
                    "stage1_estimate": evaluator.evaluate_full(candidate),
                    "bootstrap_passed": False,
                }
            )

    best_cacheable_params = dict(stage1_candidates[0]["cacheable_params"])
    _log(
        f"Stage 1 done  best_hour={best_cacheable_params.get('thor_entry_hour_utc', '')!r}"
        f"  tier={best_cacheable_params.get('thor_trade_tiers', '')}"
        f"  sl={best_cacheable_params.get('thor_sl_atr', '')}"
        f"  score={best_cacheable_params.get('thor_min_score_trade', '')}"
    )

    for trial in all_trials:
        history_rows.append(
            {
                "phase": "optuna",
                "param": str(trial.params),
                "value": trial.value,
                "trial_number": trial.number,
                "final_capital": np.nan,
                "growth_pct": np.nan,
                "max_drawdown_pct": np.nan,
                "trades": np.nan,
                "active_agents": "",
                "objective_score": trial.value if trial.value is not None else np.nan,
            }
        )

    rerank_candidates = stage1_candidates[:_FULLSIM_RERANK_TOP_K]
    _log(f"Stage 2: full-sim rerank on top-{len(rerank_candidates)} candidates ...")

    from .numba_fine_tune import evaluate_exit_params_full, fine_tune_exits

    best_params: Optional[dict] = None
    best_result: Optional[dict] = None
    best_fast_est: Optional[dict] = None
    best_rank_key: Optional[tuple[float, float, float, int]] = None

    for idx, candidate_row in enumerate(rerank_candidates, 1):
        trial = candidate_row["trial"]
        stage1_params = dict(candidate_row["cacheable_params"])
        _log(
            f"  candidate {idx}/{len(rerank_candidates)}: trial #{trial.number}"
            f"  score={trial.value:.4f}"
            f"  bootstrap={'yes' if candidate_row['bootstrap_passed'] else 'fallback'}"
        )

        if not skip_fine_tune:
            exit_delta = fine_tune_exits(
                prepared_symbols,
                stage1_params,
                evaluator,
                mae_csv_path=str(mae_path),
            )
        else:
            exit_delta = {}

        full_params = dict(stage1_params)
        full_params.update(exit_delta)
        fast_est = evaluate_exit_params_full(prepared_symbols, full_params)
        full_result = _evaluate_params(
            prepared_symbols,
            full_params,
            include_pump_outputs=True,
            sim_logger=_log,
            detailed_progress=True,
        )
        full_result["params"] = full_params
        rank_key = _fullsim_rank_key(full_result["portfolio_summary"])

        history_rows.append(
            {
                "phase": "fullsim_rerank",
                "param": str(full_params),
                "value": trial.value,
                "trial_number": trial.number,
                "final_capital": full_result["portfolio_summary"]["final_capital"],
                "growth_pct": full_result["portfolio_summary"]["growth_pct"],
                "max_drawdown_pct": full_result["portfolio_summary"]["max_drawdown_pct"],
                "trades": full_result["portfolio_summary"]["trades"],
                "active_agents": ",".join(full_result.get("active_agents", [])),
                "objective_score": rank_key[0],
            }
        )
        _log(
            f"    full sim  growth={full_result['portfolio_summary']['growth_pct']:+.2f}%"
            f"  dd={full_result['portfolio_summary']['max_drawdown_pct']:.2f}%"
            f"  trades={full_result['portfolio_summary']['trades']}"
            f"  rank={rank_key[0]:.4f}"
        )

        if best_result is None or rank_key > best_rank_key:
            best_params = full_params
            best_result = full_result
            best_fast_est = fast_est
            best_rank_key = rank_key

    if best_params is None or best_result is None or best_fast_est is None:
        raise RuntimeError("Turbo candidate rerank produced no valid full-sim result.")

    _log("Stage 3: sensitivity sweep (+/-20%) ...")
    sensitivity_rows = _sensitivity_sweep(evaluator, best_params)
    brittle = [r["param"] for r in sensitivity_rows if r["brittle"]]
    if brittle:
        _log(f"  BRITTLE params: {brittle}")

    cache_vs_sim = {
        "cached_final_capital": best_fast_est.get("final_capital", 0.0),
        "sim_final_capital": best_result["portfolio_summary"]["final_capital"],
        "cached_max_dd_pct": best_fast_est.get("max_drawdown_pct", 0.0),
        "sim_max_dd_pct": best_result["portfolio_summary"]["max_drawdown_pct"],
    }
    cap_div = abs(cache_vs_sim["cached_final_capital"] - cache_vs_sim["sim_final_capital"]) / max(
        abs(cache_vs_sim["cached_final_capital"]), 1.0
    )
    dd_div = abs(cache_vs_sim["cached_max_dd_pct"] - cache_vs_sim["sim_max_dd_pct"])
    if cap_div > 0.05 or dd_div > 3.0:
        _log(
            f"  CACHE_DIVERGENCE: capital_delta={cap_div:.1%}  dd_delta={dd_div:.1f}pp"
            "  (fast replay is approximate; full sim is ground truth)"
        )
    else:
        _log(f"  cache-vs-sim OK: capital_delta={cap_div:.1%}  dd_delta={dd_div:.1f}pp")

    best_result["sensitivity_rows"] = sensitivity_rows
    best_result["cache_vs_sim"] = cache_vs_sim
    best_result["learned_filter"] = learned_filter_obj
    best_result["optuna_study"] = study
    best_result["n_optuna_trials"] = len(study.trials)

    wf_fold_metrics = evaluator.evaluate_walkforward(best_params)
    best_result["wf_fold_metrics"] = wf_fold_metrics

    _log(
        f"DONE  growth={best_result['portfolio_summary']['growth_pct']:+.2f}%"
        f"  dd={best_result['portfolio_summary']['max_drawdown_pct']:.2f}%"
        f"  trades={best_result['portfolio_summary']['trades']}"
    )

    tuning_df = pd.DataFrame(history_rows)
    return best_params, best_result, tuning_df


_SENSITIVITY_PERTURB = {
    "thor_min_score_trade": ("float", 0.20),
    "thor_wave_strength_min": ("float", 0.20),
    "thor_top_risk_max": ("float", 0.20),
    "thor_mae_veto_atr": ("float", 0.20),
    "thor_sl_atr": ("nearest_sl", None),
    "thor_trade_cooldown_bars": ("int", 0.20),
    "baldur_warning_exit_score": ("float", 0.20),
    "learned_filter_threshold": ("float", 0.20),
    "thor_entry_hour_utc": ("skip", None),
}


def _perturb(val, ptype: str, frac: float, direction: float):
    if ptype == "float":
        return val * (1.0 + direction * frac)
    if ptype == "int":
        return max(1, int(round(val * (1.0 + direction * frac))))
    if ptype == "nearest_sl":
        from .cached_evaluator import _SL_LEVELS

        idx = _SL_LEVELS.index(val) if val in _SL_LEVELS else 0
        new_idx = max(0, min(len(_SL_LEVELS) - 1, idx + int(direction)))
        return _SL_LEVELS[new_idx]
    return val


def _sensitivity_sweep(evaluator, best_params: dict) -> list[dict]:
    from .objective import score_multifold, sensitivity_drop

    base_score = score_multifold(evaluator.evaluate_walkforward(best_params))
    rows = []

    for param, (ptype, frac) in _SENSITIVITY_PERTURB.items():
        if param not in best_params or ptype == "skip":
            continue
        base_val = best_params[param]
        row = {"param": param, "value": base_val, "base_score": base_score}
        for direction, label in [(-1, "minus20"), (+1, "plus20")]:
            trial = dict(best_params)
            trial[param] = _perturb(base_val, ptype, frac or 0.2, direction)
            score = score_multifold(evaluator.evaluate_walkforward(trial))
            drop = sensitivity_drop(base_score, score)
            row[f"{label}_score"] = score
            row[f"{label}_drop"] = drop
        max_drop = max(row.get("minus20_drop", 0.0), row.get("plus20_drop", 0.0))
        row["brittle"] = max_drop > 0.30
        rows.append(row)

    return rows
