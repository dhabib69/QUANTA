"""
norse_tuner/numba_fine_tune.py
================================
Stage 2 grid sweep over Thor's non-cacheable exit params.

This module now uses the same fast Thor capital replay as the cached evaluator
instead of scoring combos with pure ATR compounding.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional

import numpy as np

from quanta_config import Config

from .fast_capital import empty_metrics, simulate_fast_thor_metrics
from .objective import score_multifold


_ROUND1_GRID = {
    "thor_bank_atr": [3.0, 4.2, 5.4, 6.6],
    "thor_bank_fraction": [0.15, 0.25, 0.35, 0.45],
    "thor_runner_trail_atr": [3.0, 4.0, 5.0, 6.0],
}
_ROUND2_GRID = {
    "thor_trail_activate_atr": [1.0, 1.5, 2.0, 2.5],
    "thor_max_bars_pre_bank": [72, 144, 288],
    "thor_max_bars_post_bank": [288, 576, 1152],
}
_ROUND1_KEEP = 4

_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[fine_tune {time.time() - _T0:5.0f}s] {msg}", flush=True)


def _sweep_symbol(
    prep,
    params: dict,
) -> list[dict]:
    """
    Run the Thor exit sim for one symbol with a single param combo.
    Returns fast-replay trade rows for entries that pass the stage-1 filters.
    """
    from quanta_norse_agents import simulate_thor_exit_stop_market
    from quanta_norse_year_sim import _allowed_tiers, _get_cached_thor_base, _parse_allowed_hours

    allowed_tiers = _allowed_tiers(params["thor_trade_tiers"])
    allowed_hours = _parse_allowed_hours(params.get("thor_entry_hour_utc", ""))
    cooldown = int(params["thor_trade_cooldown_bars"])
    last_bar = -cooldown
    ctx = prep.feature_ctx

    rows: list[dict] = []
    for sig in prep.raw_thor_signals:
        i = int(sig["bar_idx"])
        if i - last_bar < cooldown:
            continue
        if allowed_tiers and str(sig.get("tier", "")).upper() not in allowed_tiers:
            continue
        if float(sig.get("atr", 0.0)) <= 0.0:
            continue

        base = _get_cached_thor_base(prep, sig, params)
        if allowed_hours and int(base["entry_hour_utc"]) not in allowed_hours:
            continue

        pre = base["pre_entry_state"]
        if pre["max_drawdown_atr"] > float(params["thor_mae_veto_atr"]):
            continue
        if pre["wave_strength_score"] < float(params["thor_wave_strength_min"]):
            continue
        if pre["top_risk_score"] >= float(params["thor_top_risk_max"]):
            continue
        if float(base["feature_score"]) < float(params["thor_min_score_trade"]):
            continue

        sim = simulate_thor_exit_stop_market(
            sig,
            prep.feature_ctx,
            bank_atr=float(params["thor_bank_atr"]),
            sl_atr=float(params["thor_sl_atr"]),
            bank_fraction=float(params["thor_bank_fraction"]),
            trail_activate_atr=float(params["thor_trail_activate_atr"]),
            runner_trail_atr=float(params["thor_runner_trail_atr"]),
            max_bars_pre_bank=int(params["thor_max_bars_pre_bank"]),
            max_bars_post_bank=int(params["thor_max_bars_post_bank"]),
            penetration_factor=float(params["stop_market_penetration_factor"]),
            slip_cap_atr=float(params["stop_market_slip_atr_cap"]),
            material_drawdown_atr=float(params.get("pump_material_drawdown_atr", 1.0)),
        )
        exit_bar = int(sim["exit_bar"])
        rows.append(
            {
                "symbol": prep.symbol,
                "entry_ts": int(ctx.times[i]),
                "exit_ts": int(ctx.times[exit_bar]),
                "entry_price": float(ctx.closes[i]),
                "entry_atr": float(sig["atr"]),
                "realized_atr": float(sim["realized_atr"]),
                "score": float(base["feature_score"]),
            }
        )
        last_bar = i

    return rows


def _collect_trade_arrays(
    prepared_symbols: list,
    params: dict,
) -> dict[str, np.ndarray]:
    entry_ts: list[int] = []
    exit_ts: list[int] = []
    symbols: list[str] = []
    entry_prices: list[float] = []
    entry_atrs: list[float] = []
    realized: list[float] = []
    scores: list[float] = []

    for prep in prepared_symbols:
        for row in _sweep_symbol(prep, params):
            entry_ts.append(int(row["entry_ts"]))
            exit_ts.append(int(row["exit_ts"]))
            symbols.append(str(row["symbol"]))
            entry_prices.append(float(row["entry_price"]))
            entry_atrs.append(float(row["entry_atr"]))
            realized.append(float(row["realized_atr"]))
            scores.append(float(row["score"]))

    if not entry_ts:
        return {
            "entry_ts": np.zeros(0, dtype=np.int64),
            "exit_ts": np.zeros(0, dtype=np.int64),
            "symbol": np.empty(0, dtype=object),
            "entry_price": np.zeros(0, dtype=np.float64),
            "entry_atr": np.zeros(0, dtype=np.float64),
            "realized_atr": np.zeros(0, dtype=np.float64),
            "score": np.zeros(0, dtype=np.float64),
        }

    order = np.argsort(np.asarray(entry_ts, dtype=np.int64), kind="stable")
    return {
        "entry_ts": np.asarray(entry_ts, dtype=np.int64)[order],
        "exit_ts": np.asarray(exit_ts, dtype=np.int64)[order],
        "symbol": np.asarray(symbols, dtype=object)[order],
        "entry_price": np.asarray(entry_prices, dtype=np.float64)[order],
        "entry_atr": np.asarray(entry_atrs, dtype=np.float64)[order],
        "realized_atr": np.asarray(realized, dtype=np.float64)[order],
        "score": np.asarray(scores, dtype=np.float64)[order],
    }


def _evaluate_combo(
    prepared_symbols: list,
    params: dict,
    fold_ts_bounds: list[tuple[int, int]],
) -> list[dict]:
    """
    Evaluate one bank/trail combo across walk-forward folds.
    Returns per-fold metric dicts.
    """
    trades = _collect_trade_arrays(prepared_symbols, params)
    ts_arr = trades["entry_ts"]

    if len(ts_arr) == 0:
        return [empty_metrics() for _ in fold_ts_bounds]

    bt = Config.backtest
    ev = Config.events
    fold_metrics: list[dict] = []
    for lo_ts, hi_ts in fold_ts_bounds:
        fold_mask = (ts_arr >= lo_ts) & (ts_arr < hi_ts)
        if not fold_mask.any():
            fold_metrics.append(empty_metrics())
            continue

        fold_metrics.append(
            simulate_fast_thor_metrics(
                entry_ts=trades["entry_ts"][fold_mask],
                exit_ts=trades["exit_ts"][fold_mask],
                symbol=trades["symbol"][fold_mask],
                entry_price=trades["entry_price"][fold_mask],
                entry_atr=trades["entry_atr"][fold_mask],
                realized_atr=trades["realized_atr"][fold_mask],
                score=trades["score"][fold_mask],
                stop_atr=float(params.get("thor_sl_atr", 3.0)),
                risk_pct=float(params.get("thor_risk_pct", 0.005)),
                max_leverage=float(params.get("thor_max_leverage", getattr(ev, "thor_max_leverage", 5.0))),
                capital_cap=float(params.get("thor_capital_cap", getattr(ev, "thor_capital_cap", 0.5))),
                max_concurrent_positions=int(
                    params.get("thor_max_concurrent_positions", getattr(ev, "thor_max_concurrent_positions", 3))
                ),
                initial_capital=10_000.0,
                commission_bps=float(getattr(bt, "commission_bps", 4.0)),
                slippage_bps=float(getattr(bt, "slippage_bps", 2.0)),
                compound_mode=str(getattr(ev, "compound_mode", "asymmetric_target")),
                compound_max_loss_pct=float(getattr(ev, "compound_max_loss_pct", 3.0)),
                compound_activation_score=float(getattr(ev, "compound_activation_score", 85.0)),
            )
        )

    return fold_metrics


def evaluate_exit_params_full(
    prepared_symbols: list,
    params: dict,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Full-sample fast replay estimate for a fully specified Thor param set.
    This includes the non-cacheable exit params, unlike CachedTrialEvaluator.
    """
    trades = _collect_trade_arrays(prepared_symbols, params)
    return simulate_fast_thor_metrics(
        entry_ts=trades["entry_ts"],
        exit_ts=trades["exit_ts"],
        symbol=trades["symbol"],
        entry_price=trades["entry_price"],
        entry_atr=trades["entry_atr"],
        realized_atr=trades["realized_atr"],
        score=trades["score"],
        stop_atr=float(params.get("thor_sl_atr", 3.0)),
        risk_pct=float(params.get("thor_risk_pct", 0.005)),
        max_leverage=float(params.get("thor_max_leverage", getattr(Config.events, "thor_max_leverage", 5.0))),
        capital_cap=float(params.get("thor_capital_cap", getattr(Config.events, "thor_capital_cap", 0.5))),
        max_concurrent_positions=int(
            params.get("thor_max_concurrent_positions", getattr(Config.events, "thor_max_concurrent_positions", 3))
        ),
        initial_capital=float(initial_capital),
        commission_bps=float(getattr(Config.backtest, "commission_bps", 4.0)),
        slippage_bps=float(getattr(Config.backtest, "slippage_bps", 2.0)),
        compound_mode=str(getattr(Config.events, "compound_mode", "asymmetric_target")),
        compound_max_loss_pct=float(getattr(Config.events, "compound_max_loss_pct", 3.0)),
        compound_activation_score=float(getattr(Config.events, "compound_activation_score", 85.0)),
    )


def _build_fold_ts_bounds(
    prepared_symbols: list,
    fold_bounds_rows: list[tuple[int, int]],
    mae_csv_path: Optional[str] = None,
) -> list[tuple[int, int]]:
    """
    Convert row-index fold bounds to timestamp bounds using the MAE CSV entry order.
    """
    if mae_csv_path is not None:
        import pandas as pd

        df = pd.read_csv(mae_csv_path, usecols=["entry_ts"])
        df = df.sort_values("entry_ts").reset_index(drop=True)
        ts = df["entry_ts"].to_numpy(dtype=np.int64)
        bounds = []
        for lo, hi in fold_bounds_rows:
            lo_ts = int(ts[lo]) if lo < len(ts) else 0
            hi_ts = int(ts[hi - 1]) + 1 if hi - 1 < len(ts) else 2**62
            bounds.append((lo_ts, hi_ts))
        return bounds

    all_ts = []
    for prep in prepared_symbols:
        all_ts.extend(int(t) for t in prep.feature_ctx.times if t > 0)
    all_ts.sort()
    n = len(all_ts)
    if n == 0:
        return [(0, 2**62)] * len(fold_bounds_rows)

    result = []
    for lo_row, hi_row in fold_bounds_rows:
        lo_idx = min(lo_row, n - 1)
        hi_idx = min(hi_row, n - 1)
        result.append((all_ts[lo_idx], all_ts[hi_idx]))
    return result


def _combo_rank_key(score: float, fold_metrics: list[dict]) -> tuple[float, float, float]:
    growths = [float(m.get("growth_pct", -100.0)) for m in fold_metrics]
    expectancies = [float(m.get("expectancy", 0.0)) for m in fold_metrics]
    growth_med = float(np.median(np.asarray(growths, dtype=np.float64))) if growths else -100.0
    expectancy_mean = float(np.mean(np.asarray(expectancies, dtype=np.float64))) if expectancies else 0.0
    return float(score), growth_med, expectancy_mean


def fine_tune_exits(
    prepared_symbols: list,
    stage1_params: dict,
    evaluator,
    mae_csv_path: Optional[str] = None,
) -> dict:
    """
    Two-round grid sweep over bank/trail/time-cap params.
    Returns the best exit params to merge into stage1_params.
    """
    fold_ts_bounds = _build_fold_ts_bounds(prepared_symbols, evaluator._fold_bounds, mae_csv_path)

    round1_total = int(np.prod([len(v) for v in _ROUND1_GRID.values()]))
    _log(f"Round 1: bank_atr x bank_fraction x runner_trail ({round1_total} combos)")
    best_r1_key = (-1e9, -1e9, -1e9)
    round1_ranked: list[tuple[tuple[float, float, float], dict]] = []

    r1_keys = list(_ROUND1_GRID.keys())
    r1_values = list(_ROUND1_GRID.values())
    r1_combos = list(itertools.product(*r1_values))

    for ci, combo in enumerate(r1_combos, 1):
        trial_params = dict(stage1_params)
        for key, value in zip(r1_keys, combo):
            trial_params[key] = value
        fold_metrics = _evaluate_combo(prepared_symbols, trial_params, fold_ts_bounds)
        score = score_multifold(fold_metrics)
        rank_key = _combo_rank_key(score, fold_metrics)
        combo_params = {key: value for key, value in zip(r1_keys, combo)}
        round1_ranked.append((rank_key, combo_params))
        if rank_key > best_r1_key:
            best_r1_key = rank_key
        if ci % 12 == 0 or ci == len(r1_combos):
            _log(f"  {ci}/{len(r1_combos)} done  best_score={best_r1_key[0]:.4f}")

    round1_ranked.sort(key=lambda item: item[0], reverse=True)
    round1_seeds = [params for _rank, params in round1_ranked[:_ROUND1_KEEP]]
    if not round1_seeds:
        round1_seeds = [{}]

    round2_total = int(np.prod([len(v) for v in _ROUND2_GRID.values()]) * len(round1_seeds))
    _log(
        f"Round 2: trail_activate x max_bars over top-{len(round1_seeds)} R1 seeds "
        f"({round2_total} combos)"
    )
    best_r2_key = (-1e9, -1e9, -1e9)
    best_r2_params = {}

    r2_keys = list(_ROUND2_GRID.keys())
    r2_values = list(_ROUND2_GRID.values())
    r2_combos = list(itertools.product(*r2_values))

    combo_counter = 0
    for seed in round1_seeds:
        for combo in r2_combos:
            combo_counter += 1
            trial_params = dict(stage1_params)
            trial_params.update(seed)
            for key, value in zip(r2_keys, combo):
                trial_params[key] = value
            fold_metrics = _evaluate_combo(prepared_symbols, trial_params, fold_ts_bounds)
            score = score_multifold(fold_metrics)
            rank_key = _combo_rank_key(score, fold_metrics)
            if rank_key > best_r2_key:
                best_r2_key = rank_key
                best_r2_params = dict(seed)
                best_r2_params.update({key: value for key, value in zip(r2_keys, combo)})
            if combo_counter % 24 == 0 or combo_counter == round2_total:
                _log(f"  {combo_counter}/{round2_total} done  best_score={best_r2_key[0]:.4f}")

    _log(f"fine_tune done  best_exit_score={best_r2_key[0]:.4f}  params={best_r2_params}")
    return dict(best_r2_params)
