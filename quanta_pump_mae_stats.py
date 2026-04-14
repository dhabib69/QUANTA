"""
Pump-level MAE analysis for Norse Thor/Nike signals.

Outputs:
- norse_pump_mae_rows.csv
- norse_pump_drawdown_curves.csv
- norse_mae_stats.csv
- norse_stop_decision_matrix.csv
- NORSE_MAE_STATS_REPORT.md
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from quanta_config import Config
from QUANTA_ml_engine import build_offline_feature_replay_engine
from quanta_norse_agents import (
    _LBL_CODE_TO_STR,
    _LBL_SL,
    _LBL_TIMEOUT,
    _LBL_TP,
    _pump_path_analytics,
    _sweep_thor_exits_njit,
    build_thor_contexts,
    compute_pump_state,
    extract_baldur_signals,
    find_active_thor_context,
    score_baldur_warning,
    score_thor_signal,
)

try:
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    def _njit(*a, **kw):
        def _w(f): return f
        return a[0] if (len(a) == 1 and callable(a[0])) else _w
from quanta_norse_year_sim import _ensure_feature_positions, _load_windowed_cache, _prepare_symbol


ROOT = Path(__file__).resolve().parent
ROWS_CSV = ROOT / "norse_pump_mae_rows.csv"
CURVES_CSV = ROOT / "norse_pump_drawdown_curves.csv"
STATS_CSV = ROOT / "norse_mae_stats.csv"
DECISION_CSV = ROOT / "norse_stop_decision_matrix.csv"
REPORT_MD = ROOT / "NORSE_MAE_STATS_REPORT.md"

MAX_BARS = 72
HEATMAP_BARS = 36
SL_ATR_CANDIDATES = np.ascontiguousarray(np.array([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4, 3.0], dtype=np.float64))
# Reduced grid: 4×3×2×2 = 48 combos (was 4×3×3×4 = 144)
BANK_ATR_CANDIDATES = [2.4, 3.0, 3.6, 4.2]
BANK_FRACTION_CANDIDATES = [0.15, 0.25, 0.35]
TRAIL_ACTIVATE_CANDIDATES = [1.5, 2.0]
RUNNER_TRAIL_CANDIDATES = [2.0, 3.0]
PRE_BANK_CANDIDATES = [12, 18, 24]
POST_BANK_CANDIDATES = [24, 36, 48]
BUCKET_ORDER = ["<0", "0-1", "1-3", "3-5", "5-10", ">=10"]
DEFAULT_MAX_WORKERS = 4

# Pre-flatten combo arrays for vectorized Numba kernel (built lazily at runtime)
_COMBO_BANK_ATR: np.ndarray | None = None
_COMBO_BANK_FRAC: np.ndarray | None = None
_COMBO_TRAIL_ACT: np.ndarray | None = None
_COMBO_RUNNER_TRAIL: np.ndarray | None = None


def _build_combo_arrays() -> None:
    global _COMBO_BANK_ATR, _COMBO_BANK_FRAC, _COMBO_TRAIL_ACT, _COMBO_RUNNER_TRAIL
    if _COMBO_BANK_ATR is not None:
        return
    rows_b, rows_f, rows_t, rows_r = [], [], [], []
    for b in BANK_ATR_CANDIDATES:
        for f in BANK_FRACTION_CANDIDATES:
            for t in TRAIL_ACTIVATE_CANDIDATES:
                for r in RUNNER_TRAIL_CANDIDATES:
                    rows_b.append(b); rows_f.append(f)
                    rows_t.append(t); rows_r.append(r)
    _COMBO_BANK_ATR = np.ascontiguousarray(rows_b, dtype=np.float64)
    _COMBO_BANK_FRAC = np.ascontiguousarray(rows_f, dtype=np.float64)
    _COMBO_TRAIL_ACT = np.ascontiguousarray(rows_t, dtype=np.float64)
    _COMBO_RUNNER_TRAIL = np.ascontiguousarray(rows_r, dtype=np.float64)

@_njit(fastmath=True)
def _decision_matrix_symbol_njit(
    opens, highs, lows, closes, atrs,
    entry_bars,
    sl_atr_arr,          # (K,)
    combo_bank_atr,      # (C,)
    combo_bank_frac,     # (C,)
    combo_trail_act,     # (C,)
    combo_runner_trail,  # (C,)
    max_bars_pre_bank, max_bars_post_bank,
    penetration_factor, slip_cap_atr,
):
    """
    Run all C combos × K sl_atr stops × N entry bars for one symbol.
    Returns (C, K) arrays: tp_count, sl_count, timeout_count, realized_sum,
                           positive_sum, negative_sum, dd_sum, dd_max.
    Called once per symbol → eliminates 144 × N_entries Python call overhead.
    """
    C = len(combo_bank_atr)
    K = len(sl_atr_arr)
    N = len(entry_bars)
    n_bars = len(closes)

    tp_c    = np.zeros((C, K), dtype=np.int64)
    sl_c    = np.zeros((C, K), dtype=np.int64)
    to_c    = np.zeros((C, K), dtype=np.int64)
    r_sum   = np.zeros((C, K), dtype=np.float64)
    p_sum   = np.zeros((C, K), dtype=np.float64)
    neg_sum = np.zeros((C, K), dtype=np.float64)
    dd_sum  = np.zeros((C, K), dtype=np.float64)
    dd_max  = np.zeros((C, K), dtype=np.float64)

    for ci in range(C):
        bank_atr      = combo_bank_atr[ci]
        bank_fraction = combo_bank_frac[ci]
        trail_act     = combo_trail_act[ci]
        runner_trail  = combo_runner_trail[ci]

        for ni in range(N):
            i = entry_bars[ni]
            if i < 0 or i >= n_bars:
                continue
            entry  = closes[i]
            atr_i  = atrs[i]
            if atr_i <= 0.0:
                atr_i = 1e-12

            # Per-slot state for K stops
            done           = np.zeros(K, dtype=np.int64)
            banked         = np.zeros(K, dtype=np.int64)
            bank_gain      = np.zeros(K, dtype=np.float64)
            peak_cl        = np.full(K, entry)
            runner_pk      = np.full(K, entry)
            stop_px        = np.zeros(K, dtype=np.float64)
            runner_stop_px = np.zeros(K, dtype=np.float64)
            max_dd_k       = np.zeros(K, dtype=np.float64)

            for k in range(K):
                stop_px[k]        = entry - sl_atr_arr[k] * atr_i
                runner_stop_px[k] = stop_px[k]

            bars_in_trade = 0
            all_done = 0
            end_bar = min(i + max_bars_pre_bank + max_bars_post_bank, n_bars - 1)

            for b in range(i + 1, end_bar + 1):
                bars_in_trade += 1
                lo = lows[b]; hi = highs[b]; op = opens[b]; cl = closes[b]
                dd_from_entry = (entry - lo) / atr_i
                all_done = 1
                for k in range(K):
                    if done[k]:
                        continue
                    all_done = 0
                    # max drawdown tracking
                    if dd_from_entry > max_dd_k[k]:
                        max_dd_k[k] = dd_from_entry

                    if not banked[k]:
                        # Bank leg: hit bank target?
                        bank_target = entry + bank_atr * atr_i
                        if hi >= bank_target:
                            banked[k] = 1
                            bank_gain[k] = bank_atr * bank_fraction
                            peak_cl[k]   = max(peak_cl[k], cl)
                            runner_pk[k] = peak_cl[k]
                            runner_stop_px[k] = runner_pk[k] - runner_trail * atr_i
                        else:
                            # SL hit before bank?
                            eff_stop = stop_px[k]
                            if penetration_factor > 0.0:
                                eff_stop = stop_px[k] - penetration_factor * atr_i
                            if lo <= eff_stop:
                                slip_px = min(op - eff_stop, slip_cap_atr * atr_i)
                                exit_px = max(eff_stop - slip_px, lo)
                                realized = (exit_px - entry) / atr_i
                                r_sum[ci, k]   += realized
                                neg_sum[ci, k] += realized
                                dd_sum[ci, k]  += max_dd_k[k]
                                if max_dd_k[k] > dd_max[ci, k]:
                                    dd_max[ci, k] = max_dd_k[k]
                                sl_c[ci, k] += 1
                                done[k] = 1
                            # timeout pre-bank?
                            elif bars_in_trade >= max_bars_pre_bank:
                                realized = (cl - entry) / atr_i
                                r_sum[ci, k]   += realized
                                if realized >= 0.0:
                                    p_sum[ci, k] += realized
                                else:
                                    neg_sum[ci, k] += realized
                                dd_sum[ci, k] += max_dd_k[k]
                                if max_dd_k[k] > dd_max[ci, k]:
                                    dd_max[ci, k] = max_dd_k[k]
                                to_c[ci, k] += 1
                                done[k] = 1
                    else:
                        # Runner leg
                        peak_cl[k]   = max(peak_cl[k], cl)
                        runner_pk[k] = max(runner_pk[k], cl)
                        if trail_act > 0.0:
                            new_stop = runner_pk[k] - runner_trail * atr_i
                            if new_stop > runner_stop_px[k]:
                                runner_stop_px[k] = new_stop
                        eff_runner = runner_stop_px[k]
                        if penetration_factor > 0.0:
                            eff_runner = runner_stop_px[k] - penetration_factor * atr_i
                        if lo <= eff_runner:
                            slip_px = min(op - eff_runner, slip_cap_atr * atr_i)
                            exit_px = max(eff_runner - slip_px, lo)
                            realized = bank_gain[k] + (exit_px - entry) * (1.0 - bank_fraction) / atr_i
                            r_sum[ci, k] += realized
                            if realized >= 0.0:
                                p_sum[ci, k] += realized
                            else:
                                neg_sum[ci, k] += realized
                            dd_sum[ci, k] += max_dd_k[k]
                            if max_dd_k[k] > dd_max[ci, k]:
                                dd_max[ci, k] = max_dd_k[k]
                            tp_c[ci, k] += 1
                            done[k] = 1
                        elif bars_in_trade >= max_bars_pre_bank + max_bars_post_bank:
                            realized = bank_gain[k] + (cl - entry) * (1.0 - bank_fraction) / atr_i
                            r_sum[ci, k] += realized
                            if realized >= 0.0:
                                p_sum[ci, k] += realized
                            else:
                                neg_sum[ci, k] += realized
                            dd_sum[ci, k] += max_dd_k[k]
                            if max_dd_k[k] > dd_max[ci, k]:
                                dd_max[ci, k] = max_dd_k[k]
                            to_c[ci, k] += 1
                            done[k] = 1

                if all_done:
                    break

            # Flush any still-open slots at end_bar
            for k in range(K):
                if not done[k]:
                    cl_last = closes[min(end_bar, n_bars - 1)]
                    if banked[k]:
                        realized = bank_gain[k] + (cl_last - entry) * (1.0 - bank_fraction) / atr_i
                    else:
                        realized = (cl_last - entry) / atr_i
                    r_sum[ci, k] += realized
                    if realized >= 0.0:
                        p_sum[ci, k] += realized
                    else:
                        neg_sum[ci, k] += realized
                    dd_sum[ci, k] += max_dd_k[k]
                    if max_dd_k[k] > dd_max[ci, k]:
                        dd_max[ci, k] = max_dd_k[k]
                    to_c[ci, k] += 1

    return tp_c, sl_c, to_c, r_sum, p_sum, neg_sum, dd_sum, dd_max


_MAE_WORKER_ENGINE = None
_MAE_WORKER_LOCK = None


def _tag_float(value: float) -> str:
    return str(value).replace(".", "p")


def _safe_quantile(values: pd.Series, q: float) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def _safe_mean(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _safe_pf(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return 0.0
    wins = float(clean[clean > 0].sum())
    losses = float(clean[clean < 0].sum())
    if losses == 0.0:
        return float("inf") if wins > 0 else 0.0
    return wins / abs(losses)


def _bars_to_threshold(series: np.ndarray, threshold: float) -> float:
    for idx, value in enumerate(series):
        if float(value) >= threshold:
            return float(idx)
    return np.nan


def _value_at(series: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(series):
        return np.nan
    return float(series[index])


def _outcome_bucket(max_runup_atr: float) -> str:
    if max_runup_atr < 0.0:
        return "<0"
    if max_runup_atr < 1.0:
        return "0-1"
    if max_runup_atr < 3.0:
        return "1-3"
    if max_runup_atr < 5.0:
        return "3-5"
    if max_runup_atr < 10.0:
        return "5-10"
    return ">=10"


def _score_band(score: float) -> str:
    if score >= 99.0:
        return "99+"
    if score >= 95.0:
        return "95-99"
    if score >= 90.0:
        return "90-95"
    if score >= 80.0:
        return "80-90"
    return "<80"


def _prior_gap_bucket(gap_bars: float) -> str:
    if np.isnan(gap_bars):
        return "solo"
    return "chain" if gap_bars <= 48 else "solo"


def _regime_bucket(regime_value: float) -> str:
    if not np.isfinite(regime_value):
        return "na"
    if regime_value >= 0.75:
        return "bull"
    if regime_value >= 0.50:
        return "range-up"
    if regime_value >= 0.25:
        return "range-down"
    return "bear"


def _nearest_choice(value: float, choices: list[float]) -> float:
    if not np.isfinite(value):
        return float(choices[0])
    return float(min(choices, key=lambda choice: (abs(choice - value), choice)))


def _format_cell(value) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.2f}"
    if pd.isna(value):
        return "n/a"
    return str(value)


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_none_"
    view = df if max_rows is None else df.head(max_rows)
    cols = list(view.columns)
    display_cols = [str(col) for col in cols]
    lines = [
        "| " + " | ".join(display_cols) + " |",
        "| " + " | ".join(["---"] * len(display_cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _build_warning_map(prep, warnings: list[dict] | None = None) -> dict[str, dict]:
    if warnings is None:
        warnings, _ = extract_baldur_signals(prep.df, prep.raw_thor_signals, Config.events)
    if not warnings:
        return {}
    ctx = prep.feature_ctx
    contexts = build_thor_contexts(
        prep.raw_thor_signals,
        int(getattr(Config.events, "thor_context_bars", 24)),
        0.0,
    )
    mapped: dict[str, dict] = {}
    for warning in warnings:
        thor_ctx = find_active_thor_context(contexts, int(warning["bar_idx"]))
        if thor_ctx is None:
            continue
        pump_id = f"{prep.symbol}:{thor_ctx.start_bar}"
        warning_bar = int(warning["bar_idx"])
        exit_bar = min(warning_bar + 1, len(ctx.opens) - 1)
        row = {
            "pump_id": pump_id,
            "warning_bar": warning_bar,
            "warning_score": float(score_baldur_warning(warning, ctx, thor_ctx, Config.events)),
            "warning_exit_bar": exit_bar,
            "warning_exit_price": float(ctx.opens[exit_bar]),
        }
        current = mapped.get(pump_id)
        if current is None or warning_bar < current["warning_bar"]:
            mapped[pump_id] = row
    return mapped


def _value_at_or_last(series: np.ndarray, index: int) -> float:
    if len(series) == 0:
        return np.nan
    if index < 0:
        return np.nan
    return float(series[min(index, len(series) - 1)])


def _init_mae_worker() -> None:
    global _MAE_WORKER_ENGINE
    if _MAE_WORKER_ENGINE is None:
        _MAE_WORKER_ENGINE = build_offline_feature_replay_engine(Config)


def _process_symbol_mae(task: tuple[int, str, pd.DataFrame]) -> dict[str, object]:
    _init_mae_worker()
    idx, symbol, df = task
    ev = Config.events
    local_df = df.copy()
    local_df.attrs["symbol"] = symbol
    prep = _prepare_symbol(symbol, local_df, _MAE_WORKER_ENGINE, ev)
    if not prep.raw_thor_signals:
        return {
            "index": idx,
            "rows": [],
            "curves": [],
            "decision_payload": None,
        }

    warnings, _ = extract_baldur_signals(prep.df, prep.raw_thor_signals, Config.events)
    # Only compute features at entry bars + 6-bar lookback — forward_fill handles the window.
    # OHLCV-derived metrics (runup/drawdown series) are exact regardless.
    needed_positions: set[int] = set()
    for sig in prep.raw_thor_signals:
        i = int(sig["bar_idx"])
        needed_positions.update(range(max(50, i - 6), min(i + 1, len(prep.df))))
    for warning in warnings:
        needed_positions.add(int(warning["bar_idx"]))
    _ensure_feature_positions(prep, needed_positions)

    warning_map = _build_warning_map(prep, warnings=warnings)
    ctx = prep.feature_ctx
    row_records: list[dict] = []
    curve_records: list[dict] = []
    entry_bars: list[int] = []
    prev_signal_bar = np.nan

    for sig in prep.raw_thor_signals:
        i = int(sig["bar_idx"])
        end_bar = min(i + MAX_BARS, len(prep.df) - 1)
        pump_id = f"{prep.symbol}:{i}"
        analytics = _pump_path_analytics(
            ctx,
            i,
            end_bar,
            material_drawdown_atr=float(ev.pump_material_drawdown_atr),
        )
        entry_price = float(analytics["entry_price"])
        atr0 = float(analytics["atr0"])
        peak_bar = int(analytics["peak_bar"])
        max_runup_atr = float(analytics["max_runup_atr"])
        max_mae_from_entry_atr = float(analytics["max_mae_from_entry_atr"])
        max_mae_from_peak_atr = float(analytics["max_mae_from_peak_atr"])
        max_mae_entry_bar = int(analytics["max_mae_entry_bar"])
        runup_series = analytics["runup_series"]
        drawdown_peak_series = analytics["drawdown_peak_series"]
        drawdown_entry_series = analytics["drawdown_entry_series"]
        close_delta_series = analytics["close_delta_series"]
        wave_strength_series = analytics["wave_strength_series"]
        top_risk_series = analytics["top_risk_series"]
        close_drawdown_series = analytics["close_drawdown_series"]
        bars_since_peak_series = analytics["bars_since_peak_series"]
        times = analytics["times"]

        warning = warning_map.get(pump_id)
        warning_bar = np.nan if warning is None else float(warning["warning_bar"])
        warning_delay = np.nan if warning is None else float(warning["warning_bar"] - peak_bar)
        warning_runup = np.nan
        if warning is not None:
            warning_runup = (float(warning["warning_exit_price"]) - entry_price) / max(atr0, 1e-12)

        entry_state = compute_pump_state(
            ctx,
            max(0, i - 6),
            i,
            float(ev.pump_material_drawdown_atr),
        )

        label_codes, realized_arr, exit_bar_arr, _exit_price_arr, slip_arr, _bank_arr, max_dd_arr = _sweep_thor_exits_njit(
            ctx.opens,
            ctx.highs,
            ctx.lows,
            ctx.closes,
            ctx.volume_ratio,
            ctx.quote_volume_slope,
            ctx.taker_slope,
            ctx.weighted_trend,
            ctx.bs_prob,
            ctx.participation_score,
            ctx.flow_exhaustion_score,
            ctx.close_pos,
            ctx.upper_wick_ratio,
            i,
            entry_price,
            atr0,
            SL_ATR_CANDIDATES,
            float(ev.thor_bank_atr),
            float(ev.thor_bank_fraction),
            float(ev.thor_trail_activate_atr),
            float(ev.thor_runner_trail_atr),
            int(ev.thor_max_bars_pre_bank),
            int(ev.thor_max_bars_post_bank),
            float(ev.stop_market_penetration_factor),
            float(ev.stop_market_slip_atr_cap),
            float(ev.pump_material_drawdown_atr),
        )

        for offset_i in range(len(times)):
            bar = i + offset_i
            curve_records.append(
                {
                    "pump_id": pump_id,
                    "symbol": prep.symbol,
                    "bars_since_entry": offset_i,
                    "drawdown_atr": float(close_drawdown_series[offset_i]),
                    "runup_atr": float(runup_series[offset_i]),
                    "ts": int(times[offset_i]),
                }
            )

        ts = pd.to_datetime(int(ctx.times[i]), unit="ms", utc=True)
        prior_gap_bars = np.nan if not np.isfinite(prev_signal_bar) else float(i - prev_signal_bar)
        thor_feature_score = float(score_thor_signal(sig, ctx))
        row = {
            "pump_id": pump_id,
            "symbol": prep.symbol,
            "entry_ts": int(ctx.times[i]),
            "entry_bar": i,
            "tier": str(sig.get("tier", "")),
            "nike_score": float(sig.get("score", 0.0)),
            "thor_feature_score": thor_feature_score,
            "confidence": float(sig.get("confidence", sig.get("score", 0.0))),
            "entry_mode": str(sig.get("entry_mode", sig.get("nike_entry_mode", ""))),
            "bs_floor": float(sig.get("bs_floor", sig.get("nike_bs_floor", np.nan))),
            "entry_price": entry_price,
            "entry_atr": atr0,
            "entry_quote_volume": float(ctx.quote_volume[i]),
            "entry_volume_ratio": float(ctx.volume_ratio[i]),
            "day_of_week": int(ts.dayofweek),
            "day_of_week_utc": int(ts.dayofweek),
            "hour_of_day_utc": int(ts.hour),
            "hour_of_week": int(ts.dayofweek * 24 + ts.hour),
            "regime_state": float(ctx.regime_state[i]),
            "weighted_trend": float(ctx.weighted_trend[i]),
            "bs_prob": float(ctx.bs_prob[i]),
            "participation_score": float(ctx.participation_score[i]),
            "flow_exhaustion_score": float(ctx.flow_exhaustion_score[i]),
            "prior_pump_gap_bars": prior_gap_bars,
            "max_runup_atr": max_runup_atr,
            "max_runup_pct": max_runup_atr * atr0 / max(entry_price, 1e-12) * 100.0,
            "time_to_peak_bars": int(analytics["time_to_peak_bars"]),
            "peak_bar_idx": peak_bar,
            "max_mae_from_entry_atr": max_mae_from_entry_atr,
            "max_mae_from_entry_bar": max_mae_entry_bar,
            "max_mae_from_peak_atr": max_mae_from_peak_atr,
            "bars_to_first_1atr_runup": _bars_to_threshold(runup_series, 1.0),
            "bars_to_first_3atr_runup": _bars_to_threshold(runup_series, 3.0),
            "bars_to_first_5atr_runup": _bars_to_threshold(runup_series, 5.0),
            "bars_to_first_1atr_drawdown": _bars_to_threshold(drawdown_entry_series, 1.0),
            "bars_to_first_2atr_drawdown": _bars_to_threshold(drawdown_entry_series, 2.0),
            "mfe_atr": max_runup_atr,
            "mae_to_mfe_ratio": np.nan if max_runup_atr <= 0 else max_mae_from_entry_atr / max_runup_atr,
            "close_at_bar_6": _value_at(close_delta_series, 6),
            "close_at_bar_12": _value_at(close_delta_series, 12),
            "close_at_bar_24": _value_at(close_delta_series, 24),
            "close_at_bar_36": _value_at(close_delta_series, 36),
            "close_at_bar_72": _value_at(close_delta_series, 72),
            "drawdown_at_bar_1": _value_at(drawdown_peak_series, 1),
            "drawdown_at_bar_2": _value_at(drawdown_peak_series, 2),
            "drawdown_at_bar_3": _value_at(drawdown_peak_series, 3),
            "drawdown_at_bar_6": _value_at(drawdown_peak_series, 6),
            "drawdown_at_bar_12": _value_at(drawdown_peak_series, 12),
            "drawdown_at_bar_24": _value_at(drawdown_peak_series, 24),
            "drawdown_at_bar_36": _value_at(drawdown_peak_series, 36),
            "drawdown_at_bar_72": _value_at(drawdown_peak_series, 72),
            "runup_at_bar_1": _value_at(runup_series, 1),
            "runup_at_bar_2": _value_at(runup_series, 2),
            "runup_at_bar_3": _value_at(runup_series, 3),
            "runup_at_bar_6": _value_at(runup_series, 6),
            "runup_at_bar_12": _value_at(runup_series, 12),
            "runup_at_bar_24": _value_at(runup_series, 24),
            "runup_at_bar_36": _value_at(runup_series, 36),
            "runup_at_bar_72": _value_at(runup_series, 72),
            "volume_ratio_at_entry": float(ctx.volume_ratio[i]),
            "volume_ratio_at_peak": float(ctx.volume_ratio[min(peak_bar, len(ctx.volume_ratio) - 1)]),
            "volume_ratio_at_bar_12": _value_at(ctx.volume_ratio[i:end_bar + 1], 12),
            "volume_ratio_decay_peak_to_exit": float(analytics["volume_decay_after_peak"]),
            "quote_volume_slope_at_entry": float(ctx.quote_volume_slope[i]),
            "quote_volume_slope_at_peak": float(ctx.quote_volume_slope[min(peak_bar, len(ctx.quote_volume_slope) - 1)]),
            "taker_imbalance_at_entry": float(ctx.taker_imbalance[i]),
            "taker_imbalance_at_peak": float(ctx.taker_imbalance[min(peak_bar, len(ctx.taker_imbalance) - 1)]),
            "taker_slope_at_entry": float(ctx.taker_slope[i]),
            "taker_slope_at_peak": float(ctx.taker_slope[min(peak_bar, len(ctx.taker_slope) - 1)]),
            "vpin_at_entry": float(ctx.vpin[i]),
            "vpin_at_peak": float(ctx.vpin[min(peak_bar, len(ctx.vpin) - 1)]),
            "vpin_slope_at_entry": float(ctx.vpin_slope[i]),
            "participation_score_at_entry": float(ctx.participation_score[i]),
            "participation_score_at_peak": float(ctx.participation_score[min(peak_bar, len(ctx.participation_score) - 1)]),
            "flow_exhaustion_score_at_entry": float(ctx.flow_exhaustion_score[i]),
            "flow_exhaustion_score_at_peak": float(ctx.flow_exhaustion_score[min(peak_bar, len(ctx.flow_exhaustion_score) - 1)]),
            "wave_strength_score_at_entry": float(entry_state["wave_strength_score"]),
            "wave_strength_score_at_peak": float(wave_strength_series[min(max(peak_bar - i, 0), len(wave_strength_series) - 1)]),
            "wave_strength_score_at_bar_12": _value_at_or_last(wave_strength_series, 12),
            "top_risk_score_at_entry": float(entry_state["top_risk_score"]),
            "top_risk_score_at_peak": float(top_risk_series[min(max(peak_bar - i, 0), len(top_risk_series) - 1)]),
            "top_risk_score_at_bar_12": _value_at_or_last(top_risk_series, 12),
            "upper_wick_ratio_at_entry": float(ctx.upper_wick_ratio[i]),
            "upper_wick_ratio_at_peak_bar": float(ctx.upper_wick_ratio[min(peak_bar, len(ctx.upper_wick_ratio) - 1)]),
            "max_upper_wick_ratio_post_entry": float(ctx.upper_wick_ratio[i:end_bar + 1].max()),
            "close_pos_at_entry": float(ctx.close_pos[i]),
            "close_pos_at_bar_6": _value_at(ctx.close_pos[i:end_bar + 1], 6),
            "close_pos_at_peak": float(ctx.close_pos[min(peak_bar, len(ctx.close_pos) - 1)]),
            "impulse_body_eff_at_entry": float(ctx.impulse_body_eff[i]),
            "impulse_taker_persist_at_entry": float(ctx.impulse_taker_persist[i]),
            "pre_impulse_r2_at_entry": float(ctx.pre_impulse_r2[i]),
            "atr_rank_at_entry": float(ctx.atr_rank[i]),
            "reversed_within_6_bars": bool(
                (np.nanmax(runup_series[:4]) if len(runup_series) else 0.0) >= 1.0
                and _value_at(close_delta_series, 6) <= -0.5
            ),
            "round_trip_within_12_bars": bool(
                (np.nanmax(runup_series[:13]) if len(runup_series) else 0.0) >= 1.0
                and _value_at(close_delta_series, 12) <= 0.0
            ),
            "chain_pump_flag": bool(_prior_gap_bucket(prior_gap_bars) == "chain"),
            "pre_entry_mae_atr": float(entry_state["max_drawdown_atr"]),
            "baldur_warning_fired_bar": warning_bar,
            "baldur_warning_score": float("nan") if warning is None else float(warning.get("warning_score", float("nan"))),
            "baldur_warning_delay_from_peak": warning_delay,
            "baldur_exit_runup_atr": warning_runup,
            "outcome_bucket": _outcome_bucket(max_runup_atr),
            "score_band": _score_band(thor_feature_score),
            "regime_bucket": _regime_bucket(float(ctx.regime_state[i])),
            "prior_gap_bucket": _prior_gap_bucket(prior_gap_bars),
        }

        for sl_idx, sl_atr in enumerate(SL_ATR_CANDIDATES.tolist()):
            tag = _tag_float(sl_atr)
            row[f"outcome_sl_{tag}"] = _LBL_CODE_TO_STR.get(int(label_codes[sl_idx]), "TIMEOUT")
            row[f"realized_atr_sl_{tag}"] = float(realized_arr[sl_idx])
            row[f"exit_bar_sl_{tag}"] = int(exit_bar_arr[sl_idx])
            row[f"stop_slip_atr_sl_{tag}"] = float(slip_arr[sl_idx])
            row[f"max_drawdown_during_trade_sl_{tag}"] = float(max_dd_arr[sl_idx])

        row_records.append(row)
        entry_bars.append(i)
        prev_signal_bar = float(i)

    decision_payload = None
    if entry_bars:
        decision_payload = {
            "opens": np.ascontiguousarray(ctx.opens, dtype=np.float64),
            "highs": np.ascontiguousarray(ctx.highs, dtype=np.float64),
            "lows": np.ascontiguousarray(ctx.lows, dtype=np.float64),
            "closes": np.ascontiguousarray(ctx.closes, dtype=np.float64),
            "atrs": np.ascontiguousarray(ctx.atrs, dtype=np.float64),
            "entry_bars": np.asarray(entry_bars, dtype=np.int64),
        }

    return {
        "index": idx,
        "rows": row_records,
        "curves": curve_records,
        "decision_payload": decision_payload,
    }


def _extend_symbol_rows(task: tuple[int, str, pd.DataFrame, pd.DataFrame]) -> dict[str, object]:
    idx, symbol, df, rows_sub = task
    ev = Config.events
    prep = None
    try:
        from norse_event_cache_loader import load_cached_symbol
        prep = load_cached_symbol(symbol)
    except Exception:
        prep = None
    loaded_from_cache = prep is not None
    if prep is None:
        _init_mae_worker()
        local_df = df.copy()
        local_df.attrs["symbol"] = symbol
        prep = _prepare_symbol(symbol, local_df, _MAE_WORKER_ENGINE, ev)
    if rows_sub.empty or not prep.raw_thor_signals:
        return {"index": idx, "updates": []}

    warnings, _ = extract_baldur_signals(prep.df, prep.raw_thor_signals, Config.events)
    needed_positions: set[int] = set()
    for entry_bar in rows_sub["entry_bar"].tolist():
        i = int(entry_bar)
        needed_positions.update(range(max(50, i - 6), min(i + 1, len(prep.df))))
    for warning in warnings:
        needed_positions.add(int(warning["bar_idx"]))
    if not loaded_from_cache:
        _ensure_feature_positions(prep, needed_positions)

    warning_map = _build_warning_map(prep, warnings=warnings)
    ctx = prep.feature_ctx
    signal_by_pump_id = {
        f"{symbol}:{int(sig['bar_idx'])}": sig
        for sig in prep.raw_thor_signals
    }
    updates: list[dict] = []
    for row in rows_sub.itertuples(index=False):
        i = int(row.entry_bar)
        if i < 0 or i >= len(prep.df):
            continue
        entry_state = compute_pump_state(
            ctx,
            max(0, i - 6),
            i,
            float(ev.pump_material_drawdown_atr),
        )
        warning = warning_map.get(str(row.pump_id))
        sig = signal_by_pump_id.get(str(row.pump_id))
        thor_feature_score = float(score_thor_signal(sig, ctx)) if sig is not None else np.nan
        updates.append(
            {
                "row_index": int(row.row_index),
                "thor_feature_score": thor_feature_score,
                "score_band": _score_band(thor_feature_score) if np.isfinite(thor_feature_score) else row.score_band,
                "pre_entry_mae_atr": float(entry_state["max_drawdown_atr"]),
                "baldur_warning_score": (
                    float("nan")
                    if warning is None
                    else float(warning.get("warning_score", float("nan")))
                ),
            }
        )
    return {"index": idx, "updates": updates}


def extend_existing_rows_csv(
    days: int = 365,
    max_symbols: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    t0 = time.time()
    if not ROWS_CSV.exists():
        raise FileNotFoundError(f"Missing existing rows CSV: {ROWS_CSV}")

    rows_df = pd.read_csv(ROWS_CSV)
    if rows_df.empty:
        rows_df.to_csv(ROWS_CSV, index=False)
        print("[mae-extend] rows CSV is empty; nothing to extend", flush=True)
        return rows_df

    if "pump_id" not in rows_df.columns or "entry_bar" not in rows_df.columns or "symbol" not in rows_df.columns:
        raise ValueError("Existing rows CSV is missing required columns: pump_id, symbol, entry_bar")

    working_df = rows_df.reset_index().rename(columns={"index": "row_index"})
    universe = _load_windowed_cache(days=days)
    rows_by_symbol = {
        symbol: sub.loc[:, [col for col in ["row_index", "pump_id", "entry_bar", "score_band"] if col in sub.columns]].copy()
        for symbol, sub in working_df.groupby("symbol", sort=False)
    }
    universe = [(idx, symbol, df) for idx, (symbol, df) in enumerate(universe) if symbol in rows_by_symbol]
    if max_symbols > 0:
        universe = universe[:max_symbols]
    print(f"[mae-extend] {len(universe)} symbols loaded ({days}d cache)", flush=True)

    _init_mae_worker()
    completed = [0]
    n_tasks = len(universe)

    def _run_task(task: tuple[int, str, pd.DataFrame]) -> dict[str, object]:
        idx, symbol, df = task
        result = _extend_symbol_rows((idx, symbol, df, rows_by_symbol[symbol]))
        completed[0] += 1
        return result

    if n_tasks == 0:
        raise ValueError("No symbols from the existing rows CSV matched the current cache window")

    from tqdm import tqdm
    n_workers = min(max_workers, n_tasks)
    if n_workers <= 1:
        results = [_run_task(task) for task in tqdm(universe, total=n_tasks, desc="[mae-extend] Processing")]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(_run_task, universe), total=n_tasks, desc="[mae-extend] Processing"))

    if "pre_entry_mae_atr" not in working_df.columns:
        working_df["pre_entry_mae_atr"] = np.nan
    if "baldur_warning_score" not in working_df.columns:
        working_df["baldur_warning_score"] = np.nan
    if "thor_feature_score" not in working_df.columns:
        working_df["thor_feature_score"] = np.nan

    n_updates = 0
    for result in results:
        for update in result["updates"]:
            row_index = int(update["row_index"])
            working_df.at[row_index, "thor_feature_score"] = float(update["thor_feature_score"])
            working_df.at[row_index, "score_band"] = str(update["score_band"])
            working_df.at[row_index, "pre_entry_mae_atr"] = float(update["pre_entry_mae_atr"])
            working_df.at[row_index, "baldur_warning_score"] = float(update["baldur_warning_score"])
            n_updates += 1

    working_df = working_df.drop(columns=["row_index"])
    working_df.to_csv(ROWS_CSV, index=False)
    print(f"[mae-extend] total {time.time()-t0:.0f}s  updated_rows={n_updates}  wrote {ROWS_CSV.name}", flush=True)
    return working_df


def _bucket_stats(rows_df: pd.DataFrame, baseline_tag: str) -> pd.DataFrame:
    realized_col = f"realized_atr_sl_{baseline_tag}"
    outcome_col = f"outcome_sl_{baseline_tag}"
    bucket_specs = [
        ("overall", None),
        ("outcome_bucket", "outcome_bucket"),
        ("tier", "tier"),
        ("score_band", "score_band"),
        ("hour_of_day_utc", "hour_of_day_utc"),
        ("day_of_week_utc", "day_of_week_utc"),
        ("regime_bucket", "regime_bucket"),
        ("prior_gap_bucket", "prior_gap_bucket"),
    ]
    records: list[dict] = []
    for bucket_type, column in bucket_specs:
        if column is None:
            groups = [("all", rows_df)]
        else:
            groups = rows_df.groupby(column, dropna=False)
        for bucket_value, sub in groups:
            realized = sub[realized_col]
            outcomes = sub[outcome_col]
            records.append(
                {
                    "bucket_type": bucket_type,
                    "bucket_value": bucket_value,
                    "count": int(len(sub)),
                    "tp_count": int((outcomes == "TP").sum()),
                    "sl_count": int((outcomes == "SL").sum()),
                    "timeout_count": int((outcomes == "TIMEOUT").sum()),
                    "weighted_pf": _safe_pf(realized),
                    "expectancy_atr": _safe_mean(realized),
                    "avg_realized_atr": _safe_mean(realized),
                    "max_mae_entry_mean": _safe_mean(sub["max_mae_from_entry_atr"]),
                    "max_mae_entry_p25": _safe_quantile(sub["max_mae_from_entry_atr"], 0.25),
                    "max_mae_entry_p50": _safe_quantile(sub["max_mae_from_entry_atr"], 0.50),
                    "max_mae_entry_p75": _safe_quantile(sub["max_mae_from_entry_atr"], 0.75),
                    "max_mae_entry_p90": _safe_quantile(sub["max_mae_from_entry_atr"], 0.90),
                    "max_mae_entry_p95": _safe_quantile(sub["max_mae_from_entry_atr"], 0.95),
                    "max_mae_entry_p99": _safe_quantile(sub["max_mae_from_entry_atr"], 0.99),
                    "max_mae_entry_max": float(sub["max_mae_from_entry_atr"].max()) if len(sub) else np.nan,
                    "max_mae_peak_mean": _safe_mean(sub["max_mae_from_peak_atr"]),
                    "max_mae_peak_p25": _safe_quantile(sub["max_mae_from_peak_atr"], 0.25),
                    "max_mae_peak_p50": _safe_quantile(sub["max_mae_from_peak_atr"], 0.50),
                    "max_mae_peak_p75": _safe_quantile(sub["max_mae_from_peak_atr"], 0.75),
                    "max_mae_peak_p90": _safe_quantile(sub["max_mae_from_peak_atr"], 0.90),
                    "max_mae_peak_p95": _safe_quantile(sub["max_mae_from_peak_atr"], 0.95),
                    "max_runup_mean": _safe_mean(sub["max_runup_atr"]),
                    "max_runup_p50": _safe_quantile(sub["max_runup_atr"], 0.50),
                    "max_runup_p75": _safe_quantile(sub["max_runup_atr"], 0.75),
                    "max_runup_p90": _safe_quantile(sub["max_runup_atr"], 0.90),
                    "max_runup_p95": _safe_quantile(sub["max_runup_atr"], 0.95),
                    "max_runup_max": float(sub["max_runup_atr"].max()) if len(sub) else np.nan,
                    "time_to_peak_mean": _safe_mean(sub["time_to_peak_bars"]),
                    "time_to_peak_median": _safe_quantile(sub["time_to_peak_bars"], 0.50),
                    "time_to_peak_p90": _safe_quantile(sub["time_to_peak_bars"], 0.90),
                    "bars_to_1atr_runup_mean": _safe_mean(sub["bars_to_first_1atr_runup"]),
                    "bars_to_1atr_runup_median": _safe_quantile(sub["bars_to_first_1atr_runup"], 0.50),
                    "mae_to_mfe_ratio_mean": _safe_mean(sub["mae_to_mfe_ratio"]),
                    "mae_to_mfe_ratio_median": _safe_quantile(sub["mae_to_mfe_ratio"], 0.50),
                    "volume_decay_mean": _safe_mean(sub["volume_ratio_decay_peak_to_exit"]),
                    "volume_decay_median": _safe_quantile(sub["volume_ratio_decay_peak_to_exit"], 0.50),
                    "reversed_within_6_bars_rate": float(sub["reversed_within_6_bars"].mean() * 100.0) if len(sub) else np.nan,
                    "round_trip_within_12_bars_rate": float(sub["round_trip_within_12_bars"].mean() * 100.0) if len(sub) else np.nan,
                    "chain_pump_flag_rate": float(sub["chain_pump_flag"].mean() * 100.0) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(records)


def _decision_matrix(decision_payloads: list[dict], ev) -> pd.DataFrame:
    """
    Vectorized decision matrix using _decision_matrix_symbol_njit.
    Calls the Numba kernel once per symbol (not once per combo × entry bar).
    """
    _build_combo_arrays()
    sweep_pre_bank  = int(_nearest_choice(float(ev.thor_max_bars_pre_bank),  PRE_BANK_CANDIDATES))
    sweep_post_bank = int(_nearest_choice(float(ev.thor_max_bars_post_bank), POST_BANK_CANDIDATES))
    pf = float(ev.stop_market_penetration_factor)
    sc = float(ev.stop_market_slip_atr_cap)

    C = len(_COMBO_BANK_ATR)
    K = len(SL_ATR_CANDIDATES)
    trade_count = int(sum(len(p["entry_bars"]) for p in decision_payloads))

    tp_total   = np.zeros((C, K), dtype=np.int64)
    sl_total   = np.zeros((C, K), dtype=np.int64)
    to_total   = np.zeros((C, K), dtype=np.int64)
    r_total    = np.zeros((C, K), dtype=np.float64)
    p_total    = np.zeros((C, K), dtype=np.float64)
    neg_total  = np.zeros((C, K), dtype=np.float64)
    dd_sum_tot = np.zeros((C, K), dtype=np.float64)
    dd_max_tot = np.zeros((C, K), dtype=np.float64)

    for payload in decision_payloads:
        entry_bars_i64 = np.ascontiguousarray(payload["entry_bars"], dtype=np.int64)
        tp_c, sl_c, to_c, r_s, p_s, n_s, dd_s, dd_m = _decision_matrix_symbol_njit(
            np.ascontiguousarray(payload["opens"],  dtype=np.float64),
            np.ascontiguousarray(payload["highs"],  dtype=np.float64),
            np.ascontiguousarray(payload["lows"],   dtype=np.float64),
            np.ascontiguousarray(payload["closes"], dtype=np.float64),
            np.ascontiguousarray(payload["atrs"],   dtype=np.float64),
            entry_bars_i64,
            SL_ATR_CANDIDATES,
            _COMBO_BANK_ATR, _COMBO_BANK_FRAC, _COMBO_TRAIL_ACT, _COMBO_RUNNER_TRAIL,
            sweep_pre_bank, sweep_post_bank, pf, sc,
        )
        tp_total   += tp_c;  sl_total   += sl_c;  to_total   += to_c
        r_total    += r_s;   p_total    += p_s;   neg_total  += n_s
        dd_sum_tot += dd_s
        np.maximum(dd_max_tot, dd_m, out=dd_max_tot)

    records: list[dict] = []
    for ci in range(C):
        for ki, sl_atr in enumerate(SL_ATR_CANDIDATES.tolist()):
            losses = float(neg_total[ci, ki])
            wins   = float(p_total[ci, ki])
            pf_val = float("inf") if losses == 0.0 and wins > 0 else (0.0 if losses == 0.0 else wins / abs(losses))
            records.append({
                "sl_atr":            float(sl_atr),
                "bank_atr":          float(_COMBO_BANK_ATR[ci]),
                "bank_fraction":     float(_COMBO_BANK_FRAC[ci]),
                "trail_activate_atr": float(_COMBO_TRAIL_ACT[ci]),
                "runner_trail_atr":  float(_COMBO_RUNNER_TRAIL[ci]),
                "count":             trade_count,
                "tp_count":          int(tp_total[ci, ki]),
                "sl_count":          int(sl_total[ci, ki]),
                "timeout_count":     int(to_total[ci, ki]),
                "expectancy_atr":    float(r_total[ci, ki]) / max(trade_count, 1),
                "weighted_pf":       pf_val,
                "sum_realized_atr":  float(r_total[ci, ki]),
                "avg_trade_drawdown_atr": float(dd_sum_tot[ci, ki]) / max(trade_count, 1),
                "max_trade_drawdown_atr": float(dd_max_tot[ci, ki]),
            })

    return pd.DataFrame(records).sort_values(
        ["expectancy_atr", "weighted_pf", "sum_realized_atr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _derive_recommendations(rows_df: pd.DataFrame, decision_df: pd.DataFrame) -> dict:
    winners = rows_df[rows_df["max_runup_atr"] >= 3.0]
    winner_mae_p90 = _safe_quantile(winners["max_mae_from_entry_atr"], 0.90)
    winner_mae_p50 = _safe_quantile(winners["max_mae_from_entry_atr"], 0.50)
    time_to_peak_p90 = _safe_quantile(winners["time_to_peak_bars"], 0.90)
    bars_to_3atr_median = _safe_quantile(winners["bars_to_first_3atr_runup"], 0.50)
    best_row = decision_df.iloc[0] if not decision_df.empty else pd.Series(dtype=float)
    return {
        "thor_sl_atr": _nearest_choice(winner_mae_p90, SL_ATR_CANDIDATES.tolist()),
        "thor_bank_atr": float(best_row.get("bank_atr", 3.0)),
        "thor_bank_fraction": float(best_row.get("bank_fraction", 0.25)),
        "thor_trail_activate_atr": float(best_row.get("trail_activate_atr", 2.0)),
        "thor_runner_trail_atr": float(best_row.get("runner_trail_atr", 2.5)),
        "thor_max_bars_pre_bank": int(_nearest_choice(bars_to_3atr_median, PRE_BANK_CANDIDATES)),
        "thor_max_bars_post_bank": int(_nearest_choice(time_to_peak_p90, POST_BANK_CANDIDATES)),
        "thor_mae_veto_atr": float(0.0 if not np.isfinite(winner_mae_p50) else winner_mae_p50),
        "sl_source": "P90 winner MAE from entry",
        "mae_veto_source": "P50 winner MAE from entry",
        "pre_bank_source": "median bars to first 3 ATR runup",
        "post_bank_source": "P90 winner time to peak",
    }


def _build_report(
    rows_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    recommendations: dict,
    baseline_tag: str,
) -> str:
    realized_col = f"realized_atr_sl_{baseline_tag}"

    survival_records = []
    for sl_atr in SL_ATR_CANDIDATES.tolist():
        col = f"outcome_sl_{_tag_float(sl_atr)}"
        for bucket in BUCKET_ORDER:
            sub = rows_df[rows_df["outcome_bucket"] == bucket]
            survival = float((sub[col] != "SL").mean() * 100.0) if not sub.empty else np.nan
            survival_records.append({"sl_atr": sl_atr, "outcome_bucket": bucket, "survival_pct": survival})
    survival_df = pd.DataFrame(survival_records).pivot(index="outcome_bucket", columns="sl_atr", values="survival_pct")
    survival_df = survival_df.reindex(BUCKET_ORDER).reset_index()

    curve_with_bucket = curves_df.merge(rows_df[["pump_id", "outcome_bucket"]], on="pump_id", how="left")
    heatmap_df = (
        curve_with_bucket[curve_with_bucket["bars_since_entry"] <= HEATMAP_BARS]
        .groupby(["bars_since_entry", "outcome_bucket"])["drawdown_atr"]
        .quantile(0.90)
        .unstack()
        .reindex(columns=BUCKET_ORDER)
        .reset_index()
    )

    hour_profit_df = (
        rows_df.groupby(["day_of_week_utc", "hour_of_day_utc"])[realized_col]
        .sum()
        .reset_index()
        .pivot(index="day_of_week_utc", columns="hour_of_day_utc", values=realized_col)
        .reindex(index=list(range(7)), columns=list(range(24)))
        .reset_index()
    )

    tier_wide_rows = []
    for tier in sorted(rows_df["tier"].dropna().astype(str).unique()):
        sub = rows_df[rows_df["tier"].astype(str) == tier]
        for sl_atr in SL_ATR_CANDIDATES.tolist():
            tag = _tag_float(sl_atr)
            realized = sub[f"realized_atr_sl_{tag}"]
            tier_wide_rows.append(
                {
                    "tier": tier,
                    "sl_atr": sl_atr,
                    "count": int(len(sub)),
                    "expectancy_atr": _safe_mean(realized),
                    "weighted_pf": _safe_pf(realized),
                }
            )
    tier_wide_df = pd.DataFrame(tier_wide_rows).sort_values(["tier", "sl_atr"])

    chain_df = (
        rows_df.groupby("prior_gap_bucket")[realized_col]
        .agg(["count", "mean", "sum"])
        .reset_index()
        .rename(columns={"mean": "expectancy_atr", "sum": "sum_realized_atr"})
    )

    fake_tier_df = (
        rows_df.groupby("tier")["reversed_within_6_bars"]
        .mean()
        .mul(100.0)
        .reset_index()
        .rename(columns={"reversed_within_6_bars": "fake_breakout_rate_pct"})
        .sort_values("fake_breakout_rate_pct", ascending=False)
    )
    fake_score_df = (
        rows_df.groupby("score_band")["reversed_within_6_bars"]
        .mean()
        .mul(100.0)
        .reset_index()
        .rename(columns={"reversed_within_6_bars": "fake_breakout_rate_pct"})
        .sort_values("fake_breakout_rate_pct", ascending=False)
    )

    baldur_df = rows_df[rows_df["baldur_warning_fired_bar"].notna()].copy()
    if not baldur_df.empty:
        baldur_summary_df = pd.DataFrame(
            [
                {
                    "warning_rows": int(len(baldur_df)),
                    "current_exit_expectancy_atr": _safe_mean(baldur_df[realized_col]),
                    "baldur_exit_expectancy_atr": _safe_mean(baldur_df["baldur_exit_runup_atr"]),
                    "median_warning_delay_from_peak": _safe_quantile(baldur_df["baldur_warning_delay_from_peak"], 0.50),
                }
            ]
        )
    else:
        baldur_summary_df = pd.DataFrame()

    best_matrix_df = decision_df.head(12).copy()
    symbol_top_df = (
        rows_df.groupby("symbol")[realized_col]
        .sum()
        .reset_index()
        .rename(columns={realized_col: "sum_realized_atr"})
        .sort_values("sum_realized_atr", ascending=False)
        .head(20)
    )
    regime_df = (
        rows_df.groupby("regime_bucket")[realized_col]
        .agg(["count", "mean", "sum"])
        .reset_index()
        .rename(columns={"mean": "expectancy_atr", "sum": "sum_realized_atr"})
        .sort_values("expectancy_atr", ascending=False)
    )
    stats_head_df = stats_df.sort_values(["bucket_type", "bucket_value"]).head(24)

    recommendation_lines = "\n".join(
        [
            f"- `thor_sl_atr = {recommendations['thor_sl_atr']:.2f}` from {recommendations['sl_source']}",
            f"- `thor_bank_atr = {recommendations['thor_bank_atr']:.2f}` from top decision-matrix combo",
            f"- `thor_bank_fraction = {recommendations['thor_bank_fraction']:.2f}` from top decision-matrix combo",
            f"- `thor_trail_activate_atr = {recommendations['thor_trail_activate_atr']:.2f}` from top decision-matrix combo",
            f"- `thor_runner_trail_atr = {recommendations['thor_runner_trail_atr']:.2f}` from top decision-matrix combo",
            f"- `thor_max_bars_pre_bank = {recommendations['thor_max_bars_pre_bank']}` from {recommendations['pre_bank_source']}",
            f"- `thor_max_bars_post_bank = {recommendations['thor_max_bars_post_bank']}` from {recommendations['post_bank_source']}",
            f"- `thor_mae_veto_atr = {recommendations['thor_mae_veto_atr']:.2f}` from {recommendations['mae_veto_source']}",
        ]
    )

    return f"""# Norse MAE Stats Report

## Coverage
- Pump rows: `{len(rows_df)}`
- Drawdown-curve rows: `{len(curves_df)}`
- Aggregate bucket rows: `{len(stats_df)}`
- Decision-matrix combos: `{len(decision_df)}`
- Baseline stop used for aggregate PF/expectancy tables: `{baseline_tag.replace('p', '.')}` ATR

## Winner Survival Curve
{_markdown_table(survival_df)}

## Drawdown-by-Bar Heatmap (P90 ATR)
{_markdown_table(heatmap_df, max_rows=20)}

## Hour-of-Week Profit Heatmap (sum realized ATR)
{_markdown_table(hour_profit_df, max_rows=10)}

## Tier Profitability Under Widened Stops
{_markdown_table(tier_wide_df, max_rows=20)}

## Chain vs Solo Pumps
{_markdown_table(chain_df)}

## Fake-Breakout Rate by Tier
{_markdown_table(fake_tier_df)}

## Fake-Breakout Rate by Score Band
{_markdown_table(fake_score_df)}

## Baldur Exit Efficacy
{_markdown_table(baldur_summary_df)}

## Stop Decision Matrix (top combos)
{_markdown_table(best_matrix_df)}

## Per-Symbol Top 20
{_markdown_table(symbol_top_df)}

## Regime Profitability
{_markdown_table(regime_df)}

## Aggregate Bucket Snapshot
{_markdown_table(stats_head_df)}

## Recommended Parameter Block
{recommendation_lines}
"""


def run_pump_mae_stats(
    days: int = 365,
    max_symbols: int = 0,
    skip_decision_matrix: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, object]:
    ev = Config.events
    t0 = time.time()

    universe = _load_windowed_cache(days=days)
    if max_symbols > 0:
        universe = universe[:max_symbols]

    print(f"[mae] {len(universe)} symbols loaded ({days}d cache)", flush=True)

    row_records: list[dict] = []
    curve_records: list[dict] = []
    decision_payloads: list[dict] = []
    baseline_tag = _tag_float(float(ev.thor_sl_atr))
    tasks = [(idx, symbol, df) for idx, (symbol, df) in enumerate(universe)]
    n_tasks = len(tasks)

    _init_mae_worker()  # warm up in main thread first (loads PyTorch model once)

    completed = [0]
    def _run_task(task):
        result = _process_symbol_mae(task)
        completed[0] += 1
        return result

    from tqdm import tqdm
    n_workers = min(max_workers, n_tasks)
    if n_workers <= 1:
        results = [_run_task(task) for task in tqdm(tasks, total=n_tasks, desc="[mae] Processing Symbols")]
    else:
        # ThreadPoolExecutor: much faster startup than ProcessPoolExecutor on Windows.
        # Numba and PyTorch both release the GIL, so threads run truly parallel.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(_run_task, tasks), total=n_tasks, desc="[mae] Processing Symbols"))

    for result in results:
        row_records.extend(result["rows"])
        curve_records.extend(result["curves"])
        payload = result["decision_payload"]
        if payload is not None:
            decision_payloads.append(payload)

    print(f"[mae] per-symbol done  {time.time()-t0:.0f}s  rows={len(row_records)}", flush=True)

    rows_df   = pd.DataFrame(row_records).sort_values(["entry_ts", "symbol"]).reset_index(drop=True) if row_records else pd.DataFrame()
    curves_df = pd.DataFrame(curve_records).sort_values(["pump_id", "bars_since_entry"]).reset_index(drop=True) if curve_records else pd.DataFrame()
    stats_df  = _bucket_stats(rows_df, baseline_tag) if not rows_df.empty else pd.DataFrame()

    if skip_decision_matrix or not decision_payloads:
        decision_df = pd.DataFrame()
        print("[mae] decision matrix skipped", flush=True)
    else:
        print(f"[mae] running decision matrix ({len(BANK_ATR_CANDIDATES)}×{len(BANK_FRACTION_CANDIDATES)}×{len(TRAIL_ACTIVATE_CANDIDATES)}×{len(RUNNER_TRAIL_CANDIDATES)} combos)…", flush=True)
        decision_df = _decision_matrix(decision_payloads, ev)
        print(f"[mae] decision matrix done  {time.time()-t0:.0f}s", flush=True)

    recommendations = _derive_recommendations(rows_df, decision_df) if not rows_df.empty else {}
    if rows_df.empty:
        report = "# Norse MAE Stats Report\n\n_no pump rows generated_"
    else:
        report = _build_report(rows_df, curves_df, stats_df, decision_df, recommendations, baseline_tag)

    rows_df.to_csv(ROWS_CSV, index=False)
    curves_df.to_csv(CURVES_CSV, index=False)
    stats_df.to_csv(STATS_CSV, index=False)
    decision_df.to_csv(DECISION_CSV, index=False)
    REPORT_MD.write_text(report, encoding="utf-8")

    print(f"[mae] total {time.time()-t0:.0f}s  wrote {REPORT_MD.name}", flush=True)
    return {
        "rows_df": rows_df,
        "curves_df": curves_df,
        "stats_df": stats_df,
        "decision_df": decision_df,
        "report_path": REPORT_MD,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Norse pump MAE stats")
    ap.add_argument("--days",                 type=int,  default=365,   help="Days of cache to load (default 365; use 30 for a quick test)")
    ap.add_argument("--max-symbols",          type=int,  default=0,     help="Cap number of symbols (0 = all)")
    ap.add_argument("--extend-existing-rows", action="store_true",      help="Only recompute and append the cache-tuner columns on the existing rows CSV")
    ap.add_argument("--skip-decision-matrix", action="store_true",      help="Skip the bank/trail combo grid (saves ~60%% of runtime)")
    ap.add_argument("--workers",              type=int,  default=DEFAULT_MAX_WORKERS, help="Thread workers (default 4)")
    args = ap.parse_args()

    if args.extend_existing_rows:
        rows_df = extend_existing_rows_csv(
            days=args.days,
            max_symbols=args.max_symbols,
            max_workers=args.workers,
        )
        print(f"Pump rows: {len(rows_df)}")
        print(f"Updated {ROWS_CSV.name}")
    else:
        result = run_pump_mae_stats(
            days=args.days,
            max_symbols=args.max_symbols,
            skip_decision_matrix=args.skip_decision_matrix,
            max_workers=args.workers,
        )
        rows_df = result["rows_df"]
        print(f"Pump rows: {len(rows_df)}")
        print(f"Wrote {ROWS_CSV.name}, {CURVES_CSV.name}, {STATS_CSV.name}, {DECISION_CSV.name}, {REPORT_MD.name}")
