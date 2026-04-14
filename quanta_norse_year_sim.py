"""
Feature-aware Norse year paper simulator.

Simulation-first upgrade for the Norse trio:
- Thor: breakout core holder with wide trailing stop
- Freya: continuation scalp only while Thor's wave is healthy
- Baldur: bearish top-start companion after Thor expansion weakens
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from heapq import heappop, heappush
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quanta_config import Config
from QUANTA_ml_engine import (
    build_offline_feature_replay_engine,
    extract_offline_features_for_positions,
    precompute_offline_feature_bundle,
)
from quanta_nike_live_validator import extract_nike_signals
from quanta_norse_agents import (
    build_pump_ledger,
    build_sparse_feature_context,
    build_thor_contexts,
    calc_atr,
    compute_pump_state,
    extract_baldur_signals,
    extract_freya_signals,
    find_active_thor_context,
    measure_baldur_topstart,
    score_baldur_signal,
    score_baldur_warning,
    score_freya_signal,
    score_thor_signal,
    simulate_directional_exit_stop_market,
    simulate_thor_exit_stop_market,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "feather_cache"
RUNS_DIR = ROOT / "norse_runs"
INITIAL_CAPITAL = 10000.0


@dataclass
class CandidateTrade:
    agent: str
    symbol: str
    direction: str
    entry_ts: int
    exit_ts: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    atr: float
    realized_atr: float
    label: str
    score: float
    stop_atr: float
    risk_pct: float
    max_leverage: float
    capital_cap: float
    pump_id: str
    stop_slip_atr: float
    peak_drawdown_entry_atr: float
    peak_drawdown_exit_atr: float
    top_warning_score: float = 0.0
    stop_type: str = "stop_market"
    parent_pump_id: str | None = None
    parent_agent: str | None = None
    notional_fraction_of_parent: float = 0.0


@dataclass
class PreparedSymbol:
    symbol: str
    df: pd.DataFrame
    atrs: np.ndarray
    raw_thor_signals: list[dict]
    replay_engine: object
    precomputed: dict
    klines_np: np.ndarray
    feature_cache: dict[int, np.ndarray]
    feature_ctx: object
    norse_event_cache: dict


def _safe_pf(pos_sum: float, neg_sum: float) -> float:
    if neg_sum >= 0:
        return float("inf")
    return pos_sum / abs(neg_sum)


def _format_pf(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.3f}"


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _fmt_utc_ms(ts_ms: int) -> str:
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")


def _trade_log_line(trade: CandidateTrade) -> str:
    return (
        f"{trade.agent} {trade.symbol} {_fmt_utc_ms(trade.entry_ts)} -> "
        f"{trade.label} exit={_fmt_utc_ms(trade.exit_ts)} "
        f"score={trade.score:.1f} pnl_atr={trade.realized_atr:+.2f}"
    )


def _make_run_artifacts() -> dict[str, Path]:
    run_ts = datetime.now(timezone.utc)
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_ts_utc": run_ts.isoformat(),
        "run_dir": run_dir,
        "report": run_dir / f"NORSE_YEAR_PAPER_REPORT_{run_id}.md",
        "summary_csv": run_dir / f"norse_year_paper_summary_{run_id}.csv",
        "trades_csv": run_dir / f"norse_year_paper_trades_{run_id}.csv",
        "pump_stats_csv": run_dir / f"norse_pump_stats_{run_id}.csv",
        "pump_ledger_csv": run_dir / f"norse_pump_ledger_{run_id}.csv",
        "tuning_csv": run_dir / f"norse_year_tuning_search_{run_id}.csv",
        "trade_diagnostics_csv": run_dir / f"norse_trade_diagnostics_{run_id}.csv",
        "loss_diagnostics_csv": run_dir / f"norse_loss_diagnostics_{run_id}.csv",
        "big_win_diagnostics_csv": run_dir / f"norse_big_win_diagnostics_{run_id}.csv",
        "feature_compare_csv": run_dir / f"norse_feature_compare_{run_id}.csv",
        "reason_summary_csv": run_dir / f"norse_reason_summary_{run_id}.csv",
    }


def _allowed_tiers(text: str) -> set[str]:
    return {part.strip().upper() for part in str(text or "").split(",") if part.strip()}


def _load_windowed_cache(days: int = 365) -> list[tuple[str, pd.DataFrame]]:
    files = sorted(CACHE_DIR.glob("*_5m.feather"))
    raw = []
    max_ts = 0
    for path in files:
        try:
            df = pd.read_feather(path)
        except Exception:
            continue
        if df.empty or "open_time" not in df.columns:
            continue
        ts_max = int(df["open_time"].max())
        max_ts = max(max_ts, ts_max)
        raw.append((path.stem.replace("_5m", ""), df))

    if max_ts <= 0:
        return []

    start_ts = max_ts - days * 24 * 60 * 60 * 1000
    filtered = []
    for symbol, df in raw:
        view = df[df["open_time"] >= start_ts].copy()
        if len(view) < 120:
            continue
        view.attrs["symbol"] = symbol
        filtered.append((symbol, view.reset_index(drop=True)))
    return filtered


def _prepare_symbol(symbol: str, df: pd.DataFrame, replay_engine, ev) -> PreparedSymbol:
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    atrs = calc_atr(highs, lows, closes)

    raw_thor_signals = extract_nike_signals(df)
    for sig in raw_thor_signals:
        i = int(sig["bar_idx"])
        sig["symbol"] = symbol
        sig["atr"] = float(atrs[i]) if i < len(atrs) else 0.0

    feature_bundle = precompute_offline_feature_bundle(df, engine=replay_engine)
    feature_ctx = build_sparse_feature_context(df, {})

    return PreparedSymbol(
        symbol=symbol,
        df=df,
        atrs=atrs,
        raw_thor_signals=raw_thor_signals,
        replay_engine=feature_bundle["engine"],
        precomputed=feature_bundle["precomputed"],
        klines_np=feature_bundle["klines_np"],
        feature_cache={},
        feature_ctx=feature_ctx,
        norse_event_cache={},
    )


def _ensure_feature_positions(prep: PreparedSymbol, positions) -> None:
    # Cache-loaded symbols already carry a full dense SparseFeatureContext on disk.
    # Avoid re-entering the offline feature extractor when there is no replay engine.
    if prep.replay_engine is None and len(getattr(prep.feature_ctx, "times", ())) == len(prep.df):
        return
    wanted = {int(p) for p in positions if 50 <= int(p) < len(prep.df)}
    missing = sorted(p for p in wanted if p not in prep.feature_cache)
    if not missing:
        return
    bundle = extract_offline_features_for_positions(
        prep.df,
        prep.symbol,
        missing,
        engine=prep.replay_engine,
        precomputed=prep.precomputed,
        klines_np=prep.klines_np,
    )
    prep.feature_cache.update(bundle["features_by_pos"])
    prep.feature_ctx = build_sparse_feature_context(prep.df, prep.feature_cache)


def _thor_base_cache_key(bar_idx: int, material_drawdown_atr: float) -> tuple[int, float]:
    return int(bar_idx), float(material_drawdown_atr)


def _thor_exit_cache_key(sig: dict, params: dict) -> tuple:
    return (
        int(sig["bar_idx"]),
        float(params["thor_bank_atr"]),
        float(params["thor_sl_atr"]),
        float(params["thor_bank_fraction"]),
        float(params["thor_trail_activate_atr"]),
        float(params["thor_runner_trail_atr"]),
        int(params["thor_max_bars_pre_bank"]),
        int(params["thor_max_bars_post_bank"]),
        float(params["stop_market_penetration_factor"]),
        float(params["stop_market_slip_atr_cap"]),
    )


def _get_cached_thor_base(prep: PreparedSymbol, sig: dict, params: dict) -> dict:
    cache = prep.norse_event_cache.setdefault("thor_base", {})
    i = int(sig["bar_idx"])
    key = _thor_base_cache_key(i, float(params["pump_material_drawdown_atr"]))
    cached = cache.get(key)
    if cached is not None:
        return cached

    lookback_start = max(0, i - 6)
    _ensure_feature_positions(prep, range(max(50, lookback_start), i + 1))
    ctx = prep.feature_ctx
    ts_ms = int(ctx.times[i])
    result = {
        "entry_hour_utc": int((ts_ms // 3_600_000) % 24),
        "feature_score": float(score_thor_signal(sig, ctx)),
        "pre_entry_state": compute_pump_state(
            ctx,
            lookback_start,
            i,
            float(params["pump_material_drawdown_atr"]),
        ),
    }
    cache[key] = result
    return result


def _get_cached_thor_exit(prep: PreparedSymbol, sig: dict, params: dict) -> dict:
    cache = prep.norse_event_cache.setdefault("thor_exit", {})
    key = _thor_exit_cache_key(sig, params)
    cached = cache.get(key)
    if cached is not None:
        return cached

    result = simulate_thor_exit_stop_market(
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
    cache[key] = result
    return result


def _prime_thor_event_cache(prepared_symbols: list[PreparedSymbol], params: dict) -> None:
    for prep in prepared_symbols:
        for sig in prep.raw_thor_signals:
            _get_cached_thor_base(prep, sig, params)


def _parse_allowed_hours(text: str) -> set[int]:
    """Parse thor_entry_hour_utc: '14' → {14}, '13,14,15' → {13,14,15}, '' → empty=all."""
    text = str(text or "").strip()
    if not text:
        return set()
    result = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            result.add(int(part))
    return result


def _build_thor_candidates(
    prep: PreparedSymbol,
    ev,
    params: dict,
    include_details: bool = False,
) -> tuple[list[CandidateTrade], list[dict]]:
    allowed_tiers = _allowed_tiers(params["thor_trade_tiers"])
    allowed_hours = _parse_allowed_hours(params.get("thor_entry_hour_utc", ""))
    cooldown = int(params["thor_trade_cooldown_bars"])
    last_accepted_bar = -cooldown
    trades: list[CandidateTrade] = []
    accepted_signals: list[dict] = []

    for sig in prep.raw_thor_signals:
        i = int(sig["bar_idx"])
        if i - last_accepted_bar < cooldown:
            continue
        if allowed_tiers and str(sig.get("tier", "")).upper() not in allowed_tiers:
            continue
        if float(sig.get("atr", 0.0)) <= 0.0:
            continue
        base = _get_cached_thor_base(prep, sig, params)
        ctx = prep.feature_ctx
        # Hour-of-day gate (UTC): only trade during profitable hours from MAE analysis
        if allowed_hours and int(base["entry_hour_utc"]) not in allowed_hours:
            continue

        pre_entry_state = base["pre_entry_state"]
        if pre_entry_state["max_drawdown_atr"] > float(params["thor_mae_veto_atr"]):
            continue
        if pre_entry_state["wave_strength_score"] < float(params["thor_wave_strength_min"]):
            continue
        if pre_entry_state["top_risk_score"] >= float(params["thor_top_risk_max"]):
            continue

        feature_score = float(base["feature_score"])
        if feature_score < float(params["thor_min_score_trade"]):
            continue

        sim = _get_cached_thor_exit(prep, sig, params)
        peak_drawdown_exit_atr = 0.0
        if include_details:
            _ensure_feature_positions(prep, [int(sim["exit_bar"])])
            ctx = prep.feature_ctx
            peak_drawdown_exit_atr = float(
                compute_pump_state(
                    ctx,
                    i,
                    int(sim["exit_bar"]),
                    float(params["pump_material_drawdown_atr"]),
                )["max_drawdown_atr"]
            )

        pump_id = f"{prep.symbol}:{i}"
        accepted_sig = dict(sig)
        accepted_sig["score"] = round(float(feature_score), 2)
        accepted_sig["pump_id"] = pump_id
        accepted_signals.append(accepted_sig)

        trades.append(
            CandidateTrade(
                agent="Thor",
                symbol=prep.symbol,
                direction="BULLISH",
                entry_ts=int(ctx.times[i]),
                exit_ts=int(ctx.times[int(sim["exit_bar"])]),
                entry_bar=i,
                exit_bar=int(sim["exit_bar"]),
                entry_price=float(ctx.closes[i]),
                exit_price=float(sim["exit_price"]),
                atr=float(sig["atr"]),
                realized_atr=float(sim["realized_atr"]),
                label=str(sim["label"]),
                score=float(feature_score),
                stop_atr=float(params["thor_sl_atr"]),
                risk_pct=float(params["thor_risk_pct"]),
                max_leverage=float(params["thor_max_leverage"]),
                capital_cap=float(params["thor_capital_cap"]),
                pump_id=pump_id,
                stop_slip_atr=float(sim.get("stop_slip_atr", 0.0)),
                peak_drawdown_entry_atr=float(pre_entry_state["max_drawdown_atr"]),
                peak_drawdown_exit_atr=peak_drawdown_exit_atr,
            )
        )
        last_accepted_bar = i

    return trades, accepted_signals


def _build_baldur_candidates(prep: PreparedSymbol, accepted_thor_signals: list[dict], ev, params: dict) -> list[dict]:
    warnings, raw_signals = extract_baldur_signals(prep.df, accepted_thor_signals, ev)
    _ensure_feature_positions(prep, [w["bar_idx"] for w in warnings] + [s["bar_idx"] for s in raw_signals])
    ctx = prep.feature_ctx
    thor_contexts = build_thor_contexts(accepted_thor_signals, int(getattr(ev, "thor_context_bars", 24)), 0.0)

    warning_rows: list[dict] = []
    for warning in warnings:
        thor_ctx = find_active_thor_context(thor_contexts, int(warning["bar_idx"]))
        if thor_ctx is None:
            continue
        warning_score = score_baldur_warning(warning, ctx, thor_ctx, ev)
        topstart = measure_baldur_topstart(warning, ctx.closes, ctx.lows, target_atr=1.0, max_bars=12)
        row = dict(warning)
        row["pump_id"] = f"{prep.symbol}:{thor_ctx.start_bar}"
        row["thor_entry_bar"] = int(thor_ctx.start_bar)
        row["warning_exit_bar"] = min(int(warning["bar_idx"]) + 1, len(ctx.opens) - 1)
        row["warning_exit_price"] = float(ctx.opens[row["warning_exit_bar"]])
        row["top_warning_score"] = round(float(warning_score), 2)
        row["topstart_success"] = bool(topstart.get("success", False))
        row["topstart_delay_bars"] = topstart.get("delay_bars")
        warning_rows.append(row)

    return warning_rows


def _apply_baldur_exits(
    prep: PreparedSymbol,
    thor_trades: list[CandidateTrade],
    warning_rows: list[dict],
    params: dict,
) -> list[CandidateTrade]:
    trigger_score = float(params["baldur_warning_exit_score"])
    best_warning_by_pump: dict[str, dict] = {}
    for row in warning_rows:
        if float(row.get("top_warning_score", 0.0)) < trigger_score:
            continue
        pump_id = str(row.get("pump_id", ""))
        if not pump_id:
            continue
        current = best_warning_by_pump.get(pump_id)
        if current is None or int(row["bar_idx"]) < int(current["bar_idx"]):
            best_warning_by_pump[pump_id] = row

    adjusted: list[CandidateTrade] = []
    ctx = prep.feature_ctx
    for trade in thor_trades:
        row = best_warning_by_pump.get(trade.pump_id)
        if row is None:
            adjusted.append(trade)
            continue
        exit_bar = int(row["warning_exit_bar"])
        if exit_bar <= trade.entry_bar or exit_bar >= trade.exit_bar:
            adjusted.append(trade)
            continue
        exit_price = float(row["warning_exit_price"])
        realized_atr = (exit_price - trade.entry_price) / max(trade.atr, 1e-12)
        exit_state = compute_pump_state(
            ctx,
            trade.entry_bar,
            exit_bar,
            float(params["pump_material_drawdown_atr"]),
        )
        label = "TP" if realized_atr >= 0.0 else "SL"
        adjusted.append(
            replace(
                trade,
                exit_ts=int(ctx.times[exit_bar]),
                exit_bar=exit_bar,
                exit_price=exit_price,
                realized_atr=float(realized_atr),
                label=label,
                peak_drawdown_exit_atr=float(exit_state["max_drawdown_atr"]),
                top_warning_score=float(row.get("top_warning_score", 0.0)),
            )
        )
    return adjusted


def _build_freya_pyramid_adds(
    prep: PreparedSymbol,
    thor_trades: list[CandidateTrade],
    accepted_thor_signals: list[dict],
    ev,
    params: dict,
    include_details: bool = False,
) -> tuple[list[CandidateTrade], int]:
    raw_signals = extract_freya_signals(prep.df, accepted_thor_signals, ev)
    _ensure_feature_positions(prep, [s["bar_idx"] for s in raw_signals])
    ctx = prep.feature_ctx
    thor_contexts = build_thor_contexts(accepted_thor_signals, int(getattr(ev, "thor_context_bars", 24)), 0.0)
    blocked_by_top = 0
    trades: list[CandidateTrade] = []
    used_pumps: set[str] = set()
    thor_by_pump = {trade.pump_id: trade for trade in thor_trades}

    for sig in raw_signals:
        i = int(sig["bar_idx"])
        thor_ctx = find_active_thor_context(thor_contexts, i)
        if thor_ctx is None:
            continue
        pump_id = f"{prep.symbol}:{thor_ctx.start_bar}"
        if pump_id in used_pumps:
            continue
        parent_trade = thor_by_pump.get(pump_id)
        if parent_trade is None or not (parent_trade.entry_bar < i < parent_trade.exit_bar):
            continue

        entry_state = compute_pump_state(ctx, thor_ctx.start_bar, i, float(params["pump_material_drawdown_atr"]))
        score, top_risk_score = score_freya_signal(sig, ctx, thor_ctx, ev)
        if top_risk_score >= float(params["freya_pyramid_add_top_risk_max"]):
            blocked_by_top += 1
            continue
        if score < float(params["freya_min_score_trade"]):
            continue
        if entry_state["runup_atr"] < float(params.get("freya_min_runup_atr", 0.6)):
            continue
        if entry_state["wave_strength_score"] < float(params["freya_pyramid_add_trigger_wave"]):
            continue

        peak_drawdown_exit_atr = 0.0
        if include_details:
            exit_state = compute_pump_state(
                ctx,
                thor_ctx.start_bar,
                parent_trade.exit_bar,
                float(params["pump_material_drawdown_atr"]),
            )
            peak_drawdown_exit_atr = float(exit_state["max_drawdown_atr"])
        entry_price = float(ctx.closes[i])
        exit_price = float(parent_trade.exit_price)
        realized_atr = (exit_price - entry_price) / max(float(sig["atr"]), 1e-12)
        if parent_trade.label == "TIMEOUT":
            label = "TIMEOUT"
        else:
            label = "TP" if realized_atr >= 0.0 else "SL"
        trades.append(
            CandidateTrade(
                agent="Freya",
                symbol=prep.symbol,
                direction="BULLISH",
                entry_ts=int(ctx.times[i]),
                exit_ts=int(ctx.times[parent_trade.exit_bar]),
                entry_bar=i,
                exit_bar=int(parent_trade.exit_bar),
                entry_price=entry_price,
                exit_price=exit_price,
                atr=float(sig["atr"]),
                realized_atr=float(realized_atr),
                label=label,
                score=float(score),
                stop_atr=float(params["thor_sl_atr"]),
                risk_pct=float(params["thor_risk_pct"]),
                max_leverage=float(params["thor_max_leverage"]),
                capital_cap=float(params["freya_capital_cap"]),
                pump_id=pump_id,
                stop_slip_atr=0.0,
                peak_drawdown_entry_atr=float(entry_state["max_drawdown_atr"]),
                peak_drawdown_exit_atr=peak_drawdown_exit_atr,
                parent_pump_id=pump_id,
                parent_agent="Thor",
                notional_fraction_of_parent=float(params["freya_pyramid_add_notional_fraction"]),
            )
        )
        used_pumps.add(pump_id)

    return trades, blocked_by_top


def _build_pump_outputs(prep: PreparedSymbol, thor_trades: list[CandidateTrade], warning_rows: list[dict], params: dict) -> tuple[list[dict], list[dict]]:
    baldur_first = {}
    baldur_scores = {}
    for row in warning_rows:
        pump_id = str(row.get("pump_id", ""))
        if not pump_id:
            continue
        bar_idx = int(row["bar_idx"])
        baldur_first[pump_id] = min(baldur_first.get(pump_id, bar_idx), bar_idx)
        baldur_scores[pump_id] = max(baldur_scores.get(pump_id, 0.0), float(row.get("top_warning_score", 0.0)))

    needed_positions: set[int] = set()
    for trade in thor_trades:
        needed_positions.update(range(trade.entry_bar, trade.exit_bar + 1))
    _ensure_feature_positions(prep, needed_positions)

    ledger_rows: list[dict] = []
    summary_rows: list[dict] = []
    for trade in thor_trades:
        end_bar = trade.exit_bar
        rows, summary = build_pump_ledger(
            prep.feature_ctx,
            trade.pump_id,
            trade.entry_bar,
            end_bar,
            material_drawdown_atr=float(params["pump_material_drawdown_atr"]),
        )
        if summary:
            summary["thor_exit_bar"] = trade.exit_bar
            summary["thor_exit_label"] = trade.label
            summary["baldur_end_bar"] = baldur_first.get(trade.pump_id)
            summary["baldur_warning_score"] = baldur_scores.get(trade.pump_id, 0.0)
            summary_rows.append(summary)
        ledger_rows.extend(rows)
    return ledger_rows, summary_rows


def _simulate_capital(
    trades: list[CandidateTrade],
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    free_cash = float(initial_capital)
    peak = free_cash
    max_dd = 0.0
    open_heap: list[tuple[int, int, dict]] = []
    open_pairs: set[tuple[str, str]] = set()
    open_parent_notional: dict[str, float] = {}
    open_agent_counts = {"Thor": 0, "Freya": 0, "Baldur": 0}
    open_agent_margin = {"Thor": 0.0, "Freya": 0.0, "Baldur": 0.0}
    trade_rows: list[dict] = []
    equity_curve = [(0, free_cash)]
    seq = 0

    # ── Compound Growth Engine state ─────────────────────────────────────
    ev = Config.events
    compound_mode             = str(getattr(ev, "compound_mode", "asymmetric_target"))
    compound_target_win_pct   = float(getattr(ev, "compound_target_win_pct", 10.0))
    compound_max_loss_pct     = float(getattr(ev, "compound_max_loss_pct", 3.0))
    compound_activation_score = float(getattr(ev, "compound_activation_score", 85.0))

    def _get_notional_size(equity: float, trade, stop_dist_pct: float) -> float:
        """
        Asymmetric Target Compounding (2026-04-12).
        If the trade is a high-conviction pump setup (score >= 85), we size the position
        so that hitting the stop loss costs exactly 3% of current equity.
        Otherwise, we fall back to the conservative baseline risk (0.5%).
        """
        score = getattr(trade, 'score', 0)
        if compound_mode == "asymmetric_target" and score >= compound_activation_score:
            target_loss_usd = equity * (compound_max_loss_pct / 100.0)
            return target_loss_usd / max(0.0001, stop_dist_pct)
        else:
            baseline_risk = float(getattr(trade, "risk_pct", 0.005))
            target_loss_usd = equity * baseline_risk
            return target_loss_usd / max(0.0001, stop_dist_pct)
    # ─────────────────────────────────────────────────────────────────────

    def current_equity() -> float:
        reserved = sum(pos["margin_required"] for _, _, pos in open_heap)
        return free_cash + reserved

    def close_due(until_ts: int):
        nonlocal free_cash, peak, max_dd
        while open_heap and open_heap[0][0] <= until_ts:
            _, _, pos = heappop(open_heap)
            open_pairs.discard((pos["agent"], pos["symbol"]))
            open_agent_counts[pos["agent"]] = max(0, open_agent_counts.get(pos["agent"], 0) - 1)
            open_agent_margin[pos["agent"]] = max(0.0, open_agent_margin.get(pos["agent"], 0.0) - pos["margin_required"])
            if pos.get("pump_id") and pos["agent"] == "Thor":
                open_parent_notional.pop(pos["pump_id"], None)
            free_cash += pos["margin_required"] + pos["close_cashflow"]
            equity_now = current_equity()
            peak = max(peak, equity_now)
            dd = 0.0 if peak <= 0 else (peak - equity_now) / peak
            max_dd = max(max_dd, dd)
            equity_curve.append((pos["exit_ts"], equity_now))
            trade_rows.append(pos["row"])

    for trade in sorted(trades, key=lambda t: (t.entry_ts, t.exit_ts, t.agent, t.symbol)):
        close_due(trade.entry_ts)
        pair_key = (trade.agent, trade.symbol)
        if pair_key in open_pairs:
            continue

        tuned_limit = None if params is None else params.get(f"{trade.agent.lower()}_max_concurrent_positions")
        max_agent_positions = int(
            getattr(Config.events, f"{trade.agent.lower()}_max_concurrent_positions", 1)
            if tuned_limit is None
            else tuned_limit
        )
        if open_agent_counts.get(trade.agent, 0) >= max_agent_positions:
            continue

        equity_now = current_equity()
        if equity_now <= 0 or free_cash <= 0:
            break

        stop_distance_pct = max(0.001, (trade.stop_atr * trade.atr) / max(trade.entry_price, 1e-12))
        total_notional_cap = free_cash * trade.max_leverage
        agent_margin_cap = max(0.0, equity_now * trade.capital_cap - open_agent_margin.get(trade.agent, 0.0))
        agent_notional_cap = agent_margin_cap * trade.max_leverage
        if trade.parent_pump_id and trade.parent_agent == "Thor":
            parent_notional = open_parent_notional.get(trade.parent_pump_id, 0.0)
            if parent_notional <= 0.0:
                continue
            desired_notional = parent_notional * max(0.0, trade.notional_fraction_of_parent)
            entry_notional = max(0.0, min(desired_notional, total_notional_cap, agent_notional_cap))
        else:
            # ── Asymmetric Target Compounding ──────────────────────────
            notional_by_risk = _get_notional_size(equity_now, trade, stop_distance_pct)
            if trade.notional_fraction_of_parent > 0.0 and not trade.parent_pump_id:
                notional_by_risk *= trade.notional_fraction_of_parent
            entry_notional = max(0.0, min(notional_by_risk, total_notional_cap, agent_notional_cap))

        if entry_notional <= 10.0:
            continue

        margin_required = entry_notional / max(trade.max_leverage, 1e-12)
        size_units = entry_notional / max(trade.entry_price, 1e-12)
        entry_friction = entry_notional * (commission_bps + slippage_bps) / 10000.0
        if margin_required + entry_friction > free_cash:
            continue

        free_cash -= margin_required + entry_friction
        open_agent_margin[trade.agent] = open_agent_margin.get(trade.agent, 0.0) + margin_required

        gross_pnl = (
            (trade.entry_price - trade.exit_price) * size_units
            if trade.direction == "BEARISH"
            else (trade.exit_price - trade.entry_price) * size_units
        )
        exit_notional = abs(trade.exit_price * size_units)
        exit_friction = exit_notional * (commission_bps + slippage_bps) / 10000.0
        close_cashflow = gross_pnl - exit_friction

        row = {
            "agent": trade.agent,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_ts": trade.entry_ts,
            "exit_ts": trade.exit_ts,
            "entry_price": trade.entry_price,
            "exit_price_est": trade.exit_price,
            "entry_notional": entry_notional,
            "size_units": size_units,
            "realized_atr": trade.realized_atr,
            "gross_pnl": gross_pnl,
            "net_pnl": gross_pnl - entry_friction - exit_friction,
            "label": trade.label,
            "score": trade.score,
            "hold_bars": trade.exit_bar - trade.entry_bar,
            "stop_type": trade.stop_type,
            "stop_market_slippage_atr": trade.stop_slip_atr,
            "pump_id": trade.pump_id,
            "peak_drawdown_entry_atr": trade.peak_drawdown_entry_atr,
            "peak_drawdown_exit_atr": trade.peak_drawdown_exit_atr,
            "top_warning_score": trade.top_warning_score,
            "capital_cap": trade.capital_cap,
            "parent_pump_id": trade.parent_pump_id,
            "notional_fraction_of_parent": trade.notional_fraction_of_parent,
        }

        open_pairs.add(pair_key)
        open_agent_counts[trade.agent] = open_agent_counts.get(trade.agent, 0) + 1
        if trade.agent == "Thor":
            open_parent_notional[trade.pump_id] = entry_notional
        heappush(
            open_heap,
            (
                trade.exit_ts,
                seq,
                {
                    "agent": trade.agent,
                    "symbol": trade.symbol,
                    "exit_ts": trade.exit_ts,
                    "margin_required": margin_required,
                    "close_cashflow": close_cashflow,
                    "pump_id": trade.pump_id,
                    "row": row,
                },
            ),
        )
        
        # Telemetry: Record equity AFTER margin is secured to vault
        equity_now = current_equity()
        peak = max(peak, equity_now)
        dd = 0.0 if peak <= 0 else (peak - equity_now) / peak
        max_dd = max(max_dd, dd)
        equity_curve.append((trade.entry_ts, equity_now))

        seq += 1

    close_due(2**63 - 1)

    trade_df = pd.DataFrame(trade_rows).sort_values(["entry_ts", "agent", "symbol"]).reset_index(drop=True) if trade_rows else pd.DataFrame()
    balance = max(0.0, free_cash)
    summary = {
        "initial_capital": initial_capital,
        "final_capital": balance,
        "growth_pct": ((balance / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0,
        "max_drawdown_pct": min(100.0, max_dd * 100.0),
        "trades": int(len(trade_df)),
        "equity_points": len(equity_curve),
        "equity_curve": equity_curve,
        "liquidation_hit": balance <= 0.0,
    }
    return trade_df, summary


def _summarize_trade_df(trade_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trade_df.empty or "agent" not in trade_df.columns:
        trade_df = pd.DataFrame(columns=["agent", "label", "net_pnl", "hold_bars"])
    total_net = float(trade_df["net_pnl"].sum()) if "net_pnl" in trade_df.columns else 0.0
    for agent in ["Thor", "Freya", "Baldur"]:
        sub = trade_df[trade_df["agent"] == agent]
        if sub.empty:
            rows.append(
                {
                    "agent": agent,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "timeouts": 0,
                    "tp_rate_all": 0.0,
                    "decided_win_rate": 0.0,
                    "net_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "profit_factor": 0.0,
                    "avg_hold_bars": 0.0,
                    "capital_contrib_pct": 0.0,
                }
            )
            continue
        wins = int((sub["label"] == "TP").sum())
        losses = int((sub["label"] == "SL").sum())
        timeouts = int((sub["label"] == "TIMEOUT").sum())
        decided = wins + losses
        pnl_pos = float(sub.loc[sub["net_pnl"] >= 0, "net_pnl"].sum())
        pnl_neg = float(sub.loc[sub["net_pnl"] < 0, "net_pnl"].sum())
        net = float(sub["net_pnl"].sum())
        rows.append(
            {
                "agent": agent,
                "trades": int(len(sub)),
                "wins": wins,
                "losses": losses,
                "timeouts": timeouts,
                "tp_rate_all": wins / len(sub) * 100.0,
                "decided_win_rate": wins / decided * 100.0 if decided else 0.0,
                "net_pnl": net,
                "avg_pnl": float(sub["net_pnl"].mean()),
                "profit_factor": _safe_pf(pnl_pos, pnl_neg),
                "avg_hold_bars": float(sub["hold_bars"].mean()),
                "capital_contrib_pct": (net / total_net * 100.0) if total_net not in (0.0, -0.0) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_params(
    prepared_symbols: list[PreparedSymbol],
    params: dict,
    include_pump_outputs: bool = False,
    sim_logger=None,
    detailed_progress: bool = False,
) -> dict:
    bt = Config.backtest
    commission_bps = float(getattr(bt, "commission_bps", 4.0))
    slippage_bps = float(getattr(bt, "slippage_bps", 2.0))
    include_trade_details = bool(include_pump_outputs)

    all_trades: list[CandidateTrade] = []
    thor_only: list[CandidateTrade] = []
    warning_rows: list[dict] = []
    pump_ledger_rows: list[dict] = []
    pump_summary_rows: list[dict] = []
    freya_blocked = 0

    for idx, prep in enumerate(prepared_symbols, 1):
        thor_trades, accepted_thor_signals = _build_thor_candidates(prep, Config.events, params, include_details=include_trade_details)
        warning_stats = _build_baldur_candidates(prep, accepted_thor_signals, Config.events, params)
        thor_trades = _apply_baldur_exits(prep, thor_trades, warning_stats, params)
        freya_trades, blocked = _build_freya_pyramid_adds(
            prep,
            thor_trades,
            accepted_thor_signals,
            Config.events,
            params,
            include_details=include_trade_details,
        )
        if include_pump_outputs:
            ledger_rows, summary_rows = _build_pump_outputs(prep, thor_trades, warning_stats, params)
        else:
            ledger_rows, summary_rows = [], []

        thor_only.extend(thor_trades)
        all_trades.extend(thor_trades)
        all_trades.extend(freya_trades)
        warning_rows.extend(warning_stats)
        pump_ledger_rows.extend(ledger_rows)
        pump_summary_rows.extend(summary_rows)
        freya_blocked += blocked

        if sim_logger is not None:
            if detailed_progress and (thor_trades or freya_trades or warning_stats):
                tp_count = sum(1 for trade in thor_trades if trade.label == "TP")
                sl_count = sum(1 for trade in thor_trades if trade.label == "SL")
                timeout_count = sum(1 for trade in thor_trades if trade.label == "TIMEOUT")
                sim_logger(
                    f"{prep.symbol}: Thor accepted={len(accepted_thor_signals)} "
                    f"trades={len(thor_trades)} tp={tp_count} sl={sl_count} timeout={timeout_count} "
                    f"Baldur_warnings={len(warning_stats)} Freya_adds={len(freya_trades)} blocked={blocked}"
                )
                for trade in thor_trades[:2]:
                    sim_logger(f"  {_trade_log_line(trade)}")
                if len(thor_trades) > 2:
                    sim_logger(f"  ... {len(thor_trades) - 2} more Thor trades on {prep.symbol}")
                for trade in freya_trades[:1]:
                    sim_logger(f"  {_trade_log_line(trade)}")
            elif idx == 1 or idx == len(prepared_symbols) or idx % 25 == 0:
                sim_logger(
                    f"full-sim prep {idx}/{len(prepared_symbols)} symbols "
                    f"aggregate_trades={len(all_trades)} warnings={len(warning_rows)}"
                )

    full_trade_df, full_portfolio = _simulate_capital(all_trades, INITIAL_CAPITAL, commission_bps, slippage_bps, params=params)
    full_agent_summary = _summarize_trade_df(full_trade_df)
    active_agents = sorted({trade.agent for trade in all_trades})
    trade_df = full_trade_df
    portfolio_summary = full_portfolio
    agent_summary = full_agent_summary
    thor_df, thor_summary = _simulate_capital([t for t in thor_only], INITIAL_CAPITAL, commission_bps, slippage_bps, params=params)

    warning_df = pd.DataFrame(warning_rows)
    pump_summary_df = pd.DataFrame(pump_summary_rows)
    pump_ledger_df = pd.DataFrame(pump_ledger_rows)

    top_warning_precision = float(warning_df["topstart_success"].mean() * 100.0) if not warning_df.empty else 0.0
    top_warning_delay = float(warning_df["topstart_delay_bars"].dropna().median()) if not warning_df.empty and warning_df["topstart_delay_bars"].notna().any() else None

    return {
        "params": dict(params),
        "trade_df": trade_df,
        "portfolio_summary": portfolio_summary,
        "agent_summary": agent_summary,
        "full_trade_df": full_trade_df,
        "full_agent_summary": full_agent_summary,
        "thor_df": thor_df,
        "thor_summary": thor_summary,
        "warning_df": warning_df,
        "pump_summary_df": pump_summary_df,
        "pump_ledger_df": pump_ledger_df,
        "top_warning_precision": top_warning_precision,
        "top_warning_delay": top_warning_delay,
        "freya_blocked": freya_blocked,
        "active_agents": active_agents,
    }


def _portfolio_objective(summary: dict, penalty_weight: float) -> float:
    dd_penalty = max(0.0, (float(summary["max_drawdown_pct"]) - 35.0) / 100.0) * penalty_weight
    net_growth = float(summary["final_capital"]) - float(summary.get("initial_capital", INITIAL_CAPITAL))
    return net_growth * (1.0 - dd_penalty)


def _better_result(candidate: dict, incumbent: dict, penalty_weight: float) -> bool:
    a = candidate["portfolio_summary"]
    b = incumbent["portfolio_summary"]
    score_a = _portfolio_objective(a, penalty_weight)
    score_b = _portfolio_objective(b, penalty_weight)
    if score_a != score_b:
        return score_a > score_b
    if a["final_capital"] != b["final_capital"]:
        return a["final_capital"] > b["final_capital"]
    if a["max_drawdown_pct"] != b["max_drawdown_pct"]:
        return a["max_drawdown_pct"] < b["max_drawdown_pct"]
    return a["trades"] < b["trades"]


def _tune_params(prepared_symbols: list[PreparedSymbol], base_params: dict) -> tuple[dict, dict, pd.DataFrame]:
    # ── Turbo Tuner: Optuna TPE + walk-forward CV + bootstrap CI ─────────────
    # Replaces the old coordinate-search loop (which took ~11 hours).
    # Stage 0 (event cache build) is a separate one-time run:
    #   python norse_event_cache_builder.py --days 365 --workers 6
    allow_legacy_fallback = _env_flag("QUANTA_NORSE_ALLOW_LEGACY_FALLBACK")
    try:
        from norse_tuner import run_tuning_pipeline
        return run_tuning_pipeline(prepared_symbols, base_params)
    except Exception as _turbo_exc:
        if not allow_legacy_fallback:
            raise RuntimeError(
                "Turbo Tuner failed and legacy coordinate-search fallback is disabled. "
                "Fix the stage-0 cache / MAE artifacts or set "
                "QUANTA_NORSE_ALLOW_LEGACY_FALLBACK=1 for an emergency-only fallback run."
            ) from _turbo_exc
        print(
            "[tune] Turbo Tuner failed "
            f"({_turbo_exc}), falling back to coordinate search because "
            "QUANTA_NORSE_ALLOW_LEGACY_FALLBACK=1",
            flush=True,
        )

    # ── Coordinate-search fallback (legacy, ~11 h) ────────────────────────────
    penalty_weight = float(base_params.get("norse_drawdown_penalty_weight", 1.5))
    search_space = {
        # #1 impact: hour gate. MAE shows hour-14 only has PF 1.39 vs negative elsewhere.
        "thor_entry_hour_utc": ["14", "13,14,15", "14,15", ""],
        # #2 impact: tier. Tier B has 7% lower fake-breakout rate than A.
        "thor_trade_tiers": ["A", "A,B", "B"],
        # #3 impact: SL. MAE P90 winner drawdown = 3.0 ATR — tighter = shake-out.
        "thor_sl_atr": [2.4, 3.0, 3.6, 4.2],
        # #4 impact: Baldur exit trigger. Higher = fewer false exits.
        "baldur_warning_exit_score": [70.0, 75.0, 82.0],
        # #5 impact: min score gate. Filter noise signals.
        "thor_min_score_trade": [88.0, 92.0, 96.0, 99.0],
        "thor_mae_veto_atr": [2.8, 3.62, 4.5],
        "thor_wave_strength_min": [60.0, 70.0, 80.0],
        "thor_top_risk_max": [35.0, 40.0, 50.0],
        "thor_bank_atr": [3.0, 4.2, 5.0],
        "thor_bank_fraction": [0.15, 0.35, 0.50],
        "thor_trail_activate_atr": [1.5, 2.0, 2.5],
        "thor_runner_trail_atr": [2.5, 3.0, 4.0],
        "thor_max_bars_pre_bank": [12, 18, 24],
        "thor_max_bars_post_bank": [36, 48, 72],
        "freya_min_score_trade": [70.0, 74.0, 80.0],
        "freya_pyramid_add_notional_fraction": [0.50, 0.75, 1.00],
        "freya_pyramid_add_trigger_wave": [75.0, 80.0, 85.0],
        "freya_pyramid_add_top_risk_max": [30.0, 35.0, 40.0],
        "thor_risk_pct": [0.005, 0.010, 0.015],
        "thor_max_leverage": [5.0, 8.0, 12.0],
        "thor_capital_cap": [0.50, 0.75, 1.00],
        "thor_max_concurrent_positions": [3, 5, 8],
        "freya_capital_cap": [0.25, 0.40, 0.60],
        "freya_max_leverage": [3.0, 5.0, 8.0],
        "freya_max_concurrent_positions": [1, 2, 3],
    }

    history_rows = []
    best_params = dict(base_params)
    best_result = _evaluate_params(prepared_symbols, best_params, include_pump_outputs=False)
    history_rows.append(
        {
            "phase": "baseline",
            "param": "baseline",
            "value": "baseline",
            "final_capital": best_result["portfolio_summary"]["final_capital"],
            "growth_pct": best_result["portfolio_summary"]["growth_pct"],
            "max_drawdown_pct": best_result["portfolio_summary"]["max_drawdown_pct"],
            "trades": best_result["portfolio_summary"]["trades"],
            "active_agents": ",".join(best_result["active_agents"]),
            "objective_score": _portfolio_objective(best_result["portfolio_summary"], penalty_weight),
        }
    )

    import time as _time
    _tune_t0 = _time.time()
    def _tune_log(msg: str) -> None:
        print(f"[tune {_time.time()-_tune_t0:5.0f}s] {msg}", flush=True)

    improved = True
    passes = 0
    while improved and passes < 2:
        improved = False
        passes += 1
        n_params = len(search_space)
        for p_idx, (key, values) in enumerate(search_space.items()):
            _tune_log(f"pass {passes}  param {p_idx+1}/{n_params}: {key}")
            local_best_params = dict(best_params)
            local_best_result = best_result
            for value in values:
                if best_params.get(key) == value:
                    continue
                trial_params = dict(best_params)
                trial_params[key] = value
                trial_result = _evaluate_params(prepared_symbols, trial_params, include_pump_outputs=False)
                history_rows.append(
                    {
                        "phase": f"pass_{passes}",
                        "param": key,
                        "value": value,
                        "final_capital": trial_result["portfolio_summary"]["final_capital"],
                        "growth_pct": trial_result["portfolio_summary"]["growth_pct"],
                        "max_drawdown_pct": trial_result["portfolio_summary"]["max_drawdown_pct"],
                        "trades": trial_result["portfolio_summary"]["trades"],
                        "active_agents": ",".join(trial_result["active_agents"]),
                        "objective_score": _portfolio_objective(trial_result["portfolio_summary"], penalty_weight),
                    }
                )
                if _better_result(trial_result, local_best_result, penalty_weight):
                    local_best_result = trial_result
                    local_best_params = trial_params
            if local_best_params != best_params:
                best_params = local_best_params
                best_result = local_best_result
                improved = True

    tuning_df = pd.DataFrame(history_rows)
    return best_params, best_result, tuning_df


def _render_wf_folds(result: dict) -> str:
    folds = result.get("wf_fold_metrics")
    if not folds:
        return ""
    lines = [
        "\n## Tuning — Walk-forward fold stats",
        "| Fold | Trades | Calmar | Growth% | MaxDD% | PF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    calmars = []
    for i, m in enumerate(folds):
        calmar = m.get("calmar", 0.0)
        calmars.append(calmar)
        lines.append(
            f"| {i} | {m.get('trades',0)} | {calmar:.2f} | "
            f"{m.get('growth_pct',0):+.2f} | {m.get('max_drawdown_pct',0):.2f} | "
            f"{m.get('pf',0):.3f} |"
        )
    import numpy as _np
    lines.append(
        f"| **median** | — | **{_np.median(calmars):.2f}** | — | — | — |"
    )
    return "\n".join(lines)


def _render_sensitivity(result: dict) -> str:
    rows = result.get("sensitivity_rows")
    if not rows:
        return ""
    lines = [
        "\n## Tuning — Parameter sensitivity (±20%)",
        "| Param | Value | −20% score | +20% score | Max drop | Brittle? |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['param']} | {r['value']} | "
            f"{r.get('minus20_score', float('nan')):.3f} | "
            f"{r.get('plus20_score',  float('nan')):.3f} | "
            f"{max(r.get('minus20_drop',0), r.get('plus20_drop',0)):.1%} | "
            f"{'⚠️ YES' if r['brittle'] else 'no'} |"
        )
    return "\n".join(lines)


def _render_bootstrap(result: dict) -> str:
    study = result.get("optuna_study")
    if study is None:
        return ""
    return (
        "\n## Tuning — Optuna study\n"
        f"- Trials completed: `{len(study.trials)}`\n"
        f"- Best score: `{study.best_value:.4f}`\n"
        f"- Best trial #: `{study.best_trial.number}`"
    )


def _render_learned_filter(result: dict) -> str:
    lf = result.get("learned_filter")
    if lf is None:
        return ""
    from norse_tuner.learned_filter import _SL_KEYS
    lines = ["\n## Tuning — Learned filter (top features by |coef|)"]
    for sl_key in _SL_KEYS:
        feats = lf.top_features(sl_key, n=5)
        if not feats:
            continue
        aucs = lf.auc_by_sl.get(sl_key, [])
        import numpy as _np
        avg_auc = float(_np.mean(aucs)) if aucs else float("nan")
        lines.append(f"\n**SL={sl_key} ATR**  avg_fold_AUC={avg_auc:.4f}")
        for feat, coef in feats:
            direction = "↑profit" if coef > 0 else "↓profit"
            lines.append(f"- `{feat}` coef={coef:+.4f}  {direction}")
    return "\n".join(lines)


def _render_cache_divergence(result: dict) -> str:
    cvs = result.get("cache_vs_sim")
    if cvs is None:
        return ""
    cap_div = abs(cvs["cached_final_capital"] - cvs["sim_final_capital"]) / max(abs(cvs["cached_final_capital"]), 1.0)
    dd_div  = abs(cvs["cached_max_dd_pct"] - cvs["sim_max_dd_pct"])
    flag = "⚠️ DIVERGENCE" if cap_div > 0.05 or dd_div > 3.0 else "✅ OK"
    return (
        f"\n## Tuning — Cache vs full-sim reconciliation  {flag}\n"
        f"| Metric | Cached estimate | Full sim | Delta |\n"
        f"| --- | ---: | ---: | ---: |\n"
        f"| final_capital | ${cvs['cached_final_capital']:,.2f} | ${cvs['sim_final_capital']:,.2f} | {cap_div:.1%} |\n"
        f"| max_drawdown_pct | {cvs['cached_max_dd_pct']:.2f}% | {cvs['sim_max_dd_pct']:.2f}% | {dd_div:.2f} pp |"
    )


def _classify_loss_reason(row: pd.Series, params: dict) -> str:
    if float(row.get("stop_market_slippage_atr", 0.0)) >= 0.25:
        return "stop slippage expanded the loss"
    if float(row.get("top_warning_score", 0.0)) >= float(params.get("baldur_warning_exit_score", 0.0)):
        return "Baldur warning exit fired into weakness"
    if float(row.get("max_runup_atr", 0.0)) < 1.0 and float(row.get("max_drawdown_atr", 0.0)) >= 1.5:
        return "breakout failed before any real expansion"
    if pd.notna(row.get("time_to_material_drawdown_bars")) and float(row.get("time_to_material_drawdown_bars", 999)) <= 3:
        return "early adverse reversal after entry"
    if float(row.get("final_top_risk_score", 0.0)) > float(params.get("thor_top_risk_max", 0.0)):
        return "top-risk expanded too quickly"
    if float(row.get("volume_decay_after_peak", 0.0)) < -0.35:
        return "participation collapsed after the peak"
    return "runner never developed enough before the stop"


def _classify_big_win_reason(row: pd.Series, params: dict) -> str:
    if (
        float(row.get("max_runup_atr", 0.0)) >= 5.0
        and float(row.get("final_wave_strength_score", 0.0)) >= float(params.get("thor_wave_strength_min", 0.0)) + 10.0
    ):
        return "strong wave expansion with sustained follow-through"
    if (
        float(row.get("time_to_peak_bars", 999)) <= 6
        and float(row.get("volume_decay_after_peak", -1.0)) >= -0.10
    ):
        return "fast expansion with participation holding up"
    if (
        float(row.get("top_warning_score", 0.0)) < 0.5 * float(params.get("baldur_warning_exit_score", 0.0))
        and float(row.get("final_top_risk_score", 100.0)) <= 0.8 * float(params.get("thor_top_risk_max", 100.0))
    ):
        return "low top-risk trend with clean continuation"
    return "runner capture held through normal noise"


def _build_trade_diagnostics(result: dict, params: dict) -> dict[str, pd.DataFrame | float | int]:
    trade_df = result.get("trade_df", pd.DataFrame()).copy()
    pump_summary_df = result.get("pump_summary_df", pd.DataFrame()).copy()
    if trade_df.empty:
        empty = pd.DataFrame()
        return {
            "trade_diag_df": empty,
            "loss_diag_df": empty,
            "big_win_diag_df": empty,
            "feature_compare_df": empty,
            "reason_summary_df": empty,
            "big_win_threshold": 0.0,
            "big_win_count": 0,
            "loss_count": 0,
        }

    trade_df["entry_utc"] = trade_df["entry_ts"].map(_fmt_utc_ms)
    trade_df["exit_utc"] = trade_df["exit_ts"].map(_fmt_utc_ms)
    merged = trade_df.merge(
        pump_summary_df,
        on=["pump_id", "symbol"],
        how="left",
        suffixes=("", "_pump"),
    )

    losses = merged[merged["net_pnl"] < 0.0].copy()
    positive = merged[merged["net_pnl"] > 0.0].copy()
    big_win_threshold = float(positive["net_pnl"].quantile(0.80)) if not positive.empty else 0.0
    big_wins = positive[positive["net_pnl"] >= big_win_threshold].copy() if big_win_threshold > 0 else positive.copy()

    if not losses.empty:
        losses["what_went_wrong"] = losses.apply(lambda row: _classify_loss_reason(row, params), axis=1)
    if not big_wins.empty:
        big_wins["what_went_right"] = big_wins.apply(lambda row: _classify_big_win_reason(row, params), axis=1)

    merged["outcome_bucket"] = "other"
    merged.loc[merged["net_pnl"] < 0.0, "outcome_bucket"] = "loss"
    if not big_wins.empty:
        merged.loc[merged.index.isin(big_wins.index), "outcome_bucket"] = "big_win"

    compare_cols = [
        "score",
        "realized_atr",
        "net_pnl",
        "hold_bars",
        "peak_drawdown_entry_atr",
        "peak_drawdown_exit_atr",
        "top_warning_score",
        "stop_market_slippage_atr",
        "entry_notional",
        "max_runup_atr",
        "max_drawdown_atr",
        "time_to_peak_bars",
        "time_to_material_drawdown_bars",
        "volume_decay_after_peak",
        "final_wave_strength_score",
        "final_top_risk_score",
        "baldur_warning_score",
    ]
    feature_rows: list[dict] = []
    for col in compare_cols:
        if col not in merged.columns:
            continue
        loss_vals = pd.to_numeric(losses[col], errors="coerce").dropna() if not losses.empty else pd.Series(dtype=float)
        win_vals = pd.to_numeric(big_wins[col], errors="coerce").dropna() if not big_wins.empty else pd.Series(dtype=float)
        if loss_vals.empty and win_vals.empty:
            continue
        feature_rows.append(
            {
                "feature": col,
                "loss_count": int(loss_vals.shape[0]),
                "big_win_count": int(win_vals.shape[0]),
                "loss_mean": float(loss_vals.mean()) if not loss_vals.empty else np.nan,
                "loss_median": float(loss_vals.median()) if not loss_vals.empty else np.nan,
                "big_win_mean": float(win_vals.mean()) if not win_vals.empty else np.nan,
                "big_win_median": float(win_vals.median()) if not win_vals.empty else np.nan,
                "median_delta_big_win_minus_loss": (
                    float(win_vals.median()) - float(loss_vals.median())
                    if (not loss_vals.empty and not win_vals.empty)
                    else np.nan
                ),
            }
        )
    feature_compare_df = pd.DataFrame(feature_rows).sort_values(
        "median_delta_big_win_minus_loss",
        ascending=False,
        na_position="last",
    ) if feature_rows else pd.DataFrame()

    reason_rows: list[dict] = []
    if not losses.empty and "what_went_wrong" in losses.columns:
        for reason, count in losses["what_went_wrong"].value_counts().items():
            reason_rows.append({"bucket": "loss", "reason": reason, "count": int(count)})
    if not big_wins.empty and "what_went_right" in big_wins.columns:
        for reason, count in big_wins["what_went_right"].value_counts().items():
            reason_rows.append({"bucket": "big_win", "reason": reason, "count": int(count)})

    sort_cols = ["net_pnl", "realized_atr"]
    loss_diag_df = losses.sort_values(sort_cols, ascending=[True, True]).reset_index(drop=True)
    big_win_diag_df = big_wins.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)
    trade_diag_df = merged.sort_values("entry_ts").reset_index(drop=True)
    reason_summary_df = pd.DataFrame(reason_rows)

    return {
        "trade_diag_df": trade_diag_df,
        "loss_diag_df": loss_diag_df,
        "big_win_diag_df": big_win_diag_df,
        "feature_compare_df": feature_compare_df,
        "reason_summary_df": reason_summary_df,
        "big_win_threshold": big_win_threshold,
        "big_win_count": int(len(big_win_diag_df)),
        "loss_count": int(len(loss_diag_df)),
    }


def _report_text(
    result: dict,
    prepared_symbols: list[PreparedSymbol],
    tuning_df: pd.DataFrame,
    artifact_paths: dict[str, Path | str],
    diagnostics: dict,
) -> str:
    portfolio = result["portfolio_summary"]
    thor_only = result["thor_summary"]
    agent_summary = result["full_agent_summary"]
    pump_summary_df = result["pump_summary_df"]
    params = result["params"]
    feature_compare_df = diagnostics["feature_compare_df"]
    loss_diag_df = diagnostics["loss_diag_df"]
    big_win_diag_df = diagnostics["big_win_diag_df"]
    reason_summary_df = diagnostics["reason_summary_df"]

    report = f"""# Norse Year Paper Simulation Report

## Setup
- Run ID: `{artifact_paths['run_id']}`
- Run timestamp (UTC): `{artifact_paths['run_ts_utc']}`
- Initial paper capital: `${INITIAL_CAPITAL:,.2f}`
- Window: last `365` cached days across the local `5m` universe
- Symbols included: `{len(prepared_symbols)}`
- Optimization target: `maximum final capital with >35% drawdown penalty`
- Stops modeled as: `stop-market`
- Portfolio caps: `Thor {params['thor_capital_cap']:.0%}`, `Freya {params['freya_capital_cap']:.0%}`
- Active agents: `{", ".join(result['active_agents'])}`

## Best Parameters
- Thor score floor: `{params['thor_min_score_trade']:.1f}`
- Thor tiers: `{params['thor_trade_tiers']}`
- Thor stop: `{params['thor_sl_atr']:.2f} ATR`
- Thor bank: `{params['thor_bank_atr']:.2f} ATR @ {params['thor_bank_fraction']:.0%}`
- Thor trail activate: `{params['thor_trail_activate_atr']:.2f} ATR`
- Thor trail gap: `{params['thor_runner_trail_atr']:.2f} ATR`
- Thor MAE veto: `{params['thor_mae_veto_atr']:.2f} ATR`
- Thor wave-strength min: `{params['thor_wave_strength_min']:.1f}`
- Thor top-risk max: `{params['thor_top_risk_max']:.1f}`
- Baldur warning exit score: `{params['baldur_warning_exit_score']:.1f}`
- Freya score floor: `{params['freya_min_score_trade']:.1f}`
- Freya pyramid wave min: `{params['freya_pyramid_add_trigger_wave']:.1f}`
- Freya pyramid top-risk max: `{params['freya_pyramid_add_top_risk_max']:.1f}`
- Freya add size: `{params['freya_pyramid_add_notional_fraction']:.0%}` of parent Thor notional
- Pump material drawdown threshold: `{params['pump_material_drawdown_atr']:.2f} ATR`

## Portfolio Result
- Final capital: `${portfolio['final_capital']:,.2f}`
- Growth: `{portfolio['growth_pct']:+.2f}%`
- Max drawdown: `{portfolio['max_drawdown_pct']:.2f}%`
- Executed trades: `{portfolio['trades']}`
- Liquidation hit: `{'YES' if portfolio['liquidation_hit'] else 'NO'}`

## Thor-Only Baseline
- Final capital: `${thor_only['final_capital']:,.2f}`
- Growth: `{thor_only['growth_pct']:+.2f}%`
- Max drawdown: `{thor_only['max_drawdown_pct']:.2f}%`
- Executed trades: `{thor_only['trades']}`

## Agent Statistics (Candidate Portfolio)
| Agent | Trades | TP | SL | TIMEOUT | TP Rate | Decided Win Rate | Net PnL ($) | Avg PnL ($) | Profit Factor | Avg Hold Bars | Capital Contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {row.agent} | {int(row.trades)} | {int(row.wins)} | {int(row.losses)} | {int(row.timeouts)} | "
        f"{row.tp_rate_all:.2f}% | {row.decided_win_rate:.2f}% | {row.net_pnl:+.2f} | {row.avg_pnl:+.2f} | "
        f"{_format_pf(row.profit_factor)} | {row.avg_hold_bars:.2f} | {row.capital_contrib_pct:+.2f}% |"
        for row in agent_summary.itertuples(index=False)
    )

    if not pump_summary_df.empty:
        report += f"""

## Pump Analytics
- Thor pumps recorded: `{len(pump_summary_df)}`
- Median max run-up: `{pump_summary_df['max_runup_atr'].median():.2f} ATR`
- Median max drawdown: `{pump_summary_df['max_drawdown_atr'].median():.2f} ATR`
- Median time to peak: `{pump_summary_df['time_to_peak_bars'].median():.1f}` bars
- Median volume decay after peak: `{pump_summary_df['volume_decay_after_peak'].median():+.2f}`
"""
    else:
        report += """

## Pump Analytics
- Thor pumps recorded: `0`
"""

    report += f"""

## Baldur / Freya Diagnostics
- Baldur top-warning precision: `{result['top_warning_precision']:.2f}%`
- Baldur median delay to downside: `{'n/a' if result['top_warning_delay'] is None else f"{result['top_warning_delay']:.1f} bars"}`
- Baldur warnings are used as Thor exit triggers, not standalone shorts.
- Freya blocked by top-risk: `{int(result['freya_blocked'])}`
"""

    if not reason_summary_df.empty:
        report += """

## Trade Outcome Reasons
| Bucket | Reason | Count |
| --- | --- | ---: |
""" + "\n".join(
            f"| {row.bucket} | {row.reason} | {int(row.count)} |"
            for row in reason_summary_df.itertuples(index=False)
        )

    if not loss_diag_df.empty:
        report += """

## Loss Diagnostics
""" + (
            f"- Loss trades recorded: `{diagnostics['loss_count']}`\n"
            f"- Full loss log: `{artifact_paths['loss_diagnostics_csv'].name}`\n"
        )
        report += """
| Agent | Symbol | Entry UTC | Label | Net PnL | Realized ATR | What Went Wrong |
| --- | --- | --- | --- | ---: | ---: | --- |
""" + "\n".join(
            f"| {row.agent} | {row.symbol} | {row.entry_utc} | {row.label} | "
            f"{row.net_pnl:+.2f} | {row.realized_atr:+.2f} | {row.what_went_wrong} |"
            for row in loss_diag_df.head(12).itertuples(index=False)
        )

    if not big_win_diag_df.empty:
        report += """

## Big Win Diagnostics
""" + (
            f"- Big win threshold: `net_pnl >= ${diagnostics['big_win_threshold']:.2f}`\n"
            f"- Big wins recorded: `{diagnostics['big_win_count']}`\n"
            f"- Full big-win log: `{artifact_paths['big_win_diagnostics_csv'].name}`\n"
        )
        report += """
| Agent | Symbol | Entry UTC | Net PnL | Realized ATR | What Went Right |
| --- | --- | --- | ---: | ---: | --- |
""" + "\n".join(
            f"| {row.agent} | {row.symbol} | {row.entry_utc} | "
            f"{row.net_pnl:+.2f} | {row.realized_atr:+.2f} | {row.what_went_right} |"
            for row in big_win_diag_df.head(12).itertuples(index=False)
        )

    if not feature_compare_df.empty:
        report += """

## Feature Statistics: Losses vs Big Wins
| Feature | Loss Median | Big Win Median | Delta |
| --- | ---: | ---: | ---: |
""" + "\n".join(
            f"| {row.feature} | {row.loss_median:.3f} | {row.big_win_median:.3f} | {row.median_delta_big_win_minus_loss:+.3f} |"
            for row in feature_compare_df.head(15).itertuples(index=False)
        )


    # ── Stop Loss Reversal Analysis ──────────────────────────────────────────────
    if not pump_summary_df.empty and "max_runup_atr" in pump_summary_df.columns and "max_drawdown_atr" in pump_summary_df.columns:
        _ps = pump_summary_df
        total_pumps = len(_ps)
        green_any   = int((_ps["max_runup_atr"] > 0).sum())
        green_05    = int((_ps["max_runup_atr"] >= 0.5).sum())
        green_10    = int((_ps["max_runup_atr"] >= 1.0).sum())
        green_15    = int((_ps["max_runup_atr"] >= 1.5).sum())
        green_20    = int((_ps["max_runup_atr"] >= 2.0).sum())
        runners     = _ps[_ps["max_runup_atr"] >= 2.0]
        whip_mean   = runners["max_drawdown_atr"].mean() if not runners.empty else float("nan")
        whip_med    = runners["max_drawdown_atr"].median() if not runners.empty else float("nan")
        whip_p25    = runners["max_drawdown_atr"].quantile(0.25) if not runners.empty else float("nan")
        def _pct(n: int, d: int) -> str:
            return f"{n} ({100*n/d:.1f}%)" if d > 0 else "n/a"
        report += f"""

## Stop Loss Reversal Analysis
*How many trades went green before failing, and how much red do big winners see?*

### Max Run-Up Distribution (all {total_pumps} pumps)
| Run-Up Reached | Count |
| --- | ---: |
| ≥ 0.0 ATR (any green) | {_pct(green_any, total_pumps)} |
| ≥ 0.5 ATR | {_pct(green_05, total_pumps)} |
| ≥ 1.0 ATR | {_pct(green_10, total_pumps)} |
| ≥ 1.5 ATR | {_pct(green_15, total_pumps)} |
| ≥ 2.0 ATR | {_pct(green_20, total_pumps)} |

### Biggest Winners (≥ +2.0 ATR) — How Far Into Red Did They Go?
| Metric | Value |
| --- | ---: |
| Count | {len(runners)} |
| P25 (cleanest take-offs) | {whip_p25:.2f} ATR |
| Median deepest red | {whip_med:.2f} ATR |
| Mean deepest red | {whip_mean:.2f} ATR |

> **Key insight:** ~{100*green_05/total_pumps:.0f}% of all signals went ≥ +0.5 ATR. Of big winners (≥ +2.0 ATR), the median max draw-down before mooning was **{whip_med:.2f} ATR** — which is why the SL must stay wide.
"""

    report += f"""

## Tuning Search
- Trials logged: `{len(tuning_df)}`
- Objective score: `{_portfolio_objective(portfolio, float(params['norse_drawdown_penalty_weight'])):.2f}`
- Method: `Optuna TPE + walk-forward CV + bootstrap CI + Numba exit fine-tune`
- Optuna trials: `{result.get('n_optuna_trials', 'n/a')}`
{_render_wf_folds(result)}
{_render_sensitivity(result)}
{_render_bootstrap(result)}
{_render_learned_filter(result)}
{_render_cache_divergence(result)}

## Output Files
- [`{artifact_paths['summary_csv'].name}`]({artifact_paths['summary_csv'].name})
- [`{artifact_paths['trades_csv'].name}`]({artifact_paths['trades_csv'].name})
- [`{artifact_paths['pump_stats_csv'].name}`]({artifact_paths['pump_stats_csv'].name})
- [`{artifact_paths['pump_ledger_csv'].name}`]({artifact_paths['pump_ledger_csv'].name})
- [`{artifact_paths['tuning_csv'].name}`]({artifact_paths['tuning_csv'].name})
- [`{artifact_paths['trade_diagnostics_csv'].name}`]({artifact_paths['trade_diagnostics_csv'].name})
- [`{artifact_paths['loss_diagnostics_csv'].name}`]({artifact_paths['loss_diagnostics_csv'].name})
- [`{artifact_paths['big_win_diagnostics_csv'].name}`]({artifact_paths['big_win_diagnostics_csv'].name})
- [`{artifact_paths['feature_compare_csv'].name}`]({artifact_paths['feature_compare_csv'].name})
- [`{artifact_paths['reason_summary_csv'].name}`]({artifact_paths['reason_summary_csv'].name})
"""
    return report


def run_year_simulation() -> tuple[dict, pd.DataFrame]:
    import time as _time
    _t0 = _time.time()
    artifact_paths = _make_run_artifacts()
    def _log(msg: str) -> None:
        elapsed = _time.time() - _t0
        print(f"[sim {elapsed:6.0f}s] {msg}", flush=True)

    ev = Config.events
    universe = _load_windowed_cache(days=365)
    _log(f"{len(universe)} symbols loaded from feather cache")

    # ── Try on-disk event cache first (built by norse_event_cache_builder.py) ─
    allow_cache_miss_fallback = _env_flag("QUANTA_NORSE_ALLOW_CACHE_MISS_FALLBACK")
    try:
        from norse_event_cache_loader import load_cached_symbol, validate_cache_universe
        _use_event_cache = True
    except ImportError:
        _use_event_cache = False
        if not allow_cache_miss_fallback:
            raise RuntimeError(
                "Norse event cache loader is unavailable. "
                "The canonical year-sim path requires stage-0 cache hits. "
                "Set QUANTA_NORSE_ALLOW_CACHE_MISS_FALLBACK=1 only for an emergency uncached run."
            )

    if _use_event_cache:
        cache_state = validate_cache_universe(universe)
        if not cache_state["complete"]:
            sample_text = ", ".join(
                f"{row['symbol']}:{row['reason']}" for row in cache_state["sample_failures"][:5]
            ) or "n/a"
            if not allow_cache_miss_fallback:
                raise RuntimeError(
                    "Stage-0 Norse event cache is incomplete or stale. "
                    f"manifest_present={cache_state['manifest_present']} "
                    f"manifest_complete={cache_state['manifest_complete']} "
                    f"valid={cache_state['symbols_valid']}/{cache_state['symbols_total']} "
                    f"invalid={cache_state['symbols_invalid']} "
                    f"sample_failures=[{sample_text}]. "
                    "Rebuild with: python norse_event_cache_builder.py --days 365 --workers 6 --force --clean"
                )
            _log(
                "event cache failed validation but uncached fallback is enabled "
                f"(valid={cache_state['symbols_valid']}/{cache_state['symbols_total']})"
            )
        else:
            manifest = cache_state["manifest"] or {}
            _log(
                "validated stage-0 event cache "
                f"(valid={cache_state['symbols_valid']}/{cache_state['symbols_total']} "
                f"schema={manifest.get('schema_version')} "
                f"built_at_ms={manifest.get('built_at_ms')})"
            )

    replay_engine = None
    prepared_symbols: list[PreparedSymbol] = []
    cache_hits = cache_misses = 0

    for idx, (symbol, df) in enumerate(universe):
        prepared = None
        if _use_event_cache:
            prepared = load_cached_symbol(symbol)
            if prepared is not None:
                cache_hits += 1
        if prepared is None:
            if not allow_cache_miss_fallback:
                raise RuntimeError(
                    f"Validated stage-0 cache expected a hit for {symbol}, but the loader returned a miss."
                )
            if replay_engine is None:
                _log("loading feature engine (cache miss) …")
                replay_engine = build_offline_feature_replay_engine(Config)
            prepared = _prepare_symbol(symbol, df, replay_engine, ev)
            cache_misses += 1
        if prepared.raw_thor_signals:
            prepared_symbols.append(prepared)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(universe):
            _log(
                f"prepared {idx+1}/{len(universe)}  "
                f"active={len(prepared_symbols)}  "
                f"cache_hits={cache_hits}  misses={cache_misses}"
            )
    _log(f"symbol prep done — {len(prepared_symbols)} active  hits={cache_hits}  misses={cache_misses}")

    base_params = {
        "thor_risk_pct": float(ev.thor_risk_pct),
        "thor_max_leverage": float(ev.thor_max_leverage),
        "thor_capital_cap": float(ev.thor_capital_cap),
        "thor_sl_atr": float(ev.thor_sl_atr),
        "thor_bank_atr": float(ev.thor_bank_atr),
        "thor_bank_fraction": float(ev.thor_bank_fraction),
        "thor_trail_activate_atr": float(ev.thor_trail_activate_atr),
        "thor_runner_trail_atr": float(ev.thor_runner_trail_atr),
        "thor_max_bars_pre_bank": int(ev.thor_max_bars_pre_bank),
        "thor_max_bars_post_bank": int(ev.thor_max_bars_post_bank),
        "thor_min_score_trade": float(ev.thor_min_score_trade),
        "thor_trade_tiers": str(ev.thor_trade_tiers),
        "thor_trade_cooldown_bars": int(ev.thor_trade_cooldown_bars),
        "thor_max_concurrent_positions": int(ev.thor_max_concurrent_positions),
        "thor_mae_veto_atr": float(ev.thor_mae_veto_atr),
        "thor_wave_strength_min": float(ev.thor_wave_strength_min),
        "thor_top_risk_max": float(ev.thor_top_risk_max),
        "thor_entry_hour_utc": str(getattr(ev, "thor_entry_hour_utc", "14")),
        "baldur_risk_pct": float(ev.baldur_risk_pct),
        "baldur_max_leverage": float(ev.baldur_max_leverage),
        "baldur_capital_cap": float(ev.baldur_capital_cap),
        "baldur_tp_atr": float(ev.baldur_tp_atr),
        "baldur_sl_atr": float(ev.baldur_sl_atr),
        "baldur_max_bars": int(ev.baldur_max_bars),
        "baldur_min_score_trade": float(ev.baldur_min_score_trade),
        "baldur_top_risk_min": float(ev.baldur_top_risk_min),
        "baldur_warning_exit_score": float(ev.baldur_warning_exit_score),
        "freya_risk_pct": float(ev.freya_risk_pct),
        "freya_max_leverage": float(ev.freya_max_leverage),
        "freya_capital_cap": float(ev.freya_capital_cap),
        "freya_tp_atr": float(ev.freya_tp_atr),
        "freya_sl_atr": float(ev.freya_sl_atr),
        "freya_max_bars": int(ev.freya_max_bars),
        "freya_min_score_trade": float(ev.freya_min_score_trade),
        "freya_max_concurrent_positions": int(ev.freya_max_concurrent_positions),
        "freya_baldur_block_threshold": float(ev.freya_baldur_block_threshold),
        "freya_pyramid_add_notional_fraction": float(ev.freya_pyramid_add_notional_fraction),
        "freya_pyramid_add_trigger_wave": float(ev.freya_pyramid_add_trigger_wave),
        "freya_pyramid_add_top_risk_max": float(ev.freya_pyramid_add_top_risk_max),
        "freya_min_runup_atr": float(getattr(ev, "freya_min_runup_atr", 0.6)),
        "baldur_min_runup_atr": float(ev.baldur_min_runup_atr),
        "baldur_upper_wick_min": float(ev.baldur_upper_wick_min),
        "pump_material_drawdown_atr": float(ev.pump_material_drawdown_atr),
        "stop_market_penetration_factor": float(ev.stop_market_penetration_factor),
        "stop_market_slip_atr_cap": float(ev.stop_market_slip_atr_cap),
        "norse_drawdown_penalty_weight": float(ev.norse_drawdown_penalty_weight),
    }

    _log("priming Thor event cache…")
    _prime_thor_event_cache(prepared_symbols, base_params)
    _log("Thor event cache primed")

    _log("running baseline eval…")
    best_params, best_result, tuning_df = _tune_params(prepared_symbols, base_params)

    # Preserve Turbo-tuner diagnostics before overwriting best_result with the
    # full-detail eval (which does not carry these keys).
    _TUNER_KEYS = (
        "wf_fold_metrics", "sensitivity_rows", "optuna_study",
        "n_optuna_trials", "cache_vs_sim", "learned_filter",
    )
    _tuner_meta = {k: best_result.get(k) for k in _TUNER_KEYS}

    _log("building final detailed result…")
    if (
        "trade_df" not in best_result
        or "pump_summary_df" not in best_result
        or "pump_ledger_df" not in best_result
        or "full_agent_summary" not in best_result
    ):
        detailed_result = _evaluate_params(
            prepared_symbols,
            best_params,
            include_pump_outputs=True,
            sim_logger=_log,
            detailed_progress=True,
        )
        for key, value in best_result.items():
            if key not in detailed_result:
                detailed_result[key] = value
        best_result = detailed_result

    # Restore tuner diagnostics so _report_text() can render them.
    for k, v in _tuner_meta.items():
        if v is not None:
            best_result[k] = v

    _log(f"tuning done — growth={best_result['portfolio_summary']['growth_pct']:.2f}%  dd={best_result['portfolio_summary']['max_drawdown_pct']:.2f}%  trades={best_result['portfolio_summary']['trades']}")
    best_result["params"] = best_params
    diagnostics = _build_trade_diagnostics(best_result, best_params)
    best_result["artifact_paths"] = artifact_paths
    best_result["diagnostics"] = diagnostics

    best_result["full_agent_summary"].to_csv(artifact_paths["summary_csv"], index=False)
    best_result["trade_df"].to_csv(artifact_paths["trades_csv"], index=False)
    best_result["pump_summary_df"].to_csv(artifact_paths["pump_stats_csv"], index=False)
    best_result["pump_ledger_df"].to_csv(artifact_paths["pump_ledger_csv"], index=False)
    tuning_df.to_csv(artifact_paths["tuning_csv"], index=False)
    diagnostics["trade_diag_df"].to_csv(artifact_paths["trade_diagnostics_csv"], index=False)
    diagnostics["loss_diag_df"].to_csv(artifact_paths["loss_diagnostics_csv"], index=False)
    diagnostics["big_win_diag_df"].to_csv(artifact_paths["big_win_diagnostics_csv"], index=False)
    diagnostics["feature_compare_df"].to_csv(artifact_paths["feature_compare_csv"], index=False)
    diagnostics["reason_summary_df"].to_csv(artifact_paths["reason_summary_csv"], index=False)

    report = _report_text(best_result, prepared_symbols, tuning_df, artifact_paths, diagnostics)
    artifact_paths["report"].write_text(report, encoding="utf-8")

    return best_result, tuning_df


if __name__ == "__main__":
    result, tuning = run_year_simulation()
    print(result["full_agent_summary"].to_string(index=False))
    print(f"\nFinal capital: ${result['portfolio_summary']['final_capital']:,.2f} ({result['portfolio_summary']['growth_pct']:+.2f}%)")
    print(f"Active agents: {', '.join(result['active_agents'])}")
    artifacts = result.get("artifact_paths", {})
    run_dir = artifacts.get("run_dir")
    if run_dir:
        print(f"Wrote timestamped run artifacts to {run_dir}")
