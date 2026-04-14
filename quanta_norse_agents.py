"""
Norse specialist identity and rule-based companion agents.

Internal specialist keys stay unchanged for compatibility. This module adds:
- user-facing display names
- live-execution allowlist helpers
- Thor context tracking
- rule-based Baldur/Freya cache validators
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # no-op fallback
        def _wrap(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap

DISPLAY_AGENT_NAMES: Dict[str, str] = {
    "nike": "Thor",
    "baldur": "Baldur",
    "freya": "Freya",
    "athena": "Athena",
    "ares": "Ares",
    "hermes": "Hermes",
    "artemis": "Artemis",
    "chronos": "Chronos",
    "hephaestus": "Hephaestus",
}


def display_agent_name(agent_key: Optional[str]) -> str:
    key = str(agent_key or "").strip().lower()
    if not key:
        return "Unknown"
    return DISPLAY_AGENT_NAMES.get(key, key.title())


def parse_live_model_specialists(raw_value: object) -> set[str]:
    if isinstance(raw_value, (list, tuple, set)):
        return {str(x).strip().lower() for x in raw_value if str(x).strip()}
    text = str(raw_value or "").strip()
    if not text:
        return {"nike"}
    return {part.strip().lower() for part in text.split(",") if part.strip()}


@dataclass
class ThorContext:
    symbol: str
    entry_price: float
    atr: float
    start_bar: int
    expiry_bar: int
    score: float
    tier: str
    source_agent: str = "Thor"


@dataclass
class SparseFeatureContext:
    symbol: str
    feature_map: Dict[int, np.ndarray]
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    times: np.ndarray
    atrs: np.ndarray
    vol_avg: np.ndarray
    volume_ratio: np.ndarray
    mean_volume_ratio: np.ndarray
    volume_ratio_slope: np.ndarray
    quote_volume: np.ndarray
    quote_volume_slope: np.ndarray
    vpin: np.ndarray
    vpin_slope: np.ndarray
    taker_imbalance: np.ndarray
    taker_slope: np.ndarray
    regime_state: np.ndarray
    weighted_trend: np.ndarray
    bull_ratio: np.ndarray
    bear_ratio: np.ndarray
    bs_prob: np.ndarray
    bs_time_decay: np.ndarray
    bs_iv_ratio: np.ndarray
    impulse_body_eff: np.ndarray
    impulse_taker_persist: np.ndarray
    pre_impulse_r2: np.ndarray
    atr_rank: np.ndarray
    depth_delta: np.ndarray
    vol_delta: np.ndarray
    vpin_delta: np.ndarray
    close_pos: np.ndarray
    upper_wick_ratio: np.ndarray
    participation_score: np.ndarray
    flow_exhaustion_score: np.ndarray


@njit(fastmath=True)
def _calc_atr_njit(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        t = hl
        if hc > t:
            t = hc
        if lc > t:
            t = lc
        tr[i] = t
    atr = np.zeros(n, dtype=np.float64)
    if n <= period:
        return atr
    s = 0.0
    for k in range(1, period + 1):
        s += tr[k]
    atr[period] = s / period
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    highs = np.ascontiguousarray(highs, dtype=np.float64)
    lows = np.ascontiguousarray(lows, dtype=np.float64)
    closes = np.ascontiguousarray(closes, dtype=np.float64)
    return _calc_atr_njit(highs, lows, closes, int(period))


def calc_vol_avg20(volumes: np.ndarray) -> np.ndarray:
    n = len(volumes)
    out = np.zeros(n, dtype=np.float64)
    for i in range(20, n):
        out[i] = volumes[i - 20:i].mean()
    return out


def _rolling_slope(values: np.ndarray, window: int = 5) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2) + 1e-12
    for i in range(window - 1, n):
        y = values[i - window + 1:i + 1]
        y_mean = y.mean()
        out[i] = np.sum((x - x_mean) * (y - y_mean)) / denom
    return out


def _forward_fill(values: np.ndarray, default: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if len(out) == 0:
        return out
    last = default
    for i in range(len(out)):
        v = out[i]
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return out


def neutralize_feature_vector(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).copy()
    if out.shape[0] < 278:
        padded = np.zeros(278, dtype=np.float64)
        padded[:out.shape[0]] = out
        out = padded

    bad = ~np.isfinite(out)
    if not bad.any():
        return out

    out[bad] = 0.0
    neutral_defaults = {
        49: 0.5, 50: 0.5, 51: 0.0, 52: 0.0,
        165: 1.0, 166: 1.0, 167: 1.0, 168: 1.0, 169: 1.0, 170: 1.0, 171: 1.0,
        193: 0.5, 194: 0.5, 195: 0.5, 196: 0.5, 197: 0.5, 198: 0.5, 199: 0.5,
        207: 0.0, 208: 0.0, 209: 0.0, 210: 0.0, 211: 0.0, 212: 0.0, 213: 0.0,
        231: 0.5,
        239: 0.5, 240: 0.0, 241: 0.0, 242: 0.0, 243: 0.0, 244: 0.0, 245: 0.0,
        246: 0.0, 247: 0.0, 248: 0.0, 249: 0.0,
        254: 0.0, 255: 0.0, 256: 0.5, 257: 0.0,
        270: 0.0, 271: 0.0, 272: 0.0, 273: 0.5, 274: 0.0,
        275: 0.4, 276: 0.5, 277: 1.0,
    }
    for idx, value in neutral_defaults.items():
        if bad[idx]:
            out[idx] = value
    return out


def build_sparse_feature_context(df, features_by_pos: Dict[int, np.ndarray]) -> SparseFeatureContext:
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    times = df["open_time"].values.astype(np.int64)
    n = len(closes)

    atrs = calc_atr(highs, lows, closes)
    vol_avg = calc_vol_avg20(volumes)
    candle_range = np.maximum(highs - lows, 1e-12)
    close_pos = (closes - lows) / candle_range
    upper_wick_ratio = (highs - np.maximum(opens, closes)) / candle_range
    quote_volume = closes * volumes
    raw_volume_ratio = np.where(vol_avg > 0, volumes / np.maximum(vol_avg, 1e-12), 1.0)

    vpin = np.full(n, np.nan, dtype=np.float64)
    taker = np.full(n, np.nan, dtype=np.float64)
    regime = np.full(n, np.nan, dtype=np.float64)
    weighted = np.full(n, np.nan, dtype=np.float64)
    bull_ratio = np.full(n, np.nan, dtype=np.float64)
    bear_ratio = np.full(n, np.nan, dtype=np.float64)
    bs_prob = np.full(n, np.nan, dtype=np.float64)
    bs_time = np.full(n, np.nan, dtype=np.float64)
    bs_iv = np.full(n, np.nan, dtype=np.float64)
    impulse_body_eff = np.full(n, np.nan, dtype=np.float64)
    impulse_taker = np.full(n, np.nan, dtype=np.float64)
    pre_impulse_r2 = np.full(n, np.nan, dtype=np.float64)
    atr_rank = np.full(n, np.nan, dtype=np.float64)
    depth_delta = np.full(n, np.nan, dtype=np.float64)
    vol_delta = np.full(n, np.nan, dtype=np.float64)
    vpin_delta = np.full(n, np.nan, dtype=np.float64)
    mean_volume_ratio = np.full(n, np.nan, dtype=np.float64)
    sparse_map: Dict[int, np.ndarray] = {}

    for pos, feat in features_by_pos.items():
        vec = neutralize_feature_vector(feat)
        sparse_map[int(pos)] = vec
        bull_ratio[pos] = vec[49]
        bear_ratio[pos] = vec[50]
        weighted[pos] = vec[52]
        mean_volume_ratio[pos] = float(np.mean(vec[165:172]))
        vpin[pos] = vec[193]
        taker[pos] = vec[207]
        regime[pos] = vec[231]
        bs_prob[pos] = vec[275]
        bs_time[pos] = vec[276]
        bs_iv[pos] = vec[277]
        impulse_body_eff[pos] = vec[270]
        impulse_taker[pos] = vec[271]
        pre_impulse_r2[pos] = vec[272]
        atr_rank[pos] = vec[273]
        depth_delta[pos] = vec[274]
        vol_delta[pos] = vec[262]
        vpin_delta[pos] = vec[267]

    vpin = _forward_fill(vpin, 0.5)
    taker = _forward_fill(taker, 0.0)
    regime = _forward_fill(regime, 0.5)
    weighted = _forward_fill(weighted, 0.0)
    bull_ratio = _forward_fill(bull_ratio, 0.5)
    bear_ratio = _forward_fill(bear_ratio, 0.5)
    bs_prob = _forward_fill(bs_prob, 0.4)
    bs_time = _forward_fill(bs_time, 0.5)
    bs_iv = _forward_fill(bs_iv, 1.0)
    impulse_body_eff = _forward_fill(impulse_body_eff, 0.0)
    impulse_taker = _forward_fill(impulse_taker, 0.0)
    pre_impulse_r2 = _forward_fill(pre_impulse_r2, 0.0)
    atr_rank = _forward_fill(atr_rank, 0.5)
    depth_delta = _forward_fill(depth_delta, 0.0)
    vol_delta = _forward_fill(vol_delta, 0.0)
    vpin_delta = _forward_fill(vpin_delta, 0.0)
    mean_volume_ratio = _forward_fill(mean_volume_ratio, 1.0)

    quote_volume_slope = _rolling_slope(np.log1p(np.maximum(quote_volume, 0.0)), 5)
    volume_ratio_slope = _rolling_slope(raw_volume_ratio, 5)
    taker_slope = _rolling_slope(taker, 5)
    vpin_slope = _rolling_slope(vpin, 5)

    participation_score = 100.0 * (
        0.35 * np.clip((mean_volume_ratio - 1.0) / 1.5, 0.0, 1.0)
        + 0.20 * np.clip((quote_volume_slope + 0.02) / 0.12, 0.0, 1.0)
        + 0.20 * np.clip((taker + 0.10) / 0.50, 0.0, 1.0)
        + 0.10 * np.clip(impulse_taker, 0.0, 1.0)
        + 0.15 * np.clip((bull_ratio - bear_ratio + 0.2) / 0.8, 0.0, 1.0)
    )
    flow_exhaustion_score = 100.0 * (
        0.30 * np.clip((vpin - 0.50) / 0.30, 0.0, 1.0)
        + 0.20 * np.clip((vpin_slope + 0.01) / 0.08, 0.0, 1.0)
        + 0.20 * np.clip((-taker_slope + 0.01) / 0.10, 0.0, 1.0)
        + 0.15 * np.clip((upper_wick_ratio - 0.20) / 0.45, 0.0, 1.0)
        + 0.15 * np.clip((0.55 - close_pos) / 0.55, 0.0, 1.0)
    )

    return SparseFeatureContext(
        symbol=str(df.attrs.get("symbol", "")),
        feature_map=sparse_map,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        times=times,
        atrs=atrs,
        vol_avg=vol_avg,
        volume_ratio=raw_volume_ratio,
        mean_volume_ratio=mean_volume_ratio,
        volume_ratio_slope=volume_ratio_slope,
        quote_volume=quote_volume,
        quote_volume_slope=quote_volume_slope,
        vpin=vpin,
        vpin_slope=vpin_slope,
        taker_imbalance=taker,
        taker_slope=taker_slope,
        regime_state=regime,
        weighted_trend=weighted,
        bull_ratio=bull_ratio,
        bear_ratio=bear_ratio,
        bs_prob=bs_prob,
        bs_time_decay=bs_time,
        bs_iv_ratio=bs_iv,
        impulse_body_eff=impulse_body_eff,
        impulse_taker_persist=impulse_taker,
        pre_impulse_r2=pre_impulse_r2,
        atr_rank=atr_rank,
        depth_delta=depth_delta,
        vol_delta=vol_delta,
        vpin_delta=vpin_delta,
        close_pos=close_pos,
        upper_wick_ratio=upper_wick_ratio,
        participation_score=participation_score,
        flow_exhaustion_score=flow_exhaustion_score,
    )


def compute_pump_state(ctx: SparseFeatureContext, start_bar: int, bar_idx: int, material_drawdown_atr: float = 1.0) -> dict:
    start = max(0, int(start_bar))
    bar = max(start, int(bar_idx))
    atr0 = max(float(ctx.atrs[start]), 1e-12)
    highs = ctx.highs[start:bar + 1]
    lows = ctx.lows[start:bar + 1]
    closes = ctx.closes[start:bar + 1]
    peak_offsets = np.maximum.accumulate(highs)
    peak_high = float(peak_offsets[-1]) if len(peak_offsets) else float(ctx.closes[start])
    peak_offset = int(np.argmax(highs)) if len(highs) else 0
    peak_bar = start + peak_offset
    drawdowns = (peak_offsets - lows) / atr0 if len(lows) else np.array([0.0], dtype=np.float64)
    current_drawdown_atr = max(0.0, (peak_high - ctx.closes[bar]) / atr0)
    max_drawdown_atr = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    runup_atr = max(0.0, (peak_high - ctx.closes[start]) / atr0)
    volume_decay = ctx.volume_ratio[bar] - ctx.volume_ratio[min(peak_bar, len(ctx.volume_ratio) - 1)]
    quote_decay = ctx.quote_volume_slope[bar] - ctx.quote_volume_slope[min(peak_bar, len(ctx.quote_volume_slope) - 1)]
    bars_since_peak = max(0, bar - peak_bar)
    wave_strength = 100.0 * (
        0.28 * _clip01((ctx.participation_score[bar] - 45.0) / 45.0)
        + 0.22 * _clip01((ctx.taker_slope[bar] + 0.02) / 0.10)
        + 0.18 * _clip01((ctx.bs_prob[bar] - 0.35) / 0.35)
        + 0.18 * _clip01((ctx.weighted_trend[bar] + 0.15) / 0.60)
        + 0.14 * (1.0 - _clip01(current_drawdown_atr / max(material_drawdown_atr, 1e-12)))
    )
    top_risk = 100.0 * (
        0.22 * _clip01((runup_atr - 1.5) / 3.0)
        + 0.20 * _clip01(current_drawdown_atr / max(material_drawdown_atr, 1e-12))
        + 0.18 * _clip01((-volume_decay) / 1.0)
        + 0.14 * _clip01((-quote_decay) / 0.10)
        + 0.14 * _clip01((ctx.flow_exhaustion_score[bar] - 45.0) / 55.0)
        + 0.12 * _clip01(bars_since_peak / 6.0)
    )
    return {
        "runup_atr": runup_atr,
        "current_drawdown_atr": current_drawdown_atr,
        "max_drawdown_atr": max_drawdown_atr,
        "bars_since_peak": bars_since_peak,
        "peak_bar": peak_bar,
        "peak_high": peak_high,
        "volume_decay": volume_decay,
        "quote_decay": quote_decay,
        "wave_strength_score": wave_strength,
        "top_risk_score": top_risk,
    }


def _pump_path_analytics(
    ctx: SparseFeatureContext,
    start_bar: int,
    end_bar: int,
    material_drawdown_atr: float = 1.0,
) -> dict[str, object]:
    start = max(0, int(start_bar))
    end = min(int(end_bar), len(ctx.closes) - 1)
    if end < start:
        return {
            "start_bar": start,
            "end_bar": end,
            "entry_price": 0.0,
            "atr0": 1e-12,
            "offsets": np.zeros(0, dtype=np.int64),
            "times": np.zeros(0, dtype=np.int64),
            "runup_series": np.zeros(0, dtype=np.float64),
            "drawdown_peak_series": np.zeros(0, dtype=np.float64),
            "drawdown_entry_series": np.zeros(0, dtype=np.float64),
            "close_drawdown_series": np.zeros(0, dtype=np.float64),
            "close_delta_series": np.zeros(0, dtype=np.float64),
            "bars_since_peak_series": np.zeros(0, dtype=np.int64),
            "wave_strength_series": np.zeros(0, dtype=np.float64),
            "top_risk_series": np.zeros(0, dtype=np.float64),
            "peak_bar": start,
            "peak_high": 0.0,
            "max_runup_atr": 0.0,
            "max_mae_from_entry_atr": 0.0,
            "max_mae_from_peak_atr": 0.0,
            "max_mae_entry_bar": start,
            "time_to_peak_bars": 0,
            "time_to_material_drawdown_bars": None,
            "volume_decay_after_peak": 0.0,
            "final_wave_strength_score": 0.0,
            "final_top_risk_score": 0.0,
            "summary_max_drawdown_atr": 0.0,
        }

    window = slice(start, end + 1)
    highs = np.asarray(ctx.highs[window], dtype=np.float64)
    lows = np.asarray(ctx.lows[window], dtype=np.float64)
    closes = np.asarray(ctx.closes[window], dtype=np.float64)
    volume_ratio = np.asarray(ctx.volume_ratio[window], dtype=np.float64)
    quote_volume_slope = np.asarray(ctx.quote_volume_slope[window], dtype=np.float64)
    participation_score = np.asarray(ctx.participation_score[window], dtype=np.float64)
    taker_slope = np.asarray(ctx.taker_slope[window], dtype=np.float64)
    bs_prob = np.asarray(ctx.bs_prob[window], dtype=np.float64)
    weighted_trend = np.asarray(ctx.weighted_trend[window], dtype=np.float64)
    flow_exhaustion_score = np.asarray(ctx.flow_exhaustion_score[window], dtype=np.float64)
    times = np.asarray(ctx.times[window], dtype=np.int64)

    entry_price = float(ctx.closes[start])
    atr0 = max(float(ctx.atrs[start]), 1e-12)
    offsets = np.arange(len(highs), dtype=np.int64)

    running_peak = np.maximum.accumulate(highs)
    runup_series = np.maximum(0.0, (running_peak - entry_price) / atr0)
    drawdown_peak_series = np.maximum(0.0, (running_peak - lows) / atr0)
    drawdown_entry_series = np.maximum(0.0, (entry_price - lows) / atr0)
    close_drawdown_series = np.maximum(0.0, (running_peak - closes) / atr0)
    close_delta_series = (closes - entry_price) / atr0

    prev_running_peak = np.empty_like(running_peak)
    prev_running_peak[0] = -np.inf
    if len(running_peak) > 1:
        prev_running_peak[1:] = running_peak[:-1]
    first_peak_offsets = np.maximum.accumulate(np.where(highs > prev_running_peak, offsets, 0))
    peak_indices = start + first_peak_offsets
    bars_since_peak_series = offsets - first_peak_offsets

    volume_decay_series = volume_ratio - ctx.volume_ratio[peak_indices]
    quote_decay_series = quote_volume_slope - ctx.quote_volume_slope[peak_indices]
    material_scale = max(float(material_drawdown_atr), 1e-12)
    current_drawdown_series = close_drawdown_series

    wave_strength_series = 100.0 * (
        0.28 * np.clip((participation_score - 45.0) / 45.0, 0.0, 1.0)
        + 0.22 * np.clip((taker_slope + 0.02) / 0.10, 0.0, 1.0)
        + 0.18 * np.clip((bs_prob - 0.35) / 0.35, 0.0, 1.0)
        + 0.18 * np.clip((weighted_trend + 0.15) / 0.60, 0.0, 1.0)
        + 0.14 * (1.0 - np.clip(current_drawdown_series / material_scale, 0.0, 1.0))
    )
    top_risk_series = 100.0 * (
        0.22 * np.clip((runup_series - 1.5) / 3.0, 0.0, 1.0)
        + 0.20 * np.clip(current_drawdown_series / material_scale, 0.0, 1.0)
        + 0.18 * np.clip((-volume_decay_series) / 1.0, 0.0, 1.0)
        + 0.14 * np.clip((-quote_decay_series) / 0.10, 0.0, 1.0)
        + 0.14 * np.clip((flow_exhaustion_score - 45.0) / 55.0, 0.0, 1.0)
        + 0.12 * np.clip(bars_since_peak_series.astype(np.float64) / 6.0, 0.0, 1.0)
    )

    peak_offset = int(np.argmax(runup_series)) if len(runup_series) else 0
    peak_bar = start + peak_offset
    peak_high = float(running_peak[peak_offset]) if len(running_peak) else entry_price
    max_runup_atr = float(runup_series.max()) if len(runup_series) else 0.0
    max_mae_from_entry_atr = float(drawdown_entry_series.max()) if len(drawdown_entry_series) else 0.0
    max_mae_from_peak_atr = float(drawdown_peak_series.max()) if len(drawdown_peak_series) else 0.0
    max_mae_entry_bar = start + int(np.argmax(drawdown_entry_series)) if len(drawdown_entry_series) else start

    first_material_dd = None
    peak_high_last = entry_price
    peak_bar_last = start
    for offset, high in enumerate(highs):
        bar = start + int(offset)
        if float(high) >= peak_high_last:
            peak_high_last = float(high)
            peak_bar_last = bar
        if first_material_dd is None and float(close_drawdown_series[offset]) >= float(material_drawdown_atr):
            first_material_dd = bar - peak_bar_last

    return {
        "start_bar": start,
        "end_bar": end,
        "entry_price": entry_price,
        "atr0": atr0,
        "offsets": offsets,
        "times": times,
        "runup_series": runup_series,
        "drawdown_peak_series": drawdown_peak_series,
        "drawdown_entry_series": drawdown_entry_series,
        "close_drawdown_series": close_drawdown_series,
        "close_delta_series": close_delta_series,
        "bars_since_peak_series": bars_since_peak_series,
        "wave_strength_series": wave_strength_series,
        "top_risk_series": top_risk_series,
        "peak_bar": peak_bar,
        "peak_high": peak_high,
        "max_runup_atr": max_runup_atr,
        "max_mae_from_entry_atr": max_mae_from_entry_atr,
        "max_mae_from_peak_atr": max_mae_from_peak_atr,
        "max_mae_entry_bar": max_mae_entry_bar,
        "time_to_peak_bars": max(0, peak_bar - start),
        "time_to_material_drawdown_bars": None if first_material_dd is None else int(first_material_dd),
        "volume_decay_after_peak": float(ctx.volume_ratio[end] - ctx.volume_ratio[min(peak_bar, len(ctx.volume_ratio) - 1)]),
        "final_wave_strength_score": float(wave_strength_series[-1]) if len(wave_strength_series) else 0.0,
        "final_top_risk_score": float(top_risk_series[-1]) if len(top_risk_series) else 0.0,
        "summary_max_drawdown_atr": float(close_drawdown_series.max()) if len(close_drawdown_series) else 0.0,
    }


def score_thor_signal(sig: dict, ctx: SparseFeatureContext) -> float:
    i = int(sig["bar_idx"])
    base = _clip01(float(sig.get("score", 0.0)) / 100.0)
    participation = _clip01(ctx.participation_score[i] / 100.0)
    trend = _clip01((ctx.weighted_trend[i] + 0.10) / 0.70)
    bs_edge = _clip01((ctx.bs_prob[i] - 0.35) / 0.35)
    continuation = _clip01(
        0.5 * ctx.impulse_taker_persist[i]
        + 0.3 * _clip01((ctx.pre_impulse_r2[i] + 1.0) / 2.0)
        + 0.2 * _clip01(ctx.atr_rank[i])
    )
    fade_penalty = _clip01((-ctx.quote_volume_slope[i] + 0.02) / 0.10)
    exhaustion_penalty = _clip01((ctx.flow_exhaustion_score[i] - 55.0) / 45.0)
    score = 100.0 * (
        0.30 * base
        + 0.24 * participation
        + 0.16 * trend
        + 0.16 * bs_edge
        + 0.14 * continuation
    ) - 18.0 * fade_penalty - 12.0 * exhaustion_penalty
    return max(0.0, min(100.0, float(score)))


def score_baldur_warning(sig: dict, ctx: SparseFeatureContext, thor_ctx: ThorContext, cfg_events) -> float:
    i = int(sig["bar_idx"])
    pump = compute_pump_state(ctx, thor_ctx.start_bar, i, float(getattr(cfg_events, "pump_material_drawdown_atr", 1.0)))
    # Prefer OHLCV-derived values stored directly in the warning dict (always populated).
    # The sparse feature_ctx may have zeros at Baldur warning bars for cache-loaded symbols.
    raw_wick = sig.get("upper_wick_ratio", None)
    raw_close_pos = sig.get("close_pos", None)
    wick_val = float(raw_wick) if raw_wick is not None else float(ctx.upper_wick_ratio[i])
    close_pos_val = float(raw_close_pos) if raw_close_pos is not None else float(ctx.close_pos[i])
    wick = _clip01((wick_val - 0.25) / 0.50)
    weak_close = _clip01((0.55 - close_pos_val) / 0.55)
    # flow_exhaustion_score from ctx (only available when features computed at warning bar)
    flow_ex = float(ctx.flow_exhaustion_score[i])
    score = 100.0 * (
        0.38 * _clip01(pump["top_risk_score"] / 100.0)
        + 0.22 * wick
        + 0.16 * weak_close
        + 0.14 * _clip01((flow_ex - 40.0) / 60.0)
        + 0.10 * _clip01((pump["runup_atr"] - 1.5) / 3.0)
    )
    return max(0.0, min(100.0, float(score)))


def score_baldur_signal(sig: dict, ctx: SparseFeatureContext, thor_ctx: ThorContext, cfg_events) -> tuple[float, float]:
    i = int(sig["bar_idx"])
    warning_bar = int(sig.get("warning_bar_idx", max(thor_ctx.start_bar, i - 1)))
    warning_score = score_baldur_warning(
        {"bar_idx": warning_bar},
        ctx,
        thor_ctx,
        cfg_events,
    )
    pump = compute_pump_state(ctx, thor_ctx.start_bar, i, float(getattr(cfg_events, "pump_material_drawdown_atr", 1.0)))
    confirm = _clip01(float(sig.get("confirm_drop_pct", 0.0)) / 1.0)
    score = 100.0 * (
        0.32 * _clip01(warning_score / 100.0)
        + 0.30 * _clip01(pump["top_risk_score"] / 100.0)
        + 0.20 * confirm
        + 0.18 * _clip01((ctx.flow_exhaustion_score[i] - 45.0) / 55.0)
    )
    return max(0.0, min(100.0, float(score))), warning_score


def score_freya_signal(sig: dict, ctx: SparseFeatureContext, thor_ctx: ThorContext, cfg_events) -> tuple[float, float]:
    i = int(sig["bar_idx"])
    pump = compute_pump_state(ctx, thor_ctx.start_bar, i, float(getattr(cfg_events, "pump_material_drawdown_atr", 1.0)))
    base = _clip01(float(sig.get("score", 0.0)) / 100.0)
    wave = _clip01(pump["wave_strength_score"] / 100.0)
    participation = _clip01(ctx.participation_score[i] / 100.0)
    healthy_close = _clip01((ctx.close_pos[i] - 0.45) / 0.55)
    rising_wave = _clip01((ctx.quote_volume_slope[i] + 0.01) / 0.08)
    score = 100.0 * (
        0.28 * base
        + 0.28 * wave
        + 0.18 * participation
        + 0.14 * healthy_close
        + 0.12 * rising_wave
    )
    return max(0.0, min(100.0, float(score))), pump["top_risk_score"]


# Label codes for njit sims (njit cannot return Python strings)
_LBL_TP = 1
_LBL_SL = 2
_LBL_TIMEOUT = 3
_LBL_CODE_TO_STR = {_LBL_TP: "TP", _LBL_SL: "SL", _LBL_TIMEOUT: "TIMEOUT"}


@njit(fastmath=True)
def _sim_directional_exit_njit(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_idx: int,
    entry: float,
    atr_i: float,
    is_bearish: int,
    tp_atr: float,
    sl_atr: float,
    max_bars: int,
    penetration_factor: float,
    slip_cap_atr: float,
):
    n = len(closes)
    if atr_i <= 0.0:
        return _LBL_TIMEOUT, 0.0, entry_idx, entry, 0.0

    if is_bearish == 1:
        tp_price = entry - tp_atr * atr_i
        stop_price = entry + sl_atr * atr_i
    else:
        tp_price = entry + tp_atr * atr_i
        stop_price = entry - sl_atr * atr_i

    last_bar = entry_idx + max_bars
    if last_bar > n - 1:
        last_bar = n - 1

    j = entry_idx + 1
    while j <= last_bar:
        open_j = opens[j]
        high_j = highs[j]
        low_j = lows[j]
        close_j = closes[j]

        if is_bearish == 1:
            if high_j >= stop_price:
                if open_j >= stop_price:
                    exit_price = open_j
                else:
                    penetration = high_j - stop_price
                    cap = slip_cap_atr * atr_i
                    step = penetration_factor * penetration
                    if step > cap:
                        step = cap
                    exit_price = stop_price + step
                realized = (entry - exit_price) / atr_i
                slip = (exit_price - stop_price) / atr_i
                if slip < 0.0:
                    slip = 0.0
                return _LBL_SL, realized, j, exit_price, slip
            if low_j <= tp_price:
                realized = (entry - tp_price) / atr_i
                return _LBL_TP, realized, j, tp_price, 0.0
        else:
            if low_j <= stop_price:
                if open_j <= stop_price:
                    exit_price = open_j
                else:
                    penetration = stop_price - low_j
                    cap = slip_cap_atr * atr_i
                    step = penetration_factor * penetration
                    if step > cap:
                        step = cap
                    exit_price = stop_price - step
                realized = (exit_price - entry) / atr_i
                slip = (stop_price - exit_price) / atr_i
                if slip < 0.0:
                    slip = 0.0
                return _LBL_SL, realized, j, exit_price, slip
            if high_j >= tp_price:
                realized = (tp_price - entry) / atr_i
                return _LBL_TP, realized, j, tp_price, 0.0

        if j == last_bar:
            if is_bearish == 1:
                realized = (entry - close_j) / atr_i
            else:
                realized = (close_j - entry) / atr_i
            return _LBL_TIMEOUT, realized, j, close_j, 0.0

        j += 1

    return _LBL_TIMEOUT, 0.0, last_bar, closes[last_bar], 0.0


def simulate_directional_exit_stop_market(
    sig: dict,
    ctx: SparseFeatureContext,
    tp_atr: float,
    sl_atr: float,
    max_bars: int,
    penetration_factor: float = 0.35,
    slip_cap_atr: float = 0.25,
) -> dict:
    i = int(sig["bar_idx"])
    entry = float(sig["close"])
    atr_i = float(sig["atr"])
    direction = str(sig.get("direction", "BULLISH")).upper()
    is_bearish = 1 if direction == "BEARISH" else 0
    if atr_i <= 0:
        return {"label": "TIMEOUT", "realized_atr": 0.0, "exit_bar": i, "exit_price": entry, "stop_slip_atr": 0.0}

    label_code, realized, exit_bar, exit_price, slip = _sim_directional_exit_njit(
        ctx.opens, ctx.highs, ctx.lows, ctx.closes,
        i, entry, atr_i, is_bearish,
        float(tp_atr), float(sl_atr), int(max_bars),
        float(penetration_factor), float(slip_cap_atr),
    )
    return {
        "label": _LBL_CODE_TO_STR[int(label_code)],
        "realized_atr": float(realized),
        "exit_bar": int(exit_bar),
        "exit_price": float(exit_price),
        "stop_slip_atr": float(slip),
    }


@njit(fastmath=True)
def _clip01_njit(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


@njit(fastmath=True)
def _thor_entry_strength_njit(
    entry_idx: int,
    participation_score: np.ndarray,
    bs_prob: np.ndarray,
    weighted_trend: np.ndarray,
    quote_volume_slope: np.ndarray,
    close_pos: np.ndarray,
) -> float:
    participation = _clip01_njit((participation_score[entry_idx] - 55.0) / 45.0)
    bs_edge = _clip01_njit((bs_prob[entry_idx] - 0.40) / 0.30)
    trend = _clip01_njit((weighted_trend[entry_idx] + 0.05) / 0.55)
    quote_support = _clip01_njit((quote_volume_slope[entry_idx] + 0.01) / 0.08)
    healthy_close = _clip01_njit((close_pos[entry_idx] - 0.45) / 0.45)
    return (
        0.30 * participation
        + 0.25 * bs_edge
        + 0.20 * trend
        + 0.15 * quote_support
        + 0.10 * healthy_close
    )


@njit(fastmath=True)
def _thor_wave_top_scores_njit(
    entry: float,
    atr_i: float,
    peak_high: float,
    peak_bar: int,
    current_bar: int,
    close_j: float,
    volume_ratio: np.ndarray,
    quote_volume_slope: np.ndarray,
    taker_slope: np.ndarray,
    bs_prob: np.ndarray,
    weighted_trend: np.ndarray,
    participation_score: np.ndarray,
    flow_exhaustion_score: np.ndarray,
    material_drawdown_atr: float,
):
    current_drawdown_atr = 0.0
    if peak_high > close_j:
        current_drawdown_atr = (peak_high - close_j) / atr_i

    runup_atr = 0.0
    if peak_high > entry:
        runup_atr = (peak_high - entry) / atr_i

    dd_scale = material_drawdown_atr
    if dd_scale <= 0.0:
        dd_scale = 1e-12

    volume_decay = volume_ratio[current_bar] - volume_ratio[peak_bar]
    quote_decay = quote_volume_slope[current_bar] - quote_volume_slope[peak_bar]
    bars_since_peak = current_bar - peak_bar
    if bars_since_peak < 0:
        bars_since_peak = 0

    wave_strength = 100.0 * (
        0.28 * _clip01_njit((participation_score[current_bar] - 45.0) / 45.0)
        + 0.22 * _clip01_njit((taker_slope[current_bar] + 0.02) / 0.10)
        + 0.18 * _clip01_njit((bs_prob[current_bar] - 0.35) / 0.35)
        + 0.18 * _clip01_njit((weighted_trend[current_bar] + 0.15) / 0.60)
        + 0.14 * (1.0 - _clip01_njit(current_drawdown_atr / dd_scale))
    )
    top_risk = 100.0 * (
        0.22 * _clip01_njit((runup_atr - 1.5) / 3.0)
        + 0.20 * _clip01_njit(current_drawdown_atr / dd_scale)
        + 0.18 * _clip01_njit((-volume_decay) / 1.0)
        + 0.14 * _clip01_njit((-quote_decay) / 0.10)
        + 0.14 * _clip01_njit((flow_exhaustion_score[current_bar] - 45.0) / 55.0)
        + 0.12 * _clip01_njit(bars_since_peak / 6.0)
    )
    return wave_strength, top_risk, bars_since_peak


@njit(fastmath=True)
def _sim_thor_exit_njit(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volume_ratio: np.ndarray,
    quote_volume_slope: np.ndarray,
    taker_slope: np.ndarray,
    weighted_trend: np.ndarray,
    bs_prob: np.ndarray,
    participation_score: np.ndarray,
    flow_exhaustion_score: np.ndarray,
    close_pos: np.ndarray,
    upper_wick_ratio: np.ndarray,
    entry_idx: int,
    entry: float,
    atr_i: float,
    bank_atr: float,
    sl_atr: float,
    bank_fraction: float,
    trail_activate_atr: float,
    runner_trail_atr: float,
    max_bars_pre_bank: int,
    max_bars_post_bank: int,
    penetration_factor: float,
    slip_cap_atr: float,
    material_drawdown_atr: float,
):
    """Returns: (label_code, realized_atr, exit_bar, exit_price, stop_slip_atr, bank_hit, max_drawdown_during_trade_atr)."""
    n = len(closes)
    if atr_i <= 0.0:
        return _LBL_TIMEOUT, 0.0, entry_idx, entry, 0.0, 0, 0.0

    entry_strength = _thor_entry_strength_njit(
        entry_idx,
        participation_score,
        bs_prob,
        weighted_trend,
        quote_volume_slope,
        close_pos,
    )
    bank_price = entry + bank_atr * atr_i
    stop_price = entry - sl_atr * atr_i
    runner_peak_close = entry
    runner_stop = stop_price
    banked = 0
    bank_bar = -1
    bank_gain_atr = 0.0
    peak_close = entry
    peak_high = entry
    peak_bar = entry_idx
    max_dd_during_trade = 0.0  # worst (peak - low)/atr observed
    post_bank_limit = max_bars_post_bank + int(12.0 * entry_strength + 0.5)
    if post_bank_limit < max_bars_post_bank:
        post_bank_limit = max_bars_post_bank
    last_bar = entry_idx + max_bars_pre_bank + post_bank_limit
    if last_bar > n - 1:
        last_bar = n - 1

    j = entry_idx + 1
    while j <= last_bar:
        open_j = opens[j]
        high_j = highs[j]
        low_j = lows[j]
        close_j = closes[j]

        if high_j > peak_high:
            peak_high = high_j
            peak_bar = j
        dd = (peak_high - low_j) / atr_i
        if dd > max_dd_during_trade:
            max_dd_during_trade = dd

        if banked == 0:
            if low_j <= stop_price:
                if open_j <= stop_price:
                    exit_price = open_j
                else:
                    penetration = stop_price - low_j
                    cap = slip_cap_atr * atr_i
                    step = penetration_factor * penetration
                    if step > cap:
                        step = cap
                    exit_price = stop_price - step
                realized = (exit_price - entry) / atr_i
                slip = (stop_price - exit_price) / atr_i
                if slip < 0.0:
                    slip = 0.0
                return _LBL_SL, realized, j, exit_price, slip, 0, max_dd_during_trade
            if high_j >= bank_price:
                banked = 1
                bank_bar = j
                bank_gain_atr = bank_fraction * bank_atr
                if close_j > peak_close:
                    peak_close = close_j
                if peak_close > runner_peak_close:
                    runner_peak_close = peak_close
                if bank_price > runner_peak_close:
                    runner_peak_close = bank_price
                remaining_fraction = 1.0 - bank_fraction
                if remaining_fraction <= 1e-12:
                    remaining_fraction = 1e-12
                profit_cushion_atr = bank_gain_atr / remaining_fraction
                runner_buffer_atr = profit_cushion_atr * (0.18 + 0.32 * entry_strength)
                runner_buffer_cap = 0.75 * runner_trail_atr + 0.25 * bank_atr
                if runner_buffer_atr > runner_buffer_cap:
                    runner_buffer_atr = runner_buffer_cap
                runner_buffer_floor = 0.15
                if runner_buffer_atr < runner_buffer_floor:
                    runner_buffer_atr = runner_buffer_floor
                runner_stop = entry - runner_buffer_atr * atr_i
                if runner_stop < stop_price:
                    runner_stop = stop_price
            elif (j - entry_idx) >= max_bars_pre_bank:
                realized = (close_j - entry) / atr_i
                return _LBL_TIMEOUT, realized, j, close_j, 0.0, 0, max_dd_during_trade
            j += 1
            continue

        if close_j > peak_close:
            peak_close = close_j
        if peak_close > runner_peak_close:
            runner_peak_close = peak_close

        wave_strength, top_risk, bars_since_peak = _thor_wave_top_scores_njit(
            entry,
            atr_i,
            peak_high,
            peak_bar,
            j,
            close_j,
            volume_ratio,
            quote_volume_slope,
            taker_slope,
            bs_prob,
            weighted_trend,
            participation_score,
            flow_exhaustion_score,
            material_drawdown_atr,
        )
        top_tighten = _clip01_njit((top_risk - 55.0) / 35.0)
        wave_decay = _clip01_njit((65.0 - wave_strength) / 35.0)
        flow_tighten = _clip01_njit((flow_exhaustion_score[j] - 55.0) / 35.0)
        weak_close = _clip01_njit((0.60 - close_pos[j]) / 0.60)
        wick_tighten = _clip01_njit((upper_wick_ratio[j] - 0.30) / 0.45)
        stale_tighten = _clip01_njit(float(bars_since_peak) / 8.0)
        tighten = (
            0.32 * top_tighten
            + 0.20 * wave_decay
            + 0.18 * flow_tighten
            + 0.15 * weak_close
            + 0.15 * max(wick_tighten, stale_tighten)
        )
        strong_now = _clip01_njit(
            0.55 * entry_strength
            + 0.25 * _clip01_njit((wave_strength - 65.0) / 35.0)
            + 0.20 * _clip01_njit((participation_score[j] - 55.0) / 45.0)
            - 0.20 * top_tighten
        )
        trail_mult = 1.0 + 0.40 * strong_now - 0.45 * tighten
        if trail_mult < 0.70:
            trail_mult = 0.70
        elif trail_mult > 1.60:
            trail_mult = 1.60
        activate_eff = trail_activate_atr + 0.75 * strong_now - 0.60 * tighten
        if activate_eff < 0.75:
            activate_eff = 0.75
        elif activate_eff > trail_activate_atr + 1.25:
            activate_eff = trail_activate_atr + 1.25

        if (runner_peak_close - entry) / atr_i >= activate_eff:
            cand_stop = runner_peak_close - runner_trail_atr * trail_mult * atr_i
            if cand_stop > runner_stop:
                runner_stop = cand_stop
        if low_j <= runner_stop:
            if open_j <= runner_stop:
                exit_price = open_j
            else:
                penetration = runner_stop - low_j
                cap = slip_cap_atr * atr_i
                step = penetration_factor * penetration
                if step > cap:
                    step = cap
                exit_price = runner_stop - step
            runner_realized = (exit_price - entry) / atr_i
            realized = bank_gain_atr + (1.0 - bank_fraction) * runner_realized
            slip = (runner_stop - exit_price) / atr_i
            if slip < 0.0:
                slip = 0.0
            return _LBL_TP, realized, j, exit_price, slip, 1, max_dd_during_trade
        if bank_bar >= 0 and (j - bank_bar) >= post_bank_limit:
            runner_realized = (close_j - entry) / atr_i
            realized = bank_gain_atr + (1.0 - bank_fraction) * runner_realized
            return _LBL_TP, realized, j, close_j, 0.0, 1, max_dd_during_trade

        j += 1

    close_last = closes[last_bar]
    if banked == 1:
        runner_realized = (close_last - entry) / atr_i
        realized = bank_gain_atr + (1.0 - bank_fraction) * runner_realized
        return _LBL_TP, realized, last_bar, close_last, 0.0, 1, max_dd_during_trade
    realized = (close_last - entry) / atr_i
    return _LBL_TIMEOUT, realized, last_bar, close_last, 0.0, 0, max_dd_during_trade


@njit(fastmath=True)
def _sweep_thor_exits_njit(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volume_ratio: np.ndarray,
    quote_volume_slope: np.ndarray,
    taker_slope: np.ndarray,
    weighted_trend: np.ndarray,
    bs_prob: np.ndarray,
    participation_score: np.ndarray,
    flow_exhaustion_score: np.ndarray,
    close_pos: np.ndarray,
    upper_wick_ratio: np.ndarray,
    entry_idx: int,
    entry: float,
    atr_i: float,
    sl_atr_arr: np.ndarray,      # (K,) candidate stop distances
    bank_atr: float,
    bank_fraction: float,
    trail_activate_atr: float,
    runner_trail_atr: float,
    max_bars_pre_bank: int,
    max_bars_post_bank: int,
    penetration_factor: float,
    slip_cap_atr: float,
    material_drawdown_atr: float,
):
    """
    Walk the bar range ONCE and evaluate K candidate sl_atr stops simultaneously.
    Each stop gets its own state (banked, peak_close, runner_stop, etc).
    Returns (K,)-shaped arrays: label_code, realized_atr, exit_bar, exit_price, stop_slip_atr, bank_hit, max_dd_during_trade.
    """
    K = len(sl_atr_arr)
    label_out = np.zeros(K, dtype=np.int64)
    realized_out = np.zeros(K, dtype=np.float64)
    exit_bar_out = np.zeros(K, dtype=np.int64)
    exit_price_out = np.zeros(K, dtype=np.float64)
    slip_out = np.zeros(K, dtype=np.float64)
    bank_hit_out = np.zeros(K, dtype=np.int64)
    max_dd_out = np.zeros(K, dtype=np.float64)
    done = np.zeros(K, dtype=np.int64)  # 0 = still running, 1 = resolved
    banked = np.zeros(K, dtype=np.int64)
    bank_gain_atr = np.zeros(K, dtype=np.float64)
    peak_close = np.full(K, entry, dtype=np.float64)
    runner_peak_close = np.full(K, entry, dtype=np.float64)
    runner_stop = np.zeros(K, dtype=np.float64)
    stop_price_arr = np.zeros(K, dtype=np.float64)
    for k in range(K):
        stop_price_arr[k] = entry - sl_atr_arr[k] * atr_i
        runner_stop[k] = stop_price_arr[k]

    if atr_i <= 0.0:
        for k in range(K):
            label_out[k] = _LBL_TIMEOUT
            exit_bar_out[k] = entry_idx
            exit_price_out[k] = entry
        return label_out, realized_out, exit_bar_out, exit_price_out, slip_out, bank_hit_out, max_dd_out

    entry_strength = _thor_entry_strength_njit(
        entry_idx,
        participation_score,
        bs_prob,
        weighted_trend,
        quote_volume_slope,
        close_pos,
    )
    n = len(closes)
    bank_price = entry + bank_atr * atr_i
    post_bank_limit = max_bars_post_bank + int(12.0 * entry_strength + 0.5)
    if post_bank_limit < max_bars_post_bank:
        post_bank_limit = max_bars_post_bank
    last_bar = entry_idx + max_bars_pre_bank + post_bank_limit
    if last_bar > n - 1:
        last_bar = n - 1

    peak_high = entry
    peak_bar = entry_idx
    bank_bar = np.full(K, -1, dtype=np.int64)
    j = entry_idx + 1
    while j <= last_bar:
        open_j = opens[j]
        high_j = highs[j]
        low_j = lows[j]
        close_j = closes[j]

        if high_j > peak_high:
            peak_high = high_j
            peak_bar = j
        common_dd = (peak_high - low_j) / atr_i
        wave_strength, top_risk, bars_since_peak = _thor_wave_top_scores_njit(
            entry,
            atr_i,
            peak_high,
            peak_bar,
            j,
            close_j,
            volume_ratio,
            quote_volume_slope,
            taker_slope,
            bs_prob,
            weighted_trend,
            participation_score,
            flow_exhaustion_score,
            material_drawdown_atr,
        )
        top_tighten = _clip01_njit((top_risk - 55.0) / 35.0)
        wave_decay = _clip01_njit((65.0 - wave_strength) / 35.0)
        flow_tighten = _clip01_njit((flow_exhaustion_score[j] - 55.0) / 35.0)
        weak_close = _clip01_njit((0.60 - close_pos[j]) / 0.60)
        wick_tighten = _clip01_njit((upper_wick_ratio[j] - 0.30) / 0.45)
        stale_tighten = _clip01_njit(float(bars_since_peak) / 8.0)
        tighten = (
            0.32 * top_tighten
            + 0.20 * wave_decay
            + 0.18 * flow_tighten
            + 0.15 * weak_close
            + 0.15 * max(wick_tighten, stale_tighten)
        )
        strong_now = _clip01_njit(
            0.55 * entry_strength
            + 0.25 * _clip01_njit((wave_strength - 65.0) / 35.0)
            + 0.20 * _clip01_njit((participation_score[j] - 55.0) / 45.0)
            - 0.20 * top_tighten
        )
        trail_mult = 1.0 + 0.40 * strong_now - 0.45 * tighten
        if trail_mult < 0.70:
            trail_mult = 0.70
        elif trail_mult > 1.60:
            trail_mult = 1.60
        activate_eff = trail_activate_atr + 0.75 * strong_now - 0.60 * tighten
        if activate_eff < 0.75:
            activate_eff = 0.75
        elif activate_eff > trail_activate_atr + 1.25:
            activate_eff = trail_activate_atr + 1.25

        any_active = 0
        for k in range(K):
            if done[k] == 1:
                continue
            any_active = 1
            if common_dd > max_dd_out[k]:
                max_dd_out[k] = common_dd

            if banked[k] == 0:
                sp = stop_price_arr[k]
                if low_j <= sp:
                    if open_j <= sp:
                        exit_price = open_j
                    else:
                        pen = sp - low_j
                        cap = slip_cap_atr * atr_i
                        step = penetration_factor * pen
                        if step > cap:
                            step = cap
                        exit_price = sp - step
                    realized = (exit_price - entry) / atr_i
                    slip = (sp - exit_price) / atr_i
                    if slip < 0.0:
                        slip = 0.0
                    label_out[k] = _LBL_SL
                    realized_out[k] = realized
                    exit_bar_out[k] = j
                    exit_price_out[k] = exit_price
                    slip_out[k] = slip
                    bank_hit_out[k] = 0
                    done[k] = 1
                    continue
                if high_j >= bank_price:
                    banked[k] = 1
                    bank_bar[k] = j
                    bank_gain_atr[k] = bank_fraction * bank_atr
                    if close_j > peak_close[k]:
                        peak_close[k] = close_j
                    if peak_close[k] > runner_peak_close[k]:
                        runner_peak_close[k] = peak_close[k]
                    if bank_price > runner_peak_close[k]:
                        runner_peak_close[k] = bank_price
                    remaining_fraction = 1.0 - bank_fraction
                    if remaining_fraction <= 1e-12:
                        remaining_fraction = 1e-12
                    profit_cushion_atr = bank_gain_atr[k] / remaining_fraction
                    runner_buffer_atr = profit_cushion_atr * (0.18 + 0.32 * entry_strength)
                    runner_buffer_cap = 0.75 * runner_trail_atr + 0.25 * bank_atr
                    if runner_buffer_atr > runner_buffer_cap:
                        runner_buffer_atr = runner_buffer_cap
                    if runner_buffer_atr < 0.15:
                        runner_buffer_atr = 0.15
                    runner_stop[k] = entry - runner_buffer_atr * atr_i
                    if runner_stop[k] < stop_price_arr[k]:
                        runner_stop[k] = stop_price_arr[k]
                elif (j - entry_idx) >= max_bars_pre_bank:
                    realized = (close_j - entry) / atr_i
                    label_out[k] = _LBL_TIMEOUT
                    realized_out[k] = realized
                    exit_bar_out[k] = j
                    exit_price_out[k] = close_j
                    bank_hit_out[k] = 0
                    done[k] = 1
                continue

            # banked branch
            if close_j > peak_close[k]:
                peak_close[k] = close_j
            if peak_close[k] > runner_peak_close[k]:
                runner_peak_close[k] = peak_close[k]
            if (runner_peak_close[k] - entry) / atr_i >= activate_eff:
                cand_stop = runner_peak_close[k] - runner_trail_atr * trail_mult * atr_i
                if cand_stop > runner_stop[k]:
                    runner_stop[k] = cand_stop
            rs = runner_stop[k]
            if low_j <= rs:
                if open_j <= rs:
                    exit_price = open_j
                else:
                    pen = rs - low_j
                    cap = slip_cap_atr * atr_i
                    step = penetration_factor * pen
                    if step > cap:
                        step = cap
                    exit_price = rs - step
                runner_realized = (exit_price - entry) / atr_i
                realized = bank_gain_atr[k] + (1.0 - bank_fraction) * runner_realized
                slip = (rs - exit_price) / atr_i
                if slip < 0.0:
                    slip = 0.0
                label_out[k] = _LBL_TP
                realized_out[k] = realized
                exit_bar_out[k] = j
                exit_price_out[k] = exit_price
                slip_out[k] = slip
                bank_hit_out[k] = 1
                done[k] = 1
                continue
            if bank_bar[k] >= 0 and (j - bank_bar[k]) >= post_bank_limit:
                runner_realized = (close_j - entry) / atr_i
                realized = bank_gain_atr[k] + (1.0 - bank_fraction) * runner_realized
                label_out[k] = _LBL_TP
                realized_out[k] = realized
                exit_bar_out[k] = j
                exit_price_out[k] = close_j
                bank_hit_out[k] = 1
                done[k] = 1

        if any_active == 0:
            break
        j += 1

    # Unresolved tails → timeout on last bar
    close_last = closes[last_bar]
    for k in range(K):
        if done[k] == 1:
            continue
        if banked[k] == 1:
            runner_realized = (close_last - entry) / atr_i
            realized = bank_gain_atr[k] + (1.0 - bank_fraction) * runner_realized
            label_out[k] = _LBL_TP
            realized_out[k] = realized
            exit_bar_out[k] = last_bar
            exit_price_out[k] = close_last
            bank_hit_out[k] = 1
        else:
            realized = (close_last - entry) / atr_i
            label_out[k] = _LBL_TIMEOUT
            realized_out[k] = realized
            exit_bar_out[k] = last_bar
            exit_price_out[k] = close_last
            bank_hit_out[k] = 0

    return label_out, realized_out, exit_bar_out, exit_price_out, slip_out, bank_hit_out, max_dd_out


def simulate_thor_exit_stop_market(
    sig: dict,
    ctx: SparseFeatureContext,
    bank_atr: float,
    sl_atr: float,
    bank_fraction: float,
    trail_activate_atr: float,
    runner_trail_atr: float,
    max_bars_pre_bank: int,
    max_bars_post_bank: int,
    penetration_factor: float = 0.35,
    slip_cap_atr: float = 0.25,
    material_drawdown_atr: float = 1.0,
) -> dict:
    i = int(sig["bar_idx"])
    entry = float(sig["close"])
    atr_i = float(sig["atr"])
    if atr_i <= 0:
        return {"label": "TIMEOUT", "realized_atr": 0.0, "exit_bar": i, "exit_price": entry, "stop_slip_atr": 0.0}

    label_code, realized, exit_bar, exit_price, slip, bank_hit, _max_dd = _sim_thor_exit_njit(
        ctx.opens, ctx.highs, ctx.lows, ctx.closes,
        ctx.volume_ratio, ctx.quote_volume_slope, ctx.taker_slope, ctx.weighted_trend,
        ctx.bs_prob, ctx.participation_score, ctx.flow_exhaustion_score, ctx.close_pos,
        ctx.upper_wick_ratio,
        i, entry, atr_i,
        float(bank_atr), float(sl_atr), float(bank_fraction),
        float(trail_activate_atr), float(runner_trail_atr),
        int(max_bars_pre_bank), int(max_bars_post_bank),
        float(penetration_factor), float(slip_cap_atr),
        float(material_drawdown_atr),
    )
    result = {
        "label": _LBL_CODE_TO_STR[int(label_code)],
        "realized_atr": float(realized),
        "exit_bar": int(exit_bar),
        "exit_price": float(exit_price),
        "stop_slip_atr": float(slip),
    }
    if int(label_code) == _LBL_TP or int(label_code) == _LBL_TIMEOUT:
        result["bank_hit"] = bool(bank_hit)
    return result


def build_pump_ledger(ctx: SparseFeatureContext, pump_id: str, start_bar: int, end_bar: int, material_drawdown_atr: float = 1.0) -> tuple[List[dict], dict]:
    rows: List[dict] = []
    start = max(0, int(start_bar))
    end = min(int(end_bar), len(ctx.closes) - 1)
    if end < start:
        return rows, {}

    analytics = _pump_path_analytics(ctx, start, end, material_drawdown_atr)
    entry_price = float(analytics["entry_price"])
    offsets = analytics["offsets"]
    times = analytics["times"]
    runup_series = analytics["runup_series"]
    close_drawdown_series = analytics["close_drawdown_series"]
    bars_since_peak_series = analytics["bars_since_peak_series"]
    wave_strength_series = analytics["wave_strength_series"]
    top_risk_series = analytics["top_risk_series"]

    for offset in offsets:
        offset_i = int(offset)
        bar = start + offset_i
        rows.append(
            {
                "pump_id": pump_id,
                "symbol": ctx.symbol,
                "bar_idx": bar,
                "ts": int(times[offset_i]),
                "bars_since_entry": offset_i,
                "bars_since_peak": int(bars_since_peak_series[offset_i]),
                "cumulative_return_pct": ((ctx.closes[bar] / max(entry_price, 1e-12)) - 1.0) * 100.0,
                "runup_atr": float(runup_series[offset_i]),
                "drawdown_atr": float(close_drawdown_series[offset_i]),
                "volume_ratio": float(ctx.volume_ratio[bar]),
                "quote_volume_slope": float(ctx.quote_volume_slope[bar]),
                "taker_slope": float(ctx.taker_slope[bar]),
                "vpin_slope": float(ctx.vpin_slope[bar]),
                "wave_strength_score": float(wave_strength_series[offset_i]),
                "top_risk_score": float(top_risk_series[offset_i]),
            }
        )

    summary = {
        "pump_id": pump_id,
        "symbol": ctx.symbol,
        "start_bar": start,
        "end_bar": end,
        "start_ts": int(ctx.times[start]),
        "end_ts": int(ctx.times[end]),
        "max_runup_atr": float(analytics["max_runup_atr"]),
        "max_drawdown_atr": float(analytics["summary_max_drawdown_atr"]),
        "time_to_peak_bars": int(analytics["time_to_peak_bars"]),
        "time_to_material_drawdown_bars": analytics["time_to_material_drawdown_bars"],
        "volume_decay_after_peak": float(analytics["volume_decay_after_peak"]),
        "final_wave_strength_score": float(analytics["final_wave_strength_score"]),
        "final_top_risk_score": float(analytics["final_top_risk_score"]),
    }
    return rows, summary


def build_thor_contexts(
    thor_signals: Iterable[dict],
    context_bars: int,
    min_score: float,
) -> List[ThorContext]:
    contexts: List[ThorContext] = []
    for sig in thor_signals:
        score = float(sig.get("score", 0.0))
        if score < min_score:
            continue
        atr = float(sig.get("atr", 0.0))
        if atr <= 0:
            continue
        bar_idx = int(sig["bar_idx"])
        contexts.append(
            ThorContext(
                symbol=str(sig.get("symbol", "")),
                entry_price=float(sig["close"]),
                atr=atr,
                start_bar=bar_idx,
                expiry_bar=bar_idx + int(context_bars),
                score=score,
                tier=str(sig.get("tier", "")),
            )
        )
    return contexts


def find_active_thor_context(
    contexts: List[ThorContext],
    bar_idx: int,
) -> Optional[ThorContext]:
    active: Optional[ThorContext] = None
    for ctx in contexts:
        if ctx.start_bar < bar_idx <= ctx.expiry_bar:
            if active is None or ctx.start_bar > active.start_bar:
                active = ctx
    return active


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ts_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def extract_freya_signals(
    df,
    thor_signals: List[dict],
    cfg_events,
) -> List[dict]:
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    times = df["open_time"].values
    atrs = calc_atr(highs, lows, closes)
    vol_avg = calc_vol_avg20(volumes)

    contexts = build_thor_contexts(
        thor_signals,
        int(getattr(cfg_events, "thor_context_bars", 24)),
        float(getattr(cfg_events, "thor_context_min_score", 72.0)),
    )
    cooldown = int(getattr(cfg_events, "freya_cooldown_bars", 4))
    min_move_pct = float(getattr(cfg_events, "freya_momentum_pct", 0.25))
    min_body_eff = float(getattr(cfg_events, "freya_body_eff_min", 0.50))
    min_vol_mult = float(getattr(cfg_events, "freya_vol_mult", 1.20))
    thor_score_min = float(getattr(cfg_events, "freya_thor_score_min", 60.0))
    allow_tier_c = bool(getattr(cfg_events, "freya_allow_tier_c", False))
    min_wait_bars = int(getattr(cfg_events, "freya_min_wait_bars", 2))
    min_runup_atr = float(getattr(cfg_events, "freya_min_runup_atr", 0.8))
    pullback_min_atr = float(getattr(cfg_events, "freya_pullback_min_atr", 0.20))
    pullback_max_atr = float(getattr(cfg_events, "freya_pullback_max_atr", 1.10))
    pullback_floor_atr = float(getattr(cfg_events, "freya_pullback_floor_atr", 0.20))
    min_reclaim_atr = float(getattr(cfg_events, "freya_min_reclaim_atr", 0.05))
    max_extension_atr = float(getattr(cfg_events, "freya_max_extension_atr", 0.30))
    max_peak_gap_atr = float(getattr(cfg_events, "freya_max_peak_gap_atr", 0.75))
    peak_gap_chase_floor_atr = float(getattr(cfg_events, "freya_peak_gap_chase_floor_atr", 0.02))
    chase_vol_max = float(getattr(cfg_events, "freya_chase_vol_max", 1.50))
    chase_move_max = float(getattr(cfg_events, "freya_chase_move_max", 0.40))
    upper_wick_max = float(getattr(cfg_events, "freya_upper_wick_max", 0.25))
    close_pos_min = float(getattr(cfg_events, "freya_close_pos_min", 0.65))
    min_quote_vol20 = float(getattr(cfg_events, "freya_min_quote_vol20", 20000.0))

    signals: List[dict] = []
    last_sig = -cooldown
    used_contexts: set[tuple[int, str]] = set()
    for i in range(21, len(closes)):
        if i - last_sig < cooldown:
            continue
        ctx = find_active_thor_context(contexts, i)
        if ctx is None or atrs[i] <= 0 or vol_avg[i] <= 0:
            continue
        if i <= ctx.start_bar + min_wait_bars:
            continue
        context_key = (ctx.start_bar, str(df.attrs.get("symbol", "")))
        if context_key in used_contexts:
            continue
        if ctx.score < thor_score_min:
            continue
        if not allow_tier_c and str(ctx.tier).upper() == "C":
            continue

        recent_peak = float(np.max(highs[ctx.start_bar:i]))
        runup_atr = (recent_peak - ctx.entry_price) / max(ctx.atr, 1e-12)
        if runup_atr < min_runup_atr:
            continue

        pullback_start = max(ctx.start_bar + 1, i - 2)
        recent_low = float(np.min(lows[pullback_start:i]))
        pullback_depth_atr = (recent_peak - recent_low) / max(ctx.atr, 1e-12)
        if pullback_depth_atr < pullback_min_atr or pullback_depth_atr > pullback_max_atr:
            continue
        if recent_low < ctx.entry_price + pullback_floor_atr * ctx.atr:
            continue
        if not (closes[i - 1] < highs[i - 1] and closes[i - 1] <= closes[i - 2]):
            continue

        body = closes[i] - opens[i]
        candle_range = highs[i] - lows[i]
        if body <= 0 or candle_range <= 0:
            continue
        body_eff = body / candle_range
        upper_wick_ratio = (highs[i] - closes[i]) / candle_range
        close_pos = (closes[i] - lows[i]) / candle_range
        move_pct = ((closes[i] / max(closes[i - 1], 1e-12)) - 1.0) * 100.0
        vol_ratio = volumes[i] / max(vol_avg[i], 1e-12)
        quote_vol20 = closes[i] * vol_avg[i]
        reclaim_level = max(closes[i - 1], opens[i - 1])
        reclaim_atr = (closes[i] - reclaim_level) / max(ctx.atr, 1e-12)
        extension_atr = max(0.0, (closes[i] - recent_peak) / max(ctx.atr, 1e-12))
        peak_gap_atr = max(0.0, (recent_peak - closes[i]) / max(ctx.atr, 1e-12))
        exact_peak_chase = (
            peak_gap_atr <= peak_gap_chase_floor_atr
            and (vol_ratio > chase_vol_max or move_pct > chase_move_max)
        )
        if (
            move_pct < min_move_pct
            or body_eff < min_body_eff
            or vol_ratio < min_vol_mult
            or quote_vol20 < min_quote_vol20
            or reclaim_atr < min_reclaim_atr
            or extension_atr > max_extension_atr
            or peak_gap_atr > max_peak_gap_atr
            or upper_wick_ratio > upper_wick_max
            or close_pos < close_pos_min
            or closes[i] < ctx.entry_price
            or highs[i] < highs[i - 1]
            or exact_peak_chase
        ):
            continue

        score = 100.0 * (
            0.25 * _clip01((ctx.score - thor_score_min) / 20.0)
            + 0.25 * _clip01((reclaim_atr - min_reclaim_atr) / max(0.25, max_peak_gap_atr - min_reclaim_atr))
            + 0.20 * _clip01((pullback_depth_atr - pullback_min_atr) / max(pullback_max_atr - pullback_min_atr, 1e-6))
            + 0.15 * _clip01((vol_ratio - min_vol_mult) / 1.5)
            + 0.15 * (1.0 - _clip01(peak_gap_atr / max(max_peak_gap_atr, 1e-6)))
        )
        signals.append(
            {
                "agent_key": "freya",
                "display_agent": "Freya",
                "source_regime_agent": "Thor",
                "thor_context_active": True,
                "freya_context_valid": True,
                "baldur_top_warning": False,
                "symbol": str(df.attrs.get("symbol", "")),
                "bar_idx": i,
                "date": _ts_str(times[i]),
                "close": float(closes[i]),
                "atr": float(atrs[i]),
                "direction": "BULLISH",
                "score": round(score, 2),
                "body_eff": round(float(body_eff), 3),
                "vol_ratio": round(float(vol_ratio), 2),
                "move_pct": round(float(move_pct), 3),
                "runup_atr": round(float(runup_atr), 3),
                "pullback_depth_atr": round(float(pullback_depth_atr), 3),
                "reclaim_atr": round(float(reclaim_atr), 3),
                "extension_atr": round(float(extension_atr), 3),
                "peak_gap_atr": round(float(peak_gap_atr), 3),
                "thor_score": round(float(ctx.score), 2),
                "thor_tier": ctx.tier,
            }
        )
        last_sig = i
        used_contexts.add(context_key)
    return signals


def extract_baldur_signals(
    df,
    thor_signals: List[dict],
    cfg_events,
) -> tuple[List[dict], List[dict]]:
    closes = df["close"].values.astype(np.float64)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    times = df["open_time"].values
    atrs = calc_atr(highs, lows, closes)
    vol_avg = calc_vol_avg20(volumes)

    contexts = build_thor_contexts(
        thor_signals,
        int(getattr(cfg_events, "thor_context_bars", 24)),
        float(getattr(cfg_events, "thor_context_min_score", 60.0)),
    )
    min_runup_atr = float(getattr(cfg_events, "baldur_min_runup_atr", 1.0))
    min_upper_wick = float(getattr(cfg_events, "baldur_upper_wick_min", 0.25))
    max_close_pos = float(getattr(cfg_events, "baldur_close_pos_max", 0.40))
    min_drop_pct = float(getattr(cfg_events, "baldur_confirm_drop_pct", 0.25))

    warnings: List[dict] = []
    signals: List[dict] = []
    for i in range(22, len(closes) - 1):
        ctx = find_active_thor_context(contexts, i)
        if ctx is None or atrs[i] <= 0:
            continue
        peak_high = float(np.max(highs[ctx.start_bar:i + 1]))
        runup_atr = (peak_high - ctx.entry_price) / max(ctx.atr, 1e-12)
        if runup_atr < min_runup_atr:
            continue

        candle_range = highs[i] - lows[i]
        if candle_range <= 0:
            continue
        upper_wick = highs[i] - max(opens[i], closes[i])
        close_pos = (closes[i] - lows[i]) / candle_range
        vol_ratio = volumes[i] / max(vol_avg[i], 1e-12) if vol_avg[i] > 0 else 0.0

        top_warning = (
            highs[i] >= peak_high * 0.95
            and (closes[i] < opens[i] or closes[i] < closes[i - 1])
            and upper_wick / candle_range >= min_upper_wick * 0.5
            and close_pos <= max_close_pos * 1.5
        )

        if not top_warning:
            continue

        warning = {
            "agent_key": "baldur",
            "display_agent": "Baldur",
            "source_regime_agent": "Thor",
            "thor_context_active": True,
            "freya_context_valid": False,
            "baldur_top_warning": True,
            "symbol": str(df.attrs.get("symbol", "")),
            "bar_idx": i,
            "date": _ts_str(times[i]),
            "close": float(closes[i]),
            "atr": float(atrs[i]),
            "runup_atr": round(float(runup_atr), 3),
            "upper_wick_ratio": round(float(upper_wick / candle_range), 3),
            "close_pos": round(float(close_pos), 3),
            "vol_ratio": round(float(vol_ratio), 2),
            "thor_score": round(float(ctx.score), 2),
            "thor_tier": ctx.tier,
        }
        warnings.append(warning)

        j = i + 1
        confirm_drop = ((closes[i] / max(closes[j], 1e-12)) - 1.0) * 100.0
        if (
            closes[j] < closes[i]
            and lows[j] < lows[i]
            and confirm_drop >= min_drop_pct
        ):
            signal = dict(warning)
            signal["bar_idx"] = j
            signal["date"] = _ts_str(times[j])
            signal["close"] = float(closes[j])
            signal["atr"] = float(atrs[j])
            signal["direction"] = "BEARISH"
            signal["warning_bar_idx"] = i
            signal["confirm_drop_pct"] = round(float(confirm_drop), 3)
            signal["score"] = round(
                100.0 * (
                    0.45 * _clip01(runup_atr / max(min_runup_atr * 1.5, 1e-6))
                    + 0.30 * _clip01(upper_wick / candle_range)
                    + 0.25 * _clip01(confirm_drop / max(min_drop_pct * 3.0, 1e-6))
                ),
                2,
            )
            signals.append(signal)
    return warnings, signals


def simulate_directional_exit(
    sig: dict,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_atr: float,
    sl_atr: float,
    max_bars: int,
) -> dict:
    i = int(sig["bar_idx"])
    entry = float(sig["close"])
    atr_i = float(sig["atr"])
    direction = str(sig.get("direction", "BULLISH")).upper()
    if atr_i <= 0:
        return {"label": "TIMEOUT", "realized_atr": 0.0, "exit_bar": i}

    if direction == "BEARISH":
        tp_price = entry - tp_atr * atr_i
        sl_price = entry + sl_atr * atr_i
    else:
        tp_price = entry + tp_atr * atr_i
        sl_price = entry - sl_atr * atr_i

    last_bar = min(i + int(max_bars), len(closes) - 1)
    for j in range(i + 1, last_bar + 1):
        if direction == "BEARISH":
            if highs[j] >= sl_price:
                return {"label": "SL", "realized_atr": -sl_atr, "exit_bar": j}
            if lows[j] <= tp_price:
                return {"label": "TP", "realized_atr": tp_atr, "exit_bar": j}
        else:
            if lows[j] <= sl_price:
                return {"label": "SL", "realized_atr": -sl_atr, "exit_bar": j}
            if highs[j] >= tp_price:
                return {"label": "TP", "realized_atr": tp_atr, "exit_bar": j}

    realized = (
        (entry - closes[last_bar]) / atr_i
        if direction == "BEARISH"
        else (closes[last_bar] - entry) / atr_i
    )
    return {"label": "TIMEOUT", "realized_atr": float(realized), "exit_bar": last_bar}


def measure_baldur_topstart(
    sig: dict,
    closes: np.ndarray,
    lows: np.ndarray,
    target_atr: float = 1.0,
    max_bars: int = 12,
) -> dict:
    i = int(sig["bar_idx"])
    entry = float(sig["close"])
    atr_i = float(sig["atr"])
    if atr_i <= 0:
        return {"success": False, "delay_bars": None}
    target_price = entry - target_atr * atr_i
    last_bar = min(i + int(max_bars), len(closes) - 1)
    for j in range(i + 1, last_bar + 1):
        if lows[j] <= target_price:
            return {"success": True, "delay_bars": j - i}
    return {"success": False, "delay_bars": None}
