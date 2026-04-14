"""
QUANTA Nike Live Validator
===========================
1. Fetches top-N gainers over the last 30 days from Binance Futures (REST)
2. Downloads their 5-minute OHLCV for the last 30 days
3. Runs the Nike extractor with configurable params
4. Labels every signal with triple-barrier (TP / SL / timeout)
5. Grid-searches the best Nike parameter set
6. Reports per-coin and aggregate accuracy

Usage:
    python quanta_nike_live_validator.py               # default: top 40 gainers
    python quanta_nike_live_validator.py --top 60      # top 60 gainers
    python quanta_nike_live_validator.py --no-grid     # skip param search
    python quanta_nike_live_validator.py --proxy       # route through QUANTA proxy

WebSocket live mode (streams candles after backtest):
    python quanta_nike_live_validator.py --live
"""

import sys, os, time, argparse, json, asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BASE_URL   = "https://fapi.binance.com"
INTERVAL   = "5m"
DAYS_BACK  = 30
BARS_TOTAL = DAYS_BACK * 24 * 12          # 8,640 bars per coin (30 days × 288 bars/day)
MAX_PER_REQ= 1500                          # Binance klines limit per request
LOOKBACK   = 20                            # for ATR and avg-body calc
ATR_PERIOD = 14
SEP  = "─" * 100
DSEP = "═" * 100

# Default Nike params (loaded from config)
try:
    import sys as _sys
    _sys.stdout = open(os.devnull, 'w')    # suppress emoji print from config
    from quanta_config import Config as _cfg
    _sys.stdout = sys.__stdout__
    _DEF = dict(
        body_ratio_mult = _cfg.events.nike_body_ratio_mult,
        quiet_body_pct  = _cfg.events.nike_quiet_body_pct,
        vol_mult        = _cfg.events.nike_vol_mult,
        body_min        = _cfg.events.nike_body_min,
        immediate_body_ratio_mult = _cfg.events.nike_immediate_body_ratio_mult,
        immediate_body_min        = _cfg.events.nike_immediate_body_min,
        immediate_vol_mult        = _cfg.events.nike_immediate_vol_mult,
        continuation_vol_mult     = _cfg.events.nike_continuation_vol_mult,
        score_body_ratio_weight   = _cfg.events.nike_score_body_ratio_weight,
        score_vol_ratio_weight    = _cfg.events.nike_score_vol_ratio_weight,
        score_body_eff_weight     = _cfg.events.nike_score_body_eff_weight,
        score_quiet_weight        = _cfg.events.nike_score_quiet_weight,
        score_confirm_weight      = _cfg.events.nike_score_confirm_weight,
        tier_a_confidence         = _cfg.events.nike_tier_a_confidence,
        tier_b_confidence         = _cfg.events.nike_tier_b_confidence,
        tier_c_confidence         = _cfg.events.nike_tier_c_confidence,
        tier_a_size_mult          = _cfg.events.nike_tier_a_size_mult,
        tier_b_size_mult          = _cfg.events.nike_tier_b_size_mult,
        tier_c_size_mult          = _cfg.events.nike_tier_c_size_mult,
        tier_b_bs_floor           = _cfg.events.nike_tier_b_bs_floor,
        tier_c_bs_floor           = _cfg.events.nike_tier_c_bs_floor,
        tp_atr          = _cfg.events.nike_tp_atr,
        sl_atr          = _cfg.events.nike_sl_atr,
        max_bars        = _cfg.events.nike_max_bars,
        bank_atr        = _cfg.events.nike_bank_atr,
        bank_fraction   = _cfg.events.nike_bank_fraction,
        runner_trail_atr= _cfg.events.nike_runner_trail_atr,
        max_bars_pre_bank = _cfg.events.nike_max_bars_pre_bank,
        max_bars_post_bank= _cfg.events.nike_max_bars_post_bank,
    )
except Exception:
    _sys.stdout = sys.__stdout__
    _DEF = dict(body_ratio_mult=5.0, quiet_body_pct=0.5, vol_mult=1.5,
                body_min=0.4, immediate_body_ratio_mult=8.0,
                immediate_body_min=0.55, immediate_vol_mult=2.0,
                continuation_vol_mult=1.0,
                score_body_ratio_weight=0.30, score_vol_ratio_weight=0.25,
                score_body_eff_weight=0.20, score_quiet_weight=0.10, score_confirm_weight=0.15,
                tier_a_confidence=84.0, tier_b_confidence=78.0, tier_c_confidence=72.0,
                tier_a_size_mult=1.15, tier_b_size_mult=1.0, tier_c_size_mult=0.75,
                tier_b_bs_floor=0.30, tier_c_bs_floor=0.35,
                tp_atr=2.0, sl_atr=0.8, max_bars=24,
                bank_atr=2.0, bank_fraction=0.5, runner_trail_atr=1.5,
                max_bars_pre_bank=24, max_bars_post_bank=36)

# Grid for parameter search
GRID = {
    "body_ratio_mult": [4.0, 5.0, 6.0, 7.0, 8.0],
    "quiet_body_pct":  [0.3, 0.5, 0.7, 1.0],
    "vol_mult":        [1.2, 1.5, 1.8, 2.0],
}

# ─── BINANCE REST ─────────────────────────────────────────────────────────────

def _session(proxy_port: int | None = None) -> requests.Session:
    s = requests.Session()
    if proxy_port:
        p = f"http://127.0.0.1:{proxy_port}"
        s.proxies = {"http": p, "https": p}
    s.headers.update({"User-Agent": "QUANTA/11.7"})
    return s


def get_futures_symbols(sess: requests.Session) -> list[str]:
    """Return all active USDT-margined perpetual symbols."""
    r = sess.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=15)
    r.raise_for_status()
    return [
        s["symbol"]
        for s in r.json()["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    ]


def get_30d_change(sess: requests.Session, symbol: str) -> tuple[str, float]:
    """
    Compute 30-day % price change.
    Fetches the close price 30 days ago (1 daily bar) vs current mark price.
    """
    try:
        ts_30d = int((time.time() - DAYS_BACK * 86400) * 1000)
        r = sess.get(
            f"{BASE_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1d", "startTime": ts_30d, "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return symbol, 0.0
        open_30d = float(data[0][1])
        # current price
        r2 = sess.get(f"{BASE_URL}/fapi/v1/ticker/price",
                      params={"symbol": symbol}, timeout=5)
        r2.raise_for_status()
        cur = float(r2.json()["price"])
        pct = (cur - open_30d) / open_30d * 100 if open_30d > 0 else 0.0
        return symbol, pct
    except Exception:
        return symbol, 0.0


def fetch_klines_5m(sess: requests.Session, symbol: str, bars: int = BARS_TOTAL) -> pd.DataFrame | None:
    """
    Download `bars` × 5m candles for `symbol`.
    Returns DataFrame with columns: open_time, open, high, low, close, volume
    """
    end_ts   = int(time.time() * 1000)
    start_ts = end_ts - bars * 5 * 60 * 1000
    all_rows = []
    cur_start = start_ts

    while cur_start < end_ts:
        for attempt in range(5):
            try:
                r = sess.get(
                    f"{BASE_URL}/fapi/v1/klines",
                    params={
                        "symbol":    symbol,
                        "interval":  INTERVAL,
                        "startTime": cur_start,
                        "limit":     MAX_PER_REQ,
                    },
                    timeout=20,
                )
                if r.status_code == 429:
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                batch = r.json()
                break
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    time.sleep(2 ** attempt + 1)
                    continue
                print(f"  [WARN] {symbol} HTTP error: {e}")
                return pd.DataFrame(all_rows) if all_rows else None
            except Exception as e:
                print(f"  [WARN] {symbol} fetch error: {e}")
                return pd.DataFrame(all_rows) if all_rows else None
        else:
            break   # all retries failed

        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts >= end_ts or len(batch) < MAX_PER_REQ:
            break
        cur_start = last_ts + 5 * 60 * 1000
        time.sleep(0.12)   # 120ms between requests per symbol (~8 req/s)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","trades","tbv","tqv","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"])
    df = df.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    atr = np.zeros(n)
    atr[period] = tr[1:period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_vol_avg20(volumes: np.ndarray) -> np.ndarray:
    n = len(volumes)
    va = np.zeros(n)
    for i in range(20, n):
        va[i] = volumes[i - 20:i].mean()
    return va


# ─── NIKE EXTRACTOR (pure python, param-configurable for grid search) ─────────

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _confirm_strength(entry_close: float, entry_high: float, entry_low: float, setup: dict) -> float:
    hold_span = max(setup["close"] - setup["mid"], 1e-12)
    break_span = max(setup["high"] - setup["mid"], 1e-12)
    hold = _clip01((entry_low - setup["mid"]) / hold_span)
    close_push = _clip01((entry_close - setup["close"]) / hold_span)
    breakout = _clip01((entry_high - setup["high"]) / break_span)
    return (hold + close_push + breakout) / 3.0


def _nike_score(meta: dict, confirm_strength: float) -> float:
    body_ratio_norm = _clip01((meta["body_ratio"] - _DEF["body_ratio_mult"]) /
                              max(4.0, _DEF["immediate_body_ratio_mult"] - _DEF["body_ratio_mult"]))
    vol_ratio_norm = _clip01((meta["vol_ratio"] - _DEF["vol_mult"]) /
                             max(2.0, _DEF["immediate_vol_mult"] - _DEF["vol_mult"] + 1.5))
    body_eff_norm = _clip01((meta["body_eff"] - _DEF["body_min"]) /
                            max(0.20, _DEF["immediate_body_min"] - _DEF["body_min"] + 0.05))
    quiet_norm = _clip01((_DEF["quiet_body_pct"] - meta["avg_prior_%"]) /
                         max(_DEF["quiet_body_pct"], 1e-12))

    score = 100.0 * (
        _DEF["score_body_ratio_weight"] * body_ratio_norm +
        _DEF["score_vol_ratio_weight"] * vol_ratio_norm +
        _DEF["score_body_eff_weight"] * body_eff_norm +
        _DEF["score_quiet_weight"] * quiet_norm +
        _DEF["score_confirm_weight"] * _clip01(confirm_strength)
    )
    return round(score, 2)


def _tier_payload(tier: str, entry_mode: str, meta: dict, confirm_strength: float) -> dict:
    if tier == "A":
        confidence = _DEF["tier_a_confidence"]
        size_mult = _DEF["tier_a_size_mult"]
        bs_floor = 0.0
    elif tier == "B":
        confidence = _DEF["tier_b_confidence"]
        size_mult = _DEF["tier_b_size_mult"]
        bs_floor = _DEF["tier_b_bs_floor"]
    else:
        confidence = _DEF["tier_c_confidence"]
        size_mult = _DEF["tier_c_size_mult"]
        bs_floor = _DEF["tier_c_bs_floor"]

    out = dict(meta)
    out["tier"] = tier
    out["entry_mode"] = entry_mode
    out["score"] = _nike_score(meta, confirm_strength)
    out["confidence"] = confidence
    out["size_mult"] = size_mult
    out["bs_floor"] = bs_floor
    return out

def extract_nike_signals(
    df: pd.DataFrame,
    body_ratio_mult: float = _DEF["body_ratio_mult"],
    quiet_body_pct:  float = _DEF["quiet_body_pct"],
    vol_mult:        float = _DEF["vol_mult"],
    body_min:        float = _DEF["body_min"],
) -> list[dict]:
    """
    Pure-Python Nike trigger (mirrors fast_extract_nike logic but param-flexible).
    Returns list of signal dicts.
    """
    closes  = df["close"].values.astype(np.float64)
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    times   = df["open_time"].values

    n        = len(closes)
    lookback = LOOKBACK
    vol_avg  = calc_vol_avg20(volumes)
    signals  = []
    last_sig = -lookback

    min_gap = max(lookback + 1, 6)   # minimum bars between signals (cooldown)

    def setup_metrics(idx: int):
        if idx < lookback:
            return None

        body_i = closes[idx] - opens[idx]
        if body_i <= 0.0 or body_i / max(opens[idx], 1e-12) < 0.01:
            return None

        rng = highs[idx] - lows[idx]
        if rng <= 0.0:
            return None
        body_eff = body_i / rng
        if body_eff < body_min:
            return None

        prior_bodies = np.abs(closes[idx - lookback:idx] - opens[idx - lookback:idx])
        avg_body = prior_bodies.mean()
        if avg_body <= 0.0:
            return None

        body_ratio = body_i / avg_body
        if body_ratio < body_ratio_mult:
            return None

        avg_body_pct = (avg_body / max(opens[idx], 1e-12)) * 100.0
        if avg_body_pct >= quiet_body_pct:
            return None

        if vol_avg[idx] <= 0.0:
            return None
        vol_ratio = volumes[idx] / vol_avg[idx]
        if vol_ratio < vol_mult:
            return None

        return {
            "bar_idx": idx,
            "close": closes[idx],
            "high": highs[idx],
            "mid": 0.5 * (opens[idx] + closes[idx]),
            "body_pct_%": round(body_i / max(opens[idx], 1e-12) * 100, 3),
            "body_ratio": float(body_ratio),
            "avg_prior_%": round(avg_body_pct, 3),
            "vol_ratio": float(vol_ratio),
            "body_eff": float(body_eff),
            "immediate_ok": (
                body_eff >= _DEF["immediate_body_min"] and
                body_ratio >= _DEF["immediate_body_ratio_mult"] and
                vol_ratio >= _DEF["immediate_vol_mult"]
            ),
        }

    for i in range(lookback + 1, n):
        if i - last_sig < min_gap:
            continue

        current_setup = setup_metrics(i)
        if current_setup is not None and current_setup["immediate_ok"]:
            ts = datetime.fromtimestamp(int(times[i]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            sig = _tier_payload("A", "immediate", {
                "bar_idx":       i,
                "date":          ts,
                "close":         round(closes[i], 8),
                "body_pct_%":    current_setup["body_pct_%"],
                "body_ratio":    round(current_setup["body_ratio"], 1),
                "avg_prior_%":   current_setup["avg_prior_%"],
                "vol_ratio":     round(current_setup["vol_ratio"], 1),
                "body_eff":      round(current_setup["body_eff"], 3),
                "mid":           current_setup["mid"],
                "high":          current_setup["high"],
            }, 1.0)
            signals.append(sig)
            last_sig = i
            continue

        prev_setup = setup_metrics(i - 1)
        if prev_setup is not None:
            confirm_ok = (
                lows[i] >= prev_setup["mid"] and
                closes[i] >= prev_setup["close"] and
                highs[i] >= prev_setup["high"]
            )
            if confirm_ok:
                ts = datetime.fromtimestamp(int(times[i]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                sig = _tier_payload("B", "confirm", {
                    "bar_idx":       i,
                    "date":          ts,
                    "close":         round(closes[i], 8),
                    "body_pct_%":    prev_setup["body_pct_%"],
                    "body_ratio":    round(prev_setup["body_ratio"], 1),
                    "avg_prior_%":   prev_setup["avg_prior_%"],
                    "vol_ratio":     round(prev_setup["vol_ratio"], 1),
                    "body_eff":      round(prev_setup["body_eff"], 3),
                    "mid":           prev_setup["mid"],
                    "high":          prev_setup["high"],
                }, _confirm_strength(closes[i], highs[i], lows[i], prev_setup))
                signals.append(sig)
                last_sig = i
                continue

        prev2_setup = setup_metrics(i - 2)
        if prev2_setup is None:
            continue

        continuation_ok = (
            lows[i - 1] >= prev2_setup["mid"] and
            closes[i] >= prev2_setup["close"] and
            highs[i] >= prev2_setup["high"]
        )
        if not continuation_ok:
            continue

        entry_vol_ratio = volumes[i] / vol_avg[i] if vol_avg[i] > 0.0 else 0.0
        if entry_vol_ratio < _DEF["continuation_vol_mult"]:
            continue

        first_hold = _clip01((lows[i - 1] - prev2_setup["mid"]) / max(prev2_setup["close"] - prev2_setup["mid"], 1e-12))
        confirm_strength = 0.5 * first_hold + 0.5 * _confirm_strength(closes[i], highs[i], lows[i], prev2_setup)

        ts = datetime.fromtimestamp(int(times[i]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        sig = _tier_payload("C", "continuation", {
            "bar_idx":       i,
            "date":          ts,
            "close":         round(closes[i], 8),
            "body_pct_%":    prev2_setup["body_pct_%"],
            "body_ratio":    round(prev2_setup["body_ratio"], 1),
            "avg_prior_%":   prev2_setup["avg_prior_%"],
            "vol_ratio":     round(prev2_setup["vol_ratio"], 1),
            "body_eff":      round(prev2_setup["body_eff"], 3),
            "mid":           prev2_setup["mid"],
            "high":          prev2_setup["high"],
        }, confirm_strength)
        signals.append(sig)
        last_sig = i

    return signals


# ─── TRIPLE BARRIER LABEL ─────────────────────────────────────────────────────

def simulate_nike_exit(
    sig: dict,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    bank_atr: float = _DEF["bank_atr"],
    sl_atr: float = _DEF["sl_atr"],
    bank_fraction: float = _DEF["bank_fraction"],
    runner_trail_atr: float = _DEF["runner_trail_atr"],
    max_bars_pre_bank: int = _DEF["max_bars_pre_bank"],
    max_bars_post_bank: int = _DEF["max_bars_post_bank"],
) -> dict:
    """Simulate Nike v2's bank-and-runner exit and return detailed outcome stats."""
    i = sig["bar_idx"]
    entry = closes[i]
    atr_i = atrs[i]
    if atr_i <= 0:
        return {"label": "TIMEOUT", "realized_atr": 0.0, "bank_hit": False, "exit_bar": i}

    bank_price = entry + bank_atr * atr_i
    stop_price = entry - sl_atr * atr_i
    n = len(closes)

    banked = False
    runner_peak = entry
    runner_stop = entry
    bank_gain_atr = 0.0
    last_bar = min(i + max_bars_post_bank, n - 1)

    for j in range(i + 1, last_bar + 1):
        high_j = highs[j]
        low_j = lows[j]
        close_j = closes[j]

        if not banked:
            if low_j <= stop_price:
                return {"label": "SL", "realized_atr": -sl_atr, "bank_hit": False, "exit_bar": j}
            if high_j >= bank_price:
                banked = True
                bank_gain_atr = bank_fraction * bank_atr
                runner_peak = max(bank_price, high_j)
                runner_stop = entry
                if j - i >= max_bars_post_bank:
                    realized = bank_gain_atr + (1.0 - bank_fraction) * ((close_j - entry) / atr_i)
                    return {"label": "TP", "realized_atr": realized, "bank_hit": True, "exit_bar": j}
                continue
            if j - i >= max_bars_pre_bank:
                realized = (close_j - entry) / atr_i
                return {"label": "TIMEOUT", "realized_atr": realized, "bank_hit": False, "exit_bar": j}
            continue

        runner_peak = max(runner_peak, high_j)
        runner_stop = max(runner_stop, runner_peak - runner_trail_atr * atr_i)
        if low_j <= runner_stop:
            realized = bank_gain_atr + (1.0 - bank_fraction) * ((runner_stop - entry) / atr_i)
            return {"label": "TP", "realized_atr": realized, "bank_hit": True, "exit_bar": j}
        if j - i >= max_bars_post_bank:
            realized = bank_gain_atr + (1.0 - bank_fraction) * ((close_j - entry) / atr_i)
            return {"label": "TP", "realized_atr": realized, "bank_hit": True, "exit_bar": j}

    if banked:
        realized = bank_gain_atr + (1.0 - bank_fraction) * ((closes[last_bar] - entry) / atr_i)
        return {"label": "TP", "realized_atr": realized, "bank_hit": True, "exit_bar": last_bar}
    realized = (closes[last_bar] - entry) / atr_i
    return {"label": "TIMEOUT", "realized_atr": realized, "bank_hit": False, "exit_bar": last_bar}


def label_signal(
    sig: dict,
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    atrs:   np.ndarray,
    tp_atr: float = _DEF["tp_atr"],
    sl_atr: float = _DEF["sl_atr"],
    max_bars: int  = _DEF["max_bars"],
) -> str:
    """Backward-compatible label wrapper for Nike v2 exit simulation."""
    return simulate_nike_exit(sig, closes, highs, lows, atrs)["label"]


# ─── EVALUATE ONE COIN ────────────────────────────────────────────────────────

def evaluate_coin(
    symbol: str,
    df: pd.DataFrame,
    params: dict,
    tp_atr:   float = _DEF["tp_atr"],
    sl_atr:   float = _DEF["sl_atr"],
    max_bars: int   = _DEF["max_bars"],
    verbose:  bool  = False,
) -> dict:
    closes  = df["close"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    atrs    = calc_atr(highs, lows, closes)

    signals = extract_nike_signals(df, **params)

    tp_n = sl_n = to_n = 0
    labeled = []
    for sig in signals:
        lbl = label_signal(sig, closes, highs, lows, atrs, tp_atr, sl_atr, max_bars)
        sig["label"] = lbl
        labeled.append(sig)
        if lbl == "TP":      tp_n += 1
        elif lbl == "SL":    sl_n += 1
        else:                to_n += 1

    n_sig   = len(labeled)
    decided = tp_n + sl_n               # TP or SL (timeout excluded from accuracy)
    acc     = tp_n / decided * 100 if decided > 0 else 0.0

    if verbose and labeled:
        print(f"\n  {symbol}  [{n_sig} signals  TP={tp_n}  SL={sl_n}  TO={to_n}  acc={acc:.0f}%]")
        for s in labeled:
            mark = "✓" if s["label"] == "TP" else ("✗" if s["label"] == "SL" else "~")
            print(f"    {mark} {s['date']}  body={s['body_pct_%']:+.2f}%  "
                  f"ratio={s['body_ratio']:.1f}×  quiet={s['avg_prior_%']:.3f}%  "
                  f"vol={s['vol_ratio']:.1f}×  → {s['label']}")

    return {
        "symbol":   symbol,
        "n_bars":   len(df),
        "n_sig":    n_sig,
        "tp":       tp_n,
        "sl":       sl_n,
        "timeout":  to_n,
        "decided":  decided,
        "acc_%":    round(acc, 1),
        "signals":  labeled,
    }


def evaluate_coin(
    symbol: str,
    df: pd.DataFrame,
    params: dict,
    tp_atr:   float = _DEF["tp_atr"],
    sl_atr:   float = _DEF["sl_atr"],
    max_bars: int   = _DEF["max_bars"],
    verbose:  bool  = False,
) -> dict:
    """Nike v2 evaluation with tier breakdown and bank-and-runner exit simulation."""
    closes  = df["close"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    atrs    = calc_atr(highs, lows, closes)

    signals = extract_nike_signals(df, **params)

    tp_n = sl_n = to_n = 0
    realized_atr_total = 0.0
    pos_atr_total = 0.0
    neg_atr_total = 0.0
    tier_stats = {
        "A": {"signals": 0, "tp": 0, "sl": 0, "timeout": 0, "realized_atr": 0.0},
        "B": {"signals": 0, "tp": 0, "sl": 0, "timeout": 0, "realized_atr": 0.0},
        "C": {"signals": 0, "tp": 0, "sl": 0, "timeout": 0, "realized_atr": 0.0},
    }
    labeled = []

    for sig in signals:
        sim = simulate_nike_exit(sig, closes, highs, lows, atrs)
        lbl = sim["label"]
        sig["label"] = lbl
        sig["realized_atr"] = round(float(sim["realized_atr"]), 4)
        sig["bank_hit"] = bool(sim["bank_hit"])
        sig["exit_bar"] = int(sim["exit_bar"])
        labeled.append(sig)

        if lbl == "TP":
            tp_n += 1
        elif lbl == "SL":
            sl_n += 1
        else:
            to_n += 1

        realized = float(sim["realized_atr"])
        realized_atr_total += realized
        if realized >= 0:
            pos_atr_total += realized
        else:
            neg_atr_total += realized

        tier = str(sig.get("tier", "B"))
        row = tier_stats.setdefault(tier, {"signals": 0, "tp": 0, "sl": 0, "timeout": 0, "realized_atr": 0.0})
        row["signals"] += 1
        row["realized_atr"] += realized
        if lbl == "TP":
            row["tp"] += 1
        elif lbl == "SL":
            row["sl"] += 1
        else:
            row["timeout"] += 1

    n_sig = len(labeled)
    decided = tp_n + sl_n
    acc = tp_n / decided * 100 if decided > 0 else 0.0
    expectancy_atr = realized_atr_total / n_sig if n_sig > 0 else 0.0
    weighted_pf = (pos_atr_total / abs(neg_atr_total)) if neg_atr_total < 0 else float("inf")

    if verbose and labeled:
        pf_txt = "inf" if weighted_pf == float("inf") else f"{weighted_pf:.2f}"
        print(f"\n  {symbol}  [{n_sig} signals  TP={tp_n}  SL={sl_n}  TO={to_n}  acc={acc:.0f}%  PF={pf_txt}  exp={expectancy_atr:+.2f} ATR]")
        for s in labeled:
            mark = "âœ“" if s["label"] == "TP" else ("âœ—" if s["label"] == "SL" else "~")
            print(f"    {mark} {s['date']}  tier={s.get('tier', '?')}  score={s.get('score', 0):.1f}  "
                  f"body={s['body_pct_%']:+.2f}%  ratio={s['body_ratio']:.1f}Ã—  "
                  f"quiet={s['avg_prior_%']:.3f}%  vol={s['vol_ratio']:.1f}Ã—  "
                  f"â†’ {s['label']} ({s.get('realized_atr', 0.0):+.2f} ATR)")

    return {
        "symbol": symbol,
        "n_bars": len(df),
        "n_sig": n_sig,
        "tp": tp_n,
        "sl": sl_n,
        "timeout": to_n,
        "decided": decided,
        "acc_%": round(acc, 1),
        "weighted_pf": weighted_pf,
        "expectancy_atr": expectancy_atr,
        "tier_stats": tier_stats,
        "signals": labeled,
    }


# ─── GRID SEARCH ─────────────────────────────────────────────────────────────

def grid_search(coin_data: dict[str, pd.DataFrame],
                tp_atr: float, sl_atr: float, max_bars: int) -> pd.DataFrame:
    """
    Try all combinations from GRID, return DataFrame sorted by TP accuracy.
    """
    from itertools import product

    keys   = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    rows   = []

    print(f"\n  Grid search: {len(combos)} param combos × {len(coin_data)} coins …")

    for combo in combos:
        params = dict(zip(keys, combo))
        # add fixed params not in grid
        for k, v in _DEF.items():
            if k not in params:
                params[k] = v

        total_tp = total_sl = total_sig = 0
        for df in coin_data.values():
            sigs = extract_nike_signals(df, **{k: params[k] for k in
                   ["body_ratio_mult","quiet_body_pct","vol_mult","body_min"]})
            closes = df["close"].values.astype(np.float64)
            highs  = df["high"].values.astype(np.float64)
            lows   = df["low"].values.astype(np.float64)
            atrs   = calc_atr(highs, lows, closes)
            for sig in sigs:
                lbl = label_signal(sig, closes, highs, lows, atrs, tp_atr, sl_atr, max_bars)
                total_sig += 1
                if lbl == "TP":   total_tp += 1
                elif lbl == "SL": total_sl += 1

        decided = total_tp + total_sl
        acc = total_tp / decided * 100 if decided > 0 else 0.0
        # Weighted PF: each TP wins tp_atr units, each SL loses sl_atr units
        weighted_pf = (total_tp * tp_atr) / (total_sl * sl_atr) if total_sl > 0 else float("inf")

        rows.append({
            "body_ratio_mult": params["body_ratio_mult"],
            "quiet_body_pct":  params["quiet_body_pct"],
            "vol_mult":        params["vol_mult"],
            "n_sig":           total_sig,
            "tp":              total_tp,
            "sl":              total_sl,
            "acc_%":           round(acc, 1),
            "profit_factor":   round(weighted_pf, 2),   # dollar-weighted PF
        })

    df_grid = pd.DataFrame(rows).sort_values(
        ["acc_%","profit_factor"], ascending=False
    ).reset_index(drop=True)
    return df_grid


# ─── LIVE WEBSOCKET MONITOR ───────────────────────────────────────────────────

async def live_monitor(symbols: list[str], params: dict,
                       tp_atr: float, sl_atr: float, max_bars: int):
    """
    Subscribe to 5m kline streams for the top gainers.
    Fires Nike check on each closed candle.
    Tracks open signals and reports TP/SL/timeout outcomes.
    """
    try:
        import websockets
    except ImportError:
        print("[LIVE] websockets library not found. pip install websockets")
        return

    streams  = "/".join(f"{s.lower()}@kline_5m" for s in symbols[:20])  # max 20 streams
    ws_url   = f"wss://fstream.binance.com/stream?streams={streams}"

    # rolling buffer per symbol: last 30 bars
    buffers: dict[str, list] = {s: [] for s in symbols}
    open_signals: dict[str, dict] = {}    # symbol → active signal info

    print(f"\n{DSEP}")
    print(f" LIVE MONITOR — {len(symbols[:20])} coins  (Ctrl-C to stop)")
    print(DSEP)

    try:
        async with websockets.connect(ws_url, ping_interval=20) as ws:
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                data   = msg.get("data", {})
                k      = data.get("k", {})
                sym    = data.get("stream", "").split("@")[0].upper()
                closed = k.get("x", False)

                if not closed:
                    continue

                bar = {
                    "open_time": k["t"],
                    "open":      float(k["o"]),
                    "high":      float(k["h"]),
                    "low":       float(k["l"]),
                    "close":     float(k["c"]),
                    "volume":    float(k["v"]),
                }
                buffers[sym].append(bar)
                if len(buffers[sym]) > 50:        # keep rolling 50-bar window
                    buffers[sym].pop(0)

                if len(buffers[sym]) < LOOKBACK + 2:
                    continue

                df_live = pd.DataFrame(buffers[sym])

                # ── Check if open signal hit TP/SL ──────────────────────────
                if sym in open_signals:
                    sig = open_signals[sym]
                    closes  = df_live["close"].values.astype(np.float64)
                    highs   = df_live["high"].values.astype(np.float64)
                    lows    = df_live["low"].values.astype(np.float64)
                    atrs    = calc_atr(highs, lows, closes)
                    # Use last bar to check against stored entry levels
                    h, l = bar["high"], bar["low"]
                    if h >= sig["tp_price"]:
                        pnl = (sig["tp_price"] - sig["entry"]) / sig["entry"] * 100
                        ts  = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                        print(f"  [{ts}] {sym:<18} TP HIT  +{pnl:.2f}%  (entry {sig['entry']:.6g})")
                        del open_signals[sym]
                    elif l <= sig["sl_price"]:
                        pnl = (sig["sl_price"] - sig["entry"]) / sig["entry"] * 100
                        ts  = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                        print(f"  [{ts}] {sym:<18} SL HIT  {pnl:.2f}%  (entry {sig['entry']:.6g})")
                        del open_signals[sym]
                    elif len(buffers[sym]) - sig["entry_buf_len"] >= max_bars:
                        ts  = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                        print(f"  [{ts}] {sym:<18} TIMEOUT  (entry {sig['entry']:.6g})")
                        del open_signals[sym]

                # ── Check for new Nike signal ────────────────────────────────
                if sym not in open_signals:
                    new_sigs = extract_nike_signals(df_live, **params)
                    if new_sigs:
                        # Check if last signal is on the latest closed bar
                        latest = new_sigs[-1]
                        if latest["bar_idx"] == len(df_live) - 1:
                            closes2 = df_live["close"].values.astype(np.float64)
                            highs2  = df_live["high"].values.astype(np.float64)
                            lows2   = df_live["low"].values.astype(np.float64)
                            atrs2   = calc_atr(highs2, lows2, closes2)
                            atr_i   = atrs2[-1]
                            entry   = bar["close"]
                            tp_p    = entry + tp_atr * atr_i
                            sl_p    = entry - sl_atr * atr_i
                            open_signals[sym] = {
                                "entry":        entry,
                                "tp_price":     tp_p,
                                "sl_price":     sl_p,
                                "entry_buf_len":len(buffers[sym]),
                            }
                            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                            print(f"\n  [{ts}] NIKE SIGNAL  {sym:<18}  "
                                  f"body={latest['body_pct_%']:+.2f}%  "
                                  f"ratio={latest['body_ratio']:.1f}x  "
                                  f"quiet={latest['avg_prior_%']:.3f}%  "
                                  f"vol={latest['vol_ratio']:.1f}x")
                            print(f"         entry={entry:.6g}  "
                                  f"TP={tp_p:.6g} (+{tp_atr}×ATR)  "
                                  f"SL={sl_p:.6g} (-{sl_atr}×ATR)")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[LIVE] WebSocket error: {e}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",      type=int, default=40,    help="Number of top gainers to fetch")
    ap.add_argument("--no-grid",  action="store_true",      help="Skip grid search")
    ap.add_argument("--proxy",    action="store_true",      help="Route through QUANTA proxy (port 52681)")
    ap.add_argument("--live",     action="store_true",      help="Run live WebSocket monitor after backtest")
    ap.add_argument("--tp",       type=float, default=_DEF["tp_atr"],   help="TP in ATR multiples")
    ap.add_argument("--sl",       type=float, default=_DEF["sl_atr"],   help="SL in ATR multiples")
    ap.add_argument("--max-bars", type=int,   default=_DEF["max_bars"], help="Max bars to hold")
    args = ap.parse_args()

    proxy_port = 52681 if args.proxy else None
    sess = _session(proxy_port)

    # ─── 1. Get all symbols ───────────────────────────────────────────────────
    print(DSEP)
    print(" QUANTA Nike Live Validator")
    print(f" Fetching top {args.top} 30-day gainers from Binance Futures …")
    print(DSEP)

    print("  [1/4] Loading exchange symbols …", end=" ", flush=True)
    try:
        all_symbols = get_futures_symbols(sess)
    except Exception as e:
        print(f"\n[ERROR] Cannot reach Binance: {e}")
        sys.exit(1)
    print(f"{len(all_symbols)} symbols found")

    # ─── 2. Compute 30-day % change (parallel) ───────────────────────────────
    print(f"  [2/4] Computing 30-day returns for {len(all_symbols)} symbols …", end=" ", flush=True)
    changes: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(get_30d_change, sess, sym): sym for sym in all_symbols}
        done = 0
        for f in as_completed(futs):
            sym, pct = f.result()
            changes[sym] = pct
            done += 1
            if done % 50 == 0:
                print(f"{done}…", end=" ", flush=True)

    top_gainers = sorted(changes.items(), key=lambda x: x[1], reverse=True)[:args.top]
    print(f"\n  Top {args.top} gainers identified")
    print(f"\n  {'Rank':<5} {'Symbol':<20} {'30d Change':>12}")
    print(f"  {SEP[:50]}")
    for rank, (sym, pct) in enumerate(top_gainers, 1):
        print(f"  {rank:<5} {sym:<20} {pct:>+11.1f}%")

    top_symbols = [s for s, _ in top_gainers]

    # ─── 3. Download 5m OHLCV ────────────────────────────────────────────────
    print(f"\n  [3/4] Downloading {DAYS_BACK}d of 5m data for each coin …")
    coin_data: dict[str, pd.DataFrame] = {}

    def _dl(sym):
        df = fetch_klines_5m(sess, sym)
        return sym, df

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_dl, sym): sym for sym in top_symbols}
        for i, f in enumerate(as_completed(futs), 1):
            sym, df = f.result()
            if df is not None and len(df) >= LOOKBACK + 50:
                coin_data[sym] = df
                print(f"  {i:>3}/{args.top}  {sym:<20}  {len(df):>6} bars  "
                      f"{datetime.fromtimestamp(df['open_time'].iloc[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → "
                      f"{datetime.fromtimestamp(df['open_time'].iloc[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
            else:
                print(f"  {i:>3}/{args.top}  {sym:<20}  SKIPPED (insufficient data)")

    print(f"\n  {len(coin_data)} coins with sufficient data")

    # ─── 4. Evaluate with current params ─────────────────────────────────────
    print(f"\n{DSEP}")
    print(f" EVALUATION WITH CURRENT PARAMS")
    print(f"  body_ratio_mult = {_DEF['body_ratio_mult']}×  |  "
          f"quiet_body_pct = {_DEF['quiet_body_pct']}%  |  "
          f"vol_mult = {_DEF['vol_mult']}×")
    print(f"  TP = {args.tp}×ATR  |  SL = {args.sl}×ATR  |  max_bars = {args.max_bars}")
    print(DSEP)

    current_params = {k: _DEF[k] for k in ["body_ratio_mult","quiet_body_pct","vol_mult","body_min"]}
    results = []
    for sym in sorted(coin_data.keys()):
        res = evaluate_coin(sym, coin_data[sym], current_params,
                            args.tp, args.sl, args.max_bars, verbose=True)
        results.append(res)

    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "signals"} for r in results])
    df_res = df_res.sort_values("acc_%", ascending=False).reset_index(drop=True)

    total_sig  = df_res["n_sig"].sum()
    total_tp   = df_res["tp"].sum()
    total_sl   = df_res["sl"].sum()
    total_dec  = df_res["decided"].sum()
    overall_acc= total_tp / total_dec * 100 if total_dec > 0 else 0.0
    pf         = total_tp / total_sl if total_sl > 0 else float("inf")

    print(f"\n{DSEP}")
    print(f" PER-COIN SUMMARY  (sorted by accuracy)")
    print(DSEP)
    print(f"  {'Symbol':<22} {'Bars':>6}  {'Signals':>8}  {'TP':>5}  {'SL':>5}  {'TO':>5}  {'Acc%':>7}  {'PF':>6}")
    print(f"  {SEP}")
    for _, row in df_res.iterrows():
        pf_c = row['tp'] / row['sl'] if row['sl'] > 0 else float('inf')
        print(f"  {row['symbol']:<22} {row['n_bars']:>6}  {row['n_sig']:>8}  "
              f"{row['tp']:>5}  {row['sl']:>5}  {row['timeout']:>5}  "
              f"{row['acc_%']:>7.1f}%  "
              f"{'inf':>6}" if pf_c == float('inf') else
              f"  {row['symbol']:<22} {row['n_bars']:>6}  {row['n_sig']:>8}  "
              f"{row['tp']:>5}  {row['sl']:>5}  {row['timeout']:>5}  "
              f"{row['acc_%']:>7.1f}%  {pf_c:>6.2f}")

    print(f"\n  {'TOTAL':<22} {'':>6}  {total_sig:>8}  {total_tp:>5}  {total_sl:>5}  "
          f"{df_res['timeout'].sum():>5}  {overall_acc:>7.1f}%  {pf:>6.2f}")
    print(f"\n  Overall: {total_sig} signals | {overall_acc:.1f}% accuracy | "
          f"Profit factor {pf:.2f}")

    # ─── 5. Grid search ───────────────────────────────────────────────────────
    if not args.no_grid:
        df_grid = grid_search(coin_data, args.tp, args.sl, args.max_bars)

        print(f"\n{DSEP}")
        print(f" GRID SEARCH RESULTS  (top 20 combinations)")
        print(DSEP)
        print(f"  {'body_ratio':>11}  {'quiet_pct':>10}  {'vol_mult':>9}  "
              f"{'N_Sig':>7}  {'TP':>5}  {'SL':>5}  {'Acc%':>7}  {'PF':>7}")
        print(f"  {SEP}")
        for _, row in df_grid.head(20).iterrows():
            print(f"  {row['body_ratio_mult']:>11.1f}×  "
                  f"{row['quiet_body_pct']:>9.1f}%  "
                  f"{row['vol_mult']:>9.1f}×  "
                  f"{row['n_sig']:>7}  "
                  f"{row['tp']:>5}  "
                  f"{row['sl']:>5}  "
                  f"{row['acc_%']:>7.1f}%  "
                  f"{row['profit_factor']:>7.2f}")

        best = df_grid.iloc[0]
        print(f"\n  BEST PARAMS (highest accuracy with most signals):")
        print(f"    body_ratio_mult = {best['body_ratio_mult']}×")
        print(f"    quiet_body_pct  = {best['quiet_body_pct']}%")
        print(f"    vol_mult        = {best['vol_mult']}×")
        print(f"    → Acc = {best['acc_%']}%  |  PF = {best['profit_factor']}  |  Signals = {best['n_sig']}")

        # Export grid results
        grid_out = Path("nike_grid_results.csv")
        df_grid.to_csv(grid_out, index=False)
        print(f"\n  Full grid results → {grid_out.resolve()}")

        # Apply best params to config suggestion
        print(f"\n  To apply best params, update quanta_config.py:")
        print(f"    nike_body_ratio_mult: float = {best['body_ratio_mult']}")
        print(f"    nike_quiet_body_pct:  float = {best['quiet_body_pct']}")
        print(f"    nike_vol_mult:        float = {best['vol_mult']}")

    # ─── 6. Save signal log ───────────────────────────────────────────────────
    all_sigs = []
    for r in results:
        for sig in r["signals"]:
            all_sigs.append({"symbol": r["symbol"], **sig})
    if all_sigs:
        sig_out = Path("nike_live_signals.csv")
        pd.DataFrame(all_sigs).to_csv(sig_out, index=False)
        print(f"\n  Signal log → {sig_out.resolve()}")

    print(DSEP)

    # ─── 7. Live monitor ──────────────────────────────────────────────────────
    if args.live:
        best_params = current_params
        if not args.no_grid:
            best_params = {
                "body_ratio_mult": best["body_ratio_mult"],
                "quiet_body_pct":  best["quiet_body_pct"],
                "vol_mult":        best["vol_mult"],
                "body_min":        _DEF["body_min"],
            }
        asyncio.run(live_monitor(top_symbols, best_params, args.tp, args.sl, args.max_bars))


if __name__ == "__main__":
    main()
