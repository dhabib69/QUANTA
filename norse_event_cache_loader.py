"""
norse_event_cache_loader.py
============================
Fast loader: reads the per-symbol .npz + .events.json written by
norse_event_cache_builder.py and reconstructs a PreparedSymbol in ~50 ms
instead of the 5-8 min that _prepare_symbol + feature extraction takes.

Public API
----------
    from norse_event_cache_loader import load_cached_symbol, cache_exists

    prep = load_cached_symbol("BTCUSDT")   # None on cache miss / stale
    if prep is None:
        prep = _prepare_symbol(...)        # fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "norse_event_cache"
MANIFEST_PATH = CACHE_DIR / "_manifest.json"
SCHEMA_VERSION = 3   # must match norse_event_cache_builder.py

_REQUIRED_ARRAY_KEYS = (
    "opens",
    "highs",
    "lows",
    "closes",
    "volumes",
    "times",
    "atrs",
    "vol_avg",
    "volume_ratio",
    "mean_volume_ratio",
    "volume_ratio_slope",
    "quote_volume",
    "quote_volume_slope",
    "vpin",
    "vpin_slope",
    "taker_imbalance",
    "taker_slope",
    "regime_state",
    "weighted_trend",
    "bull_ratio",
    "bear_ratio",
    "bs_prob",
    "bs_time_decay",
    "bs_iv_ratio",
    "impulse_body_eff",
    "impulse_taker_persist",
    "pre_impulse_r2",
    "atr_rank",
    "depth_delta",
    "vol_delta",
    "vpin_delta",
    "close_pos",
    "upper_wick_ratio",
    "participation_score",
    "flow_exhaustion_score",
    "klines_np",
)
_REQUIRED_JSON_KEYS = (
    "symbol",
    "n_bars",
    "thor_signals",
    "baldur_warnings",
    "baldur_signals",
    "freya_signals",
    "max_open_time_ms",
)
_ARRAY_LENGTH_KEYS = tuple(key for key in _REQUIRED_ARRAY_KEYS if key != "klines_np")


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def load_cache_manifest() -> Optional[dict]:
    if not _is_nonempty_file(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    return manifest


def _load_cache_payload(
    symbol: str,
    expected_max_open_time_ms: Optional[int] = None,
    include_arrays: bool = False,
) -> tuple[Optional[dict], str]:
    npz_path = CACHE_DIR / f"{symbol}_5m.npz"
    json_path = CACHE_DIR / f"{symbol}_5m.events.json"

    if not (_is_nonempty_file(npz_path) and _is_nonempty_file(json_path)):
        return None, "missing_or_empty_files"

    try:
        with open(json_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None, "bad_json"

    missing_json_keys = [key for key in _REQUIRED_JSON_KEYS if key not in meta]
    if missing_json_keys:
        return None, f"missing_json_keys:{','.join(missing_json_keys)}"
    if meta.get("schema_version") != SCHEMA_VERSION:
        return None, "schema_mismatch"
    if str(meta.get("symbol", "")) != symbol:
        return None, "symbol_mismatch"

    cache_token = meta.get("cache_token")
    if not cache_token:
        return None, "missing_cache_token"

    try:
        n_bars = int(meta["n_bars"])
    except Exception:
        return None, "bad_n_bars"
    if n_bars <= 0:
        return None, "bad_n_bars"

    try:
        max_open_time_ms = int(meta["max_open_time_ms"])
    except Exception:
        return None, "bad_max_open_time_ms"
    if expected_max_open_time_ms is not None and max_open_time_ms < int(expected_max_open_time_ms):
        return None, "stale_cache"

    data = {} if include_arrays else None
    try:
        with np.load(npz_path, allow_pickle=False) as arr:
            if "cache_token" not in arr.files:
                return None, "missing_npz_cache_token"
            try:
                npz_token = str(arr["cache_token"].item())
            except Exception:
                return None, "bad_npz_cache_token"
            if npz_token != str(cache_token):
                return None, "cache_token_mismatch"

            missing_array_keys = [key for key in _REQUIRED_ARRAY_KEYS if key not in arr.files]
            if missing_array_keys:
                return None, f"missing_array_keys:{','.join(missing_array_keys)}"

            for key in _ARRAY_LENGTH_KEYS:
                value = np.asarray(arr[key])
                if value.ndim != 1 or len(value) != n_bars:
                    return None, f"bad_shape:{key}"
                if include_arrays:
                    data[key] = value

            klines_np = np.asarray(arr["klines_np"])
            if klines_np.ndim != 2 or klines_np.shape[0] != n_bars:
                return None, "bad_shape:klines_np"
            if include_arrays:
                data["klines_np"] = klines_np
    except Exception:
        return None, "bad_npz"

    payload = {"meta": meta}
    if include_arrays:
        payload["data"] = data
    return payload, "ok"


def inspect_cache_pair(symbol: str, expected_max_open_time_ms: Optional[int] = None) -> tuple[bool, str]:
    payload, reason = _load_cache_payload(
        symbol,
        expected_max_open_time_ms=expected_max_open_time_ms,
        include_arrays=False,
    )
    return payload is not None, reason


def validate_cache_universe(universe) -> dict:
    manifest = load_cache_manifest()
    invalid_by_reason: dict[str, int] = {}
    sample_failures: list[dict] = []
    valid_symbols = 0

    for symbol, df in universe:
        ok, reason = inspect_cache_pair(symbol, expected_max_open_time_ms=int(df["open_time"].max()))
        if ok:
            valid_symbols += 1
            continue
        invalid_by_reason[reason] = invalid_by_reason.get(reason, 0) + 1
        if len(sample_failures) < 10:
            sample_failures.append({"symbol": symbol, "reason": reason})

    total_symbols = len(universe)
    invalid_symbols = total_symbols - valid_symbols
    manifest_complete = bool(manifest and manifest.get("complete"))
    manifest_matches = bool(
        manifest
        and int(manifest.get("symbols_total", -1)) == total_symbols
        and int(manifest.get("symbols_valid", -1)) == valid_symbols
        and int(manifest.get("symbols_failed", -1)) == 0
    )
    return {
        "complete": manifest_complete and manifest_matches and invalid_symbols == 0,
        "manifest_present": manifest is not None,
        "manifest_complete": manifest_complete,
        "manifest": manifest,
        "symbols_total": total_symbols,
        "symbols_valid": valid_symbols,
        "symbols_invalid": invalid_symbols,
        "invalid_by_reason": invalid_by_reason,
        "sample_failures": sample_failures,
    }


def cache_exists(symbol: str) -> bool:
    ok, _ = inspect_cache_pair(symbol)
    return ok


def load_cached_symbol(symbol: str) -> Optional["PreparedSymbol"]:
    """
    Return a PreparedSymbol loaded entirely from disk, or None if:
      - cache files don't exist
      - schema_version mismatch (stale cache)
      - zero-byte / corrupt / mismatched cache pair
      - any read error
    """
    payload, _ = _load_cache_payload(symbol, include_arrays=True)
    if payload is None:
        return None
    meta = payload["meta"]
    data = payload["data"]

    from quanta_norse_agents import SparseFeatureContext
    from quanta_norse_year_sim import PreparedSymbol

    def _a(key: str) -> np.ndarray:
        return data[key]

    ctx = SparseFeatureContext(
        symbol=symbol,
        feature_map={},
        opens=_a("opens"),
        highs=_a("highs"),
        lows=_a("lows"),
        closes=_a("closes"),
        volumes=_a("volumes"),
        times=_a("times").astype(np.int64),
        atrs=_a("atrs"),
        vol_avg=_a("vol_avg"),
        volume_ratio=_a("volume_ratio"),
        mean_volume_ratio=_a("mean_volume_ratio"),
        volume_ratio_slope=_a("volume_ratio_slope"),
        quote_volume=_a("quote_volume"),
        quote_volume_slope=_a("quote_volume_slope"),
        vpin=_a("vpin"),
        vpin_slope=_a("vpin_slope"),
        taker_imbalance=_a("taker_imbalance"),
        taker_slope=_a("taker_slope"),
        regime_state=_a("regime_state"),
        weighted_trend=_a("weighted_trend"),
        bull_ratio=_a("bull_ratio"),
        bear_ratio=_a("bear_ratio"),
        bs_prob=_a("bs_prob"),
        bs_time_decay=_a("bs_time_decay"),
        bs_iv_ratio=_a("bs_iv_ratio"),
        impulse_body_eff=_a("impulse_body_eff"),
        impulse_taker_persist=_a("impulse_taker_persist"),
        pre_impulse_r2=_a("pre_impulse_r2"),
        atr_rank=_a("atr_rank"),
        depth_delta=_a("depth_delta"),
        vol_delta=_a("vol_delta"),
        vpin_delta=_a("vpin_delta"),
        close_pos=_a("close_pos"),
        upper_wick_ratio=_a("upper_wick_ratio"),
        participation_score=_a("participation_score"),
        flow_exhaustion_score=_a("flow_exhaustion_score"),
    )

    df = pd.DataFrame(
        {
            "open_time": _a("times").astype(np.int64),
            "open": _a("opens"),
            "high": _a("highs"),
            "low": _a("lows"),
            "close": _a("closes"),
            "volume": _a("volumes"),
        }
    )
    df.attrs["symbol"] = symbol

    prep = PreparedSymbol(
        symbol=symbol,
        df=df,
        atrs=_a("atrs"),
        raw_thor_signals=meta["thor_signals"],
        replay_engine=None,
        precomputed={},
        klines_np=_a("klines_np"),
        feature_cache={},
        feature_ctx=ctx,
        norse_event_cache={
            "baldur_warnings": meta["baldur_warnings"],
            "baldur_signals": meta["baldur_signals"],
            "freya_signals": meta["freya_signals"],
        },
    )
    return prep
