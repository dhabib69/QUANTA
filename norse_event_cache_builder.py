"""
norse_event_cache_builder.py
============================
One-time script: builds a compact on-disk cache of all Norse events +
SparseFeatureContext arrays for every symbol in the 365-day windowed
feather cache.

After this runs (~30-45 min, 6 threads), every subsequent year-sim
and tuning run loads from disk in ~10 s instead of spending 20-30 min
on _prepare_symbol + extract_offline_features_for_positions.

Usage
-----
    python norse_event_cache_builder.py [--days 365] [--workers 6] [--force]

Idempotent: skips symbols whose .npz mtime >= source feather max_ts.
Use --force to rebuild everything.

Outputs per symbol
------------------
  norse_event_cache/{SYMBOL}_5m.npz        - SparseFeatureContext arrays + klines_np
  norse_event_cache/{SYMBOL}_5m.events.json - thor/baldur/freya signal lists
  norse_event_cache/_manifest.json          - run metadata
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "norse_event_cache"
MANIFEST_PATH = CACHE_DIR / "_manifest.json"
SCHEMA_VERSION = 3   # bump when NPZ schema changes
_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[cache {time.time() - _T0:6.0f}s] {msg}", flush=True)


def _clean(obj):
    """Recursively convert numpy scalars → Python primitives for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _ctx_to_arrays(ctx) -> dict:
    """Materialise all 33 SparseFeatureContext fields into a flat dict."""
    return {
        "opens":                  np.asarray(ctx.opens,                  dtype=np.float64),
        "highs":                  np.asarray(ctx.highs,                  dtype=np.float64),
        "lows":                   np.asarray(ctx.lows,                   dtype=np.float64),
        "closes":                 np.asarray(ctx.closes,                 dtype=np.float64),
        "volumes":                np.asarray(ctx.volumes,                dtype=np.float64),
        "times":                  np.asarray(ctx.times,                  dtype=np.int64),
        "atrs":                   np.asarray(ctx.atrs,                   dtype=np.float64),
        "vol_avg":                np.asarray(ctx.vol_avg,                dtype=np.float64),
        "volume_ratio":           np.asarray(ctx.volume_ratio,           dtype=np.float64),
        "mean_volume_ratio":      np.asarray(ctx.mean_volume_ratio,      dtype=np.float64),
        "volume_ratio_slope":     np.asarray(ctx.volume_ratio_slope,     dtype=np.float64),
        "quote_volume":           np.asarray(ctx.quote_volume,           dtype=np.float64),
        "quote_volume_slope":     np.asarray(ctx.quote_volume_slope,     dtype=np.float64),
        "vpin":                   np.asarray(ctx.vpin,                   dtype=np.float64),
        "vpin_slope":             np.asarray(ctx.vpin_slope,             dtype=np.float64),
        "taker_imbalance":        np.asarray(ctx.taker_imbalance,        dtype=np.float64),
        "taker_slope":            np.asarray(ctx.taker_slope,            dtype=np.float64),
        "regime_state":           np.asarray(ctx.regime_state,           dtype=np.float64),
        "weighted_trend":         np.asarray(ctx.weighted_trend,         dtype=np.float64),
        "bull_ratio":             np.asarray(ctx.bull_ratio,             dtype=np.float64),
        "bear_ratio":             np.asarray(ctx.bear_ratio,             dtype=np.float64),
        "bs_prob":                np.asarray(ctx.bs_prob,                dtype=np.float64),
        "bs_time_decay":          np.asarray(ctx.bs_time_decay,          dtype=np.float64),
        "bs_iv_ratio":            np.asarray(ctx.bs_iv_ratio,            dtype=np.float64),
        "impulse_body_eff":       np.asarray(ctx.impulse_body_eff,       dtype=np.float64),
        "impulse_taker_persist":  np.asarray(ctx.impulse_taker_persist,  dtype=np.float64),
        "pre_impulse_r2":         np.asarray(ctx.pre_impulse_r2,         dtype=np.float64),
        "atr_rank":               np.asarray(ctx.atr_rank,               dtype=np.float64),
        "depth_delta":            np.asarray(ctx.depth_delta,            dtype=np.float64),
        "vol_delta":              np.asarray(ctx.vol_delta,              dtype=np.float64),
        "vpin_delta":             np.asarray(ctx.vpin_delta,             dtype=np.float64),
        "close_pos":              np.asarray(ctx.close_pos,              dtype=np.float64),
        "upper_wick_ratio":       np.asarray(ctx.upper_wick_ratio,       dtype=np.float64),
        "participation_score":    np.asarray(ctx.participation_score,    dtype=np.float64),
        "flow_exhaustion_score":  np.asarray(ctx.flow_exhaustion_score,  dtype=np.float64),
    }


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _cache_pair_is_current(npz_path: Path, json_path: Path, df_max_ts: int) -> bool:
    if not (_is_nonempty_file(npz_path) and _is_nonempty_file(json_path)):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("schema_version") != SCHEMA_VERSION
            or not meta.get("cache_token")
            or meta.get("max_open_time_ms", 0) < df_max_ts
        ):
            return False
        with np.load(npz_path, allow_pickle=False) as arr:
            if "cache_token" not in arr.files:
                return False
            return str(arr["cache_token"].item()) == str(meta["cache_token"])
    except Exception:
        return False


def _write_json_atomic(target_path: Path, payload: dict) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".tmp.json",
        dir=CACHE_DIR,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=float)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
    finally:
        _safe_unlink(temp_path)


def _write_npz_atomic(target_path: Path, arrays: dict) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".tmp.npz",
        dir=CACHE_DIR,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, target_path)
    finally:
        _safe_unlink(temp_path)


def _stage_json_temp(target_path: Path, payload: dict) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".tmp.json",
        dir=CACHE_DIR,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=float)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        _safe_unlink(temp_path)
        raise


def _stage_npz_temp(target_path: Path, arrays: dict) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.",
        suffix=".tmp.npz",
        dir=CACHE_DIR,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        return temp_path
    except Exception:
        _safe_unlink(temp_path)
        raise


def _write_cache_pair_atomic(npz_path: Path, json_path: Path, arrays: dict, events: dict) -> None:
    npz_temp_path = None
    json_temp_path = None
    try:
        npz_temp_path = _stage_npz_temp(npz_path, arrays)
        json_temp_path = _stage_json_temp(json_path, events)
        os.replace(npz_temp_path, npz_path)
        npz_temp_path = None
        os.replace(json_temp_path, json_path)
        json_temp_path = None
    finally:
        if npz_temp_path is not None:
            _safe_unlink(npz_temp_path)
        if json_temp_path is not None:
            _safe_unlink(json_temp_path)


def _purge_stale_temp_files() -> None:
    for temp_path in CACHE_DIR.glob(".*.tmp.*"):
        _safe_unlink(temp_path)


def _clean_cache_outputs() -> int:
    deleted = 0
    patterns = (
        "*_5m.npz",
        "*_5m.events.json",
        "_manifest.json",
        ".*.tmp.*",
    )
    for pattern in patterns:
        for path in CACHE_DIR.glob(pattern):
            if path.is_file():
                _safe_unlink(path)
                deleted += 1
    return deleted


def _build_one(
    symbol: str,
    df,
    replay_engine,
    ev,
    force: bool = False,
) -> tuple[str, str]:
    """Build cache for one symbol.  Returns ('built'|'skip'|'fail', symbol)."""
    npz_path  = CACHE_DIR / f"{symbol}_5m.npz"
    json_path = CACHE_DIR / f"{symbol}_5m.events.json"
    df_max_ts = int(df["open_time"].max())

    if not force and _cache_pair_is_current(npz_path, json_path, df_max_ts):
        return "skip", symbol

    # ----- import heavy deps inside thread so they're not serialised --------
    from quanta_norse_year_sim import _prepare_symbol, _ensure_feature_positions
    from quanta_norse_agents import extract_baldur_signals, extract_freya_signals

    try:
        prep = _prepare_symbol(symbol, df, replay_engine, ev)

        # Prime features for every signal bar (Thor + Baldur + Freya)
        all_signal_bars: list[int] = [int(s["bar_idx"]) for s in prep.raw_thor_signals]

        baldur_warnings, baldur_signals = extract_baldur_signals(
            prep.df, prep.raw_thor_signals, ev
        )
        freya_signals = extract_freya_signals(prep.df, prep.raw_thor_signals, ev)

        extra_bars = (
            [int(w["bar_idx"]) for w in baldur_warnings]
            + [int(s["bar_idx"]) for s in baldur_signals]
            + [int(s["bar_idx"]) for s in freya_signals]
        )
        _ensure_feature_positions(prep, all_signal_bars + extra_bars)

        # ---- write NPZ -------------------------------------------------------
        cache_token = uuid.uuid4().hex
        arrays = _ctx_to_arrays(prep.feature_ctx)
        arrays["klines_np"] = np.asarray(prep.klines_np, dtype=np.float64)
        arrays["cache_token"] = np.asarray(cache_token)

        # ---- write events JSON -----------------------------------------------
        events = {
            "schema_version":    SCHEMA_VERSION,
            "cache_token":       cache_token,
            "symbol":            symbol,
            "max_open_time_ms":  df_max_ts,
            "n_bars":            int(len(prep.df)),
            "thor_signals":      _clean(prep.raw_thor_signals),
            "baldur_warnings":   _clean(baldur_warnings),
            "baldur_signals":    _clean(baldur_signals),
            "freya_signals":     _clean(freya_signals),
        }
        _write_cache_pair_atomic(npz_path, json_path, arrays, events)

        return "built", symbol

    except Exception as exc:
        return f"fail:{exc}", symbol


def main(days: int = 365, workers: int = 6, force: bool = False, clean: bool = False) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _purge_stale_temp_files()
    _safe_unlink(MANIFEST_PATH)
    if clean:
        deleted = _clean_cache_outputs()
        _log(f"cleaned cache dir  deleted_files={deleted}")

    from quanta_config import Config
    from quanta_norse_year_sim import _load_windowed_cache
    from QUANTA_ml_engine import build_offline_feature_replay_engine

    ev = Config.events
    replay_engine = build_offline_feature_replay_engine(Config)
    universe = _load_windowed_cache(days=days)
    _log(f"{len(universe)} symbols loaded — starting cache build (workers={workers})")

    built = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_build_one, sym, df, replay_engine, ev, force): sym
            for sym, df in universe
        }
        for done_i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                status, _ = fut.result()
            except Exception as exc:
                status = f"fail:{exc}"

            if status == "skip":
                skipped += 1
            elif status == "built":
                built += 1
            else:
                failed += 1
                _log(f"  FAIL {sym}: {status}")

            if done_i % 20 == 0 or done_i == len(futures):
                _log(
                    f"  {done_i}/{len(futures)} done  "
                    f"built={built}  skipped={skipped}  failed={failed}"
                )

    valid_symbols = sum(
        1
        for sym, df in universe
        if _cache_pair_is_current(
            CACHE_DIR / f"{sym}_5m.npz",
            CACHE_DIR / f"{sym}_5m.events.json",
            int(df["open_time"].max()),
        )
    )
    complete = failed == 0 and valid_symbols == len(universe)

    manifest = {
        "schema_version":   SCHEMA_VERSION,
        "built_at_ms":      int(time.time() * 1000),
        "days_window":      days,
        "symbols_total":    len(universe),
        "symbols_built":    built,
        "symbols_skipped":  skipped,
        "symbols_failed":   failed,
        "symbols_valid":    valid_symbols,
        "complete":         complete,
        "duration_seconds": round(time.time() - _T0, 1),
    }
    if complete:
        _write_json_atomic(MANIFEST_PATH, manifest)
    else:
        _log(
            "manifest withheld because cache is incomplete "
            f"(valid={valid_symbols}/{len(universe)} failed={failed})"
        )

    _log(
        f"DONE  built={built}  skipped={skipped}  failed={failed}  "
        f"total={round(time.time()-_T0,0):.0f}s"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Norse event + feature cache")
    ap.add_argument("--days",    type=int, default=365)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force",   action="store_true", help="Rebuild all symbols")
    ap.add_argument("--clean",   action="store_true", help="Delete existing cache files in norse_event_cache before rebuilding")
    args = ap.parse_args()
    main(days=args.days, workers=args.workers, force=args.force, clean=args.clean)
