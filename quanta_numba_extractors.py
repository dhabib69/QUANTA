"""
Numba-optimized event extraction functions for QUANTA specialist models.
All magic numbers are loaded from quanta_config.EventExtractionConfig.

Since @njit functions cannot access Python objects at runtime, we read the
config at MODULE LOAD TIME and bind values to module-level constants that
Numba treats as compile-time literals.
"""
import numpy as np
from numba import njit

# ── Load config once at import time ──
from quanta_config import Config as _cfg
_ev = _cfg.events

# ── Shared ──
_MIN_GAP = _ev.min_gap
_LOOKBACK = _ev.lookback_window
_TP_WEIGHT = _ev.tp_weight
_SL_WEIGHT = _ev.sl_weight

# ── Athena (V12: fallback defaults — Pantheon removed) ──
_ATH_CUSUM_MULT  = getattr(_ev, 'athena_cusum_mult',  1.5)
_ATH_CUSUM_FLOOR = getattr(_ev, 'athena_cusum_floor', 0.003)
_ATH_TP_ATR      = getattr(_ev, 'athena_tp_atr',      2.5)
_ATH_SL_ATR      = getattr(_ev, 'athena_sl_atr',      1.0)
_ATH_MAX_BARS    = getattr(_ev, 'athena_max_bars',     72)

# ── Ares (fallback defaults) ──
_ARES_CUSUM_MULT  = getattr(_ev, 'ares_cusum_mult',  1.5)
_ARES_CUSUM_FLOOR = getattr(_ev, 'ares_cusum_floor', 0.003)
_ARES_TP_ATR      = getattr(_ev, 'ares_tp_atr',      2.5)
_ARES_SL_ATR      = getattr(_ev, 'ares_sl_atr',      1.0)
_ARES_MAX_BARS    = getattr(_ev, 'ares_max_bars',     72)

# ── Hermes (fallback defaults) ──
_HER_RANGE_MULT  = getattr(_ev, 'hermes_range_mult',  0.015)
_HER_RANGE_FLOOR = getattr(_ev, 'hermes_range_floor', 0.005)
_HER_BUY_ZONE    = getattr(_ev, 'hermes_buy_zone',    0.2)
_HER_SELL_ZONE   = getattr(_ev, 'hermes_sell_zone',   0.8)
_HER_TP_ATR      = getattr(_ev, 'hermes_tp_atr',      1.5)
_HER_SL_ATR      = getattr(_ev, 'hermes_sl_atr',      1.0)
_HER_MAX_BARS    = getattr(_ev, 'hermes_max_bars',     48)

# ── Artemis (fallback defaults) ──
_ART_CUSUM_MULT  = getattr(_ev, 'artemis_cusum_mult',     1.2)
_ART_CUSUM_FLOOR = getattr(_ev, 'artemis_cusum_floor',    0.003)
_ART_VOL_SURGE   = getattr(_ev, 'artemis_vol_surge_mult', 1.8)
_ART_TP_ATR      = getattr(_ev, 'artemis_tp_atr',         2.0)
_ART_SL_ATR      = getattr(_ev, 'artemis_sl_atr',         1.0)
_ART_MAX_BARS    = getattr(_ev, 'artemis_max_bars',        60)

# ── Chronos (fallback defaults) ──
_CHR_LOOKBACK    = getattr(_ev, 'chronos_lookback',    48)
_CHR_CUSUM_MULT  = getattr(_ev, 'chronos_cusum_mult',  2.0)
_CHR_CUSUM_FLOOR = getattr(_ev, 'chronos_cusum_floor', 0.004)
_CHR_TP_ATR      = getattr(_ev, 'chronos_tp_atr',      2.0)
_CHR_SL_ATR      = getattr(_ev, 'chronos_sl_atr',      1.2)
_CHR_MAX_BARS    = getattr(_ev, 'chronos_max_bars',     72)

# ── Hephaestus (fallback defaults) ──
_HEP_WINDOW      = getattr(_ev, 'heph_window',         96)
_HEP_SUP_PCTL    = getattr(_ev, 'heph_support_pctl',   10.0)
_HEP_RES_PCTL    = getattr(_ev, 'heph_resist_pctl',    90.0)
_HEP_TOL_MULT    = getattr(_ev, 'heph_tolerance_mult', 0.5)
_HEP_TOL_FLOOR   = getattr(_ev, 'heph_tolerance_floor',0.005)
_HEP_TP_ATR      = getattr(_ev, 'heph_tp_atr',         2.0)
_HEP_SL_ATR      = getattr(_ev, 'heph_sl_atr',         1.0)
_HEP_MAX_BARS    = getattr(_ev, 'heph_max_bars',        60)

# ── Thor (active in V12) ──
_THOR_BODY_MIN        = _ev.thor_body_min
_THOR_BODY_RATIO_MULT = _ev.thor_body_ratio_mult
_THOR_BODY_LOOKBACK   = _ev.thor_body_lookback
_THOR_QUIET_BODY_PCT  = _ev.thor_quiet_body_pct / 100.0
_THOR_VOL_MULT        = _ev.thor_vol_mult
_THOR_IMMEDIATE_BODY_RATIO_MULT = _ev.thor_immediate_body_ratio_mult
_THOR_IMMEDIATE_BODY_MIN        = _ev.thor_immediate_body_min
_THOR_IMMEDIATE_VOL_MULT        = _ev.thor_immediate_vol_mult
_THOR_CONTINUATION_VOL_MULT     = _ev.thor_continuation_vol_mult
_THOR_TP_ATR          = _ev.thor_tp_atr
_THOR_SL_ATR          = _ev.thor_sl_atr
_THOR_MAX_BARS        = _ev.thor_max_bars


# ═══════════════════════════════════════════════════════════════════
# Triple Barrier Labeler (López de Prado 2018 Ch.3)
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_triple_barrier_label(closes, highs, lows, atrs, idx, direction, tp_atr_mult, sl_atr_mult, max_bars):
    """
    Direction: 1 for long, -1 for short
    
    Standardized Labels (v11.7):
      1 = BULLISH Outcome (Price went UP or stayed above entry on SL hit shorts)
      0 = BEARISH Outcome (Price went DOWN or stayed below entry on SL hit longs)
    
    Returns: (label, weight)
    """
    N = len(closes)
    if idx + max_bars >= N:
        return -1, 0.0
        
    entry = closes[idx]
    atr = atrs[idx]
    
    if atr <= 0 or entry <= 0:
        return -1, 0.0
        
    if direction == 1:
        tp_price = entry + atr * tp_atr_mult
        sl_price = entry - atr * sl_atr_mult
    else:
        tp_price = entry - atr * tp_atr_mult
        sl_price = entry + atr * sl_atr_mult
        
    for bar in range(1, max_bars + 1):
        high = highs[idx + bar]
        low = lows[idx + bar]
        
        if direction == 1:
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price
            
            if tp_hit and sl_hit:
                close = closes[idx + bar]
                if close >= entry: return 1, _TP_WEIGHT
                else: return 0, _SL_WEIGHT
            if tp_hit: return 1, _TP_WEIGHT
            if sl_hit: return 0, _SL_WEIGHT
        else:
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price
            
            if tp_hit and sl_hit:
                close = closes[idx + bar]
                if close <= entry: return 0, _TP_WEIGHT
                else: return 1, _SL_WEIGHT
            if tp_hit: return 0, _TP_WEIGHT # Success for Short -> 0 (Bearish)
            if sl_hit: return 1, _SL_WEIGHT # Loss for Short -> 1 (Bullish)
            
    return -1, 0.0


# ═══════════════════════════════════════════════════════════════════
# Athena: Breakout Continuation (Long)
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_athena(closes, highs, lows, atrs, atr_pct, cusum_pos, orig_idx):
    N = len(closes)
    out_pos = np.zeros(N, dtype=np.int64)
    out_labels = np.zeros(N, dtype=np.int32)
    out_weights = np.zeros(N, dtype=np.float64)
    count = 0
    
    last_pos = -_MIN_GAP
    
    for i in range(_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_ATH_CUSUM_MULT * atr_pct[i], _ATH_CUSUM_FLOOR)
        
        if cusum_pos[i] > threshold:
            max_high = np.max(highs[i-_LOOKBACK:i])
            if highs[i] >= max_high:
                label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, 1, _ATH_TP_ATR, _ATH_SL_ATR, _ATH_MAX_BARS)
                if label != -1:
                    out_pos[count] = orig_idx[i]
                    out_labels[count] = label
                    out_weights[count] = weight
                    count += 1
                    cusum_pos[i] = 0.0
                    last_pos = i
                    
    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Ares: Short / Downtrend
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_ares(closes, highs, lows, atrs, atr_pct, cusum_neg, orig_idx):
    N = len(closes)
    out_pos = np.zeros(N, dtype=np.int64)
    out_labels = np.zeros(N, dtype=np.int32)
    out_weights = np.zeros(N, dtype=np.float64)
    count = 0
    
    last_pos = -_MIN_GAP
    
    for i in range(_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_ARES_CUSUM_MULT * atr_pct[i], _ARES_CUSUM_FLOOR)
        
        if cusum_neg[i] < -threshold:
            # Removed strict new-low gate — CUSUM negative is sufficient
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, -1, _ARES_TP_ATR, _ARES_SL_ATR, _ARES_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                cusum_neg[i] = 0.0
                last_pos = i
                    
    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Hermes: Range-Bound Mean Reversion
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_hermes(closes, highs, lows, atrs, atr_pct, orig_idx):
    N = len(closes)
    out_pos = np.zeros(N, dtype=np.int64)
    out_labels = np.zeros(N, dtype=np.int32)
    out_weights = np.zeros(N, dtype=np.float64)
    count = 0
    
    last_pos = -_MIN_GAP
    
    for i in range(_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        
        range_high = np.max(highs[i-_LOOKBACK:i])
        range_low = np.min(lows[i-_LOOKBACK:i])
        range_pct = (range_high - range_low) / (closes[i] + 1e-8)
        
        range_threshold = max(_HER_RANGE_MULT * atr_pct[i], _HER_RANGE_FLOOR)
        if range_pct >= range_threshold: continue
        
        total_range = range_high - range_low
        if total_range < 1e-8: continue
        
        pos_in_range = (closes[i] - range_low) / total_range
        
        if pos_in_range < _HER_BUY_ZONE:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, 1, _HER_TP_ATR, _HER_SL_ATR, _HER_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                last_pos = i
        elif pos_in_range > _HER_SELL_ZONE:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, -1, _HER_TP_ATR, _HER_SL_ATR, _HER_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                last_pos = i
                
    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Artemis: Volume Breakout
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_artemis(closes, highs, lows, atrs, atr_pct, cusum_pos, cusum_neg, volumes, vol_avg, orig_idx):
    """Artemis: Stealth Volume Accumulation — CUSUM + volume surge WITHOUT structural break.
    Bidirectional (v11.5b):
      - Bullish: CUSUM_pos + vol_surge + NOT new_high  (hidden accumulation)
      - Bearish: CUSUM_neg + vol_surge + NOT new_low   (hidden distribution)
    Distinct from Athena/Ares (require structural break). Thor requires no CUSUM.
    """
    N = len(closes)
    out_pos    = np.zeros(N * 2, dtype=np.int64)
    out_labels = np.zeros(N * 2, dtype=np.int32)
    out_weights = np.zeros(N * 2, dtype=np.float64)
    count = 0

    # ── Bullish stealth accumulation ──
    last_pos = -_MIN_GAP
    for i in range(_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_ART_CUSUM_MULT * atr_pct[i], _ART_CUSUM_FLOOR)
        if cusum_pos[i] > threshold:
            new_high  = closes[i] > np.max(highs[i-_LOOKBACK:i])  # fixed: was i-1 (off-by-one, excluded most recent bar)
            vol_surge = (volumes[i] > _ART_VOL_SURGE * vol_avg[i]) if vol_avg[i] > 0 else False
            if vol_surge and not new_high:
                label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, 1, _ART_TP_ATR, _ART_SL_ATR, _ART_MAX_BARS)
                if label != -1:
                    out_pos[count]     = orig_idx[i]
                    out_labels[count]  = label
                    out_weights[count] = weight
                    count += 1
                    cusum_pos[i] = 0.0
                    last_pos = i

    # ── Bearish stealth distribution ──
    last_pos = -_MIN_GAP
    for i in range(_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_ART_CUSUM_MULT * atr_pct[i], _ART_CUSUM_FLOOR)
        if cusum_neg[i] < -threshold:
            new_low   = closes[i] < np.min(lows[i-_LOOKBACK:i])  # fixed: was i-1 (off-by-one, excluded most recent bar)
            vol_surge = (volumes[i] > _ART_VOL_SURGE * vol_avg[i]) if vol_avg[i] > 0 else False
            if vol_surge and not new_low:
                label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, -1, _ART_TP_ATR, _ART_SL_ATR, _ART_MAX_BARS)
                if label != -1:
                    out_pos[count]     = orig_idx[i]
                    out_labels[count]  = label
                    out_weights[count] = weight
                    count += 1
                    cusum_neg[i] = 0.0
                    last_pos = i

    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Chronos: Mean Reversion from Extremes
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_chronos(closes, highs, lows, atrs, atr_pct, cusum_pos, cusum_neg, orig_idx):
    N = len(closes)
    out_pos = np.zeros(N * 2, dtype=np.int64)
    out_labels = np.zeros(N * 2, dtype=np.int32)
    out_weights = np.zeros(N * 2, dtype=np.float64)
    count = 0
    
    # Longs (oversold reversal)
    last_pos = -_MIN_GAP
    for i in range(_CHR_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_CHR_CUSUM_MULT * atr_pct[i], _CHR_CUSUM_FLOOR)
        if cusum_neg[i] < -threshold:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, 1, _CHR_TP_ATR, _CHR_SL_ATR, _CHR_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                cusum_neg[i] = 0.0
                last_pos = i
                
    # Shorts (overbought reversal)
    last_pos = -_MIN_GAP
    for i in range(_CHR_LOOKBACK, N):
        if i - last_pos < _MIN_GAP: continue
        threshold = max(_CHR_CUSUM_MULT * atr_pct[i], _CHR_CUSUM_FLOOR)
        if cusum_pos[i] > threshold:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, -1, _CHR_TP_ATR, _CHR_SL_ATR, _CHR_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                cusum_pos[i] = 0.0
                last_pos = i
                
    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Utility: Numba-safe percentile (Numba lacks np.percentile)
# ═══════════════════════════════════════════════════════════════════

@njit
def numba_percentile(arr, q):
    sorted_arr = np.sort(arr)
    idx = (len(arr) - 1) * (q / 100.0)
    lower = int(np.floor(idx))
    upper = int(np.ceil(idx))
    if lower == upper:
        return sorted_arr[lower]
    weight = idx - lower
    return sorted_arr[lower] * (1.0 - weight) + sorted_arr[upper] * weight


# ═══════════════════════════════════════════════════════════════════
# Hephaestus: Support/Resistance Bounce
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_hephaestus(closes, highs, lows, atrs, atr_pct, orig_idx):
    N = len(closes)
    out_pos = np.zeros(N * 2, dtype=np.int64)
    out_labels = np.zeros(N * 2, dtype=np.int32)
    out_weights = np.zeros(N * 2, dtype=np.float64)
    count = 0
    
    last_pos = -_MIN_GAP
    
    for i in range(_HEP_WINDOW, N):
        if i - last_pos < _MIN_GAP: continue
        
        support = numba_percentile(lows[i-_HEP_WINDOW:i], _HEP_SUP_PCTL)
        resistance = numba_percentile(highs[i-_HEP_WINDOW:i], _HEP_RES_PCTL)
        
        price = closes[i]
        tolerance = max(_HEP_TOL_MULT * atr_pct[i], _HEP_TOL_FLOOR)
        
        if abs(price - support) / (support + 1e-8) < tolerance:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, 1, _HEP_TP_ATR, _HEP_SL_ATR, _HEP_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                last_pos = i
        elif abs(price - resistance) / (resistance + 1e-8) < tolerance:
            label, weight = fast_triple_barrier_label(closes, highs, lows, atrs, i, -1, _HEP_TP_ATR, _HEP_SL_ATR, _HEP_MAX_BARS)
            if label != -1:
                out_pos[count] = orig_idx[i]
                out_labels[count] = label
                out_weights[count] = weight
                count += 1
                last_pos = i
                
    return out_pos[:count], out_labels[:count], out_weights[:count]


# ═══════════════════════════════════════════════════════════════════
# Apollo: Macro Momentum Shift
# ═══════════════════════════════════════════════════════════════════

@njit
def fast_extract_thor(closes, highs, lows, opens, atrs, volumes, vol_avg20, orig_idx):
    """Thor: Tiered breakout detector (v12.0).

    Tier A:
      - same-bar entry when the setup candle is already extreme
    Tier B:
      - one-bar-later entry when the next candle confirms the setup
    Tier C:
      - two-bar continuation after the setup if follow-through persists and
        the entry bar volume does not collapse
    """
    N = len(closes)
    out_pos    = np.zeros(N, dtype=np.int64)
    out_labels = np.zeros(N, dtype=np.int32)
    out_weights= np.zeros(N, dtype=np.float64)
    count      = 0
    last_pos   = -_MIN_GAP
    lookback   = _THOR_BODY_LOOKBACK

    for i in range(lookback + 1, N):
        if i - last_pos < _MIN_GAP:
            continue

        body_i = closes[i] - opens[i]
        avg_body = 0.0
        for j in range(i - lookback, i):
            avg_body += abs(closes[j] - opens[j])
        avg_body /= lookback
        candle_range = highs[i] - lows[i]
        body_eff = body_i / candle_range if candle_range > 0.0 else 0.0
        body_ratio = body_i / avg_body if avg_body > 0.0 else 0.0
        quiet_ok = opens[i] > 0.0 and avg_body / opens[i] <= _THOR_QUIET_BODY_PCT
        vol_ratio = volumes[i] / vol_avg20[i] if vol_avg20[i] > 0.0 else 0.0

        setup_ok = (
            body_i > 0.0 and
            body_i / (opens[i] + 1e-12) >= 0.01 and
            avg_body > 0.0 and
            body_eff >= _THOR_BODY_MIN and
            body_ratio >= _THOR_BODY_RATIO_MULT and
            quiet_ok and
            vol_ratio >= _THOR_VOL_MULT
        )

        if setup_ok:
            immediate_ok = (
                body_eff >= _THOR_IMMEDIATE_BODY_MIN and
                body_ratio >= _THOR_IMMEDIATE_BODY_RATIO_MULT and
                vol_ratio >= _THOR_IMMEDIATE_VOL_MULT
            )
            if immediate_ok:
                label, weight = fast_triple_barrier_label(
                    closes, highs, lows, atrs, i,
                    1, _THOR_TP_ATR, _THOR_SL_ATR, _THOR_MAX_BARS
                )
                if label != -1:
                    out_pos[count]    = orig_idx[i]
                    out_labels[count] = label
                    out_weights[count]= weight
                    count    += 1
                    last_pos  = i
                    continue

        # Tier B: one-bar-later confirmation from setup_idx = i - 1.
        setup_idx = i - 1
        if setup_idx >= lookback:
            setup_body = closes[setup_idx] - opens[setup_idx]
            setup_range = highs[setup_idx] - lows[setup_idx]
            setup_eff = setup_body / setup_range if setup_range > 0.0 else 0.0
            setup_avg_body = 0.0
            for j in range(setup_idx - lookback, setup_idx):
                setup_avg_body += abs(closes[j] - opens[j])
            setup_avg_body /= lookback
            setup_ratio = setup_body / setup_avg_body if setup_avg_body > 0.0 else 0.0
            setup_quiet_ok = opens[setup_idx] > 0.0 and setup_avg_body / opens[setup_idx] <= _THOR_QUIET_BODY_PCT
            setup_vol_ratio = volumes[setup_idx] / vol_avg20[setup_idx] if vol_avg20[setup_idx] > 0.0 else 0.0

            prev_setup_ok = (
                setup_body > 0.0 and
                setup_body / (opens[setup_idx] + 1e-12) >= 0.01 and
                setup_avg_body > 0.0 and
                setup_eff >= _THOR_BODY_MIN and
                setup_ratio >= _THOR_BODY_RATIO_MULT and
                setup_quiet_ok and
                setup_vol_ratio >= _THOR_VOL_MULT
            )
            if prev_setup_ok:
                setup_mid = 0.5 * (opens[setup_idx] + closes[setup_idx])
                confirm_ok = (
                    lows[i] >= setup_mid and
                    closes[i] >= closes[setup_idx] and
                    highs[i] >= highs[setup_idx]
                )
                if confirm_ok:
                    label, weight = fast_triple_barrier_label(
                        closes, highs, lows, atrs, i,
                        1, _THOR_TP_ATR, _THOR_SL_ATR, _THOR_MAX_BARS
                    )
                    if label != -1:
                        out_pos[count]    = orig_idx[i]
                        out_labels[count] = label
                        out_weights[count]= weight
                        count    += 1
                        last_pos  = i
                        continue

        # Tier C: two-bar continuation from setup_idx = i - 2.
        setup_idx = i - 2
        confirm1_idx = i - 1
        if setup_idx < lookback:
            continue

        setup_body = closes[setup_idx] - opens[setup_idx]
        setup_range = highs[setup_idx] - lows[setup_idx]
        setup_eff = setup_body / setup_range if setup_range > 0.0 else 0.0
        setup_avg_body = 0.0
        for j in range(setup_idx - lookback, setup_idx):
            setup_avg_body += abs(closes[j] - opens[j])
        setup_avg_body /= lookback
        setup_ratio = setup_body / setup_avg_body if setup_avg_body > 0.0 else 0.0
        setup_quiet_ok = opens[setup_idx] > 0.0 and setup_avg_body / opens[setup_idx] <= _THOR_QUIET_BODY_PCT
        setup_vol_ratio = volumes[setup_idx] / vol_avg20[setup_idx] if vol_avg20[setup_idx] > 0.0 else 0.0

        prev2_setup_ok = (
            setup_body > 0.0 and
            setup_body / (opens[setup_idx] + 1e-12) >= 0.01 and
            setup_avg_body > 0.0 and
            setup_eff >= _THOR_BODY_MIN and
            setup_ratio >= _THOR_BODY_RATIO_MULT and
            setup_quiet_ok and
            setup_vol_ratio >= _THOR_VOL_MULT
        )
        if not prev2_setup_ok:
            continue

        setup_mid = 0.5 * (opens[setup_idx] + closes[setup_idx])
        entry_vol_ratio = volumes[i] / vol_avg20[i] if vol_avg20[i] > 0.0 else 0.0
        continuation_ok = (
            lows[confirm1_idx] >= setup_mid and
            closes[i] >= closes[setup_idx] and
            highs[i] >= highs[setup_idx] and
            entry_vol_ratio >= _THOR_CONTINUATION_VOL_MULT
        )
        if not continuation_ok:
            continue

        label, weight = fast_triple_barrier_label(
            closes, highs, lows, atrs, i,
            1, _THOR_TP_ATR, _THOR_SL_ATR, _THOR_MAX_BARS
        )
        if label != -1:
            out_pos[count]    = orig_idx[i]
            out_labels[count] = label
            out_weights[count]= weight
            count    += 1
            last_pos  = i

    return out_pos[:count], out_labels[:count], out_weights[:count]

# Backward-compatibility alias
fast_extract_nike = fast_extract_thor
