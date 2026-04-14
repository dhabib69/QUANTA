"""
QUANTA Liquidation Cascade Engine (2026-04-12)
================================================
Identifies short-squeeze liquidation cluster zones above entry price
to extend TP targets and boost position size when a cascade is imminent.

Theory:
    When many traders hold leveraged SHORT positions, they all have a
    liquidation price above entry. If price reaches that cluster, ALL
    those shorts are force-BOUGHT simultaneously — amplifying the pump.
    
    This is the mechanism behind crypto's infamous "short squeezes":
    - Funding goes deeply negative (shorts paying longs)
    - Open interest is elevated relative to volume
    - A catalyst (like a Nike/Thor setup) triggers the squeeze cascade
    
References:
    - Makarov & Schoar (2020) "Trading and Arbitrage in Crypto Markets"
    - Brunnermeier & Pedersen (2009) "Market Liquidity and Funding Liquidity"
    - Binance open interest + funding rate endpoints (live data)
"""

import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class LiquidationCascadeEngine:
    """
    Estimates liquidation cascade potential for a symbol at a given price.
    
    Usage:
        engine = LiquidationCascadeEngine(exchange_api)
        result = engine.get_cascade_signal(symbol, entry_price, atr_pct)
        if result['cascade_potential']:
            use result['target_price'] as TP
            apply result['size_boost'] to position notional
    """

    def __init__(self, exchange_api=None, config=None):
        self._api = exchange_api
        self._cfg = config
        # Cache OI/funding data: symbol -> {ts, oi, funding, oi_prev}
        self._cache: dict = {}
        self._cache_ttl = 120.0  # 2 minutes
        # Rolling OI history per symbol for surge detection
        self._oi_history: dict = {}  # symbol -> deque(maxlen=4) of (ts, oi) tuples

        # Config defaults (overridden from EventExtractionConfig if available)
        if config is not None:
            ev = getattr(config, 'events', None)
            self.oi_surge_threshold   = getattr(ev, 'liq_oi_surge_threshold',   0.12)
            self.funding_extreme      = getattr(ev, 'liq_funding_extreme',      -0.025)
            self.cascade_size_boost   = getattr(ev, 'liq_cascade_size_boost',    1.25)
            self.max_target_dist_pct  = getattr(ev, 'liq_max_target_dist_pct',  15.0)
            self.min_oi_vol_ratio     = getattr(ev, 'liq_min_oi_vol_ratio',      1.5)
            self.enabled              = getattr(ev, 'liq_enabled',               True)
        else:
            self.oi_surge_threshold  = 0.12
            self.funding_extreme     = -0.025
            self.cascade_size_boost  = 1.25
            self.max_target_dist_pct = 15.0
            self.min_oi_vol_ratio    = 1.5
            self.enabled             = True

    # ── Public API ──────────────────────────────────────────────────────────

    def get_cascade_signal(
        self,
        symbol: str,
        entry_price: float,
        atr_pct: float,
        volume_24h: float = 0.0,
    ) -> dict:
        """
        Returns a dict describing the liquidation cascade potential for this entry.

        Keys:
            cascade_potential (bool)  — True if conditions favour a squeeze
            target_price (float)      — Estimated cascade target (entry_price if no signal)
            target_dist_pct (float)   — Distance to target as % of entry_price
            size_boost (float)        — Recommended notional multiplier (1.0 if no signal)
            funding_rate (float)      — Last funding rate (%)
            oi_surge_pct (float)      — OI change over last 4 readings (%)
            reason (str)              — Human-readable explanation
        """
        base = {
            "cascade_potential": False,
            "target_price":      entry_price,
            "target_dist_pct":   0.0,
            "size_boost":        1.0,
            "funding_rate":      0.0,
            "oi_surge_pct":      0.0,
            "reason":            "no_data",
        }

        if not self.enabled or self._api is None:
            base["reason"] = "disabled_or_no_api"
            return base

        try:
            funding, oi = self._fetch_funding_and_oi(symbol)
        except Exception as exc:
            logger.debug("[LiqEng] fetch failed for %s: %s", symbol, exc)
            base["reason"] = f"fetch_error: {exc}"
            return base

        base["funding_rate"] = funding

        # ── OI surge detection ───────────────────────────────────────────
        oi_surge_pct = self._compute_oi_surge(symbol, oi)
        base["oi_surge_pct"] = oi_surge_pct

        # ── OI/volume ratio ─────────────────────────────────────────────
        oi_vol_ratio = (oi / volume_24h) if volume_24h > 0 else 0.0

        # ── Score conditions ─────────────────────────────────────────────
        # Each condition contributes independently.  We need at least 2 of 3
        # to confirm a cascade setup (avoids false positives).
        conditions_met = 0
        reasons = []

        if funding <= self.funding_extreme:
            conditions_met += 1
            reasons.append(f"funding={funding:.4f}%<{self.funding_extreme}")

        if oi_surge_pct >= self.oi_surge_threshold * 100:
            conditions_met += 1
            reasons.append(f"oi_surge={oi_surge_pct:.1f}%>{self.oi_surge_threshold*100:.0f}%")

        if oi_vol_ratio >= self.min_oi_vol_ratio:
            conditions_met += 1
            reasons.append(f"oi_vol_ratio={oi_vol_ratio:.2f}>{self.min_oi_vol_ratio}")

        if conditions_met < 2:
            base["reason"] = "insufficient_signal(" + ",".join(reasons) + ")"
            return base

        # ── Estimate cascade target ──────────────────────────────────────
        # Conservative estimate: 3 ATR above entry (roughly where clustered stops sit)
        # Extended estimate: up to max_target_dist_pct
        #
        # Empirical: when funding < -0.03%, typical squeeze moves 8-12% per Makarov 2020
        # We cap at max_target_dist_pct for safety
        base_target_pct = min(
            3.0 * atr_pct,                  # conservative ATR-based
            self.max_target_dist_pct / 2    # half the max distance
        )
        # Boost target distance when funding is extremely negative
        funding_severity = max(0.0, (abs(funding) - abs(self.funding_extreme)) / abs(self.funding_extreme))
        cascade_target_pct = min(
            base_target_pct * (1 + funding_severity * 0.5),
            self.max_target_dist_pct
        )

        target_price = entry_price * (1.0 + cascade_target_pct / 100.0)
        size_mult = min(self.cascade_size_boost, 1.0 + (conditions_met - 1) * 0.15)

        base.update({
            "cascade_potential": True,
            "target_price":      round(target_price, 10),
            "target_dist_pct":   round(cascade_target_pct, 2),
            "size_boost":        round(size_mult, 3),
            "reason":            "cascade(cond=%d): %s" % (conditions_met, "; ".join(reasons)),
        })
        logger.info(
            "[LiqEng] %s cascade signal: target=%.4f (+%.1f%%) boost=%.2fx | %s",
            symbol, target_price, cascade_target_pct, size_mult, base["reason"]
        )
        return base

    # ── Private helpers ──────────────────────────────────────────────────

    def _fetch_funding_and_oi(self, symbol: str):
        """Return (funding_rate_pct, open_interest_usd). Cached for _cache_ttl seconds."""
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < self._cache_ttl:
            return cached["funding"], cached["oi"]

        funding = 0.0
        oi = 0.0

        try:
            fr = self._api.get_funding_rate(symbol)
            if fr is not None:
                funding = float(fr)
        except Exception:
            pass

        try:
            oi_raw = self._api.get_open_interest(symbol)
            if oi_raw is not None:
                oi = float(oi_raw)
        except Exception:
            pass

        self._cache[symbol] = {"ts": time.time(), "funding": funding, "oi": oi}
        return funding, oi

    def _compute_oi_surge(self, symbol: str, current_oi: float) -> float:
        """
        Track rolling OI history and compute % surge over last 4 readings.
        Returns 0.0 if fewer than 2 readings.
        """
        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=4)

        history = self._oi_history[symbol]
        history.append((time.time(), current_oi))

        if len(history) < 2 or history[0][1] <= 0:
            return 0.0

        oldest_oi = history[0][1]
        surge_pct = (current_oi - oldest_oi) / oldest_oi * 100.0
        return surge_pct

    def inject_oi_snapshot(self, symbol: str, oi_value: float):
        """Allow the bot's existing OI polling to feed the engine directly."""
        if symbol not in self._oi_history:
            self._oi_history[symbol] = deque(maxlen=4)
        self._oi_history[symbol].append((time.time(), float(oi_value)))

    def update_cache(self, symbol: str, funding: float, oi: float):
        """Manual cache update — called by bot if it fetches funding/OI elsewhere."""
        self._cache[symbol] = {"ts": time.time(), "funding": funding, "oi": oi}
        self.inject_oi_snapshot(symbol, oi)


# ── Offline / simulation helper ──────────────────────────────────────────────

def estimate_cascade_boost_offline(
    funding_rate: float,
    oi_surge_pct: float,
    oi_vol_ratio: float,
    atr_pct: float,
    cfg=None,
) -> dict:
    """
    Pure-Python, no-API version for use in the Norse year simulator.
    Takes pre-computed funding, OI surge, OI/vol ratio and outputs the same
    dict structure as LiquidationCascadeEngine.get_cascade_signal().
    
    In the simulator these values come from the feature vector
    (indices 212=funding_rate, 213=open_interest stored in quanta_features).
    """
    ev = getattr(cfg, 'events', None) if cfg else None
    oi_surge_threshold  = getattr(ev, 'liq_oi_surge_threshold',   0.12) if ev else 0.12
    funding_extreme     = getattr(ev, 'liq_funding_extreme',      -0.025) if ev else -0.025
    cascade_size_boost  = getattr(ev, 'liq_cascade_size_boost',    1.25) if ev else 1.25
    max_target_dist_pct = getattr(ev, 'liq_max_target_dist_pct',  15.0) if ev else 15.0
    min_oi_vol_ratio    = getattr(ev, 'liq_min_oi_vol_ratio',      1.5) if ev else 1.5

    base = {
        "cascade_potential": False,
        "target_dist_pct":   0.0,
        "size_boost":        1.0,
        "reason":            "offline",
    }

    conditions_met = 0
    if funding_rate <= funding_extreme:
        conditions_met += 1
    if oi_surge_pct >= oi_surge_threshold * 100:
        conditions_met += 1
    if oi_vol_ratio >= min_oi_vol_ratio:
        conditions_met += 1

    if conditions_met < 2:
        return base

    base_target_pct = min(3.0 * atr_pct, max_target_dist_pct / 2)
    severity = max(0.0, (abs(funding_rate) - abs(funding_extreme)) / max(abs(funding_extreme), 1e-9))
    cascade_target_pct = min(base_target_pct * (1 + severity * 0.5), max_target_dist_pct)
    size_mult = min(cascade_size_boost, 1.0 + (conditions_met - 1) * 0.15)

    base.update({
        "cascade_potential": True,
        "target_dist_pct":   round(cascade_target_pct, 2),
        "size_boost":        round(size_mult, 3),
        "reason":            f"cascade(cond={conditions_met})",
    })
    return base
