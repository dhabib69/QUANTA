"""
QUANTA Nike Screener
====================
Nike is NOT a per-coin agent. It is a market-wide screener.

Architecture:
  - Subscribes to ALL Binance Futures symbols on 5m klines (own WS connections)
  - Maintains a lightweight 30-bar rolling buffer per symbol (no shared CandleStore)
  - On every closed 5m bar → runs the Nike check inline
  - When Nike fires → pushes a NikeSignal onto an output queue
  - QUANTA_bot consumes the queue and enters the trade directly

Why separate from the main prediction pipeline:
  - Nike doesn't need features, ML, or multi-timeframe analysis
  - It watches ALL 542 symbols, not just the active_coins watchlist
  - Its signal IS the entry — no second-pass ML needed
  - Keeps the main CandleStore clean (no lock contention from 542 extra symbols)

Signal flow:
  NikeScreener.signal_queue  →  Bot._nike_signal_consumer  →  opportunity dict  →  rl_opportunities
"""

import asyncio
import json
import logging
import math
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
try:
    import sys, os
    _old = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    from quanta_config import Config as _cfg
    sys.stdout = _old
    _ev = _cfg.events
    BODY_RATIO_MULT = _ev.nike_body_ratio_mult    # 15.0
    QUIET_BODY_PCT  = _ev.nike_quiet_body_pct     # 0.5  (as %)
    VOL_MULT        = _ev.nike_vol_mult            # 2.0
    BODY_MIN        = _ev.nike_body_min            # 0.6
    IMMEDIATE_BODY_RATIO_MULT = _ev.nike_immediate_body_ratio_mult
    IMMEDIATE_BODY_MIN        = _ev.nike_immediate_body_min
    IMMEDIATE_VOL_MULT        = _ev.nike_immediate_vol_mult
    CONTINUATION_VOL_MULT     = _ev.nike_continuation_vol_mult
    SCORE_BODY_RATIO_WEIGHT   = _ev.nike_score_body_ratio_weight
    SCORE_VOL_RATIO_WEIGHT    = _ev.nike_score_vol_ratio_weight
    SCORE_BODY_EFF_WEIGHT     = _ev.nike_score_body_eff_weight
    SCORE_QUIET_WEIGHT        = _ev.nike_score_quiet_weight
    SCORE_CONFIRM_WEIGHT      = _ev.nike_score_confirm_weight
    TIER_A_CONFIDENCE         = _ev.nike_tier_a_confidence
    TIER_B_CONFIDENCE         = _ev.nike_tier_b_confidence
    TIER_C_CONFIDENCE         = _ev.nike_tier_c_confidence
    TIER_A_SIZE_MULT          = _ev.nike_tier_a_size_mult
    TIER_B_SIZE_MULT          = _ev.nike_tier_b_size_mult
    TIER_C_SIZE_MULT          = _ev.nike_tier_c_size_mult
    TIER_B_BS_FLOOR           = _ev.nike_tier_b_bs_floor
    TIER_C_BS_FLOOR           = _ev.nike_tier_c_bs_floor
    TP_ATR          = _ev.nike_tp_atr              # 2.0
    SL_ATR          = _ev.nike_sl_atr              # 0.8
    MAX_BARS        = _ev.nike_max_bars            # 12
    BANK_ATR        = _ev.nike_bank_atr
    BANK_FRACTION   = _ev.nike_bank_fraction
    RUNNER_TRAIL_ATR= _ev.nike_runner_trail_atr
    MAX_BARS_PRE    = _ev.nike_max_bars_pre_bank
    MAX_BARS_POST   = _ev.nike_max_bars_post_bank
except Exception:
    sys.stdout = sys.__stdout__
    BODY_RATIO_MULT = 5.0
    QUIET_BODY_PCT  = 0.5
    VOL_MULT        = 1.5
    BODY_MIN        = 0.4
    IMMEDIATE_BODY_RATIO_MULT = 8.0
    IMMEDIATE_BODY_MIN        = 0.55
    IMMEDIATE_VOL_MULT        = 2.0
    CONTINUATION_VOL_MULT     = 1.0
    SCORE_BODY_RATIO_WEIGHT   = 0.30
    SCORE_VOL_RATIO_WEIGHT    = 0.25
    SCORE_BODY_EFF_WEIGHT     = 0.20
    SCORE_QUIET_WEIGHT        = 0.10
    SCORE_CONFIRM_WEIGHT      = 0.15
    TIER_A_CONFIDENCE         = 84.0
    TIER_B_CONFIDENCE         = 78.0
    TIER_C_CONFIDENCE         = 72.0
    TIER_A_SIZE_MULT          = 1.15
    TIER_B_SIZE_MULT          = 1.00
    TIER_C_SIZE_MULT          = 0.75
    TIER_B_BS_FLOOR           = 0.30
    TIER_C_BS_FLOOR           = 0.35
    TP_ATR          = 2.0
    SL_ATR          = 0.8
    MAX_BARS        = 24
    BANK_ATR        = 2.0
    BANK_FRACTION   = 0.50
    RUNNER_TRAIL_ATR= 1.5
    MAX_BARS_PRE    = 24
    MAX_BARS_POST   = 36

# ── Internal constants ────────────────────────────────────────────────────────
LOOKBACK         = 20          # bars for avg-body and vol-avg calculation
BUFFER_SIZE      = 80          # enough for setup, continuation checks, and lightweight local history
COOLDOWN_BARS    = 48          # ~4h: suppress re-fires on same coin after a signal
ATR_PERIOD       = 14
WS_BASE          = "wss://fstream.binance.com/stream?streams="
MAX_STREAMS      = 180         # streams per WS connection (Binance limit = 200)
RECONNECT_DELAY  = 3           # seconds before reconnect on drop


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class NikeSignal:
    symbol:        str
    timestamp:     float          # Unix timestamp of the trigger bar
    date_str:      str            # human-readable UTC
    close:         float          # close price of trigger bar
    atr:           float          # ATR at trigger bar (for TP/SL calc)
    tp_price:      float
    sl_price:      float
    body_pct:      float          # anomalous candle body %
    body_ratio:    float          # how many × avg prior body
    avg_prior_pct: float          # prior avg body % of price (quietness measure)
    vol_ratio:     float          # volume spike vs 20-bar avg
    body_eff:      float          # body efficiency (0–1)
    tier:          str            # A / B / C
    entry_mode:    str            # immediate / confirm / continuation
    score:         float          # deterministic Nike quality score (0-100)
    confidence:    float          # mapped live confidence used by bot
    size_mult:     float          # tier-aware sizing multiplier
    bs_floor:      float          # tier-specific BS/Kou veto floor above baseline


# ── Lightweight per-symbol buffer ─────────────────────────────────────────────

class _SymbolBuffer:
    """Minimal 30-bar OHLCV + ATR state for one symbol."""

    __slots__ = ('o', 'h', 'l', 'c', 'v', 'atr', '_atr_initialized',
                 'last_signal_bar', 'bar_count')

    def __init__(self):
        self.o   = deque(maxlen=BUFFER_SIZE)
        self.h   = deque(maxlen=BUFFER_SIZE)
        self.l   = deque(maxlen=BUFFER_SIZE)
        self.c   = deque(maxlen=BUFFER_SIZE)
        self.v   = deque(maxlen=BUFFER_SIZE)
        self.atr = 0.0
        self._atr_initialized = False
        self.last_signal_bar  = -COOLDOWN_BARS   # absolute bar counter, not deque index
        self.bar_count        = 0

    def push(self, o: float, h: float, l: float, c: float, v: float) -> None:
        prev_c = self.c[-1] if self.c else c
        self.o.append(o); self.h.append(h); self.l.append(l)
        self.c.append(c); self.v.append(v)
        self.bar_count += 1
        # Wilder ATR update
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        if not self._atr_initialized:
            if len(self.c) >= ATR_PERIOD:
                # Seed: simple mean of first ATR_PERIOD TRs
                trs = []
                cs = list(self.c)
                hs = list(self.h)
                ls = list(self.l)
                for i in range(1, ATR_PERIOD + 1):
                    trs.append(max(
                        hs[-i] - ls[-i],
                        abs(hs[-i] - cs[-(i+1)]) if len(cs) > i else hs[-i]-ls[-i],
                        abs(ls[-i] - cs[-(i+1)]) if len(cs) > i else 0,
                    ))
                self.atr = float(np.mean(trs))
                self._atr_initialized = True
        else:
            self.atr = (self.atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD

    def ready(self) -> bool:
        return len(self.c) >= LOOKBACK + 2


# ── Nike check (pure Python, inline) ─────────────────────────────────────────

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _confirm_strength(entry_close: float, entry_high: float, entry_low: float, setup: dict) -> float:
    hold_span = max(setup['close'] - setup['mid'], 1e-12)
    break_span = max(setup['high'] - setup['mid'], 1e-12)
    hold = _clip01((entry_low - setup['mid']) / hold_span)
    close_push = _clip01((entry_close - setup['close']) / hold_span)
    breakout = _clip01((entry_high - setup['high']) / break_span)
    return (hold + close_push + breakout) / 3.0


def _nike_score(meta: dict, confirm_strength: float) -> float:
    body_ratio_norm = _clip01((meta['body_ratio'] - BODY_RATIO_MULT) / max(4.0, IMMEDIATE_BODY_RATIO_MULT - BODY_RATIO_MULT))
    vol_ratio_norm = _clip01((meta['vol_ratio'] - VOL_MULT) / max(2.0, IMMEDIATE_VOL_MULT - VOL_MULT + 1.5))
    body_eff_norm = _clip01((meta['body_eff'] - BODY_MIN) / max(0.20, IMMEDIATE_BODY_MIN - BODY_MIN + 0.05))
    quiet_norm = _clip01((QUIET_BODY_PCT - meta['avg_prior_pct']) / max(QUIET_BODY_PCT, 1e-12))

    score = 100.0 * (
        SCORE_BODY_RATIO_WEIGHT * body_ratio_norm +
        SCORE_VOL_RATIO_WEIGHT * vol_ratio_norm +
        SCORE_BODY_EFF_WEIGHT * body_eff_norm +
        SCORE_QUIET_WEIGHT * quiet_norm +
        SCORE_CONFIRM_WEIGHT * _clip01(confirm_strength)
    )
    return round(score, 2)


def _tier_payload(tier: str, entry_mode: str, close: float, atr: float, meta: dict, confirm_strength: float) -> dict:
    if tier == 'A':
        confidence = TIER_A_CONFIDENCE
        size_mult = TIER_A_SIZE_MULT
        bs_floor = 0.0
    elif tier == 'B':
        confidence = TIER_B_CONFIDENCE
        size_mult = TIER_B_SIZE_MULT
        bs_floor = TIER_B_BS_FLOOR
    else:
        confidence = TIER_C_CONFIDENCE
        size_mult = TIER_C_SIZE_MULT
        bs_floor = TIER_C_BS_FLOOR

    return {
        'close': close,
        'atr': atr,
        'body_pct': meta['body_pct'],
        'body_ratio': meta['body_ratio'],
        'avg_prior_pct': meta['avg_prior_pct'],
        'vol_ratio': meta['vol_ratio'],
        'body_eff': meta['body_eff'],
        'tier': tier,
        'entry_mode': entry_mode,
        'score': _nike_score(meta, confirm_strength),
        'confidence': confidence,
        'size_mult': size_mult,
        'bs_floor': bs_floor,
    }


def _setup_metrics(c, o, h, l, v, idx: int) -> Optional[dict]:
    if idx < LOOKBACK:
        return None

    body_i = c[idx] - o[idx]
    if body_i <= 0.0 or body_i / max(o[idx], 1e-12) < 0.01:
        return None

    rng = h[idx] - l[idx]
    if rng <= 0.0:
        return None
    body_eff = body_i / rng
    if body_eff < BODY_MIN:
        return None

    prior_bodies = [abs(c[j] - o[j]) for j in range(idx - LOOKBACK, idx)]
    avg_body = sum(prior_bodies) / LOOKBACK
    if avg_body <= 0.0:
        return None

    body_ratio = body_i / avg_body
    if body_ratio < BODY_RATIO_MULT:
        return None

    avg_body_pct = (avg_body / max(o[idx], 1e-12)) * 100.0
    if avg_body_pct >= QUIET_BODY_PCT:
        return None

    vol_window = v[idx - 20:idx]
    if not vol_window:
        return None
    vol_avg20 = sum(vol_window) / len(vol_window)
    if vol_avg20 <= 0.0:
        return None
    vol_ratio = v[idx] / vol_avg20
    if vol_ratio < VOL_MULT:
        return None

    return {
        'close': c[idx],
        'range': rng,
        'body_pct': body_i / max(o[idx], 1e-12) * 100.0,
        'body_ratio': body_ratio,
        'avg_prior_pct': avg_body_pct,
        'vol_ratio': vol_ratio,
        'body_eff': body_eff,
        'immediate_ok': (
            body_eff >= IMMEDIATE_BODY_MIN and
            body_ratio >= IMMEDIATE_BODY_RATIO_MULT and
            vol_ratio >= IMMEDIATE_VOL_MULT
        ),
        'high': h[idx],
        'mid': 0.5 * (o[idx] + c[idx]),
    }

def _nike_check(buf: _SymbolBuffer) -> Optional[dict]:
    """
    Run the Nike trigger on the most-recently closed bar.
    Returns a dict of signal metadata or None.
    """
    if not buf.ready():
        return None

    i = len(buf.c) - 1   # index of last bar in the deque
    current_bar = buf.bar_count - 1

    # Cooldown guard
    if current_bar - buf.last_signal_bar < COOLDOWN_BARS:
        return None

    c = list(buf.c)
    o = list(buf.o)
    h = list(buf.h)
    l = list(buf.l)
    v = list(buf.v)

    current_setup = _setup_metrics(c, o, h, l, v, i)
    if current_setup and current_setup['immediate_ok']:
        buf.last_signal_bar = current_bar
        return _tier_payload('A', 'immediate', c[i], buf.atr, current_setup, 1.0)

    setup_idx = i - 1
    prev_setup = _setup_metrics(c, o, h, l, v, setup_idx) if setup_idx >= LOOKBACK else None
    if prev_setup is not None:
        confirm_ok = (
            l[i] >= prev_setup['mid'] and
            c[i] >= prev_setup['close'] and
            h[i] >= prev_setup['high']
        )
        if confirm_ok:
            buf.last_signal_bar = current_bar
            strength = _confirm_strength(c[i], h[i], l[i], prev_setup)
            return _tier_payload('B', 'confirm', c[i], buf.atr, prev_setup, strength)

    setup_idx = i - 2
    follow_idx = i - 1
    prev2_setup = _setup_metrics(c, o, h, l, v, setup_idx) if setup_idx >= LOOKBACK else None
    if prev2_setup is None:
        return None

    continuation_ok = (
        l[follow_idx] >= prev2_setup['mid'] and
        c[i] >= prev2_setup['close'] and
        h[i] >= prev2_setup['high']
    )
    if not continuation_ok:
        return None

    vol_window = v[i - 20:i]
    if not vol_window:
        return None
    vol_avg20 = sum(vol_window) / len(vol_window)
    entry_vol_ratio = v[i] / vol_avg20 if vol_avg20 > 0 else 0.0
    if entry_vol_ratio < CONTINUATION_VOL_MULT:
        return None

    buf.last_signal_bar = current_bar
    first_hold = _clip01((l[follow_idx] - prev2_setup['mid']) / max(prev2_setup['close'] - prev2_setup['mid'], 1e-12))
    strength = 0.5 * first_hold + 0.5 * _confirm_strength(c[i], h[i], l[i], prev2_setup)
    return _tier_payload('C', 'continuation', c[i], buf.atr, prev2_setup, strength)


# ── NikeScreener ─────────────────────────────────────────────────────────────

class NikeScreener:
    """
    Market-wide Nike screener.

    Usage:
        screener = NikeScreener(symbols=all_symbols, on_signal=callback)
        screener.start()            # non-blocking
        sig = screener.signal_queue.get()   # blocks until first signal
        screener.stop()
    """

    def __init__(
        self,
        symbols: List[str],
        on_signal: Optional[Callable[[NikeSignal], None]] = None,
        proxy_url: Optional[str] = None,
    ):
        self.symbols      = list(symbols)
        self.on_signal    = on_signal           # optional callback; queue always filled
        self.proxy_url    = proxy_url
        self.signal_queue: queue.Queue[NikeSignal] = queue.Queue(maxsize=500)

        self._buffers: Dict[str, _SymbolBuffer] = defaultdict(_SymbolBuffer)
        self._stop    = threading.Event()
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._signals_total = 0
        self._bars_total    = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the screener in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="NikeScreener", daemon=True
        )
        self._thread.start()
        logging.info(f"ThorScreener started — watching {len(self.symbols)} symbols on 5m")

    def stop(self) -> None:
        self._stop.set()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def stats(self) -> dict:
        return {
            'symbols':       len(self.symbols),
            'bars_processed':self._bars_total,
            'signals_fired': self._signals_total,
            'queue_size':    self.signal_queue.qsize(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._supervisor())

    async def _supervisor(self) -> None:
        """Split ALL symbols across multiple WS connections (180 streams each)."""
        # 5m only — we don't need other timeframes for the Nike check
        streams_all = [f"{s.lower()}@kline_5m" for s in self.symbols]
        chunks = [
            streams_all[i:i + MAX_STREAMS]
            for i in range(0, len(streams_all), MAX_STREAMS)
        ]
        logging.info(
            f"ThorScreener: {len(self.symbols)} symbols → "
            f"{len(chunks)} WS connections"
        )
        tasks = [
            asyncio.create_task(self._connection_loop(chunk, idx))
            for idx, chunk in enumerate(chunks)
        ]
        # Stagger start to avoid rate-limiting
        for i, task in enumerate(tasks):
            if i > 0:
                await asyncio.sleep(1.5)
        await asyncio.gather(*tasks)

    async def _connection_loop(self, stream_chunk: List[str], conn_id: int) -> None:
        url = WS_BASE + "/".join(stream_chunk)
        while not self._stop.is_set():
            try:
                await self._connect(url, conn_id)
            except Exception as exc:
                if not self._stop.is_set():
                    logging.debug(f"ThorScreener conn-{conn_id}: {exc} — reconnecting in {RECONNECT_DELAY}s")
                    await asyncio.sleep(RECONNECT_DELAY)

    async def _connect(self, url: str, conn_id: int) -> None:
        import ssl
        import aiohttp

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE

        connector     = aiohttp.TCPConnector(ssl=ssl_ctx)
        ws_kwargs     = dict(
            heartbeat   = 20,
            max_msg_size= 2 ** 20,
            ssl         = ssl_ctx,
        )
        session_kwargs = {"connector": connector}

        if self.proxy_url:
            if self.proxy_url.startswith("socks"):
                try:
                    from aiohttp_socks import ProxyConnector
                    connector = ProxyConnector.from_url(self.proxy_url, ssl=ssl_ctx, rdns=True)
                    session_kwargs = {"connector": connector}
                    ws_kwargs.pop("ssl", None)
                except ImportError:
                    pass
            else:
                ws_kwargs["proxy"] = self.proxy_url

        async with aiohttp.ClientSession(**session_kwargs) as sess:
            async with sess.ws_connect(url, **ws_kwargs) as ws:
                async for msg in ws:
                    if self._stop.is_set():
                        return
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._handle(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return

    def _handle(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            d    = data.get("data", data)
            if d.get("e") != "kline":
                return
            k      = d["k"]
            closed = k["x"]
            if not closed:
                return   # only process fully-closed bars

            symbol = k["s"]
            buf    = self._buffers[symbol]
            buf.push(
                float(k["o"]), float(k["h"]), float(k["l"]),
                float(k["c"]), float(k["v"])
            )
            self._bars_total += 1

            sig_meta = _nike_check(buf)
            if sig_meta is None:
                return

            # Build NikeSignal
            entry    = sig_meta['close']
            atr      = sig_meta['atr']
            tp_price = entry + TP_ATR * atr if atr > 0 else entry * (1 + 0.02)
            sl_price = entry - SL_ATR * atr if atr > 0 else entry * (1 - 0.008)

            ts  = int(k["T"]) / 1000   # close time of bar
            sig = NikeSignal(
                symbol        = symbol,
                timestamp     = ts,
                date_str      = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                close         = entry,
                atr           = atr,
                tp_price      = tp_price,
                sl_price      = sl_price,
                body_pct      = sig_meta['body_pct'],
                body_ratio    = sig_meta['body_ratio'],
                avg_prior_pct = sig_meta['avg_prior_pct'],
                vol_ratio     = sig_meta['vol_ratio'],
                body_eff      = sig_meta['body_eff'],
                tier          = sig_meta['tier'],
                entry_mode    = sig_meta['entry_mode'],
                score         = sig_meta['score'],
                confidence    = sig_meta['confidence'],
                size_mult     = sig_meta['size_mult'],
                bs_floor      = sig_meta['bs_floor'],
            )

            self._signals_total += 1

            # Push to queue (non-blocking, drop oldest if full)
            try:
                self.signal_queue.put_nowait(sig)
            except queue.Full:
                try:
                    self.signal_queue.get_nowait()   # drop oldest
                    self.signal_queue.put_nowait(sig)
                except Exception:
                    pass

            # Optional callback
            if self.on_signal:
                try:
                    self.on_signal(sig)
                except Exception as e:
                    logging.debug(f"ThorScreener on_signal callback error: {e}")

            logging.info(
                f"THOR  {symbol:<18}  "
                f"tier={sig.tier}  "
                f"body={sig.body_pct:+.2f}%  "
                f"ratio={sig.body_ratio:.1f}x  "
                f"quiet={sig.avg_prior_pct:.3f}%  "
                f"vol={sig.vol_ratio:.1f}x  "
                f"score={sig.score:.1f}  "
                f"entry={sig.close:.6g}  "
                f"TP={sig.tp_price:.6g}  SL={sig.sl_price:.6g}"
            )

        except Exception as e:
            logging.debug(f"ThorScreener _handle error: {e}")
