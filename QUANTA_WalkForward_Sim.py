"""
QUANTA v12  Walk-Forward OOS Simulation
==========================================
Simulates the WHOLE bot bar-by-bar on historical data.

Key design principle: ZERO lookahead.
  - Each bar is processed in strict time order, as if the bot is running live.
  - Thor signals are detected using the exact same _SymbolBuffer + _thor_check
    logic from quanta_thor_screener.py  the live bot's inline check.
  - No pre-extracted events. No snapshots. The sim discovers signals itself,
    candle by candle, exactly as the live bot would.
  - SparseFeatureContext is built causally from OHLCV data up to current bar.
  - Position management uses the same Thor v2 two-phase exit (bank + runner)
    as QUANTA_trading_core._tick_thor_v2.

Walk-forward structure (Pardo 2008 + LdP AFML Ch.7):
  Train [0:T-purge]  purge 48c  Test [T:T+S]  step S  repeat.
  Train window is only used to warm up the per-symbol Thor buffers.

References:
  - Pardo (2008)  Walk-Forward Optimization
  - Lpez de Prado (2018) AFML Ch.7 7.4  purge gap (max_bars = 48)
  - Bailey et al. (2014)  deflated Sharpe ratio
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("wf_sim")

#  Config 
from quanta_config import Config as _cfg
_ev = _cfg.events

#  Live Thor detection (exact same logic as the live bot) 
from quanta_thor_screener import (
    _SymbolBuffer,
    _thor_check,
    COOLDOWN_BARS as _THOR_COOLDOWN,
)

#  Thor context scoring
from quanta_norse_agents import (
    SparseFeatureContext,
    build_sparse_feature_context,
    score_thor_signal,
    calc_atr,
    calc_vol_avg20,
)

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── Thor CatBoost model (offline inference) ─────────────────────────────────
# IMPULSE domain mask — 102 features, matches thor_gen1.cbm training exactly.
# Derived from QUANTA_ml_engine.py _IMPULSE definition (lines 576-588).
_IMPULSE_MASK = np.array([
    0,1,2,4,7,8,9,11,14,15,16,18,21,22,23,25,28,29,30,32,
    35,36,37,39,42,43,44,46,
    58,59,60,73,74,75,
    163,164,165,166,167,168,169,170,171,
    191,192,193,194,195,196,197,198,199,
    200,201,202,203,204,205,206,207,208,209,210,211,212,213,
    214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,
    232,
    258,259,260,261,262,263,264,265,266,267,
    270,271,272,273,274,275,276,277,
], dtype=np.int64)

_THOR_MODEL   = None
_REPLAY_ENG   = None

def _find_thor_model() -> Optional[str]:
    """Return absolute path to thor_gen1.cbm, searching common locations."""
    candidates = [
        Path("models/thor_gen1.cbm"),
        Path(__file__).parent / "models" / "thor_gen1.cbm",
        Path(__file__).parent.parent / "models" / "thor_gen1.cbm",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def _load_thor_assets():
    """Load Thor CatBoost model + offline replay engine once."""
    global _THOR_MODEL, _REPLAY_ENG
    if _THOR_MODEL is not None:
        return True
    model_path = _find_thor_model()
    if model_path is None:
        log.warning("thor_gen1.cbm not found — falling back to OHLCV proxy")
        return False
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(model_path)
        _THOR_MODEL = m
        log.info("Thor gen1 model loaded (%d trees) from %s", m.tree_count_, model_path)
    except Exception as e:
        log.warning("Thor model load failed: %s", e)
        return False
    try:
        from QUANTA_ml_engine import build_offline_feature_replay_engine as _boe
        _REPLAY_ENG = _boe(cfg=_cfg)
        log.info("Offline replay engine ready")
    except Exception as e:
        log.warning("Replay engine unavailable: %s", e)
        _THOR_MODEL = None
        return False
    return True

#  Walk-forward constants
_BT           = getattr(_cfg, "backtest", None)
_TRAIN_DAYS   = 60    # 60d train → 10 OOS windows from 365d dataset (hardcoded — config has 180d)
_TEST_DAYS    = getattr(_BT, "test_window_days",    30)
_STEP_DAYS    = getattr(_BT, "step_days",           30)
_COMMISSION   = getattr(_BT, "commission_bps",     4.0) / 10_000
_SLIPPAGE     = getattr(_BT, "slippage_bps",       2.0) / 10_000
_CANDLES_DAY  = 288        # 5m candles per day
_PURGE_GAP    = 48         # max_bars across all agents

#  Thor operating params — calibrated from MAE Stats Report
# Source: NORSE_MAE_STATS_REPORT.md — Recommended Parameter Block + Decision Matrix
_MAX_CONC          = int(getattr(_ev, "thor_max_concurrent_positions", 10))
_MIN_SCORE         = float(getattr(_ev, "thor_min_score_trade",       68.0))
_COOLDOWN_BRS      = int(getattr(_ev, "thor_trade_cooldown_bars",     36))
_RISK_PCT          = float(getattr(_ev, "thor_risk_pct",              0.005))
_BANK_ATR          = 4.20   # MAE report: top decision-matrix combo (was 5.40)
_BANK_FRAC         = 0.35   # MAE report: top decision-matrix combo (was 0.45)
_TRAIL_ACTIVATE_ATR= 1.50   # MAE report: trail only kicks in 1.5 ATR above bank price
_TRAIL_ATR         = 2.00   # MAE report: tighter runner trail (was 6.00)
_SL_ATR            = 3.00   # MAE report: P90 winner MAE from entry (was 2.40)
_MAX_PRE           = 48     # MAE report: 4 hours — after that timeout pre-bank (was 144)
_MAX_POST          = 96     # MAE report: 8 hours post-bank runner window (was 1152)
_MAE_VETO_BARS     = 5      # MAE report: if adverse > _MAE_VETO_ATR in first 5 bars, exit early
_MAE_VETO_ATR      = 3.62   # MAE report: P50 winner MAE from entry
# Hour-of-day UTC filter — from hour-of-week profit heatmap (hours with PF < 0.85)
_SKIP_UTC_HOURS    = frozenset({0, 7, 10, 11, 12})  # worst: 12→PF0.51, 0→0.68, 10→0.75, 7/11→0.81

#  Entry quality gates (Norse learned filter)
_WAVE_MIN     = float(getattr(_ev, "thor_wave_strength_min",    40.0))   # wave proxy gate
_PRE_R2_MAX   = float(getattr(_ev, "thor_pre_impulse_r2_max",  0.70))   # exhaustion veto

# ── Gompertz Hazard Model — Dynamic Bank Target (Khairul's Identity companion) ──
# λ(t) = λ₀·e^(γt)  — collapse hazard accelerates as pump ages.
# Optimal bank: t* = ln(n_eff/λ₀)/γ  where n_eff = λ(t*) crossover.
# Calibrated from WF micro-pump anchors:
#   t₁=0.104d (30 bars, avg bank 4.20 ATR): n=0.700, λ(t₁)=0.700 → λ₀=0.517
#   t₂=0.226d (65 bars, avg runner 6.09 ATR): collapse dominant → λ(t₂)=1.0
# Ref: reference_journals.md — "λ(t): The Pump Collapse Hazard Companion Equation"
_N_PUMP_MICRO    = 0.700   # prior micro-pump velocity day⁻¹ at CUSUM fire
_GOMPERTZ_L0     = 0.517   # λ₀: baseline collapse hazard day⁻¹
_GOMPERTZ_GAMMA  = 2.92    # γ: hazard acceleration day⁻¹
_DYN_BANK_MIN    = 2.00    # floor — never bank below 2 ATR
_DYN_BANK_MAX    = 10.00   # ceiling — cap to prevent runaway targets

#  Pyramid averaging-in (3-layer strategy, calibrated from MAE stats)
#  Layer 1 — Thor entry (normal)
#  Layer 2 — Add at +0.5 ATR, SL = 0.5 ATR below add entry (= original entry price)
#  Layer 3 — Recovery re-entry when add SL hits; target = median pump runup 3.77 ATR
_PYR_TRIGGER_ATR  = 0.5    # price must reach entry + 0.5 ATR to trigger add
_PYR_ADD_SL_ATR   = 0.5    # add SL is 0.5 ATR below add entry
_PYR_ADD_FRAC     = 0.50   # add size = 50% of layer-1 size (matches Norse Freya notional_fraction)
_PYR_RECOVERY_ATR = 3.77   # recovery re-entry target = median pump runup from MAE stats (overall p50 = 3.80)


#
# Gompertz dynamic bank target
#

import math as _math

_K_RUNNER_HAZARD  = 2.5   # exit runner when λ(t) = 2.5 × n  (collapse risk 2.5× growth)
_K_PRE_HAZARD     = 1.5   # abort pre-bank when λ(t) = 1.5 × n (mild deceleration = give up)

def _pump_n_eff(entry: float, close: float, bars_open: int) -> float:
    """Observed pump velocity n_eff (day⁻¹), blended with prior for early-bar stability.

    Full trust in observed rate after 24 bars (2 hours).
    Uses _N_PUMP_MICRO as prior when bars < 3 or price hasn't moved up yet.
    """
    t_days = bars_open * (5.0 / 1440.0)
    if bars_open >= 3 and close > entry and t_days > 1e-9:
        n_raw = _math.log(close / entry) / t_days
        alpha = min(1.0, bars_open / 24.0)
        return alpha * n_raw + (1.0 - alpha) * _N_PUMP_MICRO
    return _N_PUMP_MICRO


def _gompertz_t_star(n_eff: float, k: float = 1.0) -> float:
    """Days until λ(t) = k × n_eff  →  t = ln(k·n_eff / λ₀) / γ.

    k=1.0  → optimal bank (n = λ crossover)
    k>1.0  → runner timeout (λ has overtaken n by factor k, EV clearly declining)
    Returns 0.0 if n_eff ≤ λ₀ (already in decline at entry).
    """
    target = k * n_eff
    if target <= _GOMPERTZ_L0:
        return 0.0
    return _math.log(target / _GOMPERTZ_L0) / _GOMPERTZ_GAMMA


def _dynamic_bank_atr(n_eff: float, atr_pct: float) -> float:
    """Optimal bank ATR via Gompertz hazard model (Khairul's Identity companion).

    t* = ln(n_eff / λ₀) / γ  → bank_atr = (e^(n_eff·t*) − 1) / atr_pct
    Fast pumps → higher bank.  Slow / stalling pumps → lower bank (early lock-in).
    """
    if n_eff <= _GOMPERTZ_L0:
        ratio = max(n_eff / _GOMPERTZ_L0, 0.1)
        return max(_DYN_BANK_MIN, _BANK_ATR * (ratio ** 0.5))

    t_star     = _gompertz_t_star(n_eff, k=1.0)
    price_mult = _math.exp(n_eff * t_star)
    atr_units  = (price_mult - 1.0) / atr_pct
    return float(max(_DYN_BANK_MIN, min(_DYN_BANK_MAX, atr_units)))


def _dynamic_pre_timeout(n_eff: float) -> int:
    """Max bars before bank hit — based on Gompertz pump deceleration.

    Pre-bank timeout = bars until λ(t) = _K_PRE_HAZARD × n_eff.
    If a coin stalls for longer than this, collapse risk has overtaken growth — abandon.
    Floor: 12 bars (1h).  Ceiling: 144 bars (12h).
    """
    t_abort = _gompertz_t_star(n_eff, k=_K_PRE_HAZARD)   # days
    bars    = int(t_abort * 288)                           # 288 bars/day at 5m
    return max(12, min(144, bars))


def _dynamic_post_timeout(n_eff: float, bars_at_bank: int) -> int:
    """Max bars after bank hit (runner phase) — based on Gompertz hazard.

    Runner window = bars from bank until λ(t) = _K_RUNNER_HAZARD × n_eff.
    Replaces hardcoded _MAX_POST = 96 (previously 1152 = 4 days before MAE calibration).
    Floor: 12 bars (1h).  Ceiling: 576 bars (2 days).
    """
    t_bank_days   = bars_at_bank * (5.0 / 1440.0)
    t_runner_end  = _gompertz_t_star(n_eff, k=_K_RUNNER_HAZARD)  # days from entry
    runner_days   = max(0.0, t_runner_end - t_bank_days)
    bars          = int(runner_days * 288)
    return max(12, min(576, bars))


#
# Position tracker
#

@dataclass
class SimPosition:
    symbol:       str
    entry_bar:    int
    entry_price:  float
    atr:          float
    size:         float
    bank_price:   float
    sl_price:     float
    bank_hit:          bool  = False
    runner_peak:       float = 0.0
    bars_open:         int   = 0
    partial_pnl:       float = 0.0   # realised at bank
    score:             float = 0.0
    tier:              str   = ""
    trail_active:      bool  = False  # True once runner_peak >= bank_price + TRAIL_ACTIVATE_ATR
    lowest_price:      float = 0.0    # track for MAE veto in early bars
    dyn_bank_atr:      float = _BANK_ATR      # live Gompertz optimal bank target (ATR units)
    n_eff:             float = _N_PUMP_MICRO  # observed pump velocity day⁻¹ (blended)
    dyn_max_pre:       int   = _MAX_PRE       # dynamic pre-bank timeout (bars)
    dyn_max_post:      int   = _MAX_POST      # dynamic runner timeout (bars, set at bank hit)
    # ── Pyramid Layer 2: add at +0.5 ATR ─────────────────────────────────────
    add1_active:    bool  = False   # add position currently open
    add1_entry:     float = 0.0     # fill price for the add
    add1_sl:        float = 0.0     # SL for add (0.5 ATR below add entry)
    add1_size:      float = 0.0     # units (50% of layer-1 size)
    add1_pnl:       float = 0.0     # realised PnL from add close
    # ── Pyramid Layer 3: recovery re-entry when add SL hits ───────────────────
    add2_active:    bool  = False   # recovery position currently open
    add2_entry:     float = 0.0     # fill price for recovery
    add2_sl:        float = 0.0     # SL = original entry - _SL_ATR * atr
    add2_target:    float = 0.0     # TP = original entry + _PYR_RECOVERY_ATR * atr
    add2_size:      float = 0.0
    add2_pnl:       float = 0.0


@dataclass
class SimTrade:
    symbol:      str
    entry_bar:   int
    exit_bar:    int
    entry_price: float
    exit_price:  float
    pnl:         float    # total PnL including all pyramid layers
    pnl_pct:     float
    barrier_hit: str
    score:       float
    tier:        str
    window_id:   int
    add1_pnl:      float = 0.0         # Layer 2 pyramid contribution
    add2_pnl:      float = 0.0         # Layer 3 recovery contribution
    layers:        int   = 1           # 1 = main only, 2 = add fired, 3 = recovery fired
    dyn_bank_atr:  float = _BANK_ATR   # Gompertz bank target ATR at exit
    n_eff:         float = _N_PUMP_MICRO  # observed pump velocity day⁻¹ at close


# 
# Position manager  Thor v2 two-phase exit, no side-effects
# 

class SimPositionManager:

    def __init__(self, balance: float):
        self.balance      = balance
        self.positions:   Dict[str, SimPosition] = {}
        self.trades:      List[SimTrade]          = []
        self._last_close: Dict[str, int]          = {}
        self.equity_curve: List[Tuple]            = []   # (bar, balance) after each close

    #  guards 

    def can_open(self, symbol: str, bar: int) -> bool:
        if symbol in self.positions:
            return False
        if len(self.positions) >= _MAX_CONC:
            return False
        if bar - self._last_close.get(symbol, -99999) < _COOLDOWN_BRS:
            return False
        return True

    #  open 

    def open(self, symbol: str, bar: int, close: float, atr: float,
             score: float, tier: str, w_id: int):
        entry = close * (1 + _SLIPPAGE + _COMMISSION)
        sl_dist = atr * _SL_ATR
        # ── Continuous compounding: risk scales linearly with score ──
        # At score=_MIN_SCORE (68): 0.5% risk
        # At score=100:             3.0% risk
        # This matches the distribution of CatBoost P(win)*100 scores which
        # cluster in 68-85 range — the old binary 85-threshold was never firing.
        _max_risk  = float(getattr(_ev, 'thor_compound_max_loss_pct', 3.0)) / 100.0
        _score_range = max(100.0 - _MIN_SCORE, 1.0)
        _effective_risk = _RISK_PCT + (score - _MIN_SCORE) / _score_range * (_max_risk - _RISK_PCT)
        _effective_risk = float(np.clip(_effective_risk, _RISK_PCT, _max_risk))
        size    = (self.balance * _effective_risk) / max(sl_dist, 1e-8)

        self.positions[symbol] = SimPosition(
            symbol=symbol, entry_bar=bar, entry_price=entry,
            atr=atr, size=size,
            bank_price=entry + atr * _BANK_ATR,
            sl_price=entry  - atr * _SL_ATR,
            runner_peak=entry,
            lowest_price=entry,
            score=score, tier=tier,
        )

    #  tick (called every bar for every open position) 

    def tick(self, symbol: str, bar: int,
             high: float, low: float, close: float, w_id: int):
        pos = self.positions.get(symbol)
        if pos is None:
            return

        pos.bars_open += 1

        # ── Pyramid Layer 2: trigger add when price +0.5 ATR above entry ────────
        add1_trigger_px = pos.entry_price + pos.atr * _PYR_TRIGGER_ATR
        if not pos.add1_active and not pos.bank_hit:
            if high >= add1_trigger_px:
                pos.add1_entry  = add1_trigger_px * (1 + _SLIPPAGE + _COMMISSION)
                pos.add1_sl     = add1_trigger_px - pos.atr * _PYR_ADD_SL_ATR
                pos.add1_size   = pos.size * _PYR_ADD_FRAC
                pos.add1_active = True

        # ── Pyramid Layer 2: tick add SL ──────────────────────────────────────
        if pos.add1_active:
            if low <= pos.add1_sl:
                # Add SL hit — close add, credit/debit balance
                fill_add     = pos.add1_sl * (1 - _SLIPPAGE - _COMMISSION)
                pos.add1_pnl = (fill_add - pos.add1_entry) * pos.add1_size
                self.balance += pos.add1_pnl
                pos.add1_active = False
                # ── Pyramid Layer 3: recovery re-entry at the add SL level ───
                # Target = original entry + median pump runup (3.77 ATR from MAE stats)
                if not pos.add2_active and not pos.bank_hit:
                    pos.add2_entry  = fill_add * (1 + _SLIPPAGE + _COMMISSION)
                    pos.add2_sl     = pos.entry_price - pos.atr * _SL_ATR  # same as original SL
                    pos.add2_target = pos.entry_price + pos.atr * _PYR_RECOVERY_ATR
                    pos.add2_size   = pos.size * _PYR_ADD_FRAC
                    pos.add2_active = True

        # ── Pyramid Layer 3: tick recovery ────────────────────────────────────
        if pos.add2_active:
            if low <= pos.add2_sl:
                fill_r2      = pos.add2_sl * (1 - _SLIPPAGE - _COMMISSION)
                pos.add2_pnl = (fill_r2 - pos.add2_entry) * pos.add2_size
                self.balance += pos.add2_pnl
                pos.add2_active = False
            elif high >= pos.add2_target:
                fill_r2      = pos.add2_target * (1 - _SLIPPAGE - _COMMISSION)
                pos.add2_pnl = (fill_r2 - pos.add2_entry) * pos.add2_size
                self.balance += pos.add2_pnl
                pos.add2_active = False

        # Track lowest price for MAE veto
        if low < pos.lowest_price:
            pos.lowest_price = low

        if not pos.bank_hit:
            # ── MAE early exit veto (first _MAE_VETO_BARS bars only) ──────────
            # Losers show large adverse excursion early — exit before full SL.
            # Calibrated from P50 winner MAE from entry in NORSE_MAE_STATS_REPORT.
            if pos.bars_open <= _MAE_VETO_BARS:
                mae = (pos.entry_price - pos.lowest_price) / max(pos.atr, 1e-8)
                if mae > _MAE_VETO_ATR:
                    self._close(pos, bar, pos.lowest_price, "MAE_VETO", w_id)
                    return

            # ── Gompertz dynamic targets (Khairul's Identity λ companion eq) ──
            # n_eff  = observed pump velocity (blended with prior, stabilises over 24 bars)
            # bank   = price where E[P(t)] peaks  (n = λ crossover)
            # pre    = bars until λ = 1.5×n  (pump stalling → abandon pre-bank)
            # post   = computed at bank hit: bars until λ = 2.5×n from bank level
            # Replaces: hardcoded _BANK_ATR=4.20, _MAX_PRE=48, _MAX_POST=96 (was 1152=4 days)
            pos.n_eff        = _pump_n_eff(pos.entry_price, close, pos.bars_open)
            atr_pct          = pos.atr / pos.entry_price if pos.entry_price > 1e-8 else 0.018
            pos.dyn_bank_atr = _dynamic_bank_atr(pos.n_eff, atr_pct)
            pos.dyn_max_pre  = _dynamic_pre_timeout(pos.n_eff)
            dyn_bank_px      = pos.entry_price + pos.dyn_bank_atr * pos.atr

            # Pre-bank phase
            if low <= pos.sl_price:
                self._close(pos, bar, pos.sl_price, "SL", w_id)
                return
            if high >= dyn_bank_px:
                fill = dyn_bank_px * (1 - _SLIPPAGE - _COMMISSION)
                bank_pnl = (fill - pos.entry_price) * pos.size * _BANK_FRAC
                # Bank the add layer 1 too if still open (at same price)
                if pos.add1_active:
                    add_bank_fill = fill
                    pos.add1_pnl += (add_bank_fill - pos.add1_entry) * pos.add1_size * _BANK_FRAC
                    self.balance  += pos.add1_pnl
                    pos.add1_size *= (1 - _BANK_FRAC)
                self.balance     += bank_pnl
                pos.partial_pnl   = bank_pnl
                pos.size         *= (1 - _BANK_FRAC)
                pos.bank_hit      = True
                pos.bank_price    = dyn_bank_px           # record actual dynamic bank level
                pos.sl_price      = pos.entry_price       # move to breakeven
                pos.runner_peak   = high
                # Set dynamic runner window — bars until λ = 2.5×n from this bar
                pos.dyn_max_post  = _dynamic_post_timeout(pos.n_eff, pos.bars_open)
                # Trail activation: runner must reach dyn_bank_px + TRAIL_ACTIVATE_ATR
                trail_activate_px = dyn_bank_px + pos.atr * _TRAIL_ACTIVATE_ATR
                if high >= trail_activate_px:
                    pos.trail_active = True
                    trail = max(pos.sl_price, pos.runner_peak - pos.atr * _TRAIL_ATR)
                    if low <= trail:
                        self._close(pos, bar, trail, "CHANDELIER_SL", w_id)
                return
            if pos.bars_open >= pos.dyn_max_pre:
                self._close(pos, bar, close, "TIMEOUT_PRE", w_id)
                return
        else:
            # Post-bank runner phase
            if high > pos.runner_peak:
                pos.runner_peak = high
            # Trail only activates once runner clears bank_price + TRAIL_ACTIVATE_ATR
            trail_activate_px = pos.bank_price + pos.atr * _TRAIL_ACTIVATE_ATR
            if pos.runner_peak >= trail_activate_px:
                pos.trail_active = True
            trail = (max(pos.sl_price, pos.runner_peak - pos.atr * _TRAIL_ATR)
                     if pos.trail_active else pos.sl_price)
            if low <= trail:
                self._close(pos, bar, trail, "CHANDELIER_SL", w_id)
                return
            if pos.bars_open >= pos.dyn_max_pre + pos.dyn_max_post:
                self._close(pos, bar, close, "RUNNER_TIMEOUT", w_id)

    #  close 

    def _close(self, pos: SimPosition, bar: int, raw_exit: float,
               reason: str, w_id: int):
        fill    = raw_exit * (1 - _SLIPPAGE - _COMMISSION)
        run_pnl = (fill - pos.entry_price) * pos.size
        total   = run_pnl + pos.partial_pnl
        self.balance += run_pnl

        # ── Force-close any still-open pyramid layers at the same exit price ──
        layers = 1
        if pos.add1_active:
            pnl1 = (fill - pos.add1_entry) * pos.add1_size
            pos.add1_pnl  += pnl1
            self.balance  += pnl1
            pos.add1_active = False
            layers = max(layers, 2)
        if pos.add2_active:
            pnl2 = (fill - pos.add2_entry) * pos.add2_size
            pos.add2_pnl  += pnl2
            self.balance  += pnl2
            pos.add2_active = False
            layers = max(layers, 3)

        total += pos.add1_pnl + pos.add2_pnl

        self._last_close[pos.symbol] = bar
        del self.positions[pos.symbol]

        # Per-trade equity snapshot for accurate drawdown tracking
        self.equity_curve.append((bar, self.balance))

        ref_size = pos.size / (1 - _BANK_FRAC) if pos.bank_hit else pos.size
        self.trades.append(SimTrade(
            symbol=pos.symbol, entry_bar=pos.entry_bar, exit_bar=bar,
            entry_price=pos.entry_price, exit_price=fill,
            pnl=total,
            pnl_pct=total / max(pos.entry_price * ref_size, 1e-8) * 100,
            barrier_hit=reason, score=pos.score, tier=pos.tier,
            window_id=w_id,
            add1_pnl=round(pos.add1_pnl, 4),
            add2_pnl=round(pos.add2_pnl, 4),
            layers=layers,
            dyn_bank_atr=round(pos.dyn_bank_atr, 3),
            n_eff=round(pos.n_eff, 4),
        ))

    def force_close_all(self, all_klines: Dict[str, np.ndarray],
                        bar: int, w_id: int):
        for sym in list(self.positions.keys()):
            kl = all_klines.get(sym)
            idx = min(bar, len(kl) - 1) if kl is not None else bar
            px  = float(kl[idx, 4]) if kl is not None else self.positions[sym].entry_price
            self._close(self.positions[sym], bar, px, "WINDOW_END", w_id)


# 
# SparseFeatureContext builder  causal OHLCV only, no ML needed
# 

def _thor_model_score(precomputed: dict, abs_bar: int,
                      klines_np: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Real Thor CatBoost inference using the IMPULSE domain mask (102 features).
    Returns (score [0,100], full_feature_vec) or (None, None) on failure.
    full_feature_vec is the cleaned 278-dim vector — used by callers for
    post-hoc gate checks (e.g. pre_impulse_r2 at index 272).
    """
    if _THOR_MODEL is None or _REPLAY_ENG is None:
        return None, None
    try:
        feat = _REPLAY_ENG._fast_extract_at_position(
            precomputed, abs_bar, klines_np, symbol=None
        )
        if feat is None:
            return None, None
        vec = np.asarray(feat, dtype=np.float64)
        if len(vec) < 278:
            return None, None
        # Replace NaN/inf with neutral 0
        vec = np.where(np.isfinite(vec), vec, 0.0)
        x = vec[_IMPULSE_MASK].reshape(1, -1)
        prob = float(_THOR_MODEL.predict_proba(x)[0][1])
        return max(0.0, min(100.0, prob * 100.0)), vec
    except Exception as e:
        log.debug("Thor model inference error at bar %d: %s", abs_bar, e)
        return None, None


def _wave_strength_proxy(klines_np: np.ndarray, abs_bar: int) -> float:
    """
    Wave strength from REAL taker buy flow (feather cache col [9] = taker_buy_base).
    Matches the Norse sim's wave_strength_score calculation exactly.

    Klines column layout (Binance 5m):
      [0] open_time  [1] open  [2] high  [3] low  [4] close
      [5] volume     [6] close_time  [7] quote_vol  [8] trades
      [9] taker_buy_base  [10] taker_buy_quote  [11] ignore

    wave_strength = net taker flow ratio over 20 bars → [0, 100]
      net = (taker_buy - taker_sell) / total_vol
      score = (net + 1) / 2 × 100   →  all buying=100, neutral=50, all selling=0
    """
    lo = max(0, abs_bar - 20)
    sl = klines_np[lo:abs_bar + 1]
    if len(sl) < 5:
        return 50.0

    total_vol = float(np.sum(sl[:, 5]))
    if total_vol < 1e-8:
        return 50.0

    # Use real taker buy if column exists and is populated
    if sl.shape[1] > 9:
        taker_buy  = float(np.sum(sl[:, 9]))
        taker_sell = total_vol - taker_buy
        net_ratio  = (taker_buy - taker_sell) / total_vol    # [-1, +1]
    else:
        # Fallback: directional volume proxy
        opens  = sl[:, 1].astype(np.float64)
        closes = sl[:, 4].astype(np.float64)
        vols   = sl[:, 5].astype(np.float64)
        dirs   = np.where(closes >= opens, 1.0, -1.0)
        net_ratio = float(np.sum(dirs * vols)) / total_vol

    return float(np.clip((net_ratio + 1.0) / 2.0 * 100.0, 0.0, 100.0))


def _sim_thor_score(sig: dict, klines_np: np.ndarray, abs_bar: int,
                    precomputed: Optional[dict] = None) -> Tuple[float, Optional[np.ndarray]]:
    """
    Thor quality score.

    Primary path  — real Thor CatBoost model (102 IMPULSE-mask features):
      Requires thor_gen1.cbm + offline replay engine.
      Returns 100 × P(win) from the trained model.

    Fallback path  — OHLCV-only proxy (when model unavailable):
      base(30%) + participation(24%) + trend(16%) + bs_edge(16%) + continuation(14%)
      minus fade penalty. No discriminative power vs real model but keeps the
      pipeline runnable without the model files.
    """
    # ── Primary: real model ──
    if precomputed is not None:
        score_ml, feat_vec = _thor_model_score(precomputed, abs_bar, klines_np)
        if score_ml is not None:
            return score_ml, feat_vec

    # ── Fallback: OHLCV proxy ──
    base = float(sig.get("score", 50.0)) / 100.0

    lo  = max(0, abs_bar - 24)
    hi  = abs_bar + 1
    sl  = klines_np[lo:hi]
    if len(sl) < 5:
        return base * 30.0, None

    closes  = sl[:, 4].astype(np.float64)
    opens   = sl[:, 1].astype(np.float64)
    volumes = sl[:, 5].astype(np.float64)
    highs   = sl[:, 2].astype(np.float64)
    lows    = sl[:, 3].astype(np.float64)

    vol_ratio     = float(sig.get("vol_ratio", 1.0))
    participation = float(np.clip((vol_ratio - 1.0) / 4.0, 0.0, 1.0))

    n   = len(closes)
    ema = float(closes[0])
    alpha = 2.0 / (min(10, n) + 1)
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
    price_vs_ema = float(np.clip((closes[-1] - ema) / max(ema * 0.001, 1e-12), -1.0, 1.0))
    trend = float(np.clip((price_vs_ema + 0.1) / 0.7, 0.0, 1.0))

    body_pct = float(sig.get("body_pct", 0.0)) / 100.0
    atr_pct  = (highs[-1] - lows[-1]) / max(closes[-1], 1e-8)
    bs_edge  = float(np.clip(body_pct / max(atr_pct, 1e-8) - 0.5, 0.0, 1.0))

    body_eff    = float(sig.get("body_eff", 0.5))
    vol_persist = float(np.clip(volumes[-1] / max(volumes[-2], 1e-8) - 0.3, 0.0, 1.0)) if n >= 3 else 0.5
    continuation = float(np.clip(0.6 * body_eff + 0.4 * vol_persist, 0.0, 1.0))

    v3   = float(np.mean(volumes[-4:-1])) if n >= 4 else volumes[-1]
    fade = float(np.clip((v3 - volumes[-1]) / max(v3, 1e-8), 0.0, 1.0))

    score = 100.0 * (
        0.30 * base
        + 0.24 * participation
        + 0.16 * trend
        + 0.16 * bs_edge
        + 0.14 * continuation
    ) - 15.0 * fade

    return max(0.0, min(100.0, float(score))), None


def _build_causal_context(symbol: str, klines_np: np.ndarray,
                          start: int, end: int) -> Optional[SparseFeatureContext]:
    """Build SparseFeatureContext from the raw OHLCV slice [start:end].
    All arrays are computed causally  no future data used."""
    if not _HAS_PANDAS or end - start < 50:
        return None
    try:
        sl = klines_np[start:end]
        df = pd.DataFrame(sl, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy_base",
            "taker_buy_quote","ignore",
        ])
        df.attrs["symbol"] = symbol
        for col in ("open","high","low","close","volume","quote_volume","taker_buy_base"):
            df[col] = df[col].astype(np.float64)
        # Pass empty features_by_pos  all fields default to OHLCV-derived values
        return build_sparse_feature_context(df, {})
    except Exception as e:
        log.debug("context build failed for %s: %s", symbol, e)
        return None


# 
# Cache loader
# 

def _load_cache(symbols: List[str],
                cache_dir: str = "feather_cache") -> Dict[str, np.ndarray]:
    from quanta_cache import FeatherCache
    fc = FeatherCache(cache_dir)
    out: Dict[str, np.ndarray] = {}
    for sym in symbols:
        try:
            klines = fc.get(sym, "5m", limit=200_000)
            if klines and len(klines) >= 500:
                out[sym] = np.array(klines, dtype=np.float64)
            else:
                log.warning("skip %s  insufficient cache (%d bars)",
                            sym, len(klines) if klines else 0)
        except Exception as e:
            log.warning("cache error %s: %s", sym, e)
    return out


# 
# Metrics
# 

@dataclass
class WFMetrics:
    total_trades:       int   = 0
    wins:               int   = 0
    losses:             int   = 0
    win_rate:           float = 0.0
    total_pnl:          float = 0.0
    total_pnl_pct:      float = 0.0
    profit_factor:      float = 0.0
    sharpe_ratio:       float = 0.0
    sortino_ratio:      float = 0.0
    calmar_ratio:       float = 0.0
    max_drawdown_pct:   float = 0.0
    window_count:       int   = 0
    windows_profitable: int   = 0
    window_sharpe_mean: float = 0.0
    window_sharpe_std:  float = 0.0
    tier_a_trades:      int   = 0
    tier_b_trades:      int   = 0
    tier_c_trades:      int   = 0
    chandelier_hits:    int   = 0
    sl_hits:            int   = 0
    timeout_hits:       int   = 0
    runtime_sec:        float = 0.0


def _compute_metrics(trades: List[SimTrade], equity: List[Tuple],
                     initial: float, win_mets: List[dict],
                     t0: float) -> WFMetrics:
    m = WFMetrics(runtime_sec=time.time() - t0)
    if not trades:
        return m

    m.total_trades = len(trades)
    m.wins    = sum(1 for t in trades if t.pnl > 0)
    m.losses  = m.total_trades - m.wins
    m.win_rate = m.wins / m.total_trades * 100

    pnls = [t.pnl for t in trades]
    m.total_pnl = sum(pnls)
    m.total_pnl_pct = m.total_pnl / initial * 100

    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    m.profit_factor = gw / max(gl, 1e-8)

    pcts = [t.pnl_pct for t in trades]
    if len(pcts) > 1:
        mu, sd = np.mean(pcts), np.std(pcts)
        m.sharpe_ratio = float(mu / max(sd, 1e-8) * np.sqrt(252))
        dsd = np.std([r for r in pcts if r < 0]) or 1e-8
        m.sortino_ratio = float(mu / dsd * np.sqrt(252))

    if len(equity) > 1:
        vals = np.array([v for _, v in equity])
        peak = np.maximum.accumulate(vals)
        dd   = (peak - vals) / np.maximum(peak, 1e-8) * 100
        m.max_drawdown_pct = float(np.max(dd))

    m.calmar_ratio = m.total_pnl_pct / max(m.max_drawdown_pct, 1e-8)

    for t in trades:
        bh = t.barrier_hit
        if "CHANDELIER" in bh or "RUNNER" in bh:
            m.chandelier_hits += 1
        elif bh == "SL":
            m.sl_hits += 1
        else:
            m.timeout_hits += 1
        if t.tier == "A":   m.tier_a_trades += 1
        elif t.tier == "B": m.tier_b_trades += 1
        else:               m.tier_c_trades += 1

    m.window_count = len(win_mets)
    m.windows_profitable = sum(1 for w in win_mets if w["pnl"] > 0)
    if win_mets:
        sh = [w["sharpe"] for w in win_mets]
        m.window_sharpe_mean = float(np.mean(sh))
        m.window_sharpe_std  = float(np.std(sh))

    return m


# 
# Walk-Forward Simulation
# 

class WalkForwardSim:
    """
    Simulates the full QUANTA v12 pipeline on cached historical OHLCV.

    Unlike quanta_norse_year_sim (which operates on pre-extracted events),
    this sim processes every raw candle bar-by-bar with zero lookahead:
      - Thor check via live _SymbolBuffer + _thor_check (same code as production)
      - Thor score via score_thor_signal on causal OHLCV context
      - Thor v2 two-phase exit inline

    Usage:
        sim = WalkForwardSim()
        metrics = sim.run()
        sim.save_report("wf_results.json")
    """

    def __init__(self, initial_balance: float = 10_000.0,
                 cache_dir: str = "feather_cache"):
        self.initial_balance = initial_balance
        self.cache_dir       = cache_dir
        self.trades:      List[SimTrade]        = []
        self.equity:      List[Tuple]           = []
        self._win_mets:   List[dict]            = []
        self.metrics      = WFMetrics()
        self._thor_loaded = _load_thor_assets()   # try once at init

    #  public

    def run(self, symbols: Optional[List[str]] = None,
            verbose: bool = True) -> WFMetrics:
        t0 = time.time()
        scorer = "Thor CatBoost (AUC 0.81)" if self._thor_loaded else "OHLCV proxy"

        if verbose:
            print("\n" + "=" * 72)
            print(" QUANTA v12  WALK-FORWARD OOS SIMULATION")
            print("   Bar-by-bar replay    zero lookahead    live Thor detection")
            print(f"   Scorer: {scorer}")
            print("=" * 72)

        if symbols is None:
            symbols = self._default_symbols()

        all_klines = _load_cache(symbols, self.cache_dir)
        if not all_klines:
            print(" No cached data. Run mass_download_cache.py first.")
            return self.metrics

        total_c = max(len(v) for v in all_klines.values())
        train_c = _TRAIN_DAYS * _CANDLES_DAY
        test_c  = _TEST_DAYS  * _CANDLES_DAY
        step_c  = _STEP_DAYS  * _CANDLES_DAY
        n_wins  = max(0, (total_c - train_c) // step_c)

        if verbose:
            print(f"  Symbols loaded : {len(all_klines)}")
            print(f"  Data span      : {total_c / _CANDLES_DAY:.0f} days")
            print(f"  Train/Test/Step: {_TRAIN_DAYS}d / {_TEST_DAYS}d / {_STEP_DAYS}d")
            print(f"  Windows        : {n_wins}")
            print(f"  Initial balance: ${self.initial_balance:,.2f}")
            print(f"  Min Thor score : {_MIN_SCORE}")
            print(f"  Thor v2 exit   : bank {_BANK_ATR}ATR ({_BANK_FRAC*100:.0f}%) "
                  f" trail {_TRAIL_ATR}ATR")
            print(f"  Commission/slip: {_COMMISSION*10000:.0f}bps / {_SLIPPAGE*10000:.0f}bps")
            print("=" * 72)

        if n_wins == 0:
            print(" Not enough data for even one walk-forward window.")
            print(f"   Need {train_c + test_c:,} candles ({(_TRAIN_DAYS+_TEST_DAYS)} days), "
                  f"have {total_c:,}.")
            return self.metrics

        balance  = self.initial_balance
        position = train_c
        w_id     = 0
        all_trades: List[SimTrade] = []
        equity = [(0, balance)]

        while position + test_c <= total_c:
            test_start = position
            test_end   = min(position + test_c, total_c)
            warmup_start = max(0, position - train_c)

            if verbose:
                td = (position - _PURGE_GAP - warmup_start) / _CANDLES_DAY
                od = (test_end - test_start) / _CANDLES_DAY
                print(f"\n Window {w_id + 1:02d}  "
                      f"warmup [{warmup_start}:{position}] ({td:.0f}d)  "
                      f"test [{test_start}:{test_end}] ({od:.0f}d)")

            w_trades, balance, w_equity = self._run_window(
                all_klines, warmup_start, test_start, test_end,
                balance, w_id, verbose
            )
            all_trades.extend(w_trades)
            equity.extend(w_equity)          # per-trade snapshots for accurate DD
            equity.append((test_end, balance))

            wm = self._window_summary(w_trades, w_id)
            self._win_mets.append(wm)
            if verbose:
                wr = sum(1 for t in w_trades if t.pnl > 0) / max(len(w_trades), 1) * 100
                print(f"    {len(w_trades)} trades | {wr:.0f}% WR | "
                      f"P&L ${sum(t.pnl for t in w_trades):+.2f} | "
                      f"Balance ${balance:,.2f}")

            w_id     += 1
            position += step_c

        self.trades  = all_trades
        self.equity  = equity
        self.metrics = _compute_metrics(all_trades, equity, self.initial_balance,
                                        self._win_mets, t0)
        if verbose:
            self.print_report()
        return self.metrics

    #  window simulation

    def _run_window(self, all_klines: Dict[str, np.ndarray],
                    warmup_start: int, test_start: int, test_end: int,
                    balance: float, w_id: int, verbose: bool
                    ) -> Tuple[List[SimTrade], float]:

        pm = SimPositionManager(balance)

        # Only keep symbols with enough data for the full test window.
        sym_klines = {
            sym: kl for sym, kl in all_klines.items()
            if len(kl) >= test_end
        }

        # Per-symbol pre-computed indicator bundles for offline Thor inference.
        # _precompute_coin_indicators is cheap (O(N) on full klines), called once
        # per symbol per window. If the replay engine is unavailable the dict stays
        # empty and _sim_thor_score falls back to the OHLCV proxy.
        sym_precomp: Dict[str, dict] = {}
        if self._thor_loaded and _REPLAY_ENG is not None:
            for symbol, klines_np in sym_klines.items():
                try:
                    sym_precomp[symbol] = _REPLAY_ENG._precompute_coin_indicators(klines_np)
                except Exception as e:
                    log.debug("precompute failed %s: %s", symbol, e)

        # Pre-warm one _SymbolBuffer per symbol on the training bars so ATR
        # and avg-body estimates are stable before the test window opens.
        bufs: Dict[str, "_SymbolBuffer"] = {}
        for symbol, klines_np in sym_klines.items():
            buf = _SymbolBuffer()
            for wb in range(warmup_start, test_start):
                r = klines_np[wb]
                buf.push(float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5]))
            bufs[symbol] = buf

        # Bar-by-bar test replay — OUTER loop is TIME, INNER loop is SYMBOL.
        #
        # Critical: the old code looped symbol→bars (symbol outer, bars inner).
        # That meant ALL bars of Symbol A were processed before Symbol B was
        # touched, so:
        #   - _MAX_CONC limit was meaningless (only 1 symbol active at a time)
        #   - balance updates from A bled into B's sizing without concurrent
        #     position overlap ever applying
        #   - cooldown windows for the same symbol were correct per-symbol but
        #     never competed with other symbols' signals at the same bar
        #
        # Correct structure: advance ALL symbols one bar at a time together.
        for abs_bar in range(test_start, test_end):
            for symbol, klines_np in sym_klines.items():
                row = klines_np[abs_bar]
                o, h, l, c, v = (float(row[1]), float(row[2]),
                                  float(row[3]), float(row[4]), float(row[5]))
                buf = bufs[symbol]

                # 1. Tick EXISTING position for this symbol first
                pm.tick(symbol, abs_bar, h, l, c, w_id)

                # 2. Feed this bar into the rolling buffer
                buf.push(o, h, l, c, v)

                # 3. Run live Thor check on the just-closed bar
                if pm.can_open(symbol, abs_bar):
                    # ── Hour-of-day UTC filter (MAE hour-of-week heatmap) ────
                    # Skip entry at UTC hours with PF < 0.85: 0, 7, 10, 11, 12
                    bar_ts_ms = float(row[0])       # col[0] = open_time in ms
                    utc_hour  = int((bar_ts_ms / 3_600_000) % 24)
                    if utc_hour in _SKIP_UTC_HOURS:
                        continue

                    sig = _thor_check(buf)
                    if sig is not None:
                        precomp = sym_precomp.get(symbol)

                        # ── Gate A: wave_strength proxy (Norse coef +0.37) ──
                        # Skip signals where directional volume momentum is weak.
                        wave = _wave_strength_proxy(klines_np, abs_bar)
                        if wave < _WAVE_MIN:
                            continue

                        thor_score, feat_vec = _sim_thor_score(
                            sig, klines_np, abs_bar, precomp)

                        if thor_score < _MIN_SCORE:
                            continue

                        # ── Gate B: pre_impulse_r2 veto (Norse coef -0.24) ──
                        # Skip if price was already trending linearly before the
                        # impulse — the "breakout" is just exhaustion of a run.
                        if feat_vec is not None and len(feat_vec) > 272:
                            pre_r2 = float(feat_vec[272])
                            if np.isfinite(pre_r2) and pre_r2 > _PRE_R2_MAX:
                                continue

                        pm.open(symbol, abs_bar, c, buf.atr,
                                thor_score, sig.get("tier", "?"), w_id)

        # Force-close all remaining positions at window end
        pm.force_close_all(sym_klines, test_end - 1, w_id)

        return pm.trades, pm.balance, pm.equity_curve

    #  helpers 

    def _default_symbols(self) -> List[str]:
        # Scan feather_cache for all available 5m files — no cap
        try:
            fc_path = Path(self.cache_dir)
            syms = sorted({
                p.stem.replace("_5m", "")
                for p in fc_path.glob("*_5m.feather")
            })
            if syms:
                return syms
        except Exception:
            pass
        try:
            from QUANTA_selector import QuantaSelector
            syms = QuantaSelector().get_cached_coins_for_training(limit=500)
            if syms:
                return syms
        except Exception:
            pass
        return [
            "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","DOGEUSDT",
            "XRPUSDT","ADAUSDT","AVAXUSDT","MATICUSDT","LINKUSDT",
            "DOTUSDT","UNIUSDT","LTCUSDT","ATOMUSDT","NEARUSDT",
            "APTUSDT","ARBUSDT","OPUSDT","INJUSDT","SUIUSDT",
        ]

    def _window_summary(self, trades: List[SimTrade], w_id: int) -> dict:
        if not trades:
            return {"window_id": w_id, "sharpe": 0.0,
                    "trades": 0, "pnl": 0.0, "win_rate": 0.0}
        pcts  = [t.pnl_pct for t in trades]
        mu    = float(np.mean(pcts))
        sd    = float(np.std(pcts)) if len(pcts) > 1 else 1.0
        return {
            "window_id": w_id,
            "sharpe":    float(mu / max(sd, 1e-8) * np.sqrt(252)),
            "trades":    len(trades),
            "pnl":       float(sum(t.pnl for t in trades)),
            "win_rate":  float(sum(1 for t in trades if t.pnl > 0)
                               / max(len(trades), 1) * 100),
        }

    #  report 

    def print_report(self):
        m = self.metrics
        grade = ("A" if m.sharpe_ratio > 1.5 else
                 "B" if m.sharpe_ratio > 1.0 else
                 "C" if m.sharpe_ratio > 0.5 else
                 "D" if m.sharpe_ratio > 0   else "F")
        print("\n" + "=" * 72)
        print(" WALK-FORWARD RESULTS  QUANTA v12 THOR")
        print("=" * 72)
        print(f"  Windows  : {m.window_count}  ({m.windows_profitable} profitable)")
        print(f"  Runtime  : {m.runtime_sec:.1f}s")
        print()
        print(f" PERFORMANCE ")
        print(f"  Trades    : {m.total_trades}  "
              f"(A:{m.tier_a_trades} / B:{m.tier_b_trades} / C:{m.tier_c_trades})")
        print(f"  Win Rate  : {m.win_rate:.1f}%")
        print(f"  Total P&L : ${m.total_pnl:+,.2f}  ({m.total_pnl_pct:+.1f}%)")
        print(f"  PF        : {m.profit_factor:.2f}")
        print()
        print(f" RISK-ADJUSTED ")
        print(f"  Sharpe    : {m.sharpe_ratio:.2f}  [{grade}]")
        print(f"  Sortino   : {m.sortino_ratio:.2f}")
        print(f"  Calmar    : {m.calmar_ratio:.2f}")
        print(f"  Max DD    : {m.max_drawdown_pct:.1f}%")
        print()
        print(f" EXITS ")
        print(f"  Chandelier/Runner : {m.chandelier_hits}")
        print(f"  SL                : {m.sl_hits}")
        print(f"  Timeout/WindowEnd : {m.timeout_hits}")

        # Pyramid breakdown
        if self.trades:
            l1 = sum(1 for t in self.trades if t.layers == 1)
            l2 = sum(1 for t in self.trades if t.layers == 2)
            l3 = sum(1 for t in self.trades if t.layers >= 3)
            add1_pnl = sum(t.add1_pnl for t in self.trades)
            add2_pnl = sum(t.add2_pnl for t in self.trades)
            print()
            print(f" PYRAMID LAYERS ")
            print(f"  L1 only : {l1} trades")
            print(f"  L2 add  : {l2} trades | add PnL ${add1_pnl:+,.2f}")
            print(f"  L3 recov: {l3} trades | recovery PnL ${add2_pnl:+,.2f}")
        print()
        print(f" WINDOW STABILITY ")
        print(f"  Sharpe/window : {m.window_sharpe_mean:.2f}  {m.window_sharpe_std:.2f}")
        print("=" * 72)

    def save_report(self, filepath: str = "wf_sim_results.json"):
        out = {
            "metrics":        asdict(self.metrics),
            "equity_curve":   self.equity,
            "window_metrics": self._win_mets,
            "trades": [
                {
                    "symbol":   t.symbol,
                    "entry_bar": t.entry_bar,
                    "exit_bar":  t.exit_bar,
                    "entry":    round(t.entry_price, 6),
                    "exit":     round(t.exit_price, 6),
                    "pnl":      round(t.pnl, 4),
                    "pnl_pct":  round(t.pnl_pct, 4),
                    "barrier":  t.barrier_hit,
                    "score":    round(t.score, 2),
                    "tier":     t.tier,
                    "window":   t.window_id,
                    "layers":   t.layers,
                    "add1_pnl": round(t.add1_pnl, 4),
                    "add2_pnl": round(t.add2_pnl, 4),
                }
                for t in self.trades
            ],
        }
        with open(filepath, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f" Saved  {filepath}")

    def save_run_folder(self, base_dir: str = "wf_runs") -> str:
        """
        Save a timestamped run folder matching Norse sim output structure.

        wf_runs/<YYYYMMDD_HHMMSS>/
          WF_SIM_REPORT_<ts>.md           — main markdown report
          wf_trades_<ts>.csv              — all trades (Norse-compatible columns)
          wf_wins_<ts>.csv                — winning trades only
          wf_losses_<ts>.csv              — losing trades only
          wf_window_stats_<ts>.csv        — per-window breakdown
          wf_score_distribution_<ts>.csv  — score bucket analysis
          wf_summary_<ts>.csv             — one-row summary
          wf_sim_results_<ts>.json        — full JSON dump
        """
        import csv as _csv
        from collections import defaultdict

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        m  = self.metrics
        trades = self.trades
        initial = self.initial_balance
        final   = initial + m.total_pnl

        # ── helpers ──────────────────────────────────────────────────────────
        def _write_csv(filename: str, rows: list, fieldnames: list):
            p = run_dir / filename
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = _csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            return p.name

        def _barrier_label(b: str) -> str:
            if "CHANDELIER" in b or "RUNNER" in b:
                return "TP"
            if b == "SL":
                return "SL"
            return "TIMEOUT"

        # ── 1. Trades CSV (Norse-compatible) ─────────────────────────────────
        trade_fields = [
            "agent","symbol","direction","entry_bar","exit_bar",
            "entry_price","exit_price","entry_notional","realized_atr",
            "gross_pnl","net_pnl","label","score","tier",
            "hold_bars","barrier","window_id",
        ]
        trade_rows, win_rows, loss_rows = [], [], []
        for t in trades:
            hold  = t.exit_bar - t.entry_bar
            label = _barrier_label(t.barrier_hit)
            notional = t.entry_price * (abs(t.pnl) / max(abs(t.pnl_pct / 100), 1e-8))
            row = {
                "agent": "Thor", "symbol": t.symbol, "direction": "BULLISH",
                "entry_bar": t.entry_bar, "exit_bar": t.exit_bar,
                "entry_price": round(t.entry_price, 8),
                "exit_price":  round(t.exit_price,  8),
                "entry_notional": round(notional, 4),
                "realized_atr": "",
                "gross_pnl": round(t.pnl, 4),
                "net_pnl":   round(t.pnl, 4),
                "label": label, "score": round(t.score, 4),
                "tier":  t.tier, "hold_bars": hold,
                "barrier": t.barrier_hit, "window_id": t.window_id,
            }
            trade_rows.append(row)
            (win_rows if t.pnl > 0 else loss_rows).append(row)

        _write_csv(f"wf_trades_{ts}.csv",  trade_rows, trade_fields)
        _write_csv(f"wf_wins_{ts}.csv",    win_rows,   trade_fields)
        _write_csv(f"wf_losses_{ts}.csv",  loss_rows,  trade_fields)

        # ── 2. Window stats CSV ───────────────────────────────────────────────
        win_fields = ["window","trades","wins","losses","win_rate_pct",
                      "net_pnl","sharpe","balance_end"]
        balance = initial
        win_rows_stat = []
        w_groups: dict = defaultdict(list)
        for t in trades:
            w_groups[t.window_id].append(t)
        for wm in self._win_mets:
            wid  = wm["window_id"]
            balance += wm["pnl"]
            wts  = w_groups[wid]
            wins = sum(1 for t in wts if t.pnl > 0)
            win_rows_stat.append({
                "window": wid + 1,
                "trades": wm["trades"],
                "wins": wins,
                "losses": wm["trades"] - wins,
                "win_rate_pct": round(wm["win_rate"], 2),
                "net_pnl": round(wm["pnl"], 4),
                "sharpe": round(wm["sharpe"], 4),
                "balance_end": round(balance, 4),
            })
        _write_csv(f"wf_window_stats_{ts}.csv", win_rows_stat, win_fields)

        # ── 3. Score distribution CSV ─────────────────────────────────────────
        score_fields = ["bucket_lo","bucket_hi","trades","wins","win_rate_pct",
                        "net_pnl","avg_pnl"]
        buckets = [(68,72),(72,76),(76,80),(80,85),(85,90),(90,100)]
        score_rows = []
        for lo, hi in buckets:
            bt = [t for t in trades if lo <= t.score < hi]
            if not bt:
                continue
            bw = sum(1 for t in bt if t.pnl > 0)
            score_rows.append({
                "bucket_lo": lo, "bucket_hi": hi,
                "trades": len(bt), "wins": bw,
                "win_rate_pct": round(bw / len(bt) * 100, 2),
                "net_pnl": round(sum(t.pnl for t in bt), 4),
                "avg_pnl": round(sum(t.pnl for t in bt) / len(bt), 4),
            })
        _write_csv(f"wf_score_distribution_{ts}.csv", score_rows, score_fields)

        # ── 4. Summary CSV (one row) ──────────────────────────────────────────
        summary_fields = [
            "run_id","timestamp","initial_capital","final_capital","growth_pct",
            "trades","win_rate_pct","profit_factor","sharpe","sortino",
            "max_dd_pct","windows","windows_profitable",
            "tier_a","tier_b","tier_c","chandelier_exits","sl_exits",
            "train_days","test_days","symbols","min_score",
        ]
        _write_csv(f"wf_summary_{ts}.csv", [{
            "run_id": ts, "timestamp": datetime.now().isoformat(),
            "initial_capital": initial, "final_capital": round(final, 2),
            "growth_pct": round(m.total_pnl_pct, 2),
            "trades": m.total_trades, "win_rate_pct": round(m.win_rate, 2),
            "profit_factor": round(m.profit_factor, 4),
            "sharpe": round(m.sharpe_ratio, 4),
            "sortino": round(m.sortino_ratio, 4),
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "windows": m.window_count,
            "windows_profitable": m.windows_profitable,
            "tier_a": m.tier_a_trades, "tier_b": m.tier_b_trades,
            "tier_c": m.tier_c_trades,
            "chandelier_exits": m.chandelier_hits,
            "sl_exits": m.sl_hits,
            "train_days": _TRAIN_DAYS, "test_days": _TEST_DAYS,
            "symbols": len({t.symbol for t in trades}),
            "min_score": _MIN_SCORE,
        }], summary_fields)

        # ── 5. JSON dump ──────────────────────────────────────────────────────
        json_path = run_dir / f"wf_sim_results_{ts}.json"
        self.save_report(str(json_path))

        # ── 6. Markdown report ────────────────────────────────────────────────
        gw = sum(t.pnl for t in trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in trades if t.pnl < 0))
        avg_hold = (sum(t.exit_bar - t.entry_bar for t in trades)
                    / max(len(trades), 1))
        tp_count = sum(1 for t in trades if _barrier_label(t.barrier_hit) == "TP")
        sl_count = sum(1 for t in trades if _barrier_label(t.barrier_hit) == "SL")
        to_count = m.timeout_hits

        score_vals = [t.score for t in trades] or [0]
        import numpy as _np

        md_lines = [
            "# QUANTA v12 Walk-Forward Simulation Report",
            "",
            "## Setup",
            f"- Run ID: `{ts}`",
            f"- Run timestamp: `{datetime.now().isoformat()}`",
            f"- Initial capital: `${initial:,.2f}`",
            f"- OOS windows: `{m.window_count}` × {_TEST_DAYS}d  (train {_TRAIN_DAYS}d, step {_STEP_DAYS}d)",
            f"- Symbols traded: `{len({t.symbol for t in trades})}`",
            f"- Score floor: `{_MIN_SCORE}`",
            f"- Thor v2 exit: `bank {_BANK_ATR} ATR @ {int(_BANK_FRAC*100)}%` + `trail {_TRAIL_ATR} ATR`",
            f"- SL: `{_SL_ATR} ATR`",
            f"- Commission/slippage: `{int(_COMMISSION*10000)}bps / {int(_SLIPPAGE*10000)}bps`",
            f"- Entry quality gates: wave_strength ≥ {_WAVE_MIN}, pre_impulse_r2 ≤ {_PRE_R2_MAX}",
            f"- Risk scaling: `{_RISK_PCT*100:.1f}%` (score={_MIN_SCORE}) → "
            f"`{float(getattr(_ev,'thor_compound_max_loss_pct',3.0)):.1f}%` (score=100), continuous",
            "",
            "## Portfolio Result",
            f"- Final capital: `${final:,.2f}`",
            f"- Growth: `{m.total_pnl_pct:+.2f}%`",
            f"- Max drawdown: `{m.max_drawdown_pct:.2f}%`",
            f"- Executed trades: `{m.total_trades}`",
            "",
            "## Performance Metrics",
            f"| Metric | Value |",
            f"| --- | --- |",
            f"| Win Rate | {m.win_rate:.2f}% |",
            f"| Profit Factor | {m.profit_factor:.3f} |",
            f"| Sharpe Ratio | {m.sharpe_ratio:.3f} |",
            f"| Sortino Ratio | {m.sortino_ratio:.3f} |",
            f"| Max Drawdown | {m.max_drawdown_pct:.2f}% |",
            f"| Gross Wins | ${gw:,.2f} |",
            f"| Gross Losses | ${gl:,.2f} |",
            f"| Avg Hold (bars) | {avg_hold:.1f} |",
            "",
            "## Agent Statistics",
            f"| Agent | Trades | TP | SL | TIMEOUT | TP Rate | Net PnL | Avg PnL |",
            f"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Thor | {m.total_trades} | {tp_count} | {sl_count} | {to_count} "
            f"| {tp_count/max(m.total_trades,1)*100:.2f}% "
            f"| ${m.total_pnl:+,.2f} "
            f"| ${m.total_pnl/max(m.total_trades,1):+,.2f} |",
            "",
            "## Tier Breakdown",
            f"| Tier | Trades | Win Rate | Net PnL |",
            f"| --- | ---: | ---: | ---: |",
        ]

        from collections import defaultdict as _dd
        tier_map = _dd(lambda: {"n":0,"wins":0,"pnl":0.0})
        for t in trades:
            tier_map[t.tier]["n"]   += 1
            tier_map[t.tier]["pnl"] += t.pnl
            if t.pnl > 0: tier_map[t.tier]["wins"] += 1
        for tier in ["A","B","C"]:
            s = tier_map[tier]
            if s["n"]:
                md_lines.append(
                    f"| {tier} | {s['n']} | {s['wins']/s['n']*100:.1f}% "
                    f"| ${s['pnl']:+,.2f} |")

        md_lines += [
            "",
            "## Score Distribution",
            "| Score Bucket | Trades | Win Rate | Net PnL | Avg PnL |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for lo, hi in buckets:
            bt = [t for t in trades if lo <= t.score < hi]
            if not bt: continue
            bw  = sum(1 for t in bt if t.pnl > 0)
            bpnl = sum(t.pnl for t in bt)
            md_lines.append(
                f"| [{lo}–{hi}) | {len(bt)} | {bw/len(bt)*100:.1f}% "
                f"| ${bpnl:+,.2f} | ${bpnl/len(bt):+,.2f} |")

        md_lines += [
            "",
            f"- Score min/max/mean: "
            f"`{min(score_vals):.1f}` / `{max(score_vals):.1f}` / "
            f"`{_np.mean(score_vals):.1f}`",
            "",
            "## Window Stability",
            f"| Window | Trades | WR% | Net PnL | Sharpe |",
            f"| --- | ---: | ---: | ---: | ---: |",
        ]
        for wm in self._win_mets:
            md_lines.append(
                f"| {wm['window_id']+1} | {wm['trades']} "
                f"| {wm['win_rate']:.1f}% "
                f"| ${wm['pnl']:+,.2f} "
                f"| {wm['sharpe']:.2f} |")

        md_lines += [
            "",
            f"- Mean window Sharpe: `{m.window_sharpe_mean:.2f}` ± `{m.window_sharpe_std:.2f}`",
            f"- Windows profitable: `{m.windows_profitable}/{m.window_count}`",
            "",
            "## Exit Analysis",
            f"| Exit Type | Count | Net PnL |",
            f"| --- | ---: | ---: |",
            f"| Chandelier/Runner (TP) | {m.chandelier_hits} | ${gw:+,.2f} (approx) |",
            f"| Stop Loss (SL) | {m.sl_hits} | ${-gl:+,.2f} (approx) |",
            f"| Timeout | {m.timeout_hits} | — |",
            "",
            "## Output Files",
        ]
        for fname in sorted(run_dir.iterdir()):
            md_lines.append(f"- `{fname.name}`")

        md_path = run_dir / f"WF_SIM_REPORT_{ts}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        print(f" Run folder saved → {run_dir}")
        for f in sorted(run_dir.iterdir()):
            print(f"   {f.name}  ({f.stat().st_size//1024}KB)")
        return str(run_dir)


# 
# Entry point
# 

def main():
    print("=" * 72)
    print(" QUANTA v12  WALK-FORWARD SIM")
    print("   Live Thor detection    Thor scoring    Thor v2 exit")
    print("=" * 72)

    sim = WalkForwardSim(
        initial_balance=10_000.0,
        cache_dir="feather_cache",
    )
    sim.run(verbose=True)
    sim.save_report("wf_sim_results.json")   # legacy flat file (backward compat)
    sim.save_run_folder("wf_runs")           # Norse-style timestamped folder


if __name__ == "__main__":
    main()
