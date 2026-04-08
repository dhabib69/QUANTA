"""
🔧 QUANTA Trading Core — Constants, Utilities, PaperTrading, RLMemory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Consolidated from: QUANTA_constants.py + QUANTA_utils.py + QUANTA_trading_core.py
"""

import sys
import os
import time
import math
import csv
import json
import logging
import warnings
import threading
import numpy as np
import pandas as pd
import urllib3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections import deque

# =================== RESEARCH-BACKED CONSTANTS ===================

# Technical Indicators (Wilder 1978, Appel 1979, Bollinger 2001)
RSI_PERIOD = 14
MACD_FAST = 8
MACD_SLOW = 21
MACD_SIGNAL = 5
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
ADX_PERIOD = 14
STOCH_PERIOD = 14

# ML Thresholds (López de Prado 2018)
DIRECTION_THRESHOLD = 0.12
MIN_CONFIDENCE_RL = 60
MIN_CONFIDENCE_ALERT = 70

# Training Parameters
RL_RETRAIN_THRESHOLD = 500
RL_OUTCOME_WINDOW = 3600
CATASTROPHIC_FORGETTING_BUFFER_RATIO = 0.8

# Signal Quality Constants
MAX_STREAK = 20
MAX_MAGNITUDE = 10.0
RL_ADD_COOLDOWN = 1800
CUSUM_THRESHOLD = 0.02

# Triple Barrier Method (López de Prado 2018 Ch.3)
# Research-backed for crypto: SL at 1.5× ATR survives noise,
# TP1=1:1, TP2=2:1, TP3=3:1 risk/reward
SL_RATIO  = 1.5   # ATR multiplier for stop-loss
TP1_RATIO = 1.0   # ATR multiplier for TP1 (tight — easy hit, partial close 33%)
TP2_RATIO = 2.0   # ATR multiplier for TP2 (partial close 33%, trail SL to TP1)
TP3_RATIO = 3.5   # ATR multiplier for TP3 (remaining 34%, full win)

# 3-Tier TP Weights
# TP1 = 1:1 R:R (barely profitable after fees) — low weight
# SL should outweigh TP1 so the model learns more from mistakes than marginal wins
TP1_WEIGHT = 0.3
TP2_WEIGHT = 2.5
TP3_WEIGHT = 5.0
SL_WEIGHT  = 2.0   # was 0.5 — wrong predictions now penalised 6× harder than TP1

# Time Windows & Scanning
HISTORICAL_DAYS = 180
SCAN_INTERVAL = 90
STATS_INTERVAL = 100

# Market Thresholds
WHALE_THRESHOLD = 500000
MA_SHORT = 20
MA_LONG = 50

# Specialist Model Weights
WEIGHT_FOUNDATION = 0.5
WEIGHT_HUNTER = 0.3
WEIGHT_ANCHOR = 0.2

# Feature Engineering Constants moved to Config

# GPU/CPU Configuration
USE_GPU = True

# PPO Constants (Tuned for 5.0x Veto Variance & Contextual Bandits)
PPO_HIDDEN_DIM = 256        # Expanded to 256 for deeper 268-dim feature vector absorption
PPO_LR = 0.000025           # Lowered to 2.5e-5 to prevent 5.0x reward gradients from exploding weights
PPO_EPOCHS = 4              # Standard
PPO_BATCH_SIZE = 1024       # Doubled to 1024 to smooth out highly volatile crypto tick noise
PPO_CLIP = 0.08             # Tightened to 8% to strictly prevent catastrophic forgetting on outlier trades
PPO_GAMMA = 0.99            # Standard
PPO_GAE_LAMBDA = 0.95       # Standard
PPO_ENTROPY_COEF = 0.030    # Increased to 3.0% to force the agent to keep trying Vetoes even after getting burned
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
DSR_ADAPTATION_RATE = 0.02
DSR_EPS = 1e-8

# TFT Constants (Lim et al. 2021)
TFT_HIDDEN_SIZE = 64
TFT_NUM_HEADS = 4
TFT_DROPOUT = 0.1
TFT_SEQ_LENGTH = 12 # 1 hour of 5-minute candles to provide Odin with actual temporal context
TFT_TRAIN_EPOCHS = 10
TFT_ENSEMBLE_WEIGHT = 0.0  # Obsolete
CAT_ENSEMBLE_WEIGHT = 1.0  # Obsolete

# =================== DUAL-BRAIN ARCHITECTURE CONSTANTS ===================
#
# EWC_PENALTY_MULTIPLIER (λ)
#   Kirkpatrick et al. (2017) "Overcoming Catastrophic Forgetting in Neural Networks"
#   Schwarz et al. (2018) "Progress & Compress: Continual Learning" (DeepMind)
#   For small binary-classification LSTMs, λ ∈ [100, 1000] is the effective
#   range. Values >1000 freeze the network (no new learning). Values <100
#   provide insufficient protection. 400 is the geometric mean of the
#   effective range, validated by Schwarz 2018 on sequential Atari tasks.
EWC_PENALTY_MULTIPLIER = 400.0

# ODIN_VETO_THRESHOLD
#   Standard anomaly detection practice: 97th percentile threshold
#   (Chalapathy & Chawla 2019 "Deep Learning for Anomaly Detection: A Survey")
#   For binary softmax outputs, P(crash) > 0.80 corresponds to the empirical
#   97th percentile of reconstruction error in LSTM autoencoders on financial
#   time series (Malhotra et al. 2015 "Long Short Term Memory Networks for
#   Anomaly Detection in Time Series").
ODIN_VETO_THRESHOLD = 0.80

# REPLAY_BUFFER_SIZE
#   Mnih et al. (2015) "Human-level control through deep RL" (DeepMind)
#   Original DQN used 1M transitions. For tabular/tree models (CatBoost),
#   Isele & Cosgun (2018) "Selective Experience Replay" showed 1K-10K is
#   optimal for non-neural learners. 5000 balances memory diversity with
#   staleness risk in non-stationary crypto markets.
REPLAY_BUFFER_SIZE = 5000

# REPLAY_SAMPLE_SIZE
#   Schaul et al. (2016) "Prioritized Experience Replay" (DeepMind)
#   Recommended α=0.6 priority exponent with ~20% buffer sampling ratio
#   per training step. 1000/5000 = 20% matches this recommendation.
#   S&P 500 case study: PER with 20% sampling improved Sharpe by 22%.
REPLAY_SAMPLE_SIZE = 1000

# RL_QUOTA_CHECK_INTERVAL (seconds)
#   Bot operates on 5-minute candles. Checking the RL buffer every 60s
#   ensures we detect quota completion within 1 candle of it happening,
#   without wasting CPU cycles on sub-candle polling.
RL_QUOTA_CHECK_INTERVAL = 60

# =================== CATBOOST SHARED TRAINING CONSTANTS ===================
#
# CATBOOST_BORDER_COUNT
#   CatBoost Official Docs: "Set border_count to 254 for best quality."
#   128 is the GPU default for speed. 254 is the CPU default for accuracy.
#   Since we prioritize prediction quality over training speed, use 254.
CATBOOST_BORDER_COUNT = 254

# CATBOOST_EARLY_STOPPING
#   Mayr et al. (2012) "The Evolution of Boosting Algorithms"
#   Research shows early stopping halves training time with no accuracy loss.
#   150 rounds is standard for 1000-1500 iteration CatBoost runs.
#   Too low (<50) causes underfitting; too high (>300) wastes compute.
CATBOOST_EARLY_STOPPING = 150

# CATBOOST_HARD_NEG_BOOST
#   Bengio (2009) "Curriculum Learning" + Shrivastava et al. (2016)
#   "Training Region-based Object Detectors with Online Hard Example Mining"
#   3x is the standard hard negative weight multiplier. Medical imaging
#   literature uses 3-5x. Financial data is less extreme, so 3x is ideal.
CATBOOST_HARD_NEG_BOOST = 3.0

# CATBOOST_VAL_SPLIT
#   López de Prado (2018) AFML Ch.7: "Never fit scaler on full data."
#   80/20 train/val split is the Pareto-optimal ratio for datasets
#   with 1K-100K samples (Hastie, Tibshirani & Friedman 2009 Ch.7).
CATBOOST_VAL_SPLIT = 0.2

# NOTE: Per-specialist hyperparams (iterations, lr, depth, l2, bagging_temp)
# are INTENTIONALLY diverse across the 7 agents.
# Krogh & Vedelsby (1994): ensemble_error = avg_error − diversity.
# Uniform hyperparams destroy ensemble diversity and degrade performance.
# These are NOT magic numbers; they are engineered for disagreement.


# =================== DUAL LOGGER ===================

class DualLogger:
    def __init__(self, filename):
        self.log_file = open(filename, 'a', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, message):
        try:
            self.stdout.write(message)
        except UnicodeEncodeError:
            self.stdout.write(message.encode('ascii', 'replace').decode('ascii'))
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
    def isatty(self):
        return self.stdout.isatty()


@contextmanager
def silence_sklearn_warnings():
    """Suppress sklearn warnings but allow progress output"""
    old_level = logging.getLogger().level
    try:
        logging.getLogger().setLevel(logging.CRITICAL)
        warnings.filterwarnings('ignore')
        yield
    finally:
        logging.getLogger().setLevel(old_level)


def configure_global_warnings():
    """Global configuration to suppress noisy scientific library warnings."""
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['SKLEARN_SKIP_PARALLEL_WARNING'] = '1'
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    np.seterr(all='ignore')
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='invalid value encountered')
    warnings.filterwarnings('ignore', message='divide by zero')
    warnings.filterwarnings('ignore', module='numpy')
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    urllib3.disable_warnings()
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
    
    # Global SSL bypass for Windows proxy issues
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# Execute warning suppression globally on import
configure_global_warnings()


# =================== PAPER TRADING ===================
from quanta_exchange import SmartExecutor

class PaperTrading:
    STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_trading_state.json')

    def __init__(self, initial_balance=10000, bot=None):
        self.initial_balance = initial_balance
        self.positions = {}
        self.history = []
        self.total_pnl = 0
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.trade_log_file = "trades.csv"

        # Restore persistent state if available
        self.balance = initial_balance
        self._load_state()

        # We simulate TWAP with the real SmartExecutor class, passing self as the bot mock
        from quanta_exchange import BinanceAPIEnhanced
        from quanta_config import Config
        cfg = bot.cfg if bot and hasattr(bot, 'cfg') else (Config() if callable(Config) else Config)
        self.executor = SmartExecutor(exchange=BinanceAPIEnhanced(cfg), bot=self)

        # v11.4: Risk Manager + Paper Trading Logger
        from quanta_risk_manager import RiskManager
        from quanta_paper_trading import PaperTradingLogger
        # Pass flat_all callback so circuit breaker can close positions immediately
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            flat_all_callback=self.flat_all
        )
        self.paper_logger = PaperTradingLogger()
        self._ml_engine = None  # v11.5: Set by bot for Brier score tracking
        self._rl_memory = None   # v11.5b: Set by bot to link trades → RL memory

        # Base snapshot so graph doesn't start blank
        self.paper_logger.snapshot(
            balance=self.balance, 
            open_positions=0, 
            daily_pnl=self.total_pnl
        )

        # BS implied vol tracking: per-symbol rolling deque of bars-to-hit (maxlen=50)
        # Used to compute bs_implied_vol_ratio feature (index 277). Starts empty → neutral.
        from collections import deque
        self._bs_bars_to_hit: dict = {}  # symbol -> deque(maxlen=50)

        self._init_log()

    def _load_state(self):
        """Restore balance, stats, and positions from disk so paper trading survives restarts."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.balance = state.get('balance', self.initial_balance)
                self.total_pnl = state.get('total_pnl', 0)
                self.total_trades = state.get('total_trades', 0)
                self.total_wins = state.get('total_wins', 0)
                self.total_losses = state.get('total_losses', 0)
                self.initial_balance = state.get('initial_balance', self.initial_balance)
                
                # Restore positions with datetime conversion
                positions = state.get('positions', {})
                for sym, pos in positions.items():
                    if 'time' in pos and isinstance(pos['time'], str):
                        try:
                            pos['time'] = datetime.fromisoformat(pos['time'])
                        except ValueError:
                            pos['time'] = datetime.now()
                    self.positions[sym] = pos

                logging.info(f"Paper trading state restored: balance=${self.balance:.2f}, "
                             f"PnL=${self.total_pnl:.2f}, trades={self.total_trades}, "
                             f"active={len(self.positions)}")
        except Exception as e:
            logging.warning(f"Could not load paper trading state: {e}")

    def _save_state(self):
        """Persist balance, stats, and active positions to disk."""
        try:
            # Serialize positions with datetime to ISO
            serialized_pos = {}
            for sym, pos in self.positions.items():
                p = pos.copy()
                if 'time' in p and isinstance(p['time'], datetime):
                    p['time'] = p['time'].isoformat()
                # Convert numpy arrays to lists for JSON serialization
                if 'specialist_probs' in p and isinstance(p['specialist_probs'], np.ndarray):
                    p['specialist_probs'] = p['specialist_probs'].tolist()
                serialized_pos[sym] = p

            state = {
                'balance': round(self.balance, 6),
                'initial_balance': self.initial_balance,
                'total_pnl': round(self.total_pnl, 6),
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'positions': serialized_pos,
                'last_updated': datetime.now().isoformat(),
            }
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save paper trading state: {e}")

    def _init_log(self):
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['symbol', 'entry_price', 'exit_price', 'direction', 'pnl', 'time'])

    def open_position(self, symbol, entry_price, direction, confidence, atr_percent,
                      ppo_size_mult=1.0, barrier_rr=2.0, bs_edge=None, bs_prob=None,
                      specialist=None, exit_profile=None, timeout_bars=None):
        """
        Open a paper position.

        ppo_size_mult ∈ [0.25, 2.0] — PPO position size oracle (v11.5b).
        PPO no longer gates the trade; it scales notional after Kelly.
        Agrees+confident → 2×, HOLD → 0.5–1×, Contradicts → 0.25–0.5×.
        Capped so final notional never exceeds MAX_RISK (2.5%) of balance.

        barrier_rr — TP/SL distance ratio for this specialist (Hull Ch.26).
                     Replaces hardcoded 2.0; passed from bot per specialist config.
        bs_edge    — Live BS/Kou edge minus random-walk baseline P(TP).
                     If < 0.02 (< 2% alpha), probability is penalised.
        bs_prob    — Live BS/Kou TP-before-SL probability. Blended with ML
                     confidence before Kelly sizing so execution math directly
                     affects capital allocation.
        specialist — dominant specialist key for this trade.
        exit_profile — optional specialist-specific exit config persisted on the position.
        timeout_bars — optional specialist-specific timeout metadata.
        """
        if symbol in self.positions:
            return

        # Blend ML confidence with the live barrier probability before sizing.
        prob_ml = max(0.51, min(0.99, confidence / 100.0))
        prob = prob_ml
        if bs_prob is not None:
            try:
                prob_bs = max(0.01, min(0.99, float(bs_prob)))
                prob = max(0.51, min(0.99, 0.65 * prob_ml + 0.35 * prob_bs))
            except Exception:
                prob = prob_ml

        # BS Edge filter (Hull Ch.26 / Darling-Siegert): penalise when live barrier math
        # barely beats the specialist's zero-drift baseline.
        # Threshold 0.02 (2%) — local heuristic, similar to DIRECTION_THRESHOLD=0.12.
        if bs_edge is not None and bs_edge < 0.02:
            edge_penalty = max(0.5, min(1.0, bs_edge / 0.10))
            prob = max(0.51, prob * edge_penalty)

        # Kelly Criterion: f* = p - (q / b)
        # b = barrier_rr: per-specialist TP/SL distance ratio (dynamic, replaces hardcoded 2.0)
        b = max(0.5, float(barrier_rr))
        kelly_fraction = max(0.0, prob - ((1.0 - prob) / b))

        # Scale kelly to real risk: Half-Kelly in [0.3%, 2.5%] range
        MIN_RISK = 0.003  # 0.3% minimum per trade
        MAX_RISK = 0.025  # 2.5% maximum per trade
        risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (kelly_fraction / 1.0)
        risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))

        stop_distance = atr_percent * 1.5
        size_units = (self.balance * risk_percent) / max(0.001, (stop_distance / 100)) / max(1e-8, entry_price)
        total_notional = size_units * entry_price

        # ── PPO SIZE ORACLE (v11.5b) ──
        # Apply PPO multiplier BEFORE risk manager gate.
        # Cap: PPO can never push notional above MAX_RISK of balance.
        ppo_size_mult = float(max(0.25, min(2.0, ppo_size_mult)))  # clamp defensively
        total_notional *= ppo_size_mult
        size_units     *= ppo_size_mult
        # Hard cap: never exceed MAX_RISK% of balance regardless of PPO
        max_notional = self.balance * MAX_RISK / max(0.001, stop_distance / 100)
        if total_notional > max_notional:
            total_notional = max_notional
            size_units     = total_notional / max(1e-8, entry_price)

        # ── v11.4 RISK MANAGER GATE ──
        allowed, reason = self.risk_manager.pre_trade_check(
            symbol, total_notional, direction, self.balance, self.positions
        )
        if not allowed:
            self.paper_logger.log_decision(
                symbol=symbol, direction=direction, confidence=confidence,
                action_taken='RISK_BLOCKED', risk_result=reason
            )
            return

        # Apply streak-based size reduction
        size_mult = self.risk_manager.get_size_multiplier()
        if size_mult < 1.0:
            total_notional *= size_mult
            size_units *= size_mult

        # Route through TWAP if large order (>$500), otherwise use market order simulation
        if total_notional >= 500:
            self.executor.execute_twap(symbol, direction, total_notional, entry_price,
                                       steps=5, duration_mins=2.0,
                                       confidence=confidence, atr_percent=atr_percent,
                                       specialist=specialist, exit_profile=exit_profile,
                                       timeout_bars=timeout_bars)
            # Metadata is now handled inside _execute_market_order via executor slices
        else:
            self._execute_market_order(symbol, direction, total_notional, entry_price,
                                       confidence, atr_percent,
                                       specialist=specialist, exit_profile=exit_profile,
                                       timeout_bars=timeout_bars)

        # Track in risk manager
        self.risk_manager.on_trade_opened(symbol, total_notional)
        self._save_state()  # Ensure open positions are persisted immediately

    def _execute_market_order(self, symbol, side, chunk_notional, step_price, confidence=0, atr_percent=0,
                              specialist=None, exit_profile=None, timeout_bars=None):
        """Simulate execution of a market order chunk with realistic cost model."""
        from quanta_config import Config as _cfg
        _bt = getattr(_cfg, 'backtest', None)
        commission_bps = getattr(_bt, 'commission_bps', 4.0)   # 0.04% per side
        slippage_bps   = getattr(_bt, 'slippage_bps',   2.0)   # 0.02% per side (taker slippage)

        # Entry friction: slippage moves price AGAINST us, commission deducted from balance.
        # LONG:  we pay slightly more (price * (1 + slip))
        # SHORT: we receive slightly less (price * (1 - slip))
        slip_factor = slippage_bps / 10000.0
        if side == 'BULLISH':
            fill_price = step_price * (1.0 + slip_factor)
        else:
            fill_price = step_price * (1.0 - slip_factor)

        size = chunk_notional / fill_price
        entry_commission = chunk_notional * (commission_bps / 10000.0)
        self.balance -= entry_commission  # deduct commission immediately

        if symbol in self.positions:
            # Average down / scale in
            pos = self.positions[symbol]
            new_size = pos['size'] + size
            new_entry = ((pos['entry'] * pos['size']) + (fill_price * size)) / new_size
            self.positions[symbol]['entry'] = new_entry
            self.positions[symbol]['size'] = new_size
            # Update metadata if provided
            if confidence > 0: self.positions[symbol]['confidence'] = confidence
            if atr_percent > 0: self.positions[symbol]['atr_percent'] = atr_percent
            if specialist:
                self.positions[symbol]['specialist'] = specialist
            if exit_profile:
                self.positions[symbol]['exit_profile'] = exit_profile
            if timeout_bars is not None:
                self.positions[symbol]['timeout_bars'] = timeout_bars
        else:
            # Compute ATR-based barrier prices at entry
            atr_move = fill_price * (atr_percent / 100.0)
            specialist_key = str(specialist or "").lower()
            exit_profile = exit_profile or {}
            is_nike_v2 = (
                specialist_key == 'nike' and
                isinstance(exit_profile, dict) and
                exit_profile.get('mode') == 'nike_v2' and
                side == 'BULLISH'
            )

            if is_nike_v2:
                bank_atr = float(exit_profile.get('bank_atr', 2.0))
                sl_atr = float(exit_profile.get('sl_atr', 0.8))
                bank_fraction = float(exit_profile.get('bank_fraction', 0.5))
                runner_trail_atr = float(exit_profile.get('runner_trail_atr', 1.5))
                max_pre = int(exit_profile.get('max_bars_pre_bank', timeout_bars or 24))
                max_post = int(exit_profile.get('max_bars_post_bank', max_pre))
                _tp1 = fill_price + atr_move * bank_atr
                _tp2 = _tp1
                _tp3 = _tp1
                _sl = fill_price - atr_move * sl_atr
            elif side == 'BULLISH':
                _tp1 = fill_price + atr_move * TP1_RATIO
                _tp2 = fill_price + atr_move * TP2_RATIO
                _tp3 = fill_price + atr_move * TP3_RATIO
                _sl  = fill_price - atr_move * SL_RATIO
                bank_fraction = 0.0
                runner_trail_atr = 3.0
                max_pre = int(timeout_bars or 0)
                max_post = int(timeout_bars or 0)
            else:
                _tp1 = fill_price - atr_move * TP1_RATIO
                _tp2 = fill_price - atr_move * TP2_RATIO
                _tp3 = fill_price - atr_move * TP3_RATIO
                _sl  = fill_price + atr_move * SL_RATIO
                bank_fraction = 0.0
                runner_trail_atr = 3.0
                max_pre = int(timeout_bars or 0)
                max_post = int(timeout_bars or 0)

            self.positions[symbol] = {
                'entry': fill_price, 'size': size, 'direction': side,
                'confidence': confidence, 'atr_percent': atr_percent,
                'time': datetime.now(),
                'entry_unix': time.time(),  # BS: for bars-to-hit tracking
                'specialist': specialist_key or None,
                'exit_profile': exit_profile if isinstance(exit_profile, dict) else {},
                'timeout_bars': timeout_bars,
                'specialist_probs': None,   # v11.5: Set by bot for Brier tracking
                # ATR-based barrier levels (v11.6)
                'tp1_price': _tp1, 'tp2_price': _tp2, 'tp3_price': _tp3, 'sl_price': _sl,
                # Partial close state (v11.6)
                'tp1_hit': False, 'tp2_hit': False, 'tp3_hit': False,
                'nike_bank_hit': False,
                'nike_bank_fraction': bank_fraction,
                'nike_bank_price': _tp1,
                'nike_runner_trail_atr': runner_trail_atr,
                'nike_runner_peak': fill_price,
                'max_bars_pre_bank': max_pre,
                'max_bars_post_bank': max_post,
                'original_size': size,  # for win/loss labeling after partial closes
            }

    def close_position(self, symbol, exit_price, barrier_hit='TIMEOUT', partial=False, partial_fraction=1.0):
        if symbol in self.positions:
            pos = self.positions[symbol]
            from quanta_config import Config as _cfg
            _bt = getattr(_cfg, 'backtest', None)
            commission_bps = getattr(_bt, 'commission_bps', 4.0)
            slippage_bps   = getattr(_bt, 'slippage_bps',   2.0)

            # Determine how much size to close
            close_fraction = max(0.0, min(1.0, partial_fraction)) if partial else 1.0
            close_size = pos['size'] * close_fraction

            # Exit friction: slippage moves price AGAINST us on close
            slip_factor = slippage_bps / 10000.0
            if pos['direction'] == 'BULLISH':
                fill_exit = exit_price * (1.0 - slip_factor)
            else:
                fill_exit = exit_price * (1.0 + slip_factor)

            pnl = (fill_exit - pos['entry']) * close_size if pos['direction'] == 'BULLISH' \
                  else (pos['entry'] - fill_exit) * close_size

            exit_commission = fill_exit * close_size * (commission_bps / 10000.0)
            pnl -= exit_commission  # deduct exit commission from PnL

            self.balance += pnl
            self.total_pnl += pnl

            # For partial closes: only count win/loss on FINAL close
            # SL after TP1 already hit = WIN (banked partial profit)
            is_final_close = not partial
            if is_final_close:
                self.total_trades += 1
                tp1_was_hit = pos.get('tp1_hit', False) or pos.get('nike_bank_hit', False)
                # Win if: pnl > 0, OR SL fired but TP1 already banked, OR chandelier SL (TP1+2+3 all banked)
                if pnl > 0 or (barrier_hit in ('SL', 'SL_AUTO', 'TIMEOUT', 'NIKE_RUNNER_TIMEOUT') and tp1_was_hit) or barrier_hit == 'CHANDELIER_SL':
                    self.total_wins += 1
                else:
                    self.total_losses += 1

            with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([symbol, pos['entry'], exit_price, pos['direction'], pnl, datetime.now()])

            # v11.4: Notify risk manager + paper logger (only on final close)
            if is_final_close:
                # For RL/risk: treat SL-after-TP1 and CHANDELIER_SL as positive outcome
                banked_win = pos.get('tp1_hit', False) or pos.get('nike_bank_hit', False)
                reported_pnl = abs(pnl) if (barrier_hit in ('SL', 'SL_AUTO', 'CHANDELIER_SL', 'TIMEOUT', 'NIKE_RUNNER_TIMEOUT') and banked_win) else pnl
                self.risk_manager.on_trade_closed(symbol, reported_pnl, self.balance)
            self.paper_logger.log_trade_closed(
                symbol=symbol, direction=pos['direction'],
                entry_price=pos['entry'], exit_price=exit_price,
                size=close_size, pnl=pnl,
                entry_time=str(pos.get('time', '')),
                barrier_hit=barrier_hit,
                confidence=pos.get('confidence', 0),
            )

            # v11.5: Update per-agent Brier score calibration (final close only)
            if is_final_close and pos.get('specialist_probs') is not None and hasattr(self, '_ml_engine') and self._ml_engine:
                tp1_was_hit = pos.get('tp1_hit', False) or pos.get('nike_bank_hit', False)
                outcome = 1 if (pnl > 0 or (barrier_hit in ('SL', 'SL_AUTO', 'CHANDELIER_SL', 'TIMEOUT', 'NIKE_RUNNER_TIMEOUT') and tp1_was_hit) or barrier_hit == 'CHANDELIER_SL') else 0
                try:
                    self._ml_engine.update_brier_scores(pos['specialist_probs'], outcome)
                except Exception:
                    pass

            # v11.5b: Link trade result to RL memory for PPO training (final close only)
            if is_final_close and self._rl_memory:
                try:
                    tp1_was_hit = pos.get('tp1_hit', False) or pos.get('nike_bank_hit', False)
                    # Correct barrier label: SL-after-TP1 should not penalize RL
                    effective_barrier = barrier_hit
                    if barrier_hit in ('SL', 'SL_AUTO', 'CHANDELIER_SL', 'TIMEOUT', 'NIKE_RUNNER_TIMEOUT') and tp1_was_hit:
                        effective_barrier = 'TP1'  # treat as partial win for PPO reward
                    orig_size = pos.get('original_size', close_size)
                    pnl_pct = (pnl / (pos['entry'] * orig_size)) * 100 if pos['entry'] * orig_size > 0 else 0
                    self._rl_memory.record_trade_result(
                        symbol=symbol, pnl=pnl, pnl_pct=pnl_pct,
                        barrier_hit=effective_barrier,
                        entry_price=pos['entry'], exit_price=exit_price
                    )
                except Exception as e:
                    logging.debug(f"Could not link trade to RL memory: {e}")

            # BS implied vol tracking (final close only)
            if is_final_close and barrier_hit in ('TP1', 'TP2', 'TP3', 'SL', 'TP_AUTO', 'SL_AUTO', 'CHANDELIER_SL', 'NIKE_RUNNER_TIMEOUT'):
                try:
                    from collections import deque as _deque
                    entry_unix = pos.get('entry_unix', 0)
                    if entry_unix > 0:
                        elapsed_sec = time.time() - entry_unix
                        bars_elapsed = max(1.0, elapsed_sec / 300.0)
                        if symbol not in self._bs_bars_to_hit:
                            self._bs_bars_to_hit[symbol] = _deque(maxlen=50)
                        self._bs_bars_to_hit[symbol].append(bars_elapsed)
                except Exception:
                    pass

            if partial:
                # Reduce remaining size — keep position open
                self.positions[symbol]['size'] -= close_size
            else:
                del self.positions[symbol]

            # Persist state to disk after every trade
            self._save_state()

            # Record tick for equity curve rendering
            self.paper_logger.snapshot(
                balance=self.balance,
                open_positions=len(self.positions),
                daily_pnl=self.total_pnl
            )

    def get_avg_bars_to_hit(self, symbol):
        """
        Returns mean bars-to-barrier-hit for this symbol, or None if < 5 samples.
        Used by QUANTA_ml_engine to compute bs_implied_vol_ratio feature (index 277).
        """
        dq = self._bs_bars_to_hit.get(symbol)
        if dq is None or len(dq) < 5:
            return None
        return float(sum(dq) / len(dq))

    def _is_nike_v2_position(self, pos):
        profile = pos.get('exit_profile', {})
        return (
            str(pos.get('specialist', '')).lower() == 'nike' and
            pos.get('direction') == 'BULLISH' and
            isinstance(profile, dict) and
            profile.get('mode') == 'nike_v2'
        )

    def _tick_nike_v2(self, symbol, current_price):
        pos = self.positions.get(symbol)
        if not pos:
            return

        atr_move = pos['entry'] * (pos.get('atr_percent', 1.0) / 100.0)
        if atr_move <= 0:
            self.risk_manager.heartbeat(self.balance)
            return

        bank_fraction = float(pos.get('nike_bank_fraction', 0.5))
        bank_price = float(pos.get('nike_bank_price', pos['entry'] + 2.0 * atr_move))
        max_pre = int(pos.get('max_bars_pre_bank', pos.get('timeout_bars') or 24))
        max_post = int(pos.get('max_bars_post_bank', max_pre))
        trail_atr = float(pos.get('nike_runner_trail_atr', 1.5))
        bars_elapsed = max(0.0, (time.time() - pos.get('entry_unix', time.time())) / 300.0)

        if not pos.get('nike_bank_hit', False):
            if current_price <= pos.get('sl_price', pos['entry'] - 0.8 * atr_move):
                self.close_position(symbol, current_price, barrier_hit='SL')
                self.risk_manager.heartbeat(self.balance)
                return

            if current_price >= bank_price:
                self.close_position(symbol, current_price, barrier_hit='TP1',
                                    partial=True, partial_fraction=bank_fraction)
                if symbol in self.positions:
                    self.positions[symbol]['nike_bank_hit'] = True
                    self.positions[symbol]['tp1_hit'] = True
                    self.positions[symbol]['nike_runner_peak'] = current_price
                    self.positions[symbol]['sl_price'] = self.positions[symbol]['entry']
                self.risk_manager.heartbeat(self.balance)
                return

            if bars_elapsed >= max_pre:
                self.close_position(symbol, current_price, barrier_hit='TIMEOUT')
                self.risk_manager.heartbeat(self.balance)
                return

            self.risk_manager.heartbeat(self.balance)
            return

        new_peak = max(pos.get('nike_runner_peak', current_price), current_price)
        trail_stop = max(pos.get('sl_price', pos['entry']), pos['entry'], new_peak - atr_move * trail_atr)
        self.positions[symbol]['nike_runner_peak'] = new_peak
        self.positions[symbol]['sl_price'] = trail_stop

        if current_price <= trail_stop:
            self.close_position(symbol, current_price, barrier_hit='CHANDELIER_SL')
            self.risk_manager.heartbeat(self.balance)
            return

        if bars_elapsed >= max_post:
            self.close_position(symbol, current_price, barrier_hit='NIKE_RUNNER_TIMEOUT')
            self.risk_manager.heartbeat(self.balance)
            return

        self.risk_manager.heartbeat(self.balance)

    def tick(self, symbol, current_price):
        # v11.6: ATR-based 3-tier TP with partial closes and Chandelier trailing after TP3
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        direction = pos['direction']

        if self._is_nike_v2_position(pos):
            self._tick_nike_v2(symbol, current_price)
            return

        # Retrieve stored barrier levels (fallback to old hardcoded logic if missing)
        tp1 = pos.get('tp1_price')
        tp2 = pos.get('tp2_price')
        tp3 = pos.get('tp3_price')
        sl  = pos.get('sl_price')

        if tp1 is None:
            # Legacy position opened before v11.6 — fall back to old behavior
            pnl_pct = (current_price - pos['entry']) / pos['entry'] * 100 \
                      if direction == 'BULLISH' else (pos['entry'] - current_price) / pos['entry'] * 100
            if pnl_pct >= 3:
                self.close_position(symbol, current_price, barrier_hit='TP_AUTO')
            elif pnl_pct <= -2:
                self.close_position(symbol, current_price, barrier_hit='SL_AUTO')
            self.risk_manager.heartbeat(self.balance)
            return

        is_bull = direction == 'BULLISH'

        # Chandelier trail distance: ATR × 3.0 (entry ATR as proxy — consistent, no recalc needed)
        atr_move = pos['entry'] * (pos.get('atr_percent', 1.0) / 100.0)
        chandelier_dist = atr_move * 3.0

        # ── CHANDELIER TRAILING MODE (after TP3) ──
        if pos.get('tp3_hit', False):
            # Update peak price since TP3 hit
            if is_bull:
                new_peak = max(pos.get('peak_since_tp3', current_price), current_price)
                chandelier_sl = new_peak - chandelier_dist
                self.positions[symbol]['peak_since_tp3'] = new_peak
                self.positions[symbol]['sl_price'] = max(chandelier_sl, pos.get('sl_price', 0))  # never lower the SL
            else:
                new_peak = min(pos.get('peak_since_tp3', current_price), current_price)
                chandelier_sl = new_peak + chandelier_dist
                self.positions[symbol]['peak_since_tp3'] = new_peak
                self.positions[symbol]['sl_price'] = min(chandelier_sl, pos.get('sl_price', float('inf')))  # never raise the SL

            # Check if chandelier SL was hit
            updated_sl = self.positions[symbol]['sl_price']
            if (is_bull and current_price <= updated_sl) or (not is_bull and current_price >= updated_sl):
                self.close_position(symbol, current_price, barrier_hit='CHANDELIER_SL')
            self.risk_manager.heartbeat(self.balance)
            return

        # ── STANDARD BARRIER CHECKS (before TP3) ──
        def above(price, level): return price >= level
        def below(price, level): return price <= level
        crossed_tp1 = above(current_price, tp1) if is_bull else below(current_price, tp1)
        crossed_tp2 = above(current_price, tp2) if is_bull else below(current_price, tp2)
        crossed_tp3 = above(current_price, tp3) if is_bull else below(current_price, tp3)
        crossed_sl  = below(current_price, sl)  if is_bull else above(current_price, sl)

        if crossed_tp3 and pos.get('tp2_hit', False):
            # TP3 hit — close 34%, enter Chandelier trailing mode on remainder
            self.close_position(symbol, current_price, barrier_hit='TP3',
                                partial=True, partial_fraction=0.34)
            if symbol in self.positions:
                self.positions[symbol]['tp3_hit'] = True
                self.positions[symbol]['peak_since_tp3'] = current_price
                # Set initial chandelier SL at TP3 level - dist (never below TP2)
                if is_bull:
                    init_sl = max(tp2, current_price - chandelier_dist)
                else:
                    init_sl = min(tp2, current_price + chandelier_dist)
                self.positions[symbol]['sl_price'] = init_sl

        elif crossed_tp2 and pos.get('tp1_hit', False) and not pos.get('tp2_hit', False):
            # TP2 hit — close 33%, trail SL to TP1 level
            self.close_position(symbol, current_price, barrier_hit='TP2',
                                partial=True, partial_fraction=0.333)
            if symbol in self.positions:
                self.positions[symbol]['tp2_hit'] = True
                self.positions[symbol]['sl_price'] = tp1  # trail SL to TP1

        elif crossed_tp1 and not pos.get('tp1_hit', False):
            # TP1 hit — close 33%, move SL to breakeven
            self.close_position(symbol, current_price, barrier_hit='TP1',
                                partial=True, partial_fraction=0.333)
            if symbol in self.positions:
                self.positions[symbol]['tp1_hit'] = True
                self.positions[symbol]['sl_price'] = pos['entry']  # breakeven

        elif crossed_sl:
            # SL hit — full close (win/loss determined inside close_position by tp1_hit flag)
            self.close_position(symbol, current_price, barrier_hit='SL')

        # v11.4: Risk manager heartbeat
        self.risk_manager.heartbeat(self.balance)

    def flat_all(self, price_lookup: dict = None):
        """
        Emergency flat: close every open position at market.
        Called on graceful shutdown so no orphaned positions remain.

        Args:
            price_lookup: optional {symbol: current_price} dict.
                          Falls back to entry price if unavailable (zero PnL booking).
        """
        if not self.positions:
            return
        symbols = list(self.positions.keys())
        logging.warning(f"[FLAT_ALL] Closing {len(symbols)} open positions on shutdown: {symbols}")
        for symbol in symbols:
            try:
                price = (price_lookup or {}).get(symbol, self.positions[symbol]['entry'])
                self.close_position(symbol, price, barrier_hit='SHUTDOWN_FLAT')
            except Exception as e:
                logging.error(f"[FLAT_ALL] Failed to close {symbol}: {e}", exc_info=True)
        logging.warning(f"[FLAT_ALL] Done. Remaining positions: {list(self.positions.keys())}")


# =================== ADAPTIVE CONFORMAL CALIBRATOR (v11.1) ===================

class AdaptiveConformalCalibrator:
    """
    Adaptive Conformal Inference (ACI) Calibrator for classification.

    Replaces the broken VennAbers / naive ConformalCalibrator with a
    mathematically sound system that provides:

    1. Proper nonconformity scores for classification: s_i = 1 - p_hat(y_true_i)
       (Vovk 2005 "Algorithmic Learning in a Random World")
    2. Adaptive alpha via ACI that maintains long-run coverage under distribution shift
       (Gibbs & Candes 2021 "Adaptive Conformal Inference Under Distribution Shift")
    3. Conformal prediction sets with valid coverage guarantees
       (Romano et al. 2020 "Classification with Valid Adaptive Coverage")

    The prediction set SIZE (0, 1, or 2 classes included) serves as a
    principled uncertainty measure: size 2 = uncertain, size 1 = confident.

    Usage:
        cal = AdaptiveConformalCalibrator(alpha_target=0.10)
        cal.fit(val_probs_2d, val_labels)           # Fit on validation set
        result = cal.predict(test_probs_2d)          # Predict with coverage
        result = cal.predict_single(prob_class1)     # Quick single-sample path
    """

    def __init__(self, alpha_target=0.10, aci_gamma=0.01, min_cal_samples=50):
        self.alpha_target = alpha_target    # Target miscoverage (0.10 = 90% coverage)
        self.aci_gamma = aci_gamma          # ACI adaptation rate (Gibbs & Candes 2021)
        self.min_cal_samples = min_cal_samples

        # Calibration state
        self._cal_scores = None             # Nonconformity scores from validation set
        self._quantile_threshold = None     # q_hat for prediction set construction

        # ACI online state (adapts alpha during live inference)
        self._alpha_t = alpha_target        # Current adaptive alpha
        self._n_live_updates = 0
        self._coverage_history = deque(maxlen=500)

        # Statistics for monitoring
        self._fit_n = 0
        self._fit_coverage = None

    def fit(self, val_probs, val_labels):
        """
        Compute nonconformity scores on validation set.

        Args:
            val_probs: (N, 2) array of predicted probabilities [p_class0, p_class1]
                       OR (N,) array of p(class=1)
            val_labels: (N,) array of true labels {0, 1}
        """
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int32)

        if val_probs.ndim == 1:
            # Convert p(class=1) to 2-class format
            val_probs = np.column_stack([1.0 - val_probs, val_probs])

        n = len(val_labels)
        if n < self.min_cal_samples:
            logging.warning(f"ACI: Only {n} cal samples (need {self.min_cal_samples}). Using raw probs.")
            return

        # Nonconformity score: s_i = 1 - p_hat(y_true_i)  (Vovk 2005)
        # This is the correct score for classification — measures how much
        # probability mass the model DIDN'T assign to the true class.
        self._cal_scores = np.array([
            1.0 - val_probs[i, val_labels[i]] for i in range(n)
        ], dtype=np.float64)

        # Finite-sample corrected quantile (Romano et al. 2020 Eq. 3)
        adjusted_q = min(1.0, np.ceil((1 - self.alpha_target) * (n + 1)) / n)
        self._quantile_threshold = float(np.quantile(self._cal_scores, adjusted_q))

        self._fit_n = n
        self._alpha_t = self.alpha_target  # Reset ACI state on refit

        # Compute validation coverage for logging
        pred_sets = self._construct_sets(val_probs)
        covered = sum(1 for i in range(n) if val_labels[i] in pred_sets[i])
        self._fit_coverage = covered / n

    def _construct_sets(self, probs_2d):
        """
        Construct prediction sets from 2-class probabilities.

        A class k is included if: 1 - p_k <= q_hat
        i.e., the model assigns enough probability to class k.

        Returns: list of sets, e.g. [{0}, {1}, {0,1}, set()]
        """
        q = self._quantile_threshold
        if q is None:
            # Not fitted — include the argmax class only
            return [{int(np.argmax(p))} for p in probs_2d]

        sets = []
        for p in probs_2d:
            pred_set = set()
            for k in range(len(p)):
                if 1.0 - p[k] <= q:
                    pred_set.add(k)
            sets.append(pred_set)
        return sets

    def predict(self, test_probs):
        """
        Produce calibrated predictions with conformal prediction sets.

        Args:
            test_probs: (N, 2) or (N,) probabilities

        Returns dict:
            calibrated_prob: calibrated p(class=1) — shrunk toward 0.5 by set size
            prediction_set_size: 0, 1, or 2 (uncertainty measure)
            interval_width: normalized uncertainty in [0, 1] (for backward compat)
            coverage_rate: running ACI coverage rate
        """
        test_probs = np.asarray(test_probs, dtype=np.float64)

        if test_probs.ndim == 1:
            test_probs = np.column_stack([1.0 - test_probs, test_probs])

        if self._quantile_threshold is None:
            # Not fitted — return raw with default uncertainty
            p1 = test_probs[:, 1] if test_probs.ndim == 2 else test_probs
            return {
                'calibrated_prob': p1,
                'prediction_set_size': np.ones(len(p1), dtype=np.int32),
                'interval_width': np.full(len(p1), 0.2),
                'coverage_rate': None
            }

        pred_sets = self._construct_sets(test_probs)
        set_sizes = np.array([len(s) for s in pred_sets], dtype=np.int32)

        # Calibrated probability based on prediction set
        # - Set size 2 (both classes included): shrink toward 0.5 (uncertain)
        # - Set size 1 (one class): use raw probability (confident)
        # - Set size 0 (empty set): use raw but flag as extreme
        p1_raw = test_probs[:, 1]
        calibrated = np.copy(p1_raw)

        for i in range(len(calibrated)):
            if set_sizes[i] == 2:
                # Both classes in set → model is uncertain
                # Shrink prediction toward 0.5 proportional to how close q_hat is to 1
                shrink = min(0.5, self._quantile_threshold)
                calibrated[i] = 0.5 + (p1_raw[i] - 0.5) * (1.0 - shrink)
            elif set_sizes[i] == 0:
                # Empty set → extreme prediction, keep but clip
                calibrated[i] = np.clip(p1_raw[i], 0.02, 0.98)

        calibrated = np.clip(calibrated, 0.01, 0.99)

        # Interval width for backward compatibility
        # Map set_size to interval_width: size 0→0.0, size 1→0.1, size 2→0.5
        interval_width = np.where(set_sizes == 2, 0.5, np.where(set_sizes == 1, 0.1, 0.0))

        coverage = float(np.mean(list(self._coverage_history))) if self._coverage_history else self._fit_coverage

        return {
            'calibrated_prob': calibrated,
            'prediction_set_size': set_sizes,
            'interval_width': interval_width,
            'coverage_rate': coverage
        }

    def predict_single(self, prob_class1):
        """
        Fast path for single-sample prediction (used in specialist loop).

        Args:
            prob_class1: float, probability of class 1

        Returns:
            (calibrated_prob, set_size, interval_width)
        """
        if self._quantile_threshold is None:
            return prob_class1, 1, 0.2

        p = np.array([[1.0 - prob_class1, prob_class1]])
        result = self.predict(p)
        return (
            float(result['calibrated_prob'][0]),
            int(result['prediction_set_size'][0]),
            float(result['interval_width'][0])
        )

    def update_coverage(self, true_label, predicted_probs):
        """
        ACI online update: adapt alpha based on whether the true label
        was covered by the prediction set (Gibbs & Candes 2021 Eq. 2).

        Call this AFTER observing the true outcome to maintain adaptive coverage.

        Args:
            true_label: int (0 or 1)
            predicted_probs: (2,) array of [p0, p1] used for this prediction
        """
        if self._quantile_threshold is None:
            return

        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
        if predicted_probs.ndim == 1 and len(predicted_probs) == 2:
            pred_set = set()
            for k in range(2):
                if 1.0 - predicted_probs[k] <= self._quantile_threshold:
                    pred_set.add(k)
        else:
            pred_set = {int(np.argmax(predicted_probs))}

        covered = 1.0 if true_label in pred_set else 0.0
        self._coverage_history.append(covered)

        # ACI alpha update (Gibbs & Candes 2021 Eq. 2):
        # alpha_{t+1} = alpha_t + gamma * (alpha_target - (1 - err_t))
        # where err_t = 1 if NOT covered, 0 if covered
        err_t = 1.0 - covered
        self._alpha_t = self._alpha_t + self.aci_gamma * (err_t - self.alpha_target)
        self._alpha_t = np.clip(self._alpha_t, 0.001, 0.5)  # Bound alpha to prevent degeneracy

        # Update quantile threshold with new adaptive alpha
        if self._cal_scores is not None and len(self._cal_scores) > 0:
            n = len(self._cal_scores)
            adjusted_q = min(1.0, np.ceil((1 - self._alpha_t) * (n + 1)) / n)
            self._quantile_threshold = float(np.quantile(self._cal_scores, adjusted_q))

        self._n_live_updates += 1

    @property
    def is_fitted(self):
        return self._quantile_threshold is not None

    @property
    def current_alpha(self):
        return self._alpha_t

    @property
    def realized_coverage(self):
        if not self._coverage_history:
            return self._fit_coverage
        return float(np.mean(list(self._coverage_history)))

    def __repr__(self):
        if self.is_fitted:
            cov = f"{self.realized_coverage:.1%}" if self.realized_coverage else "N/A"
            return f"ACI(alpha={self._alpha_t:.3f}, q={self._quantile_threshold:.3f}, coverage={cov}, n_cal={self._fit_n})"
        return "ACI(not fitted)"


# Backward compatibility alias
ConformalCalibrator = AdaptiveConformalCalibrator


# =================== ADWIN DRIFT DETECTOR (v11) ===================

class ADWINDriftDetector:
    """
    ADWIN (ADaptive WINdowing) Drift Detector (Bifet & Gavalda 2007).
    
    Maintains a variable-length window of recent observations.
    Detects concept drift by finding a point where the distribution of
    values before and after differs significantly (Hoeffding bound).
    
    Advantages over raw Page-Hinkley:
    - Automatically shrinks window when drift is detected
    - Provides the estimated mean of the current regime
    - No fixed lambda threshold — adapts to data
    
    Parameters:
        delta (float): Confidence parameter. Lower = fewer false alarms.
                       0.002 is standard for binary accuracy streams.
        max_window (int): Maximum window size. Prevents unbounded memory.
    """
    
    def __init__(self, delta=0.002, max_window=2000):
        self.delta = delta
        self.max_window = max_window
        self.window = deque(maxlen=max_window)
        self._drift_detected = False
        self._n_detections = 0
        self._current_mean = 0.5
    
    def update(self, x: float) -> bool:
        """
        Feed one observation. Returns True if drift detected.
        Automatically shrinks window to discard pre-drift data.
        """
        self.window.append(x)
        self._drift_detected = False
        
        if len(self.window) < 10:
            self._current_mean = np.mean(self.window) if self.window else 0.5
            return False
        
        # ADWIN core: try splitting window at each point
        # and check if the two halves have significantly different means
        n = len(self.window)
        window_arr = np.array(self.window)
        total_sum = np.sum(window_arr)
        
        best_cut = -1
        max_epsilon = 0.0
        
        # Check at logarithmically spaced points for efficiency
        check_points = set()
        step = max(1, n // 20)  # Max 20 check points
        for i in range(step, n - step + 1, step):
            check_points.add(i)
        
        for cut in check_points:
            n0 = cut
            n1 = n - cut
            if n0 < 5 or n1 < 5:
                continue
            
            mean0 = np.sum(window_arr[:cut]) / n0
            mean1 = np.sum(window_arr[cut:]) / n1
            
            # Hoeffding bound (Bifet & Gavalda 2007)
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            delta_prime = self.delta / np.log(n)
            epsilon_cut = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / delta_prime))
            
            if abs(mean0 - mean1) >= epsilon_cut:
                if abs(mean0 - mean1) > max_epsilon:
                    max_epsilon = abs(mean0 - mean1)
                    best_cut = cut
        
        if best_cut > 0:
            # Drift detected — shrink window to post-drift data
            self._drift_detected = True
            self._n_detections += 1
            # Keep only the newer half
            new_window = list(self.window)[best_cut:]
            self.window.clear()
            self.window.extend(new_window)
        
        self._current_mean = float(np.mean(self.window)) if self.window else 0.5
        return self._drift_detected
    
    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
    
    @property
    def current_mean(self) -> float:
        """Estimated accuracy in the current (post-drift) regime."""
        return self._current_mean
    
    @property
    def n_detections(self) -> int:
        return self._n_detections
    
    def reset(self):
        self.window.clear()
        self._drift_detected = False
        self._n_detections = 0
        self._current_mean = 0.5


# =================== COMBINATORIAL PURGED CROSS-VALIDATION ===================

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Lopez de Prado, AFML Ch.12 (2018).

    Generates multiple train/test splits from N groups taken k-at-a-time
    as test sets, with purge gaps to prevent Triple Barrier label leakage.

    For N=6, k=2: generates C(6,2)=15 unique train/test paths.
    Each path uses 2 contiguous groups as test, remaining 4 as train,
    with PURGE_GAP samples removed at each train/test boundary.

    Parameters:
        n_groups (int): Number of time-ordered groups to split data into.
        k_test (int): Number of groups to use as test in each split.
        purge_gap (int): Samples to remove at train/test boundaries.
        embargo_pct (float): Additional % of group size to remove after purge
                             to prevent rolling feature leakage. Default 0.01.
    """

    def __init__(self, n_groups=6, k_test=2, purge_gap=48, embargo_pct=0.01):
        self.n_groups = n_groups
        self.k_test = k_test
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct  # % of test fold size to embargo after purge

    def split(self, n_samples):
        """
        Yields (train_indices, test_indices) for each CPCV fold.

        Args:
            n_samples: Total number of samples in the dataset.

        Yields:
            Tuple of (train_idx, test_idx) numpy arrays.
        """
        from itertools import combinations

        # Create group boundaries
        group_size = n_samples // self.n_groups
        group_bounds = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n_samples
            group_bounds.append((start, end))

        # Generate all C(n_groups, k_test) combinations
        for test_groups in combinations(range(self.n_groups), self.k_test):
            test_set = set(test_groups)
            train_groups = [g for g in range(self.n_groups) if g not in test_set]

            # Build test indices
            test_idx = []
            for g in test_groups:
                s, e = group_bounds[g]
                test_idx.extend(range(s, e))

            # Build purge + embargo zones around test group boundaries
            # Purge: removes samples at boundary (prevents label leakage)
            # Embargo: removes additional samples after purge on train side
            #          (prevents rolling feature leakage, Lopez de Prado 2018)
            embargo_size = int(group_size * self.embargo_pct)
            purge_zones = set()
            for g in test_groups:
                s, e = group_bounds[g]
                # Purge before test start
                for i in range(max(0, s - self.purge_gap), s):
                    purge_zones.add(i)
                # Purge + embargo after test end
                for i in range(e, min(n_samples, e + self.purge_gap + embargo_size)):
                    purge_zones.add(i)

            # Build train indices excluding purge zones
            train_idx = []
            for g in train_groups:
                s, e = group_bounds[g]
                for i in range(s, e):
                    if i not in purge_zones:
                        train_idx.append(i)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield np.array(train_idx), np.array(test_idx)

    def n_splits(self):
        """Number of splits this CV will generate."""
        from math import comb
        return comb(self.n_groups, self.k_test)


# =================== MULTI-STREAM DRIFT MONITOR ===================

class MultiStreamDriftMonitor:
    """
    Multi-stream drift detection using 3 parallel ADWIN instances.

    Trigger logic (Gama et al. 2014):
      accuracy_adwin fires AND (calibration_adwin OR feature_adwin fires)
      -> DRIFT_CONFIRMED

    This prevents false alarms from a single noisy stream while still
    catching genuine regime changes quickly.

    Parameters:
        min_retrain_interval (int): Minimum seconds between drift confirmations.
    """

    def __init__(self, min_retrain_interval=3600):
        self.accuracy_adwin = ADWINDriftDetector(delta=0.002)
        self.calibration_adwin = ADWINDriftDetector(delta=0.005)
        self.feature_adwin = ADWINDriftDetector(delta=0.01)
        self.min_retrain_interval = min_retrain_interval
        self._last_drift_time = 0
        self._drift_confirmed = False
        self._total_confirmations = 0
        # Sliding confirmation window (instead of same-tick AND)
        self._confirmation_window = 50  # updates
        self._last_acc_drift_tick = -999
        self._last_cal_drift_tick = -999
        self._last_feat_drift_tick = -999
        self._update_count = 0

    def update(self, correct: bool, predicted_prob: float, actual_outcome: int,
               feature_delta: float = None):
        """
        Feed one prediction outcome into all streams.

        Args:
            correct: Whether the prediction was correct (bool).
            predicted_prob: Model's predicted probability for the actual class.
            actual_outcome: Actual label (0 or 1).
            feature_delta: Optional first-PC delta of standardized features.
                           Pass None to skip feature stream.

        Returns:
            bool: True if drift is confirmed (all conditions met).
        """
        import time as _time

        self._drift_confirmed = False

        # Stream 1: Accuracy
        acc_drift = self.accuracy_adwin.update(1.0 if correct else 0.0)

        # Stream 2: Calibration error
        cal_error = abs(predicted_prob - float(actual_outcome))
        cal_drift = self.calibration_adwin.update(cal_error)

        # Stream 3: Feature distribution (optional)
        feat_drift = False
        if feature_delta is not None:
            feat_drift = self.feature_adwin.update(feature_delta)

        self._update_count += 1

        # Track when each stream last fired
        if acc_drift:
            self._last_acc_drift_tick = self._update_count
        if cal_drift:
            self._last_cal_drift_tick = self._update_count
        if feat_drift:
            self._last_feat_drift_tick = self._update_count

        # Sliding window confirmation: accuracy drift within N updates of cal/feat drift
        acc_recent = (self._update_count - self._last_acc_drift_tick) <= self._confirmation_window
        cal_recent = (self._update_count - self._last_cal_drift_tick) <= self._confirmation_window
        feat_recent = (self._update_count - self._last_feat_drift_tick) <= self._confirmation_window

        if acc_recent and (cal_recent or feat_recent):
            now = _time.time()
            if now - self._last_drift_time >= self.min_retrain_interval:
                self._drift_confirmed = True
                self._last_drift_time = now
                self._total_confirmations += 1
                # Reset ticks to prevent re-firing on same drift event
                self._last_acc_drift_tick = -999
                self._last_cal_drift_tick = -999
                self._last_feat_drift_tick = -999

        return self._drift_confirmed

    @property
    def drift_confirmed(self) -> bool:
        return self._drift_confirmed

    @property
    def total_confirmations(self) -> int:
        return self._total_confirmations

    @property
    def stream_means(self) -> dict:
        """Current estimated means from each ADWIN stream."""
        return {
            'accuracy': self.accuracy_adwin.current_mean,
            'calibration': self.calibration_adwin.current_mean,
            'feature': self.feature_adwin.current_mean
        }

    def reset(self):
        self.accuracy_adwin.reset()
        self.calibration_adwin.reset()
        self.feature_adwin.reset()
        self._drift_confirmed = False


# =================== PAGE-HINKLEY DRIFT DETECTOR ===================

class PageHinkleyDriftDetector:
    """
    Page-Hinkley test for concept drift detection (Page 1954, Hinkley 1971).
    
    Monitors a stream of binary outcomes (correct/wrong predictions).
    Fires an alarm when the cumulative deviation from the running mean
    exceeds threshold λ, indicating the model's accuracy has degraded.
    
    Parameters:
        delta (float): Magnitude tolerance. Small values = more sensitive.
                       0.005 is standard for binary accuracy streams.
        lam (float):   Detection threshold. Lower = faster detection but
                       more false alarms. 50 is a good balance.
        min_samples (int): Minimum observations before testing. Prevents
                          spurious alarms during warmup.
    """
    
    def __init__(self, delta=0.005, lam=50.0, min_samples=30):
        self.delta = delta
        self.lam = lam
        self.min_samples = min_samples
        self.reset()
    
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.x_mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = float('inf')
        self._drift_detected = False
    
    def update(self, x: float) -> bool:
        """
        Feed one observation (1.0=correct, 0.0=wrong).
        Returns True if drift detected.
        """
        self.n += 1
        self.sum += x
        self.x_mean = self.sum / self.n
        self.cum_sum += (x - self.x_mean - self.delta)
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)
        
        self._drift_detected = False
        if self.n >= self.min_samples:
            if (self.cum_sum - self.min_cum_sum) > self.lam:
                self._drift_detected = True
        return self._drift_detected
    
    @property
    def drift_detected(self) -> bool:
        return self._drift_detected


# =================== RL MEMORY (FEATHER-BASED) ===================

class RLMemory:
    """Research-backed RL experience replay buffer with 3-tier TP weighting."""
    
    def __init__(self, cfg):
        import pandas as pd
        self._pd = pd
        self.cfg = cfg
        self.memory_file = Path(cfg.rl_memory_file.replace('.json', '.feather'))
        self.memory = []
        self.MAX_MEMORY = 10000
        self.active_symbols = set()
        self.drift_detector = PageHinkleyDriftDetector(delta=0.005, lam=50.0, min_samples=30)
        self.adwin_detector = ADWINDriftDetector(delta=0.002, max_window=2000)  # v11: ADWIN primary drift
        self.multi_drift = MultiStreamDriftMonitor(min_retrain_interval=3600)   # v11: 3-stream drift
        self._load_memory()

    def _load_memory(self):
        pd = self._pd
        try:
            if self.memory_file.exists():
                df = pd.read_feather(self.memory_file)
                records = df.to_dict('records')
                recent_records = records[-self.MAX_MEMORY:] if len(records) > self.MAX_MEMORY else records
                self.memory = list(recent_records)
                self.active_symbols = {
                    p['symbol'] for p in self.memory if p.get('outcome') is None
                }
                logging.info(f"✅ Loaded {len(self.memory)} RL experiences (buffer size: 10k)")
                logging.info(f"   └─ Active symbols: {len(self.active_symbols)}")
        except Exception as e:
            logging.warning(f"Could not load RL memory: {e}")
            self.memory = []
            self.active_symbols = set()

    def _save_memory(self):
        pd = self._pd
        try:
            if not self.memory:
                return
            df = pd.DataFrame(list(self.memory))
            df.to_feather(self.memory_file, compression='lz4')
        except Exception as e:
            logging.error(f"Failed to save RL memory: {e}")

    def add_prediction(self, symbol, direction, confidence, magnitude, entry_price, features, ppo_data=None):
        prediction = {
            'symbol': symbol, 'direction': direction, 'confidence': confidence,
            'magnitude': magnitude, 'entry_price': entry_price,
            'entry_time': time.time(),
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'outcome': None, 'actual_move': None, 'success': None,
            'ppo_action': ppo_data.get('ppo_action') if ppo_data else None,
            'ppo_log_prob': ppo_data.get('ppo_log_prob') if ppo_data else None,
            'ppo_value': ppo_data.get('ppo_value') if ppo_data else None
        }
        self.memory.append(prediction)
        if len(self.memory) > self.MAX_MEMORY:
            completed = [p for p in self.memory if p.get('outcome') in ('correct', 'wrong')]
            pending = [p for p in self.memory if p.get('outcome') is None]
            neutral = [p for p in self.memory if p.get('outcome') == 'neutral']
            keep_completed = completed
            keep_neutral = neutral[-500:] if len(neutral) > 500 else neutral
            remaining_budget = self.MAX_MEMORY - len(keep_completed) - len(keep_neutral)
            keep_pending = pending[-remaining_budget:] if len(pending) > remaining_budget else pending
            self.memory = keep_completed + keep_neutral + keep_pending
            evicted = self.MAX_MEMORY + 1 - len(self.memory)
            if evicted > 100:
                print(f"🗑️ RL Memory eviction: kept {len(keep_completed)} completed, {len(keep_pending)} pending, evicted {evicted}")
        self.active_symbols.add(symbol)
        if len(self.memory) % 10 == 0:
            self._save_memory()

    def check_predictions(self, bnc, ppo_memory=None, scaler=None, monitor=None):
        """3-TIER TRIPLE BARRIER METHOD (López de Prado 2018 Ch.3)"""
        # Lazy import to avoid circular dependency
        try:
            from QUANTA_agents import DifferentialSharpeRatio
            _PPO_AVAILABLE = True
        except ImportError:
            _PPO_AVAILABLE = False

        updated = 0
        checked = 0
        now = time.time()
        print(f"\n🔍 RL OUTCOME CHECK RUNNING...")
        print(f"   Total predictions: {len(self.memory)}")
        print(f"   Active symbols: {len(self.active_symbols)}")

        for pred in self.memory:
            if pred['outcome'] is not None:
                continue
            time_elapsed = now - pred['entry_time']
            # Check after 5 min — if SL/TP hit early, record immediately 
            # If no barrier hit, only mark neutral after full outcome window
            if time_elapsed < 300:  # Need at least 5 minutes of candle data
                continue
            checked += 1
            try:
                start_ms = int(pred['entry_time'] * 1000)
                # FIX 2.4: Dynamic outcome window based on prediction magnitude
                # If TP1 already hit (monitoring phase), extend to full 24h (288 candles)
                if pred.get('_tp1_hit'):
                    outcome_limit = 288  # Full 24h monitoring window
                else:
                    pred_magnitude = pred.get('magnitude', 1.0)
                    magnitude_factor = min(3.0, max(1.0, pred_magnitude))
                    outcome_limit = int(48 * magnitude_factor)
                    outcome_limit = min(144, max(36, outcome_limit))
                klines = bnc.get_klines_from(pred['symbol'], '5m', start_time=start_ms, limit=outcome_limit)
                if not klines or len(klines) < 5:
                    continue

                entry_price = pred['entry_price']
                magnitude = pred['magnitude']
                direction = pred['direction']
                # magnitude is already volatility-scaled — comparing to raw vol double-counts it
                barrier_base = min(magnitude, MAX_MAGNITUDE)

                if direction == 'BULLISH':
                    tp1_price = entry_price * (1 + barrier_base * TP1_RATIO / 100)
                    tp2_price = entry_price * (1 + barrier_base * TP2_RATIO / 100)
                    tp3_price = entry_price * (1 + barrier_base * TP3_RATIO / 100)
                    sl_price = entry_price * (1 - barrier_base * SL_RATIO / 100)
                else:
                    tp1_price = entry_price * (1 - barrier_base * TP1_RATIO / 100)
                    tp2_price = entry_price * (1 - barrier_base * TP2_RATIO / 100)
                    tp3_price = entry_price * (1 - barrier_base * TP3_RATIO / 100)
                    sl_price = entry_price * (1 + barrier_base * SL_RATIO / 100)

                tp1_hit = tp2_hit = tp3_hit = sl_hit = False
                tp1_time = tp2_time = tp3_time = sl_time = float('inf')
            
                # Track ME (Maximum Excursion) before SL hits
                max_favorable_price = entry_price

                for kline in klines:
                    candle_time = float(kline[0])
                    if candle_time < pred['entry_time'] * 1000:
                        continue
                    high = float(kline[2])
                    low = float(kline[3])
                
                    if direction == 'BULLISH':
                        if not sl_hit: max_favorable_price = max(max_favorable_price, high)
                        if high >= tp3_price and not tp3_hit: tp3_hit = True; tp3_time = candle_time
                        if high >= tp2_price and not tp2_hit: tp2_hit = True; tp2_time = candle_time
                        if high >= tp1_price and not tp1_hit: tp1_hit = True; tp1_time = candle_time
                        if low <= sl_price and not sl_hit: sl_hit = True; sl_time = candle_time
                    else:
                        if not sl_hit: max_favorable_price = min(max_favorable_price, low)
                        if low <= tp3_price and not tp3_hit: tp3_hit = True; tp3_time = candle_time
                        if low <= tp2_price and not tp2_hit: tp2_hit = True; tp2_time = candle_time
                        if low <= tp1_price and not tp1_hit: tp1_hit = True; tp1_time = candle_time
                        if high >= sl_price and not sl_hit: sl_hit = True; sl_time = candle_time

                MONITORING_WINDOW = 86400  # 24 hours in seconds
                
                if sl_hit and sl_time <= min(tp1_time, tp2_time, tp3_time):
                    # Pure SL hit (before any TP) — wrong prediction
                    success = False; outcome_tier = 'SL'; sample_weight = SL_WEIGHT; pred['outcome'] = 'wrong'
                elif sl_hit and tp1_hit and tp1_time < sl_time:
                    # SL hit AFTER TP1 — partial win: banked profit at TP1, trail stopped out
                    # Treat as TP1 for training signal — model was directionally correct
                    success = True; outcome_tier = 'TP1'; sample_weight = TP1_WEIGHT; pred['outcome'] = 'correct'
                elif tp3_hit and tp3_time < sl_time:
                    success = True; outcome_tier = 'TP3'; pred['outcome'] = 'correct'
                
                    # 🔥 DYNAMIC WEIGHTING BEYOND TP3
                    if direction == 'BULLISH':
                        beyond_ratio = (max_favorable_price - entry_price) / (tp3_price - entry_price)
                    else:
                        beyond_ratio = (entry_price - max_favorable_price) / (entry_price - tp3_price)
                
                    # Scale weight based on how far past TP3 it ran (cap at 20.0x)
                    scaled_weight = TP3_WEIGHT * max(1.0, beyond_ratio)
                    sample_weight = min(20.0, scaled_weight)
                
                elif tp2_hit and tp2_time < sl_time:
                    # TP2 hit but not TP3 — check if still within monitoring window
                    if time_elapsed < MONITORING_WINDOW:
                        # Keep monitoring for potential TP3 upgrade
                        pred['outcome'] = None  # Stay active
                        pred['_best_tier'] = 'TP2'
                        pred['_best_weight'] = TP2_WEIGHT
                        pred['_peak_price'] = max_favorable_price
                        continue  # Don't finalize yet
                    else:
                        # 24h expired, finalize at TP2
                        success = True; outcome_tier = 'TP2'; sample_weight = TP2_WEIGHT; pred['outcome'] = 'correct'
                
                elif tp1_hit and tp1_time < sl_time:
                    # TP1 hit but not TP2/TP3 — enter monitoring phase
                    if time_elapsed < MONITORING_WINDOW:
                        # Keep monitoring for potential TP2/TP3 upgrade
                        pred['outcome'] = None  # Stay active
                        pred['_best_tier'] = pred.get('_best_tier', 'TP1')
                        pred['_best_weight'] = pred.get('_best_weight', TP1_WEIGHT)
                        pred['_peak_price'] = max_favorable_price
                        pred['_tp1_hit'] = True  # Flag for re-prediction eligibility
                        
                        # Remove from active_symbols so the coin can be re-predicted
                        self.active_symbols.discard(pred['symbol'])
                        continue  # Don't finalize yet
                    else:
                        # 24h expired, finalize at best tier reached
                        best_tier = pred.get('_best_tier', 'TP1')
                        if best_tier == 'TP2':
                            success = True; outcome_tier = 'TP2'; sample_weight = TP2_WEIGHT
                        else:
                            success = True; outcome_tier = 'TP1'; sample_weight = TP1_WEIGHT
                        pred['outcome'] = 'correct'
                else:
                    # No barrier hit yet — only mark neutral after full outcome window
                    if time_elapsed < self.cfg.rl_outcome_check_time:
                        continue  # Let it cook longer
                    # Neutral: predicted a move, nothing happened — still informative for monitor/PPO
                    # sample_weight=0 so CatBoost ignores it, but PPO gets a small penalty
                    success = None; outcome_tier = 'NONE'; sample_weight = 0.0; pred['outcome'] = 'neutral'

                pred['success'] = success
                pred['outcome_tier'] = outcome_tier
                pred['sample_weight'] = float(sample_weight)

                # PPO experience storage (v11.5b: PPO is a SIZE ORACLE, not a gate)
                if ppo_memory is not None and pred.get('ppo_action') is not None:
                    try:
                        magnitude    = pred.get('magnitude', 5.0)
                        ml_dir       = pred.get('direction', 'NONE')
                        ppo_size_mult = float(pred.get('ppo_size_mult', 1.0))

                        # Base move magnitude (fraction of account risk)
                        if success is True:
                            move_mag = (magnitude / 100.0) * sample_weight
                        elif success is False:
                            move_mag = (magnitude * SL_RATIO / 100.0) * sample_weight
                        else:
                            move_mag = 0.005  # Neutral — small baseline cost

                        # ── PPO SIZER REWARD ──────────────────────────────────────────
                        # Reward = (outcome sign) × move_mag × ppo_size_mult
                        #
                        # Logic: PPO chose how much to size. If it sized up a winner →
                        # big positive reward. If it sized up a loser → big negative.
                        # If it sized down (contradict/hold) a winner → small positive.
                        # If it sized down a loser → small negative.
                        #
                        # This teaches PPO: "agree with good ML signals, doubt bad ones."
                        # It can't block trades anymore so it MUST learn to size correctly.
                        # ─────────────────────────────────────────────────────────────
                        if success is True:
                            raw_return =  move_mag * ppo_size_mult   # Win: reward ∝ size
                        elif success is False:
                            raw_return = -move_mag * ppo_size_mult   # Loss: penalty ∝ size
                        else:
                            raw_return = -move_mag * 0.5             # Neutral: mild penalty

                        reward = raw_return
                        try:
                            if hasattr(ppo_memory, '_dsr'):
                                reward = ppo_memory._dsr.compute(raw_return)
                            elif _PPO_AVAILABLE:
                                if not hasattr(ppo_memory, '_dsr'):
                                    ppo_memory._dsr = DifferentialSharpeRatio()
                                reward = ppo_memory._dsr.compute(raw_return)
                        except Exception:
                            pass
                        state = np.array(pred['features'], dtype=np.float32)
                        if scaler:
                            state = scaler.transform([state])[0]
                            
                        # 🔥 Part E: Meta-State augmented with CatBoost probabilities 
                        if 'specialist_probs' in pred:
                            state = np.concatenate([state, pred['specialist_probs']])
                        else:
                            state = np.concatenate([state, np.zeros(7)])
                            
                        ppo_memory.store_memory(
                            state, pred['ppo_action'], pred['ppo_log_prob'],
                            reward, True, pred['ppo_value']
                        )
                    except Exception as e:
                        logging.debug(f"PPO memory store error: {e}")

                current_price = float(klines[-1][4])
                actual_move = (current_price - entry_price) / entry_price * 100
                pred['actual_move'] = actual_move
                updated += 1
                
                # C3: Feed outcome into Page-Hinkley drift detector
                if success is not None:  # Only feed definitive outcomes
                    val = 1.0 if success else 0.0
                    self.drift_detector.update(val)
                    self.adwin_detector.update(val)  # v11: ADWIN primary drift

                    # v11: Multi-stream drift (accuracy + calibration + feature)
                    predicted_prob = pred.get('confidence', 50.0) / 100.0
                    actual_outcome_int = 1 if success else 0
                    self.multi_drift.update(
                        correct=bool(success),
                        predicted_prob=predicted_prob,
                        actual_outcome=actual_outcome_int
                    )

                    # v11: Feed outcome into real-time model monitor
                    if monitor:
                        pred_class = 1 if direction == 'BULLISH' else 0
                        actual_class = 1 if success else 0
                        confidence_val = pred.get('confidence', 50.0) / 100.0
                        monitor.log_prediction(pred_class, confidence_val, actual_class)

                # Also feed neutral outcomes to monitor (calibration signal — model predicted
                # a big move, nothing happened → confidence was miscalibrated)
                if success is None and monitor:
                    try:
                        pred_class = 1 if direction == 'BULLISH' else 0
                        confidence_val = pred.get('confidence', 50.0) / 100.0
                        # Treat neutral as wrong for calibration (predicted move didn't happen)
                        monitor.log_prediction(pred_class, confidence_val, 1 - pred_class)
                    except Exception:
                        pass
                
                symbol_still_pending = any(
                    p['outcome'] is None and p['symbol'] == pred['symbol']
                    for p in self.memory if p is not pred
                )
                if not symbol_still_pending:
                    self.active_symbols.discard(pred['symbol'])

                if success:
                    emoji = "✅" if outcome_tier in ('TP3', 'TP2') else "✓"
                    print(f"   {emoji} {pred['symbol']}: {direction} ({outcome_tier}, weight={sample_weight:.1f}x, endpoint: {actual_move:+.1f}%)")
                elif success is False:
                    print(f"   ❌ {pred['symbol']}: {direction} ({outcome_tier}, weight={sample_weight:.1f}x, endpoint: {actual_move:+.1f}%)")
                else:
                    print(f"   ⚪ {pred['symbol']}: {direction} (no barrier hit - neutral)")
            except Exception as e:
                logging.debug(f"Outcome check error {pred.get('symbol', 'unknown')}: {e}")
                continue

        # C3: Check for concept drift (v11: ADWIN primary + Page-Hinkley secondary + multi-stream)
        adwin_drift = self.adwin_detector.drift_detected
        ph_drift = self.drift_detector.drift_detected
        multi_drift = self.multi_drift.drift_confirmed  # AND/OR 3-stream confirmation
        drift_detected = adwin_drift or ph_drift or multi_drift
        
        if updated > 0:
            self._save_memory()
            completed = [p for p in self.memory if p['outcome'] in ['correct', 'wrong']]
            if completed:
                wins = sum(1 for p in completed if p['success'])
                tp3_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP3')
                tp2_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP2')
                tp1_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP1')
                win_rate = (wins / len(completed)) * 100
                print(f"\n✅ RL CHECK COMPLETE: {updated}/{checked} predictions verified")
                print(f"📊 Win Rate: {wins}/{len(completed)} ({win_rate:.1f}%)")
                print(f"🎯 TP Tiers: TP3={tp3_count} (5x) | TP2={tp2_count} (2.5x) | TP1={tp1_count} (1x)")
                
                # Fetch and print top coins using get_stats logic
                stats = self.get_stats()
                if stats.get('top_coins'):
                    print(f"🌟 Top Performing Coins (min. 3 trades):")
                    for sym, rate in stats['top_coins']:
                        print(f"   • {sym}: {rate:.1f}%")
                        
                if drift_detected:
                    detector_name = "ADWIN" if adwin_drift else "Page-Hinkley"
                    adwin_mean = f"{self.adwin_detector.current_mean:.1%}"
                    print(f"⚠️  CONCEPT DRIFT DETECTED ({detector_name}) — current accuracy: {adwin_mean}, detections: {self.adwin_detector.n_detections}")
            else:
                print(f"\n✅ RL CHECK COMPLETE: {updated}/{checked} predictions verified")
        else:
            print(f"   No predictions ready to check yet (need 1h elapsed)")
        return updated, drift_detected

    def get_training_data(self):
        completed = [p for p in self.memory if p['outcome'] in ['correct', 'wrong']]
        if len(completed) < 10:
            return None, None, None
        X_list = []
        y_list = []
        weight_list = []
        expected_shape = None
        for p in completed:
            try:
                features = np.array(p['features'], dtype=np.float32)
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue
                if expected_shape is None:
                    expected_shape = features.shape[0]
                if features.shape[0] != expected_shape:
                    continue
                sample_weight = p.get('sample_weight', 1.0)
                if sample_weight == 0.0:
                    continue
                X_list.append(features)
                y_list.append(1 if p['success'] else 0)
                weight_list.append(sample_weight)
            except Exception:
                continue
        if len(X_list) < 10:
            return None, None, None
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        weights = np.array(weight_list, dtype=np.float32)
        weights = np.maximum(weights, 0.01)
        tp3_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP3')
        tp2_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP2')
        tp1_count = sum(1 for p in completed if p.get('outcome_tier') == 'TP1')
        sl_count = sum(1 for p in completed if p.get('outcome_tier') == 'SL')
        logging.info(f"✅ Training data: {len(X_list)} samples from {len(completed)} completed")
        logging.info(f"   🎯 TP Tiers: TP3={tp3_count} (5x) | TP2={tp2_count} (2.5x) | TP1={tp1_count} (1x) | SL={sl_count} (0.5x)")
        logging.info(f"   📊 Effective samples: {weights.sum():.1f} (after weighting)")
        logging.info(f"   💾 Buffer: {len(self.memory)}/10,000 | Active: {len(self.active_symbols)} symbols")
        return X, y, weights

    def record_trade_result(self, symbol, pnl, pnl_pct, barrier_hit, entry_price, exit_price):
        """Link paper trade result to the matching RL prediction for PPO training.

        Finds the most recent pending/active prediction for this symbol and
        enriches it with real trade PnL data. This makes PPO learn from actual
        trade outcomes, not just directional accuracy.
        """
        # Find matching prediction (most recent for this symbol that's still pending or active)
        matched = None
        for pred in reversed(self.memory):
            if pred.get('symbol') == symbol and pred.get('outcome') is None:
                matched = pred
                break
            # Also match recently resolved ones (within 60s) in case RL resolved first
            if (pred.get('symbol') == symbol and
                pred.get('outcome') in ('correct', 'wrong') and
                not pred.get('trade_pnl') and
                time.time() - pred.get('entry_time', 0) < 86400):
                matched = pred
                break

        if matched is None:
            return

        # Enrich with real trade data
        matched['trade_pnl'] = round(pnl, 4)
        matched['trade_pnl_pct'] = round(pnl_pct, 4)
        matched['trade_barrier'] = barrier_hit
        matched['trade_exit_price'] = exit_price
        matched['trade_closed_at'] = time.time()

        # If RL hasn't resolved this yet, resolve it now based on actual PnL
        if matched.get('outcome') is None:
            matched['outcome'] = 'correct' if pnl > 0 else 'wrong'
            matched['success'] = pnl > 0
            matched['actual_move'] = pnl_pct
            if barrier_hit in ('TP_AUTO', 'TP1', 'TP2', 'TP3'):
                matched['outcome_tier'] = barrier_hit
                matched['sample_weight'] = 2.5 if 'TP2' in barrier_hit or 'TP3' in barrier_hit else 1.0
            elif barrier_hit in ('SL_AUTO', 'SL'):
                matched['outcome_tier'] = 'SL'
                matched['sample_weight'] = 0.5
            else:
                matched['outcome_tier'] = 'TIMEOUT'
                matched['sample_weight'] = 0.3
            self.active_symbols.discard(symbol)

        # Feed drift detectors with trade result
        try:
            self.drift_detector.update(1 if pnl > 0 else 0)
            self.adwin_detector.update(1 if pnl > 0 else 0)
        except Exception:
            pass

        self._save_memory()
        logging.debug(f"RL Memory linked trade: {symbol} PnL=${pnl:.2f} ({pnl_pct:.2f}%) barrier={barrier_hit}")

    def get_stats(self):
        completed = [p for p in self.memory if p['outcome'] in ['correct', 'wrong']]
        pending = [p for p in self.memory if p['outcome'] is None]
        replaced = [p for p in self.memory if p['outcome'] == 'replaced']
        neutral = [p for p in self.memory if p['outcome'] == 'neutral']
        
        # Calculate per-symbol win rates (min 3 trades)
        symbol_stats = {}
        for p in completed:
            sym = p['symbol']
            if sym not in symbol_stats:
                symbol_stats[sym] = {'trades': 0, 'wins': 0}
            symbol_stats[sym]['trades'] += 1
            if p['success']:
                symbol_stats[sym]['wins'] += 1
                
        top_coins = []
        if symbol_stats:
            valid_coins = {
                sym: (s['wins'] / s['trades']) * 100 
                for sym, s in symbol_stats.items() if s['trades'] >= 3
            }
            # Sort by winrate (desc), then trades (desc)
            top_coins = sorted(
                valid_coins.items(), 
                key=lambda x: (x[1], symbol_stats[x[0]]['trades']), 
                reverse=True
            )[:5]
            
        if not completed:
            return {
                'total': len(self.memory), 'pending': len(pending), 'completed': 0,
                'replaced': len(replaced), 'neutral': len(neutral),
                'correct': 0, 'correct_pct': 0, 'avg_move': 0,
                'buffer_usage': len(self.memory) / self.MAX_MEMORY * 100,
                'unique_symbols': len(self.active_symbols),
                'top_coins': top_coins
            }
        correct = sum(1 for p in completed if p['success'])
        accuracy = (correct / len(completed)) * 100 if completed else 0
        avg_move = np.mean([abs(p['actual_move']) for p in completed if p['actual_move'] is not None]) if completed else 0
        return {
            'total': len(self.memory), 'pending': len(pending),
            'completed': len(completed), 'replaced': len(replaced),
            'neutral': len(neutral), 'correct': correct,
            'correct_pct': accuracy, 'avg_move': avg_move,
            'buffer_usage': len(self.memory) / self.MAX_MEMORY * 100,
            'unique_symbols': len(self.active_symbols),
            'top_coins': top_coins
        }
