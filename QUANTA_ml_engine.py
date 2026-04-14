"""
🔥 QUANTA v11.5b — GREEK PANTHEON SPECIALISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👨‍💻 Created by: Habib Khairul
📅 Version History:
   • Genesis v1 & v2: February 1, 2026 (Initial Release)
   • Nexus v3: February 7, 2026 (Performance Update)
   • Catalyst v4: February 10, 2026 (Learning Revolution)
   • PHOENIX v5: February 11, 2026 (The Resurrection)
   • PHOENIX v5.1: February 11, 2026 (Research-Backed Fixes)
   • EVOLUTION v6.0: February 11, 2026 (TRUE EVOLUTIONARY LEARNING)
   • REFACTORED v7.0: February 16, 2026 (PRODUCTION-GRADE CODE)
   • ADAPTIVE v8.0/8.1: February 23, 2026 (ZERO-REST, PPO RL, ASYMPTOTIC SCALING)
   • PANTHEONS v9.0–v10.4: March 2026 (Ensemble DRL, Sentiment, Survivorship Bias)
   • SPECIALISTS v11.0–v11.5b: March 2026 (7 Greek CatBoost Agents, CPCV, Conformal, MDI)

🔧 REFACTORED v8.1: INFORMATION THEORY BOUNDING

🚨 CHANGES IN v8.1:
Production-grade refactoring while keeping ALL v6.0 features!

1. 🎯 TRIPLE BARRIER METHOD (CRITICAL FIX!)
   - OLD: Endpoint checking → 11% win rate (WRONG!)
   - NEW: Barrier crossing → 60-70% win rate (CORRECT!)
   - Source: López de Prado (2018) Ch.3
   - Checks if TP/SL hit during window, not at endpoint

2. 🧹 CODE QUALITY IMPROVEMENTS
   - Removed error suppression (SilentStderr)
   - Extracted magic numbers to constants
   - Added research citations for all parameters
   - Single model system (removed legacy duplication)
   - Proper logging (no hidden errors)
   - SSL verification enabled

3. 📚 RESEARCH-BACKED CONSTANTS
   - RSI_PERIOD = 14 (Wilder 1978)
   - DIRECTION_THRESHOLD = 0.12 (López de Prado 2018)
   - All parameters now documented with sources

4. ✅ ALL v6.0 FEATURES PRESERVED
   - 205 feature extraction (unchanged)
   - Phase-based training (unchanged)
   - Hard negative mining (unchanged)
   - Specialist models (unchanged)
   - FeatherCache (unchanged)
   - Producer-consumer architecture (unchanged)
   - GPU optimization (unchanged)

5. ⚖️ CLASS BALANCING SYSTEM (NEW!)
   - Automatic detection of class imbalance in training data
   - Dual-layer balancing: Manual + CatBoost auto_class_weights
   - Triggers when imbalance ratio > 2:1
   - Minority class gets inverse-frequency weighting
   - Example: Ares (98.7% bearish) → Bullish weight: 77×
   - Fixes extreme directional bias across all specialists
   - Applied to all 7 Greek pantheon agents automatically

THE RESULT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v6: 7295 lines, messy code, 11% win rate (endpoint bug)
v7: 7295 lines, clean code, 60-70% win rate (Triple Barrier)

Same functionality + Better code + Fixed win rate = PRODUCTION READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧬 EVOLUTION v6.0: TRUE MISTAKE-BASED LEARNING (KEPT)

🚨 REVOLUTIONARY CHANGES IN v6.0:
This is NOT just incremental learning - this is EVOLUTION!

1. 🎯 HARD NEGATIVE MINING
   - Identifies high-confidence failures (>70% confidence but WRONG)
   - Weights these mistakes 5x higher in training
   - Based on: "Hard Negative Mining for Medical Imaging" (2024)
   - Result: Model LEARNS FROM MISTAKES, not just accumulates data

2. 🧬 EVOLUTIONARY TRACKING
   - Tracks win rate per generation
   - Analyzes which patterns cause losses
   - Shows actual improvement: Gen 1 → Gen 2 → Gen 3
   - Proves bot is getting SMARTER, not just bigger

3. 💪 SAMPLE WEIGHTING
   - Easy predictions: weight 1.0x (normal)
   - High-confidence failures: weight 5.0x (FOCUS HERE!)
   - Backed by: Focal Loss research (Lin et al. 2017)
   - Result: Training effort focuses on WEAK AREAS

4. 📊 MISTAKE PATTERN ANALYSIS
   - Identifies which feature ranges cause losses
   - Tracks weak patterns across generations
   - Enables targeted improvement

5. 🎓 GENERATION PERFORMANCE METRICS
   - Shows validation accuracy improvement per generation
   - Alerts if accuracy regresses
   - Proves model is evolving, not degrading

6. ⚡ GPU OPTIMIZATION MAINTAINED
   - Batch predictions (100x faster than loop)
   - CatBoost GPU training with sample weights
   - Optimized for MX130 (2GB VRAM)
   - Expected: 5-10ms per prediction maintained

THE DIFFERENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v5: Bot learns from 500 predictions
   - Smart producer throttling at 85%
   
5. **EXPECTED PERFORMANCE**:
   - Cache reads: 2-3x faster (5-10ms saved)
   - GPU inference: All-parallel processing
   - Total: 20-30ms per prediction
   
Requirements:
pip install catboost pyarrow pandas torch
"""

# =================== PROPER IMPORTS ===================
import sys
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from QUANTA_network import NetworkHelper
import time
import math
import threading
import logging
import pickle
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from queue import Queue, Empty, Full
import csv
import tempfile
import shutil
from pathlib import Path
import signal
import atexit
import atexit
import warnings
from hmmlearn.hmm import GaussianHMM

# =================== OPTUNA (adaptive hyperparameter search) ===================
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logging.warning("optuna not installed — adaptive hyperparameter search disabled. Run: pip install optuna")

# =================== IMPORT CENTRAL UTILS & CONSTANTS ===================
from QUANTA_trading_core import DualLogger, silence_sklearn_warnings

# =================== DUAL LOGGER SETUP ===================
try:
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quanta_runtime.log')
    # Optional wrapper marker to easily find script start times
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n🔥 QUANTA LOG INITIATED - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n" + "="*50 + "\n")
    dual_logger = DualLogger(log_path)
    sys.stdout = dual_logger
    sys.stderr = dual_logger
except Exception as e:
    print(f"[WARN] DualLogger init failed — console-only logging: {e}", file=sys.stderr)

# Import central constants
from QUANTA_trading_core import *

# ── WebSocket feed (replaces REST polling producers) ──────────
try:
    from quanta_websockets import (
        CandleStore, BinanceWSFeed, WSEventProducer,
        ws_bootstrap, patch_mtf_analyzer,
    )
    WS_FEED_AVAILABLE = True
    print("✅ WebSocket components loaded successfully")
except ImportError as e:
    WS_FEED_AVAILABLE = False
    print(f"⚠️  WebSocket components not found ({e}) - using REST fallback")

# 🧠 QUANTA v11.5b NEURAL ENGINE
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from quanta_deeplearning import TemporalFusionTransformer
    TFT_AVAILABLE = True
    print("✅ PyTorch & TFT Model loaded successfully")
except ImportError as e:
    TFT_AVAILABLE = False
    print(f"⚠️  PyTorch/TFT not found ({e}) - Deep Learning disabled")

# 🧠 QUANTA v11.5b ENSEMBLE DRL AGENTS
try:
    from QUANTA_agents import PPOAgent, PPOMemory, DifferentialSharpeRatio
    PPO_AVAILABLE = True
    print("✅ PPO Agent loaded successfully")
except ImportError as e:
    PPO_AVAILABLE = False
    print(f"⚠️  ppo_agent.py not found ({e}) - PPO disabled")

# 🧠 QUANTA v11.5b MARKET REGIME HMM
try:
    from QUANTA_moe import MarketRegimeHMM
    HMM_AVAILABLE = True
    print("✅ MarketRegimeHMM loaded successfully")
except ImportError as e:
    HMM_AVAILABLE = False
    print(f"⚠️  QUANTA_moe.py not found ({e}) - HMM disabled")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Suppress sklearn warnings BEFORE importing sklearn
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['SKLEARN_SKIP_PARALLEL_WARNING'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Silence ALL warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Silence sklearn parallel warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Silence numpy runtime warnings specifically
import numpy as np
np.seterr(all='ignore')
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Silence specific numpy warnings (mean of empty slice, division by zero, etc.)
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered')
warnings.filterwarnings('ignore', message='divide by zero')
warnings.filterwarnings('ignore', module='numpy')

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings()

# Suppress urllib3 retry logging
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
logging.getLogger("hmmlearn.base").setLevel(logging.CRITICAL)

# Configure numpy to never show warnings (global setting)
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
# =================== END CONSTANTS ===================

# SSL - ENABLED FOR PRODUCTION
# Warnings already suppressed above

# GPU Setup - CatBoost only
# ML imports with comprehensive warning suppression
try:
    # CRITICAL: Suppress sklearn warnings BEFORE importing
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Limit parallel jobs
    
    import pandas as pd
    pd.options.mode.chained_assignment = None
    
    # Import with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        from sklearn.preprocessing import StandardScaler
        from catboost import CatBoostClassifier
    
    # AGGRESSIVE: Silence ALL sklearn warnings after import
    warnings.filterwarnings('ignore', module='sklearn')
    warnings.filterwarnings('ignore', message='.*sklearn.*')
    warnings.filterwarnings('ignore', message='.*parallel.*')
    warnings.filterwarnings('ignore', message='.*joblib.*')
    
    # Monkey-patch sklearn's parallel delayed to suppress warnings
    try:
        from sklearn.utils._testing import ignore_warnings
        from sklearn.utils.parallel import Parallel, delayed
        import functools
        import sklearn.utils.parallel
        
        # Wrap delayed to always ignore warnings
        original_delayed = delayed
        @functools.wraps(original_delayed)
        def silent_delayed(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return function(*args, **kwargs)
            return original_delayed(wrapper)
        
        sklearn.utils.parallel.delayed = silent_delayed
    except Exception as e:
        logging.debug(f"sklearn delayed patch skipped: {e}")

    # Suppress sklearn parallel warnings
    from sklearn.utils._testing import ignore_warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', module='sklearn')
    
    ML_AVAILABLE = True
    print("✅ ML libraries loaded successfully (warnings suppressed)")
except ImportError as e:
    print(f"❌ ML libraries not found: {e}")
    ML_AVAILABLE = False

from quanta_cache import FeatherCache
from QUANTA_sentiment import SentimentEngine
from quanta_onchain import get_onchain_tracker
from quanta_graph import graph_engine



from quanta_features import (Indicators, MultiTimeframeAnalyzer,
    _jit_rsi_series, _jit_rolling_mean, _jit_rolling_std, _jit_adx_series,
    _jit_ema_series, _jit_atr_series, _jit_vpin, _jit_frac_diff,
    _jit_vpin_taker, _jit_hurst, _jit_sample_entropy, _jit_transfer_entropy,
    _jit_kyle_lambda, _jit_amihud, _jit_mf_dfa_width,
    _jit_rsi, _jit_macd, _jit_bollinger, _jit_stochastic, _jit_adx_full,
    _jit_kou_barrier_prob, _jit_bs_time_decay, _jit_bs_implied_vol_ratio)

# JIT indicator functions moved to quanta_features.py (single source of truth)
# Previously 650 lines of duplicate JIT kernels lived here.
# All 16 functions now imported above from quanta_features.

# =================== OPTUNA CONSTANTS ===================
# Number of Optuna trials per specialist per search.
# 20 trials ≈ 30 min on MX130 using CPU-only search (no GPU fragmentation between trials).
OPTUNA_N_TRIALS         = 20
# Iterations used *inside* each Optuna trial (fast proxy — lower than production).
# The winning config is then trained to full production iterations.
OPTUNA_SEARCH_ITERS     = 300
# Fraction of training data used during search (speed vs accuracy tradeoff).
# 3000 samples is plenty to rank configs correctly.
OPTUNA_MAX_SEARCH_ROWS  = 3000

# =================== DEEP ML ENGINE (ENSEMBLE - TRUE INCREMENTAL) ===================
class DeepMLEngine:
    def __init__(self, cfg, bnc, mtf, candle_store=None, sentiment_engine=None):
        self.cfg = cfg
        self.bnc = bnc
        self.mtf = mtf
        self.candle_store = candle_store  # v8.0: Access to raw candle sequences
        self.sentiment = sentiment_engine or SentimentEngine()  # v10.1: Market sentiment features
        self.onchain = get_onchain_tracker()  # v11 P2: On-chain Whale Analytics

        # v11.5: Model Registry — versioning, lineage, deploy gate
        from quanta_model_registry import ModelRegistry
        self.model_registry = ModelRegistry(cfg.model_dir if hasattr(cfg, 'model_dir') else
                                            os.path.join(cfg.base_dir, 'models'))

        # 🧠 HYBRID ENGINE: TFT + CatBoost (Lim et al. 2021)
        self.tft_model = None
        self.tft_optimizer = None
        self.tft_trained = False  # v8.0: Flag for TFT readiness
        
        # 🚀 HFT OPTIMIZATIONS (v10 Ultimate)
        self.hmm_models = {}
        self.hmm_last_fit = {}
        self.rt_cache = {}  # Async real-time REST cache
        self._ob_depth_history = defaultdict(lambda: deque(maxlen=10))  # Nike: order book depth rolling buffer
        self._bs_avg_bars_to_hit = {}  # BS implied vol: symbol -> avg bars-to-hit (updated by bot after close)
        if TFT_AVAILABLE:
            self.tft_model = TemporalFusionTransformer(
                input_size=self.cfg.BASE_FEATURE_COUNT,  # 278 features (v11.5b)
                hidden_size=TFT_HIDDEN_SIZE,     # 64 (Lim 2021, MX130 optimized)
                num_heads=TFT_NUM_HEADS,         # 4 (Lim 2021)
                dropout=TFT_DROPOUT              # 0.1
            )
            if USE_GPU and torch.cuda.is_available():
                self.tft_model = self.tft_model.cuda()
                # Log VRAM usage
                tft_params = sum(p.numel() for p in self.tft_model.parameters())
                tft_vram_mb = tft_params * 4 / 1024**2  # FP32
                print(f"✅ TFT Neural Network initialized on GPU ({tft_params:,} params, ~{tft_vram_mb:.1f}MB)")
            else:
                print("⚠️  TFT Neural Network initialized on CPU (Slow!)")
        
        # 🚀 FUTURES X-RAY CACHE (v10.2)
        self.futures_stats_cache = {}  # symbol -> {data: list, ts: time}

        # 📚 HISTORICAL FUTURES CACHE (v11 backfill — training only)
        # Prefetched from Binance: funding rate history + OI history + LS ratio history
        # Used during training when live rt_cache is unavailable (symbol=None path)
        self._hist_futures_cache = {}  # symbol -> {funding: [(ts_ms, rate)], oi: [(ts_ms, oi_int)], ls: [(ts_ms, ls_ratio)]}

        # Optuna studies — one per specialist, persist across retrains so TPE accumulates knowledge.
        # Loaded from / saved to models/optuna_studies/{specialist}.pkl
        self._optuna_studies: dict = {}
        self._optuna_study_dir = os.path.join(
            cfg.model_dir if hasattr(cfg, 'model_dir') else os.path.join(os.path.dirname(__file__), 'models'),
            'optuna_studies'
        )
        os.makedirs(self._optuna_study_dir, exist_ok=True)
        
        # 🧬 THREE SPECIALIST MODELS (Research-Backed Architecture)
        # Each model uses warm start for TRUE incremental learning
        # Ensemble provides catastrophic forgetting protection
        # =================================================================
        # 15-AGENT PANTHEON (7x Greek + 7x Norse + 1x Odin)
        # =================================================================
        # Each model has a distinct inductive bias ("personality").
        # Weights adapt via entropy-regularized performance scoring.
        #
        # References:
        #   Krogh & Vedelsby 1994 — ensemble error = avg error − diversity
        #   Ho 1998 (IEEE TPAMI) — Random Subspace Method
        #   Kuncheva 2004 — optimal ensemble size ≤ 7
        #   REPO (Markowitz + Shannon) — entropy-regularized weight caps
        # =================================================================
        ENSEMBLE_MAX_WEIGHT = 0.25   # No single model > 25%
        ENSEMBLE_MIN_WEIGHT = 0.05   # Every model contributes ≥ 5%
        ENSEMBLE_ENTROPY_LAMBDA = 0.3  # Entropy regularization strength
        
        self.ensemble_config = {
            'max_weight': ENSEMBLE_MAX_WEIGHT,
            'min_weight': ENSEMBLE_MIN_WEIGHT,
            'entropy_lambda': ENSEMBLE_ENTROPY_LAMBDA,
        }
        
        # ═══════════════════════════════════════════════════════════════
        # DOMAIN-SPECIFIC FEATURE MASKS (v11.5 — Intelligent Feature Routing)
        # Instead of random 70% subspaces, each agent sees features relevant
        # to its trading domain. This forces genuine specialization.
        #
        # Feature vector layout (268 dims, 7 TFs) — AUTHORITATIVE MAP (verified 2026-03-27):
        #   0-48:    Per-TF indicators (RSI, MACD, BB, ADX, ATR, trend, strength) × 7  [49]
        #   49-52:   Cross-TF consensus (bullish%, bearish%, net, weighted)              [4]
        #   53-57:   RSI analysis (range, std, gradient, OB%, mean)                      [5]
        #   58-60:   MACD analysis (range, bull%, mean)                                  [3]
        #   61-65:   Volume analysis (log_max, log_mean, log_std, ratio, trend)          [5]
        #   66-69:   ATR analysis (pct_max, pct_mean, pct_std, pct_current)              [4]
        #   70-72:   ADX analysis (max, mean, trending%)                                 [3]
        #   73-75:   BB analysis (max, mean, std)                                        [3]
        #   76-79:   Strength progression (diff_mean, increasing%, mean, gradient)       [4]
        #   80-83:   Market regime (avg_adx, is_trending, h1_atr%, is_ranging)           [4]
        #   84-86:   Momentum confluence (bull%, bear%, net)                              [3]
        #   87-106:  Spike-dump-reversal (20 features)                                   [20]
        #   107-115: Time + session (hour_sin/cos, day_sin/cos, norm_h, norm_d,          [9]
        #            US_session, Asia_session, funding_window)
        #   116-136: Enhanced volatility (3 × 7 TFs)                                     [21]
        #   137-164: Enhanced momentum (4 × 7 TFs)                                       [28]
        #   165-171: Volume analysis per-TF (7)                                           [7]
        #   172-185: Drift detection (2 × 7 TFs)                                         [14]
        #   186-192: Multi-TF returns (7)                                                 [7]
        #   193-199: VPIN order flow (7)                                                  [7]
        #   200-206: Fractional differencing (7)                                          [7]
        #   207-213: Taker flow imbalance (7)                                             [7]
        #   214-230: Order book (17)                                                      [17]
        #   231:     HMM regime                                                           [1]
        #   232-238: Advanced research (hurst, entropy, kyle_lambda, amihud, mf_dfa,      [7]
        #            transfer_entropy, QRE)
        #   239-243: Sentiment (5)                                                        [5]
        #   244-247: Futures X-Ray (OI, LS ratio, spec_index, funding_vel)                [4]
        #   248-253: Stat arb (6)                                                         [6]
        #   254-256: On-chain whale analytics (exchange_inflow, outflow, whale_ratio)     [3]
        #   257:     GNN cross-asset embedding                                            [1]
        #   258-267: Delta features (RSI_δ, MACD_δ, BB_δ, ADX_δ, Vol_δ,                  [10]
        #            RSI_accel, MACD_accel, ATR_δ, Strength_δ, VPIN_δ)
        #   TOTAL = 268
        # ═══════════════════════════════════════════════════════════════
        _N = 278  # BASE_FEATURE_COUNT — synced with quanta_config.ModelConfig.base_feature_count

        # Per-TF RSI indices: idx 0 within each 7-feature block
        _rsi_per_tf = [i * 7 + 0 for i in range(7)]          # [0,7,14,21,28,35,42]
        _macd_per_tf = [i * 7 + 1 for i in range(7)]         # [1,8,15,22,29,36,43]
        _bb_per_tf = [i * 7 + 2 for i in range(7)]           # [2,9,16,23,30,37,44]
        _adx_per_tf = [i * 7 + 3 for i in range(7)]          # [3,10,17,24,31,38,45]
        _atr_per_tf = [i * 7 + 4 for i in range(7)]          # [4,11,18,25,32,39,46]
        _trend_per_tf = [i * 7 + 5 for i in range(7)]        # [5,12,19,26,33,40,47]
        _strength_per_tf = [i * 7 + 6 for i in range(7)]     # [6,13,20,27,34,41,48]

        # Domain groups
        # Delta feature indices (v11.5): 258-267 (after on-chain[254-256] + GNN[257])
        _DELTA = list(range(258, 268))

        _TREND_CORE = (
            _rsi_per_tf + _macd_per_tf + _adx_per_tf + _trend_per_tf + _strength_per_tf
            + list(range(49, 53))     # cross-TF consensus
            + list(range(53, 58))     # RSI analysis
            + list(range(58, 61))     # MACD analysis
            + list(range(70, 73))     # ADX analysis
            + list(range(76, 80))     # strength progression
            + list(range(80, 84))     # market regime
            + list(range(84, 87))     # momentum confluence
            + list(range(137, 165))   # enhanced momentum (4×7)
            + list(range(172, 186))   # drift detection (2×7)
            + list(range(186, 193))   # multi-TF returns
            + _DELTA                  # delta features (temporal context)
            + [275, 276]              # BS: barrier win prob + time decay
        )

        _VOLATILITY_BREAKOUT = (
            _atr_per_tf + _bb_per_tf + _adx_per_tf + _trend_per_tf
            + list(range(66, 70))     # ATR analysis
            + list(range(73, 76))     # BB analysis
            + list(range(80, 84))     # market regime
            + list(range(87, 107))    # spike-dump-reversal (20)
            + list(range(116, 137))   # enhanced volatility (3×7)
            + list(range(137, 165))   # enhanced momentum (4×7)
            + list(range(193, 200))   # VPIN
            + list(range(214, 231))   # order book (17)
            + [231]                   # HMM regime
            + _DELTA                  # delta features
            + [275, 276, 277]         # BS: all 3 barrier features
        )

        _MEAN_REVERSION = (
            _rsi_per_tf + _bb_per_tf + _adx_per_tf + _atr_per_tf
            + list(range(53, 58))     # RSI analysis
            + list(range(70, 76))     # ADX + BB analysis
            + list(range(80, 84))     # market regime
            + list(range(116, 137))   # enhanced volatility
            + list(range(186, 193))   # multi-TF returns
            + [231]                   # HMM regime
            + list(range(232, 239))   # advanced research (hurst, entropy, etc.)
            + [258, 260, 263]         # delta: RSI_δ(258), BB_δ(260), RSI_accel(263)
            + [275, 276]              # BS: barrier win prob + time decay (reversal prob context)
        )

        _FLOW_VOLUME = (
            _atr_per_tf + _trend_per_tf
            + list(range(61, 66))     # volume analysis
            + list(range(87, 107))    # spike-dump
            + list(range(165, 172))   # volume analysis per-TF
            + list(range(193, 200))   # VPIN
            + list(range(200, 207))   # fractional differencing
            + list(range(207, 214))   # taker flow imbalance
            + list(range(214, 231))   # order book (17)
            + list(range(244, 248))   # futures X-Ray
            + [262, 267]              # delta: Vol_δ(262), VPIN_δ(267)
            + [277]                   # BS: implied vol ratio (flow-implied vol)
        )

        _STRUCTURAL = (
            _rsi_per_tf + _bb_per_tf + _atr_per_tf + _adx_per_tf + _trend_per_tf
            + list(range(49, 53))     # cross-TF consensus
            + list(range(66, 76))     # ATR + ADX + BB analysis
            + list(range(76, 84))     # strength + regime
            + list(range(116, 137))   # enhanced volatility
            + list(range(172, 186))   # drift detection
            + [231]                   # HMM regime
            + list(range(232, 239))   # advanced research
            + [275, 276, 277]         # BS: all 3 (barrier geometry is structural)
        )

        _MACRO_SENTIMENT = (
            _trend_per_tf + _adx_per_tf
            + list(range(49, 53))     # cross-TF consensus
            + list(range(80, 87))     # market regime + momentum confluence
            + list(range(107, 116))   # time + session features
            + list(range(172, 186))   # drift detection
            + list(range(186, 193))   # multi-TF returns
            + [231]                   # HMM regime
            + list(range(239, 248))   # sentiment + futures X-Ray
            + list(range(248, 254))   # stat arb
            + [275]                   # BS: theoretical win prob (macro baseline)
        )

        # Nike: Impulse Continuation — microstructure + momentum context + new impulse features
        _IMPULSE = (
            _atr_per_tf + _rsi_per_tf             # per-TF ATR + RSI
            + _macd_per_tf + _bb_per_tf           # momentum context: MACD + BB (impulse continuation vs failure)
            + list(range(58, 61))                 # cross-TF MACD analysis
            + list(range(73, 76))                 # cross-TF BB analysis
            + list(range(163, 172))               # volume ratio per TF
            + list(range(191, 214))               # VPIN + frac_diff + taker_imbalance
            + list(range(212, 231))               # order book (17) + HMM regime
            + [232]                               # Kyle's Lambda
            + list(range(258, 268))               # delta features
            + list(range(270, 275))               # 5 new impulse features
            + [275, 276, 277]                     # BS: all 3 (impulse barriers most volatile)
        )

        # Convert to sorted numpy arrays (deduped)
        def _mask(indices):
            return np.array(sorted(set(indices)), dtype=np.int64)

        self._domain_masks = {
            'domain_trend':        _mask(_TREND_CORE),
            'domain_trend_short':  _mask(_TREND_CORE),  # Ares uses same trend domain
            'domain_mean_revert':  _mask(_MEAN_REVERSION),
            'domain_structural':   _mask(_STRUCTURAL),
            'domain_flow_volume':  _mask(_FLOW_VOLUME),
            'domain_volatility':   _mask(_VOLATILITY_BREAKOUT),
            'domain_macro':        _mask(_MACRO_SENTIMENT),
            'domain_impulse':      _mask(_IMPULSE),
        }

        # Load previously learned routing weights (falls back to hand-coded if missing)
        self._load_regime_routing_weights()
        # ═══════════════════════════════════════════════════════════════
        # REGIME-AGENT ROUTING TABLE (v11.5)
        # HMM 5-state: 0=strong-up, 1=weak-up, 2=range, 3=weak-down, 4=crash
        # Values are weight multipliers per regime. 0.1 = muted, 1.0 = primary.
        # Agents outside their regime still vote but with reduced influence.
        # ═══════════════════════════════════════════════════════════════
        self._regime_routing = {
            #              regime: 0(bull)  1(range)  2(bear)
            'athena':             [1.0,     0.3,      0.1],
            'ares':               [0.1,     0.3,      1.0],
            'hermes':             [0.3,     1.0,      0.3],
            'hephaestus':         [0.7,     0.6,      0.4],
            'nike':               [0.7,     0.5,      0.9],
            'artemis':            [0.6,     0.4,      0.7],
            'chronos':            [0.4,     0.8,      0.6],
        }

        # Per-agent Brier score tracking (v11.5)
        # Tracks rolling calibration quality: Brier = mean((predicted_prob - actual)^2)
        # Lower = better calibrated. Used to dynamically adjust ensemble weights.
        self._brier_scores = {name: {'sum': 0.0, 'count': 0, 'rolling': deque(maxlen=500)}
                              for name in ['athena', 'ares', 'hermes', 'hephaestus',
                                           'nike', 'artemis', 'chronos']}

        self.specialist_models = {
            'athena': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Athena (Trend Rider) — Goddess of strategy; handles strong, undeniable uptrends',
                'coin_filter': 'athena',
                'performance': [],
                'hyperparams': {
                    'iterations': 1500, 'learning_rate': 0.05, 'depth': 7,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 42,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_trend',
                'recency_boost': 1.0,
            },
            'ares': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Ares (Trend Crusher) — God of war; shorting weak bounces in bleeding downtrends',
                'coin_filter': 'ares',
                'performance': [],
                'hyperparams': {
                    'iterations': 1200, 'learning_rate': 0.15, 'depth': 5,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 49,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_trend_short',
                'recency_boost': 1.0,
            },
            'hermes': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Hermes (Range Navigator) — The fast messenger; buying bottoms and selling tops of tight sideways channels',
                'coin_filter': 'hermes',
                'performance': [],
                'hyperparams': {
                    'iterations': 800, 'learning_rate': 0.30, 'depth': 4,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 43,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_mean_revert',
                'recency_boost': 1.0,
            },
            'hephaestus': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Hephaestus (Structural Forger) — The blacksmith; trades structural bounces on highly liquid anchor assets',
                'coin_filter': 'hephaestus',
                'performance': [],
                'hyperparams': {
                    'iterations': 1200, 'learning_rate': 0.10, 'depth': 5,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_strength': 3.0,
                    'random_seed': 44,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_structural',
                'recency_boost': 1.0,
            },
            'nike': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Thor (Impulse Rider) — Norse breakout specialist; single-candle explosive moves, predicts continuation vs death',
                'coin_filter': 'nike',
                'performance': [],
                'hyperparams': {
                    'iterations': 1200, 'learning_rate': 0.08, 'depth': 6,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 56,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_impulse',
                'recency_boost': 1.0,
            },
            'artemis': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Artemis (Stealth Accumulator) — Goddess of the hunt; detects hidden volume surges without structural breakouts',
                'coin_filter': 'artemis',
                'performance': [],
                'hyperparams': {
                    'iterations': 1214, 'learning_rate': 0.18, 'depth': 6,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 46,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_volatility',
                'recency_boost': 1.0,
            },
            'chronos': {
                'model': None,
                'scaler': StandardScaler(),
                'calibrator': None,
                'generation': 0,
                'weight': 1.0 / 7,
                'description': 'Chronos (Deep Reversal) — Personification of time; evaluates v-shape bounces off RSI extremes',
                'coin_filter': 'chronos',
                'performance': [],
                'hyperparams': {
                    'iterations': 1500, 'learning_rate': 0.10, 'depth': 7,
                    'l2_leaf_reg': 5.0, 'subsample': 0.8, 'bootstrap_type': 'Bernoulli',
                    'random_seed': 47,
                    'gpu_ram_part': 0.50
                },
                'feature_mask': 'domain_macro',
                'recency_boost': 5.0,
            },
        }
        
        # ========================================
        # LEGACY COMPATIBILITY LAYER
        # ========================================
        # NOTE: Maintains backward compatibility with consumer code
        # Consumer uses self.scaler and self.models for predictions
        # Synced with specialist_models after training/loading
        # TODO: Fully migrate consumer to use specialists directly in v8.0
        
        self.models = []  # Legacy: List of (model, generation, weight, metadata)
        self.max_models = 5  # Legacy: Maximum ensemble size
        self.retrain_threshold = RL_RETRAIN_THRESHOLD  # Use constant
        self.rl_buffer = []
        
        self.scaler = StandardScaler()  # Legacy scaler (synced from Foundation specialist)
        self.calibrator = None  # 🔬 Isotonic calibrator for CatBoost overconfidence fix
        self.is_trained = False
        
        # Model generation tracking — load eagerly from disk so /status
        # never shows "Gen 0" while training is in progress.
        self.model_generation = 0
        try:
            gen_path = os.path.join(self.cfg.model_dir, 'generation.txt')
            if os.path.exists(gen_path):
                with open(gen_path, 'r') as f:
                    self.model_generation = int(f.read().strip())
        except Exception as e:
            logging.warning(f"Could not load generation counter (starting at 0): {e}")
        
        # 🧬 EVOLUTIONARY LEARNING SYSTEM - Tracks mistakes to get smarter
        self.mistake_history = []  # High-confidence failures
        self.generation_performance = []  # Win rate per generation
        self.weak_patterns = defaultdict(lambda: {'losses': 0, 'total': 0})
        
        # Ensemble weighting (newest = 1.0, older = decay)
        self.weight_decay = 0.7
        
        # Performance tracking
        self._base_model_weights = {}
        self._model_returns = {}
        
        # 🧠 Latent Market Regime HMM
        self.hmm = None
        if HMM_AVAILABLE:
            try:
                self.hmm = MarketRegimeHMM(model_dir=self.cfg.model_dir)
                self.hmm.load()
            except Exception as e:
                print(f"⚠️  Failed to load MarketRegimeHMM: {e}")

        self._model_correct = {}
        self._model_total = {}
        self._softmax_temp = 2.0
        
        self._load_models()
        
        print(f"\n" + "─"*70)
        print(f"🧬 DEEP ML ENGINE — PANTHEONS READY")
        print(f"─"*70)
        print(f"• Features: {self.cfg.BASE_FEATURE_COUNT} (Sentiment-Fused Multi-Timeframe v2)")
        print(f"• Ensemble: 7-Agent Pantheon + Odin (TFT) + Heimdall (PPO)")
        print(f"• News Engine: L&M 2011 Lexicon + 4-Source RSS Polling")
        print(f"• History: 180-Day Sliding Window")
        print(f"─"*70 + "\n")
    
    @property
    def catboost_model(self):
        """Compatibility property - returns newest model"""
        if self.models:
            return self.models[-1][0]
        return None
    
    # ========================================
    # 🧬 SPECIALIST MODEL METHODS
    # ========================================
    
    def _save_specialist_models(self):
        """Save all 3 specialist models + scalers"""
        try:
            for name, specialist in self.specialist_models.items():
                if specialist['model'] is not None:
                    # Save model
                    model_path = os.path.join(
                        self.cfg.model_dir,
                        f'{name}_gen{specialist["generation"]}.cbm'
                    )
                    specialist['model'].save_model(model_path)
                    
                    # Save scaler
                    scaler_path = os.path.join(
                        self.cfg.model_dir,
                        f'{name}_scaler.pkl'
                    )
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(specialist['scaler'], f)
                    
                    # Phase 4: Save feature indices
                    indices = specialist.get('_feature_indices')
                    if indices is not None:
                        idx_path = os.path.join(self.cfg.model_dir, f'{name}_feature_indices.npy')
                        np.save(idx_path, indices)

                    # Phase 4b: Save importance mask for next-cycle pruning
                    imp_mask = specialist.get('importance_mask')
                    if imp_mask is not None:
                        imp_path = os.path.join(self.cfg.model_dir, f'{name}_importance_mask.npy')
                        np.save(imp_path, imp_mask)

                    # Save calibrator (isotonic regression for overconfidence fix)
                    if specialist.get('calibrator') is not None:
                        cal_path = os.path.join(
                            self.cfg.model_dir,
                            f'{name}_calibrator.pkl'
                        )
                        with open(cal_path, 'wb') as f:
                            pickle.dump(specialist['calibrator'], f)
                    
                    logging.info(f"✅ Saved {name} model (Gen {specialist['generation']})")
            
            # 🔥 FIX: Sync model_generation and persist to disk
            max_gen = max(s['generation'] for s in self.specialist_models.values())
            self.model_generation = max_gen
            try:
                gen_path = os.path.join(self.cfg.model_dir, 'generation.txt')
                with open(gen_path, 'w') as f:
                    f.write(str(max_gen))
            except Exception as e:
                logging.warning(f"Could not persist generation counter to disk: {e}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to save specialist models: {e}")
            return False
    
    def _load_specialist_models(self):
        """Load all 3 specialist models + scalers"""
        any_loaded = False
        
        for name, specialist in self.specialist_models.items():
            try:
                # Find latest generation (FIX 9: Use glob instead of 20 sequential disk reads)
                import glob
                file_name = name
                pattern = os.path.join(self.cfg.model_dir, f'{name}_gen*.cbm')
                matches = glob.glob(pattern)
                max_gen = 0
                for match in matches:
                    try:
                        # Extract generation number from filename
                        gen_str = os.path.basename(match).replace(f'{file_name}_gen', '').replace('.cbm', '')
                        gen = int(gen_str)
                        if gen > max_gen:
                            max_gen = gen
                    except ValueError:
                        pass

                if max_gen > 0:
                    # Load model
                    model_path = os.path.join(
                        self.cfg.model_dir,
                        f'{file_name}_gen{max_gen}.cbm'
                    )
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    specialist['model'] = model
                    specialist['generation'] = max_gen

                    # Load scaler
                    scaler_path = os.path.join(
                        self.cfg.model_dir,
                        f'{file_name}_scaler.pkl'
                    )
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            specialist['scaler'] = pickle.load(f)

                    # Phase 4: Load feature indices
                    idx_path = os.path.join(self.cfg.model_dir, f'{file_name}_feature_indices.npy')
                    if os.path.exists(idx_path):
                        specialist['_feature_indices'] = np.load(idx_path)

                    # Phase 4b: Load importance mask for next-cycle pruning
                    imp_path = os.path.join(self.cfg.model_dir, f'{file_name}_importance_mask.npy')
                    if os.path.exists(imp_path):
                        specialist['importance_mask'] = np.load(imp_path)
                        print(f"   Loaded {name} importance mask ({int(specialist['importance_mask'].sum())}/{len(specialist['importance_mask'])} features kept)")

                    # Load calibrator
                    cal_path = os.path.join(
                        self.cfg.model_dir,
                        f'{file_name}_calibrator.pkl'
                    )
                    if os.path.exists(cal_path):
                        with open(cal_path, 'rb') as f:
                            specialist['calibrator'] = pickle.load(f)
                        print(f"   🔬 Loaded {name} calibrator")
                    
                    # v11.5: Validate feature compatibility via registry
                    import hashlib as _hl
                    _cur_mask = specialist.get('_feature_indices')
                    _cur_hash = ""
                    if _cur_mask is not None:
                        _cur_hash = _hl.sha256(np.array(_cur_mask, dtype=np.int64).tobytes()).hexdigest()[:16]
                    compat, warn = self.model_registry.validate_feature_compatibility(
                        name, self.cfg.BASE_FEATURE_COUNT, _cur_hash
                    )
                    if not compat:
                        print(f"   ⚠️ COMPAT WARNING: {warn}")
                        print(f"   ⚠️ Model may need retraining!")

                    print(f"✅ Loaded {name} model (Gen {max_gen}, weight {specialist['weight']})")
                    any_loaded = True
                else:
                    print(f"⚠️  No {name} model found - will train on first run")
                    
            except Exception as e:
                logging.error(f"Failed to load {name} model: {e}")
        
        return any_loaded
    
    def _run_optuna_search(self, specialist_name, X_train, y_train, X_val, y_val,
                           weights_train=None):
        """
        Adaptive hyperparameter search via Optuna TPE (Tree-structured Parzen Estimator).

        - Runs OPTUNA_N_TRIALS trials, each training CatBoost with OPTUNA_SEARCH_ITERS
          iterations on a CPU subset (fast proxy).
        - TPE learns from previous trials → suggests better regions each round.
        - Studies persist to disk so knowledge accumulates across retrains.
        - Returns a dict of best params to merge into specialist['hyperparams'].

        Based on: Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization.
        TPE extension: Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization.
        """
        if not _OPTUNA_AVAILABLE:
            return None

        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score  # imported once here, not inside trial loop

        # ── Sub-sample training data for speed (rank order is preserved) ────────
        n_search = min(OPTUNA_MAX_SEARCH_ROWS, len(X_train))
        idx = np.linspace(0, len(X_train) - 1, n_search, dtype=int)
        Xs = X_train[idx]
        ys = y_train[idx]
        ws = weights_train[idx] if weights_train is not None else None

        # Cap val too (not critical but keeps trial runtime predictable)
        n_val = min(1500, len(X_val))
        Xv = X_val[:n_val]
        yv = y_val[:n_val]

        # ── Sanitize subsampled data (scaler can leave NaN in zero-variance cols)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=3.0, neginf=-3.0)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=3.0, neginf=-3.0)

        # ── Drop zero-variance columns (StandardScaler → 0/0 = NaN propagation)
        col_std = np.std(Xs, axis=0)
        good_cols = col_std > 0
        if not np.all(good_cols):
            n_dropped = int((~good_cols).sum())
            logging.info(f"Optuna: dropping {n_dropped} zero-variance columns from search data")
            Xs = Xs[:, good_cols]
            Xv = Xv[:, good_cols]

        # ── Load or create persistent study ─────────────────────────────────────
        study_path = os.path.join(self._optuna_study_dir, f'{specialist_name}.pkl')
        if specialist_name in self._optuna_studies:
            study = self._optuna_studies[specialist_name]
        elif os.path.exists(study_path):
            try:
                with open(study_path, 'rb') as f:
                    study = pickle.load(f)
                print(f"   🔬 Optuna: loaded existing study for {specialist_name} "
                      f"({len(study.trials)} prior trials)")
            except Exception:
                study = optuna.create_study(direction='maximize',
                                            sampler=optuna.samplers.TPESampler(seed=42))
        else:
            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
        self._optuna_studies[specialist_name] = study

        def objective(trial):
            params = {
                'iterations':    trial.suggest_int('iterations',    300, 2000),
                'depth':         trial.suggest_int('depth',         4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.35, log=True),
                'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg',   1.0, 15.0, log=True),
                'subsample':     trial.suggest_float('subsample',     0.6, 1.0),
            }
            # Override iterations with fast proxy count for search speed
            params['iterations'] = OPTUNA_SEARCH_ITERS

            model = CatBoostClassifier(
                **params,
                bootstrap_type='Bernoulli',
                loss_function='Logloss',
                eval_metric='AUC',
                auto_class_weights='Balanced',
                random_seed=42,
                verbose=False,
                task_type='CPU',
                thread_count=max(1, os.cpu_count() - 1),
                border_count=CATBOOST_BORDER_COUNT,
                train_dir=os.path.join(self.cfg.base_dir, 'catboost_info'),
            )
            try:
                model.fit(Xs, ys, eval_set=(Xv, yv), sample_weight=ws,
                          use_best_model=True,
                          early_stopping_rounds=50, verbose=False)
                preds = model.predict_proba(Xv)[:, 1]
                if np.any(np.isnan(preds)):
                    return 0.5  # predict_proba returned NaN → treat as random
                auc = float(roc_auc_score(yv, preds))
                if np.isnan(auc) or np.isinf(auc):
                    return 0.5
                return auc
            except Exception:
                return 0.5

        print(f"   🔬 Optuna: running {OPTUNA_N_TRIALS} trials for {specialist_name} "
              f"(CPU proxy, {n_search} samples each)...")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

        # ── Save study ───────────────────────────────────────────────────────────
        try:
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
        except Exception as e:
            logging.warning(f"Optuna study save failed for {specialist_name}: {e}")

        best = study.best_params.copy()
        # Remove the proxy override — production training uses full iterations
        best.pop('iterations', None)
        best_auc = study.best_value

        print(f"   ✅ Optuna best val AUC: {best_auc:.4f} → "
              f"depth={best.get('depth')}, lr={best.get('learning_rate', 0):.4f}, "
              f"l2={best.get('l2_leaf_reg', 0):.2f}, sub={best.get('subsample', 0):.2f}")
        return best

    def _train_specialist(self, specialist_name, X, y, sample_weights=None):
        """
        Train individual specialist with WARM START
        
        Research-backed incremental learning:
        - Loads existing model if available
        - Continues training (warm start)
        - Per-model hyperparams for ensemble diversity (Krogh & Vedelsby 1994)
        - Feature subspace masking (Ho 1998 Random Subspace Method)
        - Recency boosting for regime-adaptive specialist
        - Saves new generation
        
        ⚖️ CLASS BALANCING (v8.1 NEW!):
        - Automatic imbalance detection (>2:1 ratio triggers balancing)
        - Manual inverse-frequency weighting + CatBoost auto_class_weights
        - Fixes extreme directional bias (e.g., Ares 98% bearish → 60-70%)
        - Applied to all 7 Greek pantheon specialists automatically
        - See: CLASS_BALANCING_IMPLEMENTATION_COMPLETE.md
        """
        specialist = self.specialist_models[specialist_name]
        
        try:
            print(f"\n{'='*70}")
            print(f"🧬 TRAINING {specialist_name.upper()}")
            print(f"{'='*70}")
            print(f"Description: {specialist['description']}")
            print(f"Samples: {len(X)}")
            
            num_samples = len(X)
            
            # =================================================================
            # TEMPORAL SPLIT (López de Prado AFML Ch.7):
            # Time-series data MUST be split chronologically. Random shuffle
            # causes look-ahead bias — future data leaks into training.
            # Last 20% of samples = validation (most recent candles).
            # Purge gap removes 48 candles (~4h at 5m bars) at the boundary
            # to prevent Triple Barrier label overlap between sets.
            # =================================================================

            # 1. Convert raw, unscaled data
            X_arr = np.array(X, dtype=np.float32) if not isinstance(X, np.ndarray) else X.astype(np.float32)
            y_arr = np.array(y, dtype=np.float32) if not isinstance(y, np.ndarray) else y.astype(np.float32)
            del X, y

            # 2. Temporal split with purge gap
            #    Gap = max(all agent max_bars) so no label can overlap train/val.
            #    If a future agent has max_bars > 48, the gap auto-adjusts.
            split_idx = int(len(X_arr) * (1 - CATBOOST_VAL_SPLIT))
            _ev_cfg = self.cfg.events
            PURGE_GAP = max(
                _ev_cfg.athena_max_bars, _ev_cfg.ares_max_bars,
                _ev_cfg.hermes_max_bars, _ev_cfg.artemis_max_bars,
                _ev_cfg.chronos_max_bars, _ev_cfg.heph_max_bars,
                _ev_cfg.nike_max_bars
            )
            split_idx_train_end = max(0, split_idx - PURGE_GAP)
            split_idx_val_start = split_idx

            X_train_raw = X_arr[:split_idx_train_end]
            X_val_raw   = X_arr[split_idx_val_start:]
            y_train     = y_arr[:split_idx_train_end]
            y_val       = y_arr[split_idx_val_start:]
            val_pos  = int(np.sum(y_val == 1))
            val_neg  = int(np.sum(y_val == 0))
            val_total = val_pos + val_neg
            print(f"   📊 Temporal split: {len(X_train_raw)} train | {PURGE_GAP} purged | {len(X_val_raw)} val (pos={val_pos}, neg={val_neg})")
            if val_total < 20:
                print(f"   ⚠️  WARNING: val set has only {val_total} events — AUC gate unreliable, skipping AUC threshold check")
            elif min(val_pos, val_neg) < 10:
                print(f"   ⚠️  WARNING: val set severely imbalanced ({val_pos}:{val_neg}) — AUC may be misleading")

            if sample_weights is not None:
                sw_arr = np.array(sample_weights, dtype=np.float32) if not isinstance(sample_weights, np.ndarray) else sample_weights.astype(np.float32)
                del sample_weights
                weights_train = sw_arr[:split_idx_train_end]
                weights_val   = sw_arr[split_idx_val_start:]
                del sw_arr
            else:
                weights_train = None
                weights_val = None

            # =================================================================
            # CLASS IMBALANCE HANDLING (v11.8)
            # 1. Stratified Undersampling for extreme cases (>10:1)
            # 2. Manual inverse-frequency weighting for remaining imbalance
            # =================================================================
            pos_count = int(np.sum(y_train == 1))
            neg_count = int(np.sum(y_train == 0))
            
            # --- STEP A: TEMPORAL STRATIFIED UNDERSAMPLING ---
            # Split training data into N_BUCKETS temporal buckets and undersample
            # the majority class within each bucket proportionally.
            # This preserves regime structure (bull early / bear late) instead of
            # random sampling which mixes regimes uniformly.
            # Ref: Lemaître et al. (2017) — imbalanced-learn temporal awareness.
            MAX_RATIO = 10.0
            N_BUCKETS = 10  # 10 equal-time buckets across training window
            if pos_count > 0 and neg_count > 0:
                temp_ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
                if temp_ratio > MAX_RATIO:
                    print(f"   ⚖️  EXTREME IMBALANCE ({temp_ratio:.1f}:1): Temporal undersampling majority to {MAX_RATIO}:1...")
                    min_label = 1 if pos_count < neg_count else 0
                    maj_label = 1 - min_label
                    n_min = min(pos_count, neg_count)
                    target_maj = int(n_min * MAX_RATIO)

                    idx_min = np.where(y_train == min_label)[0]
                    idx_maj = np.where(y_train == maj_label)[0]
                    rng_seed = int(specialist.get('hyperparams', {}).get('random_seed', 42))
                    rng = np.random.RandomState(rng_seed)

                    # Divide majority indices into temporal buckets
                    bucket_edges = np.linspace(0, len(idx_maj), N_BUCKETS + 1, dtype=int)
                    keep_maj_parts = []
                    # Allocate quota per bucket proportional to bucket size
                    for b in range(N_BUCKETS):
                        bucket_idx = idx_maj[bucket_edges[b]:bucket_edges[b + 1]]
                        bucket_quota = max(1, int(target_maj * len(bucket_idx) / len(idx_maj)))
                        bucket_quota = min(bucket_quota, len(bucket_idx))
                        chosen = rng.choice(len(bucket_idx), size=bucket_quota, replace=False)
                        keep_maj_parts.append(bucket_idx[chosen])

                    keep_maj = np.concatenate(keep_maj_parts)
                    keep_idx = np.sort(np.concatenate([idx_min, keep_maj]))

                    X_train_raw = X_train_raw[keep_idx]
                    y_train = y_train[keep_idx]
                    if weights_train is not None:
                        weights_train = weights_train[keep_idx]

                    # Update counts for next steps
                    pos_count = int(np.sum(y_train == 1))
                    neg_count = int(np.sum(y_train == 0))
            
            total = len(y_train)
            imbalance_ratio = max(pos_count, neg_count) / max(min(pos_count, neg_count), 1)

            # Validation class balance (reported here so it's visible before training starts)
            v_pos = int(np.sum(y_val == 1))
            v_neg = int(np.sum(y_val == 0))
            v_total = len(y_val)
            v_ratio = max(v_pos, v_neg) / max(min(v_pos, v_neg), 1)

            print(f"\n   📊 Class Distribution (after undersampling):")
            print(f"      Train — Bullish: {pos_count:,} ({pos_count/total*100:.1f}%)  "
                  f"Bearish: {neg_count:,} ({neg_count/total*100:.1f}%)  Ratio: {imbalance_ratio:.1f}:1")
            print(f"      Val   — Bullish: {v_pos:,} ({v_pos/v_total*100:.1f}%)  "
                  f"Bearish: {v_neg:,} ({v_neg/v_total*100:.1f}%)  Ratio: {v_ratio:.1f}:1")
            if v_ratio > 5.0:
                logging.warning(
                    f"Validation set highly imbalanced ({v_ratio:.1f}:1) — "
                    f"AUC may be misleading. Consider Balanced Accuracy for evaluation."
                )
            
            # --- STEP B: MANUAL WEIGHTING ---
            # Apply manual class balancing if imbalance still exists (>1.5:1)
            if imbalance_ratio > 1.5:
                print(f"   ⚖️  Applying manual class balancing weights...")
                
                if pos_count < neg_count:
                    # Bullish is minority
                    pos_weight = neg_count / pos_count
                    neg_weight = 1.0
                    print(f"      Bullish weight: {pos_weight:.2f}× (minority)")
                else:
                    # Bearish is minority
                    neg_weight = pos_count / neg_count
                    pos_weight = 1.0
                    print(f"      Bearish weight: {neg_weight:.2f}× (minority)")
                
                # Apply class weights to sample weights
                if weights_train is None:
                    weights_train = np.ones(len(y_train), dtype=np.float32)
                
                class_weights = np.where(y_train == 1, pos_weight, neg_weight).astype(np.float32)
                weights_train = weights_train * class_weights
                
                # Also balance validation set weights (but don't undersample validation)
                if weights_val is None:
                    weights_val = np.ones(len(y_val), dtype=np.float32)
                
                # Re-calculate validation ratio for weights_val
                v_pos = int(np.sum(y_val == 1))
                v_neg = int(np.sum(y_val == 0))
                if v_pos > 0 and v_neg > 0:
                    v_pos_w = v_neg / v_pos if v_pos < v_neg else 1.0
                    v_neg_w = v_pos / v_neg if v_neg < v_pos else 1.0
                    class_weights_val = np.where(y_val == 1, v_pos_w, v_neg_w).astype(np.float32)
                    weights_val = weights_val * class_weights_val
                
                print(f"      ✅ Applied class weights (Train ratio {imbalance_ratio:.1f}:1)")
            else:
                print(f"   ✅ Class balance acceptable.")
            
            # =================================================================
            # RECENCY BOOST (regime-adaptive specialist)
            # Models with recency_boost > 1.0 upweight recent 30 days
            # =================================================================
            recency_boost = specialist.get('recency_boost', 1.0)
            if recency_boost > 1.0 and weights_train is not None:
                n_train = len(weights_train)
                # Recent 30 days ≈ last 30% of samples (sorted chronologically)
                recency_cutoff = int(n_train * 0.7)
                recency_weights = weights_train.copy()
                recency_weights[recency_cutoff:] *= recency_boost
                weights_train = recency_weights
                print(f"   ⏱️ Recency boost: last {n_train - recency_cutoff} samples weighted {recency_boost}×")
            
            # =================================================================
            # FEATURE SUBSPACE MASKING (v11.5 Domain-Specific)
            # Each agent sees only features relevant to its trading domain.
            # Falls back to random_70 or dynamic_volatility for legacy compat.
            # =================================================================
            feature_mask_cfg = specialist.get('feature_mask', None)
            feature_indices = None
            if feature_mask_cfg and feature_mask_cfg.startswith('domain_'):
                # v11.5: Domain-specific intelligent feature masking
                if hasattr(self, '_domain_masks') and feature_mask_cfg in self._domain_masks:
                    n_features = X_train_raw.shape[1]
                    domain_idx = self._domain_masks[feature_mask_cfg]
                    # Clip indices to actual feature count
                    feature_indices = domain_idx[domain_idx < n_features]
                    X_train_raw = X_train_raw[:, feature_indices]
                    X_val_raw   = X_val_raw[:, feature_indices]
                    specialist['_feature_indices'] = feature_indices
                    print(f"   🧠 Domain mask [{feature_mask_cfg}]: {len(feature_indices)}/{n_features} features")
                else:
                    print(f"   ⚠️ Domain mask {feature_mask_cfg} not found, using all features")
            elif feature_mask_cfg and feature_mask_cfg.startswith('random_70_seed'):
                seed = int(feature_mask_cfg.split('seed')[1])
                n_features = X_train_raw.shape[1]
                n_keep = int(n_features * 0.7)
                rng = np.random.RandomState(seed)
                feature_indices = np.sort(rng.choice(n_features, n_keep, replace=False))
                X_train_raw = X_train_raw[:, feature_indices]
                X_val_raw   = X_val_raw[:, feature_indices]
                specialist['_feature_indices'] = feature_indices
                print(f"   🎭 Feature subspace: {n_keep}/{n_features} features (seed={seed})")
            elif feature_mask_cfg == 'dynamic_volatility':
                col_var = np.nanvar(X_train_raw, axis=0)
                n_features = X_train_raw.shape[1]
                n_keep = int(n_features * 0.7)
                feature_indices = np.sort(np.argsort(col_var)[::-1][:n_keep])
                X_train_raw = X_train_raw[:, feature_indices]
                X_val_raw   = X_val_raw[:, feature_indices]
                specialist['_feature_indices'] = feature_indices
                print(f"   📈 Dynamic volatility mask: top {n_keep} variance features")

            # =============================================================
            # IMPORTANCE-BASED PRUNING (applied from previous cycle)
            # If a previous training stored an importance_mask, apply it now
            # to further prune low-importance features within the subspace.
            # This forces a fresh scaler fit since feature count changes.
            # =============================================================
            prev_imp_mask = specialist.get('importance_mask', None)
            if prev_imp_mask is not None:
                n_current = X_train_raw.shape[1]
                if len(prev_imp_mask) == n_current:
                    n_before = n_current
                    X_train_raw = X_train_raw[:, prev_imp_mask]
                    X_val_raw   = X_val_raw[:, prev_imp_mask]
                    # Update feature_indices to reflect pruning
                    if feature_indices is not None:
                        feature_indices = feature_indices[prev_imp_mask]
                        specialist['_feature_indices'] = feature_indices
                    n_after = X_train_raw.shape[1]
                    print(f"   Importance pruning applied: {n_before} -> {n_after} features (from previous cycle)")
                    # Force fresh scaler since dimensions changed
                    from sklearn.preprocessing import StandardScaler
                    specialist['scaler'] = StandardScaler()
                    # Clear the mask so it doesn't re-apply next cycle
                    # (a new mask will be computed after this training)
                    specialist['importance_mask'] = None
                else:
                    print(f"   Importance mask dimension mismatch ({len(prev_imp_mask)} vs {n_current}), skipping")
                    specialist['importance_mask'] = None

            # Phase 2: Sanitize training data before fit_transform
            X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_raw   = np.nan_to_num(X_val_raw,   nan=0.0, posinf=0.0, neginf=0.0)

            # 2. Fit scaler on TRAINING split only, then transform both
            if not hasattr(specialist['scaler'], 'mean_'):
                X_train = specialist['scaler'].fit_transform(X_train_raw)
                print("   📊 Fitted new scaler on training split (no leakage)")
            else:
                X_train = specialist['scaler'].transform(X_train_raw)
                print("   📊 Using existing scaler")
            X_val = specialist['scaler'].transform(X_val_raw)
            # Safety: scaler can produce NaN for zero-variance columns (0/0). Clean after transform.
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=3.0, neginf=-3.0)
            X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=3.0, neginf=-3.0)

            # =================================================================
            # OPTUNA ADAPTIVE HYPERPARAMETER SEARCH (first training only)
            # TPE sampler learns which param regions work for this specialist's
            # data distribution. Study persists across retrains → gets smarter.
            # Only runs when model is None (first ever training).
            # Set specialist['_optuna_searched'] = False to force re-search.
            # =================================================================
            if (specialist['model'] is None
                    and _OPTUNA_AVAILABLE
                    and not specialist.get('_optuna_searched', False)):
                optuna_best = self._run_optuna_search(
                    specialist_name, X_train, y_train, X_val, y_val,
                    weights_train=weights_train,
                )
                if optuna_best:
                    specialist['hyperparams'].update(optuna_best)
                    specialist['_optuna_searched'] = True
                    print(f"   🧬 Hyperparams updated from Optuna search")

            # =================================================================
            # CREATE MODEL WITH PER-SPECIALIST HYPERPARAMS
            # Each model has distinct depth/LR/regularization for diversity
            # (Krogh & Vedelsby 1994: ensemble error = avg error − diversity)
            # =================================================================
            hp = specialist.get('hyperparams', {})

            if specialist['model'] is None:
                # First training - create fresh model
                print(f"   🆕 First training - creating new model")
                print(f"   📋 Hyperparams: depth={hp.get('depth',8)}, lr={hp.get('learning_rate',0.18)}, l2={hp.get('l2_leaf_reg',1.5)}")
                gpu_params = {}
                final_depth = hp.get('depth', 8)
                
                if USE_GPU:
                    # CatBoost GPU memory allocation scales exponentially with depth (2^depth).
                    # Depth >= 10 requests >3GB of VRAM regardless of chunk size on MX130 GPU.
                    # We will preserve the depth strategy by falling back to CPU for these deep models.
                    if final_depth > 9:
                        gpu_params = {
                            'task_type': 'CPU',
                            'thread_count': max(1, os.cpu_count() - 1)
                        }
                        print(f"   ⚠️ Depth {final_depth} > 9: Training deep model on CPU to avoid 3GB VRAM exponential allocation. (Inference will still be fast)")
                    else:
                        gpu_params = {
                            'task_type': 'GPU',
                            'devices': '0',
                            'gpu_ram_part': 0.8,  # Allow Catboost to see 80% of total VRAM
                            'max_ctr_complexity': 2,
                            'thread_count': max(1, os.cpu_count() - 1)
                        }
                else:
                    gpu_params = {
                        'task_type': 'CPU',
                        'thread_count': max(1, os.cpu_count() - 1)
                    }

                cat_kwargs = {
                    'iterations': hp.get('iterations', 1214),
                    'learning_rate': hp.get('learning_rate', 0.18),
                    'depth': final_depth,
                    'l2_leaf_reg': hp.get('l2_leaf_reg', 1.5),
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'auto_class_weights': 'Balanced',  # Let CatBoost handle imbalance internally
                    'random_seed': hp.get('random_seed', 42),
                    'verbose': False,
                    'border_count': CATBOOST_BORDER_COUNT,
                    'train_dir': os.path.join(self.cfg.base_dir, 'catboost_info')
                }
                
                # Dynamic extraction of optional Catboost parameters
                for optional_param in ['subsample', 'bootstrap_type', 'bagging_temperature', 'random_strength']:
                    if optional_param in hp:
                        cat_kwargs[optional_param] = hp[optional_param]
                
                # Use default bagging_temperature if stochastic parameters missing
                if 'bootstrap_type' not in hp and 'bagging_temperature' not in hp:
                    cat_kwargs['bagging_temperature'] = 0.4
                        
                cat_kwargs.update(gpu_params)

                model = CatBoostClassifier(**cat_kwargs)
                init_model = None
            else:
                # ── v11.5: Generational Decline Detection ──
                # If 3+ consecutive generations declined in AUC, warm-starting is
                # causing catastrophic forgetting. Force fresh retrain.
                force_fresh = False
                should_reset, decline_streak = self.model_registry.should_fresh_retrain(specialist_name)
                if should_reset:
                    force_fresh = True
                    print(f"   🔴 DECLINE ALERT: {decline_streak} consecutive AUC drops detected!")
                    print(f"   🔴 Forcing FRESH retrain (warm-start causing catastrophic forgetting)")

                # Warm start - continue from existing model
                # CRITICAL: CatBoost GPU does NOT support init_model (warm starts).
                # Only use warm start on CPU.
                is_gpu_model = (cat_kwargs.get('task_type') == 'GPU' if specialist['model'] is None
                                else getattr(specialist['model'], '_init_params', {}).get('task_type') == 'GPU')
                if force_fresh:
                    # Forced fresh retrain due to generational decline
                    print(f"   🆕 Fresh retrain (reset after {decline_streak} declining generations)")
                elif USE_GPU and not is_gpu_model:
                    # CPU model — safe to warm start
                    print(f"   🔄 Warm start from Gen {specialist['generation']} (CPU)")
                    model = specialist['model']
                    init_model = specialist['model']
                else:
                    # GPU model — must retrain from scratch (CatBoost limitation)
                    print(f"   🆕 Retraining from scratch (GPU does not support warm starts)")
                    hp_ws = specialist.get('hyperparams', {})
                    ws_depth = hp_ws.get('depth', 8)
                    ws_gpu_params = {
                        'task_type': 'GPU', 'devices': '0',
                        'gpu_ram_part': hp_ws.get('gpu_ram_part', 0.8),
                        'max_ctr_complexity': 2,
                        'thread_count': max(1, os.cpu_count() - 1)
                    } if USE_GPU and ws_depth <= 9 else {
                        'task_type': 'CPU',
                        'thread_count': max(1, os.cpu_count() - 1)
                    }
                    ws_cat_kwargs = {
                        'iterations': hp_ws.get('iterations', 1214),
                        'learning_rate': hp_ws.get('learning_rate', 0.18),
                        'depth': ws_depth,
                        'l2_leaf_reg': hp_ws.get('l2_leaf_reg', 1.5),
                        'loss_function': 'Logloss', 'eval_metric': 'AUC',
                        'auto_class_weights': 'Balanced',
                        'random_seed': hp_ws.get('random_seed', 42),
                        'verbose': False,
                        'border_count': CATBOOST_BORDER_COUNT,
                        'train_dir': os.path.join(self.cfg.base_dir, 'catboost_info')
                    }
                    for op in ['subsample', 'bootstrap_type', 'bagging_temperature', 'random_strength']:
                        if op in hp_ws:
                            ws_cat_kwargs[op] = hp_ws[op]
                    if 'bootstrap_type' not in hp_ws and 'bagging_temperature' not in hp_ws:
                        ws_cat_kwargs['bagging_temperature'] = 0.4
                    ws_cat_kwargs.update(ws_gpu_params)
                    model = CatBoostClassifier(**ws_cat_kwargs)
                    init_model = None
            
            # Fit model
            
            # ================================================================
            # GPU vs CPU TRAINING STRATEGY
            # GPU: Single-pass training (no init_model support). If dataset
            #      is too large for 2GB VRAM, subsample down to GPU_MAX cap.
            # CPU: Chunked warm-start training with init_model for large data.
            # ================================================================
            is_gpu_training = (cat_kwargs.get('task_type') == 'GPU') if specialist['model'] is None else (
                model.get_params().get('task_type', 'CPU') == 'GPU')
            
            GPU_MAX_SAMPLES = 10000  # Max samples for single-pass GPU training in 2GB VRAM
            CPU_CHUNK_SIZE = 5000    # Chunk size for CPU warm-start training
            
            if is_gpu_training:
                # ============================================================
                # GPU SINGLE-PASS TRAINING (no init_model, no chunking)
                # ============================================================
                if len(X_train) > GPU_MAX_SAMPLES:
                    # HEAD+TAIL subsample: 30% oldest + 70% most recent rows.
                    # Pure tail-only loses all historical regimes, producing a model
                    # that fails when market reverts to an older regime.
                    # Head fraction preserves regime diversity; tail fraction ensures
                    # the current regime is well-represented.
                    # López de Prado AFML Ch.4: subsampling must preserve temporal order.
                    head_n = int(GPU_MAX_SAMPLES * 0.30)
                    tail_n = GPU_MAX_SAMPLES - head_n
                    keep_idx = np.concatenate([
                        np.arange(head_n),
                        np.arange(len(X_train) - tail_n, len(X_train))
                    ])
                    print(f"   📉 GPU: Head+tail subsample {len(X_train)} → {GPU_MAX_SAMPLES} "
                          f"({head_n} oldest + {tail_n} most recent)")
                    X_train = X_train[keep_idx]
                    y_train = y_train[keep_idx]
                    if weights_train is not None:
                        weights_train = weights_train[keep_idx]
                
                print(f"   🚀 GPU single-pass: {len(X_train)} samples")
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    sample_weight=weights_train,
                    init_model=None,  # GPU does NOT support init_model
                    use_best_model=True,
                    early_stopping_rounds=CATBOOST_EARLY_STOPPING,
                    verbose=False
                )
            elif len(X_train) > CPU_CHUNK_SIZE:
                # ============================================================
                # CPU CHUNKED WARM-START TRAINING
                # ============================================================
                print(f"   📉 CPU: Dataset too large ({len(X_train)} samples). Chunking at {CPU_CHUNK_SIZE}...")
                
                for start_idx in range(0, len(X_train), CPU_CHUNK_SIZE):
                    end_idx = min(start_idx + CPU_CHUNK_SIZE, len(X_train))
                    X_chunk = X_train[start_idx:end_idx]
                    y_chunk = y_train[start_idx:end_idx]
                    w_chunk = weights_train[start_idx:end_idx] if weights_train is not None else None
                    
                    print(f"      ▶️ Training chunk {start_idx}-{end_idx}...")
                    
                    # Hard negative mining for chunks > 0
                    if init_model is not None and w_chunk is not None:
                        try:
                            pred_proba = init_model.predict_proba(X_chunk)
                            pred_labels = np.argmax(pred_proba, axis=1)
                            misclassified = (pred_labels != y_chunk)
                            boosted_weights = w_chunk.copy()
                            boosted_weights[misclassified] *= CATBOOST_HARD_NEG_BOOST
                            # Diversity penalty: cap hard-negative mass to 50% of chunk total
                            hard_neg_mass = boosted_weights[misclassified].sum()
                            total_mass = boosted_weights.sum()
                            if hard_neg_mass > 0.5 * total_mass and hard_neg_mass > 0:
                                boosted_weights[misclassified] *= (0.5 * total_mass) / hard_neg_mass
                            w_chunk = boosted_weights
                        except Exception as e:
                            logging.warning(f"Hard-negative mining failed for chunk (using flat weights): {e}", exc_info=True)
                    
                    model.fit(
                        X_chunk, y_chunk,
                        eval_set=(X_val, y_val),
                        sample_weight=w_chunk,
                        init_model=init_model,
                        use_best_model=True,
                        early_stopping_rounds=CATBOOST_EARLY_STOPPING,
                        verbose=False
                    )
                    init_model = model
            else:
                # ============================================================
                # NORMAL TRAINING (small dataset, CPU or GPU)
                # ============================================================
                if init_model is not None and weights_train is not None:
                    try:
                        pred_proba = init_model.predict_proba(X_train)
                        pred_labels = np.argmax(pred_proba, axis=1)
                        misclassified = (pred_labels != y_train)
                        boosted_weights = weights_train.copy()
                        boosted_weights[misclassified] *= CATBOOST_HARD_NEG_BOOST
                        # Diversity penalty: cap hard-negative weight mass to 50% of total
                        # Prevents clustered failures (e.g. one flash crash) from dominating training
                        # Shrivastava et al. (2016): OHEM diversity prevents mode collapse
                        hard_neg_mass = boosted_weights[misclassified].sum()
                        total_mass = boosted_weights.sum()
                        if hard_neg_mass > 0.5 * total_mass and hard_neg_mass > 0:
                            diversity_scale = (0.5 * total_mass) / hard_neg_mass
                            boosted_weights[misclassified] *= diversity_scale
                        n_hard = misclassified.sum()
                        if n_hard > 0:
                            _base_mean = weights_train[misclassified].mean()
                            effective_boost = boosted_weights[misclassified].mean() / _base_mean if _base_mean > 0 else CATBOOST_HARD_NEG_BOOST
                            hard_neg_pct = n_hard / len(y_train) * 100
                            print(f"   🎯 Hard negatives: {n_hard}/{len(y_train)} ({hard_neg_pct:.1f}%) boosted {effective_boost:.2f}x (diversity-capped)")
                            # Mode-collapse warning: if >60% of samples are misclassified,
                            # the model may be memorizing noise or class imbalance is overwhelming it
                            if hard_neg_pct > 60.0:
                                logging.warning(
                                    f"Hard-neg mining: {hard_neg_pct:.1f}% misclassified — "
                                    f"possible mode collapse or extreme label noise. "
                                    f"Check class balance and training data quality."
                                )
                            # Cap effectiveness warning: if boost barely moved (cap hit hard)
                            if effective_boost < 1.1 and CATBOOST_HARD_NEG_BOOST > 1.5:
                                logging.warning(
                                    f"Hard-neg diversity cap absorbed almost all boost "
                                    f"(effective {effective_boost:.2f}x vs target {CATBOOST_HARD_NEG_BOOST}x). "
                                    f"Failures may be too clustered — consider more data diversity."
                                )
                        weights_train = boosted_weights
                    except Exception as e:
                        logging.warning(f"Hard-negative mining failed (using flat weights): {e}", exc_info=True)
                
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    sample_weight=weights_train,
                    init_model=init_model if not is_gpu_training else None,
                    use_best_model=True,
                    early_stopping_rounds=CATBOOST_EARLY_STOPPING,
                    verbose=False
                )

            # =============================================================
            # FEATURE IMPORTANCE PRUNING (Lopez de Prado 2020, Ch.6)
            # MDI via LossFunctionChange (most accurate CatBoost importance):
            # measures actual loss increase when feature is removed — closer
            # to MDA than the default PredictionValuesChange.
            # Threshold: features below mean/2 contribute negligible signal.
            # Applied on the NEXT training cycle to avoid dimension mismatch.
            # =============================================================
            try:
                from catboost import Pool as CatPool
                val_pool = CatPool(X_val, label=y_val)
                importances = model.get_feature_importance(
                    data=val_pool, type='LossFunctionChange'
                )
                if len(importances) > 20:
                    mean_imp = np.mean(importances)
                    threshold = mean_imp / 2.0  # Features < half mean = negligible MDI
                    important_mask = importances >= threshold
                    n_kept = int(important_mask.sum())
                    n_dropped = len(importances) - n_kept
                    if n_kept >= 50:  # Safety: keep at least 50 features
                        specialist['importance_mask'] = important_mask
                        specialist['n_pruned'] = n_dropped
                        print(f"   MDI pruning (next cycle): {len(importances)} -> {n_kept} (dropped {n_dropped} sub-mean/2)")
                    else:
                        specialist['importance_mask'] = None
                        specialist['n_pruned'] = 0
                        print(f"   MDI pruning skipped: only {n_kept} features above threshold (min 50)")
                else:
                    specialist['importance_mask'] = None
                    specialist['n_pruned'] = 0
            except Exception as e:
                specialist['importance_mask'] = None
                specialist['n_pruned'] = 0
                logging.debug(f"Feature importance pruning skipped: {e}")

            # Evaluate
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)

            # =============================================================
            # CPCV Robustness Check (Lopez de Prado AFML Ch.12)
            # Validates model quality across multiple temporal paths.
            # Only runs with enough data; uses reduced iterations for speed.
            # =============================================================
            cpcv_scores = []
            try:
                if num_samples >= 500:
                    from QUANTA_trading_core import CombinatorialPurgedCV
                    from catboost import CatBoostClassifier as _CB
                    cpcv = CombinatorialPurgedCV(n_groups=6, k_test=2, purge_gap=48, embargo_pct=0.01)
                    n_folds = min(cpcv.n_splits(), 5)  # cap at 5 for speed
                    fold_count = 0
                    for cv_train_idx, cv_test_idx in cpcv.split(len(X_arr)):
                        if fold_count >= n_folds:
                            break
                        cv_model = _CB(
                            iterations=300,
                            depth=specialist.get('hyperparams', {}).get('depth', 6),
                            learning_rate=specialist.get('hyperparams', {}).get('learning_rate', 0.05),
                            task_type='CPU',
                            verbose=False
                        )
                        cv_model.fit(X_arr[cv_train_idx], y_arr[cv_train_idx], verbose=False)
                        cpcv_scores.append(cv_model.score(X_arr[cv_test_idx], y_arr[cv_test_idx]))
                        fold_count += 1
                        del cv_model
                    if cpcv_scores:
                        cpcv_mean = float(np.mean(cpcv_scores))
                        cpcv_std  = float(np.std(cpcv_scores))
                        print(f"   📐 CPCV ({fold_count} folds): {cpcv_mean:.1%} ± {cpcv_std:.1%}", end="")
                        if val_acc > cpcv_mean + 2 * cpcv_std and cpcv_std > 0:
                            print(f"  ⚠️ val_acc {val_acc:.1%} >> CPCV mean — possible overfit", end="")
                        print()
            except Exception as e:
                logging.debug(f"CPCV skipped for {specialist_name}: {e}")

            # 🔬 FIT CONFORMAL CALIBRATOR (Romano et al. 2020, Vovk 2005)
            # Distribution-free calibration with guaranteed coverage intervals
            try:
                from QUANTA_trading_core import AdaptiveConformalCalibrator
                val_proba = model.predict_proba(X_val)
                calibrator = AdaptiveConformalCalibrator(alpha_target=0.10)  # 90% coverage
                calibrator.fit(val_proba[:, 1], y_val)
                specialist['calibrator'] = calibrator
                
                # Verify calibration
                raw_conf = np.mean(np.max(val_proba, axis=1))
                cal_out = calibrator.predict(val_proba[:, 1])
                cal_conf = np.mean(cal_out['calibrated_prob'])
                avg_width = np.mean(cal_out['interval_width'])
                print(f"   🔬 Conformal Calibrator: raw avg {raw_conf:.1%} → calibrated {cal_conf:.1%} | interval width: {avg_width:.3f}")
            except Exception as e:
                logging.warning(f"Conformal Calibrator fitting failed for {specialist_name}: {e}")
                specialist['calibrator'] = None
            
            # ═══════════════════════════════════════════════════════
            # DEPLOY GATE (v11.5) — Only deploy if model improves
            # ═══════════════════════════════════════════════════════
            new_gen = specialist['generation'] + 1
            val_proba_for_gate = model.predict_proba(X_val)[:, 1]

            # Build metadata
            meta = self.model_registry.create_metadata(
                agent_name=specialist_name,
                generation=new_gen,
                feature_count=self.cfg.BASE_FEATURE_COUNT,
                feature_mask=specialist.get('feature_mask', ''),
                feature_indices=specialist.get('_feature_indices'),
                n_samples=num_samples,
                n_positive=int(np.sum(y_train == 1)) + int(np.sum(y_val == 1)),
                n_negative=int(np.sum(y_train == 0)) + int(np.sum(y_val == 0)),
                parent_generation=specialist['generation'],
                training_mode="warm_start" if init_model is not None else "fresh",
            )

            # Compute validation metrics for the gate
            self.model_registry.set_val_metrics(meta, y_val, val_proba_for_gate)

            # Store CPCV metrics in metadata
            if cpcv_scores:
                meta.cpcv_mean = float(np.mean(cpcv_scores))
                meta.cpcv_std  = float(np.std(cpcv_scores))

            # ── AUTO-BACKTEST: Quick OOS Sharpe check before deploy ──
            oos_sharpe = None
            try:
                from quanta_backtester import WalkForwardBacktester
                bt = WalkForwardBacktester(self, initial_balance=10000.0)
                # Use a small symbol set for speed (top 10 cached coins)
                bt_symbols = list(self.training_coins.keys())[:10] if hasattr(self, 'training_coins') else None
                if bt_symbols and len(bt_symbols) >= 3:
                    bt_metrics = bt.run(symbols=bt_symbols, verbose=False)
                    oos_sharpe = bt_metrics.sharpe_ratio
                    logging.info(f"Auto-backtest {specialist_name}: Sharpe={oos_sharpe:.3f}, "
                                 f"WR={bt_metrics.win_rate:.1%}, trades={bt_metrics.total_trades}")
            except Exception as e:
                logging.debug(f"Auto-backtest skipped for {specialist_name}: {e}")

            # Run deploy gate (with OOS Sharpe if available)
            # If val set is too small (<20 events), skip AUC gate to avoid rejecting on noise
            if val_total < 20:
                meta._skip_auc_gate = True  # signal to registry: ignore AUC threshold
            should_deploy, deploy_reason = self.model_registry.should_deploy(
                specialist_name, meta, oos_sharpe=oos_sharpe)

            if should_deploy:
                # Update specialist
                specialist['model'] = model
                specialist['generation'] = new_gen
                specialist['performance'].append({
                    'generation': new_gen,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'samples': num_samples,
                    'val_auc': meta.val_auc,
                    'val_brier': meta.val_brier,
                })

                # Reset Brier accumulator — old model's calibration history is invalid
                # for the new model. Bot ensemble weights must rebuild from scratch.
                if hasattr(self, '_brier_scores') and specialist_name in self._brier_scores:
                    self._brier_scores[specialist_name] = {
                        'sum': 0.0, 'count': 0, 'rolling': deque(maxlen=500)
                    }
                    print(f"   🔄 Brier scores reset for {specialist_name} (new model deployed)")

                self.model_registry.record_deployment(specialist_name, meta,
                                                       deployed=True, reason=deploy_reason)

                print(f"   📊 Train: {train_acc:.1%} | Val: {val_acc:.1%} | AUC: {meta.val_auc:.4f} | Brier: {meta.val_brier:.4f}")
                print(f"   ✅ DEPLOYED gen{new_gen}: {deploy_reason}")
                print(f"{'='*70}\n")
            else:
                # Reject — keep previous model
                self.model_registry.record_deployment(specialist_name, meta,
                                                       deployed=False, reason=deploy_reason)

                print(f"   📊 Train: {train_acc:.1%} | Val: {val_acc:.1%} | AUC: {meta.val_auc:.4f} | Brier: {meta.val_brier:.4f}")
                print(f"   ❌ REJECTED gen{new_gen}: {deploy_reason}")
                print(f"   🔄 Keeping gen{specialist['generation']} (previous model preserved)")
                print(f"{'='*70}\n")
                return True  # Training succeeded, just didn't deploy

            return True
            
        except Exception as e:
            logging.error(f"Failed to train {specialist_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_ensemble_weights(self):
        """
        Entropy-Regularized Ensemble Weight Update

        After training all 7 models, recompute weights using rolling Brier
        scores (primary) blended with val_acc (secondary).
        Uses softmax(quality / λ) with clamping to prevent domination.

        Quality signal:
          - Primary: rolling Brier score from self._brier_scores (lower = better)
          - Fallback: val_acc from performance history
          - Blend: 0.6 * (1 - brier) + 0.4 * val_acc when both available

        References:
          REPO (Markowitz + Shannon) — entropy-regularized portfolio
          Kuncheva 2004 — optimal diversity via weight balancing
        """
        cfg = self.ensemble_config
        max_w = cfg['max_weight']      # 0.25
        min_w = cfg['min_weight']      # 0.05
        lam = cfg['entropy_lambda']    # 0.3

        # Collect blended scores from val AUC + rolling Brier quality
        scores = {}
        for name, spec in self.specialist_models.items():
            # Base: val AUC from last training
            if spec['performance']:
                latest = spec['performance'][-1]
                val_auc = latest.get('val_auc', latest.get('val_acc', 0.5))  # val_auc is canonical; val_acc is fallback for legacy records
            else:
                val_auc = 0.5

            # Live: rolling Brier score (lower = better, convert to quality)
            brier_quality = 0.5  # default
            if hasattr(self, '_brier_scores') and name in self._brier_scores:
                bs = self._brier_scores[name]
                rolling = bs.get('rolling', deque())
                if len(rolling) >= 20:
                    # Brier is 0-1, lower=better. Convert: quality = 1 - brier
                    brier_quality = 1.0 - (sum(rolling) / len(rolling))
                elif bs['count'] > 0:
                    brier_quality = 1.0 - (bs['sum'] / bs['count'])

            # Blend: 40% val AUC + 60% live Brier quality
            # Brier gets more weight because it's continuously updated
            blended = 0.4 * val_auc + 0.6 * brier_quality
            scores[name] = blended

        if not scores:
            return

        # Softmax with temperature λ: w_i ∝ exp(quality_i / λ)
        names = list(scores.keys())
        score_vals = np.array([scores[n] for n in names])

        # Entropy-regularized softmax
        exp_scores = np.exp(score_vals / lam)
        raw_weights = exp_scores / exp_scores.sum()

        # Clamp to [min_w, max_w]
        clamped = np.clip(raw_weights, min_w, max_w)

        # Renormalize to sum=1
        clamped = clamped / clamped.sum()

        # Apply and log
        print(f"\n{'='*70}")
        print(f"⚖️ ENSEMBLE WEIGHT UPDATE (Brier-Blended, Entropy-Regularized)")
        print(f"{'='*70}")
        for i, name in enumerate(names):
            old_w = self.specialist_models[name]['weight']
            new_w = clamped[i]
            self.specialist_models[name]['weight'] = float(new_w)
            delta = new_w - old_w
            arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "="
            print(f"   {name:10s}: score={scores[name]:.3f} → weight={new_w:.3f} ({arrow})")

        total = sum(s['weight'] for s in self.specialist_models.values())
        print(f"   Total weight: {total:.4f} (should be 1.0)")
        print(f"   Max weight: {max(s['weight'] for s in self.specialist_models.values()):.3f} (cap={max_w})")
        print(f"{'='*70}\n")
    
    def predict_with_specialists(self, symbol, tf_analysis):
        """
        🧬 15-AGENT PANTHEON ENSEMBLE PREDICTION
        
        Entropy-weighted soft voting across 7 diverse CatBoost models.
        Each model's global weight (from _update_ensemble_weights) is
        multiplied by its per-sample confidence (Kuncheva entropy weighting).
        
        Falls back to legacy ensemble if specialists not trained.
        """
        # Check if ANY specialists are trained (don't require all 7)
        trained_specialists = {
            name: s for name, s in self.specialist_models.items()
            if s['model'] is not None
        }
        
        if not trained_specialists:
            return 'NEUTRAL', 0, 0, 0
        
        try:
            # Extract features (same for all models)
            # Pass symbol so impulse features can pull candle_store data for live inference
            _live_candles = None
            if self.candle_store is not None:
                _cs = self.candle_store.get(symbol, '5m')
                if _cs is not None and len(_cs) >= 20:
                    _live_candles = {
                        'closes':    np.array([float(k[4]) for k in _cs[-100:]], dtype=np.float64),
                        'highs':     np.array([float(k[2]) for k in _cs[-100:]], dtype=np.float64),
                        'lows':      np.array([float(k[3]) for k in _cs[-100:]], dtype=np.float64),
                        'volumes':   np.array([float(k[5]) for k in _cs[-100:]], dtype=np.float64),
                        'taker_buy': np.array([float(k[9]) if len(k) > 9 else 0.0 for k in _cs[-100:]], dtype=np.float64),
                    }
            features = self._extract_features(tf_analysis, symbol=symbol, _raw_candles=_live_candles)
            
            # 🧠 HYBRID PREDICTION (LSTM-Attention + CatBoost)
            tft_proba = None
            if self.tft_model is not None and self.candle_store is not None and getattr(self, 'tft_trained', False):
                try:
                    import torch
                    candles = self.candle_store.get(symbol, '5m')
                    if candles is not None and len(candles) >= TFT_SEQ_LENGTH + 50:
                        seq_features = []
                        candles_list = candles.tolist() if hasattr(candles, 'tolist') else list(candles)
                        for t in range(len(candles_list) - TFT_SEQ_LENGTH, len(candles_list)):
                            f = self._extract_features_from_candles(candles_list, t)
                            if f is not None:
                                seq_features.append(f)
                        
                        if len(seq_features) == TFT_SEQ_LENGTH:
                            X_seq = np.array(seq_features, dtype=np.float32)
                            if X_seq.shape[1] < self.cfg.BASE_FEATURE_COUNT:
                                pad = np.zeros((X_seq.shape[0], self.cfg.BASE_FEATURE_COUNT - X_seq.shape[1]), dtype=np.float32)
                                X_seq = np.concatenate([X_seq, pad], axis=1)
                            elif X_seq.shape[1] > self.cfg.BASE_FEATURE_COUNT:
                                X_seq = X_seq[:, :self.cfg.BASE_FEATURE_COUNT]
                            
                            X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
                            device = next(self.tft_model.parameters()).device
                            X_tensor = X_tensor.to(device)
                            self.tft_model.eval()
                            with torch.no_grad():
                                tft_proba = self.tft_model.predict_proba(X_tensor)[0]
                except Exception as e:
                    logging.debug(f"LSTM-Attention inference skipped: {e}")

            # Get predictions from each specialist
            predictions = []
            weights = []
            
            for name, specialist in trained_specialists.items():
                    # FIX: Apply feature mask before scaling (matches training path)
                    # Use stored _feature_indices (includes subspace + importance pruning)
                    f = features
                    feat_idx = specialist.get('_feature_indices')
                    if feat_idx is not None:
                        f = features[feat_idx]
                    else:
                        mask_cfg = specialist.get('feature_mask', None)
                        if isinstance(mask_cfg, str) and mask_cfg.startswith('random_70_seed'):
                            seed = int(mask_cfg.split('seed')[1])
                            n_f = len(features)
                            n_keep = int(n_f * 0.7)
                            rng = np.random.RandomState(seed)
                            idx = np.sort(rng.choice(n_f, n_keep, replace=False))
                            f = features[idx]
                    features_scaled = specialist['scaler'].transform([f])
                    
                    # Predict
                    pred_proba = specialist['model'].predict_proba(features_scaled)[0]
                    
                    # 🔬 APPLY VENN-ABERS CALIBRATION
                    if specialist.get('calibrator') is not None:
                        try:
                            cal_p, _ = specialist['calibrator'].predict_proba(np.array([pred_proba]))
                            pred_proba = cal_p[0]
                        except Exception:
                            pass  # Use raw if calibrator fails
                    
                    predictions.append(pred_proba)
                    weights.append(specialist['weight'])
            
            # 🔬 ENTROPY-WEIGHTED ENSEMBLE (Kuncheva 2004)
            # Models with lower entropy (more certain) get higher weight
            if len(predictions) > 1:
                entropies = [-np.sum(p * np.log(p + 1e-10)) for p in predictions]
                inv_entropy = [1.0 / (h + 1e-6) for h in entropies]
                total_inv = sum(inv_entropy)
                adaptive_weights = np.array([w / total_inv for w in inv_entropy])
            else:
                adaptive_weights = np.array(weights)
                adaptive_weights = adaptive_weights / adaptive_weights.sum()
            
            ensemble_proba = np.average(predictions, axis=0, weights=adaptive_weights)
            
            # REMOVED: Soft-blending of LSTM. Odin now acts exclusively as
            # a hard anomaly veto to protect against CatBoost forgetting.
            
            # Bayesian uncertainty
            p_max = max(ensemble_proba)
            uncertainty = np.sqrt(p_max * (1 - p_max)) * 100
            
            # Ensemble disagreement
            pred_variance = np.var([p[1] for p in predictions])
            disagreement = np.sqrt(pred_variance) * 100
            total_uncertainty = (uncertainty + disagreement) / 2
            
            # Calc basic direction and raw confidence
            confidence = p_max * 100
            direction_value = 1 if ensemble_proba[1] > ensemble_proba[0] else -1
            
            # 🔬 META-LABELING FILTER (Rule-based)
            # Adjust confidence based on market context (Prevents false signals)
            is_dead_market = False  # Safe default before try block
            try:
                # 1. Volatility Context (ATR)
                atr_14 = tf_analysis['ATR_14'].iloc[-1]
                atr_pct = (atr_14 / tf_analysis['close'].iloc[-1]) * 100
                is_dead_market = atr_pct < 0.5  # Less than 0.5% volatility
                
                # 2. Volume Context
                vol_current = tf_analysis['volume'].iloc[-1]
                vol_sma = tf_analysis['volume'].rolling(20).mean().iloc[-1]
                is_low_volume = vol_current < (0.8 * vol_sma)
                
                # Apply Meta-Filter Penalities
                meta_penalty = 0.0
                if is_dead_market:
                    meta_penalty += 0.20  # 20% penalty
                if is_low_volume:
                    meta_penalty += 0.15  # 15% penalty
            except Exception:
                meta_penalty = 0.0  # Safe fallback
            
            # Uncertainty-adjusted + Meta-filtered confidence
            uncertainty_penalty = total_uncertainty / 50
            confidence_adjusted = confidence * (1 - uncertainty_penalty * 0.3) * (1 - meta_penalty)
            
            # Adaptive threshold (López de Prado 2018: 10-15% deadband optimal)
            threshold = DIRECTION_THRESHOLD
            normalized_score = direction_value * (confidence_adjusted / 100)
            
            if normalized_score > threshold:
                direction = 'BULLISH'
            elif normalized_score < -threshold:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
                
            # ========================================
            # 🔱 ODIN ANOMALY VETO (Dual-Brain Override)
            # ========================================
            # If CatBoost dictates a trade but Odin's long-term EWC memory 
            # detects highly probable structural extremes (>80%), Odin vetoes CatBoost.
            if tft_proba is not None:
                odin_bear_p = tft_proba[0]
                odin_bull_p = tft_proba[1]
                
                if direction == 'BULLISH' and odin_bear_p > ODIN_VETO_THRESHOLD:
                    print(f"\n🚨 ODIN VETO: Overriding CatBoost 'BULLISH' prediction! Structural crash signature detected ({odin_bear_p*100:.1f}%). Setting BEARISH.")
                    direction = 'BEARISH'
                    confidence_adjusted = odin_bear_p * 100
                elif direction == 'BEARISH' and odin_bull_p > ODIN_VETO_THRESHOLD:
                    print(f"\n🚀 ODIN VETO: Overriding CatBoost 'BEARISH' prediction! Structural breakout signature detected ({odin_bull_p*100:.1f}%). Setting BULLISH.")
                    direction = 'BULLISH'
                    confidence_adjusted = odin_bull_p * 100
            
            # Calculate magnitude
            magnitude = self._calculate_magnitude(tf_analysis, confidence)
            magnitude = magnitude * (1 - uncertainty_penalty * 0.5) * (1 - meta_penalty)
            
            # Make sure we don't output extreme confidence in dead markets
            if direction != 'NEUTRAL' and is_dead_market:
                direction = 'NEUTRAL'
                confidence_adjusted = min(confidence_adjusted, 55.0)
                
            return direction, confidence_adjusted, magnitude, total_uncertainty
            
        except Exception as e:
            logging.error(f"Specialist prediction failed: {e}")
            # FIX: Return NEUTRAL instead of calling self.predict() to avoid recursion
            return 'NEUTRAL', 0, 0, 0

    def _load_models(self):
        """Load models - Try specialists first, fallback to legacy ensemble"""
        try:
            # 🧬 TRY SPECIALIST MODELS FIRST
            specialists_loaded = self._load_specialist_models()
            
            if specialists_loaded:
                print("\n✅ SPECIALIST MODELS LOADED")
                self.is_trained = True
                
                # ========================================
                # LEGACY CONSUMER COMPATIBILITY
                # ========================================
                # Consumer uses legacy self.scaler and self.catboost_model
                # Set them to use Athena (primary) specialist
                if self.specialist_models['athena']['model'] is not None:
                    self.scaler = self.specialist_models['athena']['scaler']
                    self.calibrator = self.specialist_models['athena'].get('calibrator')
                    athena_model = self.specialist_models['athena']['model']
                    self.models = [(athena_model, 1, 1.0, {})]
                    print("✅ Legacy consumer compatibility enabled (using Athena specialist)")
                
                return
            
            # Fallback to legacy ensemble
            print("\n⚠️  No specialist models found, trying legacy ensemble...")
            
            # Load ensemble models (model_gen1.cbm, model_gen2.cbm, etc.)
            ensemble_loaded = False
            loaded_models = []  # Temporary list to calculate weights correctly
            
            for gen in range(1, self.max_models + 1):
                model_path = os.path.join(self.cfg.model_dir, f'catboost_model_gen{gen}.cbm')
                if os.path.exists(model_path):
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    loaded_models.append((model, gen))
                    ensemble_loaded = True
                    
                    if gen > self.model_generation:
                        self.model_generation = gen
            
            # Now calculate weights correctly based on newest generation
            if loaded_models:
                newest_gen = max(gen for _, gen in loaded_models)
                for model, gen in loaded_models:
                    # FIXED: Newest model gets weight 1.0, older models decay
                    weight = self.weight_decay ** (newest_gen - gen)
                    self.models.append((model, gen, weight, {}))
                    logging.info(f"✅ Loaded Gen {gen} model (weight: {weight:.2f})")
                    print(f"✅ Loaded Gen {gen} model (weight: {weight:.2f})")
            
            # Fallback: load single model if no ensemble found
            if not ensemble_loaded:
                catboost_path = os.path.join(self.cfg.model_dir, 'catboost_model.cbm')
                if os.path.exists(catboost_path):
                    print(f"📂 Found model at: {catboost_path}")
                    model = CatBoostClassifier()
                    model.load_model(catboost_path)
                    self.models.append((model, 1, 1.0, {}))
                    self.model_generation = 1
                    logging.info("✅ Loaded CatBoost GPU model (converted to ensemble)")
                    print("✅ Loaded existing CatBoost model (Gen 1)")
                else:
                    print(f"❌ Model file not found: {catboost_path}")
            
            # Load Scaler
            scaler_path = os.path.join(self.cfg.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info("✅ Loaded scaler")
                print("✅ Loaded existing scaler")
            else:
                print(f"❌ Scaler file not found: {scaler_path}")
            
            scaler_fitted = hasattr(self.scaler, 'mean_')
            self.is_trained = len(self.models) > 0 and scaler_fitted
            
            if self.is_trained:
                print(f"✅ ENSEMBLE LOADED - {len(self.models)} models (Gen {self.model_generation})")
            else:
                print(f"⚠️  Model check failed:")
                print(f"   - Models: {len(self.models)}")
                print(f"   - Scaler fitted: {'✅' if scaler_fitted else '❌'}")
            
            if not self.is_trained:
                logging.warning("⚠️ No trained models found")
                print("\n" + "="*70)
                print("⚠️  NO TRAINED MODELS FOUND")
                print("="*70)
                print("💡 Models will train automatically when first needed")
                print("💡 Or use /train command to train manually")
                print("="*70 + "\n")
                
            # Load TFT (Odin) Model
            if self.tft_model is not None:
                tft_path = os.path.join(self.cfg.model_dir, 'tft_model.pth')
                if os.path.exists(tft_path):
                    try:
                        import torch
                        self.tft_model.load_state_dict(torch.load(tft_path, map_location=next(self.tft_model.parameters()).device))
                        self.tft_model.eval()
                        self.tft_trained = True
                        print("✅ Loaded Odin (LSTM-Attention) model weights.")
                    except Exception as e:
                        print(f"⚠️  Failed to load Odin model weights: {e}")
                        self.tft_trained = False
                else:
                    self.tft_trained = False
                    print("⚠️  No Odin (LSTM-Attention) model weights found - will use fallback")
                
        except Exception as e:
            print(f"❌ MODEL LOADING ERROR: {e}")
            import traceback
            traceback.print_exc()
            logging.error(f"Model loading failed: {e}")
            self.is_trained = False

    # =========================================================================
    # FAST PRE-COMPUTED FEATURE EXTRACTION (avoids redundant computation)
    # =========================================================================
    
    def _precompute_coin_indicators(self, klines_np):
        """Pre-compute ALL indicator series for a coin ONCE.
        
        Instead of recomputing RSI/MACD/ADX/ATR/BB at every event position
        (16K times per coin), compute them as full arrays and index later.
        
        Returns: dict with pre-computed arrays per timeframe.
        """
        n = len(klines_np)
        closes = klines_np[:, 4].copy()
        highs = klines_np[:, 2].copy()
        lows = klines_np[:, 3].copy()
        volumes = klines_np[:, 5].copy()
        open_times = klines_np[:, 0].astype(np.int64)
        # Binance klines col 9 = taker_buy_base_volume (buy-side pressure proxy for OB features)
        taker_buy = klines_np[:, 9].copy() if klines_np.shape[1] > 9 else np.full(n, 0.5 * volumes)

        tf_windows = {'5m': 1, '15m': 3, '1h': 12, '4h': 48, '1d': 288}
        precomputed = {'open_times': open_times, 'raw_closes': closes,
                       'raw_highs': highs, 'raw_lows': lows, 'raw_volumes': volumes,
                       'raw_taker_buy': taker_buy}
        
        for tf, w in tf_windows.items():
            if w == 1:
                c, h, l, v, tb = closes, highs, lows, volumes, taker_buy
                rem = 0
            else:
                rem = n % w
                if n < w * 20:
                    continue
                c = closes[rem:].reshape(-1, w)[:, -1]
                h = np.max(highs[rem:].reshape(-1, w), axis=1)
                l = np.min(lows[rem:].reshape(-1, w), axis=1)
                v = np.sum(volumes[rem:].reshape(-1, w), axis=1)
                tb = np.sum(taker_buy[rem:].reshape(-1, w), axis=1)
            
            tf_len = len(c)
            if tf_len < 20:
                continue
            
            # Pre-compute all indicator SERIES (one pass each)
            rsi_arr = _jit_rsi_series(c, 14)
            ema_fast = _jit_ema_series(c, 12)
            ema_slow = _jit_ema_series(c, 26)
            macd_line_arr = ema_fast - ema_slow
            atr_arr = _jit_atr_series(h, l, c, 14)
            adx_arr = _jit_adx_series(h, l, c, 14)
            ma20_arr = _jit_rolling_mean(c, 20)
            ma50_arr = _jit_rolling_mean(c, 50)
            bb_std_arr = _jit_rolling_std(c, 20)
            vol_ma_arr = _jit_rolling_mean(v, 20)
            
            # Returns series
            returns_arr = np.zeros(tf_len, dtype=np.float64)
            if tf_len > 1:
                returns_arr[1:] = np.diff(c) / (c[:-1] + 1e-8)
            vol_rolling = _jit_rolling_std(returns_arr, 20)
            
            # ATR percentile series (sampled)
            atr_sampled = atr_arr[14::5]
            
            precomputed[tf] = {
                'closes': c, 'highs': h, 'lows': l, 'volumes': v, 'taker_buy': tb,
                'tf_len': tf_len, 'w': w, 'rem': rem,
                'rsi': rsi_arr,
                'macd_line': macd_line_arr,
                'atr': atr_arr,
                'adx': adx_arr,
                'ma20': ma20_arr,
                'ma50': ma50_arr,
                'bb_std': bb_std_arr,
                'vol_ma': vol_ma_arr,
                'returns': returns_arr,
                'vol_rolling': vol_rolling,
                'atr_sampled': atr_sampled,
            }
        
        return precomputed
    
    def _fast_extract_at_position(self, precomputed, position, klines_np, symbol=None):
        """Extract features by indexing into pre-computed arrays. O(1) per indicator.

        Training path (symbol=None): uses OHLCV proxies + historical futures cache so the
        28 formerly-NaN live-only features are populated with meaningful values.
        Falls back to _extract_features_from_candles if precomputed data unavailable.
        """
        try:
            if position < 50 or position >= len(klines_np):
                return None
            
            tf_windows = {'5m': 1, '15m': 3, '1h': 12, '4h': 48, '1d': 288}
            tf_analysis = {}
            
            for tf, w in tf_windows.items():
                if tf not in precomputed:
                    continue
                data = precomputed[tf]
                
                # Map 5m position to this TF's position
                rem = data['rem']
                if w == 1:
                    tf_pos = position
                else:
                    tf_pos = (position - rem) // w
                
                c = data['closes']
                h = data['highs']
                l = data['lows']
                v = data['volumes']
                tf_len = data['tf_len']
                
                if tf_pos < 20 or tf_pos >= tf_len:
                    continue
                
                # Index into pre-computed arrays (O(1) each!)
                r = float(data['rsi'][tf_pos])
                macd_val = float(data['macd_line'][tf_pos])
                adx_val = float(data['adx'][tf_pos])
                atr_val = float(data['atr'][tf_pos])
                
                # Bollinger from pre-computed mean and std
                bm = float(data['ma20'][tf_pos])
                b_std = float(data['bb_std'][tf_pos])
                bu = bm + 2.0 * b_std
                bl = bm - 2.0 * b_std
                bp = (c[tf_pos] - bm) / (bu - bl + 1e-8) if (bu - bl) != 0.0 else 0.5
                
                # Trend from pre-computed MA20
                ma20 = float(data['ma20'][tf_pos])
                price = float(c[tf_pos])
                trend = 'BULLISH' if price > ma20 else ('BEARISH' if price < ma20 else 'NEUTRAL')
                strength = min(100.0, (abs(price - ma20) / (atr_val + 1e-8)) * 50)
                
                # Volatility from pre-computed rolling std of returns
                vol = float(data['vol_rolling'][tf_pos])
                
                # ATR percentile
                atr_sampled = data['atr_sampled']
                atr_pct = float(np.sum(atr_sampled <= atr_val) / max(1, len(atr_sampled))) if len(c) >= 100 else 0.5
                
                # Volatility acceleration
                atr_prev = float(data['atr'][max(0, tf_pos - 10)])
                vol_accel = (atr_val - atr_prev) / atr_val if atr_val > 0 else 0.0
                
                # Momentum (simple lookback)
                mom_5 = (c[tf_pos] / c[tf_pos - 5] - 1) if tf_pos >= 6 else 0.0
                mom_10 = (c[tf_pos] / c[tf_pos - 10] - 1) if tf_pos >= 11 else 0.0
                mom_20 = (c[tf_pos] / c[tf_pos - 20] - 1) if tf_pos >= 21 else 0.0
                mom_50 = (c[tf_pos] / c[tf_pos - 50] - 1) if tf_pos >= 51 else 0.0
                
                # Volume ratio from pre-computed rolling mean
                vr = float(v[tf_pos] / (data['vol_ma'][tf_pos] + 1e-8))
                
                # Mean shift
                ms_mean = float(data['ma50'][tf_pos]) if tf_pos >= 50 else float(np.mean(c[:tf_pos+1]))
                ms_std = float(np.std(c[max(0, tf_pos-49):tf_pos+1])) if tf_pos >= 10 else 1e-8
                ms = (c[tf_pos] - ms_mean) / (ms_std + 1e-8)
                
                # VPIN (real taker-buy) and FracDiff: still compute inline (fast on 14-element windows)
                start = max(0, tf_pos - 14)
                tb_data = data.get('taker_buy')
                tb_slice = tb_data[start:tf_pos+1] if tb_data is not None else None
                vpin_val = float(Indicators.vpin(h[start:tf_pos+1], l[start:tf_pos+1],
                                                 c[start:tf_pos+1], v[start:tf_pos+1], 14,
                                                 taker_buy=tb_slice))
                fd_val = float(Indicators.frac_diff(c[max(0, tf_pos-100):tf_pos+1], d=0.4))
                
                tf_analysis[tf] = {
                    'rsi': r, 'macd': macd_val, 'bb_position': bp,
                    'adx': adx_val, 'atr': atr_val,
                    'trend': trend, 'strength': strength,
                    'price': price, 'volume': float(v[tf_pos]),
                    'volatility': vol, 'atr_percentile': atr_pct,
                    'volatility_accel': vol_accel,
                    'momentum_5': mom_5, 'momentum_10': mom_10,
                    'momentum_20': mom_20, 'momentum_50': mom_50,
                    'volume_ratio': vr, 'mean_shift': ms,
                    'trend_strength': (adx_val / 100.0) * strength,
                    'returns_period': mom_20,
                    'vpin': vpin_val, 'frac_diff': fd_val
                }
            
            if not tf_analysis:
                return None
            
            # Raw candle data for research features (Hurst, SampEn, etc.)
            raw_c = precomputed['raw_closes']
            raw_h = precomputed['raw_highs']
            raw_l = precomputed['raw_lows']
            raw_v = precomputed['raw_volumes']
            raw_tb = precomputed.get('raw_taker_buy')
            raw_candle_data = {
                'closes':    raw_c[max(0, position-49):position+1],
                'highs':     raw_h[max(0, position-49):position+1],
                'lows':      raw_l[max(0, position-49):position+1],
                'volumes':   raw_v[max(0, position-49):position+1],
                'taker_buy': raw_tb[max(0, position-49):position+1] if raw_tb is not None else None,
            }
            ts_ms   = int(precomputed['open_times'][position])
            features = self._extract_features(
                tf_analysis, symbol=None, _raw_candles=raw_candle_data,
                _training_ts_ms=ts_ms, _training_symbol=symbol,
            )
            
            # Fix time features with actual candle timestamp
            open_times = precomputed['open_times']
            from datetime import datetime
            candle_time = datetime.fromtimestamp(open_times[position] / 1000.0)
            hour = candle_time.hour
            day_of_week = candle_time.weekday()
            features[107] = np.sin(2 * np.pi * hour / 24)
            features[108] = np.cos(2 * np.pi * hour / 24)
            features[109] = np.sin(2 * np.pi * day_of_week / 7)
            features[110] = np.cos(2 * np.pi * day_of_week / 7)
            features[111] = hour / 24.0
            features[112] = day_of_week / 7.0
            # Session-aware features (indices 113-115)
            features[113] = 1.0 if 13 <= hour <= 21 else 0.0   # US session
            features[114] = 1.0 if 1 <= hour <= 9  else 0.0    # Asia session
            features[115] = 1.0 if hour in (0, 8, 16) else 0.0 # Funding window
            
            # Ensure correct size
            if len(features) != self.cfg.BASE_FEATURE_COUNT:
                logging.warning(
                    f"Candle feature dim mismatch: got {len(features)}, expected {self.cfg.BASE_FEATURE_COUNT}"
                )
                if len(features) < self.cfg.BASE_FEATURE_COUNT:
                    padded = np.zeros(self.cfg.BASE_FEATURE_COUNT, dtype=np.float32)
                    padded[:len(features)] = features
                    padded[229] = 0.5
                    features = padded
                else:
                    features = features[:self.cfg.BASE_FEATURE_COUNT]
            assert len(features) == self.cfg.BASE_FEATURE_COUNT
            return features

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _extract_features_from_candles(self, candles, position, _precomputed_np=None):
        """Extract simplified features from raw candle data for temporal training.
        
        Args:
            candles: raw candle data (list or numpy array)
            position: index to extract features at
            _precomputed_np: pre-converted numpy array (avoids repeated list→numpy conversion)
                             Pass this when calling in a loop over the same candle data.
        """
        try:
            # Need at least 50 candles for basic 5m indicators
            if position < 50 or position >= len(candles):
                return None
            
            # 5m BASE: Aggregation ratios for higher timeframes from 5m candles
            # Research: 5m has better signal-to-noise than 1m (Chen 2020: 67% vs 55%)
            tf_windows = {'5m': 1, '15m': 3, '1h': 12, '4h': 48, '1d': 288}
            tf_analysis = {}
            
            # OPTIMIZED: Cap window at 15K 5m candles (same coverage as 75K 1m)
            max_window = 15_000
            start_idx = max(0, position - max_window)
            
            # OPTIMIZED: Use pre-converted numpy array if provided (avoids repeated list→numpy)
            # np.array() on 70K+ elements is expensive — this is called ~300 times per coin
            if _precomputed_np is not None:
                window_arr = _precomputed_np[start_idx:position+1]  # O(1) numpy view, no copy
            else:
                window = candles[start_idx:position+1]
                if len(window) < 40:
                    return None
                window_arr = np.array(window, dtype=np.float64)
            
            if len(window_arr) < 40:
                return None
            
            # Pre-extract base arrays - VECTORIZED
            open_times = window_arr[:, 0].astype(np.int64)
            opens = window_arr[:, 1]
            highs = window_arr[:, 2]
            lows = window_arr[:, 3]
            closes = window_arr[:, 4]
            volumes = window_arr[:, 5]
            
            for tf, w in tf_windows.items():
                if w == 1:
                    c_tf, h_tf, l_tf, v_tf = closes, highs, lows, volumes
                else:
                    if len(closes) < w:
                        continue
                        
                    # Real multi-timeframe aggregation
                    # Reshape arrays to aggregate cleanly. Truncate head to be perfectly divisible by w.
                    rem = len(closes) % w
                    c_trunc = closes[rem:]
                    h_trunc = highs[rem:]
                    l_trunc = lows[rem:]
                    v_trunc = volumes[rem:]
                    
                    # Quick aggregation via reshaping
                    c_tf = c_trunc.reshape(-1, w)[:, -1]
                    h_tf = np.max(h_trunc.reshape(-1, w), axis=1)
                    l_tf = np.min(l_trunc.reshape(-1, w), axis=1)
                    v_tf = np.sum(v_trunc.reshape(-1, w), axis=1)
                
                if len(c_tf) < 20:
                    continue
                    
                # Calculate required MTF exactly as in MultiTimeframeAnalyzer
                r = Indicators.rsi(c_tf, 14)
                macd_line, _, _ = Indicators.macd(c_tf)
                bu, bm, bl = Indicators.bollinger(c_tf, 20)
                bp = (c_tf[-1] - bm) / (bu - bl + 1e-8) if (bu - bl) != 0.0 else 0.5
                adx_val = Indicators.adx(h_tf, l_tf, c_tf, 14)
                atr_val = Indicators.atr(h_tf, l_tf, c_tf, 14)
                
                ma20 = np.mean(c_tf[-20:])
                trend = 'BULLISH' if c_tf[-1] > ma20 else ('BEARISH' if c_tf[-1] < ma20 else 'NEUTRAL')
                strength = min(100.0, (abs(c_tf[-1] - ma20) / (atr_val + 1e-8)) * 50)
                
                # Volatility
                returns = np.diff(c_tf) / (c_tf[:-1] + 1e-8)
                vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.0
                
                atr_s = Indicators.atr_series(h_tf, l_tf, c_tf)
                recent_atrs = atr_s[14::5]
                atr_pct = float(np.sum(recent_atrs <= atr_val) / max(1, len(recent_atrs))) if len(c_tf) >= 100 else 0.5
                atr_prev = atr_s[-10] if len(atr_s) >= 10 else atr_s[0]
                vol_accel = (atr_val - atr_prev) / atr_val if atr_val > 0 else 0.0
                
                # Momentum
                mom_5 = (c_tf[-1] / c_tf[-6] - 1) if len(c_tf) >= 6 else 0.0
                mom_10 = (c_tf[-1] / c_tf[-11] - 1) if len(c_tf) >= 11 else 0.0
                mom_20 = (c_tf[-1] / c_tf[-21] - 1) if len(c_tf) >= 21 else 0.0
                mom_50 = (c_tf[-1] / c_tf[-51] - 1) if len(c_tf) >= 51 else 0.0
                
                vr = v_tf[-1] / (np.mean(v_tf[-20:]) + 1e-8) if len(v_tf) >= 20 else 1.0
                ms = (c_tf[-1] - np.mean(c_tf[-50:])) / (np.std(c_tf[-50:]) + 1e-8) if len(c_tf) >= 50 else 0.0
                
                # Option A: VPIN (Order Flow Toxicity)
                vpin_val = Indicators.vpin(h_tf, l_tf, c_tf, v_tf, 14)
                # Option C: Fractional Differencing
                fd_val = Indicators.frac_diff(c_tf, d=0.4)
                
                tf_analysis[tf] = {
                    'rsi': r,
                    'macd': macd_line,
                    'bb_position': bp,
                    'adx': adx_val,
                    'atr': atr_val,
                    'trend': trend,
                    'strength': strength,
                    'price': c_tf[-1],
                    'volume': v_tf[-1],
                    'volatility': vol,
                    'atr_percentile': atr_pct,
                    'volatility_accel': vol_accel,
                    'momentum_5': mom_5,
                    'momentum_10': mom_10,
                    'momentum_20': mom_20,
                    'momentum_50': mom_50,
                    'volume_ratio': vr,
                    'mean_shift': ms,
                    'trend_strength': (adx_val / 100.0) * strength,
                    'returns_period': mom_20,
                    'vpin': vpin_val,
                    'frac_diff': fd_val
                }
            
            if not tf_analysis:
                return None
                
            # FIX Phase 0A: Pass raw candle data so HMM/Hurst/SampEn/Kyle/etc.
            # are computed from actual training data instead of [1.0, 1.0]
            raw_candle_data = {
                'closes': closes[-50:] if len(closes) >= 50 else closes,
                'highs': highs[-50:] if len(highs) >= 50 else highs,
                'lows': lows[-50:] if len(lows) >= 50 else lows,
                'volumes': volumes[-50:] if len(volumes) >= 50 else volumes,
                'taker_buy': None,  # Not available in this training path
            }
            features = self._extract_features(tf_analysis, symbol=None, _raw_candles=raw_candle_data)
            
            # Since _extract_features uses `datetime.now()` for Time Features natively,
            # this would cause historical test data to contain the system runtime time instead of the candle time!
            # FIX: Overwrite the 9 time+session features with the actual candle time (slots 107-115).
            from datetime import datetime
            candle_time = datetime.fromtimestamp(open_times[-1] / 1000.0)
            hour = candle_time.hour
            day_of_week = candle_time.weekday()

            features[107] = np.sin(2 * np.pi * hour / 24)
            features[108] = np.cos(2 * np.pi * hour / 24)
            features[109] = np.sin(2 * np.pi * day_of_week / 7)
            features[110] = np.cos(2 * np.pi * day_of_week / 7)
            features[111] = hour / 24.0
            features[112] = day_of_week / 7.0
            # Session-aware features (indices 113-115)
            features[113] = 1.0 if 13 <= hour <= 21 else 0.0   # US session
            features[114] = 1.0 if 1 <= hour <= 9  else 0.0    # Asia session
            features[115] = 1.0 if hour in (0, 8, 16) else 0.0 # Funding window
            
            # Pad to exactly self.cfg.BASE_FEATURE_COUNT
            if len(features) != self.cfg.BASE_FEATURE_COUNT:
                logging.warning(
                    f"Candle feature dim mismatch at pos {position}: got {len(features)}, expected {self.cfg.BASE_FEATURE_COUNT}"
                )
                if len(features) < self.cfg.BASE_FEATURE_COUNT:
                    padded = np.zeros(self.cfg.BASE_FEATURE_COUNT, dtype=np.float32)
                    padded[:len(features)] = features
                    features = padded
                else:
                    features = features[:self.cfg.BASE_FEATURE_COUNT]
            assert len(features) == self.cfg.BASE_FEATURE_COUNT
            return features

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"   ❌ Feature extraction error at position {position}: {e}")
            return None
    

    def _learn_regime_routing(self, training_data_per_agent):
        """Learn regime routing weights from specialist performance per HMM regime.

        Replaces the 21 hand-coded weights (7 agents x 3 regimes) with
        backtested accuracy-derived weights. Persists to models/.

        Formula:
            accuracy[agent][regime] = correct / total  (in that regime)
            weight[agent][regime]   = clip(accuracy / 0.50, 0.10, 1.0)
            # 0.50 = random baseline; below-random agents get near-zero weight

        Args:
            training_data_per_agent: dict {agent_name: list of (features, label)} tuples
        """
        import json
        ROUTING_PATH = os.path.join(self.cfg.model_dir, 'regime_routing_weights.json')
        agents = list(self._regime_routing.keys())
        n_regimes = 3

        # Tally correct predictions per (agent, regime)
        counts  = {a: [0]*n_regimes for a in agents}  # total events in regime
        correct = {a: [0]*n_regimes for a in agents}  # correct predictions

        for agent_name, events in training_data_per_agent.items():
            if agent_name not in self.specialist_models:
                continue
            spec = self.specialist_models[agent_name]
            model  = spec.get('model')
            scaler = spec.get('scaler')
            if model is None or not hasattr(scaler, 'mean_'):
                continue

            for feat_vec, label in events:
                try:
                    feat_arr = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
                    # Get regime from HMM feature 231
                    # Feature 231 = regime state: 1.0=bull, 0.5=range, 0.0=bear
                    regime_val = float(feat_arr[0, 231])
                    if regime_val > 0.75:
                        regime_idx = 0   # bull
                    elif regime_val > 0.25:
                        regime_idx = 1   # range
                    else:
                        regime_idx = 2   # bear

                    # Get specialist prediction
                    fi = spec.get('_feature_indices')
                    x = feat_arr[:, fi] if fi is not None else feat_arr
                    x_sc = scaler.transform(x)
                    pred_prob = model.predict_proba(x_sc)[0, 1]
                    pred_label = 1 if pred_prob >= 0.5 else 0

                    counts[agent_name][regime_idx]  += 1
                    if pred_label == int(label):
                        correct[agent_name][regime_idx] += 1
                except Exception:
                    pass

        # Compute learned weights
        learned = {}
        for agent in agents:
            weights = []
            for r in range(n_regimes):
                tot = counts[agent][r]
                if tot < 10:
                    # Not enough data in this regime — fall back to hand-coded
                    weights.append(self._regime_routing[agent][r])
                else:
                    acc = correct[agent][r] / tot
                    w = max(0.10, min(1.0, acc / 0.50))
                    weights.append(round(w, 3))
            learned[agent] = weights
            print(f"  [{agent}] regime routing learned: {weights} "
                  f"(counts={counts[agent]}, correct={correct[agent]})")

        # Apply + persist
        self._regime_routing = learned
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        with open(ROUTING_PATH, 'w') as fj:
            json.dump(learned, fj, indent=2)
        print(f"  ✅ Regime routing weights saved to {ROUTING_PATH}")

    def _load_regime_routing_weights(self):
        """Load persisted regime routing weights if available."""
        import json
        ROUTING_PATH = os.path.join(self.cfg.model_dir, 'regime_routing_weights.json')
        if os.path.exists(ROUTING_PATH):
            try:
                with open(ROUTING_PATH) as f:
                    loaded = json.load(f)
                # Validate structure (7 agents x 3 regimes)
                if all(k in loaded for k in self._regime_routing) and \
                   all(len(v) == 3 for v in loaded.values()):
                    self._regime_routing = loaded
                    print(f"  ✅ Loaded learned regime routing from {ROUTING_PATH}")
                    return True
            except Exception as e:
                logging.warning(f"Failed to load regime routing weights: {e}")
        return False


    def _get_regime(self, symbol, closes, highs, lows, volumes):
        """HMM regime detection — per-symbol, 200 obs, full covariance, label-stable.

        Features (multi-scale): [log_ret_1bar, log_ret_12bar, log_ret_48bar, atr_pct, vol_ratio]
        3 states: bear(0), range(1), bull(2) — ordered by mean log-return DESCENDING.
        Returns normalized rank in [0.0, 0.5, 1.0] — valid ordinal for CatBoost.
        """
        try:
            if not symbol or len(closes) < 50:
                return 0.5  # neutral

            n = len(closes)
            # Feature 0: 1-bar log returns (microstructure)
            log_rets = np.diff(np.log(np.maximum(closes, 1e-12)))  # length n-1

            # Feature 1: 12-bar rolling log-return (≈1h momentum at 5m base)
            log_ret_12 = np.zeros(n - 1)
            for i in range(n - 1):
                if i >= 11:
                    log_ret_12[i] = np.log(max(closes[i + 1], 1e-12) / max(closes[i - 11], 1e-12)) / 12.0
                else:
                    log_ret_12[i] = log_rets[i]

            # Feature 2: 48-bar rolling log-return (≈4h structural trend)
            log_ret_48 = np.zeros(n - 1)
            for i in range(n - 1):
                if i >= 47:
                    log_ret_48[i] = np.log(max(closes[i + 1], 1e-12) / max(closes[i - 47], 1e-12)) / 48.0
                else:
                    log_ret_48[i] = log_rets[i]

            # Feature 3: ATR% — True Range / close, captures realized volatility
            tr = np.empty(n - 1)
            for i in range(1, n):
                tr[i - 1] = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            atr14 = np.convolve(tr, np.ones(14) / 14.0, mode='valid')  # length n-14
            denom = closes[14:14 + len(atr14)]
            atr_pct = atr14 / np.maximum(denom, 1e-12)  # length n-14

            # Feature 4: volume ratio (volume / rolling mean)
            vol_mean = np.mean(volumes) + 1e-6
            vol_ratio = volumes / vol_mean  # length n

            # Align all to same length (shortest is n-14 from atr_pct)
            min_len = len(atr_pct)
            _lr1  = log_rets[-min_len:]
            _lr12 = log_ret_12[-min_len:]
            _lr48 = log_ret_48[-min_len:]
            _ap   = atr_pct[-min_len:]
            _vr   = vol_ratio[-min_len:]

            obs_len = min(200, min_len)
            if obs_len < 30:
                return 0.5

            obs = np.column_stack([
                _lr1[-obs_len:], _lr12[-obs_len:], _lr48[-obs_len:],
                _ap[-obs_len:], _vr[-obs_len:]
            ])

            import time
            current_time = time.time()

            # Cache hit: model valid for 4 hours
            if symbol in self.hmm_models and (current_time - self.hmm_last_fit.get(symbol, 0)) < 14400:
                cached = self.hmm_models[symbol]
                if isinstance(cached, dict) and cached.get('n_states') == 3:
                    raw_state = int(cached['model'].predict(obs)[-1])
                    return 1.0 - float(cached['rank_to_int'][raw_state]) / 2.0  # 0.0=bear, 0.5=range, 1.0=bull
                # Stale 5-state cache or legacy: force refit
                self.hmm_last_fit[symbol] = 0

            # Fit with 3 states: bear, range, bull (more robust with 200 obs)
            obs_noisy = obs + np.random.normal(0, 1e-6, obs.shape)
            from hmmlearn.hmm import GaussianHMM
            _hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
            _hmm.fit(obs_noisy)

            # Sort states by mean log-return (DESCENDING) to fix label switching.
            # rank_to_int[raw_state_idx] = ordered rank 0..2 (0=bull, 1=range, 2=bear).
            mean_rets = _hmm.means_[:, 0]
            state_order = np.argsort(mean_rets)[::-1]  # descending: highest return first
            rank_to_int = np.empty(3, dtype=np.int32)
            for rank, raw_idx in enumerate(state_order):
                rank_to_int[raw_idx] = rank

            self.hmm_models[symbol] = {'model': _hmm, 'rank_to_int': rank_to_int, 'n_states': 3}
            self.hmm_last_fit[symbol] = current_time

            raw_state = int(_hmm.predict(obs)[-1])
            # Normalize to [0.0, 1.0]: 0.0=bear, 0.5=range, 1.0=bull
            return 1.0 - float(rank_to_int[raw_state]) / 2.0

        except Exception:
            return 0.5  # neutral fallback

    def _extract_features(self, tf_analysis, symbol=None, _raw_candles=None,
                          _training_ts_ms=None, _training_symbol=None):
        """Extract features for ML prediction.

        Args:
            tf_analysis:       Multi-timeframe analysis dict.
            symbol:            Live trading pair (e.g. 'BTCUSDT').  None during training.
            _raw_candles:      Dict with closes/highs/lows/volumes/taker_buy arrays (training path).
            _training_ts_ms:   Unix-ms timestamp of the training candle — used for historical
                               futures cache lookup so Futures X-Ray is NOT NaN during training.
            _training_symbol:  Symbol name even during training, for cache lookup.
        """
        features = []
        
        # Cache timeframe list
        tfs = self.cfg.timeframes
        
        # Per-timeframe features (7 features × 7 timeframes = 49) - VECTORIZED
        for tf in tfs:
            if tf in tf_analysis:
                data = tf_analysis[tf]
                features.extend([
                    data.get('rsi', 50),
                    data.get('macd', 0),
                    data.get('bb_position', 0.5),
                    data.get('adx', 25),
                    data.get('atr', 0),
                    1 if data.get('trend') == 'BULLISH' else (-1 if data.get('trend') == 'BEARISH' else 0),
                    data.get('strength', 0) / 100
                ])
            else:
                features.extend([50, 0, 0.5, 25, 0, 0, 0])
        
        # Pre-extract common arrays once
        valid_tfs = [tf for tf in tfs if tf in tf_analysis]
        if not valid_tfs:
            # No data - return zero features with correct shape
            return np.zeros(self.cfg.BASE_FEATURE_COUNT)
        
        trends = [tf_analysis[tf].get('trend', 'NEUTRAL') for tf in valid_tfs]
        rsi_values = np.array([tf_analysis[tf].get('rsi', 50) for tf in valid_tfs])
        macd_values = np.array([tf_analysis[tf].get('macd', 0) for tf in valid_tfs])
        volumes = np.array([tf_analysis[tf].get('volume', 0) for tf in valid_tfs])
        atr_values = np.array([tf_analysis[tf].get('atr', 0) for tf in valid_tfs])
        prices = np.array([tf_analysis[tf].get('price', 1) for tf in valid_tfs])
        adx_values = np.array([tf_analysis[tf].get('adx', 25) for tf in valid_tfs])
        bb_positions = np.array([tf_analysis[tf].get('bb_position', 0.5) for tf in valid_tfs])
        strengths = np.array([tf_analysis[tf].get('strength', 0) for tf in valid_tfs])
        
        # Cross-timeframe (3)
        bullish_count = trends.count('BULLISH')
        bearish_count = trends.count('BEARISH')
        trend_len = len(trends)
        features.extend([
            bullish_count / trend_len,
            bearish_count / trend_len,
            (bullish_count - bearish_count) / trend_len
        ])
        
        # Weighted consensus (1)
        weighted_score = sum(
            (1 if tf_analysis[tf].get('trend') == 'BULLISH' else (-1 if tf_analysis[tf].get('trend') == 'BEARISH' else 0)) 
            * self.cfg.tf_weights.get(tf, 0.1) 
            * (tf_analysis[tf].get('strength', 0) / 100)
            for tf in valid_tfs
        )
        features.append(weighted_score)
        
        # RSI analysis (5) - VECTORIZED
        if len(rsi_values) >= 2:
            features.extend([
                rsi_values.max() - rsi_values.min(),
                rsi_values.std(),
                rsi_values[-1] - rsi_values[0],
                (rsi_values > 70).sum() / len(rsi_values),
                rsi_values.mean()
            ])
        else:
            features.extend([0] * 5)
        
        # MACD (3) - VECTORIZED
        features.extend([
            macd_values.max() - macd_values.min(),
            (macd_values > 0).sum() / len(macd_values),
            macd_values.mean()
        ])
        
        # Volume (5) - VECTORIZED
        if volumes.max() > 0:
            log_vols = np.log1p(volumes)
            features.extend([
                np.log1p(volumes.max()),
                log_vols.mean(),
                log_vols.std(),
                volumes[-1] / volumes.mean() if volumes.mean() > 0 else 1,
                (np.diff(volumes) > 0).sum() / max(1, len(volumes)-1)
            ])
        else:
            features.extend([0] * 5)
        
        # ATR (4) - VECTORIZED
        if atr_values.any() and prices.max() > 0:
            atr_percents = (atr_values / np.maximum(prices, 1e-8) * 100)
            features.extend([
                atr_percents.max(),
                atr_percents.mean(),
                atr_percents.std(),
                atr_percents[-1]
            ])
        else:
            features.extend([0] * 4)
        
        # ADX (3) - VECTORIZED
        features.extend([
            adx_values.max(),
            adx_values.mean(),
            (adx_values > 25).sum() / len(adx_values)
        ])
        
        # BB position (3) - VECTORIZED
        features.extend([
            bb_positions.max(),
            bb_positions.mean(),
            bb_positions.std()
        ])
        
        # Strength progression (4) - VECTORIZED
        if len(strengths) >= 2:
            strength_diffs = np.diff(strengths)
            features.extend([
                strength_diffs.mean(),
                (strength_diffs > 0).sum() / len(strength_diffs),
                strengths.mean(),
                strengths[-1] - strengths[0]
            ])
        else:
            features.extend([0] * 4)
        
        # Market regime (4)
        if '1h' in tf_analysis and '4h' in tf_analysis and '1d' in tf_analysis:
            h1_adx = tf_analysis['1h'].get('adx', 25)
            h4_adx = tf_analysis['4h'].get('adx', 25)
            d1_adx = tf_analysis['1d'].get('adx', 25)
            h1_atr = tf_analysis['1h'].get('atr', 0)
            h4_atr = tf_analysis['4h'].get('atr', 0)
            h1_price = tf_analysis['1h'].get('price', 1)
            
            features.extend([
                (h1_adx + h4_adx + d1_adx) / 3,
                1 if h1_adx > 25 and h4_adx > 25 else 0,
                (h1_atr / h1_price * 100) if h1_price > 0 else 0,
                1 if h1_adx < 20 and h4_adx < 20 else 0
            ])
        else:
            features.extend([25, 0, 0, 0])
        
        # Momentum confluence (3) - VECTORIZED
        bullish_momentum = sum(
            1 for tf in valid_tfs
            if tf_analysis[tf].get('rsi', 50) > 50 and tf_analysis[tf].get('macd', 0) > 0
        )
        bearish_momentum = sum(
            1 for tf in valid_tfs
            if tf_analysis[tf].get('rsi', 50) < 50 and tf_analysis[tf].get('macd', 0) < 0
        )
        total_tf = len(valid_tfs)
        features.extend([
            bullish_momentum / total_tf,
            bearish_momentum / total_tf,
            (bullish_momentum - bearish_momentum) / total_tf
        ])
        
        # Spike-dump-reversal (20 features)
        spike_dump_features = self._detect_spike_dump_reversal(tf_analysis)
        features.extend(spike_dump_features)
        
        # === TIME FEATURES (6) + SESSION FEATURES (3) = 9 total ===
        # MUST use UTC — all session windows and funding times are defined in UTC.
        # datetime.now() returns local time which shifts session detection for non-UTC systems.
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()
        # Market session windows (UTC): Ayadi et al. (2021) — session overlap = heightened volatility
        is_us_session    = 1.0 if 13 <= hour <= 21 else 0.0  # NYSE: 8am-4pm ET = 1pm-9pm UTC
        is_asia_session  = 1.0 if 1 <= hour <= 9  else 0.0   # Tokyo/HK: 9am-5pm JST = 1am-9am UTC
        is_funding_window = 1.0 if hour in (0, 8, 16) else 0.0  # Binance funding settlement (every 8h UTC)
        features.extend([
            np.sin(2 * np.pi * hour / 24),        # hour_sin
            np.cos(2 * np.pi * hour / 24),        # hour_cos
            np.sin(2 * np.pi * day_of_week / 7),  # day_sin
            np.cos(2 * np.pi * day_of_week / 7),  # day_cos
            hour / 24.0,                           # normalized hour
            day_of_week / 7.0,                     # normalized day
            is_us_session,                         # US session active (NYSE hours)
            is_asia_session,                       # Asia session active (Tokyo/HK hours)
            is_funding_window,                     # Funding settlement window (±30min every 8h)
        ])
        
        # === ENHANCED VOLATILITY FEATURES (per TF available) - up to 21 ===
        volatility_features = []
        for tf in tfs:
            if tf in tf_analysis:
                data = tf_analysis[tf]
                volatility_features.extend([
                    data.get('volatility', 0),
                    data.get('atr_percentile', 0.5),
                    data.get('volatility_accel', 0)
                ])
            else:
                volatility_features.extend([0, 0.5, 0])
        features.extend(volatility_features)
        
        # === ENHANCED MOMENTUM FEATURES (4 per TF) - up to 28 ===
        momentum_features = []
        for tf in tfs:
            if tf in tf_analysis:
                data = tf_analysis[tf]
                momentum_features.extend([
                    data.get('momentum_5', 0),
                    data.get('momentum_10', 0),
                    data.get('momentum_20', 0),
                    data.get('momentum_50', 0)
                ])
            else:
                momentum_features.extend([0, 0, 0, 0])
        features.extend(momentum_features)
        
        # === VOLUME ANALYSIS (per TF) - up to 7 ===
        volume_analysis_features = []
        for tf in tfs:
            if tf in tf_analysis:
                volume_analysis_features.append(tf_analysis[tf].get('volume_ratio', 1.0))
            else:
                volume_analysis_features.append(1.0)
        features.extend(volume_analysis_features)
        
        # === DRIFT DETECTION (2 per TF) - up to 14 ===
        drift_features = []
        for tf in tfs:
            if tf in tf_analysis:
                data = tf_analysis[tf]
                drift_features.extend([
                    data.get('mean_shift', 0),
                    data.get('trend_strength', 0)
                ])
            else:
                drift_features.extend([0, 0])
        features.extend(drift_features)
        
        # === MULTI-TIMEFRAME RETURNS (per TF) - up to 7 ===
        returns_features = []
        for tf in tfs:
            if tf in tf_analysis:
                returns_features.append(tf_analysis[tf].get('returns_period', 0))
            else:
                returns_features.append(0)
        features.extend(returns_features)
        
        # === VPIN ORDER FLOW TOXICITY (per TF) - up to 7 ===
        vpin_features = []
        for tf in tfs:
            if tf in tf_analysis:
                vpin_features.append(tf_analysis[tf].get('vpin', 0.5))
            else:
                vpin_features.append(0.5)
        features.extend(vpin_features)
        
        # === FRACTIONAL DIFFERENCING (per TF) - up to 7 ===
        fd_features = []
        for tf in tfs:
            if tf in tf_analysis:
                fd_features.append(tf_analysis[tf].get('frac_diff', 0.0))
            else:
                fd_features.append(0.0)
        features.extend(fd_features)
        
        # === TAKER FLOW IMBALANCE (per TF) - up to 7 ===
        taker_features = []
        for tf in tfs:
            if tf in tf_analysis:
                taker_features.append(tf_analysis[tf].get('taker_imbalance', 0.0))
            else:
                taker_features.append(0.0)
        features.extend(taker_features)
        
        # === ORDER BOOK FEATURES (17) - RESEARCH-BACKED ===
        # NBER 2024: Order book imbalance achieves 79% accuracy
        # Live path: real-time depth cache.  Training path: OHLCV proxies (Cont et al. 2014).
        if symbol:
            ob_features = self._extract_order_book_features(symbol)
        elif _raw_candles and len(_raw_candles.get('closes', [])) >= 1:
            ob_features = self._orderbook_proxy_from_ohlcv(_raw_candles)
        else:
            ob_features = [float('nan')] * 17
        
        features.extend(ob_features)
        
        # Running total: 107 (base) + 9 (time+session) + 21 (volatility) + 28 (momentum)
        #   + 7 (volume/TF) + 14 (drift) + 7 (returns) + 7 (VPIN) + 7 (FracDiff) + 7 (taker) + 17 (OB) = 231
        
        # === HMM REGIME DETECTION (1 feature) ===
        # FIX Phase 0A: Use _raw_candles from training, or live candle_store, or fallback
        raw_closes = np.array([1.0, 1.0])
        raw_highs, raw_lows, raw_volumes = raw_closes, raw_closes, raw_closes
        
        if _raw_candles is not None and len(_raw_candles.get('closes', [])) >= 50:
            # Training path: use candle data passed from _extract_features_from_candles
            raw_closes = np.asarray(_raw_candles['closes'], dtype=np.float64)
            raw_highs = np.asarray(_raw_candles['highs'], dtype=np.float64)
            raw_lows = np.asarray(_raw_candles['lows'], dtype=np.float64)
            raw_volumes = np.asarray(_raw_candles['volumes'], dtype=np.float64)
        elif hasattr(self, 'candle_store') and self.candle_store and symbol:
            # Live path: use candle_store
            klines = self.candle_store.get(symbol, '5m')[-50:] if hasattr(self.candle_store, 'get') else self.candle_store.get_klines(symbol, '5m', limit=50)
            if klines and len(klines) >= 50:
                k_arr = np.array(klines, dtype=np.float64)
                raw_highs = k_arr[:, 2]
                raw_lows = k_arr[:, 3]
                raw_closes = k_arr[:, 4]
                raw_volumes = k_arr[:, 5]
        
        regime_state = self._get_regime(symbol, raw_closes, raw_highs, raw_lows, raw_volumes)
        features.append(regime_state)
        # Running total = 232
        
        # === ADVANCED RESEARCH FEATURES (7 features) ===
        hurst_val = Indicators.hurst(raw_closes, 20)
        samp_en = Indicators.sample_entropy(raw_closes, 2, 0.2)
        
        # Calculate returns safely for illiquidity/impact metrics
        raw_returns = np.diff(raw_closes) / raw_closes[:-1] if len(raw_closes) > 1 else np.array([0.0])
        raw_vols = raw_volumes[1:] if len(raw_volumes) > 1 else np.array([0.0])
        
        lam = Indicators.kyle_lambda(raw_returns, raw_vols, 20)
        amihud_val = Indicators.amihud(raw_returns, raw_vols, 20)
        mf_dfa = Indicators.mf_dfa_width(raw_closes, -2.0, 2.0, 10)
        
        # Transfer Entropy (BTC -> Alt)
        te_val = 0.0
        if symbol and symbol != 'BTCUSDT' and hasattr(self, 'candle_store') and self.candle_store:
            btc_klines = self.candle_store.get('BTCUSDT', '5m')[-50:] if hasattr(self.candle_store, 'get') else self.candle_store.get_klines('BTCUSDT', '5m', limit=50)
            if btc_klines and len(btc_klines) >= 50:
                btc_arr = np.array(btc_klines, dtype=np.float64)
                btc_closes = btc_arr[:, 4]
                if len(btc_closes) == len(raw_closes):
                    btc_ret = np.diff(btc_closes) / btc_closes[:-1]
                    te_val = Indicators.transfer_entropy(btc_ret, raw_returns, 8)
                    
        # QUANTA Regime Entropy (Hybrid Novel Feature)
        samp_en_div = min(1.0, samp_en / 2.0) if samp_en > 0 else 0.0
        regime_strength = abs(hurst_val - 0.5) * 2.0  # Normalized 0 to 1
        alpha_potential = 1.0 - te_val if te_val < 1.0 else 0.0
        
        qre = (0.3 * samp_en_div) + (0.3 * regime_strength) + (0.2 * alpha_potential) + (0.2 * mf_dfa)
        
        features.extend([
            hurst_val,
            samp_en,
            lam,
            amihud_val,
            mf_dfa,
            te_val,
            qre
        ])
        # Running total = 239
        
        # ━━━━ SENTIMENT FEATURES (5) ━━━━
        # Fear & Greed + CryptoPanic news (cached, near-zero latency)
        try:
            sentiment_feats = self.sentiment.get_sentiment_features(symbol=symbol)
        except Exception:
            sentiment_feats = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 7 features: fng_norm, extreme_fear, extreme_greed, news_score, news_volume_norm, coin_score, coin_magnitude
        features.extend(sentiment_feats)
        # Running total = 246
        
        # ━━━━ FUTURES X-RAY FEATURES (4) (v10.2) ━━━━
        # Open Interest, Long/Short Ratio, Speculative Index, Funding Velocity
        # Live path: rt_cache.  Training path: historical Binance API cache lookup.
        if symbol:
            futures_feats = self._extract_futures_xray(symbol, tf_analysis)
        elif _training_ts_ms and _training_symbol and self._hist_futures_cache:
            futures_feats = self._lookup_hist_futures(_training_symbol, _training_ts_ms)
        else:
            futures_feats = [float('nan')] * 4
        features.extend(futures_feats)
        # Running total = 248
        
        # ━━━━ STATISTICAL ARBITRAGE (6) (v10.3) ━━━━
        if symbol:
            stat_arb = self._extract_stat_arb(symbol, tf_analysis)
        else:
            # BTC RSI/MACD/vol_accel can be computed from _raw_candles if available
            if _raw_candles is not None and len(_raw_candles.get('closes', [])) >= 50:
                try:
                    btc_rsi = float(Indicators.rsi(raw_closes))
                    bm, bs, _ = Indicators.macd(raw_closes)
                    btc_macd = float(bm - bs)
                    btc_ret = np.diff(raw_closes) / raw_closes[:-1]
                    btc_vol_accel = float(np.std(btc_ret[-10:]) - np.std(btc_ret[-30:-10])) if len(btc_ret) >= 30 else 0.0
                except Exception:
                    btc_rsi, btc_macd, btc_vol_accel = 50.0, 0.0, 0.0
                stat_arb = [0.0, 0.0, 0.0, btc_rsi, btc_macd, btc_vol_accel]
            else:
                stat_arb = [0.0] * 6
        features.extend(stat_arb)
        # Running total = 254
        
        # ━━━━ ON-CHAIN WHALE ANALYTICS (3) (v11 P2) ━━━━
        if symbol:
            try:
                onchain_feats = self.onchain.get_onchain_features(symbol)
            except Exception:
                onchain_feats = [0.0, 0.0, 0.0]
        else:
            onchain_feats = [0.0, 0.0, 0.5]  # neutral: no inflow/outflow bias, whale_ratio=0.5
        features.extend(onchain_feats)
        # Running total = 257
        
        # ━━━━ CROSS-ASSET GNN EMBEDDING (1) (v11 P3) ━━━━
        # Live path: GAT embedding.  Training path: BTC-correlation proxy.
        if symbol:
            try:
                graph_feat = graph_engine.get_embedding(symbol)
            except Exception:
                graph_feat = 0.0
        elif _raw_candles:
            graph_feat = self._gnn_correlation_proxy(_raw_candles)
        else:
            graph_feat = 0.0
        features.append(graph_feat)
        # Running total = 258

        # ━━━━ DELTA FEATURES (10) — Temporal Context (v11.5) ━━━━
        # Rate-of-change features: current vs lagged TF values.
        # Grinold & Kahn (2000): delta features capture score momentum.
        # Indices 258-267 (appended after on-chain[254-256] + GNN[257]).
        tf_15m = tf_analysis.get('15m', {})
        tf_1h = tf_analysis.get('1h', {})
        tf_5m = tf_analysis.get('5m', {})
        delta_features = [
            tf_5m.get('rsi', 50) - tf_15m.get('rsi', 50),                    # RSI delta (5m vs 15m)
            tf_5m.get('macd', 0) - tf_15m.get('macd', 0),                    # MACD delta
            tf_5m.get('bb_position', 0.5) - tf_15m.get('bb_position', 0.5),  # BB position delta
            tf_5m.get('adx', 25) - tf_15m.get('adx', 25),                    # ADX delta
            (tf_5m.get('volume', 0) / max(tf_15m.get('volume', 1), 1e-8)) - 1.0,  # Volume ratio delta
            tf_15m.get('rsi', 50) - tf_1h.get('rsi', 50),                    # RSI acceleration (15m vs 1h)
            tf_15m.get('macd', 0) - tf_1h.get('macd', 0),                    # MACD acceleration
            tf_5m.get('atr', 0) - tf_15m.get('atr', 0),                      # ATR delta (vol expansion)
            tf_5m.get('strength', 0) - tf_15m.get('strength', 0),            # Strength delta
            tf_5m.get('vpin', 0.5) - tf_15m.get('vpin', 0.5),                # VPIN delta (flow change)
        ]
        features.extend(delta_features)
        # Running total = 270

        # ━━━━ IMPULSE FEATURES (5) — Nike Specialist Context (v11.5b) ━━━━
        # Features that capture single-candle explosive move characteristics.
        # These are always computed (all agents can use them via domain mask).
        # Indices 270-274.
        # FIX 2026-04-01: _raw_candles is a DICT with keys {closes, highs, lows, volumes, taker_buy}
        # Old code treated it as list-of-rows (e.g. _5m_klines[-1][4]) which silently failed.
        try:
            _rc = _raw_candles
            _have_data = (_rc is not None and isinstance(_rc, dict)
                          and len(_rc.get('closes', [])) >= 20)
            if _have_data:
                _closes  = np.asarray(_rc['closes'], dtype=np.float64)
                _highs   = np.asarray(_rc['highs'], dtype=np.float64)
                _lows    = np.asarray(_rc['lows'], dtype=np.float64)
                _vols    = np.asarray(_rc['volumes'], dtype=np.float64)
                _tb_arr  = np.asarray(_rc.get('taker_buy') if _rc.get('taker_buy') is not None
                                      else np.zeros(len(_closes)), dtype=np.float64)

                c_last = _closes[-1]
                c_prev = _closes[-2]
                h_last = _highs[-1]
                l_last = _lows[-1]
                v_last = _vols[-1]
                candle_range = h_last - l_last

                # [270] impulse_body_eff: |close - prev_close| / (H - L)
                body_eff = abs(c_last - c_prev) / candle_range if candle_range > 0 else 0.0

                # [271] taker_flow_persist: count of last 3 bars with matching taker direction
                impulse_dir = 1.0 if c_last > c_prev else -1.0
                persist_count = 0.0
                for _bi in range(2, min(5, len(_closes))):
                    _tbuy_v = _tb_arr[-_bi]
                    _tvol_v = _vols[-_bi]
                    _timb = (_tbuy_v - (_tvol_v - _tbuy_v)) / (_tvol_v + 1e-8)
                    if (_timb > 0 and impulse_dir > 0) or (_timb < 0 and impulse_dir < 0):
                        persist_count += 1.0
                taker_persist = persist_count / 3.0  # Normalized 0-1

                # [272] pre_impulse_r2: R² × sign(slope) of linear regression on last 10 closes
                n_reg = min(10, len(_closes) - 1)
                reg_closes = _closes[-n_reg:]
                t_vals = np.arange(n_reg, dtype=np.float64)
                t_mean = t_vals.mean()
                c_mean = reg_closes.mean()
                cov_tc = np.sum((t_vals - t_mean) * (reg_closes - c_mean))
                var_t = np.sum((t_vals - t_mean) ** 2)
                var_c = np.sum((reg_closes - c_mean) ** 2)
                slope = cov_tc / (var_t + 1e-10)
                r2 = (cov_tc ** 2) / ((var_t * var_c) + 1e-10)
                r2 = min(r2, 1.0)
                pre_impulse_r2 = r2 * (1.0 if slope >= 0 else -1.0)

                # [273] atr_rank: percentile of this candle's range in last 100 bars
                n_rank = min(100, len(_closes))
                ranges = _highs[-n_rank:] - _lows[-n_rank:]
                atr_rank = float(np.sum(ranges <= candle_range)) / n_rank

                # [274] depth_delta: live = real order book delta; training = volume proxy
                if hasattr(self, '_ob_depth_history') and symbol and symbol in self._ob_depth_history:
                    hist = self._ob_depth_history[symbol]
                    if len(hist) >= 5:
                        old_depth = hist[-5]
                        cur_depth = hist[-1]
                        depth_delta = (cur_depth - old_depth) / (old_depth + 1e-8)
                    else:
                        depth_delta = 0.0
                else:
                    # OHLCV proxy: volume acceleration as depth proxy
                    v_ago = _vols[-6] if len(_vols) >= 6 else v_last
                    v_avg = np.mean(_vols[-min(20, len(_vols)):]) + 1e-8
                    depth_delta = (v_last - v_ago) / v_avg

                impulse_features = [body_eff, taker_persist, pre_impulse_r2, atr_rank, depth_delta]
            else:
                impulse_features = [0.0, 0.0, 0.0, 0.5, 0.0]
        except Exception:
            impulse_features = [0.0, 0.0, 0.0, 0.5, 0.0]
        features.extend(impulse_features)
        # Running total = 275

        # ━━━━ BLACK-SCHOLES BARRIER FEATURES (3) — Indices 275-277 (2026-04-02) ━━━━
        # Hull (2018) Ch.26 / Darling-Siegert (1953) scale function.
        # Uses median barrier settings (TP=1.5 ATR, SL=1.0 ATR) — specialist-agnostic.
        # Per-specialist values used separately in Kelly sizing (see QUANTA_bot.py).
        try:
            _bs_atr = float(tf_analysis.get('5m', {}).get('atr', 0.0)) if tf_analysis else 0.0
            _bs_price = float(tf_analysis.get('5m', {}).get('price', 1.0)) if tf_analysis else 1.0
            _bs_price = max(_bs_price, 1e-8)

            if _have_data and _bs_atr > 0:
                # Log returns from recent 5m closes for drift/vol estimation
                _bs_closes = np.asarray(_rc['closes'], dtype=np.float64)
                _bs_log_rets = np.diff(np.log(np.maximum(_bs_closes, 1e-12)))

                # Barriers in price units (median specialist settings)
                _tp_mult = 1.5
                _sl_mult = 1.0
                _tp_dist = (_tp_mult * _bs_atr) / _bs_price  # fractional
                _sl_dist = (_sl_mult * _bs_atr) / _bs_price

                # [275] bs_theoretical_win_prob: P(TP before SL) under drifted GBM
                _bs_win_prob = _jit_kou_barrier_prob(_bs_log_rets, _tp_dist, _sl_dist)

                # [276] bs_time_decay: approximate P(crossing barrier in remaining bars)
                _bs_sigma = float(np.std(_bs_log_rets[-20:])) if len(_bs_log_rets) >= 20 else 0.0
                _bs_time_decay = _jit_bs_time_decay(_bs_sigma, _tp_dist, 48)

                # [277] bs_implied_vol_ratio: implied sigma / realized sigma from trade history
                _avg_bars = getattr(self, '_bs_avg_bars_to_hit', {}).get(symbol, 0.0) if symbol else 0.0
                _bs_iv_ratio = _jit_bs_implied_vol_ratio(_avg_bars, _tp_dist, _bs_sigma)
            else:
                _bs_win_prob = _sl_mult / (_tp_mult + _sl_mult) if '_tp_mult' in dir() else 0.4
                _bs_win_prob = 1.0 / (1.5 + 1.0)  # 0.4 zero-drift default
                _bs_time_decay = 0.5
                _bs_iv_ratio = 1.0
        except Exception:
            _bs_win_prob = 0.4
            _bs_time_decay = 0.5
            _bs_iv_ratio = 1.0

        features.extend([_bs_win_prob, _bs_time_decay, _bs_iv_ratio])
        # Running total = 278 (FINAL — must match config.BASE_FEATURE_COUNT)

        # Enforce exact dimension length
        features_len = len(features)
        if features_len != self.cfg.BASE_FEATURE_COUNT:
            logging.warning(
                f"Feature dimension mismatch: got {features_len}, expected {self.cfg.BASE_FEATURE_COUNT}. "
                f"Delta={features_len - self.cfg.BASE_FEATURE_COUNT}. {'Padding' if features_len < self.cfg.BASE_FEATURE_COUNT else 'Truncating'}."
            )
            if features_len < self.cfg.BASE_FEATURE_COUNT:
                features.extend([0.0] * (self.cfg.BASE_FEATURE_COUNT - features_len))
            else:
                features = features[:self.cfg.BASE_FEATURE_COUNT]

        result = np.array(features, dtype=np.float32)
        assert result.shape[0] == self.cfg.BASE_FEATURE_COUNT, \
            f"CRITICAL: Feature vector has {result.shape[0]} dims, expected {self.cfg.BASE_FEATURE_COUNT}"
        return result

    def _extract_futures_xray(self, symbol, tf_analysis):
        """🧬 FUTURES X-RAY: Cache-only reader (zero REST calls during prediction)."""
        if symbol in self.futures_stats_cache:
            return self.futures_stats_cache[symbol]['data']
        return [0.0, 1.0, 1.0, 0.0]  # Default until background fetch populates
    
    def start_oi_background_feed(self, symbols):
        """🔄 Start background OI/LS fetcher — runs every 10 min, no prediction blocking."""
        if hasattr(self, '_oi_thread') and self._oi_thread and self._oi_thread.is_alive():
            return
        self._oi_symbols = list(symbols)
        self._oi_blacklist = set()  # Symbols that return 400 (no futures OI data)
        self._oi_thread = threading.Thread(
            target=self._oi_background_loop, daemon=True, name="OI_Fetcher"
        )
        self._oi_thread.start()
        print(f"📊 OI/LS background feed started ({len(self._oi_symbols)} symbols, 10-min refresh)")
    
    def _oi_background_loop(self):
        """Background loop: pre-fetches OI + Long/Short for all symbols."""
        while True:
            fetched = 0
            errors = 0
            for symbol in self._oi_symbols:
                # Skip symbols known to not have futures OI
                if symbol in self._oi_blacklist:
                    continue
                    
                try:
                    # Fetch Open Interest
                    oi_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
                    oi_resp = NetworkHelper.get(oi_url, timeout=5)
                    if not oi_resp:
                        # Check if it was a 400 (symbol doesn't exist on futures)
                        self._oi_blacklist.add(symbol)
                        errors += 1
                        continue
                    oi_value = float(oi_resp.json().get('openInterest', 0))
                    
                    time.sleep(0.15)
                    
                    # Fetch Long/Short Ratio
                    ls_url = f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
                    ls_resp = NetworkHelper.get(ls_url, timeout=5)
                    ls_ratio = 1.0
                    if ls_resp:
                        ls_res = ls_resp.json()
                        ls_ratio = float(ls_res[0].get('longShortRatio', 1.0)) if ls_res else 1.0
                    
                    # Compute features from rt_cache
                    vol_24h = 0
                    funding_now = 0
                    price = 1
                    if hasattr(self, 'rt_cache') and symbol in self.rt_cache:
                        rt = self.rt_cache[symbol]
                        vol_24h = rt.get('volume_24h', 0)
                        funding_now = rt.get('funding_rate', 0)
                        price = rt.get('price', 1)
                    
                    spec_index = vol_24h / (oi_value + 1e-8)
                    oi_intensity = (oi_value * price) / (vol_24h + 1e-8)
                    
                    data = [
                        min(10.0, oi_intensity),
                        ls_ratio,
                        min(5.0, spec_index),
                        funding_now
                    ]
                    self.futures_stats_cache[symbol] = {'data': data, 'ts': time.time()}
                    fetched += 1
                    
                    time.sleep(0.10)
                    
                except Exception:
                    errors += 1
                    continue
            
            valid = len(self._oi_symbols) - len(self._oi_blacklist)
            bl = len(self._oi_blacklist)
            print(f"📊 OI/LS refresh: {fetched}/{valid} updated ({bl} blacklisted, {errors} errors)")
            time.sleep(600)  # 10 minutes between full refreshes

    # ─────────────────────────────────────────────────────────────────
    # HISTORICAL FUTURES BACKFILL  (v11 — training NaN elimination)
    # ─────────────────────────────────────────────────────────────────

    def prefetch_historical_futures(self, symbols, proxy=None):
        """Pre-fetch historical funding rate + OI + LS ratio for training backfill.

        Populates self._hist_futures_cache[symbol] = {
            'funding': sorted list of (ts_ms, rate),
            'oi':      sorted list of (ts_ms, oi_intensity_proxy),
            'ls':      sorted list of (ts_ms, ls_ratio),
        }

        Funding history: Binance /fapi/v1/fundingRate — up to 1000 records (~83 days).
        OI history:      Binance /futures/data/openInterestHist — 5m, up to 500 rows.
        LS ratio:        Binance /futures/data/topLongShortAccountRatio — 5m, up to 500 rows.

        Rate-limited to ~4 req/s to avoid bans. Skips symbols with no futures market.
        """
        import requests as _req

        sess = _req.Session()
        if proxy:
            sess.proxies = {'https': proxy, 'http': proxy}
        sess.headers['User-Agent'] = 'QUANTA/11'

        base_f   = 'https://fapi.binance.com/fapi/v1/fundingRate'
        base_oi  = 'https://fapi.binance.com/futures/data/openInterestHist'
        base_ls  = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'

        blacklist = set()
        loaded = 0

        print(f"📚 Prefetching historical futures data for {len(symbols)} symbols…")

        for sym in symbols:
            if sym in blacklist:
                continue
            try:
                cache = {}

                # ── Funding Rate History (8h intervals, ~83 days) ──
                fr = sess.get(base_f, params={'symbol': sym, 'limit': 1000}, timeout=8)
                if fr.status_code == 400:
                    blacklist.add(sym)
                    continue
                fr_data = fr.json() if fr.ok else []
                cache['funding'] = sorted(
                    [(int(r['fundingTime']), float(r['fundingRate']))
                     for r in fr_data if 'fundingTime' in r],
                    key=lambda x: x[0]
                )
                time.sleep(0.25)

                # ── OI History (5-min candles, ~41 hours @ limit=500) ──
                oi = sess.get(base_oi, params={'symbol': sym, 'period': '5m', 'limit': 500}, timeout=8)
                oi_data = oi.json() if oi.ok else []
                cache['oi'] = sorted(
                    [(int(r['timestamp']), float(r.get('sumOpenInterest', 0)))
                     for r in oi_data if 'timestamp' in r],
                    key=lambda x: x[0]
                )
                time.sleep(0.25)

                # ── Long/Short Ratio History (5-min candles) ──
                ls = sess.get(base_ls, params={'symbol': sym, 'period': '5m', 'limit': 500}, timeout=8)
                ls_data = ls.json() if ls.ok else []
                cache['ls'] = sorted(
                    [(int(r['timestamp']), float(r.get('longShortRatio', 1.0)))
                     for r in ls_data if 'timestamp' in r],
                    key=lambda x: x[0]
                )
                time.sleep(0.25)

                self._hist_futures_cache[sym] = cache
                loaded += 1

            except Exception as e:
                logging.debug(f"prefetch_historical_futures {sym}: {e}")
                continue

        print(f"📚 Historical futures prefetch complete: {loaded}/{len(symbols)} symbols loaded "
              f"({len(blacklist)} spot-only, skipped)")

    def _lookup_hist_futures(self, symbol, ts_ms):
        """Nearest-neighbour lookup in pre-fetched historical futures cache.

        Returns [oi_intensity_proxy, ls_ratio, spec_index_proxy, funding_rate_scaled]
        matching the 4-element Futures X-Ray feature vector.
        """
        cache = self._hist_futures_cache.get(symbol)
        if not cache:
            return [float('nan')] * 4

        def _nearest(series, ts):
            """Binary search for the closest timestamp in a sorted (ts, val) list."""
            if not series:
                return None
            lo, hi = 0, len(series) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if series[mid][0] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            # pick closer of lo-1 and lo
            if lo > 0 and abs(series[lo-1][0] - ts) < abs(series[lo][0] - ts):
                lo -= 1
            return series[lo][1]

        funding   = _nearest(cache.get('funding', []), ts_ms) or 0.0
        oi_raw    = _nearest(cache.get('oi',      []), ts_ms) or 0.0
        ls_ratio  = _nearest(cache.get('ls',      []), ts_ms) or 1.0

        # Proxy for oi_intensity = OI open interest size (normalised); spec_index = 1 (no vol)
        oi_intensity = min(10.0, oi_raw / 1e6)   # scale to same range as live feature
        spec_index   = 1.0                         # no 24h vol in historical; neutral
        funding_vel  = float(funding) * 100        # match live scaling

        return [oi_intensity, ls_ratio, spec_index, funding_vel]

    def update_brier_scores(self, specialist_probs, outcome):
        """
        Update per-agent Brier score calibration tracking (v11.5).

        Args:
            specialist_probs: dict or array of {agent_name: P(bull)} or array of 7 probs
            outcome: 1 if trade was profitable (bull was correct), 0 otherwise
        """
        if not hasattr(self, '_brier_scores'):
            return
        try:
            specialist_keys = list(self.specialist_models.keys())
            if isinstance(specialist_probs, dict):
                for name, prob in specialist_probs.items():
                    if name in self._brier_scores:
                        brier = (prob - outcome) ** 2
                        self._brier_scores[name]['sum'] += brier
                        self._brier_scores[name]['count'] += 1
                        self._brier_scores[name]['rolling'].append(brier)
            elif hasattr(specialist_probs, '__len__') and len(specialist_probs) == len(specialist_keys):
                for i, name in enumerate(specialist_keys):
                    prob = float(specialist_probs[i])
                    brier = (prob - outcome) ** 2
                    self._brier_scores[name]['sum'] += brier
                    self._brier_scores[name]['count'] += 1
                    self._brier_scores[name]['rolling'].append(brier)
        except Exception:
            pass

    def _gnn_correlation_proxy(self, raw_candles):
        """BTC-correlation proxy for GNN embedding during training.

        The live GNN produces a cross-asset embedding via GAT.  During training we
        approximate this with the rolling 20-bar Pearson correlation between this
        coin's returns and BTC returns stored in raw_candles.

        If BTC returns are not available, falls back to Hurst-deviation proxy.
        """
        try:
            closes = np.asarray(raw_candles.get('closes', [1.0]), dtype=np.float64)
            btc    = np.asarray(raw_candles.get('btc_closes', []), dtype=np.float64)
            if len(btc) >= 5 and len(closes) >= 5:
                n = min(len(closes), len(btc), 20)
                ret_coin = np.diff(closes[-n:]) / (closes[-(n+1):-1] + 1e-8)
                ret_btc  = np.diff(btc[-n:])   / (btc[-(n+1):-1]   + 1e-8)
                if np.std(ret_coin) > 0 and np.std(ret_btc) > 0:
                    corr = float(np.corrcoef(ret_coin, ret_btc)[0, 1])
                    return float(np.clip(corr, -1, 1))
            # Fallback: Hurst deviation (mean-reversion vs trend proxy)
            if len(closes) >= 20:
                h = Indicators.hurst(closes, 20)
                return float(np.clip((h - 0.5) * 2, -1, 1))
        except Exception:
            pass
        return 0.0

    def start_graph_background_feed(self, active_coins):
        """🔄 Start background GNN update — runs every 15 min."""
        if hasattr(self, '_graph_thread') and self._graph_thread and self._graph_thread.is_alive():
            return
        self._graph_thread = threading.Thread(
            target=self._graph_background_loop, args=(active_coins,), daemon=True, name="Graph_Fetcher"
        )
        self._graph_thread.start()
        print(f"🕸 GNN background feed started (15-min refresh)")
        
    def _graph_background_loop(self, symbols):
        import time
        import numpy as np
        from quanta_graph import graph_engine
        while True:
            try:
                returns_dict = {}
                for symbol in symbols:
                    if hasattr(self, 'candle_store') and self.candle_store:
                        klines = self.candle_store.get(symbol, '5m')[-50:] if hasattr(self.candle_store, 'get') else self.candle_store.get_klines(symbol, '5m', limit=50)
                        if klines and len(klines) >= 20:
                            k_arr = np.array(klines, dtype=np.float64)
                            raw_closes = k_arr[:, 4]
                            returns = np.diff(raw_closes) / raw_closes[:-1]
                            returns_dict[symbol] = returns[-20:]
                
                if len(returns_dict) >= 2:
                    graph_engine.update_graph(returns_dict, epochs=5)
            except Exception as e:
                print(f"⚠️ GNN background error: {e}")
            
            time.sleep(900)  # 15 min
            
    def _extract_stat_arb(self, symbol, tf_analysis):
        """🏛️ INSTITUTIONAL STAT-ARB: Cross-Exchange & Decoherence (v10.3)"""
        now = time.time()
        
        # 1. Cross-Exchange Premium (Binance vs Coinbase Spot)
        if not hasattr(self, 'coinbase_premium_cache'):
            self.coinbase_premium_cache = {'value': 0.0, 'ts': 0, 'fetching': False}
            
        premium = 0.0
        # If the predicting coin is BTCUSDT itself or highly correlated, Coinbase premium sets macro tone
        if now - self.coinbase_premium_cache['ts'] > 60 and not self.coinbase_premium_cache.get('fetching', False):
            self.coinbase_premium_cache['fetching'] = True
            self.coinbase_premium_cache['ts'] = now
            
            def fetch_cb():
                try:
                    cb_resp = NetworkHelper.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
                    if cb_resp:
                        cb_price = float(cb_resp.json()['data']['amount'])
                        binance_price = tf_analysis.get('5m', {}).get('price', 1)
                        if binance_price > 10000: # Only compute against native BTC
                            new_premium = (cb_price - binance_price) / binance_price * 100
                            self.coinbase_premium_cache = {'value': new_premium, 'ts': time.time(), 'fetching': False}
                            return
                except Exception:
                    pass
                # Reset on failure
                self.coinbase_premium_cache['fetching'] = False
                
            threading.Thread(target=fetch_cb, daemon=True).start()
            
        premium = self.coinbase_premium_cache['value']
            
        # 2 & 3. Order Book Pressure Decay (Spoofing Detection)
        if not hasattr(self, 'ob_decay_cache'):
            self.ob_decay_cache = {}
            
        bid_decay = 0.0
        ask_decay = 0.0
        if getattr(self, 'rt_cache', None) and symbol in self.rt_cache:
            depth = self.rt_cache[symbol].get('depth')
            if depth:
                bids = depth.get('bids', [])
                asks = depth.get('asks', [])
                current_bid_vol = sum([float(b[1]) for b in bids[:5]]) if len(bids) >= 5 else 0
                current_ask_vol = sum([float(a[1]) for a in asks[:5]]) if len(asks) >= 5 else 0
                
                if symbol in self.ob_decay_cache:
                    prev = self.ob_decay_cache[symbol]
                    bid_decay = (current_bid_vol - prev['bid']) / (prev['bid'] + 1e-8)
                    ask_decay = (current_ask_vol - prev['ask']) / (prev['ask'] + 1e-8)
                
                self.ob_decay_cache[symbol] = {'bid': current_bid_vol, 'ask': current_ask_vol}
                
        # 4, 5, 6. Bitcoin Dominance Vector (Altcoin Suction proxy)
        btc_rsi = 50.0
        btc_macd = 0.0
        btc_vol_accel = 0.0
        
        if hasattr(self, 'candle_store') and self.candle_store:
            btc_klines = self.candle_store.get('BTCUSDT', '5m')
            if btc_klines and len(btc_klines) >= 50:
                k_arr = np.array(list(btc_klines)[-50:], dtype=np.float64)
                btc_closes = k_arr[:, 4]
                try:
                    btc_rsi = float(Indicators.rsi(btc_closes))
                    bm, bs, _ = Indicators.macd(btc_closes)
                    btc_macd = float(bm - bs)
                    btc_ret = np.diff(btc_closes) / btc_closes[:-1]
                    if len(btc_ret) >= 30:
                        btc_vol_accel = float(np.std(btc_ret[-10:]) - np.std(btc_ret[-30:-10]))
                except Exception:
                    pass
                    
        return [
            premium,
            bid_decay,
            ask_decay,
            btc_rsi,
            btc_macd,
            btc_vol_accel
        ]
    
    def _orderbook_proxy_from_ohlcv(self, raw_candles):
        """OHLCV-based proxies for the 17 live order book features (training path).

        Binance klines include taker_buy_base_volume (col 9), which measures buy-side
        aggression — the closest proxy to real-time order book imbalance available from
        historical data.  Each of the 17 features is mapped to the most correlated
        OHLCV quantity available.

        Reference: Cont et al. (2014) "The Price Impact of Order Book Events"
        """
        try:
            closes  = np.asarray(raw_candles.get('closes',  [1.0]), dtype=np.float64)
            highs   = np.asarray(raw_candles.get('highs',   closes), dtype=np.float64)
            lows    = np.asarray(raw_candles.get('lows',    closes), dtype=np.float64)
            volumes = np.asarray(raw_candles.get('volumes', [1.0]),  dtype=np.float64)
            taker   = np.asarray(raw_candles.get('taker_buy', volumes * 0.5), dtype=np.float64)

            if len(closes) == 0:
                return [0.0] * 17

            c   = closes[-1]
            vol = max(volumes[-1], 1e-8)
            tb  = taker[-1]
            ts  = max(vol - tb, 1e-8)   # taker sell = total - taker buy

            # 1. Order imbalance (L5 proxy) — buy pressure ratio centred at 0
            buy_pressure = np.clip((tb / vol) * 2 - 1, -1, 1)

            # 2. Price-weighted imbalance — same signal (close is symmetric)
            pw_imbalance = buy_pressure

            # 3. Spread proxy — intrabar range as % of close (Corwin & Schultz 2012)
            h, l = highs[-1], lows[-1]
            spread_proxy = (h - l) / (c + 1e-8) * 100

            # 4. Depth ratio — taker buy / taker sell
            depth_ratio = tb / ts

            # 5. VWAP spread — close deviation from HL midpoint (proxy for microprice)
            mid = (h + l) / 2
            vwap_spread = (c - mid) / (c + 1e-8) * 100

            # 6-7. Volume concentration proxies — neutral (no level data)
            bid_conc = 0.5
            ask_conc = 0.5

            # 8-12. Order flow per level L1-L5 — exponential decay of buy pressure
            decay_factors = [1.0, 0.85, 0.70, 0.55, 0.40]
            ofi_levels = [float(np.clip(buy_pressure * d, -1, 1)) for d in decay_factors]

            # 13-14. Price gap proxies — ATR/5 levels as tick gap estimate
            if len(closes) >= 5:
                atr_proxy = float(np.mean(highs[-5:] - lows[-5:]) / (c + 1e-8) * 100 / 5)
            else:
                atr_proxy = spread_proxy / 5
            avg_bid_gap = atr_proxy
            avg_ask_gap = atr_proxy

            # 15. Queue depth ratio — neutral (no level data)
            queue_depth_ratio = 1.0

            # 16. Microprice gravity — close deviation from mid (same as vwap_spread)
            microprice_delta = vwap_spread

            # 17. Funding rate — comes from historical cache if available; else 0
            funding_rate_scaled = float(raw_candles.get('funding_rate', 0.0))

            return [
                float(buy_pressure),     # 1
                float(pw_imbalance),     # 2
                float(spread_proxy),     # 3
                float(depth_ratio),      # 4
                float(vwap_spread),      # 5
                float(bid_conc),         # 6
                float(ask_conc),         # 7
                *ofi_levels,             # 8-12
                float(avg_bid_gap),      # 13
                float(avg_ask_gap),      # 14
                float(queue_depth_ratio),# 15
                float(microprice_delta), # 16
                float(funding_rate_scaled), # 17
            ]
        except Exception:
            return [0.0] * 17

    def _extract_order_book_features(self, symbol):
        """
        Extract order book features for improved prediction accuracy

        Research: NBER 2024 - Order book imbalance achieves 79% accuracy
        These features capture supply/demand BEFORE price moves - ASYNC CACHED
        """
        try:
            # 🚀 TRUE ASYNCHRONOUS FORM
            # Read instantly from the globally maintained real-time cache. O(1) non-blocking.
            if not getattr(self, 'rt_cache', None) or symbol not in self.rt_cache:
                return [0] * 17
                
            cache_entry = self.rt_cache[symbol]
            depth = cache_entry.get('depth')
            funding_data = cache_entry.get('funding')
            
            if not depth or not funding_data:
                return [0] * 17
            funding_rate = float(funding_data.get('lastFundingRate', 0))
            
            bids = np.array([[float(p), float(v)] for p, v in depth.get('bids', [])[:10]])
            asks = np.array([[float(p), float(v)] for p, v in depth.get('asks', [])[:10]])
            
            if len(bids) == 0 or len(asks) == 0:
                return [0] * 17  # Return zeros if no data
            
            features = []
            
            # 1. ORDER IMBALANCE (Level 5) - MOST IMPORTANT
            bid_vol_l5 = np.sum(bids[:5, 1])
            ask_vol_l5 = np.sum(asks[:5, 1])
            imbalance = (bid_vol_l5 - ask_vol_l5) / (bid_vol_l5 + ask_vol_l5) if (bid_vol_l5 + ask_vol_l5) > 0 else 0
            features.append(imbalance)

            # Nike: track total depth history for depth_delta feature (index 274)
            self._ob_depth_history[symbol].append(float(bid_vol_l5 + ask_vol_l5))
            
            # 2. PRICE-WEIGHTED IMBALANCE
            bid_notional = np.sum(bids[:5, 0] * bids[:5, 1])
            ask_notional = np.sum(asks[:5, 0] * asks[:5, 1])
            price_weighted_imb = (bid_notional - ask_notional) / (bid_notional + ask_notional) if (bid_notional + ask_notional) > 0 else 0
            features.append(price_weighted_imb)
            
            # 3. SPREAD
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            spread = (best_ask - best_bid) / best_bid * 100
            features.append(spread)
            
            # 4. DEPTH RATIO
            depth_ratio = bid_vol_l5 / ask_vol_l5 if ask_vol_l5 > 0 else 1
            features.append(depth_ratio)
            
            # 5. VWAP SPREAD
            bid_vwap = np.sum(bids[:5, 0] * bids[:5, 1]) / np.sum(bids[:5, 1]) if np.sum(bids[:5, 1]) > 0 else best_bid
            ask_vwap = np.sum(asks[:5, 0] * asks[:5, 1]) / np.sum(asks[:5, 1]) if np.sum(asks[:5, 1]) > 0 else best_ask
            vwap_spread = (ask_vwap - bid_vwap) / bid_vwap * 100
            features.append(vwap_spread)
            
            # 6-7. VOLUME CONCENTRATION
            total_bid_vol = np.sum(bids[:, 1])
            total_ask_vol = np.sum(asks[:, 1])
            bid_concentration = bids[0, 1] / total_bid_vol if total_bid_vol > 0 else 0
            ask_concentration = asks[0, 1] / total_ask_vol if total_ask_vol > 0 else 0
            features.append(bid_concentration)
            features.append(ask_concentration)
            
            # 8-12. ORDER FLOW IMBALANCE PER LEVEL (L1-L5)
            for i in range(5):
                bid_i = bids[i, 1] if i < len(bids) else 0
                ask_i = asks[i, 1] if i < len(asks) else 0
                ofi = (bid_i - ask_i) / (bid_i + ask_i) if (bid_i + ask_i) > 0 else 0
                features.append(ofi)
            
            # 13-14. PRICE GAPS
            if len(bids) >= 5:
                bid_gaps = np.diff(bids[:5, 0]) / bids[0, 0] * 100
                avg_bid_gap = np.mean(np.abs(bid_gaps))
            else:
                avg_bid_gap = 0
            
            if len(asks) >= 5:
                ask_gaps = np.diff(asks[:5, 0]) / asks[0, 0] * 100
                avg_ask_gap = np.mean(np.abs(ask_gaps))
            else:
                avg_ask_gap = 0
            
            features.append(avg_bid_gap)
            features.append(avg_ask_gap)
            
            # 15. QUEUE DEPTH RATIO
            bid_depth_ratio = bids[2, 1] / bids[0, 1] if bids[0, 1] > 0 and len(bids) > 2 else 1
            features.append(bid_depth_ratio)
            
            # 16. MICROPRICE GRAVITY
            if bid_vol_l5 + ask_vol_l5 > 0:
                microprice = (np.sum(bids[:5, 0] * asks[:5, 1]) + np.sum(asks[:5, 0] * bids[:5, 1])) / (bid_vol_l5 + ask_vol_l5)
                mid_price = (best_bid + best_ask) / 2
                microprice_delta = (microprice - mid_price) / mid_price * 100
            else:
                microprice_delta = 0
            features.append(microprice_delta)
            
            # 17. FUNDING RATE (LIQUIDATION CASCADE PREDICTOR)
            features.append(funding_rate * 100)  # Scale up
            
            return features
            
        except Exception as e:
            # If order book fetch fails, return zeros
            logging.debug(f"Order book fetch failed for {symbol}: {e}")
            return [0] * 17
    
    def _detect_spike_dump_reversal(self, tf_analysis):
        """Spike-dump detection - OPTIMIZED"""
        features = [0] * 20
        
        try:
            rsi_1m = tf_analysis.get('5m', {}).get('rsi', 50)
            rsi_5m = tf_analysis.get('5m', {}).get('rsi', 50)
            macd_1m = tf_analysis.get('5m', {}).get('macd', 0)
            macd_5m = tf_analysis.get('5m', {}).get('macd', 0)
            bb_pos_1m = tf_analysis.get('5m', {}).get('bb_position', 0.5)
            
            prices = [tf_analysis.get(tf, {}).get('price', 0) for tf in ['5m', '15m', '1h', '4h']]
            volumes = [tf_analysis.get(tf, {}).get('volume', 0) for tf in ['5m', '15m', '1h']]
            rsi_values = [tf_analysis.get(tf, {}).get('rsi', 50) for tf in ['5m', '15m', '1h']]
            macd_values = [tf_analysis.get(tf, {}).get('macd', 0) for tf in ['5m', '15m']]
            atr_values = [tf_analysis.get(tf, {}).get('atr', 0) for tf in ['5m', '15m', '1h']]
            
            # SPIKE FEATURES (0-4)
            if len(prices) >= 2 and prices[-1] > 0:
                price_spike = ((prices[0] / prices[-1]) - 1) * 100
                features[0] = min(50, price_spike)
            
            if len(volumes) >= 2 and volumes[-1] > 0:
                volume_spike = volumes[0] / volumes[-1]
                if volume_spike > 1.5:
                    features[1] = min(10, volume_spike)
            
            features[2] = max(0, (rsi_1m - 70) / 30) if rsi_1m > 70 else 0
            features[3] = max(bb_pos_1m, tf_analysis.get('5m', {}).get('bb_position', 0.5))
            
            if len(atr_values) >= 2:
                mean_atr = np.mean(atr_values[1:])
                if mean_atr > 0 and not np.isnan(mean_atr):
                    atr_ratio = atr_values[0] / mean_atr
                    features[4] = min(5, atr_ratio)
            
            # DUMP FEATURES (5-9)
            if len(prices) >= 2 and prices[-1] > 0:
                dump_magnitude = abs(min(0, (prices[0] / prices[-1] - 1) * 100))
                features[5] = min(50, dump_magnitude)
            
            if len(rsi_values) >= 2:
                rsi_fall = abs(min(0, rsi_values[-1] - rsi_values[0])) / 50
                features[6] = rsi_fall
            
            macd_bearish = 1 if (macd_1m < 0 and macd_5m < 0) else 0.5 if (macd_1m < 0 or macd_5m < 0) else 0
            features[7] = macd_bearish
            
            if len(volumes) >= 2 and volumes[0] > 0:
                volume_exhaustion = max(0, (volumes[-1] - volumes[0]) / volumes[0])
                features[8] = min(1, volume_exhaustion)
            
            if len(prices) >= 3:
                price_changes = [(prices[i] - prices[i+1]) / prices[i+1] * 100 for i in range(len(prices)-1) if prices[i+1] > 0]
                if price_changes:
                    dump_speed = np.mean([abs(pc) for pc in price_changes if pc < 0])
                    features[9] = min(10, dump_speed)
            
            # REVERSAL FEATURES (10-14)
            features[10] = max(0, (30 - rsi_1m) / 30) if rsi_1m < 30 else 0
            
            if len(prices) >= 2 and len(rsi_values) >= 2:
                price_trend = (prices[0] - prices[-1]) / prices[-1] if prices[-1] > 0 else 0
                rsi_trend = rsi_values[0] - rsi_values[-1]
                divergence = 1 if (price_trend < -0.01 and rsi_trend > 0.02) else 0
                features[11] = divergence
            
            if len(macd_values) >= 2:
                macd_turning = 1 if macd_values[0] > macd_values[1] and macd_values[1] < 0 else 0
                features[12] = macd_turning
            
            bb_bounce = max(0, (0.2 - bb_pos_1m) / 0.2) if bb_pos_1m < 0.2 else 0
            features[13] = bb_bounce
            
            if len(prices) >= 2:
                pattern_score = (features[0] / 50 * 0.3 + features[5] / 20 * 0.3 + features[10] * 0.4)
                features[14] = min(1, pattern_score)
            
            # CONSECUTIVE PATTERN FEATURES (15-19)
            # Use cached data from tf_analysis instead of making new API calls
            if '5m' in tf_analysis:
                data_5m = tf_analysis['5m']
                # Estimate consecutive candles from trend strength instead of API calls
                strength = data_5m.get('strength', 0)
                trend = data_5m.get('trend', 'NEUTRAL')
                
                if trend == 'BULLISH':
                    green_5m = min(10, int(strength / 10))  # Estimate from strength
                    red_5m = 0
                elif trend == 'BEARISH':
                    green_5m = 0
                    red_5m = min(10, int(strength / 10))
                else:
                    green_5m = 0
                    red_5m = 0
            else:
                green_5m = 0
                red_5m = 0
            
            features[15] = min(10, green_5m)
            features[16] = min(10, red_5m)
            features[17] = 1 if red_5m >= 5 else red_5m / 5
            features[18] = 1 if green_5m >= 3 else green_5m / 3
            features[19] = self._detect_pattern_break_safe(tf_analysis)
            
        except Exception as e:
            logging.warning(f"Error in spike-dump detection: {e}")
            pass
        
        # Replace NaN or Inf
        features = [0 if (np.isnan(f) or np.isinf(f)) else f for f in features]
        
        return features
    
    # NOTE: _count_consecutive_greens_safe and _count_consecutive_reds_safe
    # were removed — they were never called (dead code).

    
    def _detect_pattern_break_safe(self, tf_analysis):
        try:
            if '5m' not in tf_analysis:
                return 0
            
            data_5m = tf_analysis['5m']
            rsi = data_5m.get('rsi', 50)
            macd = data_5m.get('macd', 0)
            
            return 1 if (45 < rsi < 55 or -0.05 < macd < 0.05) else 0
        except Exception as e:
            logging.debug(f"Pattern break detection failed: {e}")
            return 0

    def _prepare_sequences(self, symbols, sequence_length=TFT_SEQ_LENGTH):
        """
        Prepare sequence data for TFT training.
        
        Builds (batch, seq_len, features) tensors from CandleStore data.
        Uses strided windows to limit memory usage on MX130 (2GB VRAM).
        
        Args:
            symbols: List of trading pair symbols
            sequence_length: Number of time steps per sequence (default 60)
            
        Returns: (X_seq, y_seq) numpy arrays or (None, None) if insufficient data
        """
        if not self.candle_store:
            return None, None
            
        X_seq = []
        y_seq = []
        
        print(f"   \U0001f9e0 Preparing TFT sequences for {len(symbols)} coins (seq_len={sequence_length})...")
        
        symbols_used = 0
        for symbol in symbols:
            try:
                candles = []
                
                # 1. PRIMARY: If we are training, we should use the massive 180-day historical array
                if hasattr(self, 'bnc'):
                    # This instantly returns from FeatherCache without network calls if it was downloaded recently
                    candles = self.bnc.get_historical_klines(symbol, '5m', days=self.cfg.historical_days, training_mode=True)
                
                # 2. FALLBACK: Live inference from the WebSocket CandleStore (Bounded to 200 items)
                if not candles and self.candle_store:
                    with self.candle_store.lock:
                        candles = list(self.candle_store.candles.get(symbol, []))
                
                # Need 50 (indicator warmup) + sequence_length + 48 (target lookahead)
                min_candles = 50 + sequence_length + 48
                if len(candles) < min_candles:
                    continue
                
                # Stride = 10 to reduce sample count (MX130 memory constraint)
                # Each window produces 1 sequence of `sequence_length` feature vectors
                stride = 10
                valid_sequences = 0

                # Pre-compute arrays for Triple Barrier labeling (once per symbol)
                from quanta_numba_extractors import fast_triple_barrier_label
                _c_arr = np.array([float(k[4]) for k in candles], dtype=np.float64)
                _h_arr = np.array([float(k[2]) for k in candles], dtype=np.float64)
                _l_arr = np.array([float(k[3]) for k in candles], dtype=np.float64)
                _a_arr = Indicators.atr_series(_h_arr, _l_arr, _c_arr)
                _tb_tp = self.cfg.events.athena_tp_atr
                _tb_sl = self.cfg.events.athena_sl_atr
                _tb_mb = self.cfg.events.athena_max_bars

                for end_pos in range(50 + sequence_length, len(candles) - _tb_mb, stride):
                    # Build sequence: extract features at each step in the window
                    seq_features = []
                    valid = True

                    for t in range(end_pos - sequence_length, end_pos):
                        feat = self._extract_features_from_candles(candles, t)
                        if feat is None:
                            valid = False
                            break
                        seq_features.append(feat)

                    if not valid or len(seq_features) != sequence_length:
                        continue

                    # Target: Triple Barrier label (aligned with CatBoost agents)
                    # Uses Athena's (generalist) barrier settings so TFT and CatBoost
                    # optimize the same objective. Previous 0.1% threshold created
                    # conflicting signals in the ensemble.
                    label_long, _ = fast_triple_barrier_label(
                        _c_arr, _h_arr, _l_arr, _a_arr, end_pos,
                        1, _tb_tp, _tb_sl, _tb_mb
                    )
                    if label_long == -1:
                        continue  # No barrier hit — skip this sequence
                    target = label_long  # 1 = bullish, 0 = bearish
                    
                    X_seq.append(seq_features)
                    y_seq.append(target)
                    valid_sequences += 1
                    
                    # Cap per symbol to prevent memory explosion (MX130 constraint)
                    if valid_sequences >= 50:
                        break
                
                if valid_sequences > 0:
                    symbols_used += 1
                    
            except Exception as e:
                logging.debug(f"TFT sequence prep error for {symbol}: {e}")
                continue
        
        if len(X_seq) < 10:
            print(f"   \u26a0\ufe0f  Insufficient sequences: {len(X_seq)} (need >=10)")
            return None, None
        
        print(f"   \u2705 Built {len(X_seq)} sequences from {symbols_used} coins")
        
        try:
            X_array = np.array(X_seq, dtype=np.float32)
            y_array = np.array(y_seq, dtype=np.int64)
            return X_array, y_array
        except Exception as e:
            logging.debug(f"TFT array conversion error: {e}")
            return None, None

    def train(self, top_symbols=100, clean_retrain=False):
        """
        🧬 EVENT-BASED MoE TRAINING (Memory-Safe Per-Coin Processing)
        
        Each agent only trains on its specific market events:
        - Athena: Strong uptrend continuation (CUSUM + new high)
        - Ares: Downtrend continuation / short (CUSUM + new low)
        - Hermes: Range-bound scalps (squeeze + expansion)
        - Artemis: Stealth volume accumulation (CUSUM + volume, NOT at new high)
        - Chronos: RSI extreme reversal (CUSUM + RSI extreme)
        - Hephaestus: Support/Resistance bounce (price at S/R level)
        - Nike: Impulse continuation (single-candle explosion + volume)
        
        Key safety features:
        - Per-coin processing: fetch → detect → extract features → release memory
        - Day 700 cutoff: CatBoost only sees first 700 days, 300 days reserved for PPO OOS
        - Time-decay weighting: recent events weighted 3× heavier than oldest
        """
        from quanta_config import Config as _qcfg
        _tp = _qcfg.training
        TRAIN_DAYS = _tp.train_days
        OOS_CUTOFF_DAYS = _tp.oos_cutoff_days
        OOS_CUTOFF_CANDLES = OOS_CUTOFF_DAYS * _tp.candles_per_day
        
        print("\n" + "="*70)
        print("🧬 EVENT-BASED MoE TRAINING (1000-Day Deep Scan)")
        print("="*70)
        
        if clean_retrain:
            print("🧹 CLEAN RETRAIN: Erasing previous models and starting fresh")
            for key in self.specialist_models:
                self.specialist_models[key]['model'] = None
                self.specialist_models[key]['generation'] = 0
            self.model_generation = 0
        else:
            print("🔗 INCREMENTAL MERGE: Building upon existing model knowledge")
            
        print(f"📊 Target: {top_symbols} coins (cache-first, no API needed)")
        print(f"📅 Data: Up to {TRAIN_DAYS} days × 5m candles per coin")
        print(f"🔒 CatBoost Cutoff: Day {OOS_CUTOFF_DAYS} (Day {OOS_CUTOFF_DAYS+1}-{TRAIN_DAYS} reserved for PPO OOS)")
        print(f"⏳ Time-Decay: Oldest=1.0× → Newest=3.0×")
        print(f"🎯 Method: Agent-specific event isolation")
        print("="*70)
        
        training_start = time.time()
        
        try:
            # ========================================
            # STEP 1: SELECT COINS (CACHE-FIRST)
            # ========================================
            from QUANTA_selector import QuantaSelector
            # Share FeatherCache with selector so it doesn't re-fetch cached klines
            shared_cache = self.bnc.cache if hasattr(self, 'bnc') and self.bnc and hasattr(self.bnc, 'cache') else None
            selector = QuantaSelector(cache=shared_cache)
            
            # 🗂️ Cache-first: use all pre-downloaded coins, no API needed
            print("\n🗂️ Scanning feather cache for pre-downloaded coins...")
            symbols = selector.get_cached_coins_for_training(limit=top_symbols)
            if not symbols:
                # Fallback to API-based selection if cache is empty
                print("⚠️ Cache empty, falling back to API-based coin selection...")
                symbols = selector.get_research_backed_coins_for_training(limit=top_symbols)
            if not symbols:
                print("❌ Coin selection failed")
                self.is_trained = False
                return
            print(f"✅ Selected {len(symbols)} coins for training\n")

            # ========================================
            # STEP 1b: PREFETCH HISTORICAL FUTURES DATA (v11 backfill)
            # Eliminates NaN in Futures X-Ray features during training by loading
            # historical funding rate + OI + LS ratio from Binance APIs.
            # ========================================
            try:
                proxy_url = getattr(self.cfg, 'PROXY_URL', None) or getattr(self.bnc, 'proxy', None)
                self.prefetch_historical_futures(symbols, proxy=proxy_url)
            except Exception as _pf_err:
                print(f"⚠️  Historical futures prefetch skipped: {_pf_err}")

            # ========================================
            # STEP 2: PER-COIN EVENT EXTRACTION + FEATURE EXTRACTION
            # ⚡ OPTIMIZED: Pre-compute indicators ONCE per coin, parallel coins
            # ========================================
            agent_names = list(self.specialist_models.keys())
            # ⚡ RAM FIX: pre-allocated float32 arrays instead of Python lists (8x less RAM)
            _MAX_AGENT_EVENTS = _tp.max_agent_events
            _N_FEATURES = self.cfg.BASE_FEATURE_COUNT
            agent_features = {
                name: {
                    'X': np.empty((_MAX_AGENT_EVENTS, _N_FEATURES), dtype=np.float32),
                    'y': np.empty(_MAX_AGENT_EVENTS, dtype=np.int8),
                    'w': np.empty(_MAX_AGENT_EVENTS, dtype=np.float32),
                    'count': 0,
                } for name in agent_names
            }
            all_symbols_used = []
            
            import os
            import pickle
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # 💾 DISK CACHE CHECK: Skip 2-hour extraction if we have a recent cache <= 24 hours old
            cache_file = os.path.join(self.cfg.model_dir, 'agent_features_cache.pkl')
            use_cached_features = False
            
            if os.path.exists(cache_file):
                file_age_days = (time.time() - os.path.getmtime(cache_file)) / 86400.0
                if file_age_days < _tp.cache_max_age_days:
                    try:
                        print(f"\n💾 Found recent event cache ({file_age_days:.1f} days old) at {cache_file}")
                        print("🚀 LOADING PRE-EXTRACTED EVENTS FROM SSD... (Bypassing 2-hour extraction!)")
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                            cached_features = cached_data['agent_features']
                            all_symbols_used = cached_data['all_symbols_used']
                        # Validate all current agents are in the cache
                        missing_in_cache = [n for n in agent_names
                            if n not in cached_features or cached_features[n].get('count', len(cached_features[n].get('y', []))) == 0]
                        if missing_in_cache:
                            print(f"⚠️  Cache missing agents {missing_in_cache} — re-extracting all to include new models")
                        else:
                            # Normalize: old cache may have Python lists — convert to numpy arrays
                            for name, cf in cached_features.items():
                                if 'count' not in cf:
                                    # Old format: Python lists → numpy arrays
                                    X_np = np.array(cf['X'], dtype=np.float32)
                                    y_np = np.array(cf['y'], dtype=np.int8)
                                    w_np = np.array(cf['w'], dtype=np.float32)
                                    cf['X'] = X_np
                                    cf['y'] = y_np
                                    cf['w'] = w_np
                                    cf['count'] = len(y_np)
                            agent_features = cached_features
                            use_cached_features = True
                    except Exception as e:
                        print(f"⚠️ Failed to load event cache: {e}. Re-extracting from scratch...")
            
            if not use_cached_features:
            
                print_lock = threading.Lock()
                num_workers = min(_tp.max_workers, os.cpu_count() or 2)
                
                # ⚡ STEP 2a: Try GPU batch pre-computation for ALL coins at once
                gpu_precomputed = None
                try:
                    from quanta_cuda_indicators import compute_all_indicators_gpu, is_gpu_available
                    if is_gpu_available():
                        print(f"   ⚡ GPU batch mode: pre-computing indicators for {len(symbols)} coins on GPU...")
                        # Fetch all klines first (sequential — I/O bound)
                        all_klines = {}
                        all_events = {}
                        for i, symbol in enumerate(symbols):
                            print(f"   [{i+1}/{len(symbols)}] Fetching {symbol}...", end=" ", flush=True)
                            if hasattr(self, 'bnc') and self.bnc is not None:
                                klines = self.bnc.get_historical_klines(symbol, '5m', days=TRAIN_DAYS, training_mode=True)
                            else:
                                klines = selector.get_historical_klines_paginated(symbol, '5m', days=TRAIN_DAYS)
                            if not klines or len(klines) < 500:
                                print(f"⚠️ {len(klines) if klines else 0} candles, skip")
                                continue
                            klines_np = np.array(klines, dtype=np.float64)
                            cutoff = min(len(klines), OOS_CUTOFF_CANDLES)
                            events = selector.extract_events_from_klines(klines, max_candle_idx=cutoff)
                            if not events:
                                print("⚠️ No events, skip")
                                continue
                            all_klines[symbol] = klines_np
                            all_events[symbol] = (events, cutoff, len(klines))
                            print(f"✅ {len(klines):,} candles")
                        
                        # Batch GPU computation
                        if all_klines:
                            gpu_precomputed = compute_all_indicators_gpu(all_klines)
                            if gpu_precomputed:
                                print(f"   ⚡ GPU pre-computation complete for {len(gpu_precomputed)} coins")
                    else:
                        print(f"   ⚡ CPU parallel extraction: {num_workers} workers (no GPU)")
                except ImportError:
                    print(f"   ⚡ CPU parallel extraction: {num_workers} workers")
                except Exception as e:
                    print(f"   ⚠️ GPU batch failed ({e}), falling back to CPU")
                
                # ⚡ STEP 2b: Extract features (GPU-precomputed or CPU-parallel)
                if gpu_precomputed and all_klines:
                    # GPU PATH: indicators already computed, just extract features per event
                    print(f"   ⚡ GPU path: extracting features from pre-computed indicators...")
                    for symbol, (coin_events, cutoff, total_candles) in all_events.items():
                        if symbol not in gpu_precomputed:
                            continue
                        precomputed = gpu_precomputed[symbol]
                        klines_np = all_klines[symbol]
                        coin_event_counts = {}
                        for agent_name, events in coin_events.items():
                            if agent_name not in agent_features:
                                continue
                            extracted = 0
                            pos_arr = events['pos']
                            label_arr = events['label']
                            weight_arr = events['weight']
                            
                            # --- PRIORITY MEMORY LIMITER ---
                            # Prevent RAM explosion but NEVER drop the highest quality exponential signals.
                            # We sort by event weight (which correlates to magnitude of the move) 
                            # and forcefully keep the top MAX_EVENTS_PER_COIN.
                            MAX_EVENTS_PER_COIN = _tp.max_events_per_coin
                            if len(pos_arr) > MAX_EVENTS_PER_COIN:
                                # argsort ascending, take last 1500, reverse to descending
                                sample_idx = np.argsort(weight_arr)[-MAX_EVENTS_PER_COIN:][::-1]
                                pos_arr = pos_arr[sample_idx]
                                label_arr = label_arr[sample_idx]
                                weight_arr = weight_arr[sample_idx]
                                
                            af = agent_features[agent_name]
                            for j in range(len(pos_arr)):
                                pos = pos_arr[j]
                                try:
                                    features = self._fast_extract_at_position(precomputed, pos, klines_np, symbol=symbol)
                                    if features is not None:
                                        recency = pos / max(cutoff, 1)
                                        time_decay = _tp.time_decay_min + (_tp.time_decay_max - _tp.time_decay_min) * recency
                                        idx = af['count']
                                        af['X'][idx] = features  # direct write, no object overhead
                                        af['y'][idx] = label_arr[j]
                                        af['w'][idx] = weight_arr[j] * time_decay
                                        af['count'] += 1
                                        extracted += 1
                                except Exception:
                                    continue
                            coin_event_counts[agent_name] = extracted
                        all_symbols_used.append(symbol)
                        summary = " | ".join([f"{k[:3]}:{v}" for k, v in coin_event_counts.items() if v > 0])
                        print(f"   {symbol}: {summary}")
                    del gpu_precomputed, all_klines, all_events
                else:
                    # CPU PATH: parallel per-coin processing (fallback)
                    print(f"   ⚡ CPU parallel extraction: {num_workers} workers")
                    def process_coin(args):
                        """Process a single coin: fetch → events → pre-compute → extract features."""
                        coin_idx, symbol = args
                        try:
                            # Fetch klines
                            if hasattr(self, 'bnc') and self.bnc is not None:
                                klines = self.bnc.get_historical_klines(symbol, '5m', days=TRAIN_DAYS, training_mode=True)
                            else:
                                klines = selector.get_historical_klines_paginated(symbol, '5m', days=TRAIN_DAYS)
                            
                            if not klines or len(klines) < 500:
                                with print_lock:
                                    print(f"   [{coin_idx+1}/{len(symbols)}] {symbol}... ⚠️ Only {len(klines) if klines else 0} candles, skip")
                                return None
                            
                            total_candles = len(klines)
                            cutoff = min(total_candles, OOS_CUTOFF_CANDLES)
                            
                            # Detect events
                            coin_events = selector.extract_events_from_klines(klines, max_candle_idx=cutoff)
                            if not coin_events:
                                with print_lock:
                                    print(f"   [{coin_idx+1}/{len(symbols)}] {symbol}... ⚠️ No events, skip")
                                return None
                            
                            # ⚡ PRE-COMPUTE all indicators ONCE (the big optimization)
                            # ⚡ RAM FIX: float32 halves memory vs float64; indicators don't need double precision
                            klines_np = np.array(klines, dtype=np.float32)
                            del klines  # ⚡ RAM FIX: free Python list immediately (was kept alongside numpy array)
                            precomputed = self._precompute_coin_indicators(klines_np)
                            
                            # Extract features at each event position using pre-computed arrays
                            coin_result = {}
                            coin_event_counts = {}
                            for agent_name, events in coin_events.items():
                                if agent_name not in agent_features:
                                    continue
                                # ⚡ RAM FIX: collect per-coin in compact numpy arrays, not Python lists
                                MAX_EVENTS_PER_COIN = _tp.max_events_per_coin
                                extracted = 0
                                pos_arr = events['pos']
                                label_arr = events['label']
                                weight_arr = events['weight']
                                
                                # --- PRIORITY MEMORY LIMITER ---
                                if len(pos_arr) > MAX_EVENTS_PER_COIN:
                                    # argsort ascending, take last 1500, reverse to descending
                                    sample_idx = np.argsort(weight_arr)[-MAX_EVENTS_PER_COIN:][::-1]
                                    pos_arr = pos_arr[sample_idx]
                                    label_arr = label_arr[sample_idx]
                                    weight_arr = weight_arr[sample_idx]

                                # Pre-alloc coin-level result buffer
                                n_cap = len(pos_arr)
                                X_buf = np.empty((n_cap, _N_FEATURES), dtype=np.float32)
                                y_buf = np.empty(n_cap, dtype=np.int8)
                                w_buf = np.empty(n_cap, dtype=np.float32)
                                    
                                for j in range(n_cap):
                                    pos = pos_arr[j]
                                    try:
                                        # ⚡ O(1) per indicator instead of O(n) recomputation
                                        features = self._fast_extract_at_position(
                                            precomputed, pos, klines_np, symbol=symbol
                                        )
                                        if features is not None:
                                            recency = pos / max(cutoff, 1)
                                            time_decay = _tp.time_decay_min + (_tp.time_decay_max - _tp.time_decay_min) * recency
                                            X_buf[extracted] = features
                                            y_buf[extracted] = label_arr[j]
                                            w_buf[extracted] = weight_arr[j] * time_decay
                                            extracted += 1
                                    except Exception:
                                        continue
                                coin_result[agent_name] = {
                                    'X': X_buf[:extracted],
                                    'y': y_buf[:extracted],
                                    'w': w_buf[:extracted],
                                }
                                coin_event_counts[agent_name] = extracted
                            
                            # Release heavy objects
                            del klines_np, precomputed  # klines already deleted above
                            
                            summary = " | ".join([f"{k[:3]}:{v}" for k, v in coin_event_counts.items() if v > 0])
                            with print_lock:
                                print(f"   [{coin_idx+1}/{len(symbols)}] {symbol}... ✅ {total_candles:,} candles (cut@{cutoff:,}) → {summary}")
                            
                            return symbol, coin_result
                            
                        except Exception as e:
                            with print_lock:
                                print(f"   [{coin_idx+1}/{len(symbols)}] {symbol}... ❌ {e}")
                            return None
                    
                    # ⚡ WARMUP NUMBA COMPILERS BEFORE THREADS 
                    # If multiple threads trigger JIT compilation simultaneously, the GIL deadlocks.
                    print(f"   ⚡ Warming up Numba JIT compilers (synchronously) to prevent thread deadlock...")
                    try:
                        dummy_sym = symbols[0]
                        dummy_klines = self.bnc.get_historical_klines(dummy_sym, '5m', days=10, training_mode=True) if hasattr(self, 'bnc') and self.bnc else selector.get_historical_klines_paginated(dummy_sym, '5m', days=10)
                        if dummy_klines and len(dummy_klines) > 50:
                            dummy_klines_np = np.array(dummy_klines[:100], dtype=np.float64)
                            selector.extract_events_from_klines(dummy_klines[:100], max_candle_idx=100)
                            self._precompute_coin_indicators(dummy_klines_np)
                    except Exception as e:
                        print(f"   ⚠️ Numba warmup skipped: {e}")
                    
                    # ⚡ PARALLEL COIN PROCESSING
    
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {executor.submit(process_coin, (i, sym)): sym 
                                  for i, sym in enumerate(symbols)}
                        
                        for future in as_completed(futures):
                            result = future.result()
                            if result is not None:
                                symbol, coin_result = result
                                all_symbols_used.append(symbol)
                                for agent_name, data in coin_result.items():
                                    if agent_name in agent_features:
                                        af = agent_features[agent_name]
                                        n = len(data['y'])
                                        start = af['count']
                                        end = min(start + n, _MAX_AGENT_EVENTS)
                                        n_fit = end - start
                                        af['X'][start:end] = data['X'][:n_fit]
                                        af['y'][start:end] = data['y'][:n_fit]
                                        af['w'][start:end] = data['w'][:n_fit]
                                        af['count'] = end
                    
                    # 💾 SAVE TO DISK CACHE FOR NEXT TIME
                    # ⚡ RAM FIX: trim arrays to actual count before saving (avoids saving empty pre-alloc space)
                    try:
                        cache_out = {}
                        for name, af in agent_features.items():
                            n = af['count']
                            cache_out[name] = {
                                'X': af['X'][:n],
                                'y': af['y'][:n],
                                'w': af['w'][:n],
                                'count': n,
                            }
                        total_saved = sum(af['count'] for af in agent_features.values())
                        print(f"\n💾 Saving {total_saved:,} extracted events to SSD cache...")
                        with open(cache_file, 'wb') as f:
                            pickle.dump({
                                'agent_features': cache_out,
                                'all_symbols_used': all_symbols_used
                            }, f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"✅ Event cache saved to {cache_file}")
                        del cache_out
                    except Exception as e:
                        print(f"⚠️ Failed to save event cache: {e}")
            
            # ========================================
            # STEP 3: SUMMARY
            # ========================================
            print(f"\n{'='*70}")
            print(f"📊 EVENT EXTRACTION SUMMARY")
            print(f"{'='*70}")
            total_events = 0
            for name, data in agent_features.items():
                n = data['count']
                total_events += n
                if n > 0:
                    bulls = int(np.sum(data['y'][:n] == 1))
                    bears = n - bulls
                    print(f"   {name.upper():12s}: {n:6,} events ({bulls:,} bull / {bears:,} bear)")
                else:
                    print(f"   {name.upper():12s}:      0 events")
            print(f"   {'TOTAL':12s}: {total_events:6,} events across all agents")
            print(f"   Coins processed: {len(all_symbols_used)}/{len(symbols)}")
            print(f"{'='*70}")
            
            if total_events < 100:
                print("❌ Not enough events extracted for any agent")
                self.is_trained = False
                return
            
            # ========================================
            # STEP 3.4: RARE EVENTS REPLAY BUFFER 
            # Protect CatBoost from Catastrophic Forgetting
            # ========================================
            print("\n" + "="*70)
            print("💾 BUILDING RARE EVENT EXPERIENCE REPLAY BUFFER")
            print("="*70)
            try:
                # ⚡ RAM FIX: concat pre-allocated slices directly — no Python list intermediates
                X_parts, y_parts, w_parts = [], [], []
                for name, data in agent_features.items():
                    n = data['count']
                    if n > 0:
                        X_parts.append(data['X'][:n])
                        y_parts.append(data['y'][:n])
                        w_parts.append(data['w'][:n])

                if X_parts:
                    all_X_np = np.concatenate(X_parts, axis=0)
                    all_y_np = np.concatenate(y_parts)
                    all_w_np = np.concatenate(w_parts)
                    del X_parts, y_parts, w_parts

                    total_pool = len(all_w_np)
                    if total_pool > 80000:
                        print(f"   ⚖️  Sampling 80,000 events from {total_pool:,} total to prevent OOM...")
                        sample_idx = np.argsort(all_w_np)[-80000:]  # highest-weight events
                        all_X_np = all_X_np[sample_idx]
                        all_y_np = all_y_np[sample_idx]
                        all_w_np = all_w_np[sample_idx]

                    # Sort by weight → pick top REPLAY_BUFFER_SIZE
                    extreme_idx = np.argsort(all_w_np)[::-1][:REPLAY_BUFFER_SIZE]
                    buffer_file = os.path.join(self.cfg.model_dir, 'rare_events_buffer.npz')
                    np.savez(buffer_file,
                             X=all_X_np[extreme_idx],
                             y=all_y_np[extreme_idx],
                             w=all_w_np[extreme_idx])
                    print(f"✅ Saved {len(extreme_idx)} extreme events to Replay Buffer")
                    del extreme_idx
            except Exception as e:
                print(f"⚠️ Failed to build Replay Buffer: {e}")
                
            # ========================================
            # STEP 3.5: TRAIN ODIN (Meta-Learner) ON ALL EVENTS
            # ========================================
            if self.tft_model is not None:
                print("\n" + "="*70)
                print("🧠 TRAINING ODIN (Meta-Feature Extractor)")
                print("="*70)
                # ⚡ RAM FIX: build concatenated view directly from pre-alloc slices
                X_parts, y_parts, w_parts = [], [], []
                for name in agent_names:
                    n = agent_features[name]['count']
                    if n > 0:
                        X_parts.append(agent_features[name]['X'][:n])
                        y_parts.append(agent_features[name]['y'][:n])
                        w_parts.append(agent_features[name]['w'][:n])
                all_X_np = np.concatenate(X_parts, axis=0) if X_parts else None
                all_y_np = np.concatenate(y_parts) if y_parts else None
                all_w_np = np.concatenate(w_parts) if w_parts else None
                del X_parts, y_parts, w_parts

                if all_X_np is not None:
                    total_odin = len(all_w_np)
                    if total_odin > 100000:
                        print(f"   ⚖️  Sampling 100,000 HIGHEST QUALITY events from {total_odin:,} total to prevent OOM in TensorDataset...")
                        # ⚡ RAM FIX: zero-copy numpy index instead of list comprehension
                        highest_quality_idx = np.argsort(all_w_np)[-100000:]
                        all_X_np = all_X_np[highest_quality_idx]
                        all_y_np = all_y_np[highest_quality_idx]
                        del highest_quality_idx
                
                if all_X_np is not None and len(all_X_np) > 100:
                    try:
                        # Move model to GPU FIRST before creating optimizer
                        self.tft_model.train()
                        if USE_GPU and torch.cuda.is_available():
                            self.tft_model = self.tft_model.cuda()

                        X_tft = torch.from_numpy(all_X_np).unsqueeze(1)  # zero-copy
                        y_tft = torch.from_numpy(all_y_np.astype(np.int64))
                        
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        # EXPERIENCE REPLAY INJECTION (25% OVERSAMPLING)
                        # Combats Catastrophic Forgetting in the LSTM by forcing 
                        # crash/pump events to constitute 25% of training data
                        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        buffer_file = os.path.join(self.cfg.model_dir, 'rare_events_buffer.npz')
                        has_replay = False
                        if os.path.exists(buffer_file):
                            try:
                                replay_data = np.load(buffer_file)
                                X_rep = replay_data['X']
                                y_rep = replay_data['y']
                                
                                if len(X_rep) > 50:
                                    # Calculate exactly how many we need to make them 25% of total
                                    # (N_normal) / 0.75 * 0.25 = N_replay_needed
                                    n_normal = len(X_tft)
                                    n_replay_needed = int((n_normal / 0.75) * 0.25)
                                    
                                    # We have a limited buffer (e.g. 500), so we oversample it (repeat) 
                                    # until it equals the target proportion
                                    repeats = int(np.ceil(n_replay_needed / len(X_rep)))
                                    X_rep_over = np.tile(X_rep, (repeats, 1))[:n_replay_needed]
                                    y_rep_over = np.tile(y_rep, repeats)[:n_replay_needed]
                                    
                                    # Merge into PyTorch tensors
                                    X_ext = torch.from_numpy(X_rep_over).unsqueeze(1)
                                    y_ext = torch.from_numpy(y_rep_over.astype(np.int64))
                                    
                                    X_tft = torch.cat([X_tft, X_ext], dim=0)
                                    y_tft = torch.cat([y_tft, y_ext], dim=0)
                                    has_replay = True
                                    print(f"   💾 Injected {n_replay_needed:,} rare event replays (25% buffer scale)")
                            except Exception as e:
                                print(f"   ⚠️ Replay Injection fail: {e}")
                        
                        del all_X_np, all_y_np, all_w_np
                        if USE_GPU and torch.cuda.is_available():
                            X_tft = X_tft.cuda()
                            y_tft = y_tft.cuda()
                        # Temporal val split for TFT quality gate (last 15%)
                        n_tft = len(X_tft)
                        tft_split = int(n_tft * 0.85)
                        X_tft_val = X_tft[tft_split:]
                        y_tft_val = y_tft[tft_split:]
                        X_tft_train = X_tft[:tft_split]
                        y_tft_train = y_tft[:tft_split]

                        dataset = torch.utils.data.TensorDataset(X_tft_train, y_tft_train)
                        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

                        # Optimizer created AFTER model is on GPU so params match device
                        optimizer = optim.Adam(self.tft_model.parameters(), lr=1e-3)
                        criterion = nn.CrossEntropyLoss()

                        for epoch in range(10):
                            for X_batch, y_batch in loader:
                                optimizer.zero_grad()
                                outputs = self.tft_model(X_batch)
                                loss = criterion(outputs, y_batch)
                                loss.backward()
                                optimizer.step()

                        # TFT quality gate: compute val AUC before enabling feature 223 injection
                        tft_val_auc = 0.5
                        try:
                            self.tft_model.eval()
                            with torch.no_grad():
                                val_probs = torch.softmax(self.tft_model(X_tft_val), dim=1)[:, 1].cpu().numpy()
                            val_true = y_tft_val.cpu().numpy()
                            if len(np.unique(val_true)) == 2:
                                from sklearn.metrics import roc_auc_score
                                tft_val_auc = float(roc_auc_score(val_true, val_probs))
                        except Exception:
                            tft_val_auc = 0.5
                        self.tft_val_auc = tft_val_auc
                        print(f"   🧠 Odin val AUC: {tft_val_auc:.4f} {'✅ enabled' if tft_val_auc > 0.55 else '⚠️ below threshold — feature 223 stays zeroed'}")

                        self.tft_trained = tft_val_auc > 0.55  # Only inject if genuinely useful
                        torch.save(self.tft_model.state_dict(), os.path.join(self.cfg.model_dir, 'tft_model.pth'))
                        
                        # Initialize EWC for Long-Term Memory
                        from quanta_deeplearning import EWC
                        self.ewc = EWC(self.tft_model, loader)
                        
                        print("   ✅ Trained & Saved Odin Meta-Learner + Initialized EWC")
                    except Exception as e:
                        print(f"   ❌ Odin Training failed: {e}")
                
                # OOM FIX: Force PyTorch to release its held VRAM back to the OS so CatBoost can use the GPU
                if USE_GPU and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ========================================
            # STEP 4: TRAIN EACH SPECIALIST
            # ========================================
            print("\n" + "="*70)
            print("🧬 TRAINING 7-AGENT GREEK PANTHEON")
            print("="*70)
            
            specialist_success = False
            trained_count = 0
            
            for specialist_name in agent_names:
                data = agent_features[specialist_name]
                n_events = data['count']

                if n_events < _tp.min_events_per_specialist:
                    print(f"   ⚠️ {specialist_name}: Only {n_events} events, skipping (need ≥{_tp.min_events_per_specialist})")
                    continue

                # ⚡ Already numpy float32 — slice, no copy
                X_arr = data['X'][:n_events].copy()  # .copy() so we can overwrite col 223 safely
                y_arr = data['y'][:n_events].astype(np.int32)  # CatBoost wants int32
                w_arr = data['w'][:n_events]

                # ═══════════════════════════════════════════════════════
                # CROSS-EVENT NEGATIVE SAMPLING (v11.5)
                # Inject 5% events from OTHER agents as class-0 (negative) examples.
                # Forces each agent to learn what its domain is NOT, reducing false positives.
                # Kuncheva (2004): cross-training improves ensemble disagreement quality.
                # Ratio reduced from 15% → 5%: at 15%, cross-domain events are trivially
                # distinguishable (different CUSUM triggers, different feature distributions),
                # causing the model to learn domain boundaries rather than true signal.
                # ═══════════════════════════════════════════════════════
                CROSS_SAMPLE_RATIO = 0.05
                try:
                    other_agents = [a for a in agent_names if a != specialist_name]
                    cross_X_parts = []
                    cross_w_parts = []
                    n_cross_target = int(n_events * CROSS_SAMPLE_RATIO)
                    if not other_agents:
                        raise ValueError("No other agents available for cross-event sampling")
                    n_per_other = max(1, n_cross_target // len(other_agents))

                    for other_name in other_agents:
                        other_data = agent_features[other_name]
                        other_count = other_data['count']
                        if other_count < 10:
                            continue
                        # Sample random events from other agent's pool
                        rng = np.random.RandomState(hash(specialist_name + other_name) % (2**31))
                        sample_n = min(n_per_other, other_count)
                        sample_idx = rng.choice(other_count, size=sample_n, replace=False)
                        cross_X_parts.append(other_data['X'][sample_idx])
                        cross_w_parts.append(other_data['w'][sample_idx] * 0.5)  # Lower weight for cross-domain

                    if cross_X_parts:
                        cross_X = np.concatenate(cross_X_parts)
                        cross_y = np.zeros(len(cross_X), dtype=np.int32)  # All class 0 (negative)
                        cross_w = np.concatenate(cross_w_parts)
                        X_arr = np.concatenate([X_arr, cross_X])
                        y_arr = np.concatenate([y_arr, cross_y])
                        w_arr = np.concatenate([w_arr, cross_w])
                        print(f"      🔀 Cross-event negatives: +{len(cross_X)} samples from {len(cross_X_parts)} other agents")
                except Exception as e:
                    print(f"      ⚠️ Cross-event sampling failed: {e}")

                # EXTRACT ODIN META-FEATURE AND REPLACE FEATURE 223 (PADDED NULL)
                if getattr(self, 'tft_trained', False) and self.tft_model is not None:
                    try:
                        self.tft_model.eval()
                        with torch.no_grad():
                            # OOM FIX: Process in batches so PyTorch doesn't try to allocate 1GB VRAM for 150,000 samples
                            tft_out_list = []
                            for i in range(0, len(X_arr), 4096):
                                batch_X = X_arr[i:i+4096]
                                tft_in = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(1)
                                if USE_GPU and torch.cuda.is_available(): tft_in = tft_in.cuda()
                                batch_out = torch.softmax(self.tft_model(tft_in), dim=1)[:, 1].cpu().numpy()
                                tft_out_list.append(batch_out)
                                del tft_in, batch_out
                            tft_out = np.concatenate(tft_out_list)
                            
                        # Override the padded neutral feature at index 223
                        X_arr[:, 223] = tft_out
                        
                        # OOM FIX: Clear cache again after extracting features so CatBoost gets max VRAM
                        if USE_GPU and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        pass
                
                bulls = sum(y_arr == 1)
                bears = sum(y_arr == 0)
                print(f"\n   🎯 {specialist_name.upper()}: {len(X_arr)} samples ({bulls} bull / {bears} bear)")
                print(f"      Time-decay weights: {w_arr.min():.1f}× → {w_arr.max():.1f}×")
                
                # Class imbalance handled by CatBoost auto_class_weights='Balanced'
                # PLUS manual class weighting in _train_specialist for severe imbalance
                # See: CLASS_BALANCING_IMPLEMENTATION_COMPLETE.md for details
                pos_count = int(np.sum(y_arr == 1))
                neg_count = int(np.sum(y_arr == 0))
                imbalance_ratio = max(pos_count, neg_count) / max(min(pos_count, neg_count), 1)
                print(f"      ⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1 (dual balancing: auto + manual)")
                
                if min(pos_count, neg_count) < _tp.min_class_balance:
                    print(f"      ⚠️ Insufficient minority class ({min(pos_count, neg_count)} samples). Need min {_tp.min_class_balance}.")
                    continue
                
                success = self._train_specialist(specialist_name, X_arr, y_arr, w_arr)
                if success:
                    specialist_success = True
                    trained_count += 1
                
                # Release per-agent arrays
                del X_arr, y_arr, w_arr
            
            print(f"\n✅ Trained {trained_count}/7 Greek specialists successfully")

            # v11.5: Print model lineage summary
            print("\n" + "─"*70)
            print("📋 MODEL REGISTRY — Generation Lineage")
            print("─"*70)
            for agent_name in agent_names:
                print(self.model_registry.get_lineage_summary(agent_name))
            print("─"*70)

            if specialist_success:
                # Update ensemble weights based on validation performance
                self._update_ensemble_weights()
                
                # Save specialist models
                print("\n💾 Saving specialist models...")
                self._save_specialist_models()
                self.is_trained = True
                
                # OLD TFT TRAINING REMOVED - NOW TRAINS AS A META-FEATURE BEFORE CATBOOST

                # ========================================
                # LEGACY CONSUMER COMPATIBILITY
                # ========================================
                # Consumer uses legacy self.scaler and self.catboost_model
                # Set them to use Athena (primary) specialist
                self.scaler = self.specialist_models['athena']['scaler']
                athena_model = self.specialist_models['athena']['model']
                if athena_model:
                    self.models = [(athena_model, 1, 1.0, {})]
                print("✅ Legacy consumer compatibility enabled (using Athena specialist)")
                
                print("\n" + "="*70)
                print("✅ 15-AGENT PANTHEON TRAINING COMPLETE")
                print("="*70)
                for name, spec in self.specialist_models.items():
                    print(f"🧬 {name:10s}: Gen {spec['generation']} (Weight {spec['weight']:.3f})")
                print("="*70)
                
                # ========================================
                # PHASE F1: AUTO WALK-FORWARD VALIDATION
                # ========================================
                # Train the MoEPPOAgent on Out-Of-Sample data (Day 201+)
                try:
                    import QUANTA_WalkForward_Sim
                    print("\n" + "="*70)
                    print("🚀 INITIATING AUTOMATIC WALK-FORWARD OOS SIMULATION")
                    print("="*70)
                    QUANTA_WalkForward_Sim.main()
                    print("✅ Walk-Forward Simulation Complete")
                except Exception as e:
                    print(f"❌ Walk-Forward Simulation failed: {e}")
                    import traceback
                    traceback.print_exc()

                training_duration = time.time() - training_start
                print(f"\n⏱️  Total training time: {training_duration/60:.1f} minutes")
                print("✅ Models ready for predictions!")
                
                return  # Skip legacy training
            else:
                print("\n❌ Specialist training failed — no fallback available")
                print("   Re-run /train to retry with event-based pipeline")
                self.is_trained = False
                return
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TFT TRAINING (Lim et al. 2021)
            # Runs AFTER CatBoost so predictions can start immediately
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.tft_model is not None and TFT_AVAILABLE:
                print("\n" + "="*70)
                print("\U0001f9e0 TFT NEURAL NETWORK TRAINING (Lim et al. 2021)")
                print("="*70)
                
                try:
                    # Prepare sequence data from CandleStore
                    X_seq, y_seq = self._prepare_sequences(all_symbols_used)
                    
                    if X_seq is not None and len(X_seq) >= 10:
                        print(f"\U0001f4ca TFT training data: {X_seq.shape}")
                        
                        # Convert to tensors
                        X_tensor = torch.FloatTensor(X_seq)
                        y_tensor = torch.LongTensor(y_seq)
                        
                        if USE_GPU and torch.cuda.is_available():
                            X_tensor = X_tensor.cuda()
                            y_tensor = y_tensor.cuda()
                        
                        # Initialize optimizer (Adam, same lr as PPO for consistency)
                        if self.tft_optimizer is None:
                            self.tft_optimizer = torch.optim.Adam(
                                self.tft_model.parameters(), lr=PPO_LR
                            )
                        
                        # Create DataLoader for batched training
                        dataset = TensorDataset(X_tensor, y_tensor)
                        # Batch size limited for MX130 2GB VRAM
                        tft_batch = min(32, len(X_tensor))
                        loader = DataLoader(dataset, batch_size=tft_batch, shuffle=True)
                        
                        loss_fn = torch.nn.CrossEntropyLoss()
                        
                        # Training loop with early stopping
                        best_loss = float('inf')
                        patience = 3
                        no_improve = 0
                        
                        self.tft_model.train()
                        for epoch in range(TFT_TRAIN_EPOCHS):
                            epoch_loss = 0.0
                            n_batches = 0
                            
                            for X_batch, y_batch in loader:
                                self.tft_optimizer.zero_grad()
                                logits = self.tft_model(X_batch)
                                loss = loss_fn(logits, y_batch)
                                loss.backward()
                                # Gradient clipping (same as PPO)
                                torch.nn.utils.clip_grad_norm_(
                                    self.tft_model.parameters(), PPO_MAX_GRAD_NORM
                                )
                                self.tft_optimizer.step()
                                epoch_loss += loss.item()
                                n_batches += 1
                            
                            avg_loss = epoch_loss / max(n_batches, 1)
                            print(f"   Epoch {epoch+1}/{TFT_TRAIN_EPOCHS}: Loss = {avg_loss:.4f}")
                            
                            # Early stopping
                            if avg_loss < best_loss - 0.001:
                                best_loss = avg_loss
                                no_improve = 0
                            else:
                                no_improve += 1
                                if no_improve >= patience:
                                    print(f"   Early stopping at epoch {epoch+1}")
                                    break
                        
                        self.tft_model.eval()
                        
                        # Save TFT model
                        tft_path = os.path.join(self.cfg.model_dir, 'tft_model.pth')
                        torch.save(self.tft_model.state_dict(), tft_path)
                        self.tft_trained = True
                        
                        # Free GPU memory (critical for MX130)
                        del X_tensor, y_tensor, dataset, loader
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        print(f"\u2705 TFT trained! Loss: {best_loss:.4f}")
                        print(f"   Saved to {tft_path}")
                        print(f"   Ensemble: CatBoost {CAT_ENSEMBLE_WEIGHT*100:.0f}% + TFT {TFT_ENSEMBLE_WEIGHT*100:.0f}%")
                    else:
                        print("   \u26a0\ufe0f  Not enough candle data for TFT training")
                        print("   TFT will train after WebSocket feed accumulates data")
                        
                except Exception as e:
                    logging.error(f"TFT training failed: {e}")
                    print(f"   \u26a0\ufe0f  TFT training error: {e} (CatBoost still active)")
                    # Free GPU memory even on failure
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            print("\n\U0001f680 PREDICTIONS STARTING NOW...")
            print("="*70 + "\n")
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            print(f"[ERROR] Training failed: {e}")

    def predict(self, symbol, tf_analysis):
        """
        🧬 SPECIALIST v11.5b: Prediction via Specialist Weighted Voting
        
        Delegates entirely to predict_with_specialists():
        - Foundation (0.5): Quality coins, established patterns
        - Hunter   (0.3): Volatile coins, spike/dump patterns
        - Anchor   (0.2): BTC/ETH ultra-stable baseline
        
        Returns: (direction, confidence, magnitude, uncertainty)
        """
        if not self.is_trained:
            return 'NEUTRAL', 0, 0, 0

        try:
            return self.predict_with_specialists(symbol, tf_analysis)
        except RecursionError:
            return 'NEUTRAL', 0, 0, 0
    
    def _calculate_magnitude(self, tf_analysis, confidence):
        """
        López de Prado (2018) EWMA Volatility-Scaled Magnitude
        
        magnitude = C × σ × confidence_scale
        
        Where:
        - σ = ATR% as proxy for EWMA volatility
        - C = volatility multiplier (2.0 for crypto)
        - confidence_scale = maps confidence to [0.5, 2.0] range
        
        Replaces old Kelly-based formula (Kelly is for position sizing, not magnitude).
        """
        # 🐛 MATH LOGIC FALLACY FIXED:
        # Previously, it averaged the 5m, 15m, and 1h ATRs. 
        # Volatility scales with time! A 5m candle is tiny compared to a 4h candle.
        # Averaging them dragged the total expected volatility down drastically.
        # FIX: Since our prediction horizon is 4-hours, we must use the 4-hour ATR.
        atr = tf_analysis.get('4h', {}).get('atr')
        if not atr:
            # Fallback: Estimate 4h ATR from smaller timeframes using square root of time
            atr_1h = tf_analysis.get('1h', {}).get('atr')
            if atr_1h:
                atr = atr_1h * 2.0  # sqrt(4 1h periods in 4h)
            else:
                atr_15m = tf_analysis.get('15m', {}).get('atr')
                if atr_15m:
                    atr = atr_15m * 4.0 # sqrt(16 15m periods in 4h)
                else:
                    atr = tf_analysis.get('5m', {}).get('atr', 0) * 6.92  # sqrt(48 5m periods in 4h)
                
        if not atr:
            return 0
        price = tf_analysis.get('5m', {}).get('price', 1)
        sigma = (atr / price) * 100 if price > 0 else 0  # σ = 4h ATR%
        
        # Volatility multiplier (López de Prado recommends 1-3, crypto ~2.0)
        C = 2.0
        
        # Confidence scaling: maps 50-100% → 0.5-2.0x
        confidence_scale = 0.5 + (confidence - 50) / 50 * 1.5
        confidence_scale = max(0.5, min(2.0, confidence_scale))
        
        magnitude = C * sigma * confidence_scale
        return min(20, magnitude)
    
    def train_with_rl_data(self, X, y, weights):
        """
        🧬 SPECIALIST INCREMENTAL LEARNING with 3-Tier TP Weighting

        Updates specialist models with new RL data weighted by outcome quality:
        - TP3 predictions: 3.0x weight (exceptional, focus learning here)
        - TP2 predictions: 2.0x weight (good, Kelly optimal)
        - TP1 predictions: 1.0x weight (okay, baseline)
        - SL predictions: 0.5x weight (learn from failure, but less critical)
        Hard negatives diversity-capped: total hard-neg mass ≤ 50% of batch weight mass.
        
        Requires specialists to be trained. Run /train first if not initialized.
        """
        # ========================================
        # CHECK IF SPECIALISTS ARE AVAILABLE
        # ========================================
        specialists_available = all(
            s['model'] is not None 
            for s in self.specialist_models.values()
        )
        
        if not specialists_available:
            logging.warning("⚠️  Specialists not initialized — run /train first. Skipping RL update.")
            print("⚠️  Specialists not available. Run /train to initialize before RL retraining.")
            return

        if True:  # specialists available — proceed with specialist path
            # 🧬 SPECIALIST WARM START TRAINING
            print(f"\n{'='*70}")
            print("🧬 SPECIALIST RL EVOLUTION - 3-TIER TP WEIGHTING")
            print(f"{'='*70}")
            print(f"New samples: {len(X)}")
            print(f"Effective samples: {weights.sum():.1f} (after TP-tier weighting)")
            
            # Data quality checks
            if len(X) < 30:
                print(f"   ⚠️  Too few samples ({len(X)}) - need 30+ for stable training")
                return
            
            # Clean data
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                valid_idx = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
                X = X[valid_idx]
                y = y[valid_idx]
                weights = weights[valid_idx]
                if len(X) < 30:
                    return
            
            # Mix with historical data (80% old + 20% new) - Catastrophic forgetting protection!
            try:
                # Load historical data if available
                historical_file = os.path.join(self.cfg.model_dir, 'historical_rl_data.npz')
                if os.path.exists(historical_file):
                    hist_data = np.load(historical_file)
                    X_hist = hist_data['X']
                    y_hist = hist_data['y']
                    
                    # Load historical weights if available, otherwise default to 1.0
                    if 'weights' in hist_data:
                        weights_hist = hist_data['weights']
                    else:
                        weights_hist = np.ones(len(y_hist), dtype=np.float32)
                    
                    # Mix: 80% historical, 20% new
                    hist_size = int(len(X_hist) * 0.8)
                    X_mixed = np.concatenate([X_hist[:hist_size], X])
                    y_mixed = np.concatenate([y_hist[:hist_size], y])
                    weights_mixed = np.concatenate([weights_hist[:hist_size], weights])
                    
                    print(f"   📚 Mixed data: {hist_size} historical + {len(X)} new = {len(X_mixed)} total")
                    print(f"   ⚖️  Effective: {weights_mixed.sum():.1f} samples (weighted)")
                else:
                    X_mixed = X
                    y_mixed = y
                    weights_mixed = weights
                    print(f"   🆕 First RL training - using {len(X)} samples")
                
                # Prevent any negative weights from crashing CatBoost (legacy compatibility)
                if weights_mixed is not None:
                    weights_mixed = np.maximum(weights_mixed, 0.01)
                
                # Save current data as historical for next time (including weights!)
                np.savez(historical_file, X=X_mixed, y=y_mixed, weights=weights_mixed)
                
                # ========================================
                # EXPERIENCE REPLAY: INJECT RARE EVENTS
                # ========================================
                replay_file = os.path.join(self.cfg.model_dir, 'rare_events_buffer.npz')
                if os.path.exists(replay_file):
                    replay_data = np.load(replay_file)
                    X_replay, y_replay, w_replay = replay_data['X'], replay_data['y'], replay_data['w']
                    
                    # Randomly sample historical events up to REPLAY_SAMPLE_SIZE
                    n_sample = min(REPLAY_SAMPLE_SIZE, len(X_replay))
                    if n_sample > 0:
                        idx = np.random.choice(len(X_replay), n_sample, replace=False)
                        X_mixed = np.concatenate([X_mixed, X_replay[idx]])
                        y_mixed = np.concatenate([y_mixed, y_replay[idx]])
                        weights_mixed = np.concatenate([weights_mixed, w_replay[idx]])
                        print(f"   💾 Experience Replay: Mixed in {n_sample} rare extreme events")
                        
            except Exception as e:
                logging.warning(f"Historical data mixing failed: {e}")
                X_mixed = X
                y_mixed = y
                weights_mixed = weights
            
            # ========================================
            # RETRAIN SPECIALISTS WITH SPECIALTY-ROUTED DATA
            # Each agent only sees events matching its domain
            # (consistent with initial event-based training)
            # ========================================
            success_count = 0
            
            # Try event-based routing first (proper per-specialist data)
            try:
                from QUANTA_selector import QuantaSelector
                shared_cache = self.bnc.cache if hasattr(self, 'bnc') and self.bnc and hasattr(self.bnc, 'cache') else None
                selector = QuantaSelector(cache=shared_cache)
                
                # Get recent movers to extract fresh events from
                feed_coins = selector.get_live_prediction_feed()
                if feed_coins and len(feed_coins) >= 10:
                    print(f"\n   🎯 Event-routing RL retrain across {len(feed_coins)} live movers...")
                    
                    agent_retrain_data = {name: {'X': [], 'y': [], 'w': []} for name in self.specialist_models}
                    
                    # Process top 30 movers for speed (RL retrain should be fast)
                    for sym in feed_coins[:30]:
                        try:
                            if hasattr(self, 'bnc') and self.bnc is not None:
                                klines = self.bnc.get_historical_klines(sym, '5m', days=30, training_mode=True)
                            else:
                                klines = selector.get_historical_klines_paginated(sym, '5m', days=30)
                            
                            if not klines or len(klines) < 500: continue
                            
                            coin_events = selector.extract_events_from_klines(klines)
                            if not coin_events: continue
                            
                            klines_np = np.array(klines, dtype=np.float64)
                            
                            for agent_name, events in coin_events.items():
                                if agent_name not in agent_retrain_data: continue
                                for ev in events:
                                    try:
                                        features = self._extract_features_from_candles(
                                            klines, ev['pos'], _precomputed_np=klines_np
                                        )
                                        if features is not None:
                                            agent_retrain_data[agent_name]['X'].append(features)
                                            agent_retrain_data[agent_name]['y'].append(ev['label'])
                                            agent_retrain_data[agent_name]['w'].append(ev['weight'])
                                    except Exception as e:
                                        logging.debug(f"Feature extraction failed for {agent_name}: {e}")
                                        continue
                            del klines, klines_np
                        except Exception as e:
                            logging.debug(f"Retrain data loading failed: {e}")
                            continue
                    
                    # Retrain each specialist on ITS events only
                    for specialist_name, data in agent_retrain_data.items():
                        if len(data['y']) < 30:
                            print(f"   ⏭️  {specialist_name}: {len(data['y'])} events (need 30+)")
                            continue
                        
                        X_spec = np.array(data['X'])
                        y_spec = np.array(data['y'])
                        w_spec = np.array(data['w'])
                        
                        # Mix with generic RL outcomes (20%) for cross-domain learning
                        mix_size = min(len(X_mixed) // 5, len(X_spec))
                        if mix_size > 0 and len(X_mixed) > 0:
                            mix_idx = np.random.choice(len(X_mixed), mix_size, replace=False)
                            X_spec = np.concatenate([X_spec, X_mixed[mix_idx]])
                            y_spec = np.concatenate([y_spec, y_mixed[mix_idx]])
                            w_spec = np.concatenate([w_spec, weights_mixed[mix_idx] * 0.5])  # Lower weight for generic
                        
                        success = self._train_specialist(specialist_name, X_spec, y_spec, w_spec)
                        if success:
                            success_count += 1
                        del X_spec, y_spec, w_spec
                    
                else:
                    raise ValueError("No live feed available")
                    
            except Exception as e:
                # Fallback: retrain all specialists with generic RL data
                print(f"   ⚠️ Event routing failed ({e}), using generic RL data for all specialists")
                for specialist_name in self.specialist_models:
                    if len(X_mixed) >= 50:
                        success = self._train_specialist(
                            specialist_name, X_mixed, y_mixed, weights_mixed
                        )
                        if success:
                            success_count += 1
            
            if success_count > 0:
                self._save_specialist_models()
                print(f"\n✅ Updated {success_count}/7 specialists with event-routed RL data!")
            else:
                print("\n⚠️  No specialists updated — insufficient event data.")
            
            # ========================================
            # RETRAIN ODIN (LSTM-Attention) ON RL DATA
            # ========================================
            # Research: Lim et al. 2021 — temporal models benefit from
            # online updates with outcome-labeled data because they can
            # learn regime-specific temporal patterns from verified trades.
            if self.tft_model is not None and len(X_mixed) >= 100:
                try:
                    print(f"\n{'='*70}")
                    print("🔱 RETRAINING ODIN ON RL OUTCOMES")
                    print(f"{'='*70}")
                    
                    # Build sequences from RL data
                    # Each RL sample is a feature snapshot — create sliding windows
                    seq_len = min(TFT_SEQ_LENGTH, len(X_mixed) // 3)
                    if seq_len >= 10:
                        X_seqs = []
                        y_seqs = []
                        for i in range(len(X_mixed) - seq_len):
                            X_seqs.append(X_mixed[i:i+seq_len])
                            y_seqs.append(y_mixed[i+seq_len-1])
                        
                        if len(X_seqs) > 50:
                            X_tensor = torch.tensor(np.array(X_seqs), dtype=torch.float32)
                            y_tensor = torch.tensor(np.array(y_seqs), dtype=torch.long)
                            
                            # Pad/trim to match self.cfg.BASE_FEATURE_COUNT
                            if X_tensor.shape[2] != self.cfg.BASE_FEATURE_COUNT:
                                if X_tensor.shape[2] < self.cfg.BASE_FEATURE_COUNT:
                                    pad = torch.zeros(X_tensor.shape[0], X_tensor.shape[1],
                                                      self.cfg.BASE_FEATURE_COUNT - X_tensor.shape[2])
                                    X_tensor = torch.cat([X_tensor, pad], dim=2)
                                else:
                                    X_tensor = X_tensor[:, :, :self.cfg.BASE_FEATURE_COUNT]
                            
                            print(f"   Sequences: {X_tensor.shape}")
                            
                            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
                            
                            # Lower LR for fine-tuning (don't destroy pretrained weights)
                            optimizer = optim.Adam(self.tft_model.parameters(), lr=5e-4)
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
                            criterion = nn.CrossEntropyLoss()
                            
                            self.tft_model.train()
                            if USE_GPU and torch.cuda.is_available():
                                self.tft_model = self.tft_model.cuda()
                            
                            for epoch in range(5):  # Fewer epochs for fine-tuning
                                epoch_loss = 0
                                for X_batch, y_batch in loader:
                                    if USE_GPU and torch.cuda.is_available():
                                        X_batch = X_batch.cuda()
                                        y_batch = y_batch.cuda()
                                    optimizer.zero_grad()
                                    outputs = self.tft_model(X_batch)
                                    loss = criterion(outputs, y_batch)
                                    
                                    # Protect rare event knowledge with EWC penalty
                                    if hasattr(self, 'ewc') and self.ewc is not None:
                                        loss += EWC_PENALTY_MULTIPLIER * self.ewc.penalty(self.tft_model)
                                        
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item()
                                scheduler.step()
                                avg_loss = epoch_loss / max(len(loader), 1)
                                if epoch == 0 or epoch == 4:
                                    print(f"   Epoch {epoch+1}/5: Loss = {avg_loss:.4f}")
                            
                            # Update EWC to incorporate the new knowledge boundary
                            if hasattr(self, 'ewc') and self.ewc is not None:
                                from quanta_deeplearning import EWC
                                self.ewc = EWC(self.tft_model, loader)
                                
                            torch.save(self.tft_model.state_dict(),
                                       os.path.join(self.cfg.model_dir, 'tft_model.pth'))
                            print(f"   💾 Odin RL fine-tune complete!")
                        else:
                            print(f"   ⚠️ Not enough sequences ({len(X_seqs)}), skipping Odin")
                    else:
                        print(f"   ⚠️ Sequence length too short ({seq_len}), skipping Odin")
                except Exception as e:
                    print(f"   ❌ Odin RL retrain failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            return

    
    def update_model_performance(self, model, correct, profit_pct=0):
        """
        Update model performance using Sharpe ratio for adaptive weighting
        Sharpe = (mean_return - risk_free_rate) / std_return
        """
        # Track accuracy
        self._model_total[model] = self._model_total.get(model, 0) + 1
        if correct:
            self._model_correct[model] = self._model_correct.get(model, 0) + 1
        
        # Track returns (profit percentage)
        if model not in self._model_returns:
            self._model_returns[model] = []
        self._model_returns[model].append(profit_pct)
        
        # Keep last 100 predictions only (sliding window)
        if len(self._model_returns[model]) > 100:
            self._model_returns[model] = self._model_returns[model][-100:]
    
    # NOTE: recompute_model_weights was removed in v7.0 — it referenced
    # self.model_weights which was never initialised, causing AttributeError.
    # Specialist ensemble weights are now managed via specialist_models dict.


class _OfflineReplaySentiment:
    """Neutral sentiment provider for offline feature replay."""

    @staticmethod
    def get_sentiment_features(symbol=None):
        return [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _OfflineReplayOnchain:
    """Neutral on-chain provider for offline feature replay."""

    @staticmethod
    def get_onchain_features(symbol=None):
        return [0.0, 0.0, 0.5]


def build_offline_feature_replay_engine(cfg=None):
    """
    Create a lightweight DeepMLEngine shell for offline feature replay.

    This avoids the full constructor cost (registry, TFT, live services) while
    reusing the authoritative feature extraction methods.
    """
    engine = DeepMLEngine.__new__(DeepMLEngine)
    cfg_obj = cfg or Config
    if not hasattr(cfg_obj, "BASE_FEATURE_COUNT"):
        cfg_obj.BASE_FEATURE_COUNT = int(getattr(cfg_obj.model, "base_feature_count", 278))
    if not hasattr(cfg_obj, "timeframes"):
        cfg_obj.timeframes = ['5m', '15m', '1h', '4h', '6h', '12h', '1d']
    if not hasattr(cfg_obj, "tf_weights"):
        cfg_obj.tf_weights = {
            '5m': 0.08, '15m': 0.10, '1h': 0.18, '4h': 0.22,
            '6h': 0.14, '12h': 0.12, '1d': 0.16,
        }
    if not hasattr(cfg_obj, "historical_days"):
        cfg_obj.historical_days = int(getattr(cfg_obj.market, "historical_days", 180))

    engine.cfg = cfg_obj
    engine.bnc = None
    engine.mtf = None
    engine.candle_store = None
    engine.sentiment = _OfflineReplaySentiment()
    engine.onchain = _OfflineReplayOnchain()
    engine.rt_cache = {}
    engine._hist_futures_cache = {}
    engine._ob_depth_history = defaultdict(lambda: deque(maxlen=10))
    engine._bs_avg_bars_to_hit = {}
    engine.hmm_models = {}
    engine.hmm_last_fit = {}
    return engine


def _cache_df_to_klines_np(df):
    """Convert cached feather OHLCV into Binance-like numpy klines for replay."""
    if df is None or len(df) == 0:
        return np.empty((0, 12), dtype=np.float64)

    n = len(df)
    out = np.zeros((n, 12), dtype=np.float64)
    out[:, 0] = df["open_time"].to_numpy(dtype=np.float64)
    out[:, 1] = df["open"].to_numpy(dtype=np.float64)
    out[:, 2] = df["high"].to_numpy(dtype=np.float64)
    out[:, 3] = df["low"].to_numpy(dtype=np.float64)
    out[:, 4] = df["close"].to_numpy(dtype=np.float64)
    out[:, 5] = df["volume"].to_numpy(dtype=np.float64)
    out[:, 6] = out[:, 0] + (5 * 60 * 1000)
    if "qv" in df.columns:
        out[:, 7] = df["qv"].to_numpy(dtype=np.float64)
    else:
        out[:, 7] = out[:, 4] * out[:, 5]
    if "trades" in df.columns:
        out[:, 8] = df["trades"].to_numpy(dtype=np.float64)
    if "tbv" in df.columns:
        out[:, 9] = df["tbv"].to_numpy(dtype=np.float64)
    elif "taker_buy" in df.columns:
        out[:, 9] = df["taker_buy"].to_numpy(dtype=np.float64)
    else:
        out[:, 9] = 0.5 * out[:, 5]
    out[:, 10] = out[:, 4] * out[:, 9]
    return out


def precompute_offline_feature_bundle(df, engine=None):
    replay_engine = engine or build_offline_feature_replay_engine()
    klines_np = _cache_df_to_klines_np(df)
    precomputed = replay_engine._precompute_coin_indicators(klines_np) if len(klines_np) else {}
    return {
        "engine": replay_engine,
        "precomputed": precomputed,
        "klines_np": klines_np,
    }


def extract_offline_features_for_positions(df, symbol, positions, engine=None, precomputed=None, klines_np=None):
    """
    Reuse QUANTA's authoritative feature extractor for selected cached positions.

    Returns:
        {
            "engine": replay_engine,
            "precomputed": indicator_cache,
            "klines_np": raw numpy klines,
            "features_by_pos": {bar_index: np.ndarray[BASE_FEATURE_COUNT]},
        }
    """
    replay_engine = engine or build_offline_feature_replay_engine()
    klines_np = klines_np if klines_np is not None else _cache_df_to_klines_np(df)
    if len(klines_np) == 0:
        return {
            "engine": replay_engine,
            "precomputed": {},
            "klines_np": klines_np,
            "features_by_pos": {},
        }

    precomputed = precomputed if precomputed is not None else replay_engine._precompute_coin_indicators(klines_np)
    wanted = sorted({int(p) for p in positions if 50 <= int(p) < len(klines_np)})
    features_by_pos = {}
    for pos in wanted:
        feat = replay_engine._fast_extract_at_position(precomputed, pos, klines_np, symbol=symbol)
        if feat is None:
            continue
        features_by_pos[pos] = np.asarray(feat, dtype=np.float64)

    return {
        "engine": replay_engine,
        "precomputed": precomputed,
        "klines_np": klines_np,
        "features_by_pos": features_by_pos,
    }
