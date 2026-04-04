# QUANTA v7.0 — Full Code Documentation
**Created by Habib Khairul**  
*For internal use — reference this before making any adjustments to the code*

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [File Structure](#2-file-structure)
3. [Config](#3-config)
4. [Constants (Research-Backed)](#4-constants)
5. [NetworkHelper](#5-networkhelper)
6. [FreeProxyManager](#6-freeproxymanager)
7. [FeatherCache](#7-feathercache)
8. [BinanceAPIEnhanced](#8-binanceapienhanced)
9. [CandleStore & WebSocket Feed (ws_feed.py)](#9-candlestore--websocket-feed)
10. [MultiTimeframeAnalyzer (MTF)](#10-multitimeframeanalyzer)
11. [Indicators](#11-indicators)
12. [FearGreed](#12-feargreed)
13. [DeepMLEngine](#13-deepmlengine)
14. [Feature Engineering (205 Features)](#14-feature-engineering)
15. [Triple Barrier Method](#15-triple-barrier-method)
16. [RLMemory](#16-rlmemory)
17. [Hard Negative Mining & Evolutionary Learning](#17-hard-negative-mining--evolutionary-learning)
18. [PaperTrading](#18-papertrading)
19. [WSEventProducer (Pipeline Producer)](#19-wseventproducer)
20. [consumer_worker (GPU Prediction)](#20-consumer_worker)
21. [Bot Class — Main Orchestrator](#21-bot-class)
22. [Telegram Commands](#22-telegram-commands)
23. [Startup Flow](#23-startup-flow)
24. [Confidence Tiers & Alert Logic](#24-confidence-tiers--alert-logic)
25. [Adjustable Parameters Cheatsheet](#25-adjustable-parameters-cheatsheet)

---

## 1. Architecture Overview

```
Binance Futures WebSocket
        │  (push, <1s latency, 0 API cost)
        ▼
   CandleStore (RAM)
   dict[symbol][tf] → deque(200 candles)
        │
        │  on every 1m candle CLOSE
        ▼
   WSEventProducer
   SharedThreadPool(8 workers)
   analyze() reads from RAM
        │
        ▼
   data_queue (Queue, maxsize=8000)
        │
        ▼
   consumer_worker  ← GPU thread
   Batch predict (512 at once)
   CatBoost ensemble → confidence %
        │
        ├── >= 70% → Telegram alert + RL queue
        ├── 60-69% → RL queue only (silent learning)
        └── < 60%  → Discarded
        │
        ▼
   RLMemory (10k buffer)
   Check outcomes after 1 hour
   Triple Barrier Method → WIN / LOSS
        │  (500 outcomes reached)
        ▼
   Retrain → Hard Negative Mining
           → New model generation
           → Ensemble updated
```

**Key principle:** Data flows to you via WebSocket (push). You never ask Binance for data during live operation after initial bootstrap. Zero API cost, zero rate limit pressure.

---

## 2. File Structure

```
QUANTA/
├── bot.py              — Main bot, all classes
├── ws_feed.py          — WebSocket feed (CandleStore, BinanceWSFeed, WSEventProducer)
├── feather_cache/      — Disk cache for historical candle data (.feather files)
├── ml_models_pytorch/  — Saved CatBoost model files (.cbm) + scalers (.pkl)
├── quanta_data/        — Persistent prediction storage (triple redundancy .feather)
├── rl_memory.feather   — RL experience replay buffer
└── trades.csv          — Paper trading log
```

---

## 3. Config

**Location:** `class Config` (~line 1200)

All runtime settings in one place. Change here, affects whole bot.

| Parameter | Default | What it does |
|---|---|---|
| `timeframes` | `['1m','5m','15m','1h','4h','1d','1w']` | TFs analyzed per coin |
| `tf_weights` | 1w=0.30, 1d=0.20, 4h=0.15... | Weight of each TF in consensus score |
| `scan_interval` | 90s | How often stats are reported (pipeline runs continuously) |
| `ml_confidence_min` | 65% | Minimum to send Telegram alert |
| `ml_confidence_rl_min` | 60% | Minimum to store in RL buffer |
| `rl_retrain_threshold` | 500 | Completed outcomes before retraining |
| `rl_outcome_check_time` | 3600s (1h) | How long to wait before checking if prediction was right |
| `rl_check_interval` | 1800s (30m) | How often to run the outcome check loop |
| `queue_size` | 8000 | Max items in data_queue |
| `num_producers` | 12 | Number of REST producer threads (fallback only, WS mode ignores this) |
| `gpu_batch_size` | 512 | How many coins GPU predicts at once |
| `cache_enabled` | True | Enable FeatherCache disk storage |
| `historical_days` | 90 | Days of history for initial training |
| `model_dir` | `ml_models_pytorch/` | Where models are saved |

---

## 4. Constants

**Location:** Lines ~297–361 (after imports)

Extracted from magic numbers — all research-cited. Change here to tune behavior globally.

| Constant | Value | Source | Effect |
|---|---|---|---|
| `RSI_PERIOD` | 14 | Wilder 1978 | RSI calculation window |
| `DIRECTION_THRESHOLD` | 0.12 | López de Prado 2018 | Minimum predicted move to count as directional |
| `FOCAL_LOSS_GAMMA` | 2.5 | Lin et al. 2017 | Hard negative mining focus strength |
| `HARD_NEGATIVE_WEIGHT` | 5.0 | Lin et al. 2017 | How much more hard failures are weighted in training |
| `RL_RETRAIN_THRESHOLD` | 500 | Rolnick 2019 | Completed predictions needed to trigger retrain |
| `RL_OUTCOME_WINDOW` | 3600 | — | Seconds after prediction to check outcome |
| `MIN_CONFIDENCE_RL` | 60 | — | Store in RL buffer above this |
| `MIN_CONFIDENCE_ALERT` | 70 | — | Send Telegram alert above this |
| `SL_RATIO` | 0.5 | Industry standard | Stop loss = 50% of predicted magnitude |
| `TP1/2/3_RATIO` | 0.5 / 1.0 / 1.5 | Kelly Criterion | Three take profit tiers |
| `TP1/2/3_WEIGHT` | 1.0 / 2.5 / 5.0 | Sharpe optimization | Training weight given per TP tier hit |
| `CATASTROPHIC_FORGETTING_BUFFER_RATIO` | 0.8 | Kirkpatrick 2017 | 80% old data + 20% new in retraining mix |
| `WEIGHT_FOUNDATION/HUNTER/ANCHOR` | 0.5 / 0.3 / 0.2 | Empirical | Specialist model ensemble weights |
| `HISTORICAL_DAYS` | 90 | López de Prado | Min days for regime coverage in training |

---

## 5. NetworkHelper

**Location:** `class NetworkHelper` (~line 780)

Singleton session manager for all HTTP requests.

**What it does:**
- Creates one persistent `requests.Session` with connection pooling (40 connections)
- Applies retry strategy: 5 retries, exponential backoff (2, 4, 8, 16, 32s)
- Circuit breaker: opens after 20 consecutive failures per endpoint, resets after 15s
- Rate limiter: light sleep every 20 requests to Binance domains
- Proxy injection: if `_PROXY_URL` is set, routes all requests through it

**To adjust:**
- Retry count → `Retry(total=5, ...)`
- Pool size → `pool_maxsize=40`
- Circuit breaker threshold → `if breaker['failures'] >= 20`
- Circuit breaker cooldown → `cooldown_period = 15`

---

## 6. FreeProxyManager

**Location:** `class FreeProxyManager` (~line 1063)

Auto-finds working proxies when direct Binance connection is blocked (e.g. Indonesia).

**Flow:**
1. Fetches proxy lists from 7 public sources (ProxyScrape, GitHub repos)
2. Tests up to 80 proxies concurrently (15s timeout)
3. First working one wins, applied to `NetworkHelper` session
4. At startup you can also enter your own proxy port — this overrides auto-search

**Proxy sources:** ProxyScrape SOCKS5/HTTP, TheSpeedX list, hookzof, monosans  
**Test URL:** `fapi.binance.com/fapi/v1/ping`

**Note:** WebSocket connections (ws_feed.py) handle SOCKS separately via `python-socks[asyncio]`

---

## 7. FeatherCache

**Location:** `class FeatherCache` (~line 449)

Disk-based cache for historical OHLCV data. 2.5× faster than Parquet (Arrow format, LZ4 compression).

**Two layers:**
1. **Memory cache** — Python dict, instant access, capped at 500 pairs
2. **Disk cache** — `.feather` files in `feather_cache/` folder

**Flow for `get_klines()` call:**
```
Request → Memory cache hit? → Return instantly
        → Disk cache hit?   → Load feather, update memory cache, return
        → Cache miss        → REST API call → save to disk + memory → return
```

**Key settings:**
- `MAX_MEMORY_CACHE = 500` — max pairs in memory layer
- Files stored as `{SYMBOL}_{INTERVAL}.feather`
- Tracks: hits, misses, API calls saved, bytes saved

**When it's used:** Only during bootstrap and for `1d`/`1w` refresh in WS mode. During live operation, WS pushes data directly to CandleStore (no disk involved).

---

## 8. BinanceAPIEnhanced

**Location:** `class BinanceAPIEnhanced` (~line 1386)

Wraps all Binance REST API calls with caching, fallback logic, and error handling.

**Key methods:**

| Method | What it does |
|---|---|
| `get_pairs()` | Gets all USDT perpetual pairs from `/exchangeInfo`. 8 attempts, fallback to `/ticker/24hr`, then hardcoded list |
| `get_klines(symbol, interval, limit)` | Gets OHLCV data. Checks memory cache → FeatherCache → REST API |
| `get_klines_from(symbol, interval, start_time, limit)` | Gets klines anchored to a specific timestamp (used by RL outcome checking) |
| `get_top_movers(limit)` | Returns top N coins by 24h volume change from `/ticker/24hr` |
| `get_sniper_coins(limit)` | Momentum-scored selection — best coins for live prediction (see below) |
| `get_research_backed_coins(limit)` | For training data selection — liquidity + regime diversity |
| `validate_symbols(symbols)` | Tests each symbol against klines endpoint, filters invalid ones |
| `warmup_cache_research(symbols, days)` | Bulk-fetches historical data for training |

**Sniper coin selection logic:**
Scores each coin 0–1 on: 1h return (30%), 4h return (20%), volume surge (15%), ADX+direction (20%), RSI (10%), consecutive bullish/bearish candles (5%). Returns top N/2 bullish + top N/2 bearish coins merged. Balanced between directions to avoid directional bias in training.

---

## 9. CandleStore & WebSocket Feed

**Location:** `ws_feed.py`

The new data layer replacing the old 12-producer REST polling system.

### CandleStore
In-memory buffer for all live candle data.

```python
store[symbol][timeframe] → deque(maxlen=200)
```

- `update(symbol, tf, kline)` — called by WS on every tick, replaces in-progress candle or appends new one
- `seed(symbol, tf, klines)` — bulk-loads history at bootstrap
- `get(symbol, tf)` — returns plain list snapshot for analysis (thread-safe copy)
- `is_ready(symbol, tf, min_candles=50)` — True if enough history to compute indicators

### BinanceWSFeed
Manages WebSocket connections to `fstream.binance.com`.

- Subscribes to `{symbol}@kline_{tf}` streams for all (symbol, tf) combinations
- Max 180 streams per connection (conservative, Binance limit is 200)
- Multiple connections auto-spawned if needed (e.g. 75 coins × 5 TFs = 375 streams → 3 connections)
- Auto-reconnects on disconnect (3s delay)
- SOCKS proxy support via `python-socks[asyncio]` — reads `_PROXY_URL` from `bot.py` automatically
- On candle close (`"x": true`), fires `on_candle_close(symbol, tf)` callback

### WSEventProducer
Replaces the 12-producer REST polling loop.

- Listens for `on_candle_close` events from BinanceWSFeed
- Only triggers on `1m` closes (every minute, per coin) — higher TF closes automatically update CandleStore
- Debounce: skips if analysis already in-flight for that symbol
- Throttle: skips if `data_queue` > 80% full
- Submits `analyze()` to **one shared ThreadPoolExecutor(8 workers)** — never recreated
- Puts result item onto `data_queue` for consumer_worker

### Static TF Refresh
`1d` and `1w` candles don't need WebSocket (they barely change). They're:
- Fetched via REST at bootstrap
- Refreshed every 4h (1d) / 24h (1w) by a background thread

---

## 10. MultiTimeframeAnalyzer

**Location:** `class MultiTimeframeAnalyzer` (~line 3260)

Runs technical analysis across all 7 timeframes for a given symbol.

**analyze(symbol) returns:**
```python
{
  '1m': { trend, strength, price, rsi, macd, bb_position, adx, volume, atr,
          volatility, atr_percentile, volatility_accel,
          momentum_5/10/20/50, volume_ratio, mean_shift, trend_strength, returns_period },
  '5m': { ... },
  '15m': { ... },
  '1h': { ... },
  '4h': { ... },
  '1d': { ... },
  '1w': { ... }
}
```

**Trend scoring logic (per TF):**
- RSI > 70 → +20, RSI < 30 → -20, RSI > 50 → +10, else -10
- MACD line > signal AND histogram > 0 → +20 (bearish: -20)
- Price > MA20 > MA50 → +20 (bearish: -20)
- BB position > 0.8 → +10 (< 0.2 → -10)
- Stochastic both > 80 → +10 (both < 20 → -10)
- Score > 30 → BULLISH, < -30 → BEARISH, else NEUTRAL
- Strength = `min(100, abs(score) × (ADX/25))`

**Cache:** 30-second in-memory cache per symbol (keyed by `symbol_time//30`). After WS patch, reads from CandleStore (RAM) instead of calling get_klines().

**WS patch adds:**
- Reads from CandleStore instead of REST
- Uses shared ThreadPoolExecutor (never recreated)
- Falls back to REST for any TF not yet warmed up in CandleStore

---

## 11. Indicators

**Location:** `class Indicators` (~line 2966)

All pure-function technical indicators. Used by MTF analyzer.

| Method | Input | Output |
|---|---|---|
| `rsi(closes, period=14)` | Price list | RSI value 0–100 |
| `macd(closes)` | Price list | (macd_line, signal_line, histogram) |
| `bollinger(closes, period=20, std=2)` | Price list | (upper, middle, lower) |
| `atr(highs, lows, closes, period=14)` | OHLC lists | ATR value |
| `adx(highs, lows, closes, period=14)` | OHLC lists | ADX value |
| `adx_full(highs, lows, closes)` | OHLC lists | (ADX, +DI, -DI) |
| `stochastic(highs, lows, closes, k=14, d=3)` | OHLC lists | (K, D) |

All use `np.seterr(all='ignore')` — numpy warnings silenced globally.

---

## 12. FearGreed

**Location:** `class FearGreed` (~line 2936)

Fetches Crypto Fear & Greed Index from `alternative.me`.

- Cached for 15 minutes (900s)
- Two fallback URLs tried
- Returns `{'value': 0-100, 'label': 'Extreme Fear/Fear/Neutral/Greed/Extreme Greed'}`
- Displayed in Telegram alerts and startup message
- Default fallback: `{'value': 50, 'label': 'Neutral'}`

**To adjust:** Change cache duration at `if (time.time() - self._cached_at) < 900`

---

## 13. DeepMLEngine

**Location:** `class DeepMLEngine` (~line 3422)

The ML brain. CatBoost-only ensemble with 3 specialist models and evolutionary learning.

### Three Specialist Models

| Model | Weight | Trained on | Purpose |
|---|---|---|---|
| `foundation` | 0.50 | Top 50 volume coins, >90 days history | Reliable baseline, highest trust |
| `hunter` | 0.30 | Volatile/small-cap coins | Pump/dump detection |
| `anchor` | 0.20 | BTC, ETH, top 10 only | Conservative safety net |

### Legacy Ensemble
Alongside specialists, maintains a rolling ensemble of up to 5 models:
- New model added with weight 1.0
- Older models decayed by 0.7× each retrain
- Oldest dropped when > 5 models
- Prediction = weighted average across all ensemble models

### CatBoost Settings (per model)
```
iterations=500, learning_rate=0.05, depth=6
loss_function=Logloss, eval_metric=AUC
task_type=GPU (or CPU fallback)
early_stopping_rounds=50
```

### Warm Start (Incremental Learning)
Each retrain uses `init_model=existing_model` — CatBoost continues from where it left off instead of starting fresh. This is how catastrophic forgetting is prevented.

### Prediction Flow
```
features (205-dim) → specialist_scaler.transform() → catboost.predict_proba()
                   → weighted vote across 3 specialists + legacy ensemble
                   → final confidence %  +  direction (BULLISH/BEARISH)
                   → Bayesian uncertainty estimate
```

### Bayesian Uncertainty
Multiple CatBoost models vote. Variance across votes = uncertainty.  
High uncertainty → skip prediction (not sent as alert even if confidence > 70%)

### Magnitude Calculation
Based on ATR of the 1h timeframe. Uncertainty-adjusted: `magnitude × (1 - uncertainty_penalty × 0.5)`. Min threshold: 2.5% move required to send alert.

### Saving
- Models: `ml_models_pytorch/{name}_gen{N}.cbm`
- Scalers: `ml_models_pytorch/{name}_scaler.pkl`
- Generation counter: `ml_models_pytorch/generation.txt`
- Training metadata: `ml_models_pytorch/training_metadata.json`
- Historical data buffer: `ml_models_pytorch/historical_data.pkl`

---

## 14. Feature Engineering

**Location:** `_extract_features()` and `_extract_features_from_1m()` in `DeepMLEngine`

**Total: 205 features per prediction**

| Group | Count | What |
|---|---|---|
| Per-timeframe (7 TFs × 7 features) | 49 | RSI, MACD, BB position, ADX, ATR, trend direction, strength |
| Cross-timeframe consensus | 4 | Bullish %, bearish %, net consensus, weighted score |
| RSI analysis | 5 | Range, std, drift, overbought count, mean |
| MACD analysis | 3 | Range, positive count, mean |
| Volume analysis | 5 | Log max, log mean, log std, relative volume, volume trend |
| ATR analysis | 4 | Max%, mean%, std%, current% |
| ADX analysis | 3 | Max, mean, strong trend count |
| BB analysis | 3 | Width, mean position, squeeze |
| Momentum | 9 | Multi-window returns at 5/10/20/50 periods per TF |
| 1m-based deep features | ~115 | From raw 1m candles: MAs, volatility regimes, wick analysis, spike detection, pattern recognition |

The 1m-based features include:
- 5 moving averages (5/10/20/50/100 MA) and their crossovers
- Volatility: rolling std, ATR percentile, acceleration
- Momentum: 5/10/20/50 period returns
- Volume profile: log-normalized, trends
- Bollinger Band position and width
- RSI and Stochastic
- Wick analysis: upper/lower wick ratios (spike/rejection detection)
- Consecutive bullish/bearish candles
- Gap analysis between candles
- Higher highs / lower lows counting

---

## 15. Triple Barrier Method

**Location:** `RLMemory.check_predictions()` (~line 6183)  
**Source:** López de Prado (2018), Chapter 3

Determines if a prediction was correct by checking whether price hit TP or SL *during* the 1-hour window, not just at the end.

**Barriers for BULLISH prediction:**
```
TP1 = entry × (1 + magnitude × 0.50 / 100)   — 50% of predicted move
TP2 = entry × (1 + magnitude × 1.00 / 100)   — 100% of predicted move
TP3 = entry × (1 + magnitude × 1.50 / 100)   — 150% of predicted move
SL  = entry × (1 - magnitude × 0.50 / 100)   — -50% of predicted move
```

**Outcome tiers and training weights:**
| Outcome | Weight | Meaning |
|---|---|---|
| TP3 hit | 5.0× | Exceptional prediction — learn heavily from this |
| TP2 hit | 2.5× | Full target hit — learn well from this |
| TP1 hit | 1.0× | Partial win — baseline learning |
| SL hit | 0.5× | Loss — still learn, but lighter weight |
| No barrier hit | Timeout | Neutral outcome, weak signal |

**Data fetch for outcome check:** Uses `get_klines_from(symbol, '5m', start_time=entry_time)` — anchored to prediction time, not current time. This ensures correct 1-hour window even if the check runs late.

---

## 16. RLMemory

**Location:** `class RLMemory` (~line 6080)

Reinforcement Learning experience replay buffer.

**Buffer:** `deque(maxlen=10000)` — FIFO, auto-drops oldest when full

**Each stored prediction contains:**
```python
{
  symbol, direction, confidence, magnitude,
  entry_price, entry_time,
  features,          # 205-dim array (raw, pre-scaler)
  outcome,           # None until checked, then: 'TP3'/'TP2'/'TP1'/'SL'/'TIMEOUT'
  actual_move,       # % price moved
  success,           # True/False
  weight             # Training weight based on outcome tier
}
```

**Flow:**
1. Prediction made → stored with `outcome=None`
2. Every 30 minutes: `check_predictions()` runs
3. Predictions older than 1 hour get their outcome determined via Triple Barrier
4. When 500 completed outcomes accumulated → retrain triggered
5. After retrain → buffer cleaned (keep pending + last 24h completed)

**Saved to:** `rl_memory.feather` (LZ4 compressed, fast Feather format)

---

## 17. Hard Negative Mining & Evolutionary Learning

**Location:** `DeepMLEngine.train()` (~line 5600)  
**Source:** Focal Loss (Lin et al. 2017), Hard Negative Mining for Medical Imaging (2024)

The core of why the bot gets smarter over generations, not just bigger.

### Hard Negative Mining
After each retrain cycle:
1. Take all completed predictions where `confidence > 70%` but outcome = LOSS
2. These are "hard negatives" — the model was very confident but wrong
3. Apply `sample_weight = HARD_NEGATIVE_WEIGHT (5.0×)` to these in training
4. CatBoost focuses gradient updates on these failure cases

**Effect:** Model learns to recognize the patterns that previously fooled it at high confidence.

### Training Data Mix (Catastrophic Forgetting Prevention)
```
X_train = 80% historical buffer + 20% new RL data
```
Never train on just new data — always mix with old to preserve existing knowledge.

### Scaler Drift Detection
Before refitting the `StandardScaler`, runs a Kolmogorov-Smirnov test on feature distributions. Only refits if distribution shifted > 30%. Prevents model degradation from inconsistent scaling.

### Generation Tracking
After each retrain:
```python
generation_performance.append({
  'generation': N,
  'val_acc': X.XX,
  'hard_negatives': N,
  'samples': N
})
```
Compares val_acc gen-over-gen. Prints: EVOLVED / STABLE / REGRESSION.

---

## 18. PaperTrading

**Location:** `class PaperTrading` (~line 6022)

Simulated trading to track theoretical performance.

- Starting balance: $10,000
- Position sizing: `risk% = min(5, confidence/20) / 100`
- Stop distance: `ATR × 1.5`
- Auto-closes when: +3% profit OR -2% loss
- Logs every trade to `trades.csv`

**Note:** This runs in parallel with predictions. It doesn't affect the ML or alerts — purely informational tracking.

---

## 19. WSEventProducer

**Location:** `ws_feed.py — class WSEventProducer`

Replaces the old `producer_worker` threads.

**Trigger:** Every 1m candle close for any tracked symbol  
**Analysis:** Calls `mtf.analyze(symbol)` → reads from CandleStore (RAM, no I/O)  
**Pre-features:** Extracts 205 features in the producer thread to offload GPU consumer  
**Queue:** Puts result dict onto `Bot.data_queue` for consumer  

**Debounce:** If analysis for symbol already in-flight → skip (prevents pile-up)  
**Throttle:** If `data_queue > 80% full` → skip (prevents consumer overload)  

**Thread model:**
- BinanceWSFeed: 1 asyncio thread per WS connection (daemon)
- Analysis: 1 shared `ThreadPoolExecutor(max_workers=8)`, lifetime = process lifetime
- Static refresh: 1 background thread, checks every 5 minutes

---

## 20. consumer_worker

**Location:** `Bot.consumer_worker()` (~line 6924)

The GPU prediction thread. Runs continuously, never stops.

**Flow:**
1. Blocks on `data_queue.get(timeout=0.05)` 
2. Collects batch up to `gpu_batch_size=512` items (non-blocking after first)
3. Stacks all pre-extracted feature arrays into matrix
4. `scaler.transform(features_batch)` — normalize
5. `catboost.predict_proba(features_batch)` — one GPU call for entire batch
6. For each result: extract confidence, direction, magnitude, uncertainty
7. `confidence >= ml_confidence_rl_min (60%)` → store in RL queue
8. `confidence >= ml_confidence_min (65%)` → put on `result_queue` for alert
9. Paper trading tick update
10. Every 5000 predictions → push stats to `stats_queue` (async, non-blocking)

**During retraining:** Switches from `data_queue` to `retrain_queue` (filled with all available coins for max learning during the training period)

---

## 21. Bot Class

**Location:** `class Bot` (~line 6521)

Main orchestrator. Owns all components.

### Key Attributes
| Attribute | Type | Purpose |
|---|---|---|
| `cfg` | Config | All settings |
| `bnc` | BinanceAPIEnhanced | REST API access |
| `mtf` | MultiTimeframeAnalyzer | Technical analysis |
| `ml` | DeepMLEngine | ML predictions |
| `rl_memory` | RLMemory | Learning buffer |
| `fg` | FearGreed | Market sentiment |
| `paper` | PaperTrading | Simulation |
| `data_queue` | Queue(8000) | Producer → Consumer |
| `result_queue` | Queue | Consumer → Alerts |
| `retrain_queue` | Queue(20000) | During retraining |
| `stats_queue` | Queue(1000) | GPU → Stats logger (async) |
| `stop_event` | threading.Event | Graceful shutdown |
| `is_retraining` | threading.Event | Retrain mode flag |
| `active_coins` | list | Current prediction targets |
| `rl_opportunities` | list | Predictions waiting for outcome |

### Triple Redundancy Storage
Predictions saved to 3 separate `.feather` files every 100 predictions + every 5 seconds background + on shutdown:
- `quanta_data/predictions_primary.feather`
- `quanta_data/predictions_backup1.feather`
- `quanta_data/predictions_backup2.feather`

### Threads Running Simultaneously
| Thread | Purpose |
|---|---|
| WS Feed (asyncio) | Receives candle data from Binance |
| WSEventProducer pool (8) | Runs MTF analysis on candle closes |
| GPU Consumer (1) | Batch predictions on data_queue |
| Stats Logger (1) | Prints stats async, never blocks GPU |
| RL Timer (1) | Shows countdown to next outcome check |
| Static TF Refresh (1) | Updates 1d/1w candles periodically |
| Telegram Listener (1) | Polls for commands |
| Auto-Save (1) | Saves predictions every 5s |

---

## 22. Telegram Commands

**Location:** `Bot.cmd()` (~line 7875)

| Command | What it does |
|---|---|
| `/start` | Shows available commands |
| `/status` | Shows generation, speed (coins/min), total processed, RL accuracy |
| `/stop` | Saves all data and exits instantly |
| `/rlstats` | Detailed RL buffer stats (total, pending, completed, win rate) |
| `/interval N` | Change scan_interval to N seconds |

---

## 23. Startup Flow

```
1. User enters proxy port (or Enter for direct)
2. Proxy patched into requests.get globally
3. Direct connection test → if fail, proxy applied
4. Startup diagnostic (ping, exchangeInfo, klines test)
5. If models not trained:
   a. get_research_backed_coins(50) — selects training coins
   b. warmup_cache_research(coins, days=90) — fetch 90 days history
   c. _train_models() — initial CatBoost training (5-15 min)
6. Telegram startup message sent
7. _scan_pipeline() called → first run initializes:
   a. get_sniper_coins(75) — selects live prediction coins
   b. validate_symbols() — filters invalid pairs
   c. ws_bootstrap() — fetch 200 candles REST (one-time)
   d. WSEventProducer.start() — WS connections open
   e. consumer_worker thread started
   f. Stats logger thread started
   g. RL timer thread started
8. Main loop: _scan_pipeline() called every 90s (just stats reporting now)
9. RL check every 30 min → if 500 outcomes → retrain
```

---

## 24. Confidence Tiers & Alert Logic

```
Prediction confidence:
  < 60%  → Discarded (below RL threshold)
  60-64% → Stored in RL buffer, no alert, silent learning
  65-69% → Telegram alert + RL buffer
  70%+   → Telegram alert (flagged high confidence) + RL buffer

Additional filters before alert:
  - magnitude < 2.5% → skip (move too small)
  - uncertainty too high → skip (model is guessing)
  - alert_cooldown (1h) per symbol → no repeat alerts

Alert content:
  - Symbol, direction, confidence %, estimated move %
  - Timeframe breakdown (1w/1d/4h/1h trends)
  - Entry price, TP, SL (ATR-based)
  - Current Fear & Greed
  - Model generation
```

---

## 25. Adjustable Parameters Cheatsheet

Quick reference for the most common adjustments.

### More alerts (lower bar)
```python
# Config class:
self.ml_confidence_min = 60      # was 65
self.ml_confidence_rl_min = 55   # was 60

# Constants:
MIN_CONFIDENCE_ALERT = 60        # was 70
```

### Faster retraining
```python
# Constants:
RL_RETRAIN_THRESHOLD = 200       # was 500 (retrain every 200 outcomes instead of 500)
```

### More coins tracked
```python
# In _scan_pipeline():
discovered_coins = self.bnc.get_sniper_coins(limit=100)   # was 75
```

### Bigger training window
```python
# Config class:
self.historical_days = 180       # was 90 (6 months instead of 3)
```

### Change outcome check window
```python
# Constants:
RL_OUTCOME_WINDOW = 1800         # was 3600 (check after 30 min instead of 1h)
```

### Reduce timeframes (faster analysis, less data)
```python
# Config class:
self.timeframes = ['1m', '5m', '15m', '1h', '4h']   # drop 1d and 1w
```

### Stronger hard negative focus
```python
# Constants:
HARD_NEGATIVE_WEIGHT = 10.0      # was 5.0 (double the focus on failures)
```

### Larger GPU batch (if you have more VRAM)
```python
# Config class:
self.gpu_batch_size = 1024       # was 512
```

### WebSocket analysis pool size
```python
# ws_feed.py — WSEventProducer.__init__:
self._pool = ThreadPoolExecutor(max_workers=16)   # was 8
```

### WS streams per connection
```python
# ws_feed.py:
MAX_STREAMS_PER_CONNECTION = 180    # conservative, Binance limit is 200
```

---

*Last updated: February 2026 — QUANTA v7.0 + WS Feed v1.0*
