# QUANTA C/C++ Refactoring Plan

## Goal

Refactor QUANTA trading bot from Python (~30,000 lines across 30+ files) to C/C++ for production-grade latency, memory efficiency, and deployment.

## Current Python Codebase

| File | Lines | Size | Role |
|------|-------|------|------|
| QUANTA_ml_engine.py | ~6000 | 291KB | Feature extraction, training, inference |
| QUANTA_bot.py | 3476 | 182KB | Main loop, consumer, ensemble pipeline |
| quanta_exchange.py | ~2000 | 78KB | Binance REST/WS API |
| QUANTA_trading_core.py | ~1800 | 78KB | Paper trading, position management, Kelly |
| quanta_features.py | ~1000 | 41KB | Technical indicators, BS barrier math |
| QUANTA_selector.py | ~1000 | 39KB | Event extraction routing |
| QUANTA_sentiment.py | ~900 | 33KB | RSS parsing, F&G, Groq sentiment |
| quanta_config.py | ~700 | 29KB | Config dataclasses |
| QUANTA_agents.py | ~800 | 30KB | PPO agent, memory, DSR reward |
| quanta_websockets.py | ~600 | 23KB | WS feed, CandleStore |
| quanta_numba_extractors.py | ~500 | 19KB | Numba JIT event extractors |
| Other ~20 files | ~12000 | ~250KB | Cache, monitor, registry, telegram, etc. |

---

## User Review Required

> [!IMPORTANT]
> **Hybrid vs Full Rewrite**: I recommend a **hybrid approach** — C/C++ core engine with Python kept for Optuna hyperparameter search and SHAP explainability (no C++ equivalents exist). The Python layer becomes a thin training-only tool; the live trading engine is pure C/C++.

> [!IMPORTANT]
> **Build System**: I recommend **CMake** + **vcpkg** for Windows dependency management. Confirm this works for your environment.

> [!WARNING]
> **Timeline**: This is a multi-week project even with aggressive execution. The ML engine alone is 6000 lines of deeply intertwined Python. Expect 4-6 phases spanning 2-4 weeks of focused work.

> [!CAUTION]
> **CatBoost Training**: CatBoost's C++ training API is limited compared to Python. Recommendation: **train in Python, infer in C++**. CatBoost models saved as `.cbm` load natively in C++ via `catboost/model_calcer.h`.

---

## Architecture Decision: C vs C++

| Component | Language | Why |
|-----------|----------|-----|
| Feature extraction (278-dim) | **C** | Pure math, no OOP needed, maximum speed |
| Event extractors (CUSUM, barriers) | **C** | Replaces Numba JIT, same performance profile |
| BS barrier math (Kou, GARCH) | **C** | Numerical functions, no allocation |
| Technical indicators | **C** | Stateless math on arrays |
| Bot architecture | **C++17** | Threads, queues, RAII, smart pointers |
| ML inference pipeline | **C++17** | CatBoost/libtorch C++ APIs require C++ |
| WebSocket feed | **C++17** | boost.beast or libwebsockets |
| Telegram API | **C++17** | libcurl wrapper |
| Config system | **C++17** | Struct with JSON serialization |
| PPO agent | **C++17** | libtorch C++ (or custom) |
| Paper trading | **C++17** | Position management, state persistence |

**Pattern**: C for hot-path math functions (called millions of times), C++ for architecture/orchestration.

---

## Dependency Mapping: Python → C/C++

| Python Library | C/C++ Replacement | Notes |
|----------------|-------------------|-------|
| NumPy | **Eigen 3.4** | Header-only, same semantics |
| Pandas | **Apache Arrow C++** | Feather read/write native |
| CatBoost | **CatBoost C API** (`catboost/model_calcer.h`) | Inference only. Train in Python |
| PyTorch | **libtorch** (C++ distribution) | TFT inference |
| hmmlearn | **Custom HMM** (50 lines of C) | GaussianHMM is simple EM |
| sklearn StandardScaler | **Custom** (10 lines) | `(x - mean) / std` |
| Numba @njit | **Native C functions** | Direct replacement, faster |
| websockets | **boost.beast** or **libwebsockets** | WS client |
| requests | **libcurl** | HTTP client |
| tqdm | Not needed | C++ progress bar trivial |
| threading | **std::thread + std::jthread** | C++17/20 |
| queue.Queue | **std::queue + std::mutex + std::condition_variable** | Thread-safe queue |
| json | **nlohmann/json** | Header-only JSON |
| Optuna | **Keep Python** | No C++ equivalent, training-only |
| SHAP | **Keep Python** | No C++ equivalent, analysis-only |
| psutil | **OS-native API** | Windows: SetProcessAffinityMask |

---

## Project Structure

```
QUANTA_cpp/
├── CMakeLists.txt
├── vcpkg.json                    # dependency manifest
├── include/
│   ├── quanta/
│   │   ├── config.h              # Config structs
│   │   ├── types.h               # Common types, constants
│   │   ├── features.h            # Feature extraction API
│   │   ├── indicators.h          # Technical indicators
│   │   ├── extractors.h          # Event extractors (CUSUM, barriers)
│   │   ├── bs_math.h             # Kou barrier, GARCH, time decay
│   │   ├── ml_engine.h           # ML inference pipeline
│   │   ├── hmm.h                 # GaussianHMM (3-state)
│   │   ├── ensemble.h            # Ensemble weighting, entropy
│   │   ├── ppo_agent.h           # PPO size oracle
│   │   ├── paper_trading.h       # Position management
│   │   ├── risk_manager.h        # Pre-trade checks, circuit breaker
│   │   ├── bot.h                 # Main bot class
│   │   ├── ws_feed.h             # WebSocket candle feed
│   │   ├── exchange.h            # Binance REST API
│   │   ├── telegram.h            # Telegram messaging
│   │   ├── sentiment.h           # Sentiment engine
│   │   ├── cache.h               # Feather/Arrow cache
│   │   ├── monitor.h             # Model monitoring
│   │   └── utils.h               # Logging, thread-safe queue
│   └── vendor/                   # Header-only libs (Eigen, nlohmann)
├── src/
│   ├── core/                     # Pure C hot-path
│   │   ├── features.c            # 278-dim feature extraction
│   │   ├── indicators.c          # RSI, MACD, BB, ATR, ADX, etc.
│   │   ├── extractors.c          # CUSUM, triple barrier, event detection
│   │   ├── bs_math.c             # Kou jump-diffusion, GARCH, time decay
│   │   └── hmm.c                 # GaussianHMM EM + predict
│   ├── engine/                   # C++ architecture
│   │   ├── ml_engine.cpp         # CatBoost inference, domain masks, ensemble
│   │   ├── ensemble.cpp          # Shannon entropy, Brier, regime routing
│   │   ├── ppo_agent.cpp         # PPO via libtorch or custom
│   │   ├── paper_trading.cpp     # Positions, Kelly, TP/SL tracking
│   │   ├── risk_manager.cpp      # 7 pre-trade checks
│   │   └── bot.cpp               # Consumer loop, producer threads
│   ├── io/                       # I/O layer
│   │   ├── ws_feed.cpp           # boost.beast WebSocket
│   │   ├── exchange.cpp          # Binance REST via libcurl
│   │   ├── telegram.cpp          # Telegram send via libcurl
│   │   ├── sentiment.cpp         # RSS + F&G fetcher
│   │   └── cache.cpp             # Arrow/Feather read/write
│   ├── config.cpp                # JSON config loader
│   └── main.cpp                  # Entry point
├── python/                       # Kept for training-only
│   ├── train.py                  # Optuna + CatBoost training
│   ├── shap_analysis.py          # SHAP explainability
│   └── export_model.py           # Export .cbm + scaler for C++ loader
└── tests/
    ├── test_features.cpp
    ├── test_extractors.cpp
    ├── test_bs_math.cpp
    └── test_ensemble.cpp
```

---

## Phased Implementation

### Phase 1 — Foundation & Core Math (Week 1)

Port the pure C hot-path functions. These have zero external dependencies and can be unit-tested independently.

#### [NEW] `include/quanta/types.h`
- Feature vector type: `double features[278]`
- Candle struct: `{double open, high, low, close, volume, taker_buy}`
- Config constants: all values from `quanta_config.py`
- Error codes enum

#### [NEW] `include/quanta/config.h` + `src/config.cpp`
- All 9 config dataclasses → C++ structs
- JSON loading via nlohmann/json
- `EventExtractionConfig`, `ModelConfig`, `RiskManagerConfig`, etc.

#### [NEW] `src/core/indicators.c`
- Port from `quanta_features.py`: RSI, MACD, BB, ATR, ADX, Stochastic
- Port: Hurst, SampleEntropy, KyleLambda, Amihud, MF-DFA, TransferEntropy
- Port: VPIN, fractional differencing
- All pure C, operates on `double*` arrays

#### [NEW] `src/core/bs_math.c`
- `jit_kou_barrier_prob()` — Kou jump-diffusion with GARCH vol
- `jit_bs_time_decay()` — single-barrier CDF proxy
- `jit_bs_implied_vol_ratio()` — implied vol heuristic
- Port GARCH(1,1) filter: `omega=1e-6, alpha=0.10, beta=0.85`

#### [NEW] `src/core/extractors.c`
- `fast_extract_athena()`, `fast_extract_ares()`, etc. — all 7 extractors
- `fast_triple_barrier_label()` — barrier labeling
- CUSUM filter logic
- Direct port from `quanta_numba_extractors.py` (already C-like Numba)

#### [NEW] `src/core/features.c`
- `extract_features()` → builds 278-dim vector
- Per-TF feature computation (7 TFs × features each)
- Time features (UTC), sentiment slot, delta features
- Impulse features (270-274), BS features (275-277)

#### [NEW] `src/core/hmm.c`
- 3-state GaussianHMM: EM training + Viterbi predict
- State sorting descending by mean log-return
- Feature 231 output: `1.0 - rank/2.0`
- Cache format: model + rank_to_int mapping

**Verification**: Unit tests comparing C output vs Python output on same input data.

---

### Phase 2 — ML Inference Engine (Week 1-2)

#### [NEW] `src/engine/ml_engine.cpp`
- Load CatBoost `.cbm` models via C API (`CalcModelPrediction`)
- Load StandardScaler params (mean/std arrays from JSON export)
- Domain mask application per specialist
- 7 specialist inference pipeline
- Brier score tracking (rolling deque)

#### [NEW] `src/engine/ensemble.cpp`
- Shannon entropy weighting: `H = -(p·log2p + (1-p)·log2(1-p))`
- Regime routing multiplication (load from `regime_routing_weights.json`)
- Brier calibration multiplier
- Event overlap discount (≥3 specialists same direction → 0.85×)
- Entropy veto (H > 0.85 → NEUTRAL)
- Disagreement threshold (std > 0.20)

#### [NEW] `src/engine/ppo_agent.cpp`
- Option A: **libtorch** — load saved PPO model, inference only
- Option B: **Custom** — small 2-layer MLP, ~100 lines, loads weights from JSON
- Size oracle formula: `ppo_size_mult ∈ [0.25, 2.0]`
- State construction: 278 features + 10 specialist signals

#### [NEW] `python/export_model.py`
- Export CatBoost `.cbm` files (already native)
- Export scaler `mean_` and `scale_` arrays to JSON
- Export PPO weights to JSON/binary
- Export regime routing weights
- Export Optuna best hyperparams

**Verification**: Compare C++ inference output vs Python on 100 test samples. Must match within floating-point tolerance.

---

### Phase 3 — Trading Core & Bot Architecture (Week 2)

#### [NEW] `src/engine/paper_trading.cpp`
- Position management: open, close, flat_all
- TP1/TP2/TP3/SL tracking with barrier checking
- Kelly sizing with `barrier_rr` and `bs_edge`
- Commission (4bps) + slippage (2bps)
- Balance persistence (JSON state file)
- `_bs_bars_to_hit` tracking per symbol

#### [NEW] `src/engine/risk_manager.cpp`
- 7 pre-trade checks (max positions, max risk, correlation, etc.)
- Circuit breaker logic
- `flat_all_callback` wiring
- RLock equivalent: `std::recursive_mutex`

#### [NEW] `src/engine/bot.cpp`
- Main consumer loop (replaces `consumer_worker`)
- Thread pool: producer threads, consumer thread, stats logger
- Thread-safe queue: `std::queue` + `std::mutex` + `std::condition_variable`
- Signal handling: SIGINT/SIGTERM → flat_all → save → exit
- Batch processing pipeline
- CUSUM per-symbol state management
- Streak boost logic
- BS veto/turbo power block
- Daily evaluator

#### [NEW] `include/quanta/utils.h`
- Thread-safe logging (dual: console + file)
- Thread-safe concurrent queue template
- Timer utilities
- `nan_to_num()` equivalent

**Verification**: Paper trading simulation — same input, same trades, same P&L.

---

### Phase 4 — I/O Layer (Week 2-3)

#### [NEW] `src/io/ws_feed.cpp`
- boost.beast WebSocket client
- Binance combined stream (`/stream?streams=...`)
- CandleStore: in-memory OHLCV ring buffers per symbol/TF
- Auto-reconnect with exponential backoff
- Candle aggregation: 5m → 15m, 1h, 4h

#### [NEW] `src/io/exchange.cpp`
- Binance REST API via libcurl
- Futures endpoints: klines, exchangeInfo, ticker, depth
- Rate limiting (Binance weight tracking)
- Proxy support (Psiphon)
- Retry with exponential backoff

#### [NEW] `src/io/telegram.cpp`
- Send message via libcurl POST
- Markdown formatting
- Command polling (optional — could keep Python telegram bot)

#### [NEW] `src/io/sentiment.cpp`
- Fear & Greed Index fetcher
- RSS parser for CryptoPanic
- Loughran-McDonald lexicon scoring
- 7 sentiment features output

#### [NEW] `src/io/cache.cpp`
- Apache Arrow C++ for Feather read/write
- Per-symbol write locks (`std::shared_mutex`)
- Cache warmup from historical data

**Verification**: WebSocket feed parity — same candles received, same events triggered.

---

### Phase 5 — Integration & Build System (Week 3)

#### [NEW] `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.20)
project(QUANTA VERSION 12.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(CURL REQUIRED)
find_package(Arrow REQUIRED)
find_package(nlohmann_json REQUIRED)

# CatBoost C API
find_library(CATBOOST_LIB catboostmodel)

# libtorch (optional, for PPO)
find_package(Torch QUIET)

add_executable(quanta src/main.cpp ...)
target_link_libraries(quanta
    Eigen3::Eigen Boost::system CURL::libcurl
    Arrow::arrow ${CATBOOST_LIB}
)
```

#### [NEW] `vcpkg.json`
```json
{
  "dependencies": [
    "eigen3", "boost-beast", "curl",
    "apache-arrow", "nlohmann-json"
  ]
}
```

#### [NEW] `src/main.cpp`
- Parse CLI args
- Load config from JSON
- Initialize bot
- Run event loop
- Graceful shutdown

**Verification**: Full integration test — bot starts, connects to Binance, receives candles, makes predictions, opens paper trades.

---

### Phase 6 — Polish & Performance (Week 3-4)

- SIMD optimization for feature extraction (AVX2)
- Memory pool allocator for feature vectors (avoid malloc per candle)
- Profile with Intel VTune or perf
- Lock-free queue (optional: moodycamel::ConcurrentQueue)
- Windows service deployment (optional)
- Benchmarking: Python vs C++ latency comparison

---

## What Stays in Python (Training-Only Toolchain)

| Component | Why Keep Python |
|-----------|----------------|
| Optuna hyperparameter search | No C++ equivalent, 20 trials is infrequent |
| SHAP explainability | Analysis tool, not latency-critical |
| CatBoost training | Python API is richer, training is offline |
| TFT training | PyTorch training API much richer in Python |
| Walk-forward backtester | Analysis tool, runs offline |
| Dashboard (Flask) | Web UI, not latency-critical |

**Workflow**: Train in Python → export `.cbm` + scaler + weights → C++ loads and infers.

---

## Expected Performance Gains

| Metric | Python (current) | C/C++ (expected) |
|--------|-----------------|------------------|
| Feature extraction (278-dim) | ~5-10ms (Numba) | **~0.1-0.5ms** |
| CatBoost inference (7 specialists) | ~2-5ms | **~0.5-1ms** (C API) |
| Ensemble pipeline | ~1-2ms | **~0.05-0.1ms** |
| End-to-end per coin | ~20-30ms | **~2-5ms** |
| Memory per coin | ~50MB (Python overhead) | **~5-10MB** |
| Startup time | ~10-15s (import overhead) | **~1-2s** |
| Binary size | N/A (interpreted) | **~5-10MB** single binary |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CatBoost C API limited docs | Medium | Use `model_calcer.h` examples, test extensively |
| boost.beast complexity | Medium | Start with simple WS client, iterate |
| Floating point differences | Low | Accept ε < 1e-6, unit test heavily |
| Build system on Windows | Medium | vcpkg handles most deps |
| MX130 GPU: no CUDA C++ | Low | All inference is CPU anyway |
| Loss of hot-reload | Medium | Fast compile with ccache |
| Debugging harder than Python | High | Extensive logging, sanitizers, unit tests |

---

## Open Questions

> [!IMPORTANT]
> 1. **Telegram**: Keep Python telegram bot as separate process, or port to C++ libcurl? Python is simpler for command handling.

> [!IMPORTANT]
> 2. **PPO**: Use libtorch (adds ~1GB dependency) or implement custom 2-layer MLP (~100 lines of C++)? Custom is lighter but no autograd for future training.

> [!IMPORTANT]
> 3. **Dashboard**: Keep Flask dashboard as-is (separate Python process), or build a simple C++ HTTP server?

> [!IMPORTANT]
> 4. **Priority**: Should we optimize for fastest possible live trading first (Phase 1-3), then add I/O (Phase 4)? Or go breadth-first?

> [!IMPORTANT]
> 5. **Do you want to keep the Python bot running in parallel during migration** for comparison testing? (Recommended: yes, until C++ matches output exactly)

---

## Verification Plan

### Automated Tests
- Unit tests for every C function (features, indicators, extractors, BS math)
- Integration test: load same `.cbm` model in Python and C++, compare predict_proba output on 1000 samples
- Paper trading parity test: replay same candle sequence, verify identical trade decisions

### Manual Verification
- Run Python and C++ bots side-by-side on live feed
- Compare: same predictions, same opportunities, same trades
- Monitor for 24h before cutting over to C++ only
