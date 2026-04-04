---
name: QUANTA Key Files Map
description: What each file does, which are critical to edit, and what each controls
type: project
---

## Core Files — Highest Risk to Edit

### quanta_config.py
Single source of truth for ALL parameters. 9 dataclasses with `__post_init__` validators.
- `Config.base_feature_count = 278` — feature vector size, propagates everywhere
- `EventExtractionConfig` — accessed as `self.cfg.events` (NOT `self.cfg.event_extraction` — known gotcha)
- Nike params: `nike_body_min`, `nike_atr_mult`, `nike_vol_mult`, `nike_tp_atr`, `nike_sl_atr`, `nike_max_bars`
- Any change here propagates to all files that read `self.cfg.*`

### QUANTA_ml_engine.py (~4000+ lines)
Feature extraction + training pipeline + ensemble inference. Most complex file.

Key methods:
- `_extract_features(tf_analysis, symbol, _raw_candles, _training_ts_ms, _training_symbol)` — builds 278-dim vector. `_raw_candles` is a DICT `{closes, highs, lows, volumes, taker_buy}`, NOT a list of rows.
- `_extract_features_from_candles(candles, position)` — training path. Overwrites time features [107:116] with actual candle timestamp after calling `_extract_features()`.
- `_train_specialist(specialist_name, X, y, sample_weights)` — full training pipeline per agent
- `_run_optuna_search(...)` — adaptive hyperparameter search, 20 trials, persists to `models/optuna_studies/`
- `predict_with_specialists(symbol, tf_analysis)` — live inference, builds `_live_candles` from candle_store
- `_update_ensemble_weights()` — uses `val_auc` key (not `val_acc`) from performance history
- Specialist dict: `self.specialist_models` — keys: 'athena','ares','hermes','artemis','chronos','hephaestus','nike'
- Domain masks: `_TREND`, `_TREND_SHORT`, `_MEAN_REVERSION`, `_FLOW_VOLUME`, `_STRUCTURAL`, `_MACRO`, `_IMPULSE`
- Brier scores: `self._brier_scores[agent]` — dict with `sum`, `count`, `rolling` (deque maxlen=500). Reset on deploy.
- Optuna studies: `self._optuna_studies[specialist]` — loaded from `models/optuna_studies/*.pkl`

Constants at module level:
- `OPTUNA_N_TRIALS = 20`, `OPTUNA_SEARCH_ITERS = 300`, `OPTUNA_MAX_SEARCH_ROWS = 3000`

### QUANTA_bot.py (~3500+ lines)
Main bot loop. Producer/consumer architecture.

Key methods:
- `Bot.__init__()` — initializes everything
- `Bot.run()` — startup sequence, launches all threads
- `consumer_worker()` — perpetual prediction loop. The heart of the bot.
- `_daily_evaluator_worker()` — uses `self.ml.rt_cache` (NOT `self.rt_cache` — fixed bug)
- `_rt_cache_updater()` — populates `self.ml.rt_cache`
- `_realtime_prediction_timer()` — triggers timed predictions

Key variables in consumer loop:
- `specialist_keys = ['athena','ares','hermes','artemis','chronos','hephaestus','nike']`
- `_overlap_discount` — checked with `locals()` not `dir()` (fixed bug)
- `regime_mults_matrix` — shape (7, batch_len), per-item per-specialist regime multipliers
- `specialists_ready` — True when specialist models exist

### QUANTA_trading_core.py
- `PaperTrading` class — position management, TP/SL tracking, balance persistence (`paper_trading_state.json`)
- `TP1_WEIGHT = 0.3`, `SL_WEIGHT = 2.0` (wrong predictions 6× heavier than marginal wins)
- `flat_all(price_lookup)` — emergency close all positions
- `barrier_base = min(magnitude, MAX_MAGNITUDE)` — no double-counting volatility
- PPO veto multiplier: `min(2.0, 1.0 + n_resolved/200)` — ramps to prevent early collapse

### quanta_numba_extractors.py
Numba JIT compiled. All event extractors + `fast_triple_barrier_label()`.

Key functions:
- `fast_extract_athena(closes, highs, atrs, atr_pct, cusum_pos, volumes, orig_idx)`
- `fast_extract_ares(closes, lows, atrs, atr_pct, cusum_neg, volumes, orig_idx)`
- `fast_extract_artemis(closes, highs, lows, atrs, atr_pct, cusum_pos, cusum_neg, volumes, vol_avg, orig_idx)` — bidirectional, buffer size N*2
- `fast_extract_nike(closes, highs, lows, atrs, volumes, vol_avg20, orig_idx)` — no CUSUM
- `fast_triple_barrier_label(closes, highs, lows, atrs, i, direction, tp_atr, sl_atr, max_bars)` → (label, weight)
- Labels: 1=BULLISH, 0=BEARISH, -1=invalid (excluded from training)

Lookback fix: Artemis uses `highs[i-_LOOKBACK:i]` NOT `[i-1]` (off-by-one was fixed 2026-04-01).

---

## Supporting Files

### quanta_features.py
- `MultiTimeframeAnalyzer` — aggregates 5m → 15m, 1h, 4h
- `Indicators` class — all technical indicators
- Advanced: Hurst, SampleEntropy, TransferEntropy, KyleLambda (correct OLS: `Cov(ΔP,signed_vol)/Var(signed_vol)`), Amihud, MF-DFA

### quanta_risk_manager.py
- `RiskManager` — RLock (re-entrant), 7 pre-trade checks, circuit breaker
- `get_size_multiplier()` called inside `on_trade_closed()` — requires RLock not Lock
- `flat_all_callback` parameter in constructor

### quanta_cache.py
- `FeatherCache` — per-symbol write lock via `_write_locks_meta` meta-lock + `_write_locks` dict
- 2.5× faster than parquet for OHLCV data

### quanta_model_registry.py
- `ModelRegistry` — RLock, atomic save via `os.replace(tmp → final)`
- `should_deploy()` — checks accuracy/Brier gates, skips when `meta._skip_auc_gate=True`
- `ModelMetadata` — tracks `val_auc` (canonical), `val_acc` (legacy), `val_brier`, generation

### quanta_monitor.py
- `ModelMonitor.log_prediction(predicted_class, predicted_prob, actual_class)` — always called with `actual_class=None` from bot. `calibration_error` always returns None (structural — Brier scores are the real calibration signal).
- `calibration_error` aligns arrays by `min_len` to avoid misalignment when actuals sparse

### QUANTA_agents.py
- `PPOAgent` — 3-action policy (HOLD/LONG/SHORT), value estimate, Differential Sharpe Ratio reward
- `PPOMemory` — experience buffer

### quanta_deeplearning.py
- TFT (Temporal Fusion Transformer) — feeds feature index 223 at inference
- Quality gate: `tft_val_auc > 0.55` required. If below, slot 223 stays 0.0.
- Cap per symbol: 50 sequences (MX130 memory)

### quanta_selector.py (QUANTA_selector.py)
- `extract_events_from_klines(symbol, klines)` — runs all 7 extractors
- `_calculate_base_indicators(df)` — computes Vol_Avg_20, Vol_Avg_50, CUSUM_pos, CUSUM_neg, ATR, ATR_pct
- Artemis call passes both `cusum_pos` AND `cusum_neg`
- Nike call uses `vol_avg20` (not vol_avg)

### quanta_archive.py
- Historical data download with 3-retry exponential backoff (1s→2s→4s)

### main.py
- Entry point: `Bot().run()`
- Concurrent module preloading, Binance weight limiter

---

## File Edit Risk Levels

| File | Risk | Why |
|------|------|-----|
| quanta_config.py | 🔴 CRITICAL | Changes propagate everywhere. Field names used in other files (`self.cfg.events` not `self.cfg.event_extraction`). |
| QUANTA_ml_engine.py | 🔴 CRITICAL | 4000+ lines, feature extraction, training, inference. One wrong index breaks everything. |
| quanta_numba_extractors.py | 🔴 CRITICAL | JIT compiled — runtime errors only show at call time. Off-by-one bugs silent. |
| QUANTA_trading_core.py | 🟡 HIGH | Position/order logic, weight constants affect training feedback |
| QUANTA_bot.py | 🟡 HIGH | Consumer loop threading, signal handling, variable scope issues |
| QUANTA_selector.py | 🟡 HIGH | Extractor routing — wrong vol_avg passed to wrong extractor |
| quanta_model_registry.py | 🟢 MEDIUM | Deploy gate logic, metadata keys |
| quanta_monitor.py | 🟢 MEDIUM | Array alignment important |
| quanta_features.py | 🟢 MEDIUM | Technical indicator math |
| quanta_risk_manager.py | 🟢 MEDIUM | Lock type matters (RLock vs Lock) |

---

## Model Files Location

`models/` directory:
- `{agent}_gen1.cbm` — CatBoost model
- `{agent}_scaler.pkl` — StandardScaler
- `{agent}_calibrator.pkl` — conformal calibrator
- `{agent}_feature_indices.npy` — domain mask indices
- `{agent}_importance_mask.npy` — pruned feature importance mask
- `tft_model.pth` — TFT weights
- `model_registry.json` — deployment history
- `generation.txt` — current generation number
- `rare_events_buffer.npz` — hard negative mining buffer
- `agent_features_cache.pkl` — cross-event sampling cache
- `optuna_studies/{agent}.pkl` — persistent Optuna TPE study per specialist

**⚠️ If feature count changes, delete all .cbm and .pkl files and retrain.**
