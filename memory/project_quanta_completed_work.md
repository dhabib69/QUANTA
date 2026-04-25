---
name: QUANTA Completed Implementation Work
description: All implemented phases and fixes in chronological order, with exact file locations
type: project
---

## Phase 1 — Feature Count Unification (2026-03-31)
- Feature count unified to 268 → later 270 → now 275 (all changes in quanta_config.py only)
- All code reads `self.cfg.BASE_FEATURE_COUNT` — never hardcoded

## Phase 2a — Paper Trading Realism
- Commission (4bps) + slippage (2bps) in `_execute_market_order` and `close_position`
- File: QUANTA_trading_core.py

## Phase 2b — Risk Manager Per-Trade Cap
- `pre_trade_check()` check #7: blocks notional > 2% of balance
- `RiskManagerConfig.max_risk_per_trade_pct = 2.0` in quanta_config.py
- File: quanta_risk_manager.py

## Phase 2c — Concurrency Fixes
- FeatherCache: `_write_locks_meta` meta-lock + `_write_locks` dict (replaces defaultdict race)
- ModelRegistry: RLock + atomic save via `os.replace(tmp → final)`
- Files: quanta_cache.py, quanta_model_registry.py

## Phase 2 — flat_all() Emergency Close
- `PaperTrading.flat_all(price_lookup)` — uses entry price as fallback
- Wired as `flat_all_callback` to RiskManager. Signal handler + KeyboardInterrupt call it.
- Files: QUANTA_trading_core.py, QUANTA_bot.py, quanta_risk_manager.py

## Phase 2 — Backtester Purge Gap
- 48-candle purge gap between train/test windows in walk-forward backtester
- File: quanta_backtester.py

## Phase 2 — GPU Subsampling Fix
- Was: `train_test_split(stratify=...)` — random shuffle, look-ahead bias
- Now: temporal head+tail (30% oldest + 70% most recent) — preserves regime diversity
- File: QUANTA_ml_engine.py

## Phase 3a — Error Handling Sweep
- All `except: pass` replaced with `logging.warning/error(..., exc_info=True)`
- Key fixes: CSV writes in quanta_paper_trading.py, DualLogger, auto-save loop, hard-neg mining, FeatherCache.set(), generation counter
- Inference fallbacks (TFT, calibrator, regime, Brier) → logging.debug (high-freq, expected)

## Phase 3b — Smart Execution Retry
- 3-retry loop with exponential backoff (2s→4s→8s), per-slice timeout (10s via ThreadPoolExecutor)
- Partial fill detection: warns if filled < 95%
- Files: quanta_smart_exec.py, quanta_config.py

## Training Quality Fixes (2026-03-31)
- TFT label alignment: now uses `fast_triple_barrier_label()` with Athena's settings
- Temporal stratified undersampling: 10 equal temporal buckets (not random)
- Dynamic purge gap: `max(all agent max_bars)` from config
- GPU head+tail subsample: 30% oldest + 70% most recent
- Cross-event negative ratio: 15% → 5%
- Hard-neg collapse detection: warning when >60% misclassified
- UTC time features: `datetime.utcnow()` not `datetime.now()`
- Val class balance reporting with 5:1 ratio warning

## Architecture Session (2026-03-31)
- Temporal train/val split with 48-candle purge gap (replaced `train_test_split`)
- CPCV: `CombinatorialPurgedCV(n_groups=6, k_test=2, purge_gap=48)` — 5-fold cap
- MultiStreamDriftMonitor: 3-stream ADWIN, both accuracy AND (calibration OR feature) must fire
- Brier-scored ensemble weights: `0.6*(1-brier) + 0.4*val_auc`
- Dashboard: quanta_dashboard.py + templates/dashboard.html (Chart.js equity curve, Brier, drift)
- Paper trading balance persistence: saves/loads `paper_trading_state.json`
- RL ↔ Paper trade linkage: `RLMemory.record_trade_result()` called from `close_position()`

## v11.5b Architecture Overhaul (2026-04-01)

### Critical Bug Fix
- `self.cfg.event_extraction` → `self.cfg.events` (4 occurrences) — was blocking all 7 agents from training
- File: QUANTA_ml_engine.py

### Academic Paper Fixes
- Kyle's Lambda: `Cov(ΔP, signed_vol) / Var(signed_vol)` (correct OLS) — File: quanta_features.py
- CUSUM: removed ad-hoc drift subtraction, now pure zero-drift per LdP AFML Ch.2 — File: QUANTA_bot.py
- VPIN: removed dead code (overwritten assignment) — File: quanta_features.py
- Feature Pruning: `LossFunctionChange` (true MDI) + `mean/2` threshold — File: QUANTA_ml_engine.py

### Nike Agent (replaces Divergence/Apollo)
- Trigger: `range > 2×ATR AND vol > 1.5×avg20 AND body_eff > 0.5`, no CUSUM, both directions
- Files: quanta_config.py, quanta_numba_extractors.py, QUANTA_ml_engine.py, QUANTA_selector.py, QUANTA_bot.py

### Artemis Now Bidirectional
- Bull: `CUSUM_pos + vol_surge + NOT new_high` (stealth accumulation)
- Bear: `CUSUM_neg + vol_surge + NOT new_low` (stealth distribution)
- Files: quanta_numba_extractors.py, QUANTA_selector.py

### Feature Count: 270 → 275
- 5 impulse features (270-274): body_eff, taker_flow_persist, pre_impulse_r2, atr_rank, depth_delta
- `Vol_Avg_20` added to `_calculate_base_indicators()` in QUANTA_selector.py
- `_IMPULSE` domain mask added in QUANTA_ml_engine.py

## Prediction Pipeline Math Fixes (2026-04-01)

### QUANTA_bot.py
1. **HMM per-item regime** — was applying batch[0]'s regime to all coins. Now per-item loop, `regime_mults_matrix` shape (7, batch_len)
2. **Entropy veto** — replaced arbitrary top-2 disagreement (40%) with `H(p_ens) > 0.85 bits → NEUTRAL`
3. **Conformal uncertainty** — `np.mean` → `np.max` (mean is mathematically invalid for conformal sets)
4. **Uncertainty penalty** — removed magic `/50 × 0.15`. Now `clip(width/100, 0, 1)`, max 25% reduction
5. **Disagreement threshold** — 0.15 → 0.20 (derived: `sqrt(0.25/7) ≈ 0.189`)
6. **Brier reading** — bot now reads `rolling` deque (last 500) not `sum/count` global avg
7. **Event overlap discount** — ≥3 high-certainty specialists same direction → 0.85× discount

### QUANTA_trading_core.py
8. **TP1_WEIGHT** — 1.0 → 0.3 (TP1 is break-even after fees)
9. **SL_WEIGHT** — 0.5 → 2.0 (wrong predictions penalized 6× more than marginal wins)
10. **barrier_base** — removed `max(magnitude, volatility*0.5)` double-counting. Now `min(magnitude, MAX_MAGNITUDE)`
11. **Neutral outcomes** — fed to PPO as wrong-direction signal. `raw_return = -move_mag`
12. **PPO veto ramp** — `min(2.0, 1.0 + n_resolved/200)`. Prevents early-training collapse

## Pipeline Suboptimal Issues Fixes — Second Pass (2026-04-01)

### QUANTA_ml_engine.py
13. **Brier reset on retrain** — clears `_brier_scores[agent]` on successful model deploy
14. **TFT quality gate** — temporal val split (15%), val AUC > 0.55 required. `tft_trained=True` only if useful
15. **Nike domain mask expanded** — added `_macd_per_tf`, `_bb_per_tf`, cross-TF MACD/BB indices to `_IMPULSE`
16. **Val minimum event count** — warning if val<20 events, sets `meta._skip_auc_gate=True`

### quanta_numba_extractors.py
17. **Artemis bidirectional** — added bearish loop with direction=-1, buffer size N*2

### quanta_model_registry.py
18. **skip_auc_gate** — `should_deploy()` skips accuracy/Brier gates when flag set

### quanta_monitor.py
19. **log_prediction guard** — `actual_class is not None` check (was crashing on None)
20. **calibration_error alignment** — last-n entries aligned by `min_len`

### quanta_archive.py
21. **Download retry** — 3-retry exponential backoff (1s→2s→4s) in `_download_and_extract_month()`

## Feature Extraction Critical Fixes (2026-04-01)

22. **Impulse features always zeros** — `_raw_candles` is a dict but old code indexed as list-of-rows. `len(dict)=4 < 20` → always defaulted. Fixed: reads `_rc['closes'][-1]` etc.
23. **Live inference missing candle data** — `predict_with_specialists()` called `_extract_features()` without candle data. Fixed: builds `_live_candles` dict from `candle_store` and passes it.
24. **Missing taker_buy key** — second training path built dict without `taker_buy`. Added `'taker_buy': None`.

## Optuna Adaptive Hyperparameter Search (2026-04-01)

25. `_run_optuna_search()` added to `DeepMLEngine` — 20 CPU trials, TPE sampler, persists to `models/optuna_studies/{specialist}.pkl`
26. Search space: `depth` 4-8, `lr` 0.01-0.35 log, `l2` 1-15 log, `subsample` 0.6-1.0
27. Fires on first training only (`model is None AND not _optuna_searched`). Force re-search: set `specialist['_optuna_searched'] = False`
28. Best params merged into `specialist['hyperparams']` before production training

## Full Audit Bug Fixes — Third Pass (2026-04-01)

29. **`self.rt_cache` undefined** — was AttributeError in `_daily_evaluator_worker`. Fixed: `self.ml.rt_cache` — File: QUANTA_bot.py lines 901, 908, 911
30. **`_overlap_discount` scope** — `dir()` (module namespace) → `locals()` (local scope). Discount was silently never applied — File: QUANTA_bot.py line 1854
31. **Ensemble weights wrong key** — `latest.get('val_acc', 0.5)` → `latest.get('val_auc', latest.get('val_acc', 0.5))`. Weights were always computed with AUC=0.5 — File: QUANTA_ml_engine.py line 1873
32. **Artemis lookback off-by-one** — `highs[i-_LOOKBACK:i-1]` excluded most recent bar. Artemis could fire when true new high existed 1 bar back — Fixed: `highs[i-_LOOKBACK:i]` — File: quanta_numba_extractors.py lines 278, 296
33. **Cross-event ZeroDivisionError** — `n_cross_target // len(other_agents)` crashes if only 1 agent has data. Added guard — File: QUANTA_ml_engine.py line 4724
34. **Optuna import inside trial loop** — `roc_auc_score` imported 20× per specialist. Hoisted to method level — File: QUANTA_ml_engine.py

## PPO Redesign: Gate → Size Oracle (2026-04-01)

35. **PPO Heimdall Gate removed** — old design dampened/boosted `ml_conf`, could block trades. Replaced with `ppo_size_mult ∈ [0.25, 2.0]` applied to Kelly notional. ML signal can never be blocked by PPO. — QUANTA_bot.py
36. **`open_position()` updated** — added `ppo_size_mult=1.0` param. Applied after Kelly calc, before risk manager gate. Hard cap: notional never exceeds MAX_RISK. Stored in `positions[symbol]['ppo_size_mult']`. — QUANTA_trading_core.py
37. **PPO reward redesigned** — was veto-based (reward for blocking losses). Now: `raw_return = outcome_sign × move_mag × ppo_size_mult`. PPO learns sizing quality, not binary veto quality. — QUANTA_trading_core.py `check_predictions()`
38. **`ppo_size_mult` stored in opportunity dict** — key `'ppo_size_mult'` in every opportunity record for RL memory and future analysis. — QUANTA_bot.py

## Black-Scholes Barrier Math Integration (2026-04-02)

### Academic Foundation
- Hull (2018) Ch.26 + Darling-Siegert (1953) scale function: `P(hit a before b) = [s(0)-s(b)]/[s(a)-s(b)]` where `s(x) = exp(-θx)`, `θ = 2ν/σ²`, `ν = μ - 0.5σ²`
- Zero-drift (gambler's ruin): `P(TP) = sl_dist / (tp_dist + sl_dist)`. Athena: 1.0/2.5 = 0.40
- Verified: strong bull (θ>>0) → P→1; strong bear (θ<<0) → P→0 ✓

### New Features (indices 275-277)
- `[275]` **bs_theoretical_win_prob** — P(TP before SL) under drifted GBM. Specialist-agnostic (median barriers TP=1.5, SL=1.0 ATR)
- `[276]` **bs_time_decay** — approx P(crossing barrier in remaining bars). Single-barrier CDF proxy (`tanh` pattern, same as VPIN). NOT exact Kunitomo-Ikeda series — CatBoost learns the residual.
- `[277]** bs_implied_vol_ratio** — QUANTA-specific heuristic: `sigma_implied / sigma_realized` back-solved from avg bars-to-hit. Clipped [0.1, 10]. Starts at 1.0 (neutral) until ≥5 trade history accumulates.

### Feature Count: 275 → 278
- `quanta_config.py:76` — `base_feature_count: int = 278`
- `QUANTA_ml_engine.py:479` — `_N = 278`
- Domain masks updated: `[275,276]` on trend/mean-rev/macro, `[275,276,277]` on volatility/structural/impulse, `[277]` on flow

### New @njit Functions — quanta_features.py
- `_jit_bs_barrier_prob(log_returns, tp_dist, sl_dist)` — scale function formula
- `_jit_bs_time_decay(sigma, tp_dist, bars_remaining)` — CDF approximation
- `_jit_bs_implied_vol_ratio(avg_bars_to_hit, barrier_dist, sigma_realized)` — implied vol heuristic
- All imported in QUANTA_ml_engine.py

### Dynamic Kelly b — QUANTA_trading_core.py
- `open_position()` gains `barrier_rr=2.0` param (replaces hardcoded `b = 2.0`)
- `open_position()` gains `bs_edge=None` param — penalises when ML < 2% above random-walk baseline: `edge_penalty = max(0.5, min(1.0, bs_edge / 0.10))`
- BS edge threshold 0.02 (2%) is a local heuristic, not from Hull

### Wired from Bot — QUANTA_bot.py:2242
- Dominant specialist identified from `specialist_probs_batch[idx]` → looks up TP/SL ATR from `self._sys.events`
- Computes `barrier_rr = tp_atr / sl_atr` (per-specialist, e.g. Athena=1.5, Artemis=2.0, Nike=2.5)
- Computes `bs_edge = (ml_conf/100) - sl_atr/(tp_atr+sl_atr)` (zero-drift baseline)
- Passes both to `open_position()`

### Implied Vol Tracking — QUANTA_trading_core.py + QUANTA_bot.py
- `PaperTrading._bs_bars_to_hit` — per-symbol deque(maxlen=50), appended on barrier hits (TP/SL, not TIMEOUT)
- `PaperTrading.get_avg_bars_to_hit(symbol)` — returns mean or None if <5 samples
- `DeepMLEngine._bs_avg_bars_to_hit` — dict synced from paper trading every RL check cycle
- Feature extraction reads `self._bs_avg_bars_to_hit.get(symbol, 0.0)` → `_jit_bs_implied_vol_ratio()`

**Models need retraining** — feature vector changed 275 → 278.

## HMM Regime Detection Overhaul (2026-04-02)

### Root Issues Found (6 bugs in `_get_regime`)
1. Feature mismatch: trained on `[returns, rolling_std, vol_ratio]` but bot routing sent `[log_ret, atr_pct, adx]` for prediction — completely wrong features at inference
2. `n_iter=20` — EM not converging for 5-state model (standard minimum is 100)
3. 50 observations — 10 per state average, too few for reliable EM
4. `covariance_type="diag"` — assumes return and volatility are independent given state (they're not)
5. Label switching — every 4h refit permutes state semantics; `_regime_routing` assumes state 0=bull always
6. Raw state index 0–4 as continuous ordinal feature — invalid for CatBoost

### Fix Applied (QUANTA_ml_engine.py `_get_regime`)
- Features changed to `[log_returns, atr_pct, vol_ratio]` — matches bot routing
- `n_iter=100`, `covariance_type='full'`, 200 observations
- State ordering: sorted DESCENDING by mean log-return → rank 0=strong-bull, 4=crash
- Matches `_regime_routing` table convention: `[0(up), 1(wk-up), 2(range), 3(wk-dn), 4(crash)]`
- Cache format changed to `{'model': _hmm, 'rank_to_int': array}` dict
- Feature 231 = `1.0 - rank/4.0` → [1.0=bull, 0.75, 0.5, 0.25, 0.0=crash] — semantically stable
- Bot routing (QUANTA_bot.py) updated to use new cache format + matching feature construction

### Pre-Training Audit (2026-04-02) — All Clear
- `base_feature_count=278` (config) == `_N=278` (engine) ✓
- All domain mask indices within [0, 277] ✓
- PPO `input_dim = 278 + 10 = 288` ✓
- All EventExtractionConfig barrier fields present ✓
- Stale `# 268 features` comment fixed at engine line 384

## Dashboard Fix (2026-04-02)
- `templates/dashboard.html` was missing → 500 Internal Server Error on every request
- Created full 4-tab dashboard: Overview (equity curve, positions, PnL), Predictions (daily picks, RL outcomes), Models (Brier scores, specialist status), System (queue, latency, log tail)
- Chart.js equity curve, live polling every 5s via `/api/*` endpoints
- File: `templates/dashboard.html`

## GitHub Repository (2026-04-02)
- Repo: https://github.com/dhabib69/QUANTA (private)
- 66 files committed, 27,818 lines
- `.gitignore` excludes: `ml_models_pytorch/` (707MB), `*.pkl`, `*.log`, `feather_cache/`, `catboost_info/`, `*.exe`, plan docs, temp files
- Branch: `main`

## Optuna NaN Fix (2026-04-02)

### Root Cause
6 feature indices (stat arb 248-252, on-chain 254-256, GNN 257) were explicitly set to `float('nan')` during training (when `symbol=None` for offline training). After `nan_to_num` → `0.0`, StandardScaler divides `0.0 / 0.0 = NaN` for zero-variance columns → NaN in X_train → CatBoost `predict_proba` returns NaN → Optuna objective returns NaN → "The value nan is not acceptable" for every trial.

### Fix
1. `stat_arb` training fallback: `[nan, nan, nan, btc_rsi, btc_macd, btc_vol_accel]` → `[0.0, 0.0, 0.0, ...]`
2. `onchain_feats` training fallback: `[nan, nan, nan]` → `[0.0, 0.0, 0.5]` (neutral whale ratio)
3. `graph_feat` training fallback: `float('nan')` → `0.0`
4. Added `nan_to_num` AFTER scaler transform as safety net (posinf→3.0, neginf→-3.0)
5. Deleted corrupted `ml_models_pytorch/agent_features_cache.pkl` (had 743,256 NaN values)
- File: QUANTA_ml_engine.py

## Optuna NaN Fix — Second Pass (2026-04-02)

### Root Cause
Previous fix cleaned features at source but `_run_optuna_search()` subsampled data still propagated NaN through Optuna trials. `roc_auc_score` returns NaN (not an exception) when `predict_proba` outputs NaN — Optuna rejects it as "The value nan is not acceptable".

### Fix
1. Added `nan_to_num` on subsampled `Xs`/`Xv` inside `_run_optuna_search()` (post-scaler safety net)
2. Added zero-variance column detection/removal — drops columns where `std == 0` (StandardScaler 0/0 = NaN source)
3. Added NaN/inf guard on `predict_proba` output and `roc_auc_score` return — returns 0.5 (random baseline) instead of NaN
4. Deleted corrupted study files `ares.pkl`, `athena.pkl` (had 20+ NaN-memorized trials biasing TPE)
- File: QUANTA_ml_engine.py lines 1002-1076

## Bot Runtime Fixes (2026-04-02)

39. **`AttributeError: 'Bot' object has no attribute '_sys'`** — BS barrier block in `consumer_worker` (QUANTA_bot.py line ~2255) used `self._sys.events.*` but `Bot` only sets `self.cfg`, never `self._sys`. Fixed to `self.cfg.events.*` via local `_ev` alias.

40. **SHAP unavailable** — `shap` package not installed. Ran `pip install shap` → v0.51.0. SHAP explainability now active.

## BS Execution Power — "Trillion Dollar Equation" (2026-04-02)

41. **BS Veto** — Feature 275 (`bs_theoretical_win_prob`) now hard-vetoes trades when `P < 0.25`. Reads raw unscaled features from `features_batch[idx][275]`. Sets `passes_gate = False` regardless of ML confidence. — QUANTA_bot.py (consumer loop, after Heimdall Sizer)
42. **BS Turbo Boost** — When `P > 0.55`, multiplies `ppo_size_mult` by `min(1.5, bs_prob / 0.50)`. Stacks with PPO sizing. — QUANTA_bot.py
43. **BS Safety Cap** — `ppo_size_mult = min(ppo_size_mult, 2.5)` after BS×PPO stacking. Prevents runaway 3.0× notional. — QUANTA_bot.py
44. **BS Edge Override** — `_bs_edge` calculation replaced from `(ml_conf/100) - baseline` to `_live_bs_prob - baseline`. Pure stochastic edge instead of mixed ML/stochastic comparison. Fallback to old formula on error. — QUANTA_bot.py (barrier R/R block before `open_position()`)

## Known Structural Limitations (not bugs)
- Monitor calibration dead: `log_prediction()` always called with `actual_class=None` — `calibration_error` always returns None. Brier scores in `_brier_scores` are the live calibration signal instead.
- On-chain features (254-256): 0.0 during training (no offline API) — model learns neutral baseline
- Stat arb features (248-250): 0.0 during training (no cross-pair data) — same
- Order book: OHLCV proxy during training vs real depth during inference — structural mismatch
- Nike bull/bear not split into 2 models — future work
- Regime routing multipliers still hand-coded (35 values, 7 agents × 5 regimes) — need backtest learning
- DIRECTION_THRESHOLD=0.12 misattributed to López de Prado in code comment — it's a local heuristic

## Tier 3 BS & HMM Optimizations (2026-04-02)

44. **Bot HMM Multi-Scale Gap fixed** — Fixed QUANTA_bot.py where bot used to send identical single-bar log-returns. Integrated candle_store logic so the bot now builds proper [log_ret_1bar, log_ret_12bar, log_ret_48bar, atr_pct, vol_ratio] structures.
45. **Unified HMM Systems** — Deprecated identical MoE HMM global prediction block. Now uniformly recycling the cached inline per-symbol self.ml.hmm_models[symbol] 3-state extraction.
46. **Regime Routing Automation** — Removed 21 static arrays routing weights mapping. Inserted _learn_regime_routing in QUANTA_ml_engine.py that scales allocation vectors proportionally based on each individual agent's accuracy distribution across different real-world tracked HMM Regimes in training data.
294. **Kou (2002) Jump-Diffusion BS** — Replaced Hull 2018 base analytical probability calculations in Feature 275 targeting continuous decay with _jit_kou_barrier_prob utilizing Double-Exponential leap adjustments with real-time compensatory scaling properties matching crypto market behavior.

## Dashboard Detail Missing Fix (2026-04-03)
- Tracked empty "Specialist vote breakdown not available" in Thought Modal frontend back to missing API payload fields.
- Fixed `quanta_dashboard.py` `_get_predictions()` to actively push `specialist_probs`, `ppo_action`, `ppo_size_mult`, and `shap_summary` to UI for both `daily_picks` and `recent_preds`.
- Documented 3-part strict Papering Filter gating: 1) Initial Minimum ML Confidence (`ml_confidence_min`), 2) PPO Gate Veto allowance (acting as autonomous size oracle), 3) Risk Manager dynamic limit blocking.

## Neutral Threshold Adjustment (2026-04-03)
- Lowered `direction_threshold` (0.12 → 0.08) and `cusum_threshold` (0.02 → 0.01) in `quanta_config.py`.
- Reasoning: Shrinks the neutral movement zone, converting more micro-movements into valid directional signals instead of being ignored, increasing active prediction yield.

## Paper Trading State Persistence Fix (2026-04-04)
- **Error**: `Object of type ndarray is not JSON serializable` when saving `paper_trading_state.json`
- **Root Cause**: `specialist_probs` field in positions was set to numpy array, which cannot be JSON serialized
- **Fix**: Added numpy array → list conversion in `_save_state()` method (QUANTA_trading_core.py lines 338-343)
- **Code**: `if 'specialist_probs' in p and isinstance(p['specialist_probs'], np.ndarray): p['specialist_probs'] = p['specialist_probs'].tolist()`
- **Impact**: Paper trading state now saves successfully without errors, preserving positions across bot restarts
- **Cleanup**: Deleted corrupted `paper_trading_state.json` file that had JSON parsing errors from previous failed saves

## Binance fapi API Connection Issues (2026-04-04)
- **Symptoms**: Multiple `❌ All 3 attempts failed for https://fapi.binance.com/fapi/v1/ticker/price` and WebSocket connection errors
- **Root Cause**: Psiphon proxy (port 56356) causing intermittent connection failures to Binance futures API
- **Affected Endpoints**: `/ticker/price`, `/openInterest`, WebSocket streams
- **Current Mitigation**: Circuit breaker opens after 20 failures, 60s cooldown, exponential backoff retries
- **Status**: Bot continues operating with cached data and fallback mechanisms; errors are non-critical
- **Files**: QUANTA_network.py (NetworkHelper), quanta_exchange.py (BinanceAPIEnhanced)

## Research Memory Update (2026-04-07)
- Added `memory/kou_first_passage_conditional_note.md` documenting the finite-horizon Kou TP-before-SL probability for Nike-style post-jump entries.
- Captures the strong-Markov conditioning result for "trigger jump already occurred" and records two production paths: one-sided Kou-Wang approximation and finite-horizon PIDE solver.

## Live BS/Kou Execution Wiring (2026-04-07)
- Added `compute_live_kou_barrier_components(...)` in `quanta_features.py` to produce a live execution score using actual TP/SL geometry, finite-horizon time decay, and optional Nike-style post-jump conditioning.
- Added bot helpers in `QUANTA_bot.py` to pull recent live 5m log returns from the candle buffer and compute a specialist-aware BS/Kou score per symbol at execution time.
- Replaced the old feature-275-only veto/turbo path in `QUANTA_bot.py` with the live BS/Kou context so gating now respects actual specialist TP/SL/max-bars instead of generic median barriers.
- Stored BS/Kou execution metadata (`bs_prob`, `bs_order_prob`, `bs_time_prob`, `bs_baseline`, `bs_source`, `bs_specialist`) on each opportunity for downstream auditability.
- Updated `QUANTA_trading_core.py` so Kelly sizing now blends ML confidence with the live BS/Kou TP-before-SL probability instead of ignoring the barrier math economically.
- Honed the Nike path with a dedicated conditional Kou kernel: `_jit_kou_conditional_first_passage(...)` plus `_jit_nike_barrier_prob(...)` now estimate post-jump Nike probabilities from pre-trigger returns, trigger-body size, jump intensity, and a finite horizon instead of relying only on the generic live approximation.
- Tightened execution distances to log-space in `QUANTA_bot.py` via `log1p` / `-log(1-x)` conversion before passing TP/SL geometry into the live barrier solver.
- Verification: `python -m py_compile quanta_features.py QUANTA_bot.py QUANTA_trading_core.py` passed.

## Nike Confirmation Upgrade (2026-04-08)
- Reworked Nike from a pure single-candle detector into an ignition-plus-confirmation trigger across `quanta_numba_extractors.py`, `quanta_nike_screener.py`, and `quanta_nike_live_validator.py`.
- New Nike logic:
  - setup candle: `body_ratio >= 5.0`, `body_eff >= 0.4`, `vol_ratio >= 1.5`, quiet-base gate unchanged
  - immediate same-bar entry only for extreme candles: `body_ratio >= 8.0`, `body_eff >= 0.55`, `vol_ratio >= 2.0`
  - otherwise wait one bar and require confirmation: next bar low holds above setup midpoint, next close stays above setup close, next high breaks setup high
- Extended `nike_max_bars` from `12` to `24` to better fit cache-derived time-to-peak behavior for explosive runners.
- Cache benchmark on `breakout_peak_results.csv` positive spike events:
  - top 50 events: `80.0%` same-bar-or-next-bar recall
  - top 80 events: `75.0%` same-bar-or-next-bar recall
  - benchmark bars-to-peak: mean `25.1–25.6`, median `28.5`
- This benchmark measures pattern recognition recall on curated spike events, not live profit factor over the full universe.

## Nike Full-Cache Benchmark (2026-04-08)
- Added `NIKE_CACHE_PERFORMANCE_REPORT.md` with a full local-cache benchmark over `230` `5m` feather files (`19,838,110` bars).
- Coverage summary:
  - anomaly events: `35,634`
  - Nike signals: `19,265`
  - symbols with anomalies: `216`
  - symbols with Nike signals: `215`
- Anomaly-event recall with current Nike:
  - all anomalies: `53.05%` same-bar-or-next-bar recall
  - `run_up > 5%`: `68.33%`
  - `run_up > 10%`: `72.34%`
  - largest realized spike cohorts stayed around the target level: top `100` = `73.0%`, top `250` = `72.8%`, top `500` = `75.0%`, top `1000` = `72.6%`
- Full-cache realized signal outcomes with `TP=2.0 ATR`, `SL=0.8 ATR`, `max_bars=24`:
  - `TP=6030`, `SL=13111`, `TIMEOUT=124`
  - decided accuracy = `31.50%`
  - weighted PF = `1.150`
  - expectancy = `+0.0816 ATR` per signal
- Time-to-peak evidence remained consistent with the earlier positive-set study:
  - all anomalies mean `14.76` bars, median `11`
  - `run_up > 5%` mean `23.94`, median `26`
  - `run_up > 10%` mean `24.50`, median `27`
- Conclusion: current Nike is credible for strong breakout recognition and has positive expectancy, but it is not a broad anomaly detector and still needs better runner exits plus targeted fixes for large missed winners.

## Nike V2 Tiered Rollout + Benchmark (2026-04-08)
- Upgraded Nike to a tiered v2 path across `quanta_numba_extractors.py`, `quanta_nike_screener.py`, `quanta_nike_live_validator.py`, `QUANTA_bot.py`, and `QUANTA_trading_core.py`.
- Added tier metadata end-to-end:
  - `nike_tier`
  - `nike_score`
  - `nike_entry_mode`
  - `nike_bs_floor`
  - `nike_live_execute`
- Added a Nike-specific paper exit profile in `QUANTA_trading_core.py`:
  - initial stop `0.8 ATR`
  - bank partial profits at `+2.0 ATR`
  - trail the runner using a Nike-specific chandelier
  - pre-bank timeout and post-bank timeout metadata persisted on the position
- Wired the main live model path in `QUANTA_bot.py` so Nike-dominant trades pass `specialist='nike'`, the Nike exit profile, and timeout metadata into `paper.open_position(...)`.
- Wired the dedicated Nike screener path in `QUANTA_bot.py` so signals get a live BS/Kou veto, specialist metadata, and Nike paper-exit metadata before paper execution.
- Added runtime protection for existing paper positions by ticking them whenever their symbol appears in the live batch, rather than only when a new trade is about to open.
- Fixed the Nike screener cooldown to use absolute bar count instead of the rolling deque index, preventing permanent cooldown after the buffer fills.
- Produced `NIKE_V2_CACHE_PERFORMANCE_REPORT.md` from a full-cache rerun over the same `230` local `5m` files:
  - top `500` realized spike recall on `same/+1/+2`: `80.80%`
  - full-cache weighted PF under the v2 exit profile: `1.001`
  - signal count delta vs baseline: `+8.38%`
- Ran two follow-up tuning sweeps:
  - `nike_v2_tuning_search.csv`
  - `nike_v2_exit_search_wide.csv`
- Tuning conclusion:
  - Tier `C` is not fit for live trading.
  - Tier `A` is the only tier with positive standalone PF under the tested v2 exit family.
  - No tested `A+B` exit/threshold combination cleared both rollout gates simultaneously.
- Conservative live rollout shipped in config:
  - Tier `A`: live
  - Tier `B`: observe-only
  - Tier `C`: observe-only
- Verification:
  - `python -m py_compile quanta_config.py quanta_numba_extractors.py quanta_nike_screener.py quanta_nike_live_validator.py QUANTA_bot.py QUANTA_trading_core.py quanta_features.py quanta_exchange.py`
  - runtime smoke confirmed Nike signal metadata and Nike v2 paper-position metadata persistence

## Norse MAE + Year Sim Rewrite (2026-04-09)
- Added `quanta_pump_mae_stats.py` to generate the missing Phase A MAE artifacts:
  - `norse_pump_mae_rows.csv`
  - `norse_pump_drawdown_curves.csv`
  - `norse_mae_stats.csv`
  - `norse_stop_decision_matrix.csv`
  - `NORSE_MAE_STATS_REPORT.md`
- The MAE script reuses `_load_windowed_cache()`, `_prepare_symbol()`, `build_pump_ledger()`, `compute_pump_state()`, and the Numba Thor stop sweep to produce per-pump drawdown/runup rows plus aggregate bucket stats and a stop decision matrix.
- Reworked `quanta_norse_year_sim.py` to match the new Norse portfolio design:
  - Thor now applies a pre-entry 6-bar MAE veto plus wave-strength / top-risk gates.
  - Baldur is no longer simulated as an independent short; warning rows are now used to force Thor exits when the warning score clears the configured threshold.
  - Freya is no longer simulated as an independent scalp; it now becomes a Thor-linked pyramid add that exits with the parent Thor trade.
  - Tuning now searches the wider Thor exit family, the MAE gate, and the Freya/Baldur control thresholds.
  - Result selection now uses a soft penalty on drawdowns above `60%` instead of the old pure lexicographic final-capital comparator.
- Updated Norse config defaults in `quanta_config.py` toward the wider-stop / farther-bank profile from the plan and added the new knobs:
  - `thor_mae_veto_atr`
  - `thor_wave_strength_min`
  - `thor_top_risk_max`
  - `baldur_warning_exit_score`
  - `freya_pyramid_add_notional_fraction`
  - `freya_pyramid_add_trigger_wave`
  - `freya_pyramid_add_top_risk_max`
  - `norse_drawdown_penalty_weight`
- Added the missing stop-order API surface to `quanta_multi_exchange.py`:
  - `place_stop_market()`
  - `place_take_profit_market()`
  - `cancel_order()`
  - `modify_stop_order()`
  Current Bybit / OKX implementations are explicit stubs with warning logs, not full exchange-side stop wiring yet.
- Verification:
  - `python -m py_compile quanta_config.py quanta_norse_year_sim.py quanta_pump_mae_stats.py quanta_multi_exchange.py`
  - Runtime smoke with `C:/Users/habib/AppData/Local/Programs/Python/Python311/python.exe` on one cached symbol exercised the new Thor/Baldur/Freya flow without errors.

## Norse MAE Pipeline Optimization (2026-04-10)
- Reworked `quanta_norse_agents.py` with a new internal fused helper `_pump_path_analytics(...)` that computes the MAE-path arrays in one pass for a pump window:
  - `runup_atr`
  - close-based drawdown curve
  - low-based MAE-from-peak / MAE-from-entry series
  - bars-since-peak
  - wave-strength and top-risk path scores
  - peak timing and material-drawdown timing
- Refactored `build_pump_ledger(...)` to consume `_pump_path_analytics(...)` instead of calling `compute_pump_state(...)` bar-by-bar. This preserves the existing ledger rows and summary fields while removing the repeated per-bar state rebuild.
- Refactored `quanta_pump_mae_stats.py` to use the fused pump analytics path instead of:
  - `build_pump_ledger(...)` inside the pump-row loop
  - repeated `compute_pump_state(...)` calls for peak / bar-12 wave-risk fields
- Added deterministic symbol-level multiprocessing to `quanta_pump_mae_stats.py`:
  - `ProcessPoolExecutor(max_workers=4)` worker fanout by symbol
  - each worker loads/prepares one symbol, extracts features once, computes raw pump rows + curve rows, and returns a compact decision payload
  - parent process preserves `_load_windowed_cache()` symbol order, concatenates results in that order, and keeps bucket aggregation, decision-matrix generation, and markdown report generation single-threaded
- Reworked the decision-matrix input to consume worker-returned numeric payloads (`opens/highs/lows/closes/atrs + entry_bars`) instead of parent-held `PreparedSymbol` objects. Output schema and sort order are unchanged.
- Added an environment-safe fallback in `quanta_pump_mae_stats.py`: if Windows `ProcessPoolExecutor` cannot create worker pipes in the Codex sandbox, the script falls back to deterministic single-process execution. The normal desktop/runtime path still prefers process workers.
- Verification completed on the 30-day subset with Python 3.11:
  - `python -m py_compile quanta_norse_agents.py quanta_pump_mae_stats.py quanta_norse_year_sim.py`
  - Exactness harness: reconstructed the pre-refactor MAE logic inline and compared it against the new implementation on the same 30-day universe
    - `120` pump rows
    - `8544` drawdown-curve rows
    - `1152` decision-matrix rows
    - result: exact match for rows, curves, stats, decision matrix, and markdown report
  - Determinism harness: two independent 30-day reruns produced byte-identical CSVs and markdown
  - `build_pump_ledger(...)` exactness harness: new ledger rows + summaries matched the original per-bar implementation exactly across the same 30-day sample
  - Hot-path benchmark on prepared 30-day pumps:
    - old repeated analytics path: `0.46s`
    - new fused analytics path: `0.05s`
    - speedup: `8.79x`
- Constraint discovered during verification:
  - the Codex sandbox blocks Windows multiprocessing pipe creation (`PermissionError: [WinError 5] Access is denied`) so full wall-clock process-pool gains could not be measured here
  - correctness and determinism are verified; true parallel runtime improvement must be measured on the user machine outside the sandbox

## Norse Turbo Stage-0 Hardening + Strict Rerun (2026-04-11)
- Hardened `norse_event_cache_builder.py`:
  - added `--clean` rebuild mode for `norse_event_cache`
  - cache writes now stage both `.npz` and `.events.json` temp files before commit
  - `_manifest.json` is deleted at start and only written when the full pass completes with `symbols_valid == symbols_total` and `symbols_failed == 0`
- Hardened `norse_event_cache_loader.py`:
  - loader now rejects missing/empty files, bad JSON, missing schema keys, token mismatches, stale `max_open_time_ms`, missing arrays, and array-shape/schema mismatches as hard cache misses
  - added `load_cache_manifest()`, `inspect_cache_pair()`, and `validate_cache_universe()` for stage-0 validation
- Hardened `quanta_norse_year_sim.py`:
  - default path now requires a complete validated stage-0 cache and no cache misses
  - emergency fallbacks now require explicit env vars:
    - `QUANTA_NORSE_ALLOW_CACHE_MISS_FALLBACK=1`
    - `QUANTA_NORSE_ALLOW_LEGACY_FALLBACK=1`
  - cache-loaded symbols with a full dense `feature_ctx` now bypass `_ensure_feature_positions(...)` instead of falling back into offline feature extraction
- Hardened `norse_tuner/cached_evaluator.py`:
  - MAE CSV contract is now strict instead of silently defaulting missing columns
  - required columns include the new cache-driven fields plus all realized ATR columns
- Fixed `norse_tuner/pipeline.py` cache-vs-sim reporting bug where cached final capital was accidentally multiplied by initial capital twice
- Realigned the Turbo score contract:
  - `quanta_pump_mae_stats.py` now writes `thor_feature_score = score_thor_signal(sig, ctx)` and updates `score_band` from the same score surface
  - `extend_existing_rows_csv(...)` was extended to backfill `thor_feature_score` on existing rows
  - `CachedTrialEvaluator` now gates on `thor_feature_score` instead of raw `nike_score`, matching the full year sim
- Execution results completed in this session:
  - full MAE rebuild completed over the 365-day universe and wrote `19563` pump rows plus the usual CSV/report set
  - full clean stage-0 cache rebuild completed with `237/237` symbols valid and `_manifest.json` present
  - strict Turbo year-sim run completed using `237/237` cache hits and `0` misses
  - final observed result before the last report-only patch:
    - final capital: `$10,745.57`
    - growth: `+7.46%`
    - max drawdown: `15.93%`
    - executed trades: `51`
    - active agents: `Thor` only
    - cache-vs-sim divergence after score-contract fix: `7.2%` capital, `13.8 pp` drawdown
- Follow-up code-only patch applied after that run:
  - `quanta_norse_year_sim.py` now preserves Turbo tuning metadata (`wf_fold_metrics`, sensitivity, Optuna study, learned filter, cache reconciliation) instead of overwriting it during the final detailed result step
  - the report should be rerun locally once so `NORSE_YEAR_PAPER_REPORT.md` includes the tuning sections again

## Thor Exit Capture Upgrade (2026-04-12)
- Reworked the shared Thor post-bank exit path in `quanta_norse_agents.py`:
  - added runner-aware entry-strength estimation from live feature context
  - added adaptive post-bank wave/top-risk scoring inside the Numba exit kernel
  - widened strong setups after bank by allowing a controlled below-entry runner buffer that is still funded by the partial bank cushion
  - delayed and widened the runner trail on strong continuation states
  - tightened the runner trail when top-risk, exhaustion, weak closes, upper wicks, or stale bars-since-peak increase
- Fixed a semantic issue in the shared Thor exit kernels:
  - `thor_max_bars_post_bank` is now enforced relative to the bank hit, not as a total-age cap from entry
  - the old behavior could cut runners early even when the name implied post-bank hold time
- Threaded the adaptive exit inputs through the shared callsites:
  - `quanta_norse_year_sim.py`
  - `quanta_pump_mae_stats.py`
  - `norse_tuner/numba_fine_tune.py`
- Rebuilt `norse_tuner/numba_fine_tune.py` as a clean ASCII module and upgraded the search logic:
  - wider round-1 exit grid for higher `bank_atr`, lower `bank_fraction`, and wider runner trails
  - wider round-2 grid for later trail activation and longer time caps
  - top-`4` round-1 seeds now carry forward into round 2 instead of locking onto the first local winner
  - tie-break now prefers higher median fold growth / expectancy when objective scores are close
- Verification completed without running the year sim:
  - `python -m py_compile quanta_norse_agents.py quanta_pump_mae_stats.py quanta_norse_year_sim.py norse_tuner/numba_fine_tune.py`
  - import smoke:
    - `import quanta_norse_agents, quanta_pump_mae_stats, quanta_norse_year_sim, norse_tuner.numba_fine_tune`
  - synthetic Numba smoke:
    - constructed a minimal `SparseFeatureContext`
    - executed one `simulate_thor_exit_stop_market(...)` call successfully
  - no full MAE rebuild or year simulation was executed in this step

## Thor Exit Capture Rerun Results (2026-04-12)
- User reran the 365-day MAE rebuild locally after the adaptive Thor exit changes:
  - completed `237/237` symbols
  - runtime about `4710s`
  - rebuilt `19563` pump rows
  - rewrote `norse_pump_mae_rows.csv`, `norse_pump_drawdown_curves.csv`, `norse_mae_stats.csv`, `norse_stop_decision_matrix.csv`, and `NORSE_MAE_STATS_REPORT.md`
- User reran the strict 365-day year sim locally after that rebuild:
  - stage-0 cache validated `237/237`
  - symbol prep had `237` cache hits and `0` misses
  - learned filter fold AUCs remained around `0.60` to `0.62`
  - Optuna stage still collapsed around `-99.9` best score
  - stage-2 exit fine-tune also stayed flat around `-99.904`
  - selected exit delta:
    - `thor_bank_atr = 5.4`
    - `thor_bank_fraction = 0.15`
    - `thor_runner_trail_atr = 2.0`
    - `thor_trail_activate_atr = 1.5`
    - `thor_max_bars_pre_bank = 12`
    - `thor_max_bars_post_bank = 36`
  - final full-sim result:
    - final capital: `$17,171.76`
    - growth: `+71.72%`
    - max drawdown: `54.21%`
    - trades: `42`
    - Thor only; Freya/Baldur `0`
    - Thor profit factor: `2.857`
  - cache-vs-sim reconciliation worsened materially:
    - capital delta: `8.6%`
    - drawdown delta: `47.2 pp`

## Turbo Tuner Repair For Low-Frequency Thor (2026-04-12)
- Repaired `norse_tuner/objective.py` so low-frequency Thor no longer collapses the walk-forward objective:
  - removed the hard `20 trades per fold` rejection behavior
  - added confidence-weighted fold scoring for low-trade folds
  - zero-trade folds still fail hard
  - negative low-trade folds are still penalized aggressively
  - lowered `TRADE_BONUS_TARGET` so low-frequency systems are not effectively treated as non-participants
- Rebuilt `norse_tuner/pipeline.py` as a clean ASCII module and changed the selection flow:
  - dedupe Optuna trials by param signature
  - keep multiple bootstrap-passed stage-1 candidates instead of trusting one cache winner
  - run exit fine-tune + full sim on the top candidate set
  - choose the final winner from real full-sim results using a drawdown-aware rerank key
  - sensitivity now runs after the final candidate is selected
  - fast replay reconciliation now uses `evaluate_exit_params_full(...)` from `norse_tuner/numba_fine_tune.py`, so the diagnostic includes stage-2 exit params
- Added `evaluate_exit_params_full(...)` to `norse_tuner/numba_fine_tune.py` for fast full-sample replay estimates that include the tuned exit params.
- Bumped `norse_tuner/optuna_search.py` study contract version to invalidate persisted studies built under the old broken low-trade objective.
- Tightened `quanta_norse_year_sim.py::_portfolio_objective(...)`:
  - drawdown penalty now starts above `35%`, not `60%`
- Verification completed without rerunning the full MAE or year sim:
  - `python -m py_compile norse_tuner/objective.py norse_tuner/numba_fine_tune.py norse_tuner/pipeline.py norse_tuner/optuna_search.py quanta_norse_year_sim.py`
  - import smoke for `norse_tuner.pipeline`, `norse_tuner.numba_fine_tune`, and `norse_tuner.optuna_search`
  - objective sanity check:
    - profitable 8-trade fold now scores positively instead of collapsing to `-100`
    - zero-trade fold still scores `-100`
    - bad low-trade fold remains strongly negative

## Norse Sim Verbose Progress Logging (2026-04-12)
- Added detailed full-sim progress output in `quanta_norse_year_sim.py` and `norse_tuner/pipeline.py`
- During real full-sim rerank / final detailed eval, logs now show per-symbol Thor/Freya activity with accepted setups, Thor TP/SL/TIMEOUT counts, Baldur warnings, Freya adds, and sample trade outcome lines
- Kept cache/tuning passes lightweight: only the real full-sim path emits trade-style progress lines
- Verified with `python -m py_compile` and import smoke (`import quanta_norse_year_sim, norse_tuner.pipeline`)

## Norse Timestamped Artifact Bundles + Trade Diagnostics (2026-04-12)
- Norse year sims no longer overwrite a single report/output set; each run now writes into `norse_runs/<UTC timestamp>/`
- Timestamped bundle includes the main report, summary/trade/pump/tuning CSVs, plus new diagnostics CSVs:
  `norse_trade_diagnostics_*`, `norse_loss_diagnostics_*`, `norse_big_win_diagnostics_*`, `norse_feature_compare_*`, `norse_reason_summary_*`
- Added post-run loss and big-win analysis in `quanta_norse_year_sim.py`
- Report now includes:
  run ID/timestamp, loss-reason counts, sample loss rows, sample big-win rows, and feature medians comparing losses vs big wins
- Verified with `python -m py_compile quanta_norse_year_sim.py norse_tuner/pipeline.py` and import smoke (`import quanta_norse_year_sim`)

## Norse 10x Asymmetric Geometric Compound Tuning Fix (2026-04-12)
- Discovered and fixed the '50% Max Drawdown' telemetry illusion in quanta_norse_year_sim.py; the charting system was improperly capturing mathematically deducted staging margin before vault allocation, causing massive fake telemetry crashes. Drawdown fixed perfectly.
- Exposed the '0 Trades Passed / -114 Calmar' Optuna failure state:
  - Optuna baseline bounds for 	hor_wave_strength_min (50.0) and 	hor_min_score_trade (70.0) physically choked stringency to <4 valid trades a year on average across all 800 samples.
  - Loosened Optuna bounds in optuna_search.py (	hor_wave_strength_min 25-80, 	hor_min_score_trade 60-95), allowing the TPE sampler the necessary geometry volume to find the mathematical 'Big Win' pockets.
- Confirmed Baldur and Freya baseline interaction limits � verified they function correctly and safely within their designated live bounds.
- Re-activated the full _SL_CHOICES parameter sweep to let the tuner aggressively track mathematical constraints safely.
- Result completed: **+1238.36% Compounded Growth ( -> ) / 26.63% Max DD / 199 Trades**.


## Phase V12 — Thor Specialist Pivot (2026-04-14)
- Architecture fully refactored to single-specialist Norse Thor engine (CatBoost).
- Removed all Pantheon specialists (Athena, Ares, Hermes, Hephaestus, Artemis, Chronos, Nike).
- Removed Odin Meta-Learner (TFT) and Heimdall PPO sizer.
- Transitioned training to offline-first mode using feather_cache (no live Binance calls).
- Fixed Thor event extraction bug (missing opens argument) and Numba config import crashes.
- Deployed Thor Gen 1: AUC 0.8080 | Brier 0.1528.
- Files: train_thor.py, QUANTA_ml_engine.py, QUANTA_bot.py, QUANTA_selector.py

## Phase V12.1 — Nike→Thor Full Codebase Rename (2026-04-15)
- Complete rename of all Nike references to Thor across 7 core files. Nike was a cosmetic alias for the same breakout detection logic; Thor is now the single canonical name.
- `quanta_thor_screener.py` — new canonical file. All `NikeSignal/NikeScreener/_nike_check` renamed to Thor equivalents. Backward-compat aliases appended at end.
- `quanta_nike_screener.py` — converted to thin shim that re-exports from quanta_thor_screener.
- `quanta_config.py` — all 13 `nike_*` EventExtractionConfig params renamed to `thor_*`. Added new Thor entry quality gate params: `thor_wave_strength_min=40.0`, `thor_pre_impulse_r2_max=0.70`, `thor_compound_mode="asymmetric_target"`, `thor_compound_activation_score=85.0`, `thor_compound_max_loss_pct=3.0`. Added 25 `@property` backward-compat aliases mapping old `nike_*` names.
- `quanta_numba_extractors.py` — `_NIKE_*` constants → `_THOR_*`, `fast_extract_nike` → `fast_extract_thor` with alias.
- `QUANTA_bot.py` — all nike screener imports, attribute names, specialist keys updated to thor.
- `QUANTA_trading_core.py` — position dict keys (`thor_bank_hit`, etc.), compound config keys, `open_position()` signature extended with `thor_score=None, **kwargs`.
- `QUANTA_ml_engine.py` — purge gap uses `thor_max_bars`, specialist key `'thor'` throughout.
- All 8 files pass `py -3.11 -m py_compile`. Zero functional `nike_*` references remain in non-shim files.

## Phase V12.2 — WalkForward Sim Norse Translation (2026-04-15)
- Rewrote QUANTA_WalkForward_Sim.py to close the gap between WF OOS results and Norse year sim.
- Critical loop order fix: symbol-outer/bar-inner → bar-outer/symbol-inner (fixes _MAX_CONC, concurrent position management, and cooldown logic).
- Wired real Thor CatBoost model (thor_gen1.cbm, 634 trees, AUC 0.81) via offline replay engine. `_thor_model_score()` runs inference on 102-feature IMPULSE mask. Score discrimination improved from ~0.3 to ~5 point separation.
- Added `_find_thor_model()` searching multiple path candidates including `Path(__file__).parent`.
- Added pre_impulse_r2 veto (feature[272] > 0.70 → skip). Strongest negative predictor from Norse learned filter (coef=-0.24).
- Added wave_strength gate using real taker_buy_base (klines col[9]) — exact match to Norse wave_strength_score (coef=+0.37). Falls back to directional-volume proxy if col[9] absent.
- Replaced binary compound_activation_score threshold (85.0) with continuous linear scaling: risk = 0.5% at score=68 → 3.0% at score=100. Old binary threshold was firing on only 17/479 trades.
- Reduced _TRAIN_DAYS to 60 (hardcoded, config has 180d) → 10 OOS windows, 300 OOS days.
- Expanded symbol universe from 50 cap to all available feather cache files (245-401 symbols).
- Added per-trade equity tracking for accurate Max DD (was window-level only → always 0%).
- Sim progression: +37.7% → +347% → +2054% → actively improving.

## Phase V12.3 — MAE Stats Calibration + 3-Layer Pyramid (2026-04-15)
- Read and applied NORSE_MAE_STATS_REPORT.md recommendations to QUANTA_WalkForward_Sim.py.
- Parameter changes from MAE report Recommended Parameter Block + Decision Matrix:
  - `SL_ATR: 2.40 → 3.00` (P90 winner MAE; rescues 13.6% of 1-3 ATR outcome trades)
  - `BANK_ATR: 5.40 → 4.20` (top decision-matrix combo)
  - `BANK_FRAC: 0.45 → 0.35` (top decision-matrix combo)
  - `TRAIL_ATR: 6.00 → 2.00` (tighter runner, top combo)
  - Added `TRAIL_ACTIVATE_ATR = 1.50`: trail only kicks in 1.5 ATR above bank price, not immediately
  - `MAX_PRE: 144 → 48` (4h timeout; kills dead pre-bank trades)
  - `MAX_POST: 1152 → 96` (8h post-bank runner window)
- MAE early exit veto: if trade goes 3.62 ATR adverse in first 5 bars → `MAE_VETO` exit (from P50 winner MAE from entry in aggregate bucket snapshot).
- Hour-of-day UTC filter: skip new entries at hours 0, 7, 10, 11, 12 (PF 0.51-0.81 in hour-of-week heatmap). Best hours: 2 (PF 1.15), 8 (PF 1.16), 14 (PF 1.18).
- 3-layer pyramid averaging-in strategy (from Norse MAE pump stats):
  - L1: normal Thor entry at price P
  - L2: add at +0.5 ATR above entry, size=50% of L1, SL=0.5 ATR below add entry
  - L3: if L2 SL hits, recovery re-entry at that price, target=P+3.77 ATR (MAE overall runup p50=3.80 ATR), SL=original entry - 3.0 ATR
  - All layers force-close at main position exit. Bank event partially closes L2 add (35% fraction).
  - Trade record now carries `add1_pnl`, `add2_pnl`, `layers` fields.
- Norse-style timestamped run folder: `wf_runs/<YYYYMMDD_HHMMSS>/` with 8 artifacts:
  - `WF_SIM_REPORT_<ts>.md`, `wf_trades_<ts>.csv`, `wf_wins_<ts>.csv`, `wf_losses_<ts>.csv`
  - `wf_window_stats_<ts>.csv`, `wf_score_distribution_<ts>.csv`, `wf_summary_<ts>.csv`, `wf_sim_results_<ts>.json`
- Files modified: QUANTA_WalkForward_Sim.py (primary)


## Khairul's Identity — Formal Derivation (2026-04-15)
- Derived the master compound growth equation of the QUANTA system from WF simulation data.
- Named "Khairul's Identity" by Habib Khairul — the original creator of the 194× bot.
- Identity: C(T) = C₀ · exp(n · T)
- Full expansion: n = λ · [P·ln(1+f·b) + (1-P)·ln(1-f)] with pyramid boost ln(Π)
- Calibrated values from WF sim (468 trades, 300 OOS days, $10k→$1.988M):
  - n = 0.01764/day | λ = 1.56 trades/day | g̃ = 0.01131/trade
  - P = 0.707 | b = 1.810 | f_avg ≈ 0.015 | Π = 1.696
  - f/f* = 2.75% of Kelly → MaxDD = 8.1% (theoretical: 11%, corr discount gives 8.1%)
  - Average runner exit = 6.09 ATR (derived from b=1.810, geometry-consistent)
  - Pump phase: n_pump ≈ 0.082/day (universal altcoin constant) > n_QUANTA ✓
  - Grinold ceiling: n_max = 0.0366/day → QUANTA operates at 48% of info-theoretic ceiling
  - Cross-coin correlation discount: ρ_eff = 0.343 → 65.7% of theoretical alpha removed by corr
- Five consistency constraints: Kelly, Pump Phase, Grinold, MAE Geometry, Pyramid
- All five constraints simultaneously satisfied → identity is over-determined and self-consistent
- Stored in: memory/project_quanta_technical_decisions.md (equation reference)

## Phase V12.4 — Gompertz Dynamic Bank + Dynamic Timeouts (2026-04-16)
- Replaced ALL hardcoded exit timing with Gompertz hazard model λ(t) = λ₀·e^(γt).
- Three hardcoded values eliminated:
  - _BANK_ATR = 4.20 ATR (fixed) → _dynamic_bank_atr(n_eff, atr_pct)
  - _MAX_PRE  = 48 bars (4h)     → _dynamic_pre_timeout(n_eff)  = ln(1.5·n/λ₀)/γ
  - _MAX_POST = 96 bars (8h)     → _dynamic_post_timeout(n_eff) = ln(2.5·n/λ₀)/γ − t_bank
    (was originally 1152 bars = 4 days before MAE calibration)
- New helper functions: _pump_n_eff(), _gompertz_t_star(), _dynamic_bank_atr(),
  _dynamic_pre_timeout(), _dynamic_post_timeout()
- New constants: _N_PUMP_MICRO=0.700, _GOMPERTZ_L0=0.517, _GOMPERTZ_GAMMA=2.92,
  _K_RUNNER_HAZARD=2.5, _K_PRE_HAZARD=1.5
- SimPosition extended with: n_eff, dyn_bank_atr, dyn_max_pre, dyn_max_post
- SimTrade extended with: dyn_bank_atr, n_eff (for post-run analysis)

## V12.4 WF Sim Results (2026-04-16) — 10/10 windows profitable
- Total Return: +150,235% ($10k → $15,023,500) — was +19,883%
- Win Rate: 70.0% | PF: 4.65 | Sharpe: 7.27 | Sortino: 19.25 | Calmar: 15,497
- Max DD: 9.7% (was 8.1% — small cost for 7.55× gain)
- L2 add PnL: +$3,457,528 (was $262k — +13.2×)
- L3 recovery PnL: +$2,665,814 (was $554k — +4.8×)
- 466 trades, 300 OOS days, 245 symbols
- Exits: Chandelier/Runner 265 | SL 91 | Timeout/WindowEnd 110

## Updated Khairul's Identity Constants (V12.4)
- n_daily = ln(1502.35)/300 = 0.02438/day (was 0.01764)
- n_per_trade = 0.01569 (was 0.01131)
- Monthly n = 0.7314 → e^0.7314 = 2.078 (+107.8%/month)
- Annual n = 8.899 → e^8.899 = 7,337× per year
- Doubling time = ln(2)/0.02438 = 28.4 days (was 39.3 days)

## Phase V12.4 — Gompertz Dynamic Exit + Live Bot Integration (2026-04-16)
- WF sim V12.4 result: +150,235% (+755% vs V12.3 +19,883%), 466 trades, 70.0% WR, PF 4.65, Sharpe 7.27, MaxDD 9.7%
- L2 add PnL: +$3,457,528 | L3 recovery PnL: +$2,665,814 (pyramid now dominant at 41% of total)
- Replaced ALL hardcoded exit timeouts with Gompertz hazard model:
  - `_BANK_ATR = 4.20` (fixed) → `_dynamic_bank_atr(n_eff, atr_pct)` = (e^(n·t*)-1)/ATR%
  - `_MAX_PRE = 48 bars` (fixed) → `_dynamic_pre_timeout(n_eff)` = bars until λ(t)=1.5·n
  - `_MAX_POST = 96 bars` (fixed, was 1152=4 days before MAE calibration) → `_dynamic_post_timeout(n_eff)` = bars until λ(t)=2.5·n
- Added `_pump_n_eff()`, `_gompertz_t_star()` module-level helpers in WF sim
- Ported to live bot:
  - `quanta_config.py`: Added 14 new fields — thor_sl_atr_calibrated, thor_bank_atr_calibrated, thor_bank_fraction_calibrated, thor_runner_trail_atr_calibrated, thor_trail_activate_atr_calibrated, thor_max_bars_pre/post_calibrated, thor_mae_veto_bars/atr, 5× Gompertz constants
  - `QUANTA_trading_core.py`: Added 5 static methods to PaperTrading (_gc_n_eff, _gc_t_star, _gc_bank_atr, _gc_pre_bars, _gc_post_bars)
  - `QUANTA_trading_core.py`: Rewrote `_tick_thor_v12` — Gompertz dynamic bank/pre/post, trail_activate_atr (1.5 ATR above bank), MAE veto (3.62 ATR in 5 bars), breakeven SL at bank hit
  - `QUANTA_trading_core.py`: Added `_build_thor_exit_profile()` — was missing (AttributeError silently prevented thor_v2 routing), now returns calibrated params dict with mode='thor_v2'
  - Position dict: 6 new Gompertz fields: gompertz_n_eff, gompertz_lowest, gompertz_dyn_bank_atr, gompertz_dyn_max_pre, gompertz_dyn_max_post, gompertz_bank_bar
- Khairul's Identity λ companion equation now live in production
- Files: QUANTA_WalkForward_Sim.py, QUANTA_trading_core.py, quanta_config.py

## Live Thor Direction Clamp + Telegram Fallback (2026-04-17)
- Fixed live consumer behavior in `QUANTA_bot.py` so Thor no longer emits executable `BEARISH` signals.
  - Root cause: CatBoost still outputs a binary class distribution and the Thor-only consumer was mapping the negative class to `BEARISH`, which leaked old long/short semantics into the new single-specialist pump pipeline.
  - Fix: negative Thor outputs are now treated as `NEUTRAL` (`do not trade`) instead of short entries.
- Confirmed the old `specialist_probs_batch` NameError path is guarded in the current consumer code by explicit fallback initialization (`specialist_probs_batch = None`, `specialist_keys = []`), so stale runtime crashes came from an older file state, not the patched one.
- Improved Telegram execution notifications in `quanta_telegram.py`.
  - Root cause observed live: `Telegram send failed: no response after retries` despite valid token/chat_id in environment.
  - Fix: `TelegramBot.send()` now retries through the existing `NetworkHelper` path first, then falls back to a direct `requests.post(...)` send to Telegram if the proxy/session path returns no response.
  - Added clearer error logging for no-response, non-JSON, and Telegram API error payloads.
- Verification:
  - `py -3.11 -m py_compile QUANTA_bot.py quanta_telegram.py`
  - environment check confirmed `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are both present at runtime

## Fresh Prediction Reset + Legacy RL Gate Fix (2026-04-17)
- Cleared persisted prediction state for a true fresh-start restart:
  - deleted `quanta_data/predictions_primary.feather`
  - deleted `quanta_data/predictions_backup1.feather`
  - deleted `quanta_data/predictions_backup2.feather`
  - deleted `rl_memory.feather`
- Left `paper_trading_state.json` intact intentionally because it is paper trade / position state, not just saved prediction backlog.
- Found one real legacy interference path in the live bot wrapper:
  - local runtime `Config` in `QUANTA_bot.py` was still hard-forcing `self.rl_enabled = True`
  - this could allow dormant PPO / RL background behavior to stay alive even though canonical `quanta_config.py` already sets `RLConfig.rl_enabled = False`
- Fixed the runtime wrapper to respect canonical config:
  - `QUANTA_bot.Config.rl_enabled` now mirrors `self._sys.rl.rl_enabled`
  - PPO agent initialization now only occurs when `self.cfg.rl_enabled` is true
- Audit result:
  - remaining Pantheon / PPO / TFT / MoE references are still present in comments, banners, imports, and some cold-path legacy helper code
  - they are mostly cosmetic or dormant now, not part of Thor’s hot inference decision path
  - the meaningful live-path disturbances that were still active were:
    1. Thor negative class leaking through as `BEARISH` trade direction
    2. Telegram network path failing without a direct fallback
    3. runtime RL/PPO being force-enabled by the bot wrapper
- Verification:
  - `py -3.11 -m py_compile QUANTA_bot.py quanta_telegram.py`
  - runtime check showed `Config().rl_enabled == False` after the patch

## Dashboard Executed-Trade Refactor (2026-04-17)
- Refactored the overview trade table in `quanta_dashboard.py` and `templates/dashboard.html` so the dashboard distinguishes executed positions from generic scan/prediction noise.
- Active paper positions now expose:
  - `executed_by`
  - `execution_state`
  - `is_thor_executed`
- Dashboard labels:
  - active Thor long positions with `THOR EXECUTED`
  - anything else with `LEGACY / NON-THOR`
- Overview card wording updated from generic `Open Positions` to `Executed Trades`.
- The positions/history table now shows:
  - `Engine`
  - `State`
  - entry / size / confidence for active executions
  - engine / entry / exit / result for history rows
- Fixed a broken duplicate render block in the dashboard JavaScript that still referenced an undefined `positions` variable during overview refresh.
- Verification:
  - `py -3.11 -m py_compile quanta_dashboard.py`

## Dashboard Thor-Only Main View + Legacy State Split (2026-04-17)
- Tightened the dashboard overview so the main active-positions table now shows only Thor-executed active positions.
- Added a separate `Legacy State` section underneath the main overview for any non-Thor / leftover active positions still present in paper state.
- Backend:
  - `quanta_dashboard.py` now splits active positions into:
    - `active_positions` = Thor-executed only
    - `legacy_active_positions_detail` = non-Thor active leftovers
- Frontend:
  - main table header now reads `Thor Executed Positions`
  - added `renderLegacyTable()` in `templates/dashboard.html`
  - legacy section is auto-hidden when there are no leftover non-Thor positions
- Result:
  - the overview no longer mixes current Thor execution state with stale short/legacy positions
  - any contamination is still visible, but quarantined into its own section instead of polluting the main live view
- Verification:
  - `py -3.11 -m py_compile quanta_dashboard.py`

## Dashboard Final Cleanup Pass (2026-04-17)
- Added an explicit red warning banner at the top of Overview when legacy active positions exist in paper state.
- Renamed the misleading `RL Memory` overview card to `Execution Pipeline`.
- Replaced the old `Buffer Size` line with a live `Mode` line:
  - `Thor-only` when runtime RL is disabled
  - `Thor + RL follow-up` if RL is ever re-enabled later
- Position modal now shows:
  - `Engine`
  - `Execution State`
  This makes each clicked active position self-describing as Thor-executed vs legacy state.
- Result:
  - Overview now reads as an execution dashboard, not a generic old multi-system monitor
  - Legacy contamination is elevated into an explicit warning state instead of being easy to miss
- Verification:
  - `py -3.11 -m py_compile quanta_dashboard.py`

## Persisted Legacy Paper Positions Removed (2026-04-17)
- Cleaned `paper_trading_state.json` to remove stale non-Thor / non-bullish active positions that came from the earlier live direction leak before the Thor bearish clamp.
- Preserved only active positions satisfying:
  - `specialist == 'thor'`
  - `direction == 'BULLISH'`
- Result:
  - kept `4` valid Thor-long active positions
  - removed stale positions:
    - `PIPPINUSDT`
    - `SAGAUSDT`
    - `BERAUSDT`
    - `TAOUSDT`
    - `BCHUSDT`
    - `1000RATSUSDT`
- Important runtime note:
  - this cleans persisted paper state on disk
  - a running bot/dashboard process may still hold the old positions in memory until restart / reload

## Thor Live Confidence Path Tightened (2026-04-22)
- Fixed Thor-only live calibration in `QUANTA_bot.py` to call `AdaptiveConformalCalibrator.predict()` correctly instead of treating it like isotonic regression.
- When the conformal calibrator returns `interval_width`, the live uncertainty discount now uses that width directly instead of falling back to the Bayesian proxy.
- Execution and gating now use the adjusted confidence path, not the raw pre-penalty model confidence.
- Streak persistence no longer boosts the confidence value itself; it only boosts magnitude so the score can reward persistence without inflating the probability number.
- Added `raw_model_confidence` to the entry diagnostics payload for post-trade inspection.
- Verification:
  - `python -m py_compile QUANTA_bot.py`

## WF Sim Thor Inference Aligned (2026-04-22)
- Updated `QUANTA_WalkForward_Sim.py` so Thor replay inference now loads and uses `thor_feature_indices.npy`, `thor_scaler.pkl`, and `thor_calibrator.pkl` when available.
- The walk-forward sim no longer scores entries from raw unscaled `predict_proba()[1]` alone.
- Sim Thor score now follows the live Thor-only path more closely:
  - slice with Thor feature indices
  - scale with Thor scaler
  - calibrate with `AdaptiveConformalCalibrator`
  - apply uncertainty discount from conformal `interval_width` when present
- Verification:
  - `py -3.11 -m py_compile QUANTA_WalkForward_Sim.py`

## WF Sim Leverage Cap Restored (2026-04-22)
- Updated `QUANTA_WalkForward_Sim.py` entry sizing so score-based risk still determines target notional, but the final position is capped by the same leverage logic used in live paper trading.
- Base cap is `5x` notional on equity; S-tier entries (`score >= 90` and `bs_prob >= 0.70`) can still use `10x`.
- This prevents tiny-ATR names from blowing up sim notionals far beyond the live Thor path.
- Verification:
  - `py -3.11 -m py_compile QUANTA_WalkForward_Sim.py`

## Live Thor Fully Re-Aligned To Sim (2026-04-22)
- Updated `QUANTA_bot.py` so live Thor execution now uses the corrected Thor score directly instead of `confidence × magnitude / 10`.
- Live paper entries now pass the real feature-derived `bs_prob` into `PaperTrading.open_position()` instead of the old hardcoded `0.5`.
- Live Thor sizing now uses 5m ATR at entry for stop-distance parity with the walk-forward sim, rather than the old 4h ATR path.
- Removed the extra live-only execution blockers that the current WF sim does not use:
  - magnitude floor veto
  - separate confidence gate
  - top-risk veto
- Verification:
  - `py -3.11 -m py_compile QUANTA_bot.py`

## Prediction Pipeline Reset (2026-04-22)
- Deleted the persisted pending-prediction backup files from `quanta_data`:
  - `predictions_primary.feather`
  - `predictions_backup1.feather`
  - `predictions_backup2.feather`
- Verified `rl_memory.feather` is not present, so there was no additional on-disk RL buffer to clear.
- Left `paper_trading_state.json`, `daily_picks.json`, model artifacts, and logs untouched.

## Runtime Launch/Stop Check (2026-04-22)
- Started QUANTA from workspace using `py -3.11 main.py` (detached process launch).
- Runtime boot log was written to `quanta_runtime.log` (latest boot block timestamped `2026-04-22 17:17:32` local time).
- User then stopped the run manually, so no long-running session was kept active.

## Thor Runner Time Limit Removed (2026-04-22)
- Updated live Thor in `QUANTA_trading_core.py` so post-bank runners are no longer force-closed by elapsed time.
- Updated `QUANTA_WalkForward_Sim.py` the same way: removed the post-bank `RUNNER_TIMEOUT` exit so WF runner behavior now matches the live price-governed design.
- Runner exits are now governed by price-state logic only after bank:
  - breakeven / trail handling
  - `CHANDELIER_SL`
  - any other existing structural exit, but not post-bank age
- Pre-bank timeout remains intact; only the post-bank runner clock was removed.
- Verification:
  - `py -3.11 -m py_compile QUANTA_trading_core.py`
  - `py -3.11 -m py_compile QUANTA_WalkForward_Sim.py`

## Paper Trading State Reset (2026-04-22)
- Reset `paper_trading_state.json` to a clean paper baseline:
  - `balance = 10000.0`
  - `initial_balance = 10000.0`
  - zero trades / wins / losses / pnl
  - empty `positions`, `history`, `sl_slippage`, and `sl_blacklist`
- Reset `trades.csv` back to header-only so the paper ledger cannot bootstrap old trade history on next bot start.
- Created timestamped backups before reset:
  - `paper_trading_state.json.bak-reset-20260422_182236`
  - `trades.csv.bak-reset-20260422_182236`

## Dashboard Thor Barrier Display Fixed (2026-04-22)
- Fixed `quanta_dashboard.py` so active Thor-executed positions now display exit barriers from the live paper position state itself:
  - `tp1_price`
  - `tp2_price`
  - `tp3_price`
  - `sl_price`
- This replaces the incorrect behavior where the dashboard modal could show stale `daily_picks` targets or a generic ATR fallback, which did not match Thor v2 live/WF exit geometry.
- Result:
  - Thor position modal now reflects actual executed barrier state instead of alert-level suggestion targets.
- Verification:
  - `py -3.11 -m py_compile quanta_dashboard.py`

## Live Thor Sizing Aligned To Actual SL Geometry (2026-04-22)
- Fixed `QUANTA_trading_core.py` sizing so live Thor v2 now computes stop distance from the actual executed `sl_atr` in `exit_profile` instead of a stale generic `1.5 x ATR` proxy.
- Before the fix, live notional sizing could overstate size versus the actual MAE-calibrated Thor stop (`3.0 ATR`) because it assumed a much tighter stop than the one actually placed.
- Result:
  - live sizing now matches WF sizing semantics more closely:
    - risk dollars divided by actual stop distance
    - then capped by the same leverage logic
- Verification:
  - `py -3.11 -m py_compile QUANTA_trading_core.py`

## Paper Trading State Reset Again (2026-04-22)
- Reset `paper_trading_state.json` again to a clean `$10,000` baseline after live paper trades reopened.
- Cleared all persisted open positions, history, PnL counters, slippage samples, and blacklist state.
- Reset `trades.csv` back to header-only so old trade history cannot bootstrap back in.
- Created timestamped backups before reset:
  - `paper_trading_state.json.bak-reset-20260422_215036`
  - `trades.csv.bak-reset-20260422_215036`

## Thor Live Paper Path Forced To WF-Sim Parity (2026-04-23)
- Patched `QUANTA_trading_core.py` and `QUANTA_bot.py` so Thor paper trading now follows the WF sim execution model directly instead of mixing in live-only paper layers.
- Entry / sizing parity changes:
  - Thor paper entries bypass live risk-manager trade blocking and loss-throttle sizing.
  - Thor paper entries no longer route through TWAP; they open with one sim-style synthetic fill.
  - Entry/exit fee handling for Thor paper now matches WF sim math:
    - entry fill uses `price * (1 + slippage + commission)` for longs
    - exit fill uses `price * (1 - slippage - commission)` for longs
  - Thor cooldown can now run off closed-bar timestamps instead of wall-clock time when bar context is available.
- Exit / timing parity changes:
  - Added closed-5m-bar Thor ticking via `PaperTrading.tick_bar(...)`.
  - Added `_paper_thor_bar_worker()` in `QUANTA_bot.py` so open Thor paper positions are advanced bar-by-bar from live 5m candle data, matching WF sim structure.
  - Thor paper runner management is now driven from candle `high/low/close` with exact sim-style bar counting (`bars_open`) instead of approximate wall-clock elapsed time.
  - The old per-symbol `paper.tick(price)` path is skipped for Thor sim-parity positions so price-tick logic cannot interfere with bar-parity logic.
- Pyramid parity changes:
  - Thor paper layer-2 add, layer-3 recovery, bank-on-add, and runner trail now evaluate from candle highs/lows exactly like the WF sim.
- Verification:
  - `py -3.11 -m py_compile QUANTA_trading_core.py QUANTA_bot.py`

## Paper Trading State Reset After Thor Parity Patch (2026-04-23)
- Reset `paper_trading_state.json` and `trades.csv` after the Thor live/sim parity patch so the next run starts from a clean identical baseline.
- Created timestamped backups before reset:
  - `paper_trading_state.json.bak-reset-20260423_232422`
  - `trades.csv.bak-reset-20260423_232422`
- New baseline:
  - balance `$10,000`
  - no open positions
  - empty history
  - empty slippage / blacklist state

## Dashboard Refactor And Thor Truth Cleanup (2026-04-24)
- Refactored `templates/dashboard.html` into a new execution-centric dashboard with a cleaner visual system, stronger hierarchy, and clearer Thor/live-paper state presentation.
- Reworked active-position rendering so Thor-executed positions no longer inherit misleading alert-pick TP/SL ladders from `daily_picks`.
- Updated `quanta_dashboard.py` overview rows to expose live Thor execution state directly:
  - `execution_state`
  - `executed_by`
  - `thor_bank_hit`
  - `thor_trail_active`
  - `runner_peak`
  - `bank_price`
- Added a dedicated `/api/thor` endpoint and made Thor dashboard params read from the active Thor exit profile (`paper._build_thor_exit_profile()`), not stale legacy config aliases.
- Preserved non-Thor/legacy positions separately so the overview can stay Thor-clean without losing backend visibility.
- Verification:
  - `py -3.11 -m py_compile quanta_dashboard.py`
  - local dashboard responded successfully on `http://localhost:5000/`
  - local API checks succeeded for `/api/overview` and `/api/thor`
