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
