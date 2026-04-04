---
name: QUANTA Technical Decisions & Critical Facts
description: Important design decisions, known gotchas, and critical implementation facts for QUANTA
type: project
---

## Critical Constants (quanta_config.py)
- `base_feature_count = 278` — single source of truth. All code reads `self.cfg.BASE_FEATURE_COUNT`. Never hardcode.
- `get_sentiment_features()` returns 7 features (fng_norm, extreme_fear, extreme_greed, news_score, news_volume_norm, coin_score, coin_magnitude)
- Commission: 4bps (0.0004) per trade
- Slippage: 2bps (0.0002) per trade
- Triple Barrier (Athena defaults): TP=1.5 ATR, SL=1.0 ATR, max_bars=48. Per-specialist values in `EventExtractionConfig` (nike: TP=2.0, SL=0.8; hermes: TP=1.0, SL=0.8; artemis: TP=2.0, SL=1.0)
- Purge gap = max(all agent max_bars) = 48 candles (~4h at 5m)
- Val split = 20% (CATBOOST_VAL_SPLIT)
- GPU_MAX_SAMPLES = 10,000 (MX130 2GB VRAM)
- CPU_CHUNK_SIZE = 5,000

## Config Field Name Gotcha
- `EventExtractionConfig` accessed as `self.cfg.events` — NOT `self.cfg.event_extraction`
- `self.cfg.event_extraction` caused AttributeError blocking all 7 agents (fixed 2026-03-31)

## Label Standardization
- Label 1 = BULLISH outcome
- Label 0 = BEARISH outcome
- Label -1 = invalid (no barrier hit, insufficient data) → excluded from training

## Thread Safety Rules
- `FeatherCache`: per-symbol write lock via `_write_locks_meta` meta-lock + `_write_locks` dict
- `ModelRegistry`: RLock (re-entrant), atomic save via `os.replace(tmp → final)`
- `RiskManager`: RLock (re-entrant — `get_size_multiplier` called inside `on_trade_closed`)
- `PaperTrading`: standard Lock

## flat_all() Pattern
- `paper.flat_all(price_lookup: dict = None)` — uses entry price as fallback
- Called on: SIGINT/SIGTERM, KeyboardInterrupt, circuit breaker trigger
- Wired as `flat_all_callback` in RiskManager constructor

## _raw_candles Format — Critical
`_raw_candles` is a DICT: `{'closes': array, 'highs': array, 'lows': array, 'volumes': array, 'taker_buy': array_or_None}`
- NOT a list of rows like Binance klines (`k[4]` for close etc.)
- Old code indexed it as list-of-rows → `len(dict)=4 < 20` → impulse features always zeros (fixed 2026-04-01)
- Correct access: `_rc['closes'][-1]`, `_rc['highs'][-1]`, etc.

## Time Features — Must Use UTC
- `datetime.utcnow()` not `datetime.now()` — Habib is UTC+8 (Malaysia)
- US session: 13-21 UTC, Asia: 1-9 UTC, Funding: 0, 8, 16 UTC
- During training: `_extract_features_from_candles()` overwrites features[107:116] with actual candle timestamp after calling `_extract_features()`. NOT using wall clock. This is correct.

## Ensemble Weighting Math
- Shannon entropy: `H = -(p·log2p + (1-p)·log2(1-p))`, certainty=1-H, weight ∝ certainty^1.5
- Alpha=1.5 is tunable, not arbitrary
- Brier quality: `1.0 - rolling_avg_brier` (rolling deque last 500, reset on retrain)
- Final blend: `0.4 * val_auc + 0.6 * brier_quality`
- Ensemble weights key in performance history: `val_auc` (NOT `val_acc` — that's the legacy fallback key)

## Entropy Veto Threshold
- `H(p_ens) > 0.85 bits → NEUTRAL`
- Binary entropy max = 1.0 bits at p=0.5
- H=0.85 corresponds to p ∈ approximately [0.26, 0.74]
- Does NOT correspond to [0.40, 0.60] as old code comment incorrectly stated

## Disagreement Threshold Derivation
- `E[std] = sqrt(p*(1-p)/n) = sqrt(0.25/7) ≈ 0.189` for 7 independent binary classifiers at p=0.5
- Threshold: 0.20 (rounded up from theoretical 0.189)
- Max discount: 20%

## Agent Domain Masks (QUANTA_ml_engine.py)
- `_MEAN_REVERSION` mask: indices [258, 260, 263] — RSI_δ, BB_δ, RSI_accel + [275, 276] BS barrier
- `_FLOW_VOLUME` mask: indices [262, 267] — vol_δ, VPIN_δ + [277] BS implied vol ratio
- `_IMPULSE` mask: per-TF ATR + RSI + MACD + BB + cross-TF + volume + VPIN + frac_diff + taker_imbalance + order book + HMM + Kyle + delta features + impulse [270-274] + BS [275,276,277]
- `_TREND_CORE`: added [275, 276]; `_VOLATILITY_BREAKOUT`: added [275,276,277]; `_STRUCTURAL`: added [275,276,277]; `_MACRO_SENTIMENT`: added [275]

## TFT Integration
- TFT output replaces feature index 223 (padded null slot) at inference time
- Quality gate: `tft_val_auc > 0.55` — below this, slot 223 stays 0.0
- Cap per symbol = 50 sequences (MX130 memory)
- TFT trained with Triple Barrier labels (Athena settings)

## PPO Role: Size Oracle (v11.5b — redesigned 2026-04-01)

**PPO is a position SIZE ORACLE, not a gate.**

Old design (removed): PPO dampened/boosted `ml_conf`, could block trades entirely.
Problem: Random early PPO with 2× veto power systematically killed good ML signals.

New design: PPO outputs `ppo_size_mult ∈ [0.25, 2.0]` applied to Kelly-calculated notional.
ML ensemble still owns direction + confidence. Trade always happens if ML passes 4 filters.

### Size Multiplier Formula
```
val_signal = sigmoid(ppo_value)  # [0, 1]

PPO agrees with ML direction  → size_mult = 1.0 + 1.0 * val_signal   # [1.0, 2.0]
PPO says HOLD                 → size_mult = 0.5 + 0.5 * val_signal   # [0.5, 1.0]
PPO contradicts ML direction  → size_mult = 0.25 + 0.25 * val_signal # [0.25, 0.5]
```
Clamped to [0.25, 2.0] defensively. Hard cap: notional never exceeds MAX_RISK (2.5%) of balance.

### Reward Signal
`raw_return = outcome_sign × move_mag × ppo_size_mult`
- Win + sized up → big positive reward
- Loss + sized up → big negative penalty
- Win + sized down → small positive
- Loss + sized down → small negative
- Neutral → -move_mag × 0.5

PPO learns: "agree with good ML signals (size up), doubt bad ones (size down)."

### Why This Is Correct
- Bad PPO → suboptimal sizes, not missed trades
- Good PPO → 2× notional on high-conviction setups = meaningful alpha
- PPO has a bounded, learnable job (sizing) vs unbounded damage potential (blocking)
- Consistent with how institutional RL is used: sizing oracle on top of a signal generator

### Code Locations
- `ppo_size_mult` init: QUANTA_bot.py (default=1.0 before PPO block)
- Heimdall Sizer block: QUANTA_bot.py (replaced Heimdall Gate)
- `open_position(ppo_size_mult=1.0)`: QUANTA_trading_core.py — applied after Kelly, before risk gate
- `ppo_size_mult` stored in `positions[symbol]` for tracking
- Reward: QUANTA_trading_core.py `check_predictions()` — scales raw_return by ppo_size_mult

## BS Execution Power — "Trillion Dollar Equation" (2026-04-02)

**Black-Scholes theoretical win probability (Feature 275) is no longer just a passive ML feature.**
It now has direct execution authority over trade gating and position sizing.

### BS Direction-Aware Probability (v11.5b+)
- For LONG trades: `bs_prob = feature[275]` (as-is)
- For SHORT trades: `bs_prob = 1.0 - feature[275]` (flipped — bear drift → high P for shorts)
- Uses **dominant specialist's barriers** (not generic median) for veto/turbo

### BS Veto (Hard Gate)
```
if bs_prob < 0.25:
    passes_gate = False  # Trade killed — stochastic drift structurally unfavorable
```
- Reads raw unscaled `features_batch[idx][275]` (Darling-Siegert scale function output)
- **Direction-corrected**: flipped for SHORT trades so bear drift boosts shorts, not penalizes them
- Overrides ML confidence — if drift says < 25% chance of hitting TP before SL, trade is fundamentally unviable
- Logged: `⛔ BS VETO: {symbol} {LONG/SHORT} rejected by stochastic drift (P={bs_prob:.2f} < 0.25, spec={dom})`

### BS Turbo Boost (Sizing Overdrive)
```
if bs_prob > 0.55:
    bs_multiplier = min(1.5, bs_prob / 0.50)
    ppo_size_mult *= bs_multiplier
```
- Stacks multiplicatively with Heimdall PPO sizing
- At P=0.55 → 1.1×, at P=0.75+ → 1.5× (capped)
- Logged: `🚀 BS TURBO: {symbol} drift favorable`

### Final Safety Cap
```
ppo_size_mult = min(ppo_size_mult, 2.5)
```
- Prevents PPO (2.0×) × BS (1.5×) = 3.0× runaway sizing
- Hard ceiling at 2.5× regardless of how PPO and BS stack

### BS Edge Override (Kelly Calibration)
```
_bs_edge = _live_bs_prob - _bs_win_prob_baseline  # was: (ml_conf/100) - baseline
```
- Old formula mixed ML confidence with zero-drift baseline (apples vs oranges)
- New formula: pure stochastic edge = drift-adjusted P minus random-walk P
- Feeds directly into Kelly edge penalty in `open_position()`
- Fallback to old formula if feature read fails

### Code Location
- BS power block: QUANTA_bot.py (after Heimdall Sizer, before `opportunity_score`)
- BS edge override: QUANTA_bot.py (in barrier R/R calculation block before `open_position()`)
- Safety cap: QUANTA_bot.py (immediately after BS power block)

**Scale function formula** (Darling-Siegert 1953, Hull Ch.26):
```
P(hit TP before SL) = [1 - exp(θ·sl_dist)] / [exp(-θ·tp_dist) - exp(θ·sl_dist)]
where θ = 2ν/σ², ν = μ - 0.5σ²  (Itô-corrected GBM drift)
```
Zero-drift case: P = sl_dist / (tp_dist + sl_dist)

**Critical sign note**: The numerator uses `exp(θ·sl_dist)` and denominator uses `exp(-θ·tp_dist) - exp(θ·sl_dist)`. Signs swapped = completely wrong directional behavior (bear drift → P→1 instead of P→0).

**`open_position()` new params (QUANTA_trading_core.py)**:
- `barrier_rr=2.0` — per-specialist TP/SL ratio, replaces hardcoded `b=2.0` in Kelly
- `bs_edge=None` — ML conf minus zero-drift baseline. If < 0.02: `prob *= max(0.5, bs_edge/0.10)`
- BS edge threshold 0.02 = local heuristic (like DIRECTION_THRESHOLD=0.12)

**Implied vol tracking**:
- `PaperTrading._bs_bars_to_hit[symbol]` — deque(maxlen=50), appended on TP/SL barrier hits only (not TIMEOUT)
- Synced to `DeepMLEngine._bs_avg_bars_to_hit` every RL check cycle in bot
- Feature 277 = 1.0 (neutral) until ≥5 trades per symbol

## Optuna Adaptive Hyperparameter Search
- Studies persist to `models/optuna_studies/{specialist}.pkl` — accumulate across retrains
- 20 trials × 300 iterations × 3000 samples per specialist ≈ 30 min on MX130 CPU
- Search space: `depth` 4-8, `lr` 0.01-0.35 log, `l2` 1-15 log, `subsample` 0.6-1.0
- Fires on first training only. Force re-search: `specialist['_optuna_searched'] = False`
- Best params merged into `specialist['hyperparams']` before production training

## Undersampling
- MAX_RATIO = 10.0 (10:1 max class imbalance before undersampling)
- N_BUCKETS = 10 (temporal buckets for stratified undersampling — preserves regime distribution)
- Random seed per specialist from `hyperparams['random_seed']`

## Cross-Event Negative Sampling
- CROSS_SAMPLE_RATIO = 0.05 (5%) — reduced from 15% to prevent trivial domain-boundary learning
- Cross-domain events weighted 0.5× (lower confidence)
- Guard: `if not other_agents: raise ValueError(...)` before `// len(other_agents)`
- Only triggered if other_count >= 10

## Hard Negative Mining
- Boost factor: CATBOOST_HARD_NEG_BOOST
- Diversity cap: 50% of total weight mass
- Collapse warning: >60% misclassified OR effective_boost < 1.1

## GPU Subsample Strategy
- Head+tail: 30% oldest + 70% most recent (preserves regime diversity + recency)

## Artemis Lookback — Fixed Off-By-One
- OLD (bug): `highs[i-_LOOKBACK:i-1]` — excluded bar i-1, window was _LOOKBACK-1 bars
- NEW (fixed): `highs[i-_LOOKBACK:i]` — correct _LOOKBACK bars, excludes only current bar
- Same fix applied to `lows` in bearish loop

## Impulse Features (270-274) at Inference
- Live inference: `predict_with_specialists()` builds `_live_candles` dict from `candle_store.get(symbol, '5m')`
- Training path 1 (precomputed): `raw_c[max(0,pos-49):pos+1]` etc. — has taker_buy
- Training path 2 (candles): builds dict with `'taker_buy': None`
- Fallback when data unavailable: `[0.0, 0.0, 0.0, 0.5, 0.0]`

## Scope Bug Pattern — locals() vs dir()
- `dir()` returns the MODULE namespace, not local variables
- Use `'varname' in locals()` to check if a variable was defined in current function scope
- `_overlap_discount` was checked with `dir()` — always False — discount never applied (fixed 2026-04-01)

## HMM Regime Model — Fixed (2026-04-02)

**`_get_regime` in QUANTA_ml_engine.py** — per-symbol GaussianHMM, 4h cache.

### Current correct implementation (v11.5b+):
- Features (multi-scale): `[log_ret_1bar, log_ret_12bar, log_ret_48bar, atr_pct, vol_ratio]` — 5 features
- **3 states**: bull(0), range(1), bear(2) — reduced from 5 to avoid overfitting with 200 obs
- `n_iter=100`, `covariance_type='full'`, 200 observations
- State ordering: **DESCENDING** by mean log-return (rank 0=bull, rank 2=bear)
- Cache format: `{'model': GaussianHMM, 'rank_to_int': int32[3], 'n_states': 3}`
- Feature 231 output: `1.0 - rank/2.0` → [1.0=bull, 0.5=range, 0.0=bear]

### `_regime_routing` table convention (QUANTA_ml_engine.py ~line 615)
```
# index: 0(bull)  1(range)  2(bear)
'athena': [1.0, 0.3, 0.1]
```
**Sort order must be DESCENDING** so rank 0 always = most bullish.

### Bot routing (QUANTA_bot.py ~line 1730)
Features sent for prediction: `[log_ret_1bar, log_ret_12bar, log_ret_48bar, atr_pct, vol_ratio]` — dynamically extracted from candle_store matching the training distribution.
Cache access: `cached['model'].predict(obs)` + `cached['rank_to_int'][raw_state]`.
`regime_idx = max(0, min(2, ...))`, fallback `[0.5]*3`.

### Unified HMM System (Tier 3 Fix)
- There is only one HMM system: `self.hmm_models[symbol]`.
- The generic `MarketRegimeHMM` (MoE) was deprecated. Both `_regime_routing` and PPO state augmentations grab values from the same cached per-symbol `self.hmm_models[symbol]` evaluated in the routing block.

## Bot Config Access Pattern
- `Bot` class (QUANTA_bot.py line 505) uses `self.cfg` — a `Config()` instance
- `self._sys` does NOT exist on `Bot` — it exists on a different class (line 318)
- Correct access: `self.cfg.events.athena_tp_atr` etc.
- `self.events` is also set at `__init__` line 339 as alias for `self.cfg.events`

## Optuna NaN — Root Cause Pattern
Training fallback features must be `0.0`, never `float('nan')`. NaN → nan_to_num → 0.0 → zero-variance column → StandardScaler `0.0/0.0 = NaN` → NaN in X_train → NaN from `predict_proba` → Optuna rejects NaN objective.

Always apply `nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)` AFTER scaler transform (already done at line ~1374 post-fix). Never put `float('nan')` in feature extraction fallback paths.

## Known Structural Limitations (not bugs)
- Monitor `calibration_error` always returns None — `log_prediction()` called with `actual_class=None`. Brier scores in `_brier_scores` are the live calibration signal.
- On-chain features (254-256): `0.0` during training — model learns neutral baseline, real values used at inference
- Stat arb features (248-250): `0.0` during training — same pattern
- Order book: OHLCV proxy during training vs real depth during inference. **Future fix:** record top 5 bid/ask levels per candle during cache warmup. Hephaestus and Nike most affected.
- DIRECTION_THRESHOLD=0.08 (lowered from 0.12) to shrink the neutral movement zone and force more active predictions.
- Regime routing multipliers: now automated via `_learn_regime_routing()` in QUANTA_ml_engine.py — scales allocation vectors based on per-agent accuracy across tracked HMM regimes

## BS Volatility: GARCH(1,1) Filtered Sigma (v11.5b+)
- `_jit_bs_barrier_prob` now uses GARCH(1,1) filtered vol instead of rolling sample std
- Crypto params: `omega=1e-6, alpha=0.10, beta=0.85` (Bollerslev 1986)
- Captures volatility clustering — single O(n) pass, seeded from sample variance
- Multi-scale drift: 60% fast (20-bar) + 40% slow (100-bar)

## Kou (2002) Jump-Diffusion BS (Tier 3 — replaces Hull analytical)
- Feature 275 now computed via `_jit_kou_barrier_prob` — double-exponential jump-diffusion
- Better captures crypto fat tails and sudden jumps vs pure GBM
- Compensatory scaling properties matching crypto market behavior
- Original Hull/Darling-Siegert continuous diffusion replaced

## Prediction Pipeline — Sound Components
| Component | Why Sound |
|-----------|-----------|
| Shannon entropy weighting | `H = -(p·log2p + (1-p)·log2(1-p))`, certainty=1-H, weight∝certainty^1.5 |
| Brier score trust | Rolling deque last 500, reset on retrain |
| Kyle's Lambda | OLS: `Cov(ΔP, signed_vol)/Var(signed_vol)` |
| CUSUM inference gate | Pure zero-drift log returns, reset after fire — LdP AFML Ch.2 exact |
| Triple Barrier labeling | LdP 2018 Ch.3 exact |
| Feature pruning | LossFunctionChange + mean/2 — proper MDI |
| Temporal split + dynamic PURGE_GAP | Sound |
| Conformal uncertainty | `np.max(interval_widths)` — mean was invalid |
| Entropy veto | `H > 0.85 bits` — theoretically grounded |
| Disagreement threshold | `sqrt(0.25/7) ≈ 0.189` — derived, not guessed |

## Specialist Agents Summary (v11.5b — 7 agents)
| Agent | Trigger | Direction |
|-------|---------|-----------|
| Athena | CUSUM_pos + new highest high | Long |
| Ares | CUSUM_neg + new lowest low | Short |
| Hermes | Volatility squeeze + expansion | Both |
| Artemis | CUSUM_pos + vol_surge + NOT new_high (bull) / CUSUM_neg + vol_surge + NOT new_low (bear) | Both |
| Chronos | CUSUM both + RSI extreme | Both |
| Hephaestus | Price pattern detection | Both |
| Nike | Single candle impulse: range>2×ATR AND vol>1.5×avg20 AND body_eff>0.5 (no CUSUM) | Both |

## How to Apply
Read this file before making any changes to training logic, feature extraction, ensemble math, or threading. These are non-obvious decisions that took significant analysis to arrive at. Many have been wrong before — assume nothing is obvious.
