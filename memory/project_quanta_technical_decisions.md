---
name: QUANTA Technical Decisions & Critical Facts
description: Important design decisions, known gotchas, and critical implementation facts for QUANTA
type: project
---

## Conditional Kou First-Passage For Nike Trigger Entries (2026-04-07)
- For Nike, the relevant barriers are `TP = 2.0 * ATR`, `SL = 0.8 * ATR`, `T = 12` bars.
- If the trigger jump has already occurred and price starts at `b+ - eps`, the correct conditional probability is the same Kou process restarted from the post-jump state. Conditioning changes the start point, not the future law.
- In shifted coordinates with interval `(0, a)`, use `a = log((entry + 2.0 * ATR) / (entry - 0.8 * ATR))` and `y0 = a - eps` when working in log-price units.
- Fast approximation: ignore the far lower barrier and use the one-sided Kou-Wang Laplace-transform formula on residual distance `eps`.
- Tight method: solve the finite-horizon backward PIDE on `(0, a)` and evaluate `u(T, y0)`.
- Current `_jit_kou_barrier_prob` in `quanta_features.py` is an infinite-horizon effective-diffusion heuristic. It is acceptable for feature 275 but should not be treated as the exact finite-horizon conditional Nike score.
- The post-trigger research note lives in `memory/kou_first_passage_conditional_note.md`.

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

## Live BS/Kou Placement (2026-04-07)
- Keep the generic Kou-style barrier probability as an ML feature, but do not rely on feature weight alone for execution decisions.
- Compute a fresh live BS/Kou score inside `QUANTA_bot.py` after direction and dominant specialist are known, so the math uses the specialist's actual TP/SL/max-bars instead of the generic feature-275 median geometry.
- For Nike long breakouts, route the live score through a dedicated conditional Kou kernel using pre-trigger returns plus the trigger-body jump to estimate post-jump TP-before-SL probability over the finite specialist horizon.
- Feed live TP/SL geometry into the barrier solver in log-space, using `log1p(tp/price)` for the upside barrier and `-log(1-sl/price)` for the downside barrier.
- Use one shared live BS/Kou context for three execution actions: veto, turbo sizing, and Kelly probability input. Avoid recomputing inconsistent barrier probabilities in separate code paths.
- In paper trading, blend ML confidence with the live BS/Kou TP-before-SL probability before Kelly sizing, then apply the BS edge penalty relative to the specialist's zero-drift baseline.

## Nike Entry Logic (2026-04-08)
- Nike should not be treated as only a first-candle detector. Cache spike studies showed many explosive winners start with an ignition candle and become clearer on the very next confirmation candle.
- Use a two-stage Nike trigger:
  - relaxed setup candle to recognize ignition earlier
  - stricter same-bar entry only for clearly extreme candles
  - otherwise wait exactly one bar for structural confirmation
- Confirmation bar requirements:
  - low remains above the midpoint of the setup candle
  - close remains above the setup close
  - high breaks the setup high
- Cache benchmark target for Nike pattern recognition is recall on curated spike events, not raw accuracy on all anomalies. The tuned rule reached `75%` recall on the top `80` spike events and `80%` recall on the top `50`.
- Extend Nike's vertical barrier from `12` to `24` bars. Positive spike events peaked around `25` bars on average and `28.5` bars at the median, so the old `12`-bar window was materially too short.

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
## Nike Benchmark Interpretation (2026-04-08)
- Keep two benchmark lenses separate:
  - anomaly-event recall asks whether Nike recognizes the breakout archetype
  - realized TP/SL outcomes ask whether the trading geometry is economically viable
- The full-cache benchmark validated the current Nike rule as a specialist, not a universal anomaly detector:
  - all-anomaly same-or-next-bar recall = `53.05%`
  - `run_up > 10%` same-or-next-bar recall = `72.34%`
  - top `100` to top `1000` realized spike cohorts stayed around `72%` to `75%` recall
- Do not loosen Nike globally just to improve all-anomaly recall. The strongest value is in high-conviction ignition/confirmation breakouts; broadening the pattern indiscriminately is likely to dilute signal quality.
- Keep `max_bars = 24` for now because it materially improves alignment with breakout maturation time, but recognize that even `24` bars still misses many true peaks. A future runner exit or trailing structure is more defensible than extending the fixed timeout indefinitely.

## Nike V2 Rollout Rule (2026-04-08)
- Nike v2 must be judged by explicit rollout gates before live promotion:
  - top `500` realized spike recall on `same/+1/+2` >= `75%`
  - full-cache weighted PF > `1.150`
  - signal count growth <= `35%`, unless PF also improves
- If the full A/B/C rollout fails those gates, do not ship it live just because recall improved.
- Full-cache v2 result:
  - recall gate passed (`80.80%`)
  - PF gate failed (`1.001`)
  - signal growth gate passed (`+8.38%`)
- Follow-up tuning showed:
  - Tier `C` consistently hurts trading quality and should remain observe-only.
  - Tier `A` is the strongest live tier.
  - Tier `A+B` preserves strong pattern recall, but no tested exit/threshold combination cleared the PF gate.
- Operational rule for now:
  - Tier `A` may trade live.
  - Tier `B` and Tier `C` should remain observe-only until a new benchmark run clears the PF gate.

## Norse Simulation Decisions (2026-04-09)
- Thor tuning is no longer a pure "maximize final capital then tie-break by drawdown" search. `quanta_norse_year_sim.py` now uses a soft penalty objective:
  - `score = final_capital * (1 - max(0, (max_drawdown_pct - 60) / 100) * penalty_weight)`
  - default `penalty_weight = 1.5`
  This preserves the user's max-profit bias but zeroes out obviously unrunnable parameter sets faster than the old lexicographic comparator.
- Thor entry selection now has three additional hard gates before sim execution:
  - 6-bar MAE veto via `thor_mae_veto_atr`
  - minimum `wave_strength_score` via `thor_wave_strength_min`
  - maximum `top_risk_score` via `thor_top_risk_max`
  These are applied using `compute_pump_state()` on the immediate pre-entry context, not realized post-trade data.
- Baldur is now treated as an exit overlay, not an alpha sleeve. In the year sim:
  - independent Baldur shorts are removed from the candidate portfolio
  - Baldur warnings above `baldur_warning_exit_score` cut the parent Thor trade at the next bar open
- Freya is now treated as a Thor pyramid-add, not an independent scalp. In the year sim:
  - the add is only allowed while the parent Thor trade is still open
  - it requires Thor already to be past `thor_trail_activate_atr`
  - it requires `wave_strength_score >= freya_pyramid_add_trigger_wave`
  - it requires `top_risk_score < freya_pyramid_add_top_risk_max`
  - the add notional is `freya_pyramid_add_notional_fraction` of the parent Thor notional and exits with the parent Thor trade
- `quanta_pump_mae_stats.py` is the new Phase A calibration source. It is intended to produce:
  - raw per-pump MAE / runup rows
  - long-format drawdown curves
  - bucketed aggregate MAE statistics
  - a Thor stop decision matrix over the widened exit family
  - a markdown report with a recommended parameter block
- `quanta_multi_exchange.py` now exposes the stop-order API shape needed by the live retrofit:
  - `place_stop_market()`
  - `place_take_profit_market()`
  - `cancel_order()`
  - `modify_stop_order()`
  Current Bybit / OKX behavior is explicit warning + stub return, not exchange-native stop coverage yet.

## Norse MAE Optimization Decisions (2026-04-10)
- The MAE optimization must preserve output semantics exactly. In particular, `quanta_pump_mae_stats.py` already mixes two drawdown conventions and that was kept intentionally:
  - long-format curve rows use close-based drawdown from the running peak
  - several pump-row MAE fields use low-based drawdown from the running peak or from entry
  Cleaning this up would have changed the benchmark data, so the refactor keeps the old behavior.
- `build_pump_ledger(...)` had mixed peak semantics before the refactor, and those semantics were preserved exactly:
  - per-row `bars_since_peak`, wave-strength, and top-risk use the first occurrence of the running max high (same as `compute_pump_state(...)`)
  - `time_to_material_drawdown_bars` uses the latest equal-high bar because the original loop reset `peak_bar` on `>=`
  - `time_to_peak_bars` and `volume_decay_after_peak` use the earliest bar that achieved the final max runup because the original summary derived them from `max(rows, key=runup_atr)` on the forward-built ledger
- New internal helper in `quanta_norse_agents.py`:
  - `_pump_path_analytics(ctx, start_bar, end_bar, material_drawdown_atr)`
  - purpose: compute the entire pump path once and feed both the MAE script and `build_pump_ledger(...)`
  - this helper is internal only; no CLI or schema change was introduced
- `quanta_pump_mae_stats.py` now prefers deterministic symbol-level multiprocessing, but parent reductions stay single-threaded:
  - workers only prepare one symbol and return raw rows / curve rows / numeric decision payloads
  - bucket aggregation, recommendation derivation, and markdown report generation remain in the parent to preserve exact ordering and byte-stable output
- Worker output order is intentionally stabilized by preserving `_load_windowed_cache()` symbol order in the parent merge. The final CSV/report ordering guarantees remain:
  - pump rows sorted by `entry_ts`, `symbol`
  - curves sorted by `pump_id`, `bars_since_entry`
  - stats sorted by existing bucket/report logic
  - decision matrix sorted by `expectancy_atr`, `weighted_pf`, `sum_realized_atr`
- `ProcessPoolExecutor` is the primary runtime path, but `quanta_pump_mae_stats.py` now falls back to deterministic single-process execution when worker creation fails with `OSError` or `PermissionError`. This fallback exists specifically because the Codex sandbox blocks Windows multiprocessing pipe creation, not because the parallel design was dropped.
- Verification standard for this refactor:
  - exact equality against the pre-refactor MAE logic on a fixed 30-day subset
  - byte-stable reruns on the same subset
  - explicit ledger-row + ledger-summary equality for `build_pump_ledger(...)`
  - hotspot benchmark focused on the analytics path, not just end-to-end wall time, because sandboxed process creation makes full parallel runtime measurement unreliable here

## Norse Turbo Strict-Path Decisions (2026-04-11)
- Stage-0 cache validity is now fail-closed by default:
  - valid run requires `_manifest.json` present with `complete=true`
  - manifest counts must match the current 365-day universe
  - every symbol pair must pass loader validation against the current `max_open_time_ms`
- `quanta_norse_year_sim.py` should not silently mix cache hits with uncached fallback on the canonical path.
  - default behavior: validated cache only, zero misses allowed
  - emergency-only escape hatches:
    - `QUANTA_NORSE_ALLOW_CACHE_MISS_FALLBACK=1`
    - `QUANTA_NORSE_ALLOW_LEGACY_FALLBACK=1`
- Cache-loaded `PreparedSymbol` objects already contain a full dense `SparseFeatureContext`.
  - `_ensure_feature_positions(...)` must no-op for that path when `replay_engine is None` and `len(feature_ctx.times) == len(df)`
  - otherwise the code falls back into offline feature extraction and can trip unrelated replay-engine assumptions
- The Turbo cache evaluator must optimize on the same Thor score used by the full sim.
  - raw `nike_score` from the detector is not the execution gate
  - canonical gating score is `score_thor_signal(sig, ctx)`
  - MAE rows now carry that as `thor_feature_score`
  - `CachedTrialEvaluator` should use `thor_feature_score`, not `nike_score`
- Current measured Norse Turbo outcome after score-contract alignment:
  - final capital `$10,745.57`
  - growth `+7.46%`
  - max drawdown `15.93%`
  - `51` executed Thor trades
  - Freya and Baldur contributed zero trades on this pass
  - cache-vs-sim divergence improved materially after the score fix but still remained non-trivial on drawdown (`13.8 pp`)
- `run_year_simulation()` must preserve the tuner metadata returned by `_tune_params(...)`.
  - the final detailed evaluation should only run when pump/trade outputs are missing
  - otherwise the returned Turbo metadata (`wf_fold_metrics`, sensitivity, Optuna study, learned filter, cache reconciliation) is lost from the final report

## Thor Exit Capture Decisions (2026-04-12)
- Thor profit capture is currently more constrained by exit shape than by raw breakout recognition. The next iteration should improve the shared exit kernel before changing entry logic or objectives.
- `thor_max_bars_post_bank` should mean post-bank age, not total age since entry.
  - enforcing it from entry suppresses runner capture when bank is reached late
  - the shared Thor sim kernels should track `bank_bar` explicitly and enforce the timeout from there
- The Thor exit kernel should use live path state, not only fixed ATR distances, after the bank hit.
  - strong continuation states should tolerate more giveback and trail later / wider
  - weakening states should tighten faster based on wave decay, top-risk rise, flow exhaustion, weak close position, upper wicks, and stale bars-since-peak
- The adaptive path should stay inside the shared Numba kernel used by both:
  - `simulate_thor_exit_stop_market(...)`
  - `_sweep_thor_exits_njit(...)`
  This keeps the MAE row generation path and the live year-sim exit path aligned.
- A controlled below-entry runner stop after bank is acceptable when the partial bank already funds the residual risk.
  - the correct limit is bounded by the realized bank cushion on the remaining position, not by a hard “runner must never dip below entry” rule
  - this is intended to reduce premature shakeouts on real breakouts, not to widen pre-bank loss risk
- Stage-2 exit tuning should not depend on a single round-1 winner.
  - keep the top few round-1 seeds
  - then sweep trail/time-cap params across those seeds
  - when objective scores are similar, prefer higher median fold growth / expectancy
- This step intentionally avoided a new MAE rebuild or year sim run.
  - compile/import verification was sufficient to validate the code edits
  - the user still needs to rerun the MAE builder and the year sim locally to measure whether the new exit family actually improves realized growth

## Thor Tuning Breakdown Observed In Full Rerun (2026-04-12)
- The latest full-year rerun proves the current Turbo objective is no longer a reliable benchmark for this strategy shape.
  - best Optuna score stayed around `-99.9`
  - stage-2 exit fine-tune scores stayed effectively flat across the whole widened search
  - final full sim still produced `+71.72%` growth with `54.21%` max drawdown and `42` trades
- Root cause is structural:
  - `MIN_TRADES_PER_FOLD = 20` in `norse_tuner/objective.py`
  - current Thor trade frequency is only `42` trades over the whole year
  - with `5` walk-forward folds, most or all folds necessarily fail the minimum-trades gate
  - this collapses the objective surface so Optuna and exit fine-tune cannot meaningfully rank candidates
- The full sim result is still real as a point backtest, but it should not be interpreted as a properly tuned optimum.
  - the candidate was selected on a nearly flat / degenerate tuning surface
  - treat this run as evidence that the new exits can raise gross returns, not that the tuner is now trustworthy
- The cache approximation is currently too weak for drawdown-sensitive tuning under the new exit family.
  - observed cache-vs-sim divergence reached `8.6%` capital and `47.2 pp` drawdown
  - likely driver: longer runner holds increase overlap / path / capital-allocation effects that the cached evaluator does not approximate well enough
- Next priority should be tuner repair, not more parameter-search runs on the current benchmark.

## Turbo Repair Decisions After The 71.72% / 54.21% Run (2026-04-12)
- The hard minimum-trades fold gate was the primary reason the Optuna surface collapsed.
  - a low-frequency Thor strategy should not be treated as invalid solely because a fold has fewer than `20` trades
  - replacement policy: use confidence-weighted fold scoring instead of binary rejection
  - zero-trade folds remain hard failures; low-trade profitable folds are discounted rather than erased
- Candidate selection should not trust one cache-selected winner.
  - keep several distinct bootstrap-passed stage-1 candidates
  - fine-tune exits for each candidate
  - rerank the top candidate set using real full-sim outputs
- The final full-sim candidate score should be stricter on drawdown than the old legacy helper.
  - drawdown around `50%+` should be actively disfavored during final selection
  - selection now uses a drawdown-aware full-sim rerank key instead of a pure cached winner handoff
- Cache-vs-sim reconciliation should include the tuned exit params whenever possible.
  - `CachedTrialEvaluator.evaluate_full(...)` only knows the cacheable stage-1 params
  - stage-2 diagnostics should therefore use a fast replay estimate that includes the exit params, not the stage-1 evaluator alone
- Persisted Optuna studies from the pre-repair objective should be discarded.
  - changed low-trade scoring means old trials are no longer on the same objective surface

## Norse Full-Sim Progress Logging (2026-04-12)
- User-facing runtime visibility matters for long Norse runs; stage-only prints were not enough.
- Detailed trade-style prints are now emitted only on the real full-sim path, not the fast cache evaluator path.
- Logging lives in `_evaluate_params(...)` and is wired from `norse_tuner/pipeline.py` with `sim_logger=_log` and `detailed_progress=True`.
- Output is intentionally compact: per-symbol summary plus a few sample Thor/Freya trade lines, with periodic aggregate checkpoints for quiet symbols.

## Norse Run Artifact Strategy + Diagnostics (2026-04-12)
- Research outputs should be append-only across runs, not overwritten in place.
- `quanta_norse_year_sim.py` now creates a UTC-timestamped run folder under `norse_runs/` and writes all artifacts there.
- Post-run diagnostics are generated from `trade_df` joined with `pump_summary_df` on `pump_id`/`symbol`.
- Loss rows get a heuristic `what_went_wrong`; big-win rows get a heuristic `what_went_right`.
- Big wins are currently defined as positive-net-PnL trades at or above the 80th percentile of positive trade `net_pnl`.
- Feature comparison output focuses on medians/means for numeric trade + pump-path fields between losses and big wins.

## Norse Thor V12 Specialist Pivot (2026-04-14)
- Decision: Collapse the 7-agent Greek Pantheon ensemble into a single high-conviction specialist (Thor).
- Rationale: The ensemble approach introduced excessive complexity (Shannon entropy vetoes, cross-agent calibration jitter) which diluted performance in fast-moving crypto markets. A single specialist focusing on impulse continuation (Nike/Thor) delivers higher alpha with lower latency.
- Feature Space: Reduced from 278 features (including ensemble meta-features) to a specific 102-feature mask (`domain_impulse`) focused on microstructure and short-term momentum.
- Offline-First: All training redirected to use local `feather_cache` to bypass Binance API rate limits and ensure perfect backtest reproducibility.
- Legacy Compat: Maintained `self.models` and `self.scaler` bindings in `DeepMLEngine` to support existing consumer code without full refactors.

## WalkForward Sim Design Decisions (2026-04-15)
- **Loop order**: bar-outer / symbol-inner is the ONLY correct structure. Symbol-outer/bar-inner makes `_MAX_CONC` meaningless and breaks concurrent position management.
- **Train window**: hardcoded 60d in sim (`_TRAIN_DAYS = 60`). Config has 180d but that yields only 6 windows from 365d data. 60d → 10 windows (300 OOS days). Never read `_BT.train_window_days` for this.
- **CatBoost inference path**: `_REPLAY_ENG._fast_extract_at_position(precomputed, abs_bar, klines_np)` → 278-dim vector → `_IMPULSE_MASK` (102 features) → `predict_proba`. Returns `(score, feat_vec)` tuple so caller can check `feat_vec[272]` (pre_impulse_r2) without double-compute.
- **Wave strength source**: klines col[9] = `taker_buy_base` (real Binance taker flow from feather cache). Formula: `net = (2×taker_buy/total_vol - 1)`, `score = (net+1)/2 × 100`. Falls back to directional-vol proxy if col[9] absent.
- **Continuous risk scaling**: score=68 → 0.5% risk, score=100 → 3.0%, linear. The old binary `activation_score=85` threshold fired on <4% of CatBoost-scored trades (scores cluster 68–85, max observed ~88.6).
- **Per-trade equity curve**: appended in `SimPositionManager._close()`. Window-level snapshots always showed 0% MaxDD. Per-trade is required.
- **pre_impulse_r2 veto index**: feature index [272] in 278-dim vector. Skip trade if > 0.70.
- **Hour-of-day filter**: extracted from `klines_np[abs_bar, 0]` (open_time ms). `utc_hour = int((ts_ms / 3_600_000) % 24)`. Skip hours in `_SKIP_UTC_HOURS = {0, 7, 10, 11, 12}`.

## MAE-Calibrated Thor Exit Parameters (2026-04-15)
Source: `NORSE_MAE_STATS_REPORT.md` (19,563 pump rows, 1.4M drawdown-curve rows, 384 decision-matrix combos)
- `SL_ATR = 3.00` — P90 winner MAE. At SL=2.4: only 49.5% of 1-3 ATR outcome trades survive. At 3.0: 63.2%. Tighter stops kill eventual winners.
- `BANK_ATR = 4.20`, `BANK_FRAC = 0.35` — top decision-matrix combo `(sl=3.0, bank=4.2, frac=0.35, activate=1.5, trail=2.0)`.
- `TRAIL_ATR = 2.00` — tighter runner trail. Old 6.0 ATR was giving back too much.
- `TRAIL_ACTIVATE_ATR = 1.50` — trail only activates 1.5 ATR above bank price. Below that SL stays at breakeven. Prevents whipsaw exits on volatile runners.
- `MAX_PRE = 48` bars (4h), `MAX_POST = 96` bars (8h). Old 144/1152 = 12h/4d was far too long.
- `MAE_VETO_ATR = 3.62`, `MAE_VETO_BARS = 5` — exit if trade goes 3.62 ATR adverse in first 5 bars. P50 winner MAE from entry = 4.29 ATR overall, recommendation = 3.62.
- **Bad hours** (UTC, PF < 0.85): 0 (0.68), 7 (0.81), 10 (0.75), 11 (0.81), 12 (0.51). **Good hours**: 2 (1.15), 8 (1.16), 14 (1.18), 13 (1.06), 6 (1.01).
- **Bad days** (UTC): Monday=0 (PF 0.81), Tuesday=1 (0.89). **Good days**: Sunday=6 (1.11), Wednesday=2 (1.07), Friday=4 (1.05).
- Baldur exit efficacy: current exit expectancy 0.47 ATR vs Baldur-triggered 0.83 ATR. Baldur fires median 15 bars BEFORE peak.

## 3-Layer Pyramid Strategy (2026-04-15)
Calibrated from MAE pump stats: overall `max_runup_p50 = 3.80 ATR`, `max_mae_entry_p50 = 4.29 ATR`.
Constants: `_PYR_TRIGGER_ATR=0.5`, `_PYR_ADD_SL_ATR=0.5`, `_PYR_ADD_FRAC=0.50`, `_PYR_RECOVERY_ATR=3.77`
- **L1**: normal Thor entry at P. SL=P-3.0×ATR, bank=P+4.2×ATR.
- **L2** (add): when `high ≥ P + 0.5×ATR`, open add (50% L1 size). Add SL = `P + 0.5×ATR - 0.5×ATR = P` (breakeven for the add). Only triggered pre-bank.
- **L3** (recovery): if L2 SL hits, open recovery at that fill price (50% L1 size). SL = P - 3.0×ATR. Target = P + 3.77×ATR (median pump runup from MAE stats). L3 closes at target independently of main position.
- Bank event: if main banks while L2 add is open, L2 also partially banks (BANK_FRAC=35%).
- All active layers force-close at main position exit price.
- `SimTrade` fields: `add1_pnl`, `add2_pnl`, `layers` (1/2/3).

## Norse vs WalkForward Gap Analysis (2026-04-15)
- Norse: $10k→$1.917M (+19,078%), 365 days, 728 trades, 50.6% WR, PF 2.37, MaxDD 43.7%
- WF best result to date: $10k→$215k (+2,054%), 270 OOS days (9 windows), 479 trades, 57.6% WR, PF 3.51
- Extrapolated WF to 365 days ≈ $730k. Residual gap ≈ 2.6×.
- Gap causes: (1) 270 vs 365 OOS days, (2) wave_strength proxy vs real aggTrades taker flow, (3) Norse Optuna params are in-sample tuned (mild overfit), (4) late-window compounding with large equity hasn't fully materialized yet in 10-window structure.

## Khairul's Identity — Master Equation Reference (2026-04-15)

### The Identity
C(T) = C₀ · e^(n·T)

n = λ · g̃   where   g̃ = P·ln(1 + f·b) + (1-P)·ln(1 - f)

### Calibrated Constants (from 194× WF sim, 300 OOS days)
- n_daily = 0.01764 day⁻¹
- g̃ = 0.01131 per trade (measured) | 0.01452 (Kelly formula at avg f) — gap = score dist skew + pyramid non-linearity
- λ = 1.56 trades/day (468 trades / 300 days)
- P = 0.707 | b = 1.810 | f_avg ≈ 0.015 | Π = 1.696
- f* = 0.545 (full Kelly) | f/f* = 2.75% of Kelly

### Five Consistency Constraints
1. Kelly: f/f*=0.0275 → n=0.01764 (conservative operating point, full-Kelly n≈0.089)
2. Pump Phase: n_QUANTA=0.01764 < n_pump=0.082 → harvesting fraction of pump ✓
3. Grinold: n_max=IC²·λ·b·(1-ρ)=0.0366 → QUANTA at 48% of info-theoretic ceiling
4. MAE Geometry: b=1.810 → average runner exits at 6.09 ATR (self-consistent with exit params)
5. Pyramid: Π=1.696 → L3 weight=27.9% explainable by MAE recovery law (3.77 ATR p50)

### Derived Quantities
- b = PF × (1-P)/P = 4.37 × 0.293/0.707 = 1.810
- f* = (P·b - (1-P)) / b = 0.987 / 1.810 = 0.5453
- IC ≈ 2×(AUC-0.5) = 0.616 | Grinold IR = IC×√BR = 12.2 (annual) | actual Sharpe = 7.15
- Cross-coin ρ_eff = (7.15/12.2)² = 0.343 → 65.7% alpha erased by altcoin co-movement
- Average runner exit: (b×SL_ATR - BANK_FRAC×BANK_ATR) / (1-BANK_FRAC) = 6.09 ATR
- Pump phase time-to-bank: ln(1+4.20×ATR%) / 0.082 ≈ 21h at ATR=1.8%
- n_monthly = 0.01764 × 30 = 0.529 | e^0.529 = 1.697 (+69.7%/month)
- Capital doubles every: ln(2)/0.01764 = 39.3 days

### Named After
Habib Khairul — builder of QUANTA. Derived from original 194× OOS walk-forward simulation, April 2026.

## Thor Paper Execution Parity With WF Sim (2026-04-23)

- Thor paper trading is now intentionally forced to follow the WF sim execution model as the source of truth.
- For Thor paper positions, live-only execution layers are bypassed:
  - no risk-manager trade blocking
  - no loss-throttle sizing
  - no TWAP routing
  - no exchange-stop / slippage-blacklist side-effects
- Thor paper timing now advances on closed 5m bars using candle `high/low/close`, with integer `bars_open`, instead of wall-clock elapsed time.
- Thor paper fill math now matches WF sim exactly:
  - long entry fill: `price × (1 + slippage + commission)`
  - long exit fill: `price × (1 - slippage - commission)`
- Design intent:
  - if WF sim says a Thor paper trade should open, size, pyramid, timeout, bank, or trail a certain way, the live paper path should do the same thing under the same candle sequence.

## Dashboard Must Reflect Executed Thor State, Not Alert Picks (2026-04-24)

- The dashboard is now intentionally execution-centric for Thor.
- Active Thor position cards must read TP/SL/bank/trailing state from the persisted paper/live position object, not from `daily_picks`.
- `daily_picks` remain alert-layer suggestions only; they are not a valid source of truth for Thor v2 exit geometry.
- Thor parameter display must read from the active exit profile builder when available (`paper._build_thor_exit_profile()`), because legacy `thor_*` aliases can drift from the calibrated live engine.
- Design intent:
  - if the dashboard shows a Thor trade, the numbers should match the engine that is currently managing that trade.
