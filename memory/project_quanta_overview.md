---
name: QUANTA Project Overview
description: What QUANTA is, full architecture, execution flow, and system design
type: project
---

## What QUANTA Is

Institutional-grade algorithmic trading engine for Binance Futures (crypto). Built by Habib Khairul. Goal: production-ready capital deployment.

**Core design philosophy:**
- 7 CatBoost specialists, each with a distinct inductive bias (domain specialization)
- Ensemble weighted by information-theoretic certainty, not raw votes
- PPO RL agent sizes every trade — acts as position size oracle (not a gate)
- Triple Barrier labeling (López de Prado 2018) — correct barrier crossing, NOT endpoint checking
- All math backed by academic papers, not guesses

---

## The 7 Specialist Agents

| Agent | Trigger | Direction | Domain Mask |
|-------|---------|-----------|-------------|
| Athena | CUSUM_pos + new highest high | Long only | domain_trend |
| Ares | CUSUM_neg + new lowest low | Short only | domain_trend_short |
| Hermes | Volatility squeeze + expansion | Both | domain_mean_revert |
| Artemis | CUSUM_pos + vol_surge + NOT new_high (bull) / CUSUM_neg + vol_surge + NOT new_low (bear) — stealth accumulation/distribution | Both | domain_volatility |
| Chronos | CUSUM both + RSI extreme reversal | Both | domain_macro |
| Hephaestus | Price pattern detection | Both | domain_structural |
| Nike | Single candle: range>2×ATR AND vol>1.5×avg20 AND body_eff>0.5 (no CUSUM) | Both | domain_impulse |

Nike replaces Divergence/Apollo (dropped 2026-04-01). Apollo was dual-CUSUM, too similar to Athena+Ares.

---

## Feature Vector — 278 Dimensions

Single source of truth: `Config.base_feature_count = 278` in quanta_config.py.
All code reads `self.cfg.BASE_FEATURE_COUNT`. Never hardcode 278.

| Index Range | Count | What |
|-------------|-------|------|
| 0-48 | 49 | Per-TF features: 7 TFs × 7 features each (RSI, MACD, BB, ADX, ATR, trend, strength) |
| 49-51 | 3 | Cross-TF consensus, RSI spread, MACD alignment |
| 52 | 1 | Weighted consensus |
| 53-57 | 5 | RSI analysis |
| 58-60 | 3 | MACD |
| 61-65 | 5 | Volume |
| 66-69 | 4 | ATR |
| 70-72 | 3 | ADX |
| 73-75 | 3 | Bollinger Bands position |
| 76-79 | 4 | Strength progression |
| 80-83 | 4 | Market regime |
| 84-86 | 3 | Momentum confluence |
| 87-106 | 20 | Spike-dump-reversal |
| 107-115 | 9 | Time + session (UTC): hour_sin, hour_cos, day_sin, day_cos, norm_hour, norm_day, US_session, Asia_session, funding_window |
| 116-136 | 21 | Enhanced volatility per TF (3×7) |
| 137-164 | 28 | Enhanced momentum per TF (4×7) |
| 165-171 | 7 | Volume ratio per TF |
| 172-185 | 14 | Drift detection per TF (2×7) |
| 186-192 | 7 | Multi-TF returns |
| 193-199 | 7 | VPIN order flow toxicity per TF |
| 200-206 | 7 | Fractional differencing per TF |
| 207-213 | 7 | Taker flow imbalance per TF |
| 214-230 | 17 | Order book features (bid/ask depth, spread, imbalance proxy) |
| 231 | 1 | HMM regime state (0-4) |
| 232-238 | 7 | Advanced research: Hurst, SampleEntropy, KyleLambda, Amihud, MF-DFA, TransferEntropy, QRE |
| 239-245 | 7 | Sentiment: fng_norm, extreme_fear, extreme_greed, news_score, news_volume_norm, coin_score, coin_magnitude |
| 246-249 | 4 | Futures X-ray: OI, L/S ratio, spec index, funding velocity |
| 248-253 | 6 | Stat arb (always NaN during training — no cross-pair offline data) |
| 254-256 | 3 | On-chain whale analytics (always NaN during training — no offline API) |
| 257 | 1 | Cross-asset GNN embedding |
| 258-267 | 10 | Delta features: RSI_δ, MACD_δ, BB_δ, ADX_δ, Vol_δ, RSI_accel, MACD_accel, ATR_δ, Strength_δ, VPIN_δ |
| 268-269 | 2 | Delta continuation |
| 270-274 | 5 | Impulse features (Nike): body_eff, taker_flow_persist, pre_impulse_r2, atr_rank, depth_delta |
| 275 | 1 | **bs_theoretical_win_prob** — P(TP before SL) via Hull/Darling-Siegert scale function. Median barriers (TP=1.5, SL=1.0 ATR). Zero-drift baseline = 0.40 for Athena. |
| 276 | 1 | **bs_time_decay** — approx P(crossing barrier in 48 bars remaining). Single-barrier CDF proxy using tanh. |
| 277 | 1 | **bs_implied_vol_ratio** — sigma_implied / sigma_realized from avg bars-to-hit history. Starts 1.0 (neutral). Clipped [0.1, 10]. |

**Important:** Features 248-250 (stat arb) and 254-256 (on-chain) are always NaN during training. Features 275-277 are always computable from OHLCV — no NaN risk. Feature 277 starts at 1.0 until ≥5 completed trades per symbol accumulate.

---

## Full Execution Flow: Cold Start to First Trade

### Phase 1 — Boot (seconds)
`main.py` → `Bot().__init__()` → `Bot().run()`

Initialized in `__init__`:
- `Config` → all parameters
- `BinanceAPIEnhanced` → exchange connection
- `MultiTimeframeAnalyzer` → aggregates 5m into 15m/1h/4h
- `DeepMLEngine` → 7 specialists + TFT + HMM + Optuna studies
- `PaperTrading` → ledger with balance persistence
- `PPOAgent` + `PPOMemory` → RL components
- `data_queue`, `retrain_queue`, `result_queue` → pipeline assembly line
- `WSEventProducer` → websocket candle listener

Nothing is trained. Nothing is trading. Empty hands.

### Phase 2 — Symbol Selection (seconds)
`get_research_backed_coins(limit=200)` → asks Binance for top coins by volume/liquidity.
Filters stablecoins, low-volume, delisted. Result: ~30-50 quality futures pairs.
Fallback chain: research_backed → top_movers → get_pairs → hardcoded 20 majors.

### Phase 3 — Cache Warmup (5-30 min, first run only)
`warmup_cache_research(missing_coins, days=365)` → downloads 365 days of 5m OHLCV for all coins.
Stored as Feather format (columnar compressed, 2.5× faster than parquet).
Subsequent runs only top up missing candles.
~21,000 candles per coin × 50 coins = ~1M rows per coin.

### Phase 4 — Initial Training (5-15 min, first run only)
`_train_models(clean_retrain=True)` fires when `ml.is_trained == False`.

Per coin, for 180-day window:
1. Compute 275 features per candle
2. Run CUSUM detector to find statistically significant events
3. Label each event with Triple Barrier (TP/SL hit within 48 bars → 1/0, timeout → -1 excluded)
4. Route events to correct specialists
5. **Optuna** runs first: 20 CPU trials × 300 iterations × 3000 samples per specialist — TPE sampler learns optimal depth/lr/l2/subsample
6. CatBoost trains with best Optuna params at full iterations
7. Calibration check: val AUC > threshold or skip gate if <20 val events
8. HMM 3-state regime model (bull/range/bear) trains per-symbol, sorted descending by mean log-return
9. TFT trains, quality gate: val AUC > 0.55 or feature 223 stays zeroed
10. PPO starts from random weights — learns from live outcomes

`ml.is_trained = True`. Training done.

### Phase 5 — Live Feeds Start
Three background feeds launch simultaneously:
- WebSocket: live 5m candle streams for all symbols → `data_queue`
- Sentiment feed: Fear & Greed, news score, coin-specific sentiment
- OI + Funding Rate feed: Binance futures open interest, long/short ratio, funding

### Phase 6 — Consumer Loop (runs forever)
Worker thread pulls batches from `data_queue`. For each batch:

**Step A: 7 Specialists Predict**
Each specialist runs `predict_proba()` on the 275-feature vector (subsetted to its domain mask).
Output: 7 × (p_bear, p_bull) arrays.

**Step B: Ensemble Weighting**
1. **Shannon entropy weighting** — certainty = 1 - H, weight ∝ certainty^1.5. Confident models dominate uncertain ones.
2. **HMM regime routing** — per-item (not per-batch). Each coin's own HMM gives regime 0-4. Specialist multipliers differ by regime (e.g. Athena boosted in trending, Hermes in ranging). `regime_mults_matrix` shape (7, batch_len).
3. **Brier score calibration** — rolling deque last 500 predictions. `mult = max(0.7, min(1.3, 1.3 - (brier/0.25)*0.6))`. Reset on retrain.
4. **Event overlap discount** — if ≥3 specialists certainty>0.7 in same direction → 0.85× (not independent votes on extreme moves).
5. **Entropy veto** — `H(p_ens) > 0.85 bits → force [0.5, 0.5]` (NEUTRAL). Corresponds to p ∈ ~[0.26, 0.74].

**Step C: 4 Pre-PPO Filters**
1. Direction threshold: |score| > 0.12 or NEUTRAL
2. Confidence floor: ≥ 50%
3. Magnitude: expected move ≥ 1%
4. CUSUM gate: log returns since last reset must cross threshold — no trading on sideways price

**Step D: Streak Boost**
Same symbol + direction within 15 min → streak counter. Persistent signals get asymptotic boost: +20% conf max, +50% magnitude max. `conf_mult = 1 + 0.20*(1 - exp(-0.5*(streak-1)))`.

**Step E: PPO (Heimdall) — Size Oracle**
State = 275 features + 7 specialist probs + divergence + entropy + HMM regime = 226-dim.
PPO outputs action ∈ {0=HOLD, 1=LONG, 2=SHORT} + value estimate.

PPO is a **position size oracle**, not a gate. ML ensemble owns the direction decision. PPO cannot block trades.

```
val_signal = sigmoid(ppo_value)  # [0,1]

PPO agrees with ML  → ppo_size_mult = 1.0 + 1.0 * val_signal   # [1.0, 2.0]
PPO says HOLD       → ppo_size_mult = 0.5 + 0.5 * val_signal   # [0.5, 1.0]
PPO contradicts ML  → ppo_size_mult = 0.25 + 0.25 * val_signal # [0.25, 0.5]
```

`passes_gate = ml_conf >= threshold` — purely ML-driven, PPO has no say.

`ppo_size_mult` is passed to `open_position()` and applied to Kelly-calculated notional (capped at MAX_RISK).

### Phase 7 — Trade Placed
`paper.open_position(symbol, price, direction, confidence, volatility, ppo_size_mult, barrier_rr, bs_edge)`

Position gets:
- Entry price (live market)
- TP1 = 0.5× magnitude (low training weight: 0.3)
- TP2 = 1.0× magnitude
- TP3 = 1.5× ATR (full target)
- SL = 1.0× ATR (training weight: 2.0 — wrong predictions hurt 6× more than marginal wins)
- Commission 4bps + slippage 2bps deducted

**Kelly sizing (v11.5b+):**
- `b = barrier_rr` (per-specialist TP/SL ratio — dynamic, replaces hardcoded 2.0)
- BS edge filter: if `(ml_conf/100 - sl/(tp+sl)) < 0.02`, Kelly probability penalised
- PPO size multiplier applied after Kelly, before risk cap

Opportunity logged to RL buffer with 30-min cooldown per symbol/direction.

### Phase 8 — PPO Learns (4h later)
RL memory checks outcomes 48 bars after signal. Measures actual TP/SL hit.
Reward = Differential Sharpe Ratio (improves risk-adjusted return curve, not just P&L).
PPO trains on accumulated experiences via `ppo_agent.update(ppo_memory)`.
When enough completed trades accumulate → background thread triggers full CatBoost retrain.

---

## Key Constants

| Constant | Value | Where |
|----------|-------|-------|
| BASE_FEATURE_COUNT | 278 | quanta_config.py |
| Commission | 4bps (0.0004) | quanta_config.py |
| Slippage | 2bps (0.0002) | quanta_config.py |
| Triple Barrier TP | 1.5 ATR | quanta_config.py (Athena settings, all agents use same) |
| Triple Barrier SL | 1.0 ATR | quanta_config.py |
| max_bars | 48 bars (~4h at 5m) | quanta_config.py |
| Purge gap | 48 candles | = max(all agent max_bars) |
| Val split | 20% | CATBOOST_VAL_SPLIT |
| GPU_MAX_SAMPLES | 10,000 | MX130 2GB VRAM constraint |
| CPU_CHUNK_SIZE | 5,000 | |
| OPTUNA_N_TRIALS | 20 | QUANTA_ml_engine.py |
| OPTUNA_SEARCH_ITERS | 300 | QUANTA_ml_engine.py |
| OPTUNA_MAX_SEARCH_ROWS | 3,000 | QUANTA_ml_engine.py |
| TP1_WEIGHT | 0.3 | QUANTA_trading_core.py |
| SL_WEIGHT | 2.0 | QUANTA_trading_core.py |
| DIRECTION_THRESHOLD | 0.12 | local heuristic (misattributed to LdP in code — NOT from paper) |
| ENTROPY_VETO_THRESHOLD | 0.85 bits | H > 0.85 → p ∈ [~0.26, ~0.74] |
| DISAGREEMENT_STD_NORM | 0.20 | derived: sqrt(0.25/7) ≈ 0.189 for 7 independent binary classifiers |

---

## What PPO Does and Doesn't Do

**PPO DOES:**
- Output `ppo_size_mult ∈ [0.25, 2.0]` — scales Kelly-calculated notional up or down
- Agree + high value → up to 2× size (go big on conviction)
- HOLD + any value → 0.5–1.0× (cautious, still trades)
- Contradict + any value → 0.25–0.5× (very small, still trades)
- Learn from realized outcomes (4h lag): reward = outcome_sign × move_mag × ppo_size_mult

**PPO does NOT:**
- Block or gate any trade — ML passes gate independently
- Modify `ml_conf` in any way
- Pick direction (that's the ML ensemble's job)
- Train on synthetic data — only real trade outcomes

---

## Architecture: What Makes It Non-Standard

1. **Domain specialization** — each specialist sees only features relevant to its trigger type (domain mask). Athena never sees impulse features. Nike never sees macro features.

2. **Triple Barrier labels** — not "did price go up" but "did price hit TP or SL first within 48 bars". This correctly models the actual trading objective.

3. **CUSUM event detection** — training samples come from statistically significant price events, not every candle. Models learn on meaningful data, not noise.

4. **Optuna per-specialist** — each specialist gets its own TPE study that accumulates knowledge across retrains. Studies persist to `models/optuna_studies/`.

5. **Ensemble entropy veto** — ensemble uncertainty itself gates the trade. A 50/50 ensemble gets vetoed regardless of individual specialist confidence.

6. **Brier score weighting** — models weighted by how well-calibrated they've been recently (rolling last 500). Reset on retrain so old model's history doesn't contaminate new model.

7. **PPO ramp veto** — prevents RL collapse in early training when ML accuracy may be < 50%.
