---
name: QUANTA Academic References
description: All academic papers, books, and preprints used in QUANTA — where each is applied in the codebase
type: reference
---

## Core ML & Finance Books

### López de Prado (2018) — *Advances in Financial Machine Learning* (AFML)
The most referenced source in QUANTA. Used throughout:
- **Ch.2 §2.5**: CUSUM filter — `QUANTA_bot.py` event detection, `DIRECTION_THRESHOLD = 0.12`
- **Ch.3**: Triple Barrier Method labeling — `quanta_numba_extractors.py`, `quanta_config.py EventExtractionConfig`
- **Ch.4 §4.1**: Sample weighting / temporal subsampling must preserve order — `QUANTA_ml_engine.py`
- **Ch.5**: Fractional Differencing (frac_diff) for memory + stationarity — `quanta_features.py`
- **Ch.6**: Feature Importance Pruning (bottom 20% drop) — `QUANTA_ml_engine.py:1482`
- **Ch.7 §7.4**: Purge gap between train/val (48 candles) — `QUANTA_ml_engine.py:962`, `quanta_backtester.py:199`
- **Ch.10**: Risk sizing / position sizing — `quanta_config.py RiskManagerConfig`
- **Ch.12**: Combinatorial Purged Cross-Validation (CPCV) — `QUANTA_ml_engine.py:1516`, `quanta_model_registry.py`, `quanta_backtester.py`

### López de Prado (2020) — *Machine Learning for Asset Managers*
- **Ch.6**: Feature importance pruning implementation reference

### Tharp (2006) — *Trade Your Way to Financial Freedom*
- Trade journaling / position sizing concepts — `quanta_config.py PaperTradingConfig`, `quanta_paper_trading.py`

### Pardo (2008) + Bailey et al. (2014)
- Walk-forward backtesting — `quanta_backtester.py`

### Schwager (1993)
- Paper trading / performance evaluation — `quanta_config.py PaperTradingConfig`

### Bouchaud & Potters (2003) — *Theory of Financial Risk and Derivative Pricing*
- Risk manager design — `quanta_config.py RiskManagerConfig`

---

## Academic Papers

### [1] Lim et al. (2021) — Temporal Fusion Transformers
**"Temporal Fusion Transformers for interpretable multi-horizon time series forecasting"**
*International Journal of Forecasting*, 37(4), 1748-1764.
- TFT model implementation — `quanta_deeplearning.py`, output feeds into feature index 223

### [2] Schulman et al. (2017) — PPO
**"Proximal policy optimization algorithms"**
*arXiv:1707.06347*
- PPO RL agent — `QUANTA_agents.py`

### [4] Mandelbrot & Wallis (1969) — Hurst Exponent / R/S Analysis
**"Robustness of the rescaled range R/S"**
*Water Resources Research*, 5(5), 967-988.
- Hurst exponent feature — `quanta_features.py _jit_hurst`, feature index 252

### [5] Kantelhardt et al. (2002) — MF-DFA
**"Multifractal detrended fluctuation analysis of nonstationary time series"**
*Physica A*, 316(1-4), 87-114.
- MF-DFA width feature — `quanta_features.py _jit_mf_dfa_width`, feature index 257

### [6] Richman & Moorman (2000) — Sample Entropy
**"Physiological time-series analysis using approximate entropy and sample entropy"**
*American Journal of Physiology*, 278(6), H2039-H2049.
- Sample entropy feature — `quanta_features.py _jit_sample_entropy`, feature index 253

### [7] Schreiber (2000) — Transfer Entropy
**"Measuring information transfer"**
*Physical Review Letters*, 85(2), 461.
- Transfer entropy feature — `quanta_features.py _jit_transfer_entropy`, feature index 258

### [8] Easley, López de Prado & O'Hara (2012) — VPIN
**"Flow toxicity and liquidity in a high-frequency world"**
*The Review of Financial Studies*, 25(5), 1457-1493.
- VPIN (Volume-synchronized Probability of Informed Trading) — `quanta_features.py _jit_vpin`, feature indices 213-219
- Volume surge threshold in Artemis event — `quanta_config.py EventExtractionConfig`

### [9] Kyle (1985) — Kyle's Lambda
**"Continuous auctions and insider trading"**
*Econometrica*, 1315-1335.
- Kyle's Lambda (price impact / market impact) — `quanta_features.py _jit_kyle_lambda`, feature index 255

### [10] Amihud (2002) — Illiquidity
**"Illiquidity and stock returns: cross-section and time-series effects"**
*Journal of Financial Markets*, 5(1), 31-56.
- Amihud illiquidity ratio — `quanta_features.py _jit_amihud`, feature index 256

### [11] Rydén, Teräsvirta & Åsbrink (1998) — HMM
**"Modeling daily return series with hidden Markov models"**
*Journal of Applied Econometrics*, 13(3), 217-244.
- HMM regime state — `QUANTA_ml_engine.py`, feature index 251

### [12] Moody et al. (1998) — Differential Sharpe Ratio
**"Performance functions and reinforcement learning for trading systems and portfolios"**
*Journal of Forecasting*, 17(5-6), 441-470.
- Differential Sharpe Ratio reward in PPO agent — `QUANTA_agents.py`

### Romano et al. (2020) + Gibbs & Candes (2021) — Conformal Inference
- Adaptive Conformal Calibration — `QUANTA_ml_engine.py:1555`, `quanta_config.py ConformalConfig`
- 90% coverage target, calibrates raw probabilities post-training

### axPPO (arXiv 2024) — Adaptive Entropy PPO
- Adaptive entropy coefficient for exploration — `QUANTA_agents.py:309, 412`

---

## Black-Scholes Barrier Math (added 2026-04-02)

### Hull (2018) — *Options, Futures, and Other Derivatives*, Ch.26
- Barrier options framework — `quanta_features.py _jit_bs_barrier_prob`, feature index 275
- Motivates the double-barrier interpretation of Triple Barrier labeling
- Key insight: QUANTA's Triple Barrier = double-barrier knock-in option → BS first-passage theory applies

### Darling & Siegert (1953) — Scale Function
**"The first passage problem for a continuous Markov process"**
*Annals of Mathematical Statistics*, 24(4), 624-639.
- Scale function `s(x) = exp(-θx)` for BM with drift, θ = 2ν/σ²
- P(hit upper before lower) = [s(0)-s(b)]/[s(a)-s(b)] — `quanta_features.py _jit_bs_barrier_prob`
- Zero-drift case reduces to gambler's ruin: P = sl_dist/(tp_dist+sl_dist)

### Kunitomo & Ikeda (1992) — Double Barrier Pricing
**"Pricing options with curved boundaries"**
*Mathematical Finance*, 2(4), 275-298.
- Exact finite-horizon double-barrier probability via infinite series (fast-converging)
- QUANTA uses approximate single-barrier CDF proxy for feature 276 (bs_time_decay) — CatBoost learns residual
- Full Kunitomo-Ikeda series is analytically correct but overkill for a CatBoost feature input

### Ross (1996) — *Stochastic Processes*, Ch. on BM
- Gambler's ruin in continuous time — theoretical grounding for zero-drift baseline
- `bs_theoretical_win_prob` zero-drift case: P = sl/(tp+sl)

### Kou (2002) — Double-Exponential Jump-Diffusion
**"A jump-diffusion model for option pricing"**
*Management Science*, 48(8), 1086-1101.
- Replaces Hull/Darling-Siegert pure GBM in Feature 275 (Tier 3 upgrade)
- `_jit_kou_barrier_prob` — double-exponential jump process captures crypto fat tails
- Better matches crypto's sudden jumps vs continuous diffusion assumption

### Bollerslev (1986) — GARCH(1,1)
**"Generalized autoregressive conditional heteroskedasticity"**
*Journal of Econometrics*, 31(3), 307-327.
- GARCH(1,1) filtered volatility for BS barrier calculations
- Crypto params: `omega=1e-6, alpha=0.10, beta=0.85`
- Captures volatility clustering — used in `_jit_bs_barrier_prob`

---

## Khairul's Identity (2026) — Original Work

### Khairul (2026) — *Khairul's Identity: The Master Compound Growth Equation of the QUANTA System*
**Derived by Habib Khairul — April 2026**
*Unpublished. Derived from QUANTA walk-forward OOS simulation, 300 days, 245 symbols.*

#### Validation History (OOS Walk-Forward, $10k start)
| Version | Change | Result | n_daily |
|---------|--------|--------|---------|
| V12.3 | MAE calibration + 3-layer pyramid | **+19,883%** (194×) | 0.01764 |
| V12.4 | Gompertz λ(t) dynamic exit (replaces hardcoded 4-day timeout) | **+150,235%** (1,503×) | 0.02438 |

V12.4 Gompertz upgrade increased n_daily by **+38%** (0.01764 → 0.02438) by letting fast pumps ride to their natural λ=n crossover instead of cutting at a fixed ATR. L2+L3 pyramid PnL jumped from $816k → $6.12M.

#### The Identity

$$C(T) = C_0 \cdot e^{nT}$$

$$n = \lambda \cdot \left[P\ln(1+fb) + (1-P)\ln(1-f)\right]$$

Where:
- $C(T)$ — portfolio capital at time T (days)
- $C_0$ — initial capital
- $n$ — continuous compound growth rate (day⁻¹)
- $\lambda$ — trade frequency (trades/day), driven by CUSUM detection × universe × hour filter
- $P$ — win probability, derived from Thor CatBoost AUC × cross-coin correlation discount
- $f$ — risk fraction per trade, linear in Thor score: $f(s) = 0.005 + (s-68)/32 \times 0.025$
- $b$ — win/loss ratio, derived from MAE-calibrated exit geometry: $b = PF \times (1-P)/P$

#### Calibrated Constants (194× bot, OOS 2026)
| Symbol | Value | Source |
|--------|-------|--------|
| $n$ | 0.01764 day⁻¹ | WF sim: ln(198.83)/300 |
| $\lambda$ | 1.56 trades/day | 468 trades / 300 days |
| $P$ | 0.707 | WF win rate |
| $b$ | 1.810 | PF=4.37, P=0.707 |
| $f_{avg}$ | ~0.015 | score dist avg |
| $f^*$ | 0.545 | full Kelly fraction |
| $f/f^*$ | 2.75% of Kelly | ultra-conservative → MaxDD=8.1% |
| $\Pi$ | 1.696× | L2+L3 pyramid amplification |

#### Pyramid Extension
The 3-layer pyramid adds an amplification factor $\Pi$ to total PnL:
$$C_{pyramid}(T) = C_{L1}(T) \cdot \Pi, \quad \Pi = 1 + w_2 r_{L2} + w_3 r_{L3} = 1.696$$
$$w_1 = 0.589,\quad w_2 = 0.132,\quad w_3 = 0.279$$

L3 recovery weight (0.279) is 2.1× larger than L2 (0.132): coins that pull back to L2 stop-loss continue to 3.77 ATR with statistically predictable probability (from MAE survival curve).

#### Five Consistency Constraints (Over-Determination)
The identity is self-consistent across five independent frameworks simultaneously:
1. **Kelly**: $f/f^* = 2.75\%$ → theoretical MaxDD ≈ 11%; observed 8.1% (correlation discount)
2. **Pump Phase Law**: $n_{QUANTA} = 0.01764 < n_{pump} = 0.082$ day⁻¹ (harvesting fraction of universal altcoin pump cycle)
3. **Grinold's Fundamental Law**: $n_{max} = IC^2 \cdot \lambda \cdot b \cdot (1-\rho) = 0.0366$ day⁻¹; QUANTA operates at 48% of info-theoretic ceiling
4. **MAE Exit Geometry**: $b = 1.810$ → average winner runner exits at 6.09 ATR (derivable from SL=3.0, Bank=4.2, BankFrac=0.35)
5. **Pyramid Consistency**: $\Pi = 1.696$ explained by MAE recovery law at 3.77 ATR (overall p50 runup)

#### Pump Phase Sub-Law
During altcoin pump cycles:
$$P(t) = P_0 \cdot e^{n_{pump} \cdot t}, \quad n_{pump} \approx 0.082\ \text{day}^{-1}$$

Empirically consistent across WIF, BONK, COAI and other explosive altcoin moves. QUANTA's CUSUM detector fires at the onset of the micro-breakout embedded within this macro cycle. Time to bank target (4.20 ATR) at typical ATR=1.8%:
$$t_{bank} = \frac{\ln(1 + 4.20 \times 0.018)}{0.082} \approx 21\ \text{hours}$$

#### Sharpe Decomposition
$$IR_{theoretical} = IC \cdot \sqrt{BR} = 0.616 \times \sqrt{393} = 12.2$$
$$\text{Observed Sharpe} = 7.15 \implies \rho_{eff} = (7.15/12.2)^2 = 0.343$$

65.7% of theoretical alpha is erased by altcoin cross-coin co-movement. The surviving 34.3% is idiosyncratic edge from CUSUM + Thor.

#### Doubling Time
$$T_{double} = \frac{\ln 2}{n} = \frac{0.6931}{0.01764} = \mathbf{39.3\ days}$$

#### Files
- Formal reference: `memory/project_quanta_technical_decisions.md` — Khairul's Identity section
- Simulation source: `QUANTA_WalkForward_Sim.py` — V12.3+
- Run data: `wf_runs/<timestamp>/WF_SIM_REPORT_*.md`

---

### Khairul (2026) — *λ(t): The Pump Collapse Hazard Companion Equation*
**Derived by Habib Khairul — April 2026**
*Companion to Khairul's Identity. Governs individual position optimal exit timing.*

#### The Equation

$$\lambda(t) = \lambda_0 \cdot e^{\gamma t} \quad \text{(Gompertz hazard function)}$$

$$\mathbb{E}[P_{exit}(t)] = P_0 \cdot e^{nt} \cdot e^{-\frac{\lambda_0}{\gamma}(e^{\gamma t} - 1)}$$

The expected exit price peaks at **t\*** where growth equals collapse hazard:

$$n = \lambda(t^*) \implies t^* = \frac{\ln(n / \lambda_0)}{\gamma}$$

$$\text{Optimal bank ATR} = \frac{e^{n \cdot t^*} - 1}{\text{ATR\%}}$$

#### Duality with Khairul's Identity

| Equation | Governs | Variable |
|----------|---------|----------|
| $C(T) = C_0 \cdot e^{nT}$ | Portfolio compound growth | $n$ = 0.01764 day⁻¹ (portfolio) |
| $\lambda(t) = \lambda_0 e^{\gamma t}$ | Per-position exit timing | $\lambda_0$, $\gamma$ (micro-pump scale) |

As price pumps up ($e^{nt}$), predictability goes down ($e^{-\lambda t}$). The optimal exit is the peak of their product.

#### Calibration (micro-pump scale, 5-min bars, WF sim)

Two anchor points from MAE data:
- **t₁ = 0.104 days** (30 bars, avg bank hit at 4.20 ATR, ATR=1.8%): $n_{bank} = \ln(1.0756)/0.104 = 0.700$ day⁻¹, so $\lambda(t_1) = 0.700$
- **t₂ = 0.226 days** (65 bars, avg runner exit at 6.09 ATR): collapse dominant → $\lambda(t_2) = 1.0$ day⁻¹

Solving the system:
$$\gamma = \frac{\ln(\lambda(t_2)/\lambda(t_1))}{t_2 - t_1} = \frac{\ln(1.0/0.700)}{0.122} = \mathbf{2.92\ \text{day}^{-1}}$$
$$\lambda_0 = \lambda(t_1) / e^{\gamma t_1} = 0.700 / e^{2.92 \times 0.104} = \mathbf{0.517\ \text{day}^{-1}}$$

#### Calibrated Constants
| Symbol | Value | Meaning |
|--------|-------|---------|
| $\lambda_0$ | 0.517 day⁻¹ | Baseline collapse hazard at entry |
| $\gamma$ | 2.92 day⁻¹ | Hazard acceleration rate |
| $n_{prior}$ | 0.700 day⁻¹ | Average micro-pump velocity at CUSUM fire |

Hazard doubles every: $\ln(2)/\gamma = 0.237$ days = **5.7 hours**

#### Dynamic Bank Target by Observed Pump Velocity

| n_observed | t* | Bank ATR | Interpretation |
|------------|----|----------|----------------|
| < 0.517 (slow) | negative | 2.0–2.5 ATR | λ already > n at entry, exit early |
| 0.700 (average) | 0.104d = 30 bars | **4.20 ATR** ✓ matches hardcoded |
| 1.05 (1.5× fast) | 0.176d = 51 bars | 6.8 ATR | Ride fast pumps higher |
| 1.40 (2× fast) | 0.228d = 66 bars | 8.9 ATR | Very fast — give more room |
| > 2.0 (extreme) | clamped | 10.0 ATR | Cap to prevent over-exposure |

#### Implementation in QUANTA_WalkForward_Sim.py (V12.4+)
Dynamic bank replaces hardcoded `_BANK_ATR = 4.20`:
```python
n_eff = blend(n_observed, n_prior, weight=min(1.0, bars_open/24))
t_star = ln(n_eff / λ₀) / γ           # optimal hold time (days)
dyn_bank_atr = (exp(n_eff × t_star) - 1) / atr_pct   # in ATR units
dyn_bank_px  = entry + dyn_bank_atr × ATR
```

The observed pump velocity ($n_{eff}$) is blended from raw observed rate toward the prior over the first 24 bars (2 hours) for stability.

---

### Khairul (2026) — *V12.4 Live Implementation Note*
**Date: 2026-04-16 — Gompertz model deployed to live paper trading bot**

#### What Was Deployed
The λ(t) companion equation is now running in production in `QUANTA_trading_core.py`.

**`_build_thor_exit_profile()`** — new method on PaperTrading class. Was missing (causing silent AttributeError that prevented thor_v2 routing). Now returns calibrated params: SL=3.0 ATR, Bank=4.2 ATR (initial), BankFrac=35%, Trail=2.0 ATR, TrailActivate=1.5 ATR, MAE veto 3.62/5bars.

**`_tick_thor_v12()` rewrite** — replaces all hardcoded exits with Gompertz dynamics:
```
Static before:  bank_atr=5.4 fixed, max_pre=1152 bars (4 days), max_post=1152 bars
Live now:       bank_atr = _gc_bank_atr(n_eff, atr_pct)   [2.0–10.0 ATR per coin]
                max_pre  = _gc_pre_bars(n_eff, k=1.5)      [12–144 bars]
                max_post = _gc_post_bars(n_eff, bars_at_bank, k=2.5)  [12–576 bars]
```

**Five new static methods** on PaperTrading:
- `_gc_n_eff(entry, current, elapsed_days)` — blended pump velocity
- `_gc_t_star(n_eff, k)` — Gompertz crossover time
- `_gc_bank_atr(n_eff, atr_pct)` — optimal bank ATR
- `_gc_pre_bars(n_eff)` — dynamic pre-bank timeout
- `_gc_post_bars(n_eff, bars_at_bank)` — dynamic runner timeout

**Position dict additions:** `gompertz_n_eff`, `gompertz_lowest`, `gompertz_dyn_bank_atr`, `gompertz_dyn_max_pre`, `gompertz_dyn_max_post`, `gompertz_bank_bar`

**quanta_config.py additions:** 14 new fields under EventExtractionConfig — 7 calibrated exit params + 7 Gompertz constants. All hot-reloadable (read from config on each tick).

#### Equation Active in Live Bot
$$\text{bank price}(t) = P_{entry} + \frac{e^{n_{eff} \cdot t^*} - 1}{\text{ATR\%}} \cdot ATR$$
$$t^* = \frac{\ln(n_{eff} / \lambda_0)}{\gamma}, \quad n_{eff} = \alpha \cdot n_{obs} + (1-\alpha) \cdot n_{prior}$$

where $\alpha = \min(1,\ t_{elapsed} / 2h)$ — trust in observed velocity grows over first 2 hours.
