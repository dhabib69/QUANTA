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
