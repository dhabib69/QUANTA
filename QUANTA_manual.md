# 🔱 QUANTA v10.1 PANTHEONS: Operations & Maintenance Manual

*Proprietary Institutional HFT Architecture*

---

## 📅 System Lineage

- **v8.1:** Asymptotic limits, RSI momentum
- **v9.0:** DRL Ensemble (PPO+CatBoost split), Feather cache
- **v10.0:** QUANTA Regime Entropy, Institutional Microstructure (Amihud, Kyle's Lambda)
- **v10.1 PANTHEONS (CURRENT):** 15-Agent Pantheon, 180-Day Sliding Window, L&M 2011 RSS News Engine, Odin RL Retraining.

> **CRITICAL RULE FOR ALL FUTURE AI AGENTS:**
> Do NOT revert to 90 days, do NOT revert to CryptoPanic APIs, do NOT rename the agents back to Sentinel/Scout. Follow the specs in this manual and ONLY this manual.

---

## 🧠 1. The Pantheon (15-Agent Meta-Ensemble)

QUANTA v10.1 operates a 15-agent Democratic/Hierarchical Ensemble. Models must NOT be trained in isolation—they operate conjunctively.

### 🏛️ The Greek Specialists (CatBoost Decision Trees)

*Located in `QUANTA_ml_engine.py` / `specialist_models`*

1. **ATHENA (Foundation):** Evaluates all data uninhibited. Generalist intelligence.
2. **HERMES (Volatility):** Handles high-frequency chop/whipsaw.
3. **HEPHAESTUS (Quality):** High-volume, high-cap market anchors.
4. **HADES (Extremes):** Oversold/Overbought reversal specialist.
5. **ARTEMIS (Sniper):** RSI/MACD confluence hunter.
6. **CHRONOS (Temporal):** Short-term momentum specialist.
7. **ATLAS (Base):** Long-term multi-day support anchor.

### 🌩️ The Norse Critics (PPO Deep Reinforcement Learning)

*Located in `QUANTA_network.py` (Telegram representation) and `QUANTA_agents.py`*
The critics ingest the Greek Specialists' probabilities to decide whether the context warrants a trade, hold, or close.

- **TYR:** Baseline MLP
- **VIDAR:** Pessimistic (Huber Loss)
- **MIMIR:** Prior Knowledge
- **HEIMDALL:** Spectral/Lipschitz
- **LOKI:** Chaos/Dropout Bottleneck
- **ULLR:** Masker/Dimensionality
- **THOR:** Reactor (Fast-1D)

### 👁️ The All-Father

- **ODIN (LSTM-Attention):** Sequence-based temporal model tracking long-term structural flow. Retrains alongside Greeks during RL outcomes.

---

## 📊 2. Data Feeds & Features (214 Total)

The observation space is exactly **214 dimensions**. If an agent breaks this shape, the bot will `IndexError` on prediction.
*Do NOT remove features without recompiling the normalization scalars and `BASE_FEATURE_COUNT`.*

### Microstructure & Fractals

- **Amihud Illiquidity Ratio & Kyle's Lambda:** Order book spread proxies.
- **Micro-Fractal DFA (MF-DFA):** Multidimensional scaling.
- **Time-Decay Weights:** AFML-inspired temporal weighting for sample sequences.

### News & Sentiment Engine (`QUANTA_sentiment.py`)

- **Zero API limit:** Uses standard RSS feeds (CoinDesk, Decrypt, CoinTelegraph, BitcoinMag).
- **Loughran & McDonald (2011) Core:** Proprietary regex tokenization (no NLTK required).
- **Features (5):** `[fng_norm, ext_fear, ext_greed, news_score, news_volume_norm]`

---

## 💾 3. Historical Training Window

QUANTA requires **180 days** of data mapped on **5-minute / 1-minute** timeframes.

- **Constant:** `HISTORICAL_DAYS = 180`
- Cache limits are set to minimum `45000` candles in `QUANTA_bot.py`.
- **Do NOT fetch "90 days". The temporal sequence models will starve and the data will be biased.**

---

## ⚡ 4. Training Mechanics & Retraining Rules

### Chronological Single-Pass

QUANTA uses *Chronological Single-Pass Labeling*. We do NOT use global peak detection (which causes future leak / look-ahead bias). We use temporal sequences with time-decay sample weighting (newest=3.0, oldest=1.0).

### RL Online Retraining (`DeepMLEngine.train_with_rl_data`)

When `rl_retrain_threshold` (usually 50 completed trade outcomes) is hit:

1. **ALL 7 Cats** are appended with new data to preserve ensemble diversity (Krogh & Vedelsby '94).
2. **ODIN (LSTM)** is fine-tuned for 5 epochs using a low learning-rate (`5e-4`) CosineAnnealingLR.
3. This process happens asynchronously in the background.

---

## 🛡️ HOW TO FIX ERRORS WITHOUT BREAKING THE BOT

1. **Array Shape Mismatches:** If `X_train` shape != 214, you messed up `_extract_features`. Count `features.extend()` arrays.
2. **CryptoPanic Rate Limits:** We don't use it anymore! It's an optional fallback. Do not add CryptoPanic wait locks.
3. **Zombie Processes:** The system uses parallel Threads. If it hangs, `Get-Process python | Stop-Process`.
4. **Cache Hangs:** Delete the `.feather` files in `QUANTA/cache/` using the OS file manager if they become corrupted.

*Written by Antigravity under direct instructions from the Creator.*
