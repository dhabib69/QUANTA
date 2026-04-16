# QUANTA BOT — MEMORY & CONTEXT
_Last Updated: 2026-04-16 — V12.4 Gompertz Dynamic Exit Live_

---

## 🏛️ CURRENT ARCHITECTURE: V12 THOR (Single Specialist)

The system has been **fully refactored** from the 7-specialist Greek Pantheon ensemble to a **single-specialist Norse Thor engine**.

### What Was Removed
- All Pantheon specialists: Athena, Ares, Hermes, Hephaestus, Artemis, Chronos, Nike
- Odin TFT (LSTM-Attention) meta-learner and feature 223 injection
- Heimdall PPO dynamic position sizer
- CEO AI (Anthropic Claude evaluator)
- Nike Screener background process
- Shannon Entropy ensemble veto logic
- Brier Score disagreement discount

### What Remains
- **Thor (Impulse Rider):** Single CatBoost model trained on `fast_extract_nike` (CUSUM-gated explosive breakout events)
- **Static 1.0x sizing:** No PPO override — pure asymmetric compounding
- **Exit Logic (hardcoded):** 5.4 ATR early bank, 6.0 ATR runaway trail, 2.0 ATR stop-loss
- **Alert System:** Confidence-gate removed from Telegram alerts — all CUSUM-passing Thor signals are relayed regardless of ml_conf score

---

## 📁 KEY FILE CHANGES

| File | Change |
|------|--------|
| `QUANTA_bot.py` | Removed ensemble pipeline, Nike screener, CEO AI, BS-veto; alerts now confidence-gated OFF |
| `QUANTA_trading_core.py` | `_tick_nike_v2` → `_tick_thor_v12`, PPO size overrides removed |
| `QUANTA_ml_engine.py` | specialist_models = thor only; Odin TFT, cross-event sampling removed |
| `QUANTA_selector.py` | dict key `nike` → `thor` in fast_extract output |

---

## 🎯 TRAINING TARGET — COMPLETED ✅
- Model file: `models/thor_gen1.cbm` (Primary CatBoost)
- Feature mask: `domain_impulse` (102 features)
- Metrics: **AUC 0.8080 | Brier 0.1528**
- Calibrator: Conformal calibrated (raw 76.8% → 35.0%)
- Status: **Gen 1 Deployed**

---

## 📁 KEY FILE CHANGES (V12 FULL CLEANUP)
- `QUANTA_ml_engine.py`: Patched `SystemConfig` to support V12 missing attrs (`BASE_FEATURE_COUNT`, `timeframes`, `tf_weights`). Removed all references to Athena/Pantheon in training and save loops.
- `QUANTA_selector.py`: Fixed `fast_extract_nike` call signature (added missing `opens` array).
- `QUANTA_bot.py`: Fixed SyntaxError in PPO logic and replaced Athena legacy routing with Thor.
- `train_thor.py`: Specialized training script created for isolated Thor runs using local `feather_cache`.

---

## 💰 CAPITAL & RISK
- Starting paper balance: **$10,000**  
- Sizing: Static 1x (No PPO Override)
- Targets: 5.4 ATR Bank (50%) | 6.0 ATR Trail (Runner)

---

---

## 🔥 WALK-FORWARD SIM — V12.4 FINAL RESULTS (2026-04-16)

| Metric | Value |
|--------|-------|
| Total Return | **+150,235%** (V12.4) / +19,883% (V12.3) |
| Win Rate | 70.0% |
| Profit Factor | 4.65 |
| Sharpe Ratio | 7.27 |
| Sortino | 19.25 |
| Calmar | 15,497 |
| Max Drawdown | **9.7%** |
| OOS Period | 300 days (10 windows × 30d) |
| Universe | 245 symbols |
| L2 Add PnL | +$3.46M |
| L3 Recovery PnL | +$2.67M |

### WF Sim Key Parameters (calibrated from NORSE_MAE_STATS_REPORT)
- `_SL_ATR = 3.00` | `_BANK_ATR = 4.20` | `_BANK_FRAC = 0.35` | `_TRAIL_ATR = 2.00`
- `_TRAIL_ACTIVATE_ATR = 1.50` (trail only kicks after 1.5 ATR above bank price)
- `_SKIP_UTC_HOURS = {0, 7, 10, 11, 12}` (low-PF hours from MAE heatmap)
- `_MAE_VETO_BARS = 5` | `_MAE_VETO_ATR = 3.62` (early exit if −3.62 ATR in first 5 bars)
- `_PYR_TRIGGER_ATR = 0.5` | `_PYR_ADD_SL = 0.5` | `_PYR_RECOVERY_ATR = 3.77`
- Continuous risk scaling: score 68→3%, score 100→max_risk (replaces binary 85 threshold)

### Exponential Growth Constants
| Period | n value | e^n |
|--------|---------|-----|
| Per trade | 0.01569 | +1.58% |
| Daily | 0.02438 | +2.47% |
| Weekly | 0.1707 | +18.6% |
| Monthly | 0.7314 | +107.8% |
| Annual | 8.899 | 7,337× |
| Doubling time | 28.4 days | — |

---

## ⏳ NEXT ACTIONS
1. **Live Deployment:** MAE-calibrated params in WF sim — need to sync back to `quanta_config.py` for live bot to use same SL/bank/trail values.
2. **Pyramid in Norse Sim:** Port 3-layer pyramid to `quanta_norse_year_sim.py` (user confirmed they improved Norse independently).
3. **Wave_strength live stream:** WF uses feather-cache taker_buy_base; live bot needs aggTrades websocket stream — verify parity.
4. **Slippage reality check:** WF models 2bps flat; live altcoin market impact on fast breakouts is 5–15bps at size.
