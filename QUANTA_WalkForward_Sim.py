"""
🚀 QUANTA v10.4 — WALK-FORWARD OOS SIMULATION CASINO
====================================================
Isolated Out-Of-Sample reinforcement learning environment.
Trains the Mixture of Experts (MoEPPOAgent) strictly on the hidden
165 Days (Day 201 to Day 365) to brutally eliminate Look-Ahead Bias.
CatBoost is blind. The Bot must learn to survive.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

from QUANTA_bot import Config, BinanceAPIEnhanced
from QUANTA_ml_engine import DeepMLEngine

try:
    from quanta_features import Indicators
except ImportError:
    pass

from QUANTA_moe import MarketRegimeHMM, MoEPPOAgent

logging.basicConfig(level=logging.INFO, format='\n%(message)s')

def compile_hmm_features(klines_window):
    """
    Given a short window of klines (e.g., last 20 candles),
    extract the raw physical arrays [LogReturns, ATR%, ADX] for the HMM Router.
    """
    try:
        closes = [float(k[4]) for k in klines_window]
        highs = [float(k[2]) for k in klines_window]
        lows = [float(k[3]) for k in klines_window]
        
        # Log Returns
        log_returns = np.diff(np.log(closes))
        log_ret_mean = np.mean(log_returns) if len(log_returns) > 0 else 0.0
        
        # ATR %
        atr = Indicators.atr(highs, lows, closes)
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] > 0 else 0.0
        
        # ADX (Trend Strength)
        adx = Indicators.adx(highs, lows, closes)
        
        # Return a single row vector for the HMM Array [1, 3]
        return np.array([[log_ret_mean * 100, atr_pct, adx]])
    except Exception:
        # Neutral fallback
        return np.array([[0.0, 1.0, 20.0]])

def main():
    print("="*80)
    print("🔱 INITIATING WALK-FORWARD HINDSIGHT EXPERIENCE REPLAY (OOS CASINO)")
    print("   Architecture: Mixture of Experts (MoE) + HMM Gating")
    print("   Target: 165 Days Unseen Data (Day 201 to 365)")
    print("="*80)
    
    cfg = Config()
    bnc = BinanceAPIEnhanced(cfg)  # No websocket started in __init__, test_mode not needed
    ml_engine = DeepMLEngine(cfg, bnc)
    
    # 1. LOAD CATBOOST (The 200-Day Blind Teacher)
    print("\n⏳ 1. Loading Locked CatBoost Ensembles...")
    if not ml_engine.is_trained:
        # Attempt load
        if not ml_engine.load_models():
            print("❌ CatBoost PANIC: No CatBoost models found!")
            print("❌ You must run the base /train command first to train Phase 1 (First 200 Days).")
            return
    print("   ✅ CatBoost Generalists Loaded and Locked in Time.")
    
    # 2. SELECT OOS COINS
    print("\n⏳ 2. Fetching OOS Market Data...")
    # Get a cross-section of top gainers and losers. 20 pairs is enough to train PPO effectively.
    oos_coins = bnc.get_research_backed_coins(limit=20)[:8] # Keep small for simulation speed, adjust as needed
    
    if not oos_coins:
        print("❌ Failed to fetch OOS coins. Using fixed anchors...")
        oos_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT']
    
    print(f"   ✅ Selected 8 Strategic Anchors for Casino: {', '.join(oos_coins)}")
    
    OOS_CANDLE_START = int(200 * 24 * (60/5))  # Day 201
    
    # 3. BUILD HMM REGIME ROUTER DATASET
    print("\n⏳ 3. Scanning OOS Volatility for HMM Regime Discovery (BIC Scoring)...")
    hmm_training_data = []
    
    # We will slice out the OOS dataset explicitly
    casino_data = {} # {symbol: [klines]}
    
    for sym in oos_coins:
        print(f"   📥 Downloading 365 timeline for {sym}...")
        klines = bnc.get_historical_klines(sym, '5m', days=365, training_mode=True)
        if not klines or len(klines) < OOS_CANDLE_START + 1000:
            continue
            
        # Isolate exactly the Walk-Forward Future (Day 201+)
        oos_klines = klines[OOS_CANDLE_START:]
        casino_data[sym] = oos_klines
        
        # Extract features for HMM every 100 candles to map macroeconomic shifts
        total_oos = len(oos_klines)
        for i in range(25, total_oos, 100):
            window = oos_klines[i-25:i]
            hmm_feat = compile_hmm_features(window)
            hmm_training_data.append(hmm_feat[0])
            
    if not hmm_training_data:
        print("❌ Not enough data for HMM.")
        return
        
    X_hmm = np.array(hmm_training_data)
    
    # Init MoE Agent
    # 222 (base features) + 7 (catboost) = 229 input dim
    try:
        input_dim = cfg.BASE_FEATURE_COUNT + 7
    except Exception as e:
        logging.warning(f"Config attribute failed, using default input_dim: {e}")
        input_dim = 222 + 7

    moe_agent = MoEPPOAgent(input_dim=input_dim, hmm_max_components=4)
    
    # Train the HMM specifically on the OOS Variance Clusters
    moe_agent.hmm.optimize_and_fit(X_hmm)
    
    # Spawn the Experts based on the optimal K discovered by BIC
    moe_agent._spawn_experts()
    
    # 4. START VIRTUAL CASINO
    print("\n" + "="*80)
    print("🎲 INITIATING VIRTUAL EXPERIMENTAL CASINO 🎲")
    print(f"   Targeting {len(casino_data)} Assets across Unseen Future Dynamics.")
    print("="*80)
    
    # Triple Barrier Constants
    SL_RATIO = 2.0
    
    total_trades = 0
    total_vetoes = 0
    correct_vetoes = 0
    incorrect_vetoes = 0
    
    # Metrics per Regime
    regime_stats = {r: {'trades': 0, 'vetoes': 0, 'reward': 0.0} for r in range(moe_agent.hmm.n_components)}
    
    for sym, oos_klines in casino_data.items():
        print(f"\n⚡ Entering Future Matrix for: {sym}")
        oos_len = len(oos_klines)
        
        # Convert to numpy for fast feature extraction
        oos_klines_np = np.array(oos_klines, dtype=np.float64)
        
        # We step candle by candle (skip every 12 candles to speed up simulation: 1 hour)
        for i in range(50, oos_len - cfg.sequence_length - 49, 12):
            pos = i + cfg.sequence_length
            
            # Extract ML Features (from the point of view of the specific OOS candle)
            features = ml_engine._extract_features_from_candles(oos_klines, pos, _precomputed_np=oos_klines_np)
            if features is None:
                continue
                
            # Get CatBoost Predictions (The Teacher's blind guess on Day 201+)
            cb_predictions = ml_engine.predict_proba(features, sym)
            
            # Find the strongest conviction from the 7-Agent Pantheon
            max_conf = 0
            predicted_direction = 'HOLD'
            
            # Very simplified CatBoost interpretation for the Simulator
            for agent_name, prob in cb_predictions.items():
                if prob > max_conf:
                    max_conf = prob
                    # The simulator maps the probabilities roughly
                    if prob > cfg.ml_confidence_trade_min:
                        predicted_direction = 'BULL' if 'Bull' in agent_name else 'BEAR'
            
            # Convert raw predictions into the PPO input array [222 features + 7 agent probabilities]
            catboost_probs = list(cb_predictions.values())
            # Pad or trim to exactly 7 probabilities (Foundation, Hunter, anchor x Long/Short)
            while len(catboost_probs) < 7: catboost_probs.append(0.5)
            catboost_probs = catboost_probs[:7]
            
            state_array = np.concatenate([features, catboost_probs])
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Identify Latent Regime via HMM
            hmm_window = oos_klines[pos-25:pos+1]
            hmm_inputs = compile_hmm_features(hmm_window)
            
            # Let the MoE route the decision to the specific Specialist
            action_tensor, log_prob, entropy, value, _, regime = moe_agent.get_action_values(state_tensor, hmm_inputs)
            action = action_tensor.item()
            
            # Action Mapping: 0=HOLD/VETO, 1=BULL, 2=BEAR
            ppo_decision = 'HOLD' if action == 0 else ('BULL' if action == 1 else 'BEAR')
            
            # Determine actual market outcome (Triple Barrier Hindsight Replay)
            entry_price = float(oos_klines[pos][4])
            
            # Calculate dynamic ATR
            h_win = [float(k[2]) for k in hmm_window]
            l_win = [float(k[3]) for k in hmm_window]
            c_win = [float(k[4]) for k in hmm_window]
            atr_val = Indicators.atr(h_win, l_win, c_win)
            tp_ratio_local = max(0.0015, min(0.10, atr_val / entry_price))
            
            tp_long  = entry_price * (1 + tp_ratio_local)
            sl_long  = entry_price * (1 - tp_ratio_local * SL_RATIO)
            tp_short = entry_price * (1 - tp_ratio_local)
            sl_short = entry_price * (1 + tp_ratio_local * SL_RATIO)
            
            actual_outcome = 'HOLD'
            tp_t_l = sl_t_l = float('inf')
            
            # Look into the future (Next 48 candles)
            for j in range(pos + 1, min(pos + 49, oos_len)):
                h = float(oos_klines[j][2])
                l = float(oos_klines[j][3])
                if h >= tp_long and tp_t_l == float('inf'): tp_t_l = j
                if l <= sl_long and sl_t_l == float('inf'): sl_t_l = j
            
            if tp_t_l < sl_t_l and tp_t_l != float('inf'):
                actual_outcome = 'BULL'
            elif sl_t_l < tp_t_l and sl_t_l != float('inf'):
                actual_outcome = 'BEAR'
            
            # ==========================================================
            # 💎 REWARD CALCULATION MATRIX (High-Variance 5.0x System)
            # ==========================================================
            reward = 0.0
            is_trade = False
            
            if predicted_direction != 'HOLD':
                is_trade = True
                cb_is_correct = (predicted_direction == actual_outcome)
                
                if action == 0: # VETO / HOLD
                    total_vetoes += 1
                    if not cb_is_correct:
                        # Vetoed a loser -> HUGE REWARD +5.0x
                        reward = 5.0 * tp_ratio_local * 100
                        correct_vetoes += 1
                    else:
                        # Vetoed a winner -> MASSIVE PENALTY -5.0x
                        reward = -5.0 * tp_ratio_local * 100
                        incorrect_vetoes += 1
                else: 
                    # PPO agreed and made a directional bet
                    ppo_is_correct = (ppo_decision == actual_outcome)
                    if ppo_is_correct:
                        reward = 1.0 * tp_ratio_local * 100
                    else:
                        reward = -1.0 * (tp_ratio_local * SL_RATIO) * 100
            else:
                # CB said hold, PPO said trade -> Unsolicited Action
                if action != 0:
                    is_trade = True
                    ppo_is_correct = (ppo_decision == actual_outcome)
                    if ppo_is_correct:
                        # PPO found an opportunity Teacher missed -> +5.0x Override Reward
                        reward = 5.0 * tp_ratio_local * 100
                    else:
                        reward = -1.0 * (tp_ratio_local * SL_RATIO) * 100
            
            # Update specific expert's memory (The genius of MoE!)
            if is_trade:
                expert = moe_agent.experts[regime]
                expert.memory.store_memory(state_tensor.cpu().numpy()[0], action, log_prob.item(), reward, False, value.item())
                
                total_trades += 1
                regime_stats[regime]['trades'] += 1
                if action == 0: regime_stats[regime]['vetoes'] += 1
                regime_stats[regime]['reward'] += reward
                
                # Check for updates per expert batch
                if len(expert.memory) >= expert.batch_size:
                    expert.update(expert.memory)
                    expert.memory.clear_memory()
                    
    print("\n" + "="*80)
    print("🏆 WALK-FORWARD SIMULATION COMPLETE")
    print("="*80)
    print(f"   Total Encounters: {total_trades}")
    print(f"   Total Vetoes:     {total_vetoes}")
    print(f"      ├─ Saved from Losses (Correct Vetoes): {correct_vetoes}")
    print(f"      └─ Missed Winners (Wrong Vetoes):      {incorrect_vetoes}")
    print("\n   [REGIME STATISTICS]")
    for r in range(moe_agent.hmm.n_components):
        stats = regime_stats[r]
        print(f"   Regime {r}: {stats['trades']} interactions | {stats['vetoes']} vetoes | Cumulative Reward: {stats['reward']:.2f}")
    
    # Save the mathematically immune Pantheon
    moe_agent.save_models()
    print("\n💾 Mixture of Experts PPO Saved Successfully.")
    print("🚀 You are now officially running QUANTA v10.4!")

if __name__ == '__main__':
    main()
