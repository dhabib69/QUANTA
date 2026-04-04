"""
🧠 QUANTA Agents — DRL & Neural Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Contains:
  • LSTMAttentionClassifier (LSTM + Multi-Head Attention)
  • PPO Agent + DifferentialSharpeRatio (Schulman 2017, Moody & Saffell 2001)
  • RunningMeanStd (Engstrom 2020 "Implementation Matters")
  • Optuna Hyperparameter Tuner
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import os
import time
import logging
import math
import json
import random
from collections import deque

from QUANTA_trading_core import *



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIFFERENTIAL SHARPE RATIO (Moody & Saffell 2001)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DifferentialSharpeRatio:
    """Online risk-adjusted reward function (Moody & Saffell 2001)."""
    def __init__(self, eta=DSR_ADAPTATION_RATE):
        self.eta = eta
        self.A_prev = 0.0
        self.B_prev = 0.0

    def reset(self):
        self.A_prev = 0.0
        self.B_prev = 0.0

    def compute(self, raw_return: float) -> float:
        R_t = raw_return
        delta_A = R_t - self.A_prev
        delta_B = R_t ** 2 - self.B_prev
        A_new = self.A_prev + self.eta * delta_A
        B_new = self.B_prev + self.eta * delta_B
        denominator = self.B_prev - self.A_prev ** 2
        if denominator < DSR_EPS:
            dsr = R_t * 0.1
        else:
            dsr = (self.B_prev * delta_A - 0.5 * self.A_prev * delta_B) / \
                  (denominator ** 1.5 + DSR_EPS)
        self.A_prev = A_new
        self.B_prev = B_new
        return float(np.clip(dsr, -5.0, 5.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACTOR-CRITIC NETWORK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RandomizedPriorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Trainable network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Fixed prior network
        self.prior = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Freeze prior
        for param in self.prior.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.net(x) + self.prior(x)

class BootstrappedMaskerCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.3),  # Zeros out 30% of inputs
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    """Separate Actor-Critic for PPO with 7-Critic Meta-Ensemble.
    
    Action space: DISCRETE 3-action (0=HOLD, 1=BUY, 2=SELL).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=PPO_HIDDEN_DIM):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1)
        )
        
        # 7-Critic Ensemble
        self.critics = nn.ModuleList()
        
        # C1: The Standard (Baseline) - Engstrom 2020
        c1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.critics.append(c1)
        
        # C2: The Pessimist (Huber) - Fujimoto 2018
        c2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.critics.append(c2)
        
        # C3: Randomized Prior (Epistemic) - Osband 2018
        c3 = RandomizedPriorCritic(input_dim, hidden_dim)
        self.critics.append(c3)
        
        # C4: Spectral Stabilizer - Gogianu 2021
        c4 = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, 1))
        )
        self.critics.append(c4)
        
        # C5: Information Bottleneck - Cobbe 2019
        c5 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
        self.critics.append(c5)
        
        # C6: Bootstrapped Masker - Osband 2016
        c6 = BootstrappedMaskerCritic(input_dim, hidden_dim)
        self.critics.append(c6)
        
        # C7: Fast Reactor - Ota 2021
        c7 = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.critics.append(c7)
        
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for Actor
        for i, layer in enumerate(self.actor):
            if isinstance(layer, nn.Linear):
                weight_gain = 0.01 if i == len(self.actor) - 2 else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=weight_gain)
                nn.init.constant_(layer.bias, 0.0)
                
        # Orthogonal init for all critics
        for critic in self.critics:
            to_init = critic.modules() if not hasattr(critic, 'net') else critic.net.modules()
            for module in to_init:
                if isinstance(module, nn.Linear):
                    if module.out_features == 1:
                        nn.init.orthogonal_(module.weight, gain=1.0)
                    else:
                        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)

    def forward(self):
        raise NotImplementedError("Use get_action_values() instead")

    def get_action_values(self, state, action=None):
        action_probs = self.actor(state)
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0 - 1e-8)
        dist = torch.distributions.Categorical(action_probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Aggregate 7 critics (Mean / Sunrise approach)
        values = [critic(state) for critic in self.critics]
        stacked_values = torch.cat(values, dim=-1)  # shape: (batch_size, 7)
        mean_value = stacked_values.mean(dim=-1, keepdim=True)  # shape: (batch_size, 1)
        
        return action, log_prob, entropy, mean_value, stacked_values


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PPO MEMORY BUFFER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PPOMemory:
    """Rollout buffer for PPO trajectories."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store_memory(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PPO AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PPOAgent:
    """PPO with Differential Sharpe Ratio (Schulman 2017 + Moody 2001).
    
    Improvements (Engstrom, Ilyas et al. 2020 "Implementation Matters"):
      B1. Observation normalization (RunningMeanStd)
      B2. Reward normalization (running discounted return std)
      B3. LR linear annealing
      B4. Value function clipping
      B5. Per-minibatch advantage normalization
    """
    def __init__(self, input_dim=None, output_dim=3, lr=PPO_LR,
                 gamma=PPO_GAMMA, gae_lambda=PPO_GAE_LAMBDA,
                 clip_ratio=PPO_CLIP, entropy_coef=PPO_ENTROPY_COEF,
                 value_coef=PPO_VALUE_COEF, max_grad_norm=PPO_MAX_GRAD_NORM,
                 max_updates=10000):
        if input_dim is None:
            try:
                from QUANTA_bot import Config
                cfg = Config()
                input_dim = cfg.BASE_FEATURE_COUNT + 10  # +7 CatBoost probs + 2 Meta (Divergence, Mean Entropy) + 1 HMM Regime
            except ImportError:
                input_dim = 222 + 10
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.K_epochs = PPO_EPOCHS
        self.batch_size = PPO_BATCH_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🧠 PPO Agent initialized on {self.device}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"      GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
        self.policy = ActorCritic(input_dim, output_dim).to(self.device)
        self.policy_old = ActorCritic(input_dim, output_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # B3: LR annealing setup
        self.initial_lr = lr
        self.max_updates = max_updates
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.mse_loss = nn.MSELoss()
        self.dsr = DifferentialSharpeRatio()
        self._update_count = 0
        
        # B1: Observation normalization (Engstrom 2020)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        
        # B2: Reward normalization — track running std of discounted returns
        self.ret_rms = RunningMeanStd(shape=())
        self._discounted_return = 0.0
        
        param_count = sum(p.numel() for p in self.policy.parameters())
        vram_est_mb = param_count * 4 * 2 / 1024**2
        print(f"      Parameters: {param_count:,} ({vram_est_mb:.1f} MB estimated)")
        print(f"      Improvements: obs_norm, reward_norm, lr_anneal, value_clip, minibatch_adv, adaptive_entropy")
        
        # E1: Adaptive exploration (axPPO, arXiv 2024)
        self._reward_ema = 0.0
        self._reward_ema_alpha = 0.1  # EMA smoothing factor
        self._base_entropy_coef = entropy_coef

    def apply_zeus_overrides(self, overrides: dict):
        """Dynamically hot-swap learning metrics instructed by ZEUS."""
        if "lr" in overrides:
            new_lr = float(overrides["lr"])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"      ⚡ ZEUS applied new PPO LR: {new_lr:.6f}")
        if "entropy" in overrides:
            self._base_entropy_coef = float(overrides["entropy"])
            self.entropy_coef = self._base_entropy_coef  # reset active entropy
            print(f"      ⚡ ZEUS applied new PPO Entropy: {self.entropy_coef:.4f}")
        if "clip" in overrides:
            self.clip_ratio = float(overrides["clip"])
            print(f"      ⚡ ZEUS applied new PPO Clip Ratio: {self.clip_ratio:.3f}")
        if "batch" in overrides:
            self.batch_size = int(overrides["batch"])
            print(f"      ⚡ ZEUS applied new PPO Batch Size: {self.batch_size}")

    def select_action(self, state):
        # B1: Normalize observation before feeding to policy
        state_normalized = self.obs_rms.normalize(np.array(state, dtype=np.float64))
        with torch.no_grad():
            state_t = torch.FloatTensor(state_normalized).unsqueeze(0).to(self.device)
            action, log_prob, _, mean_value, _ = self.policy_old.get_action_values(state_t)
        return action.item(), log_prob.item(), mean_value.item()

    def compute_dsr_reward(self, raw_return: float) -> float:
        return self.dsr.compute(raw_return)

    def update(self, memory):
        if len(memory.states) < 2:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # B1: Update observation running stats and normalize
        raw_states = np.array(memory.states)
        self.obs_rms.update(raw_states)
        normalized_states = self.obs_rms.normalize(raw_states)
        
        states = torch.FloatTensor(normalized_states).to(self.device)
        actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        raw_rewards = np.array(memory.rewards)
        dones = torch.FloatTensor(np.array(memory.dones, dtype=np.float32)).to(self.device)
        old_values = torch.FloatTensor(np.array(memory.values)).to(self.device)
        
        # B2: Reward normalization — divide by running std of discounted returns
        disc_returns = np.zeros_like(raw_rewards)
        running = 0.0
        for t in reversed(range(len(raw_rewards))):
            running = raw_rewards[t] + self.gamma * running * (1.0 - float(memory.dones[t]))
            disc_returns[t] = running
        self.ret_rms.update(disc_returns)
        normalized_rewards = raw_rewards / (np.sqrt(self.ret_rms.var) + 1e-8)
        rewards = torch.FloatTensor(normalized_rewards).to(self.device)
        
        # GAE advantage estimation
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = old_values[t + 1].item()
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + old_values[t]
        
        # B3: Linear LR annealing (Engstrom 2020)
        frac = max(1.0 - self._update_count / self.max_updates, 0.1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr * frac
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.K_epochs):
            _, log_probs, entropy, state_values_mean, state_values_stacked = self.policy.get_action_values(states, actions)
            state_values_mean = state_values_mean.squeeze()
            
            # B5: Per-minibatch advantage normalization (Andrychowicz 2020)
            adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * adv_normalized
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_normalized
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 7-Critic Value Loss Calculation
            value_loss_total = 0.0
            for i in range(7):
                v_pred = state_values_stacked[:, i]
                # Value function clipping against old ensemble mean
                v_pred_clipped = old_values + torch.clamp(
                    v_pred - old_values, -self.clip_ratio, self.clip_ratio
                )
                
                if i == 1:  # C2: Pessimist (Huber / Smooth L1 Loss)
                    v_loss_unclipped = F.smooth_l1_loss(v_pred, returns, reduction='none')
                    v_loss_clipped = F.smooth_l1_loss(v_pred_clipped, returns, reduction='none')
                else:  # Standard MSE
                    v_loss_unclipped = (v_pred - returns) ** 2
                    v_loss_clipped = (v_pred_clipped - returns) ** 2
                    
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                value_loss_total += v_loss
                
            value_loss = value_loss_total / 7.0
            
            entropy_loss = -entropy.mean()
            
            # E1: Adaptive entropy coefficient (axPPO, arXiv 2024)
            # Scale exploration inversely with reward quality
            adaptive_entropy_coef = self.entropy_coef
            
            loss = policy_loss + self.value_coef * value_loss + adaptive_entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self._update_count += 1
        
        # E1: Update reward EMA and adapt entropy coefficient for next update
        mean_reward = raw_rewards.mean()
        self._reward_ema = self._reward_ema_alpha * mean_reward + (1 - self._reward_ema_alpha) * self._reward_ema
        # Scale: negative reward EMA → more exploration (up to 3x), positive → less (down to 0.5x)
        if self._reward_ema < 0:
            entropy_scale = min(3.0, 1.0 + abs(self._reward_ema) * 10)  # Struggling → explore more
        else:
            entropy_scale = max(0.5, 1.0 - self._reward_ema * 2)  # Winning → exploit more
        self.entropy_coef = self._base_entropy_coef * entropy_scale
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "loss": loss.item(), "policy_loss": total_policy_loss / self.K_epochs,
            "value_loss": total_value_loss / self.K_epochs,
            "entropy": total_entropy / self.K_epochs, "update_count": self._update_count,
            "lr": self.optimizer.param_groups[0]['lr'],
            "entropy_coef": self.entropy_coef,
            "reward_ema": self._reward_ema,
        }

    def save(self, directory, filename="ppo_model.pth"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self._update_count,
            "dsr_state": {"A_prev": self.dsr.A_prev, "B_prev": self.dsr.B_prev},
        }, path)
        print(f"   💾 PPO Model saved to {path}")

    def load(self, directory, filename="ppo_model.pth"):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(checkpoint["model_state_dict"])
            self.policy_old.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._update_count = checkpoint.get("update_count", 0)
            dsr_state = checkpoint.get("dsr_state", {})
            self.dsr.A_prev = dsr_state.get("A_prev", 0.0)
            self.dsr.B_prev = dsr_state.get("B_prev", 0.0)
            print(f"   ✅ PPO Model loaded from {path} (updates: {self._update_count})")
            return True
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNNING MEAN/STD (Engstrom 2020 "Implementation Matters")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RunningMeanStd:
    """Online Welford running mean/variance tracker.
    
    Used for:
    - Observation normalization (state inputs to actor/critic)
    - Reward normalization (discounted return variance)
    
    Reference: Engstrom, Ilyas et al. 2020 — "the single most impactful
    code-level optimization" for PPO.
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Avoid division by zero
    
    def update(self, x):
        """Update running stats with a batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count
    
    def normalize(self, x, clip=10.0):
        """Normalize x using running stats, with clipping."""
        return np.clip(
            (x - self.mean) / (np.sqrt(self.var) + 1e-8),
            -clip, clip
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTUNA HYPERPARAMETER TUNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET_METRIC = "sortino"
CATBOOST_TRIALS = 50
DRL_TRIALS = 50
NUM_SAMPLES = 10000


class TradingSimulator:
    """Simulates trading with realistic synthetic data."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        from QUANTA_bot import Config
        self.cfg = Config()

    def generate_data(self, num_samples=NUM_SAMPLES):
        from QUANTA_bot import Config
        cfg = Config()
        X = self.rng.standard_normal((num_samples, cfg.BASE_FEATURE_COUNT)).astype(np.float32)
        signal = (X[:, 0] * 1.5 - X[:, 1] * 0.5 + X[:, 2] * 0.3
                  + self.rng.standard_normal(num_samples) * 0.8)
        y = (signal > 0).astype(np.int32)
        price_moves = np.abs(self.rng.lognormal(-3.9, 0.5, num_samples))
        return X, y, price_moves

    def compute_sortino(self, returns):
        if len(returns) == 0:
            return -10.0
        downside = returns[returns < 0]
        down_std = np.std(downside) if len(downside) > 1 else 1e-6
        return np.mean(returns) / (down_std + 1e-6)

    def simulate_trades(self, predictions, actuals, price_moves, conf_threshold=60.0):
        if len(predictions) == 0:
            return np.array([]), 0.0
        confidences = np.max(predictions, axis=1) * 100
        directions = np.argmax(predictions, axis=1)
        mask = confidences > conf_threshold
        if mask.sum() == 0:
            return np.array([]), 0.0
        taken_dirs = directions[mask]
        actual_dirs = actuals[mask]
        taken_moves = price_moves[mask]
        correct = taken_dirs == actual_dirs
        returns = np.where(correct, taken_moves, -taken_moves * 0.5)
        win_rate = correct.mean()
        return returns, win_rate


def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    sim = TradingSimulator(seed=trial.number)
    X, y, prices = sim.generate_data()
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test, prices_test = X[split:], y[split:], prices[split:]
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        **params, verbose=False, auto_class_weights='Balanced',
        early_stopping_rounds=20,
        task_type='GPU' if os.environ.get('USE_GPU') else 'CPU',
    )
    try:
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    except Exception as e:
        print(f"  ✗ Trial {trial.number} failed: {e}")
        return -10.0
    preds = model.predict_proba(X_test)
    returns, win_rate = sim.simulate_trades(preds, y_test, prices_test)
    if len(returns) == 0:
        return -10.0
    import optuna
    sortino = sim.compute_sortino(returns)
    trial.report(sortino, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return sortino


def ppo_objective(trial):
    from QUANTA_bot import Config
    cfg = Config()
    params = {
        'gamma': trial.suggest_float('ppo_gamma', 0.80, 0.999),
        'clip_ratio': trial.suggest_float('ppo_clip', 0.05, 0.3),
        'entropy_coef': trial.suggest_float('ppo_entropy', 0.001, 0.05, log=True),
        'lr': trial.suggest_float('ppo_lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('ppo_batch', [64, 128, 256, 512]),
    }
    sim = TradingSimulator(seed=trial.number + 1000)
    X, y, prices = sim.generate_data(5000)
    agent = PPOAgent(input_dim=cfg.BASE_FEATURE_COUNT, lr=params['lr'],
                     gamma=params['gamma'], clip_ratio=params['clip_ratio'],
                     entropy_coef=params['entropy_coef'])
    agent.batch_size = params['batch_size']
    dsr = DifferentialSharpeRatio()
    memory = PPOMemory()
    total_returns = []
    for episode in range(3):
        dsr.reset()
        memory.clear_memory()
        start = episode * 1500
        end = min(start + 1500, len(X))
        for i in range(start, end - 1):
            state = X[i]
            action, log_prob, value = agent.select_action(state)
            if action == 0:
                raw_return = 0.0
            elif action == 1:
                raw_return = prices[i] if y[i] == 1 else -prices[i] * 0.5
            else:
                raw_return = prices[i] if y[i] == 0 else -prices[i] * 0.5
            reward = dsr.compute(raw_return)
            total_returns.append(raw_return)
            done = (i == end - 2)
            memory.store_memory(state, action, log_prob, reward, done, value)
        if len(memory) >= params['batch_size']:
            agent.update(memory)
    returns = np.array(total_returns)
    sortino = sim.compute_sortino(returns)
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sortino


def print_results(study, name):
    print(f"\n{'─'*50}")
    print(f"  🏆 {name} — Best Sortino: {study.best_value:.4f}")
    print(f"{'─'*50}")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")


if __name__ == "__main__":
    import optuna
    from QUANTA_bot import Config
    cfg = Config()
    t_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("=" * 60)
    print("🧠 QUANTA v11.5b — HYPERPARAMETER TUNER")
    print(f"   Target: {TARGET_METRIC.upper()}")
    print(f"   Features: {cfg.BASE_FEATURE_COUNT}")
    print("=" * 60)
    print(f"\n🚀 PHASE 1: CatBoost ({CATBOOST_TRIALS} trials)")
    cat_study = optuna.create_study(study_name="quanta_v9_catboost", direction="maximize",
                                     pruner=optuna.pruners.MedianPruner())
    n_jobs = max(1, (os.cpu_count() or 4) - 1)
    cat_study.optimize(catboost_objective, n_trials=CATBOOST_TRIALS, n_jobs=n_jobs)
    print_results(cat_study, "CatBoost")
    print(f"\n🚀 PHASE 2: PPO Agent ({DRL_TRIALS} trials)")
    ppo_study = optuna.create_study(study_name="quanta_v9_ppo", direction="maximize")
    ppo_study.optimize(ppo_objective, n_trials=DRL_TRIALS, n_jobs=1)
    print_results(ppo_study, "PPO")

    print(f"\n{'='*60}")
    print("✅ ALL PHASES COMPLETE")
    print(f"   Time: {(time.time() - t_start)/60:.1f} minutes")
    print(f"\nTo apply → Copy values into QUANTA_trading_core constants")
    print(f"{'='*60}")
