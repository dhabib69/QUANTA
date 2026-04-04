import os
import numpy as np
import logging
import pickle
import time
from hmmlearn import hmm
import warnings

# Suppress hmmlearn deprecation warnings if any
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from QUANTA_agents import PPOAgent
except ImportError:
    pass # for testing isolation

class MarketRegimeHMM:
    """
    Hidden Markov Model (HMM) for Unsupervised Regime Detection.
    Detects dynamic K market regimes (Bull, Bear, Crab, Volatile) based on
    Log-Returns, Volatility (ATR%), and Trend Strength (ADX).
    """
    def __init__(self, max_components=5, model_dir='ml_models_pytorch'):
        self.max_components = max_components
        self.model_dir = model_dir
        self.model = None
        self.n_components = 3  # Default fallback
        self.is_trained = False
        self.model_path = os.path.join(model_dir, 'hmm_regime_router.pkl')
        
    def _compute_bic(self, model, X):
        """Calculate Bayesian Information Criterion to find optimal K."""
        try:
            log_likelihood = model.score(X)
            # Free parameters = transition probabilities + means + covariances
            n_features = X.shape[1]
            n_states = model.n_components
            free_params = (n_states * (n_states - 1)) + (n_states * n_features) + (n_states * n_features * (n_features + 1) / 2)
            n_samples = X.shape[0]
            bic = -2 * log_likelihood + free_params * np.log(n_samples)
            return bic
        except Exception as e:
            logging.debug(f"BIC computation failed: {e}")
            return float('inf')

    def optimize_and_fit(self, X):
        """
        Dynamically find the optimal number of Market Regimes (K) using BIC
        and fit the ultimate HMM router.
        """
        print(f"\n[HMM Router] Optimizing Latent Market Regimes via BIC...")
        best_bic = float('inf')
        best_model = None
        
        # Test K from 2 to max_components
        for k in range(2, self.max_components + 1):
            try:
                model = hmm.GaussianHMM(n_components=k, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X)
                bic = self._compute_bic(model, X)
                print(f"   ├─ K={k} Regimes | BIC Score: {bic:.2f}")
                
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception as e:
                logging.debug(f"HMM K={k} failed: {e}")
                
        if best_model is not None:
            self.model = best_model
            self.n_components = self.model.n_components
            self.is_trained = True
            print(f"   └─ ✅ Optimal Regimes Selected: K={self.n_components} (Lowest BIC)")
            self.save()
            return True
        return False
        
    def predict_regime(self, X_recent):
        """
        Predict the latent regime of the current state.
        X_recent: numpy array of shape (N, 3) -> [Log-Returns, ATR%, ADX]
        Returns: int (regime ID 0 to K-1)
        """
        if not self.is_trained or self.model is None:
            return 0 # Fallback to default regime
        try:
            # Viterbi path prediction
            regimes = self.model.predict(X_recent)
            return regimes[-1] # Return the most recent regime
        except Exception as e:
            logging.debug(f"HMM prediction failed: {e}")
            return 0

    def get_regime_probabilities(self, X_recent):
        """For Soft Gating Mixture of Experts (giving weight distributions)."""
        if not self.is_trained or self.model is None:
            probs = np.zeros(self.n_components)
            probs[0] = 1.0
            return probs
        try:
            _, posteriors = self.model.score_samples(X_recent)
            return posteriors[-1]
        except Exception as e:
            logging.debug(f"HMM regime probability failed: {e}")
            probs = np.zeros(self.n_components)
            probs[0] = 1.0
            return probs

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({'model': self.model, 'n_components': self.n_components}, f)
        except Exception as e:
            logging.error(f"Failed to save HMM: {e}")

    def load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.n_components = data['n_components']
                    self.is_trained = True
                    print(f"✅ Loaded HMM Regime Router (K={self.n_components})")
                    return True
            except Exception as e:
                logging.error(f"Failed to load HMM: {e}")
        return False


class MoEPPOAgent:
    """
    Mixture of Experts Proximal Policy Optimization.
    Routes state inputs to K specialized PPO neurological brains based on the 
    HMM Regime Router. Eliminates Catastrophic Forgetting across macroeconomic shifts.
    """
    def __init__(self, input_dim=None, hmm_max_components=4):
        self.input_dim = input_dim
        self.hmm = MarketRegimeHMM(max_components=hmm_max_components)
        self.experts = {} # Dictionary mapping Regime ID -> PPOAgent
        
        # Try loading a pre-trained HMM to know exactly how many experts to spawn
        if self.hmm.load():
            self._spawn_experts()
        else:
            # Fallback to 1 expert before formal training
            self._spawn_experts(force_k=1)
            
    def _spawn_experts(self, force_k=None):
        num_experts = force_k if force_k else self.hmm.n_components
        print(f"🧬 Spawning {num_experts} Specialized PPO Experts...")
        from QUANTA_agents import PPOAgent, PPOMemory # late import to resolve circular dependency
        self.experts = {}
        for i in range(num_experts):
            expert = PPOAgent(input_dim=self.input_dim)
            expert.memory = PPOMemory() # Attach dedicated memory buffer
            self.experts[i] = expert
        
    def get_action_values(self, state_tensor, hmm_features_np=None, action=None):
        """
        Hard Gating: Routes the entire decision strictly to the Expert 
        responsible for the current identified market regime.
        """
        if hmm_features_np is not None and self.hmm.is_trained:
            regime = self.hmm.predict_regime(hmm_features_np)
        else:
            regime = 0
            
        expert = self.experts.get(regime, self.experts[0])
        action, log_prob, entropy, mean_value, stacked_values = expert.get_action_values(state_tensor, action)
        
        return action, log_prob, entropy, mean_value, stacked_values, regime

    def select_action(self, state_array, hmm_features_np=None):
        """
        Live Production evaluation hook for QUANTA_bot.py
        """
        if hmm_features_np is not None and self.hmm.is_trained:
            regime = self.hmm.predict_regime(hmm_features_np)
        else:
            regime = 0
        expert = self.experts.get(regime, self.experts[0])
        action, log_prob, value = expert.select_action(state_array)
        return action, log_prob, value

    def save_models(self):
        self.hmm.save()
        for regime, expert in self.experts.items():
            expert.save(directory="ml_models_pytorch", filename=f"Regime_{regime}_ppo.pth")

    def load_models(self):
        hmm_loaded = self.hmm.load()
        if hmm_loaded:
            self._spawn_experts()
            for regime, expert in self.experts.items():
                expert.load(directory="ml_models_pytorch", filename=f"Regime_{regime}_ppo.pth")
            return True
        return False
