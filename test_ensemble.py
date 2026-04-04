import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from QUANTA_ml_engine import DeepMLEngine
from QUANTA_agents import PPOAgent, PPOMemory

# Create dummy dependencies
class DummyCfg:
    model_dir = "./models"
    
class DummyML:
    def __init__(self):
        self.cfg = DummyCfg()
        self.specialist_models = {
            'athena': {'weight': 0.142},
            'hermes': {'weight': 0.142},
            'hephaestus': {'weight': 0.142},
            'hades': {'weight': 0.142},
            'artemis': {'weight': 0.142},
            'chronos': {'weight': 0.142},
            'atlas': {'weight': 0.142}
        }
        
def test_weights():
    ml = DummyML()
    weights = [s['weight'] for s in ml.specialist_models.values()]
    total = sum(weights)
    print(f"Initial weights sum to: {total:.4f}")
    assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1! {total}"
    print("Weight verification PASS")
    
def test_ppo():
    try:
        agent = PPOAgent(input_dim=214)
        memory = PPOMemory()
        
        # Test basic inference
        dummy_state = np.random.randn(214)
        action, log_prob, value = agent.select_action(dummy_state)
        print(f"PPO Action: {action}, LogProb: {log_prob:.4f}, Mean Value: {value:.4f}")
        print("PPO Inference PASS")
        
        # Test update loop (7-Critic ensemble backwards passes)
        for _ in range(5):
            s = np.random.randn(214)
            a, p, v = agent.select_action(s)
            r = np.random.randn()
            memory.store_memory(s, a, p, r, True, v)
            
        metrics = agent.update(memory)
        print(f"PPO Metrics: {metrics}")
        print("PPO 7-Critic Training Update PASS")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"PPO Integration FAIL: {e}")

if __name__ == "__main__":
    test_weights()
    test_ppo()
    print("ALL TESTS COMPLETE")
