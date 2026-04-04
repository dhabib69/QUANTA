import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from QUANTA_ml_engine import DeepMLEngine

# Dummy dependencies
class DummyCfg:
    base_dir = "./"
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    tf_weights = {'5m': 0.1, '15m': 0.15, '1h': 0.25, '4h': 0.3, '1d': 0.2}

class DummyStore:
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        
        # 200 random candles: [timestamp, open, high, low, close, volume]
        t0 = 1600000000000
        candles = []
        p = 50000.0
        for i in range(200):
            p = p * (1 + np.random.normal(0, 0.001))
            candles.append([t0 + i*300*1000, p, p*1.002, p*0.998, p*1.001, 100.0])
        self.candles = {'BTCUSDT': candles}

class DummyEngine(DeepMLEngine):
    def __init__(self):
        self.cfg = DummyCfg()
        self.candle_store = DummyStore()
        
# Test extracting one sequence
engine = DummyEngine()
candles = engine.candle_store.candles['BTCUSDT']

print("Array length:", len(candles))
print("Testing single feature extraction at t=100...")
feat = engine._extract_features_from_candles(candles, 100)
if feat is None:
    print("Feat is None at t=100")
else:
    print("Feat length:", len(feat))

print("Testing single feature extraction at t=2000...")
feat = engine._extract_features_from_candles(candles, 2000)
if feat is None:
    print("Feat is None at t=2000")
else:
    print("Feat length:", len(feat))

print("Testing prepare sequences for ['BTCUSDT']...")
X, y = engine._prepare_sequences(['BTCUSDT'], 60)
if X is None:
    print("X is None!")
else:
    print("Sequences extracted:", len(X))
