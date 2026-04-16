"""
Standalone Thor V12 Training Script
Trains the Thor CatBoost model and saves to ml_models_pytorch/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from quanta_config import Config as cfg
from QUANTA_ml_engine import DeepMLEngine as MLEngine

print("=" * 70)
print("QUANTA V12 — THOR STANDALONE TRAINING")
print("=" * 70)

ml = MLEngine(cfg, bnc=None, mtf=None)

print(f"Model dir: {cfg.model_dir}")
print(f"Specialist: {list(ml.specialist_models.keys())}")
print()

ml.train(top_symbols=100, clean_retrain=False)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
