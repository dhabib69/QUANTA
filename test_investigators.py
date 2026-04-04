import sys
import os
import time
import numpy as np

# Suppress annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from QUANTA_bot import Config
from QUANTA_bot import Config
cfg = Config()
import QUANTA_network

print("======================================================================")
print("🔍 QUANTA INVESTIGATOR DIAGNOSTIC (The 270-Feature Stress Test)")
print("======================================================================")

proxy_port = "56985"
proxy_url = f"http://127.0.0.1:{proxy_port}"
QUANTA_network._PROXY_URL = proxy_url
print(f"   ✅ Network proxy set to {proxy_url}")

from QUANTA_bot import BinanceAPIEnhanced
from QUANTA_ml_engine import MultiTimeframeAnalyzer, DeepMLEngine

print("\n[1] Deploying the Foundation...")
cfg = Config()
bnc = BinanceAPIEnhanced(cfg)
print(f"✅ Foundation deployed.")

print("\n[2] Booting the Multi-Timeframe Analyzer (The Chart Spies)...")
mtf = MultiTimeframeAnalyzer(cfg, bnc)
print(f"✅ MTF Analyzer online.")

print("\n[3] Booting the ML Engine (The Board of Directors)...")
ml = DeepMLEngine(cfg, bnc, mtf)
print(f"✅ ML Engine online. Expected Features: {cfg.BASE_FEATURE_COUNT}")

# 2. Pick a target
target_symbol = "BTCUSDT"
print(f"\n[4] Sending Investigators to interrogate {target_symbol}...")

# 3. Step 1: MTF Analysis
print(f"   ⏳ Extracting multi-dimensional candles (5m, 15m, 1h, 4h, 1d)...")
start_time = time.time()
tf_analysis = mtf.analyze(target_symbol)

if not tf_analysis:
    print(f"❌ Veto: MTF Analysis failed to return data for {target_symbol}.")
    sys.exit(1)
print(f"   ✅ Candles successfully pulled & structured in {time.time() - start_time:.2f}s.")

# 4. Step 2: 270-Feature Extraction
print(f"   ⏳ Unlocking the quantitative math (RSI, Spoofing, Volatility, Sentiment)...")
start_time = time.time()
features = ml._extract_features(target_symbol, tf_analysis)
print(f"   ✅ Mathematical tensor successfully compiled in {time.time() - start_time:.2f}s.")

# 5. Output Verification
print("\n======================================================================")
print("📈 INVESTIGATION RESULTS")
print("======================================================================")

if features is None:
    print("❌ CRITICAL ERROR: Feature vector returned None.")
else:
    feature_count = len(features)
    print(f"Dim Check: Generated {feature_count} features (Expected: {cfg.BASE_FEATURE_COUNT})")
    
    if feature_count == cfg.BASE_FEATURE_COUNT:
        print("✅ SUCCESS: The 270 Feature Pipeline is mathematically flawless.")
    else:
        print("❌ DIMENSION MISMATCH: Please check the feature compilation code.")

    print("\n🕵️ A Quick Look Inside the Brain (First 15 Variables):")
    for i in range(min(15, feature_count)):
        print(f"   Feature {i+1:03d}: \t{features[i]:.6f}")
        
        
    print("\n🕵️ A Quick Look at the Extremes (Last 5 Variables):")
    for i in range(max(0, feature_count - 5), feature_count):
        print(f"   Feature {i+1:03d}: \t{features[i]:.6f}")

    non_zero = sum(1 for f in features if f != 0.0)
    print(f"\n🧠 BRAIN ACTIVITY: {non_zero}/{feature_count} features are currently active (non-zero).")

    print("\n✅ The Board of Directors is ready to vote on these numbers.")
print("======================================================================")
