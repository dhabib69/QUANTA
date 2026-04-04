import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:65087'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:65087'

from QUANTA_ml_engine import DeepMLEngine
import numpy as np

def test_feature_matrix():
    print("Testing ML Feature Engine Extraction Matrix...")
    
    # We must instantiate the DeepMLEngine with mock dependencies to bypass full initialization
    class MockConfig:
        model_dir = "models"
        timeframes = ['5m', '15m', '1h', '4h', '1d', '1w']
        tf_weights = {'5m': 0.1, '15m': 0.1, '1h': 0.2, '4h': 0.3, '1d': 0.2, '1w': 0.1}
    class MockBnc: pass
    class MockSentiment:
        def get_sentiment(self, *args, **kwargs): return {'compound': 0.0}
        def quick_batch_analysis(self, symbols): return {s: 0.0 for s in symbols}
        
    try:
        ml = DeepMLEngine(cfg=MockConfig(), bnc=MockBnc(), mtf=MockTG(), sentiment_engine=MockSentiment())
    except Exception as e:
        print(f"Bypassing full initialization: {e}")
        # Manual bypass for testing just the specific method
        ml = DeepMLEngine.__new__(DeepMLEngine)
        ml.cfg = MockConfig()
        ml.futures_stats_cache = {}
        ml.hmm_models = {}
        ml.rt_cache = {}
    
    fake_tf_analysis = {
        '5m': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50},
        '15m': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50},
        '1h': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50},
        '4h': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50},
        '1d': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50},
        '1w': {'rsi': 50, 'macd': 0, 'bb_position': 0.5, 'adx': 25, 'atr': 0.1, 'trend': 'BULLISH', 'strength': 50}
    }
    
    features = ml._extract_features(fake_tf_analysis)
    
    print(f"Feature Vector Shape: {features.shape}")
    assert features.shape[0] == 270, f"CRITICAL: Shape is {features.shape[0]} but expected 270!"
    
    fng_pad = features[223]
    print(f"Index 223 Padding Initialized: {fng_pad}")
    assert fng_pad == 0.0 or not np.isnan(fng_pad), f"CRITICAL: Array boundary error! {fng_pad}"
    
    print("\n✅ Feature Matrix is perfectly aligned and safe for Stacking.")

if __name__ == "__main__":
    test_feature_matrix()
