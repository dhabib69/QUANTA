import sys
sys.path.insert(0, r'c:\Users\habib\QUANTA')
from QUANTA_bot import Config
from quanta_exchange import BinanceAPIEnhanced
from quanta_features import MultiTimeframeAnalyzer
from QUANTA_ml_engine import DeepMLEngine

def test_gnn_integration():
    print("Initializing Config and API...")
    cfg = Config()
    bnc = BinanceAPIEnhanced(cfg)
    mtf = MultiTimeframeAnalyzer(cfg, bnc)
    
    print("Initializing DeepMLEngine...")
    ml_engine = DeepMLEngine(cfg, bnc, mtf)
    
    print("Testing GNN background feed start...")
    try:
        # Start the background feed
        ml_engine.start_graph_background_feed(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'])
        print("✅ Background feed started successfully")
    except Exception as e:
        print(f"❌ Background feed failed: {e}")
        
    print("Testing ML feature extraction (including GNN and Onchain)...")
    try:
        # Test feature extraction. Note: in QUANTA, _extract_features takes symbol and raw_state
        # We can just call it with dummy recent data or wait, _extract_features expects raw_state dictionary 
        # from the MTF. Let's just run _extract_features directly if possible, or trigger it via a mock state.
        state = {
            'symbol': 'BTCUSDT',
            'tf_analysis': {},
        }
        # mock tf_analysis to prevent crashes
        for tf in ['1m', '5m', '15m', '1h', '4h', '1d', '1w']:
            state['tf_analysis'][tf] = {
                'rsi': 50,
                'macd': 0,
                'signal': 0,
                'hist': 0,
                'bb_upper': 100,
                'bb_middle': 95,
                'bb_lower': 90,
                'atr': 5,
                'stoch_k': 50,
                'stoch_d': 50,
                'adx': 25,
                'plus_di': 25,
                'minus_di': 25,
                'latest_close': 95000,
                'volume': 1000
            }
            
        features = ml_engine._extract_features('BTCUSDT', state)
        if features is not None:
            print(f"✅ Feature extraction successful! Total features: {len(features)}")
        else:
            print("❌ Feature extraction returned None")
    except Exception as e:
        print(f"❌ Feature extraction failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_gnn_integration()
