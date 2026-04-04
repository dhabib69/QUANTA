import sys
import numpy as np
sys.path.insert(0, r'c:\Users\habib\QUANTA')
import importlib, types
# Import Config from QUANTA_bot without running main
_bot = importlib.import_module('QUANTA_bot')
Config = _bot.Config
from quanta_exchange import BinanceAPIEnhanced
from quanta_features import MultiTimeframeAnalyzer, fractional_differentiation
from QUANTA_trading_core import PaperTrading
from QUANTA_bot import Config
cfg = Config()

def run_verification():
    print("="*50)
    print("QUANTA PHASE 2 VERIFICATION TEST")
    print("="*50)

    # Test 1: ORDER BOOK SNAPSHOT (LOB)
    print("\n[Test 1] Order Book Snapshot LOB...")
    try:
        cfg = Config()
        bnc = BinanceAPIEnhanced(cfg)
        ob = bnc.get_order_book('BTCUSDT', limit=10)
        bids_total = sum(float(b[1]) for b in ob['bids'])
        asks_total = sum(float(a[1]) for a in ob['asks'])
        imbalance = (bids_total - asks_total) / (bids_total + asks_total + 1e-9)
        print(f"  LOB OK | Bids: {bids_total:.2f}  Asks: {asks_total:.2f}  Imbalance: {imbalance:+.4f}")
    except Exception as e:
        print(f"  LOB FAILED: {e}")

    # Test 2: FRACTIONAL DIFFERENTIATION
    print("\n[Test 2] Fractional Differentiation stationarity...")
    try:
        prices = np.cumsum(np.random.randn(500)) + 100
        fd = fractional_differentiation(prices, d=0.4, threshold=1e-4)
        if len(fd) > 0 and not np.all(np.isnan(fd)):
            print(f"  FracDiff OK | len={len(fd)}  mean={float(np.nanmean(fd)):.4f}  std={float(np.nanstd(fd)):.4f}")
        else:
            print("  FracDiff returned NaNs or empty array")
    except Exception as e:
        print(f"  FracDiff FAILED: {e}")

    # Test 3: KELLY CRITERION POSITION SIZING
    print("\n[Test 3] Kelly Criterion position sizing...")
    try:
        class MockBot:
            def __init__(self, c):
                self.cfg = c
        cfg = Config()
        bot_mock = MockBot(cfg)
        paper = PaperTrading(initial_balance=10000, bot=bot_mock)
        paper.open_position('BTCUSDT', 80000, 'BULLISH', confidence=60, atr_percent=1.0)
        
        import time
        for _ in range(150):
            if 'BTCUSDT' in paper.positions:
                break
            time.sleep(1.0)
        
        size_60 = paper.positions.get('BTCUSDT', {}).get('size', 0)
        if 'BTCUSDT' in paper.positions:
            del paper.positions['BTCUSDT']
            
        paper.open_position('BTCUSDT', 80000, 'BULLISH', confidence=85, atr_percent=1.0)
        
        for _ in range(150):
            if 'BTCUSDT' in paper.positions:
                break
            time.sleep(1.0)
            
        size_85 = paper.positions.get('BTCUSDT', {}).get('size', 0)
        print(f"  Kelly OK | 60% conf size={size_60:.4f} | 85% conf size={size_85:.4f}")
        assert size_85 > size_60, "Higher confidence should produce larger size"
        print(f"  Correct: size_85 > size_60")
    except Exception as e:
        print(f"  Kelly FAILED: {e}")

    print("\n" + "="*50)
    print(f"BASE_FEATURE_COUNT = {cfg.BASE_FEATURE_COUNT}")
    print("PHASE 2 VERIFICATION COMPLETE")
    print("="*50)

if __name__ == '__main__':
    run_verification()
