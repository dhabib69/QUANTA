from quanta_exchange import BinanceAPIEnhanced
from quanta_cache import FeatherCache

class DummyConfig:
    rest_url = "https://fapi.binance.com"
    cache_enabled = True
    cache_dir = "feather_cache"

def test_integration():
    print("Testing BinanceArchiveDownloader Integration...")
    
    cache = FeatherCache()
    exchange = BinanceAPIEnhanced(cfg=DummyConfig())
    
    symbol = "ETHUSDT" # Assuming ETHUSDT might not have exactly 90 days cached yet in this test context
    interval = "5m"
    days = 90 # 90 days * 288 candles/day = 25,920 candles (triggers > 10,000 threshold)
    
    print(f"Requesting {days} days of {symbol} {interval} data...")
    klines = exchange.get_historical_klines(symbol, interval, days=days, training_mode=False)
    
    if klines:
        print(f"Success! get_historical_klines returned {len(klines)} candles.")
        print(f"Sample First: {klines[0]}")
        print(f"Sample Last:  {klines[-1]}")
    else:
        print("Failed to get klines.")

if __name__ == "__main__":
    test_integration()
