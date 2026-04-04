import asyncio
from quanta_archive import BinanceArchiveDownloader

def test_archive():
    print("Testing BinanceArchiveDownloader...")
    
    # We pass None for cache_instance just to test the fetching/parsing logic
    # without actually writing to disk
    class DummyCache:
        def set(self, symbol, interval, klines):
            print(f"DummyCache: Would save {len(klines)} candles for {symbol}")
            
    downloader = BinanceArchiveDownloader(cache_instance=DummyCache())
    
    # Test fetching 90 days of BTCUSDT 5m data (approx 3 months)
    print("Fetching 90 days of BTCUSDT...")
    result = downloader.fetch_historical_archive("BTCUSDT", "5m", 90)
    
    if result:
        print(f"Success! Fetched {len(result)} total candles.")
        print("First candle sample:")
        print(f"  Time:  {result[0][0]}")
        print(f"  Close: {result[0][4]}")
        
        print("\nLast candle sample:")
        print(f"  Time:  {result[-1][0]}")
        print(f"  Close: {result[-1][4]}")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    test_archive()
