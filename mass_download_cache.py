"""
QUANTA Mass Cache Filler -- Binance Vision Archive Downloader
Downloads up to 4000 days of 5m kline data for all major futures pairs
directly from data.binance.vision ZIP archives (no API needed).
"""
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:52650"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:52650"

import sys
import logging
from quanta_cache import FeatherCache
from quanta_archive import BinanceArchiveDownloader

import requests

def get_active_futures():
    """Fetch all currently active USDT Perpetual Futures directly from Binance API"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        proxies = {
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY")
        }
        r = requests.get(url, timeout=10, proxies=proxies, verify=False)
        r.raise_for_status()
        data = r.json()
        
        symbols = [
            s['symbol'] for s in data['symbols'] 
            if s['status'] == 'TRADING' 
            and s['contractType'] == 'PERPETUAL' 
            and s['symbol'].endswith('USDT')
        ]
        return sorted(symbols)
    except Exception as e:
        print(f"Failed to fetch active symbols from API: {e}")
        sys.exit(1)

def run_mass_download():
    print("=" * 60)
    print("QUANTA Mass Cache Filler (Binance Vision Archives)")
    print("=" * 60)
    
    cache = FeatherCache("feather_cache")
    downloader = BinanceArchiveDownloader(cache)
    
    symbols = get_active_futures()
    print(f"\n[TARGET] Target: {len(symbols)} futures pairs x 4000 days of 5m data\n")
    
    success = skip = fail = 0
    
    for i, symbol in enumerate(symbols):
        print(f"--- [{i+1}/{len(symbols)}] {symbol} ", end="")
        
        # Skip if already heavily cached (>100k candles ≈ 347 days)
        cached = cache.get(symbol, "5m", 150000)
        if cached and len(cached) >= 100000:
            print(f"SKIP ({len(cached):,} candles already cached)")
            skip += 1
            continue
        
        print("downloading...")
        try:
            result = downloader.fetch_historical_archive(symbol, "5m", 4000)
            if result:
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"   [FAIL] {e}")
            fail += 1
    
    print(f"\n{'=' * 60}")
    print(f"DONE: {success} downloaded, {skip} skipped, {fail} failed")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    run_mass_download()
