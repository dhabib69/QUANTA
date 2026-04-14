import time
import requests
import logging
import threading
import json
import numpy as np
from collections import defaultdict
from QUANTA_network import NetworkHelper
from quanta_cache import FeatherCache
from quanta_features import Indicators

class BaseAPI:
    """Base class for all exchange APIs"""
    def __init__(self, name, base_url):
        self.name = name
        self.base_url = base_url
        self.request_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
    
    def get_stats(self):
        with self.lock:
            success_rate = ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 0
            return {
                'name': self.name,
                'requests': self.request_count,
                'errors': self.error_count,
                'success_rate': success_rate
            }
    
    def _log_request(self, success=True):
        with self.lock:
            self.request_count += 1
            if not success:
                self.error_count += 1

# =================== ENHANCED BINANCE API WITH CACHE ===================
class BinanceAPIEnhanced(BaseAPI):
    """Enhanced Binance API with intelligent caching"""
    def __init__(self, cfg):
        super().__init__("Binance", cfg.rest_url)
        self.cfg = cfg
        
        # Initialize cache if enabled - NEXUS: FeatherCache
        self.cache = FeatherCache(cfg.cache_dir) if cfg.cache_enabled else None
        
        # Real-time cache for recent candles
        self.kline_cache = defaultdict(lambda: defaultdict(dict))
        self.cache_duration = 30
        self.cache_lock = threading.Lock()
        
        cache_type = "FEATHER CACHE ENABLED (2.5x faster)" if cfg.cache_enabled else "NO CACHE"
        print(f"📊 Binance API with {cache_type}")
    
    def get_pairs(self):
        """Get trading pairs with enhanced error handling and fallback"""
        max_attempts = 8  # ✅ FIXED: Increased from 5 to 8 attempts
        
        for attempt in range(max_attempts):
            try:
                # Add delay between attempts
                if attempt > 0:
                    delay = min(2 ** attempt, 60)  # ✅ FIXED: Exponential backoff, max 60s (was 30s)
                    print(f"   🔄 Retry {attempt}/{max_attempts} in {delay}s...")
                    time.sleep(delay)
                
                # Make request with extended timeout
                response = NetworkHelper.get(
                    f"{self.base_url}/exchangeInfo", 
                    timeout=20,  # Increased timeout
                    max_retries=3
                )
                
                if not response:
                    logging.warning(f"⚠️ get_pairs: No response from exchangeInfo (attempt {attempt+1})")
                    continue
                
                # Validate response has content
                if not response.text or len(response.text) < 100:
                    logging.warning(f"⚠️ get_pairs: Empty/short response ({len(response.text) if response.text else 0} bytes) (attempt {attempt+1})")
                    if response.text:
                        logging.warning(f"   Response content: {response.text}")
                    continue
                
                # Try to parse JSON
                try:
                    data = response.json()
                except Exception as je:
                    logging.warning(f"⚠️ get_pairs: JSON decode error (attempt {attempt+1}): {je} | response preview: {response.text[:200]}")
                    continue
                
                # Validate response structure
                if 'symbols' not in data:
                    logging.warning(f"⚠️ get_pairs: Response missing 'symbols' key. Keys found: {list(data.keys())[:10]}")
                    continue
                
                # Extract USDT perpetual pairs
                pairs = [
                    s['symbol'] for s in data.get('symbols', [])
                    if s['symbol'].endswith('USDT') 
                    and s.get('contractType') == 'PERPETUAL' 
                    and s['status'] == 'TRADING'
                ][:100]
                
                if len(pairs) > 0:
                    self._log_request(success=True)
                    logging.info(f"✅ Successfully fetched {len(pairs)} trading pairs")
                    return pairs
                else:
                    pass  # Silenced: No valid pairs (fallback will be used)
                    continue
                    
            except requests.exceptions.ConnectionError as e:
                logging.warning(f"⚠️ get_pairs ConnectionError (attempt {attempt+1}/{max_attempts}): {e}")
                continue
                
            except requests.exceptions.Timeout as e:
                logging.warning(f"⚠️ get_pairs Timeout (attempt {attempt+1}/{max_attempts}): {e}")
                continue
                
            except Exception as e:
                logging.error(f"Unexpected error in get_pairs (attempt {attempt+1}/{max_attempts}): {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # All attempts failed - try ticker/24hr as fallback source for pairs
        logging.warning("⚠️ exchangeInfo failed, trying ticker/24hr to extract pairs...")
        try:
            ticker_response = NetworkHelper.get(
                f"{self.base_url}/ticker/24hr", timeout=20, max_retries=3
            )
            if ticker_response and ticker_response.text:
                ticker_data = ticker_response.json()
                pairs = [
                    t['symbol'] for t in ticker_data
                    if t['symbol'].endswith('USDT')
                    and float(t.get('quoteVolume', 0)) > 0
                ][:100]
                if pairs:
                    logging.info(f"✅ Got {len(pairs)} pairs from ticker/24hr fallback")
                    return pairs
        except Exception as e:
            logging.error(f"ticker/24hr fallback also failed: {e}")

        logging.error("❌ All attempts to fetch pairs failed, using hardcoded fallback list")
        self._log_request(success=False)
        return self._get_fallback_pairs()
    
    def _get_fallback_pairs(self):
        """Fallback list of popular USDT perpetual pairs - EXPANDED"""
        fallback = [
            # Top 10 by volume
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
            # High volume alts
            'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'LTCUSDT'
        ]
        return fallback
    
    def validate_symbols(self, symbols):
        """Validate symbols by testing klines endpoint - filters out 400 errors"""
        if not symbols:
            return []
        
        print(f"🔍 Validating {len(symbols)} symbols...")
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            try:
                # Quick test - fetch just 1 candle
                response = NetworkHelper.get(
                    f"{self.base_url}/klines",
                    params={
                        'symbol': symbol,
                        'interval': '5m',
                        'limit': 1
                    },
                    timeout=5,
                    max_retries=1
                )
                
                if response and response.status_code == 200:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
                    
            except Exception:
                invalid_symbols.append(symbol)
        
        # If validation fails completely (all invalid), skip validation
        if len(valid_symbols) == 0:
            print(f"  ⚠️  Validation failed for all symbols - skipping validation")
            print(f"  → Using all {len(symbols)} symbols without validation")
            return symbols
        
        if invalid_symbols:
            print(f"  ❌ Filtered out {len(invalid_symbols)} invalid symbols")
            if len(invalid_symbols) <= 5:
                print(f"     {', '.join(invalid_symbols)}")
            else:
                print(f"     {', '.join(invalid_symbols[:5])} ... and {len(invalid_symbols)-5} more")
        
        print(f"✅ {len(valid_symbols)}/{len(symbols)} symbols validated")
        return valid_symbols
    
    def get_klines(self, symbol, interval='5m', limit=1500):
        """Get klines with intelligent caching"""
        
        # For small recent requests, use real-time cache
        if limit <= 10:
            cache_key = f"{symbol}_{interval}"
            
            with self.cache_lock:
                cached = self.kline_cache[cache_key]
                
                if cached.get('time') and (time.time() - cached['time']) < self.cache_duration:
                    data = cached['data']
                    if len(data) >= limit:
                        self._log_request(success=True)
                        return data[-limit:]
        
        # For historical data, use smart cache
        if self.cache and limit > 10:
            cached_data = self.cache.get(symbol, interval, limit)
            if cached_data is not None:
                # Successfully retrieved from cache
                self._log_request(success=True)
                
                # Update real-time cache with recent data
                if len(cached_data) >= 10:
                    cache_key = f"{symbol}_{interval}"
                    with self.cache_lock:
                        self.kline_cache[cache_key] = {
                            'data': cached_data[-100:],  # Keep last 100 in memory
                            'time': time.time()
                        }
                
                return cached_data
        
        # Fallback to API for cache miss or forced update
        try:
            response = NetworkHelper.get(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': min(1500, limit)},
                timeout=8,
                adaptive_timeout=True
            )
            
            if response:
                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError):
                    self._log_request(success=False)
                    logging.debug(f"Invalid JSON response for {symbol}")
                    return []
                
                # UNIVERSAL FIX: Ensure trades field (index 8) is always an integer
                # Binance sometimes returns it as string, float, or string-float
                for kline in data:
                    if len(kline) > 8:
                        try:
                            trades = kline[8]
                            # Handle all possible formats
                            if isinstance(trades, str):
                                # It's a string - could be "59010981" or "59010981.0"
                                if '.' in trades:
                                    kline[8] = int(float(trades))  # "59010981.0" → 59010981
                                else:
                                    kline[8] = int(trades)  # "59010981" → 59010981
                            elif isinstance(trades, float):
                                kline[8] = int(trades)  # 59010981.0 → 59010981
                            # If already int, leave it
                        except (ValueError, TypeError) as e:
                            # If conversion fails, default to 0
                            kline[8] = 0
                            pass  # Silenced: Failed to convert trades field (harmless)
                
                # Update cache if we have enough data
                if self.cache and len(data) >= 100:
                    self.cache.set(symbol, interval, data)
                
                # Update real-time cache
                cache_key = f"{symbol}_{interval}"
                with self.cache_lock:
                    self.kline_cache[cache_key] = {'data': data, 'time': time.time()}
                
                self._log_request(success=True)
                return data
            else:
                self._log_request(success=False)
                logging.debug(f"klines response was None for {symbol} - trying fallback URL")
                # Try alternative Binance endpoint
                try:
                    alt_url = "https://fapi1.binance.com/fapi/v1/klines"
                    response2 = NetworkHelper.get(
                        alt_url,
                        params={'symbol': symbol, 'interval': interval, 'limit': min(1500, limit)},
                        timeout=8,
                        adaptive_timeout=False
                    )
                    if response2:
                        try:
                            data = response2.json()
                        except (ValueError, json.JSONDecodeError):
                            self._log_request(success=False)
                            logging.debug(f"Invalid JSON from alt URL for {symbol}")
                            return []
                        for kline in data:
                            if len(kline) > 8:
                                try:
                                    trades = kline[8]
                                    if isinstance(trades, str):
                                        kline[8] = int(float(trades)) if '.' in trades else int(trades)
                                    elif isinstance(trades, float):
                                        kline[8] = int(trades)
                                except (ValueError, TypeError):
                                    kline[8] = 0
                        return data
                except Exception:
                    pass
                # Try cache as fallback
                if self.cache:
                    cached_data = self.cache.get(symbol, interval, limit)
                    if cached_data:
                        return cached_data
                return []
                
        except Exception as e:
            self._log_request(success=False)
            logging.error(f"Binance get_klines {symbol} {interval}: {e}")
            
            # Try cache as last resort
            if self.cache:
                cached_data = self.cache.get(symbol, interval, limit)
                if cached_data:
                    return cached_data
            
            return []
    
    def get_klines_from(self, symbol, interval, start_time, limit=15):
        """
        Fetch klines starting from a specific timestamp (ms).
        Used by check_predictions so the correct outcome window is always captured
        regardless of when the RL check runs (avoids getting wrong-window candles).
        """
        try:
            response = NetworkHelper.get(
                f"{self.base_url}/klines",
                params={
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': int(start_time),
                    'limit': limit
                },
                timeout=10,
                adaptive_timeout=False
            )
            if response:
                data = response.json()
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
        return []

    def get_historical_klines_paginated(self, symbol, interval, days=30):
        """
        Fetch historical klines using pagination to bypass 1000 candle limit.
        This allows fetching weeks/months of 1m data.
        """
        minutes_per_candle = self._interval_to_minutes(interval)
        total_candles_needed = (days * 1440) // minutes_per_candle
        
        if total_candles_needed <= 1000:
            return self.get_klines(symbol, interval, limit=total_candles_needed)
            
        # Check cache length instantly without loading into memory (OOM safeguard)
        cached_length = 0
        if self.cache and hasattr(self.cache, 'get_length'):
            cached_length = self.cache.get_length(symbol, interval)
            
        # If cache already covers 80%, we can just extract and return it instantly
        if cached_length >= total_candles_needed * 0.8:
            cat = self.cache.get(symbol, interval, limit=total_candles_needed) or []
            if len(cat) > total_candles_needed:
                return cat[-total_candles_needed:] # Trim exact length
            return cat
            
        candles_to_fetch = total_candles_needed
        if cached_length > 0:
            candles_to_fetch = max(0, total_candles_needed - cached_length)
            
        if candles_to_fetch <= 0:
            return self.cache.get(symbol, interval, limit=total_candles_needed) or []
        
        print(f"   ⚠️ Cache has {cached_length} candles. Need {total_candles_needed}. Bridging the delta...")

        # Test API before attempting pagination
        print(f"   🔍 Testing API for {symbol}...", end=" ", flush=True)
        test_response = NetworkHelper.get(
            f"{self.base_url}/klines",
            params={'symbol': symbol, 'interval': interval, 'limit': 10},
            timeout=10
        )
        if not test_response or test_response.status_code != 200:
            print("❌ API test failed")
            return self.cache.get(symbol, interval, limit=total_candles_needed) or []
            
        print("✅ API working")
        
        # Pagination required - BACKWARD FETCHING (more reliable)
        new_klines = []
        batch_size = 1000
        
        # Calculate exactly how many batches we ALREADY have so we can offset the loop
        completed_batches = cached_length // batch_size
        num_batches_to_fetch = (candles_to_fetch // batch_size) + 1
        
        print(f"   📥 Fetching {candles_to_fetch} older candles in {num_batches_to_fetch} batches (starting at offset batch {completed_batches})...")
        
        current_end_time = int(time.time() * 1000)
        consecutive_rate_limits = 0
        reached_listing_date = False
        
        for i in range(num_batches_to_fetch):
            if reached_listing_date:
                break
            batch_success = False
            max_retries = 5
            retry_delay = 3.0  # Start with 3 seconds
            
            batch_num = completed_batches + i
            
            for attempt in range(max_retries):
                try:
                    # Fetch backwards: each batch gets 1000 candles going back in time
                    batch_end_time = current_end_time - (batch_num * batch_size * minutes_per_candle * 60 * 1000)
                    
                    response = NetworkHelper.get(
                        f"{self.base_url}/klines",
                        params={
                            'symbol': symbol,
                            'interval': interval,
                            'endTime': batch_end_time,
                            'limit': batch_size
                        },
                        timeout=20
                    )
                    
                    if response and response.text:
                        # Check for rate limit (418 error)
                        if response.status_code == 418:
                            self._log_request(success=False)
                            consecutive_rate_limits += 1
                            if attempt < max_retries - 1:
                                # Progressive backoff: more rate limits = longer wait
                                wait_time = 60 + (consecutive_rate_limits * 30)  # 60s, 90s, 120s...
                                print(f"   ⚠️  Batch {batch_num+1}/{num_batches} - RATE LIMIT (418), waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"   ❌ Batch {batch_num+1}/{num_batches} - rate limit persists after {max_retries} attempts")
                                break
                        
                        # Reset rate limit counter on success
                        consecutive_rate_limits = 0
                        
                        # Debug first response
                        if batch_num == 0 and attempt == 0 and not response.text.startswith('['):
                            print(f"\n   🔍 DEBUG - Response preview: {response.text[:200]}")
                        
                        try:
                            batch_data = response.json()
                            
                            if not isinstance(batch_data, list):
                                if attempt < max_retries - 1:
                                    print(f"   ⚠️  Batch {batch_num+1}/{num_batches} - invalid data type, retry {attempt+1}/{max_retries} (waiting {retry_delay:.1f}s)")
                                    time.sleep(retry_delay)
                                    retry_delay *= 1.5
                                    continue
                                else:
                                    print(f"   ❌ Batch {batch_num+1}/{num_batches} - failed after {max_retries} attempts")
                                    break
                            
                            # Fix trades field
                            for kline in batch_data:
                                if len(kline) > 8:
                                    try:
                                        trades = kline[8]
                                        if isinstance(trades, str):
                                            kline[8] = int(float(trades)) if '.' in trades else int(trades)
                                        elif isinstance(trades, float):
                                            kline[8] = int(trades)
                                    except (ValueError, TypeError):
                                        kline[8] = 0
                            
                            # Successfully fetched
                            new_klines = batch_data + new_klines  # Prepend because we're fetching backwards
                            batch_success = True
                            consecutive_rate_limits = 0
                            
                            # Print progress every 10 batches
                            if (i + 1) % 10 == 0 or (i + 1) == num_batches_to_fetch:
                                print(f"   ✅ Progress: {i + 1}/{num_batches_to_fetch} incremental batches fetched")
                            
                            # Check early stopping: reached listing date
                            if len(batch_data) < batch_size:
                                print(f"   📈 Reached listing date for {symbol} (fetched {len(batch_data)} in final batch). Stopping early.")
                                reached_listing_date = True
                                break
                                
                            # Patient delay to avoid rate limits
                            if batch_num < num_batches_to_fetch - 1:
                                # Base delay + extra if we've been rate limited recently
                                base_delay = 3.0
                                extra_delay = min(consecutive_rate_limits * 2, 10)  # Up to 10s extra
                                total_delay = base_delay + extra_delay
                                if extra_delay > 0:
                                    print(f"   ⏸️  Extra cooling period: {total_delay:.1f}s (recent rate limits)")
                                time.sleep(total_delay)
                            break
                            
                        except (ValueError, json.JSONDecodeError) as je:
                            self._log_request(success=False)
                            if attempt < max_retries - 1:
                                print(f"   ⚠️  Batch {batch_num+1}/{num_batches_to_fetch} - JSON error, retry {attempt+1}/{max_retries} (waiting {retry_delay:.1f}s)")
                                print(f"        Response preview: {response.text[:150]}")
                                time.sleep(retry_delay)
                                retry_delay *= 1.5
                                continue
                            else:
                                print(f"   ❌ Batch {batch_num+1}/{num_batches_to_fetch} - JSON error after {max_retries} attempts")
                                print(f"        Response: {response.text[:200]}")
                                break
                    else:
                        self._log_request(success=False)
                        if attempt < max_retries - 1:
                            print(f"   ⚠️  Batch {batch_num+1}/{num_batches_to_fetch} - no response, retry {attempt+1}/{max_retries} (waiting {retry_delay:.1f}s)")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        elif attempt == max_retries - 1:
                            print(f"   ❌ Batch {batch_num+1}/{num_batches_to_fetch} - no response after {max_retries} attempts")
                            break
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        self._log_request(success=False)
                        print(f"   ⚠️  Batch {batch_num+1}/{num_batches_to_fetch} - error: {str(e)[:100]}, retry {attempt+1}/{max_retries}")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    elif attempt == max_retries - 1:
                        self._log_request(success=False)
                        print(f"   ❌ Batch {i+1}/{num_batches_to_fetch} - failed after {max_retries} attempts: {str(e)[:100]}")
                        break
        
        # Cache the result (appends directly to local NVMe Feather array without loading)
        if self.cache and len(new_klines) > 0:
            self.cache.set(symbol, interval, new_klines)
        
        # Sequence assembled. Retrieve the final combined array natively
        final_klines = self.cache.get(symbol, interval, limit=total_candles_needed) or new_klines
        print(f"   ✅ Sequence assembled cleanly. Length: {len(final_klines)} candles")
        return final_klines
    
    def get_historical_klines(self, symbol, interval, days=365, training_mode=False):
        """Get historical klines - uses pagination for large requests.
        
        Args:
            training_mode: If True, accepts ANY cached data >= 500 candles (permanent cache).
                          This makes subsequent training runs much faster.
        """
        minutes_per_candle = self._interval_to_minutes(interval)
        required_candles = (days * 1440) // minutes_per_candle + 50
        
        # TRAINING MODE: Use cached data immediately if available (permanent cache)
        if training_mode and self.cache:
            path = self.cache._get_path(symbol, interval)
            if path.exists():
                try:
                    import pandas as pd
                    df = pd.read_feather(path)
                    if not df.empty and len(df) >= 500:
                        klines = self.cache._df_to_klines(df)
                        if len(klines) >= 500:
                            return klines
                except Exception:
                    pass  # Fall through to API fetch
        
        # ALWAYS check cache first, regardless of request size
        if self.cache:
            cached_data = self.cache.get(symbol, interval, required_candles)
            if cached_data and len(cached_data) >= min(required_candles * 0.95, 1000):
                return cached_data
        
        # Cache miss - use pagination for large requests (>1000 candles)
        if required_candles > 10000:
            print(f"🚀 Large data request ({required_candles} candles). Routing to Binance Vision Archive...")
            from quanta_archive import BinanceArchiveDownloader
            downloader = BinanceArchiveDownloader(self.cache)
            archive_klines = downloader.fetch_historical_archive(symbol, interval, days)
            
            if archive_klines and len(archive_klines) > 500:
                return archive_klines
            else:
                print("⚠️ Archive fetch failed or returned empty. Falling back to slow API pagination...")
                return self.get_historical_klines_paginated(symbol, interval, days)
                
        elif required_candles > 1000:
            return self.get_historical_klines_paginated(symbol, interval, days)
        
        # Small request - direct API call
        return self.get_klines(symbol, interval, limit=required_candles)
    
    def _interval_to_minutes(self, interval):
        mapping = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
        return mapping.get(interval, 1)

    def get_ticker(self, symbol):
        """Fetch the latest price for a symbol via REST API."""
        try:
            response = NetworkHelper.get(
                f"{self.base_url}/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if response:
                data = response.json()
                return float(data.get('price', 0))
        except Exception as e:
            logging.debug(f"get_ticker failed for {symbol}: {e}")
        return 0
    
    def get_funding_rate(self, symbol):
        try:
            response = NetworkHelper.get(
                f"{self.base_url}/premiumIndex", 
                params={'symbol': symbol}, 
                timeout=5
            )
            if response:
                self._log_request(success=True)
                try:
                    return float(response.json().get('lastFundingRate', 0)) * 100
                except (ValueError, json.JSONDecodeError):
                    self._log_request(success=False)
                    return None
            self._log_request(success=False)
            return None
        except Exception as e:
            logging.debug(f"get_funding_rate failed: {e}")
            self._log_request(success=False)
            return None

    def get_order_book(self, symbol, limit=20):
        """Fetch Limit Order Book (LOB) snapshot for microstructure analysis"""
        try:
            response = NetworkHelper.get(
                f"{self.base_url}/depth", 
                params={'symbol': symbol, 'limit': limit}, 
                timeout=5
            )
            if response:
                self._log_request(success=True)
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError):
                    self._log_request(success=False)
                    return None
            self._log_request(success=False)
            return None
        except Exception as e:
            logging.debug(f"get_order_book failed: {e}")
            self._log_request(success=False)
            return None

    def get_top_movers(self, limit=50):
        """
        🔥 RESEARCH-BACKED COIN SELECTION
        Based on: MDPI 2025, Springer Financial Innovation 2021/2025, FreqAI
        
        Selection criteria:
        1. Liquidity first (volume > $10M)
        2. Regime diversity (bull/bear/range/volatile)
        3. Multi-factor scoring (not just % change)
        4. Forced diversity slots
        
        Result: Balanced training data across ALL market conditions
        """
        try:
            print("\n🔍 Fetching market data for coin selection...")
            response = NetworkHelper.get(f"{self.base_url}/ticker/24hr", timeout=15, adaptive_timeout=True)
            if not response:
                logging.warning("get_top_movers: No response from ticker/24hr")
                return self._get_fallback_top_movers(limit)
            
            try:
                tickers = response.json()
            except (ValueError, json.JSONDecodeError):
                logging.warning("get_top_movers: Invalid JSON response")
                return self._get_fallback_top_movers(limit)
            
            if not tickers or not isinstance(tickers, list):
                logging.warning("get_top_movers: Invalid ticker data")
                return self._get_fallback_top_movers(limit)
            
            # ================================================
            # STEP 0: FETCH ACTIVE SYMBOLS (Remove Delisted)
            # ================================================
            print("🌐 Fetching active trading pairs from exchangeInfo...")
            active_symbols = set()
            try:
                info_response = NetworkHelper.get(f"{self.base_url}/exchangeInfo", timeout=10)
                if info_response:
                    info_data = info_response.json()
                    if 'symbols' in info_data:
                        for s in info_data['symbols']:
                            if s.get('status') == 'TRADING' and s.get('contractType') == 'PERPETUAL':
                                active_symbols.add(s.get('symbol'))
                if active_symbols:
                    print(f"✅ Found {len(active_symbols)} active perpetual pairs")
            except Exception as e:
                logging.warning(f"Failed to fetch exchangeInfo: {e}. Will proceed without strict delist filtering.")
            
            # ================================================
            # STEP 1: ELIGIBILITY FILTER (Research-backed)
            # ================================================
            print("📊 Applying liquidity and age filters...")
            
            candidates = []
            for t in tickers:
                symbol = t.get('symbol', '')
                if not symbol.endswith('USDT'):
                    continue
                
                # Active/Delisted filter: only accept if it's in active_symbols
                if active_symbols and symbol not in active_symbols:
                    continue
                
                # Liquidity filter: $10M+ daily volume (MDPI 2025)
                volume_usd = float(t.get('quoteVolume', 0))
                if volume_usd < 10_000_000:  # $10M minimum
                    continue
                
                # Remove stablecoins
                if any(stable in symbol for stable in ['BUSD', 'USDC', 'TUSD', 'DAI', 'FDUSD']):
                    continue
                
                candidates.append({
                    'symbol': symbol,
                    'volume_24h': volume_usd,
                    'price_change_pct': float(t.get('priceChangePercent', 0)),
                    'price': float(t.get('lastPrice', 1)),
                    'high': float(t.get('highPrice', 1)),
                    'low': float(t.get('lowPrice', 1)),
                })
            
            if len(candidates) < 20:
                logging.warning(f"Only {len(candidates)} candidates passed filter, using fallback")
                return self._get_fallback_top_movers(limit)
            
            print(f"✅ {len(candidates)} coins passed liquidity filter ($10M+ volume)")
            
            # ================================================
            # STEP 2: CALCULATE 30-DAY METRICS & SCORES
            # ================================================
            print("📈 Calculating 30-day metrics and scoring...")
            
            scored_coins = []
            for coin in candidates:
                try:
                    # Fetch 30 days of data for proper metrics
                    klines_30d = self.get_klines(coin['symbol'], '1d', limit=30)
                    if not klines_30d or len(klines_30d) < 20:
                        continue
                    
                    # Extract data
                    closes = np.array([float(k[4]) for k in klines_30d])
                    highs = np.array([float(k[2]) for k in klines_30d])
                    lows = np.array([float(k[3]) for k in klines_30d])
                    volumes = np.array([float(k[5]) for k in klines_30d])
                    
                    # 30-day momentum (absolute value - we want big movers in BOTH directions)
                    pct_change_30d = (closes[-1] / closes[0] - 1) * 100
                    abs_momentum = abs(pct_change_30d)
                    
                    # Volatility (ATR as % of price)
                    atr = Indicators.atr(highs[-14:], lows[-14:], closes[-14:])
                    atr_pct = (atr / closes[-1]) * 100 if closes[-1] > 0 else 0
                    
                    # Trend clarity (ADX)
                    adx = Indicators.adx(highs, lows, closes)
                    
                    # Volume consistency (lower stddev = more reliable)
                    volume_stddev = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 999
                    volume_consistency = 1 / (1 + volume_stddev)  # Inverse - higher is better
                    
                    # ================================================
                    # MULTI-FACTOR SCORE (Research weights)
                    # ================================================
                    score = (
                        np.log10(coin['volume_24h']) * 0.30 +      # Liquidity (MDPI 2025)
                        abs_momentum * 0.25 +                       # Momentum (both directions!)
                        min(atr_pct, 10) * 0.20 +                  # Volatility (capped at 10%)
                        adx * 0.15 +                                # Trend clarity
                        volume_consistency * 10 * 0.10              # Consistency (scaled)
                    )
                    
                    scored_coins.append({
                        'symbol': coin['symbol'],
                        'score': score,
                        'pct_change_30d': pct_change_30d,
                        'abs_momentum': abs_momentum,
                        'atr_pct': atr_pct,
                        'adx': adx,
                        'volume_24h': coin['volume_24h'],
                    })
                    
                except Exception as e:
                    # Skip coins with data issues
                    continue
            
            if len(scored_coins) < 20:
                logging.warning(f"Only {len(scored_coins)} coins scored, using fallback")
                return self._get_fallback_top_movers(limit)
            
            print(f"✅ Scored {len(scored_coins)} coins on 5 dimensions")
            
            # Sort by score
            scored_coins.sort(key=lambda x: x['score'], reverse=True)
            
            # ================================================
            # STEP 3: FORCED DIVERSITY SLOTS (Critical!)
            # ================================================
            print("🎯 Enforcing regime diversity...")
            
            selections = []
            used_symbols = set()
            
            # Slot A: Top 5 strong gainers (>+15% 30d)
            gainers = [c for c in scored_coins if c['pct_change_30d'] > 15 and c['symbol'] not in used_symbols]
            gainers.sort(key=lambda x: x['score'], reverse=True)
            slot_a = gainers[:5]
            selections.extend(slot_a)
            used_symbols.update(c['symbol'] for c in slot_a)
            print(f"   ✅ Slot A (Gainers): {len(slot_a)} coins (+15% to +{max([c['pct_change_30d'] for c in slot_a]) if slot_a else 0:.1f}%)")
            
            # Slot B: Top 5 strong losers (<-15% 30d) ← CRITICAL FOR BALANCE!
            losers = [c for c in scored_coins if c['pct_change_30d'] < -15 and c['symbol'] not in used_symbols]
            losers.sort(key=lambda x: x['score'], reverse=True)
            slot_b = losers[:5]
            selections.extend(slot_b)
            used_symbols.update(c['symbol'] for c in slot_b)
            print(f"   ✅ Slot B (Losers): {len(slot_b)} coins ({min([c['pct_change_30d'] for c in slot_b]) if slot_b else 0:.1f}% to -15%)")
            
            # Slot C: Top 5 ranging (ADX < 20 - sideways market)
            ranging = [c for c in scored_coins if c['adx'] < 20 and c['symbol'] not in used_symbols]
            ranging.sort(key=lambda x: x['score'], reverse=True)
            slot_c = ranging[:5]
            selections.extend(slot_c)
            used_symbols.update(c['symbol'] for c in slot_c)
            print(f"   ✅ Slot C (Ranging): {len(slot_c)} coins (ADX < 20)")
            
            # Slot D: Top 5 high volatility (ATR leaders)
            volatile = [c for c in scored_coins if c['symbol'] not in used_symbols]
            volatile.sort(key=lambda x: x['atr_pct'], reverse=True)
            slot_d = volatile[:5]
            selections.extend(slot_d)
            used_symbols.update(c['symbol'] for c in slot_d)
            print(f"   ✅ Slot D (Volatile): {len(slot_d)} coins ({max([c['atr_pct'] for c in slot_d]) if slot_d else 0:.2f}% ATR)")
            
            # Slot E: Fixed anchors (always include)
            anchors = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            for anchor in anchors:
                if anchor not in used_symbols:
                    # Find in scored list or add manually
                    anchor_data = next((c for c in scored_coins if c['symbol'] == anchor), None)
                    if anchor_data:
                        selections.append(anchor_data)
                        used_symbols.add(anchor)
            print(f"   ✅ Slot E (Anchors): {len([s for s in selections if s['symbol'] in anchors])} coins (BTC/ETH/BNB)")
            
            # Slot F: Fill remaining with top-scored coins
            remaining_needed = limit - len(selections)
            remaining = [c for c in scored_coins if c['symbol'] not in used_symbols]
            slot_f = remaining[:remaining_needed]
            selections.extend(slot_f)
            print(f"   ✅ Slot F (Top Score): {len(slot_f)} coins (by overall score)")
            
            # ================================================
            # RETURN FINAL SELECTION
            # ================================================
            final_symbols = [c['symbol'] for c in selections[:limit]]
            
            print(f"\n✅ SELECTED {len(final_symbols)} COINS:")
            print(f"   Gainers: {len([c for c in selections if c['pct_change_30d'] > 15])}")
            print(f"   Losers: {len([c for c in selections if c['pct_change_30d'] < -15])}")
            print(f"   Ranging: {len([c for c in selections if -15 <= c['pct_change_30d'] <= 15])}")
            print(f"   High Vol: {len([c for c in selections if c['atr_pct'] > 3])}")
            print(f"   Avg Volume: ${np.mean([c['volume_24h'] for c in selections])/1e6:.1f}M")
            
            self._log_request(success=True)
            return final_symbols
            
        except Exception as e:
            logging.error(f"get_top_movers error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_top_movers(limit)
    
    def _get_fallback_top_movers(self, limit=50):
        """
        Fallback with REGIME DIVERSITY
        Manually curated to ensure bull/bear/range representation
        """
        fallback = [
            # Anchors (always trending)
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            
            # High liquidity large caps
            'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'MATICUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT', 'ATOMUSDT',
            
            # Mid caps (diverse behavior)
            'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
            'SUIUSDT', 'INJUSDT', 'RNDRUSDT', 'TAOUSDT', 'WLDUSDT',
            
            # High volatility candidates
            'SEIUSDT', 'TIAUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FILUSDT',
            'ICPUSDT', 'ALGOUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT',
            
            # Additional diversity
            'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'GMTUSDT', 'FTMUSDT',
            'VETUSDT', 'APEUSDT', 'GALAUSDT', 'CHZUSDT', 'ENJUSDT'
        ]
        logging.info(f"📋 Using fallback selection: {len(fallback[:limit])} pairs (diversity ensured)")
        return fallback[:limit]
    
    def get_spike_dump_candidates(self, limit=20):
        """Get spike/dump candidates with fallback"""
        try:
            response = NetworkHelper.get(f"{self.base_url}/ticker/24hr", timeout=15, adaptive_timeout=True)
            if not response:
                logging.warning("get_spike_dump_candidates: No response from ticker/24hr endpoint")
                self._log_request(success=False)
                return self._get_fallback_spike_dump(limit)
            
            try:
                tickers = response.json()
            except (ValueError, json.JSONDecodeError):
                logging.warning("get_spike_dump_candidates: Invalid JSON response")
                self._log_request(success=False)
                return self._get_fallback_spike_dump(limit)
            if not tickers or not isinstance(tickers, list):
                logging.warning("get_spike_dump_candidates: Invalid ticker data received")
                self._log_request(success=False)
                return self._get_fallback_spike_dump(limit)
            
            candidates = []
            
            for t in tickers:
                if not t.get('symbol', '').endswith('USDT'):
                    continue
                
                try:
                    price_change = float(t.get('priceChangePercent', 0))
                    volume = float(t.get('volume', 0))
                    
                    if abs(price_change) > 5 and volume > 2000000:
                        candidates.append({
                            'symbol': t['symbol'],
                            'change': abs(price_change),
                            'volume': volume
                        })
                except (ValueError, KeyError, TypeError) as e:
                    logging.debug(f"Spike/dump candidate parse error: {e}")
                    continue
            
            if not candidates:
                logging.info("get_spike_dump_candidates: No significant movers found, using fallback")
                self._log_request(success=False)
                return self._get_fallback_spike_dump(limit)
            
            candidates.sort(key=lambda x: x['change'] * (x['volume'] / 1000000), reverse=True)
            self._log_request(success=True)
            result = [c['symbol'] for c in candidates[:limit]]
            logging.info(f"✅ get_spike_dump_candidates: Found {len(result)} candidates")
            return result
            
        except Exception as e:
            logging.error(f"get_spike_dump_candidates error: {e}")
            self._log_request(success=False)
            return self._get_fallback_spike_dump(limit)
    
    def _get_fallback_spike_dump(self, limit=20):
        """Fallback list prioritizing historically volatile coins"""
        fallback = [
            # Known for high volatility
            'PEPEUSDT', 'SHIBUSDT', 'FLOKIUSDT', 'BONKUSDT',
            # Meme/volatile tokens
            'DOGEUSDT', 'WIFUSDT', 'BOMEUSDT',
            # AI tokens (high volatility)
            'TAOUSDT', 'RNDRUSDT', 'WLDUSDT', 'FETUSDT',
            # Layer 2s (can spike)
            'ARBUSDT', 'OPUSDT', 'SEIUSDT', 'SUIUSDT',
            # DeFi volatility
            'UNIUSDT', 'AAVEUSDT', 'COMPUSDT',
            # Gaming (volatile)
            'AXSUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        logging.info(f"📋 Using fallback spike/dump: {len(fallback[:limit])} pairs")
        return fallback[:limit]
    
    def get_sniper_coins(self, limit=50):
        """
        🎯 SNIPER COIN SELECTION FOR LIVE PREDICTION
        Selects coins already exhibiting strong short-term momentum.

        Academic basis:
        - Momentum persistence: Jegadeesh & Titman (1993)
        - Volume confirmation: Karpoff (1987)
        - Trend strength: Wilder (1978) ADX
        - Short-term continuation: Lehmann (1990)

        Scoring weights: 1h return 0.30 | 4h return 0.20 | volume surge 0.15
                         ADX+direction 0.20 | RSI 0.10 | consecutive candles 0.05

        Returns top limit//2 bull + top limit//2 bear coins merged (balanced).
        Zero new API calls beyond /ticker/24hr — all OHLCV from FeatherCache.
        """
        try:
            response = NetworkHelper.get(f"{self.base_url}/ticker/24hr", timeout=15, adaptive_timeout=True)
            if not response:
                logging.warning("get_sniper_coins: No ticker response, falling back")
                return self._get_fallback_top_movers(limit)

            try:
                tickers = response.json()
            except Exception:
                return self._get_fallback_top_movers(limit)

            if not tickers or not isinstance(tickers, list):
                return self._get_fallback_top_movers(limit)

            STABLES = {'BUSD', 'USDC', 'TUSD', 'DAI', 'FDUSD', 'WBTC', 'STETH'}
            MIN_VOL = 2_000_000

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT'):
                    continue
                if any(s in sym for s in STABLES):
                    continue
                vol = float(t.get('quoteVolume', 0))
                if vol < MIN_VOL:
                    continue
                candidates.append(sym)

            if len(candidates) < 20:
                return self._get_fallback_top_movers(limit)

            # Sort by volume desc, score top 300 to keep computation fast
            ticker_map = {t['symbol']: t for t in tickers}
            candidates.sort(key=lambda s: float(ticker_map[s].get('quoteVolume', 0)), reverse=True)
            candidates = candidates[:300]

            bull_scores, bear_scores = [], []

            for sym in candidates:
                try:
                    # Use FeatherCache: 60 x 5m candles (~5 hours)
                    klines = self.get_klines(sym, '5m', limit=60)
                    if not klines or len(klines) < 48:
                        continue

                    closes = np.array([float(k[4]) for k in klines])
                    highs  = np.array([float(k[2]) for k in klines])
                    lows   = np.array([float(k[3]) for k in klines])
                    vols   = np.array([float(k[5]) for k in klines])

                    # --- 1h return (last 12 x 5m candles) ---
                    ret_1h = (closes[-1] / closes[-12] - 1) * 100 if closes[-12] > 0 else 0.0

                    # --- 4h return (last 48 x 5m candles) ---
                    ret_4h = (closes[-1] / closes[-48] - 1) * 100 if len(closes) >= 48 and closes[-48] > 0 else 0.0

                    # --- Volume surge (vs avg of last 12 candles) ---
                    avg_vol = np.mean(vols[-12:]) if np.mean(vols[-12:]) > 0 else 1
                    vol_ratio = vols[-1] / avg_vol
                    vol_surge = (vol_ratio - 1.2) if vol_ratio > 1.2 else 0.0

                    # --- ADX + direction ---
                    adx_val, plus_di, minus_di = Indicators.adx_full(highs, lows, closes)
                    adx_bull = (adx_val / 50.0) if (adx_val > 25 and plus_di > minus_di) else 0.0
                    adx_bear = (adx_val / 50.0) if (adx_val > 25 and minus_di > plus_di) else 0.0

                    # --- RSI ---
                    rsi = Indicators.rsi(closes)
                    rsi_bull = (rsi - 50) / 50.0 if rsi > 55 else 0.0
                    rsi_bear = (50 - rsi) / 50.0 if rsi < 45 else 0.0

                    # --- Consecutive candles (capped at 10) ---
                    consec_bull = 0
                    for i in range(len(closes) - 1, 0, -1):
                        if closes[i] > closes[i-1]:
                            consec_bull += 1
                        else:
                            break
                    consec_bull = min(consec_bull, 10)

                    consec_bear = 0
                    for i in range(len(closes) - 1, 0, -1):
                        if closes[i] < closes[i-1]:
                            consec_bear += 1
                        else:
                            break
                    consec_bear = min(consec_bear, 10)

                    # --- Composite scores ---
                    bull = (
                        max(ret_1h, 0) * 0.30 +
                        max(ret_4h, 0) * 0.20 +
                        vol_surge       * 0.15 +
                        adx_bull        * 0.20 +
                        rsi_bull        * 0.10 +
                        (consec_bull / 10.0) * 0.05
                    )
                    bear = (
                        max(-ret_1h, 0) * 0.30 +
                        max(-ret_4h, 0) * 0.20 +
                        vol_surge        * 0.15 +
                        adx_bear         * 0.20 +
                        rsi_bear         * 0.10 +
                        (consec_bear / 10.0) * 0.05
                    )

                    if bull > 0:
                        bull_scores.append((sym, bull))
                    if bear > 0:
                        bear_scores.append((sym, bear))

                except Exception as e:
                    logging.debug(f"Sniper scoring error {sym}: {e}")
                    continue

            half = max(1, limit // 2)
            bull_scores.sort(key=lambda x: x[1], reverse=True)
            bear_scores.sort(key=lambda x: x[1], reverse=True)

            top_bull = [s for s, _ in bull_scores[:half]]
            top_bear = [s for s, _ in bear_scores[:half]]

            merged = list(dict.fromkeys(top_bull + top_bear))

            if not merged:
                logging.warning("get_sniper_coins: No scored coins, falling back")
                return self._get_fallback_top_movers(limit)

            logging.info(f"🎯 Sniper: {len(top_bull)} bull + {len(top_bear)} bear = {len(merged)} coins")
            print(f"🎯 Sniper selection: {len(top_bull)} bull + {len(top_bear)} bear = {len(merged)} coins")
            return merged[:limit]

        except Exception as e:
            logging.error(f"get_sniper_coins error: {e}")
            return self._get_fallback_top_movers(limit)

    def get_research_backed_coins(self, limit=50):
        """
        🔬 RESEARCH-BACKED COIN SELECTION
        Based on academic studies (MDPI 2025, Springer 2025, FreqAI)
        
        Selection criteria:
        1. Liquidity first (volume > $10M/day)
        2. Regime diversity (gainers + losers + ranging)
        3. Volatility coverage (high + low ATR)
        4. Fixed anchors (BTC, ETH, BNB)
        
        Returns: List of 50 symbols optimized for ML training
        """
        try:
            print("\n" + "="*70)
            print("🔬 v11.5b CORE: 365-DAY SURVIVORSHIP-BIAS ELIMINATION")
            print("="*70)
            
            # Fetch all tickers
            response = NetworkHelper.get(f"{self.base_url}/ticker/24hr", timeout=20, adaptive_timeout=True)
            if not response:
                logging.warning("Failed to fetch tickers for research selection")
                return self._get_fallback_top_movers(limit)
            
            try:
                tickers = response.json()
            except (ValueError, json.JSONDecodeError):
                logging.warning("Invalid JSON in research selection")
                return self._get_fallback_top_movers(limit)
            
            # Filter: USDT perpetuals only
            usdt_pairs = [t for t in tickers if t.get('symbol', '').endswith('USDT')]
            
            print(f"📊 Total USDT pairs: {len(usdt_pairs)}")
            
            # ============================================
            # STEP 1: ELIGIBILITY FILTER
            # ============================================
            eligible = []
            for t in usdt_pairs:
                try:
                    symbol = t['symbol']
                    volume_24h = float(t.get('quoteVolume', 0))
                    
                    # Minimum volume: $10M/day (research threshold)
                    if volume_24h < 10_000_000:
                        continue
                    
                    # Skip stablecoins and wrapped assets
                    skip_keywords = ['BUSD', 'USDC', 'TUSD', 'DAI', 'WBTC', 'STETH', 'FDUSD']
                    if any(kw in symbol for kw in skip_keywords):
                        continue
                    
                    eligible.append({
                        'symbol': symbol,
                        'volume_24h': volume_24h,
                        'price_change_pct': float(t.get('priceChangePercent', 0)),
                        'price': float(t.get('lastPrice', 0))
                    })
                except (ValueError, KeyError, TypeError) as e:
                    logging.debug(f"Top mover parse error: {e}")
                    continue
            
            print(f"✅ Eligible (volume >$10M): {len(eligible)} pairs")
            
            if len(eligible) < 20:
                print("⚠️  Not enough eligible pairs, using fallback")
                return self._get_fallback_top_movers(limit)
            
            # ============================================
            # STEP 2: FETCH 365-DAY DATA FOR SCORING (v10.4)
            # ============================================
            print(f"📈 Extracting 365-day macro-regimes (Accelerated Async)...")
            
            scored_coins = []
            target_coins = eligible[:300]  # Expanded to allow for 100 gainer/100 loser pool
            completed = [0]
            
            import concurrent.futures
            
            def score_coin(coin):
                try:
                    # Get 365 days of daily candles for strict Out-Of-Sample regime detection
                    klines = self.get_klines(coin['symbol'], '1d', limit=365)
                    if not klines or len(klines) < 180: # Need at least half a year
                        return None
                    
                    closes = [float(k[4]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    # 365-Day Macro Momentum
                    pct_change_365d = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] > 0 else 0
                    
                    atr = Indicators.atr(highs, lows, closes)
                    atr_pct = (atr / closes[-1]) * 100 if closes[-1] > 0 else 0
                    
                    adx = Indicators.adx(highs, lows, closes)
                    
                    # Volume consistency
                    volume_std = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 999
                    volume_consistency = 1 / (1 + volume_std)
                    
                    # SCORING FORMULA
                    score = (
                        np.log10(coin['volume_24h']) * 0.25 + 
                        abs(pct_change_365d) * 0.25 +
                        atr_pct * 0.20 + 
                        adx * 0.15 + 
                        volume_consistency * 100 * 0.15
                    )
                    
                    return {
                        'symbol': coin['symbol'],
                        'score': score,
                        'volume_24h': coin['volume_24h'],
                        'pct_change_365d': pct_change_365d,
                        'atr_pct': atr_pct,
                        'adx': adx,
                        'price': coin['price']
                    }
                except Exception:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(score_coin, c): c for c in target_coins}
                for future in concurrent.futures.as_completed(futures):
                    completed[0] += 1
                    pct = int(completed[0] / len(target_coins) * 100)
                    print(f"\r   ⏳ Fetching Global 365-Day History: {pct}% ({completed[0]}/{len(target_coins)})   ", end='', flush=True)
                    res = future.result()
                    if res:
                        scored_coins.append(res)
            
            print() # Clear line completely
            
            if not scored_coins:
                return self._get_fallback_top_movers(limit)
            
            # ============================================
            # STEP 3: FORCED STRUCTURAL REGIME DIVERSITY (v10.4)
            # ============================================
            print(f"\n📋 Structuring Top Gainers + Top Losers (Survivorship Bias Elimination)")
            
            selections = []
            used_symbols = set()
            
            # Slot E: Fixed anchors (always include to anchor reality)
            anchors = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            for anchor in anchors:
                anchor_data = next((c for c in scored_coins if c['symbol'] == anchor), None)
                if anchor_data:
                    selections.append(anchor_data)
                    used_symbols.add(anchor)
            
            target_half = int(limit / 2) # E.g., 100 Gainers, 100 Losers
            
            # Phase 1: The Top Gainers (To map Bull Dynamics)
            gainers = [c for c in scored_coins if c['symbol'] not in used_symbols]
            gainers.sort(key=lambda x: x['pct_change_365d'], reverse=True)
            slot_gainers = gainers[:target_half]
            selections.extend(slot_gainers)
            used_symbols.update(c['symbol'] for c in slot_gainers)
            print(f"   ✅ Mapped {len(slot_gainers)} Historical Gainers (Bull Regimes)")
            
            # Phase 2: The Top Losers (To map Bear/Rug-pull Dynamics)
            losers = [c for c in scored_coins if c['symbol'] not in used_symbols]
            losers.sort(key=lambda x: x['pct_change_365d']) # Lowest first
            slot_losers = losers[:target_half] # Grab whatever room is left
            selections.extend(slot_losers)
            used_symbols.update(c['symbol'] for c in slot_losers)
            print(f"   ✅ Mapped {len(slot_losers)} Historical Bleeders (Bear/Crab Regimes)")
            
            # Fallback if limit isn't reached (e.g., extremely low eligible coins)
            remaining_needed = limit - len(selections)
            if remaining_needed > 0:
                remaining = [c for c in scored_coins if c['symbol'] not in used_symbols]
                remaining.sort(key=lambda x: x['score'], reverse=True)
                slot_remaining = remaining[:remaining_needed]
                selections.extend(slot_remaining)
                print(f"   ✅ Filled {len(slot_remaining)} remaining slots with high-volume anchors")
            
            # Final selection trim
            final_symbols = [c['symbol'] for c in selections[:limit]]
            
            print(f"\n✅ v11.5b SELECTED {len(final_symbols)} MACRO COINS:")
            print(f"   Extreme Gainers (>100%): {len([c for c in selections if c['pct_change_365d'] > 100])}")
            print(f"   Extreme Bleeders (<-40%): {len([c for c in selections if c['pct_change_365d'] < -40])}")
            print(f"   Avg Volume: ${np.mean([c['volume_24h'] for c in selections])/1e6:.1f}M")
            print("="*70 + "\n")
            
            self._log_request(success=True)
            return final_symbols
            
        except Exception as e:
            logging.error(f"Research selection failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_top_movers(limit)
    
    def categorize_coins_for_specialists(self, symbols):
        """
        🧬 CATEGORIZE COINS FOR MoE PANTHEON
        Routes coins via QuantaSelector for perfectly mathematically isolated regime sets.
        """
        if not symbols:
            return {}
            
        try:
            from QUANTA_selector import QuantaSelector
            selector = QuantaSelector(cache=self.cache if hasattr(self, 'cache') else None)
            return selector.categorize_coins_for_specialists(symbols)
        except Exception as e:
            logging.error(f"Failed to categorize for specialists: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_cache_stats(self):
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {'cache_enabled': False}
    
    def resample_1m_candles(self, klines_1m, target_interval):
        """
        🔥 LOCAL RESAMPLING - Convert 1m candles to any timeframe
        
        This saves API calls! Fetch 1m once, resample locally to 5m, 15m, 1h, etc.
        
        Args:
            klines_1m: List of 1m candles from Binance
            target_interval: '5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d'
        
        Returns:
            List of resampled candles in same Binance format
        """
        if not klines_1m or len(klines_1m) == 0:
            return []
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(klines_1m, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                        'taker_buy_base', 'taker_buy_quote']:
                df[col] = df[col].astype(float)
            df['trades'] = df['trades'].astype(int)
            
            # Set index to timestamp
            df.set_index('open_time', inplace=True)
            
            # Map interval to pandas offset
            interval_map = {
                '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1h', '2h': '2h', '4h': '4h',
                '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1D'
            }
            
            if target_interval not in interval_map:
                logging.warning(f"Unsupported interval: {target_interval}")
                return []
            
            freq = interval_map[target_interval]
            
            # Resample with OHLCV aggregation rules
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'quote_volume': 'sum',
                'trades': 'sum',
                'taker_buy_base': 'sum',
                'taker_buy_quote': 'sum'
            }).dropna()
            
            # Convert back to Binance format
            result = []
            for idx, row in resampled.iterrows():
                # Calculate close_time (end of interval)
                interval_minutes = {
                    '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120,
                    '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1440
                }
                minutes = interval_minutes.get(target_interval, 1)
                close_time = int(idx.timestamp() * 1000) + (minutes * 60 * 1000) - 1
                
                candle = [
                    int(idx.timestamp() * 1000),  # open_time
                    str(row['open']),
                    str(row['high']),
                    str(row['low']),
                    str(row['close']),
                    str(row['volume']),
                    close_time,
                    str(row['quote_volume']),
                    int(row['trades']),
                    str(row['taker_buy_base']),
                    str(row['taker_buy_quote']),
                    '0'
                ]
                result.append(candle)
            
            return result
            
        except Exception as e:
            logging.error(f"Resampling error: {e}")
            return []
    
    def get_multiframe_data(self, symbol, intervals=None, days=365):
        """
        🔥 EFFICIENT MULTI-TIMEFRAME FETCHER
        
        Fetches 1m data once, resamples locally to all needed timeframes.
        MUCH faster than fetching each timeframe separately!
        
        Args:
            symbol: Trading pair
            intervals: List of intervals (default: ['5m', '15m', '30m', '1h', '4h', '1d'])
            days: Number of days to fetch
        
        Returns:
            Dict of {interval: klines_list}
        """
        if intervals is None:
            intervals = ['5m', '15m', '30m', '1h', '4h', '1d']
        
        result = {}
        
        # Fetch 5m data (the base)
        klines_1m = self.get_historical_klines(symbol, '5m', days=days)
        
        if not klines_1m or len(klines_1m) == 0:
            return result
        
        # 5m is already fetched
        if '5m' in intervals:
            result['5m'] = klines_1m
        
        # Resample to other intervals
        for interval in intervals:
            if interval == '5m':
                continue
            
            resampled = self.resample_1m_candles(klines_1m, interval)
            if resampled:
                result[interval] = resampled
        
        return result
    
    def warmup_cache(self, symbols=None, intervals=None):
        """Warm up cache with historical data"""
        if not self.cache:
            print("⚠️ Cache not enabled")
            return
        
        if symbols is None:
            # Use basic pairs instead of get_top_movers to avoid startup hang
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                      'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT',
                      'DOTUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
                      'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'APTUSDT']
        
        if intervals is None:
            intervals = ['5m', '15m', '1h', '4h']
        
        print(f"🔥 Warming up cache for {len(symbols)} symbols...")
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    print(f"  📥 {symbol} {interval}...")
                    data = self.get_klines(symbol, interval, 1000)
                    if data and len(data) > 100:
                        print(f"    ✅ {len(data)} candles cached")
                    time.sleep(0.3)  # Rate limit
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    continue
        
        print("✅ Cache warmup complete!")
    
    def warmup_cache_research(self, symbols, days=365):
        """
        🔥 RESEARCH-BACKED CACHE WARMUP
        Fetches 180 days of 5m/1m candles for ML training
        
        Args:
            symbols: List of symbols to cache
            days: Number of days to fetch (default 180)
        """
        if not self.cache:
            print("⚠️ Cache not enabled - cannot warmup")
            return False
        
        print("\n" + "="*70)
        print(f"🔥 CACHE WARMUP - {days} DAYS OF 5M DATA (PATIENT MODE)")
        print("="*70)
        print(f"📊 Coins: {len(symbols)}")
        print(f"📈 Candles per coin: ~{days * 288:,} (5m interval)")
        print(f"💾 Total candles: ~{len(symbols) * days * 288:,}")
        print(f"⏱️  Estimated time: {len(symbols) * 0.5:.0f}-{len(symbols) * 1.5:.0f} minutes (patient mode - retries enabled)")
        print(f"☕ Grab a coffee - this is thorough but only runs once!")
        print("="*70 + "\n")
        
        import threading
        from queue import Queue
        
        start_time = time.time()
        
        # Thread-safe counters
        successful = 0
        failed = 0
        counter_lock = threading.Lock()
        
        def worker(q):
            nonlocal successful, failed
            while not q.empty():
                symbol = q.get()
                try:
                    print(f"📥 {symbol}...", end=" ", flush=True)
                    klines = self.get_historical_klines(symbol, '5m', days=days)
                    
                    with counter_lock:
                        if klines and len(klines) > 1000:
                            print(f"✅ {symbol}: {len(klines):,} candles")
                            successful += 1
                        else:
                            print(f"⚠️  {symbol}: Only {len(klines) if klines else 0} candles")
                            failed += 1
                except Exception as e:
                    print(f"❌ Error {symbol}: {e}")
                    with counter_lock:
                        failed += 1
                finally:
                    q.task_done()

        # Create queue
        work_queue = Queue()
        for sym in symbols:
            work_queue.put(sym)
            
        # Start 5 workers
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker, args=(work_queue,), daemon=True)
            t.start()
            threads.append(t)
            
        # Wait for all tasks to complete
        work_queue.join()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ CACHE WARMUP COMPLETE")
        print("="*70)
        print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"✅ Success: {successful}/{len(symbols)} coins")
        print(f"❌ Failed: {failed}/{len(symbols)} coins")
        if self.cache:
            print(f"📊 Cache hit rate: {self.cache.get_stats().get('hit_rate', 0):.1f}%")
        print("="*70 + "\n")
        
        return successful > 0

    # ── Phase-E: exchange-side stop-market orders ──────────────────────────
    def _signed_request(self, method: str, path: str, params: dict) -> dict:
        """Sign and send a Binance Futures private REST request."""
        import hashlib, hmac, urllib.parse, requests as _req, time as _time
        api_key    = getattr(self.cfg, "api_key",    "") or ""
        api_secret = getattr(self.cfg, "api_secret", "") or ""
        if not api_key or not api_secret:
            return {"error": "no_credentials"}
        params["timestamp"] = int(_time.time() * 1000)
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url   = f"{self.base_url}/{path}?{query}&signature={sig}"
        headers = {"X-MBX-APIKEY": api_key}
        try:
            if method == "POST":
                r = _req.post(url, headers=headers, timeout=5)
            elif method == "DELETE":
                r = _req.delete(url, headers=headers, timeout=5)
            else:
                r = _req.get(url, headers=headers, timeout=5)
            return r.json()
        except Exception as exc:
            logging.warning(f"[stop-market] {method} {path} failed: {exc}")
            return {"error": str(exc)}

    def place_stop_market(self, symbol: str, side: str, qty: float,
                          stop_price: float, reduce_only: bool = True) -> dict:
        """Place a STOP_MARKET order on Binance Futures."""
        params = {
            "symbol":        symbol,
            "side":          side,          # "SELL" for long, "BUY" for short
            "type":          "STOP_MARKET",
            "stopPrice":     f"{stop_price:.8f}",
            "quantity":      f"{qty:.8f}",
            "reduceOnly":    "true" if reduce_only else "false",
            "timeInForce":   "GTE_GTC",
        }
        result = self._signed_request("POST", "order", params)
        order_id = str(result.get("orderId", "")) if isinstance(result, dict) else ""
        logging.info(f"[stop-market] placed {symbol} stop@{stop_price:.6f}  orderId={order_id}")
        return {"order_id": order_id, "raw": result}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order on Binance Futures."""
        if not order_id:
            return {"error": "no_order_id"}
        params = {"symbol": symbol, "orderId": order_id}
        result = self._signed_request("DELETE", "order", params)
        logging.info(f"[stop-market] cancelled {symbol} orderId={order_id}")
        return result

    def modify_stop_order(self, symbol: str, order_id: str, new_stop_price: float) -> dict:
        """Cancel-and-replace stop order (Binance Futures doesn't support in-place modification)."""
        # Note: qty will be re-placed by the caller
        if order_id:
            self.cancel_order(symbol, order_id)
        return {"status": "cancelled_for_replace"}


class SmartExecutor:
    """C3. VWAP/TWAP Smart Execution Engine
    Splits large orders to minimize slippage on illiquid pairs.
    """
    def __init__(self, exchange, bot=None):
        self.exchange = exchange
        self.bot = bot

    def execute_twap(self, symbol, side, total_notional, current_price, steps=5, duration_mins=2.0,
                     confidence=0, atr_percent=0, specialist=None, exit_profile=None, timeout_bars=None):
        """Execute a large order using Time-Weighted Average Price (TWAP).
        
        Args:
            symbol: Pair to trade (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            total_notional: Total USD value to trade
            current_price: Estimated execution price
            steps: Number of slices to split the order into
            duration_mins: Total time to spread the executions across
            confidence: Confidence score of the signal
            atr_percent: ATR as percentage of price for barrier calculation
        """
        MIN_CHUNK = 15.0
        
        if total_notional < (MIN_CHUNK * steps):
            print(f"⚡ {symbol} TWAP canceled - order size ${total_notional:.2f} is too small. Using Market Order.")
            if self.bot:
                self.bot._execute_market_order(symbol, side, total_notional, current_price, confidence, atr_percent,
                                               specialist=specialist, exit_profile=exit_profile,
                                               timeout_bars=timeout_bars)
            return True
            
        chunk_notional = total_notional / steps
        delay_between_steps = (duration_mins * 60.0) / steps
        
        print(f"\n🔄 VWAP/TWAP Execution Started: {symbol} {side}")
        print(f"   Target: ${total_notional:.2f}")
        print(f"   Slices: {steps} chunks of ${chunk_notional:.2f}")
        print(f"   Pacing: 1 slice every {delay_between_steps:.1f}s")
        
        def _twap_worker():
            executed_notional = 0.0
            for i in range(steps):
                try:
                    price_response = NetworkHelper.get(f"{self.exchange.base_url}/ticker/price", params={'symbol': symbol}, timeout=5)
                    step_price = float(price_response.json()['price']) if price_response else current_price
                except Exception as e:
                    logging.debug(f"TWAP price fetch failed: {e}")
                    step_price = current_price
                    
                print(f"   [{i+1}/{steps}] Executing {side} ${chunk_notional:.2f} {symbol} @ ~{step_price}")
                
                if self.bot:
                    self.bot._execute_market_order(symbol, side, chunk_notional, step_price, confidence, atr_percent,
                                                   specialist=specialist, exit_profile=exit_profile,
                                                   timeout_bars=timeout_bars)
                
                executed_notional += chunk_notional
                
                if i < steps - 1:
                    time.sleep(delay_between_steps)
                    
            print(f"✅ TWAP Execution Complete: {symbol} {side} (Total: )")
            
        twap_thread = threading.Thread(target=_twap_worker, name=f"TWAP_{symbol}_{side}")
        twap_thread.daemon = True
        twap_thread.start()
        
        return True

