"""
🌐 QUANTA Network — Consolidated I/O Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Contains:
  • NetworkHelper (HTTP session pooling, circuit breaker, rate limiting)
  • FreeProxyManager (auto proxy discovery)
  • FeatherCache (blazing fast Arrow-based cache)
  • TelegramBot (command listener & notifications)
  • CandleStore (in-memory OHLCV buffer with resampling)
  • BinanceWSFeed (multiplexed WebSocket, 1m klines)
  • WSEventProducer (staggered sweep + event-driven analysis)
  • ws_bootstrap / patch_mtf_analyzer
"""

import asyncio
import os
import sys
import time
import logging
import threading
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Proxy Configuration imported from quanta_proxy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from quanta_proxy import ProxyManager


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NETWORK HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NetworkHelper:
    """Enhanced network helper with session pooling, circuit breaker, and adaptive timeouts"""
    
    _session = None
    _session_lock = threading.Lock()
    _circuit_breaker = defaultdict(lambda: {'failures': 0, 'last_fail_time': 0, 'is_open': False})
    _request_counts = defaultdict(int)
    _rate_limit_lock = threading.Lock()
    _network_errors = 0

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    cls._session.verify = False
                    cls._session.trust_env = False  # Ignore system proxies
                    ProxyManager.apply_to_session(cls._session)
                    retry_strategy = Retry(
                        total=5, backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
                        allowed_methods=["GET", "POST", "HEAD", "OPTIONS"],
                        raise_on_status=False
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=50,
                                          pool_maxsize=100, pool_block=False)
                    cls._session.mount("http://", adapter)
                    cls._session.mount("https://", adapter)
                    cls._session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'application/json, text/plain, */*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Cache-Control': 'no-cache', 'Pragma': 'no-cache'
                    })
        return cls._session

    @classmethod
    def reset_session(cls):
        with cls._session_lock:
            cls._session = None

    @classmethod
    def _check_circuit_breaker(cls, url):
        endpoint_key = url.split('?')[0]
        breaker = cls._circuit_breaker[endpoint_key]
        if breaker['is_open']:
            cooldown_period = 60  # 1 minute recovery (was 5 mins) to recover quickly from transient proxy drops
            if time.time() - breaker['last_fail_time'] > cooldown_period:
                breaker['is_open'] = False
                breaker['failures'] = 0
                logging.info(f"🔄 Circuit breaker CLOSED for {endpoint_key}")
                return True
            else:
                return False
        return True

    @classmethod
    def _record_failure(cls, url):
        endpoint_key = url.split('?')[0]
        breaker = cls._circuit_breaker[endpoint_key]
        breaker['failures'] += 1
        breaker['last_fail_time'] = time.time()
        if breaker['failures'] >= 20:
            breaker['is_open'] = True
            logging.warning(f"⚠️ Circuit breaker OPENED for {endpoint_key} (too many failures)")

    @classmethod
    def _record_success(cls, url):
        endpoint_key = url.split('?')[0]
        breaker = cls._circuit_breaker[endpoint_key]
        if breaker['failures'] > 0:
            breaker['failures'] = max(0, breaker['failures'] - 1)

    @classmethod
    def _apply_rate_limit(cls, url):
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        with cls._rate_limit_lock:
            cls._request_counts[domain] += 1
            if 'binance' in domain:
                if cls._request_counts[domain] % 20 == 0:
                    time.sleep(0.1)

    @staticmethod
    def get(url, params=None, timeout=8, max_retries=3, adaptive_timeout=True):
        if not NetworkHelper._check_circuit_breaker(url):
            logging.debug(f"🚫 Circuit breaker OPEN for {url}, skipping request")  # demoted: was WARNING, caused log flood
            return None
        NetworkHelper._apply_rate_limit(url)
        if adaptive_timeout:
            if '/ticker/24hr' in url or '/exchangeInfo' in url:
                timeout = 15  # Fail faster on heavy queries
            elif '/klines' in url:
                timeout = 8   # Fast path for klines
            else:
                timeout = 5   # Extremely aggressive fail-fast for standard queries
        session = NetworkHelper._get_session()
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = min(2 ** attempt, 10)
                    time.sleep(delay)
                
                # Explicitly pass proxy to bypass session misconfiguration
                proxy_kwargs = ProxyManager.get_requests_kwargs()
                    
                if proxy_kwargs:
                    # Session caches DNS and routing, bypass entirely!
                    # Do NOT use global requests.get as main.py patches it and it breaks!
                    response = session.request("GET", url, params=params, timeout=timeout, **proxy_kwargs)
                else:
                    response = session.get(url, params=params, timeout=timeout, verify=False)
                    
                if not response.text:
                    continue
                if response.status_code == 200:
                    if response.text.strip():
                        NetworkHelper._record_success(url)
                        return response
                    else:
                        logging.error(f"Empty response body from {url}")
                        continue
                if response.status_code == 429:
                    wait = min((2 ** attempt), 10) + (time.time() % 1)
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    wait = min((2 ** attempt), 5) + (time.time() % 1)
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    if response.status_code == 409 and 'telegram' in url.lower():
                        return None
                    # 400 = bad request (invalid symbol etc.) — NOT a server failure
                    # Do NOT trip circuit breaker for client errors
                    current_time = time.time()
                    error_key = f"{response.status_code}_{url[:50]}"
                    if not hasattr(NetworkHelper, '_last_error_log'):
                        NetworkHelper._last_error_log = {}
                    if error_key not in NetworkHelper._last_error_log or \
                       current_time - NetworkHelper._last_error_log[error_key] > 60:
                        logging.debug(f"Client error {response.status_code} from {url[:80]}... (suppressing similar for 60s)")
                        NetworkHelper._last_error_log[error_key] = current_time
                    # Only record failure for 429 (rate limit), not 400 (bad request)
                    if response.status_code == 429:
                        NetworkHelper._record_failure(url)
                    return None
                response.raise_for_status()
                NetworkHelper._record_success(url)
                return response
            except requests.exceptions.Timeout:
                NetworkHelper._record_failure(url)
                NetworkHelper._network_errors += 1
                if attempt < max_retries - 1:
                    time.sleep(min((2 ** attempt), 3) + (time.time() % 1))
            except requests.exceptions.ConnectionError:
                NetworkHelper._record_failure(url)
                NetworkHelper._network_errors += 1
                if attempt < max_retries - 1:
                    time.sleep(min((2 ** attempt), 3) + (time.time() % 1))
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP error on {url}: {e}")
                NetworkHelper._record_failure(url)
                break
            except Exception:
                NetworkHelper._record_failure(url)
                if attempt < max_retries - 1:
                    time.sleep(min((2 ** attempt), 15) + (time.time() % 1))
        NetworkHelper._record_failure(url)
        if "alternative.me" not in url:
            logging.error(f"❌ All {max_retries} attempts failed for {url}")
        return None

    @staticmethod
    def post(url, data=None, timeout=8, max_retries=3):
        session = NetworkHelper._get_session()
        for attempt in range(max_retries):
            try:
                # Explicitly pass proxy to bypass session misconfiguration
                proxy_kwargs = ProxyManager.get_requests_kwargs()
                    
                if proxy_kwargs:
                    # Do NOT use global requests.get as main.py patches it and it breaks!
                    response = session.request("POST", url, data=data, timeout=timeout, **proxy_kwargs)
                else:
                    response = session.post(url, data=data, timeout=timeout, verify=False)
                    
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) + (time.time() % 1))
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) + (time.time() % 1))
        return None


# (FreeProxyManager moved to quanta_proxy.py)





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBSOCKET FEED CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WS_TIMEFRAME = '5m'
RESAMPLE_TFS = {'15m': 3}
REST_TIMEFRAMES = ['1h', '4h', '6h', '12h', '1d']
STATIC_REFRESH_INTERVAL = {'1h': 1800, '4h': 7200, '6h': 10800, '12h': 21600, '1d': 4 * 3600}
ALL_TIMEFRAMES = ['5m', '15m', '1h', '4h', '6h', '12h', '1d']
WS_BASE = "wss://fstream.binance.com/stream?streams="
MAX_STREAMS_PER_CONNECTION = 180
CANDLE_DEPTH_1M = 3000
CANDLE_DEPTH_REST = 200
DELTA_SKIP_THRESHOLD = 0.0005
SWEEP_MIN_INTERVAL = 5
SWEEP_DEFAULT_INTERVAL = 10
SWEEP_MAX_INTERVAL = 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CANDLE STORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

