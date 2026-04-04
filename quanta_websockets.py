import asyncio
import json
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set
import numpy as np

# Use the WS constants defined in QUANTA_network
from QUANTA_network import (
    WS_BASE, MAX_STREAMS_PER_CONNECTION, CANDLE_DEPTH_1M, 
    CANDLE_DEPTH_REST, SWEEP_MIN_INTERVAL, SWEEP_DEFAULT_INTERVAL, 
    SWEEP_MAX_INTERVAL, DELTA_SKIP_THRESHOLD
)

class CandleStore:
    """Thread-safe in-memory OHLCV store with local resampling."""

    def __init__(self):
        self._store: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=CANDLE_DEPTH_REST))
        )
        self._lock = threading.Lock()
        self._bootstrapped: Set[str] = set()

    def update(self, symbol: str, interval: str, kline: list) -> None:
        with self._lock:
            buf = self._store[symbol][interval]
            if buf and buf[-1][0] == kline[0]:
                buf[-1] = kline
            else:
                buf.append(kline)

    def seed(self, symbol: str, interval: str, klines: list) -> None:
        with self._lock:
            buf = self._store[symbol][interval]
            buf.clear()
            for k in klines:
                buf.append(k)
            self._bootstrapped.add(f"{symbol}_{interval}")

    def get(self, symbol: str, interval: str) -> list:
        with self._lock:
            return list(self._store[symbol][interval])

    def is_ready(self, symbol: str, interval: str, min_candles: int = 50) -> bool:
        with self._lock:
            return len(self._store[symbol][interval]) >= min_candles

    def symbols(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def last_price(self, symbol: str) -> float:
        with self._lock:
            buf = self._store.get(symbol, {}).get('5m')
            if buf and len(buf) > 0:
                return float(buf[-1][4])
            return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BINANCE WEBSOCKET FEED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BinanceWSFeed:
    """Single WebSocket connection subscribing to klines. Auto-reconnects."""

    def __init__(self, candle_store: CandleStore, on_candle_close: Callable[[str, str], None]):
        self.store = candle_store
        self.on_close = on_candle_close
        self._symbols: List[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connections_ready = threading.Event()

    def subscribe(self, symbols: List[str]) -> None:
        self._symbols = list(symbols)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="WS-Feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._supervisor())

    async def _supervisor(self) -> None:
        tfs = ['5m', '15m', '1h', '4h', '1d', '1w']
        streams = [(s.lower(), tf) for s in self._symbols for tf in tfs]
        chunks = [streams[i:i + MAX_STREAMS_PER_CONNECTION]
                  for i in range(0, len(streams), MAX_STREAMS_PER_CONNECTION)]
        print(f"⚡ WS: {len(self._symbols)} symbols × 7 TFs "
              f"= {len(streams)} multiplexed streams across {len(chunks)} connection(s)")
        
        tasks = []
        for idx, chunk in enumerate(chunks):
            task = asyncio.create_task(self._connection_loop(chunk, idx))
            tasks.append(task)
            # Stagger connections to avoid "5 connections per second" IP ban on shared proxies
            await asyncio.sleep(1.5)
            
        self._connections_ready.set()
        await asyncio.gather(*tasks)

    async def _connection_loop(self, stream_chunk: list, conn_id: int) -> None:
        stream_str = "/".join(f"{sym}@kline_{tf}" for sym, tf in stream_chunk)
        url = WS_BASE + stream_str
        while not self._stop.is_set():
            try:
                await self._connect(url, conn_id)
            except Exception as exc:
                logging.warning(f"WS conn-{conn_id} error: {exc} — reconnecting in 2s")
                await asyncio.sleep(2)

    async def _connect(self, url: str, conn_id: int) -> None:
        import ssl
        import aiohttp
        proxy_url = None
        
        # Check ProxyManager for dynamically set proxies
        try:
            from quanta_proxy import ProxyManager
            proxy_url = ProxyManager.get_proxy()
        except ImportError:
            pass
            
        if proxy_url and proxy_url.startswith("socks://"):
            proxy_url = "socks5://" + proxy_url[8:]
            
        import ssl
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        
        ws_kwargs = dict(heartbeat=20, timeout=aiohttp.ClientWSTimeout(ws_close=3, ws_receive=10),
                         max_msg_size=2**20, ssl=ssl_ctx)
                         
        session_kwargs = {}
        if proxy_url and proxy_url.startswith("socks"):
            try:
                from aiohttp_socks import ProxyConnector
            except ImportError:
                raise RuntimeError("pip install aiohttp aiohttp-socks")
            connector = ProxyConnector.from_url(proxy_url, ssl=ssl_ctx, rdns=True)
            session_kwargs = {"connector": connector}
            ws_kwargs.pop("ssl", None)
        elif proxy_url and proxy_url.startswith("http"):
            ws_kwargs["proxy"] = proxy_url
            # Use the unverified SSL context for the TCP connector too
            # Some proxies require the SSL handshake to use the context
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            session_kwargs = {"connector": connector}
        else:
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            session_kwargs = {"connector": connector}
            
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.ws_connect(url, **ws_kwargs) as ws:
                via = "SOCKS" if proxy_url and "socks" in proxy_url else (
                    "HTTP proxy" if proxy_url else "direct")
                # Removed verbosity: print(f"✅ WS conn-{conn_id} connected ({via})")
                async for msg in ws:
                    if self._stop.is_set():
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            self._handle(msg.data)
                        except Exception as e:
                            logging.debug(f"WS parse error: {e}")
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    def _handle(self, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", msg)
        if data.get("e") != "kline":
            return
        k = data["k"]
        symbol: str = k["s"]
        tf: str = k["i"]
        kline = [k["t"], k["o"], k["h"], k["l"], k["c"], k["v"],
                 k["T"], k["q"], int(k["n"]), k["V"], k["Q"], "0"]
        self.store.update(symbol, tf, kline)
        if k["x"]:
            try:
                self.on_close(symbol, tf)
            except Exception as e:
                logging.debug(f"on_candle_close error ({symbol}/{tf}): {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOOTSTRAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ws_bootstrap(bnc, candle_store: CandleStore, symbols: List[str], max_workers: int = 4) -> None:
    tfs = ['5m', '15m', '1h', '4h', '1d', '1w']
    pairs = []
    for s in symbols:
        for tf in tfs:
            depth = CANDLE_DEPTH_1M if tf == '5m' else CANDLE_DEPTH_REST
            pairs.append((s, tf, depth))
    total = len(pairs)
    done = [0]
    lock = threading.Lock()
    print(f"\n📥 Bootstrap: fetching {total} history snapshots "
          f"({len(symbols)} symbols × 7 TFs)...")

    def fetch_one(symbol, tf, depth):
        try:
            import random
            time.sleep(random.uniform(0.1, 0.5))
            klines = bnc.get_klines(symbol, tf, limit=depth)
            if klines and len(klines) >= 50:
                candle_store.seed(symbol, tf, klines)
        except Exception as e:
            logging.debug(f"Bootstrap {symbol}/{tf}: {e}")
        finally:
            with lock:
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == total:
                    pct = done[0] / total * 100
                    print(f"   {done[0]}/{total} ({pct:.0f}%)", end="\r", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_one, s, tf, d) for s, tf, d in pairs]
        for f in as_completed(futures):
            pass
    ready_1m = sum(1 for s in symbols if candle_store.is_ready(s, '5m', 50))
    print(f"\n✅ Bootstrap complete: {ready_1m}/{len(symbols)} symbols with 5m data")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WS EVENT PRODUCER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WSEventProducer:
    """High-throughput prediction producer: event-driven + staggered sweep."""

    def __init__(self, candle_store: CandleStore, mtf, ml, data_queue, stop_event, cfg, is_training_event=None):
        self.store = candle_store
        self.mtf = mtf
        self.ml = ml
        self.data_queue = data_queue
        self.stop = stop_event
        self.is_training = is_training_event
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="WS-Analyze")
        self._in_flight: Set[str] = set()
        self._in_flight_lock = threading.Lock()
        self._last_analysis_price: Dict[str, float] = {}
        self._iteration = 0
        self._ws = BinanceWSFeed(candle_store=self.store, on_candle_close=self._on_candle_close)
        self._sweep_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        symbols = self.store.symbols() or []
        if not symbols:
            logging.warning("WSEventProducer: no symbols — bootstrap first")
            return

        # Subscribe WS feed to the same symbols that were seeded during bootstrap.
        # This keeps BinanceWSFeed._symbols in sync with CandleStore and ensures
        # the WS supervisor logs a non-zero symbol count.
        self._ws.subscribe(symbols)
        logging.info(
            "WSEventProducer: subscribing WS to %d symbols (e.g. %s)",
            len(symbols),
            ", ".join(symbols[:5]) if symbols else "none",
        )

        # _ws.start() initiates connections independently.
        self._ws.start()
        self._sweep_thread = threading.Thread(target=self._sweep_loop, name="WS-Sweep", daemon=True)
        self._sweep_thread.start()
        # Suppressed verbose logging: print(f"⚡ WSEventProducer v3.0 (Lightning) started")

    def _on_candle_close(self, symbol: str, tf: str) -> None:
        if self.stop.is_set():
            return
        self._submit_analysis(symbol, source="ws")

    def _sweep_loop(self) -> None:
        time.sleep(10)
        while not self.stop.is_set():
            try:
                # PAUSE during ML training
                if self.is_training and self.is_training.is_set():
                    time.sleep(5)
                    continue
                    
                symbols = self.store.symbols()
                if not symbols:
                    time.sleep(5)
                    continue
                qsize = self.data_queue.qsize()
                q_capacity = self.cfg.queue_size
                q_ratio = qsize / q_capacity if q_capacity > 0 else 0
                if q_ratio < 0.2:
                    sweep_interval = SWEEP_MIN_INTERVAL
                elif q_ratio < 0.6:
                    sweep_interval = SWEEP_DEFAULT_INTERVAL
                else:
                    sweep_interval = SWEEP_MAX_INTERVAL
                per_symbol_delay = sweep_interval / max(1, len(symbols))
                for symbol in symbols:
                    if self.stop.is_set():
                        break
                    current_price = self.store.last_price(symbol)
                    last_price = self._last_analysis_price.get(symbol, 0)
                    if last_price > 0 and current_price > 0:
                        delta = abs(current_price - last_price) / last_price
                        if delta < DELTA_SKIP_THRESHOLD:
                            time.sleep(per_symbol_delay)
                            continue
                    self._submit_analysis(symbol, source="sweep")
                    time.sleep(per_symbol_delay)
            except Exception as e:
                logging.debug(f"Sweep error: {e}")
                time.sleep(5)

    def _submit_analysis(self, symbol: str, source: str = "unknown") -> None:
        if self.stop.is_set() or (self.is_training and self.is_training.is_set()):
            return
        with self._in_flight_lock:
            if symbol in self._in_flight:
                return
            self._in_flight.add(symbol)
            self._iteration += 1
            iteration = self._iteration
        queue_threshold = int(self.cfg.queue_size * 0.80)
        if self.data_queue.qsize() > queue_threshold:
            with self._in_flight_lock:
                self._in_flight.discard(symbol)
            return
        self._pool.submit(self._analyze_and_queue, symbol, iteration)

    def _analyze_and_queue(self, symbol: str, iteration: int) -> None:
        try:
            fetch_start = time.time()
            tf_analysis = self.mtf.analyze(symbol)
            fetch_time = time.time() - fetch_start
            if not tf_analysis:
                return
            current_price = self.store.last_price(symbol)
            if current_price > 0:
                self._last_analysis_price[symbol] = current_price
            pre_features = None
            if self.ml is not None and self.ml.is_trained:
                try:
                    pre_features = self.ml._extract_features(tf_analysis, symbol=symbol)
                except Exception:
                    pre_features = None
            item = {"symbol": symbol, "tf_analysis": tf_analysis,
                    "worker_id": 0, "iteration": iteration}
            if pre_features is not None:
                item["pre_features"] = pre_features
            try:
                self.data_queue.put(item, timeout=5)
            except Exception:
                pass
        except Exception as e:
            logging.debug(f"WS analyze error ({symbol}): {e}")
        finally:
            with self._in_flight_lock:
                self._in_flight.discard(symbol)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MTF ANALYZER PATCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def patch_mtf_analyzer(mtf_instance) -> None:
    """Monkey-patches MultiTimeframeAnalyzer to read from CandleStore."""
    shared_pool = ThreadPoolExecutor(
        max_workers=len(mtf_instance.cfg.timeframes), thread_name_prefix="MTF-Shared")

    def set_candle_store(candle_store: CandleStore) -> None:
        mtf_instance._candle_store = candle_store

    def analyze_ws(symbol: str) -> dict:
        cs = getattr(mtf_instance, "_candle_store", None)
        if cs is None:
            return _original_analyze(symbol)
        try:
            probe = cs.get(symbol, '5m')
            if not probe or len(probe) < 50:
                return _original_analyze(symbol)
        except Exception:
            return _original_analyze(symbol)
        cache_key = f"{symbol}_{int(time.time() // mtf_instance.cache_duration)}"
        if cache_key in mtf_instance.analysis_cache:
            return mtf_instance.analysis_cache[cache_key]
        results = {}

        def analyze_tf_from_store(tf: str):
            try:
                klines = cs.get(symbol, tf)
                if not klines or len(klines) < 50:
                    return None
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                from quanta_features import Indicators
                rsi = Indicators.rsi(closes, mtf_instance.cfg.rsi_period)
                macd_line, signal_line, histogram = Indicators.macd(closes)
                bb_upper, bb_middle, bb_lower = Indicators.bollinger(closes)
                atr = Indicators.atr(highs, lows, closes)
                k_stoch, d_stoch = Indicators.stochastic(highs, lows, closes)
                adx = Indicators.adx(highs, lows, closes)
                price = closes[-1]
                ma_short = np.mean(closes[-mtf_instance.cfg.ma_short:])
                ma_long = np.mean(closes[-mtf_instance.cfg.ma_long:])
                trend_score = 0
                if rsi > 70: trend_score += 20
                elif rsi < 30: trend_score -= 20
                elif rsi > 50: trend_score += 10
                else: trend_score -= 10
                if macd_line > signal_line and histogram > 0: trend_score += 20
                elif macd_line < signal_line and histogram < 0: trend_score -= 20
                if price > ma_short > ma_long: trend_score += 20
                elif price < ma_short < ma_long: trend_score -= 20
                bb_position = ((price - bb_lower) / (bb_upper - bb_lower)
                               if (bb_upper - bb_lower) > 0 else 0.5)
                if bb_position > 0.8: trend_score += 10
                elif bb_position < 0.2: trend_score -= 10
                if k_stoch > 80 and d_stoch > 80: trend_score += 10
                elif k_stoch < 20 and d_stoch < 20: trend_score -= 10
                strength = min(100, abs(trend_score) * (adx / 25))
                if trend_score > 30: trend = 'BULLISH'
                elif trend_score < -30: trend = 'BEARISH'
                else: trend = 'NEUTRAL'
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
                if len(closes) >= 100:
                    step = 5
                    windows = range(0, len(closes) - 14, step)
                    atr_vals = np.array([Indicators.atr(highs[i:i+14], lows[i:i+14], closes[i:i+14]) for i in windows])
                    atr_percentile = float(np.mean(atr_vals <= atr)) if len(atr_vals) else 0.5
                else:
                    atr_percentile = 0.5
                volatility_accel = ((atr - Indicators.atr(highs[-10:-1], lows[-10:-1], closes[-10:-1])) / atr
                                    if atr > 0 and len(closes) >= 10 else 0)
                mom_5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                mom_10 = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
                mom_20 = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
                mom_50 = (closes[-1] / closes[-51] - 1) if len(closes) >= 51 else 0
                vol_ma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
                volume_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
                ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
                ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
                mean_shift = ma_20 - ma_50
                trend_strength = abs(mean_shift) / atr if atr > 0 else 0
                returns_period = (closes[-1] / closes[0] - 1) if closes else 0
                vpin_val = Indicators.vpin(highs, lows, closes, volumes, 14)
                fd_val = Indicators.frac_diff(closes, d=0.4)
                return (tf, {
                    'trend': trend, 'strength': int(strength), 'price': price,
                    'rsi': rsi, 'macd': histogram, 'bb_position': bb_position,
                    'adx': adx, 'volume': volumes[-1], 'atr': atr, 'symbol': symbol,
                    'volatility': volatility, 'atr_percentile': atr_percentile,
                    'volatility_accel': volatility_accel,
                    'momentum_5': mom_5, 'momentum_10': mom_10,
                    'momentum_20': mom_20, 'momentum_50': mom_50,
                    'volume_ratio': volume_ratio, 'mean_shift': mean_shift,
                    'trend_strength': trend_strength, 'returns_period': returns_period,
                    'vpin': vpin_val, 'frac_diff': fd_val
                })
            except Exception as e:
                logging.error(f"MTF WS {symbol} {tf}: {e}")
                return None

        futures = {shared_pool.submit(analyze_tf_from_store, tf): tf
                   for tf in mtf_instance.cfg.timeframes}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logging.debug(f"MTF WS future error: {e}")
                continue
            if result:
                tf, analysis = result
                results[tf] = analysis
        mtf_instance.analysis_cache[cache_key] = results
        current_time_key = time.time() // mtf_instance.cache_duration
        stale = [k for k in mtf_instance.analysis_cache
                 if int(float(k.rsplit('_', 1)[-1])) < current_time_key - 2]
        for k in stale:
            del mtf_instance.analysis_cache[k]
        return results

    _original_analyze = mtf_instance.analyze
    import types
    mtf_instance.analyze = types.MethodType(
        lambda self, symbol: analyze_ws(symbol), mtf_instance)
    mtf_instance.set_candle_store = set_candle_store
    mtf_instance._shared_pool = shared_pool
    print("✅ MTF Analyzer patched: CandleStore + numpy resampling + shared pool")
