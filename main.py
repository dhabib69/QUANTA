"""
QUANTA v10 -- Main Orchestrator
Assembles all modules and launches the trading bot.
Phase 5: Concurrent Startup -- heavy module imports run in background
threads while the user inputs the proxy port.
"""
import sys
import os
import time
import threading
import requests

# Ensure the QUANTA directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows cp1252 encoding crash
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# -- Phase 5: Concurrent Module Preloading --
_preload_results = {}

def _preload_module(name, import_fn):
    t0 = time.perf_counter()
    try:
        import_fn()
        _preload_results[name] = f"OK {time.perf_counter() - t0:.1f}s"
    except Exception as e:
        _preload_results[name] = f"SKIP ({e})"

_preload_threads = [
    threading.Thread(target=_preload_module, args=("torch",    lambda: __import__("torch")),     daemon=True),
    threading.Thread(target=_preload_module, args=("catboost", lambda: __import__("catboost")),   daemon=True),
    threading.Thread(target=_preload_module, args=("sklearn",  lambda: __import__("sklearn")),    daemon=True),
    threading.Thread(target=_preload_module, args=("numba",    lambda: __import__("numba")),      daemon=True),
    threading.Thread(target=_preload_module, args=("numpy",    lambda: __import__("numpy")),      daemon=True),
    threading.Thread(target=_preload_module, args=("pandas",   lambda: __import__("pandas")),     daemon=True),
]

print("[BOOT] Preloading heavy modules in background...")
_preload_start = time.perf_counter()
for t in _preload_threads:
    t.start()

# -- Proxy Configuration (runs BEFORE heavy imports to avoid GIL deadlock) --
# NetworkHelper import is deferred until AFTER proxy port is collected,
# because QUANTA_network imports numpy/pandas at module level which can
# deadlock with the background preload threads on Windows (GIL contention).
try:
    if sys.stdin and sys.stdin.isatty():
        _proxy_port = input("[BOOT] Enter proxy port (blank = direct): ").strip()
    else:
        _proxy_port = ""
except (EOFError, KeyboardInterrupt):
    _proxy_port = ""

# Now wait for preloads to finish BEFORE importing QUANTA_network
for t in _preload_threads:
    t.join(timeout=60)

_preload_elapsed = time.perf_counter() - _preload_start
print(f"[BOOT] Preload done ({_preload_elapsed:.1f}s wall)")
for name, status in sorted(_preload_results.items()):
    print(f"  {name}: {status}")
print()

# Safe to import now — numpy/pandas already loaded by preload threads
from QUANTA_network import NetworkHelper

if _proxy_port:
    _proxy_url = f"http://127.0.0.1:{_proxy_port}"
    from quanta_proxy import ProxyManager
    ProxyManager.set_proxy(_proxy_url)
    ProxyManager.start_watchdog()
    
    # --- AUTO PROXY ISOLATION ---
    if sys.platform == 'win32':
        try:
            import winreg, ctypes
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings", 0, winreg.KEY_WRITE)
            winreg.SetValueEx(registry_key, "ProxyEnable", 0, winreg.REG_DWORD, 0)
            winreg.CloseKey(registry_key)
            ctypes.windll.wininet.InternetSetOptionW(0, 39, 0, 0)
            ctypes.windll.wininet.InternetSetOptionW(0, 37, 0, 0)
            print("[BOOT] Proxy Isolated: System proxy disabled (Brave un-lagged). Bot is safely tunneled.")
        except Exception as e:
            print(f"[BOOT] Proxy Isolation failed: {e}")
    # ----------------------------
    
    import QUANTA_bot
    import QUANTA_network as _nh_mod
    
    os.environ["HTTP_PROXY"] = _proxy_url
    os.environ["HTTPS_PROXY"] = _proxy_url
    os.environ["http_proxy"] = _proxy_url
    os.environ["https_proxy"] = _proxy_url
    NetworkHelper.reset_session()
    print(f"[BOOT] Proxy active: {_proxy_url}")
else:
    from quanta_proxy import ProxyManager
    ProxyManager.set_proxy(None)
    print("[BOOT] Direct connection (no proxy)")

# -- Binance Weight-Aware Rate Limiter --
BINANCE_WEIGHT_LIMIT_1M = 1200
BINANCE_WEIGHT_SOFT_LIMIT = 0.90

_binance_lock = threading.Lock()
_binance_used_weight = 0
_binance_last_reset = time.time()

_original_requests_get = requests.get

def _patched_requests_get(*args, **kwargs):
    global _binance_used_weight, _binance_last_reset
    url = args[0] if args else kwargs.get("url", "")
    kwargs.setdefault("verify", False)
    
    from quanta_proxy import ProxyManager
    proxy_kwargs = ProxyManager.get_requests_kwargs()
    if proxy_kwargs and "proxies" not in kwargs:
        kwargs.update(proxy_kwargs)
        
    if "binance" in url:
        with _binance_lock:
            now = time.time()
            if now - _binance_last_reset >= 60:
                _binance_used_weight = 0
                _binance_last_reset = now
            if _binance_used_weight >= BINANCE_WEIGHT_LIMIT_1M * BINANCE_WEIGHT_SOFT_LIMIT:
                sleep_time = 60 - (now - _binance_last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                _binance_used_weight = 0
                _binance_last_reset = time.time()
    resp = _original_requests_get(*args, **kwargs)
    if "binance" in url:
        try:
            used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                _binance_used_weight = int(used)
        except Exception:
            pass
    return resp

requests.get = _patched_requests_get

# -- Launch --
if __name__ == "__main__":
    from QUANTA_bot import Bot
    Bot().run()
