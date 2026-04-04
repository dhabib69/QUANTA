"""
🌐 QUANTA Proxy Manager — Centralized Proxy Routing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provides a single source of truth for the bot's proxy configuration.
Replaces the old dynamic `_proxy_url` globals and env variables.
"""

import threading
import requests
import logging


class ProxyManager:
    """
    Global proxy state manager.
    Stores the active proxy URL and provides formatted proxy kwargs for HTTP libraries.
    """
    _proxy_url = None
    _lock = threading.Lock()
    _health_thread = None
    _stop_event = threading.Event()
    _failure_count = 0

    @classmethod
    def set_proxy(cls, url: str):
        """Set the global proxy URL (e.g., 'http://127.0.0.1:2080')."""
        with cls._lock:
            if url and not url.startswith("http"):
                cls._proxy_url = f"http://{url}"
            else:
                cls._proxy_url = url

    @classmethod
    def get_proxy(cls) -> str:
        """Get the global proxy URL."""
        with cls._lock:
            return cls._proxy_url

    @classmethod
    def get_requests_kwargs(cls) -> dict:
        """
        Get kwargs to pass to requests.get() or requests.post().
        Returns:
            {'proxies': {'http': proxy, 'https': proxy}, 'verify': False} if proxy set
            {} if no proxy
        """
        with cls._lock:
            if cls._proxy_url:
                return {
                    'proxies': {'http': cls._proxy_url, 'https': cls._proxy_url},
                    'verify': False
                }
            return {}

    @classmethod
    def apply_to_session(cls, session: requests.Session):
        """Apply the global proxy to an existing requests Session."""
        with cls._lock:
            if cls._proxy_url:
                session.proxies = {'http': cls._proxy_url, 'https': cls._proxy_url}
                session.verify = False

    @classmethod
    def start_watchdog(cls):
        """
        Starts a background thread to ping Binance every 60s.
        If the proxy fails 3 times, logs an error (auto-rotation disabled).
        """
        if cls._health_thread and cls._health_thread.is_alive():
            return
            
        cls._stop_event.clear()
        cls._health_thread = threading.Thread(target=cls._watchdog_loop, daemon=True, name="ProxyWatchdog")
        cls._health_thread.start()
        logging.info("🛡️ ProxyManager Watchdog started.")

    @classmethod
    def stop_watchdog(cls):
        """Stops the background watchdog thread."""
        cls._stop_event.set()
        if cls._health_thread:
            cls._health_thread.join(timeout=2)

    @classmethod
    def _watchdog_loop(cls):
        """Background loop to check proxy health."""
        while not cls._stop_event.is_set():
            # Wait 60 seconds between checks
            if cls._stop_event.wait(timeout=60.0):
                break
                
            current_proxy = cls.get_proxy()
            if not current_proxy:
                continue
                
            try:
                # Ping Binance through the active proxy
                resp = requests.get(
                    "https://fapi.binance.com/fapi/v1/ping",
                    proxies={"http": current_proxy, "https": current_proxy},
                    timeout=30.0,
                    verify=False
                )
                if resp.status_code == 200:
                    cls._failure_count = 0
                else:
                    cls._failure_count += 1
            except Exception as e:
                cls._failure_count += 1
                logging.debug(f"ProxyManager watchdog ping failed: {e}")

            if cls._failure_count >= 3:
                logging.error(f"⚠️ ProxyWatchdog: Active proxy {current_proxy} failed {cls._failure_count} times! Please check Psiphon 3 connection.")
                cls._failure_count = 0
