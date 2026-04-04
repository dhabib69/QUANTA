"""
QUANTA v11 Phase C: Multi-Exchange Connectivity

Abstracts exchange-specific REST/WebSocket APIs behind a unified interface.
Supports Binance (primary), Bybit, and OKX for redundant execution routing.

Academic basis:
    - Makarov & Schoar (2020) "Trading and Arbitrage in Cryptocurrency Markets" (JFE)
    - Hautsch et al. (2018) "Building Trust Takes Time: Limits to Arbitrage" (JFE)

Architecture:
    ExchangeAdapter (ABC)
    ├── BinanceAdapter   — already operational (quanta_exchange.py)
    ├── BybitAdapter      — v3 REST API
    └── OKXAdapter        — v5 REST API

Usage:
    router = ExchangeRouter([BinanceAdapter(...), BybitAdapter(...)])
    best = router.get_best_price('BTCUSDT', 'BUY')
    router.execute('BTCUSDT', 'BUY', qty=0.01, exchange='bybit')
"""

import os
import time
import hmac
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime

from QUANTA_network import NetworkHelper
from quanta_config import Config as _Cfg
_MEX = _Cfg.multi_exchange


# ═══════════════════════════════════════════════════════════
# ABSTRACT EXCHANGE ADAPTER
# ═══════════════════════════════════════════════════════════

class ExchangeAdapter(ABC):
    """Unified interface all exchange adapters must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """Returns {'bid': float, 'ask': float, 'last': float, 'volume': float}"""
        ...

    @abstractmethod
    def get_funding_rate(self, symbol: str) -> float:
        """Returns the current funding rate for a perpetual contract."""
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'MARKET') -> dict:
        """Place an order. Returns {'order_id': str, 'status': str, 'filled_price': float}"""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> dict:
        """Returns {'size': float, 'entry': float, 'pnl': float, 'side': str}"""
        ...


# ═══════════════════════════════════════════════════════════
# BYBIT ADAPTER (v3 Unified Trading API)
# ═══════════════════════════════════════════════════════════

class BybitAdapter(ExchangeAdapter):
    """
    Bybit Unified Trading v3 integration.
    Docs: https://bybit-exchange.github.io/docs/v3/
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.api_key = os.environ.get("BYBIT_API_KEY", "")
        self.api_secret = os.environ.get("BYBIT_API_SECRET", "")
        self._enabled = bool(self.api_key and self.api_secret)
        if self._enabled:
            logging.info("✅ Bybit adapter initialized")
        else:
            logging.info("⚠️ Bybit adapter: no API keys set (read-only mode)")

    @property
    def name(self):
        return "Bybit"

    def _sign(self, params: dict) -> dict:
        """Generate HMAC-SHA256 signature for Bybit v3."""
        timestamp = str(int(time.time() * 1000))
        recv_window = _MEX.recv_window
        param_str = timestamp + self.api_key + recv_window

        # Sort and concatenate params
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if sorted_params:
            param_str += sorted_params

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

    def get_ticker(self, symbol: str) -> dict:
        try:
            # Bybit uses BTCUSDT natively (same as Binance)
            resp = NetworkHelper.get(
                f"{self.BASE_URL}/v5/market/tickers",
                params={"category": "linear", "symbol": symbol},
                timeout=_MEX.api_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                tick = data.get("result", {}).get("list", [{}])[0]
                return {
                    "bid": float(tick.get("bid1Price", 0)),
                    "ask": float(tick.get("ask1Price", 0)),
                    "last": float(tick.get("lastPrice", 0)),
                    "volume": float(tick.get("turnover24h", 0))
                }
        except Exception as e:
            logging.debug(f"Bybit ticker error: {e}")
        return {"bid": 0, "ask": 0, "last": 0, "volume": 0}

    def get_funding_rate(self, symbol: str) -> float:
        try:
            resp = NetworkHelper.get(
                f"{self.BASE_URL}/v5/market/funding/history",
                params={"category": "linear", "symbol": symbol, "limit": "1"},
                timeout=_MEX.api_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                records = data.get("result", {}).get("list", [])
                if records:
                    return float(records[0].get("fundingRate", 0))
        except Exception as e:
            logging.debug(f"Bybit funding rate error: {e}")
        return 0.0

    def place_order(self, symbol, side, qty, order_type='MARKET'):
        if not self._enabled:
            return {"order_id": "DISABLED", "status": "NO_API_KEY", "filled_price": 0}

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side.upper() == "BUY" else "Sell",
            "orderType": "Market" if order_type == "MARKET" else "Limit",
            "qty": str(qty),
            "timeInForce": "GoodTillCancel"
        }

        try:
            headers = self._sign(params)
            resp = NetworkHelper.post(
                f"{self.BASE_URL}/v5/order/create",
                json=params, headers=headers, timeout=_MEX.order_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                return {
                    "order_id": result.get("orderId", ""),
                    "status": "FILLED",
                    "filled_price": float(result.get("avgPrice", 0))
                }
        except Exception as e:
            logging.error(f"Bybit order error: {e}")
        return {"order_id": "ERROR", "status": "FAILED", "filled_price": 0}

    def get_position(self, symbol):
        if not self._enabled:
            return {"size": 0, "entry": 0, "pnl": 0, "side": "NONE"}

        try:
            params = {"category": "linear", "symbol": symbol}
            headers = self._sign(params)
            resp = NetworkHelper.get(
                f"{self.BASE_URL}/v5/position/list",
                params=params, headers=headers, timeout=_MEX.api_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                positions = data.get("result", {}).get("list", [])
                if positions:
                    pos = positions[0]
                    return {
                        "size": float(pos.get("size", 0)),
                        "entry": float(pos.get("avgPrice", 0)),
                        "pnl": float(pos.get("unrealisedPnl", 0)),
                        "side": pos.get("side", "NONE")
                    }
        except Exception as e:
            logging.debug(f"Bybit position error: {e}")
        return {"size": 0, "entry": 0, "pnl": 0, "side": "NONE"}


# ═══════════════════════════════════════════════════════════
# OKX ADAPTER (v5 REST API)
# ═══════════════════════════════════════════════════════════

class OKXAdapter(ExchangeAdapter):
    """
    OKX v5 integration.
    Docs: https://www.okx.com/docs-v5/
    """

    BASE_URL = "https://www.okx.com"

    def __init__(self):
        self.api_key = os.environ.get("OKX_API_KEY", "")
        self.api_secret = os.environ.get("OKX_API_SECRET", "")
        self.passphrase = os.environ.get("OKX_PASSPHRASE", "")
        self._enabled = bool(self.api_key and self.api_secret and self.passphrase)
        if self._enabled:
            logging.info("✅ OKX adapter initialized")
        else:
            logging.info("⚠️ OKX adapter: no API keys set (read-only mode)")

    @property
    def name(self):
        return "OKX"

    def _okx_symbol(self, symbol: str) -> str:
        """Convert BTCUSDT → BTC-USDT-SWAP for OKX perpetuals."""
        base = symbol.replace("USDT", "")
        return f"{base}-USDT-SWAP"

    def get_ticker(self, symbol: str) -> dict:
        try:
            okx_sym = self._okx_symbol(symbol)
            resp = NetworkHelper.get(
                f"{self.BASE_URL}/api/v5/market/ticker",
                params={"instId": okx_sym},
                timeout=_MEX.api_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                ticks = data.get("data", [{}])
                if ticks:
                    t = ticks[0]
                    return {
                        "bid": float(t.get("bidPx", 0)),
                        "ask": float(t.get("askPx", 0)),
                        "last": float(t.get("last", 0)),
                        "volume": float(t.get("volCcy24h", 0))
                    }
        except Exception as e:
            logging.debug(f"OKX ticker error: {e}")
        return {"bid": 0, "ask": 0, "last": 0, "volume": 0}

    def get_funding_rate(self, symbol: str) -> float:
        try:
            okx_sym = self._okx_symbol(symbol)
            resp = NetworkHelper.get(
                f"{self.BASE_URL}/api/v5/public/funding-rate",
                params={"instId": okx_sym},
                timeout=_MEX.api_timeout
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                records = data.get("data", [])
                if records:
                    return float(records[0].get("fundingRate", 0))
        except Exception as e:
            logging.debug(f"OKX funding rate error: {e}")
        return 0.0

    def place_order(self, symbol, side, qty, order_type='MARKET'):
        if not self._enabled:
            return {"order_id": "DISABLED", "status": "NO_API_KEY", "filled_price": 0}
        # Full signed OKX order placement would go here
        logging.warning("OKX order placement not fully implemented — set API keys first")
        return {"order_id": "NOT_IMPL", "status": "PENDING", "filled_price": 0}

    def get_position(self, symbol):
        if not self._enabled:
            return {"size": 0, "entry": 0, "pnl": 0, "side": "NONE"}
        return {"size": 0, "entry": 0, "pnl": 0, "side": "NONE"}


# ═══════════════════════════════════════════════════════════
# EXCHANGE ROUTER (Smart Order Routing)
# ═══════════════════════════════════════════════════════════

class ExchangeRouter:
    """
    Routes orders to the exchange with the best price.

    Academic basis:
        - Makarov & Schoar (2020): Cross-exchange price dispersion creates
          arbitrageable spreads of 5-10 bps in crypto.

    Usage:
        router = ExchangeRouter([BybitAdapter(), OKXAdapter()])
        best = router.get_best_price('BTCUSDT', 'BUY')
        router.execute('BTCUSDT', 'BUY', qty=0.01)
    """

    def __init__(self, adapters: list = None):
        self.adapters = adapters or []
        self._price_cache = {}       # symbol → {exchange_name: ticker}
        self._cache_ttl = _MEX.price_cache_ttl
        self._cache_ts = {}
        logging.info(f"🔀 ExchangeRouter: {len(self.adapters)} exchanges connected")

    def add_adapter(self, adapter: ExchangeAdapter):
        self.adapters.append(adapter)

    def get_best_price(self, symbol: str, side: str) -> dict:
        """
        Fetch prices from all exchanges and return the best one.

        Returns:
            {'exchange': str, 'price': float, 'all_prices': dict}
        """
        prices = {}
        for adapter in self.adapters:
            try:
                ticker = adapter.get_ticker(symbol)
                if side.upper() == 'BUY':
                    prices[adapter.name] = ticker['ask'] if ticker['ask'] > 0 else ticker['last']
                else:
                    prices[adapter.name] = ticker['bid'] if ticker['bid'] > 0 else ticker['last']
            except Exception as e:
                logging.debug(f"Price fetch error {adapter.name}: {e}")

        if not prices:
            return {'exchange': 'NONE', 'price': 0, 'all_prices': {}}

        if side.upper() == 'BUY':
            best_exchange = min(prices, key=prices.get)  # Lowest ask for buys
        else:
            best_exchange = max(prices, key=prices.get)  # Highest bid for sells

        return {
            'exchange': best_exchange,
            'price': prices[best_exchange],
            'all_prices': prices
        }

    def get_funding_rates(self, symbol: str) -> dict:
        """Fetch funding rates from all exchanges."""
        rates = {}
        for adapter in self.adapters:
            try:
                rates[adapter.name] = adapter.get_funding_rate(symbol)
            except Exception:
                rates[adapter.name] = 0.0
        return rates

    def execute(self, symbol: str, side: str, qty: float,
                exchange: str = None, order_type: str = 'MARKET') -> dict:
        """
        Execute order on specified exchange, or auto-route to best price.
        """
        if exchange:
            adapter = next((a for a in self.adapters if a.name.lower() == exchange.lower()), None)
            if adapter:
                return adapter.place_order(symbol, side, qty, order_type)
            logging.error(f"Exchange '{exchange}' not found in router")
            return {"order_id": "NOT_FOUND", "status": "FAILED", "filled_price": 0}

        # Auto-route: find best price
        best = self.get_best_price(symbol, side)
        target = next((a for a in self.adapters if a.name == best['exchange']), None)
        if target:
            logging.info(f"🔀 Auto-routing {side} {symbol} to {best['exchange']} @ {best['price']}")
            return target.place_order(symbol, side, qty, order_type)

        return {"order_id": "NO_ROUTE", "status": "FAILED", "filled_price": 0}
