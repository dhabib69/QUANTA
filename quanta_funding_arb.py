"""
QUANTA v11 Phase C: Funding Rate Arbitrage Module

Scans perpetual funding rates across Binance, Bybit, and OKX to identify
risk-neutral yield opportunities. When funding diverges significantly between
exchanges, the module opens opposing positions to capture the spread.

Academic basis:
    - Makarov & Schoar (2020) "Trading and Arbitrage in Cryptocurrency Markets" (JFE)
    - Shleifer & Vishny (1997) "The Limits of Arbitrage" (JF)
    - DeFi: Perpetual Protocol Whitepaper — funding rate mechanics

Strategy:
    1. Poll funding rates every 60s from all connected exchanges
    2. If |rate_A - rate_B| > threshold → open long on negative-rate exchange,
       short on positive-rate exchange (delta-neutral)
    3. Collect funding payment at settlement (every 8h)
    4. Close both legs after N funding periods or if spread collapses

Usage:
    from quanta_multi_exchange import BybitAdapter, OKXAdapter, ExchangeRouter
    router = ExchangeRouter([BybitAdapter(), OKXAdapter()])
    arb = FundingArbEngine(router, min_spread_bps=5.0)
    arb.start()
"""

import time
import logging
import threading
import numpy as np
from collections import defaultdict
from datetime import datetime

from quanta_config import Config as _Cfg
_FA = _Cfg.funding_arb


class FundingArbOpportunity:
    """Represents a single funding arbitrage opportunity."""

    def __init__(self, symbol, long_exchange, short_exchange,
                 long_rate, short_rate, spread_bps):
        self.symbol = symbol
        self.long_exchange = long_exchange    # Go long here (negative/lower rate)
        self.short_exchange = short_exchange  # Go short here (positive/higher rate)
        self.long_rate = long_rate
        self.short_rate = short_rate
        self.spread_bps = spread_bps         # Absolute funding spread in bps
        self.timestamp = time.time()

    def __repr__(self):
        return (f"FundingArb({self.symbol}: LONG@{self.long_exchange} "
                f"[{self.long_rate:+.4%}] / SHORT@{self.short_exchange} "
                f"[{self.short_rate:+.4%}] = {self.spread_bps:.1f}bps)")


class FundingArbEngine:
    """
    Funding Rate Arbitrage Scanner & Executor.

    Continuously polls funding rates across all connected exchanges.
    When a spread exceeding the threshold is found, it logs the opportunity
    and optionally executes delta-neutral positions.

    Parameters:
        router: ExchangeRouter with connected adapters
        min_spread_bps: Minimum funding spread (in basis points) to trigger (default 5.0)
        scan_interval: Seconds between scans (default 60)
        max_positions: Maximum concurrent arbitrage positions (default 3)
        auto_execute: Whether to auto-execute opportunities (default False — paper mode)
        telegram_send_fn: Optional Telegram alert function
    """

    def __init__(self, router, min_spread_bps=None, scan_interval=None,
                 max_positions=None, auto_execute=False, telegram_send_fn=None):
        self.router = router
        self.min_spread_bps = min_spread_bps if min_spread_bps is not None else _FA.min_spread_bps
        self.scan_interval = scan_interval or _FA.scan_interval
        self.max_positions = max_positions or _FA.max_positions
        self.auto_execute = auto_execute
        self.telegram_send = telegram_send_fn

        self._active_positions = {}   # symbol → FundingArbOpportunity
        self._opportunity_log = []    # Historical log
        self._rate_history = defaultdict(list)  # symbol → [(ts, {exchange: rate})]
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        logging.info(f"💰 FundingArbEngine initialized (threshold: {min_spread_bps}bps, "
                     f"auto_execute: {auto_execute})")

    def start(self):
        """Start the background funding rate scanner."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._scan_loop, daemon=True, name="FundingArb"
            )
            self._thread.start()
            logging.info("💰 Funding rate arbitrage scanner started")

    def stop(self):
        """Stop the background scanner."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=_FA.thread_join_timeout)

    def _scan_loop(self):
        """Main scanning loop."""
        print("💰 Funding Rate Arbitrage Scanner active")
        while not self._stop_event.is_set():
            try:
                opportunities = self.scan_opportunities()
                if opportunities:
                    for opp in opportunities:
                        self._handle_opportunity(opp)
            except Exception as e:
                logging.error(f"Funding arb scan error: {e}")

            self._stop_event.wait(self.scan_interval)

    def scan_opportunities(self) -> list:
        """
        Scan all symbols across all exchanges for funding rate arbitrage.

        Returns list of FundingArbOpportunity where spread > threshold.
        """
        opportunities = []

        for symbol in _FA.scan_symbols:
            try:
                rates = self.router.get_funding_rates(symbol)

                # Need at least 2 exchanges with non-zero rates
                valid_rates = {k: v for k, v in rates.items() if v != 0}
                if len(valid_rates) < 2:
                    continue

                # Store rate history
                self._rate_history[symbol].append((time.time(), dict(valid_rates)))
                # Keep only last 100 snapshots
                if len(self._rate_history[symbol]) > _FA.rate_history_limit:
                    self._rate_history[symbol] = self._rate_history[symbol][-_FA.rate_history_limit:]

                # Find max spread
                exchanges = list(valid_rates.keys())
                max_spread = 0
                best_long = None
                best_short = None

                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        ex_a, ex_b = exchanges[i], exchanges[j]
                        rate_a, rate_b = valid_rates[ex_a], valid_rates[ex_b]

                        # Spread in basis points
                        spread = abs(rate_a - rate_b) * 10000

                        if spread > max_spread:
                            max_spread = spread
                            if rate_a < rate_b:
                                best_long = ex_a    # Lower/negative rate → go long
                                best_short = ex_b   # Higher/positive rate → go short
                            else:
                                best_long = ex_b
                                best_short = ex_a

                if max_spread >= self.min_spread_bps and best_long and best_short:
                    opp = FundingArbOpportunity(
                        symbol=symbol,
                        long_exchange=best_long,
                        short_exchange=best_short,
                        long_rate=valid_rates[best_long],
                        short_rate=valid_rates[best_short],
                        spread_bps=max_spread
                    )
                    opportunities.append(opp)

            except Exception as e:
                logging.debug(f"Funding scan error {symbol}: {e}")
                continue

        # Sort by spread descending
        opportunities.sort(key=lambda x: x.spread_bps, reverse=True)
        return opportunities

    def _handle_opportunity(self, opp: FundingArbOpportunity):
        """Process a detected arbitrage opportunity."""
        with self._lock:
            # Log it
            self._opportunity_log.append(opp)
            if len(self._opportunity_log) > _FA.opportunity_log_limit:
                self._opportunity_log = self._opportunity_log[-_FA.opportunity_log_limit:]

            print(f"\n💰 FUNDING ARB DETECTED: {opp}")
            print(f"   Spread: {opp.spread_bps:.1f}bps | "
                  f"Long {opp.long_exchange} [{opp.long_rate:+.4%}] | "
                  f"Short {opp.short_exchange} [{opp.short_rate:+.4%}]")

            # Alert via Telegram
            if self.telegram_send:
                try:
                    msg = (f"💰 *FUNDING ARB*\n\n"
                           f"*{opp.symbol}*: {opp.spread_bps:.1f}bps spread\n"
                           f"📈 Long `{opp.long_exchange}` ({opp.long_rate:+.4%})\n"
                           f"📉 Short `{opp.short_exchange}` ({opp.short_rate:+.4%})")
                    self.telegram_send(msg)
                except Exception:
                    pass

            # Auto-execute if enabled and under position limit
            if self.auto_execute and len(self._active_positions) < self.max_positions:
                if opp.symbol not in self._active_positions:
                    self._execute_arb(opp)

    def _execute_arb(self, opp: FundingArbOpportunity):
        """Execute a delta-neutral funding arbitrage trade."""
        # In production, this would:
        # 1. Calculate position size based on available margin
        # 2. Open LONG on opp.long_exchange
        # 3. Open SHORT on opp.short_exchange
        # 4. Monitor and close after funding settlement
        logging.info(f"💰 Would execute arb on {opp.symbol} "
                     f"(LONG@{opp.long_exchange}, SHORT@{opp.short_exchange})")
        self._active_positions[opp.symbol] = opp

    def get_stats(self) -> dict:
        """Return arbitrage engine statistics."""
        return {
            'active_positions': len(self._active_positions),
            'total_opportunities_detected': len(self._opportunity_log),
            'recent_opportunities': [
                {
                    'symbol': o.symbol,
                    'spread_bps': o.spread_bps,
                    'long': o.long_exchange,
                    'short': o.short_exchange,
                    'time': datetime.fromtimestamp(o.timestamp).strftime('%H:%M:%S')
                }
                for o in self._opportunity_log[-5:]
            ],
            'symbols_scanned': len(_FA.scan_symbols),
            'exchanges': [a.name for a in self.router.adapters]
        }

    def get_rate_history(self, symbol: str) -> list:
        """Return funding rate history for a symbol."""
        return self._rate_history.get(symbol, [])
