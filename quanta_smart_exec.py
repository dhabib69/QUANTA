"""
QUANTA v11 Phase C: TWAP/VWAP Smart Execution Engine

Splits large orders into smaller slices to minimize market impact.
Integrates with ExchangeRouter for cross-exchange execution.

Academic basis:
    - Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions" (JR)
    - Berkowitz et al. (1988) "The Total Cost of Transactions on the NYSE" (JF)
    - Kissell & Glantz (2003) "Optimal Trading Strategies" — TWAP/VWAP benchmarks

Strategies:
    1. TWAP — Time-Weighted Average Price: equal slices at fixed intervals
    2. VWAP — Volume-Weighted Average Price: slices weighted by historical volume profile
    3. Iceberg — Show only a fraction of the total order at any time

Usage:
    from quanta_multi_exchange import ExchangeRouter
    exec_engine = SmartExecutionEngine(router)
    exec_engine.twap('BTCUSDT', 'BUY', total_qty=1.0, duration_min=10, num_slices=5)
    exec_engine.vwap('ETHUSDT', 'SELL', total_qty=10.0, duration_min=30)
"""

import time
import math
import logging
import threading
import concurrent.futures
import numpy as np
from datetime import datetime
from collections import defaultdict

from quanta_config import Config as _Cfg
_SE = _Cfg.smart_exec


class ExecutionSlice:
    """A single slice of a larger order."""

    def __init__(self, slice_id, symbol, side, qty, scheduled_time):
        self.slice_id = slice_id
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.scheduled_time = scheduled_time
        self.executed = False
        self.filled_price = 0.0
        self.exchange = ""
        self.execution_time = 0.0
        self.order_result = None

    def __repr__(self):
        status = f"FILLED@{self.filled_price:.2f}" if self.executed else "PENDING"
        return f"Slice#{self.slice_id}({self.qty:.4f} {self.side} {status})"


class ExecutionPlan:
    """Tracks the full execution plan for a smart order."""

    def __init__(self, plan_id, symbol, side, total_qty, strategy, slices):
        self.plan_id = plan_id
        self.symbol = symbol
        self.side = side
        self.total_qty = total_qty
        self.strategy = strategy  # 'TWAP', 'VWAP', 'ICEBERG'
        self.slices = slices
        self.created_at = time.time()
        self.completed = False
        self.cancelled = False

    @property
    def filled_qty(self):
        return sum(s.qty for s in self.slices if s.executed)

    @property
    def avg_fill_price(self):
        filled = [(s.qty, s.filled_price) for s in self.slices if s.executed and s.filled_price > 0]
        if not filled:
            return 0.0
        total_val = sum(q * p for q, p in filled)
        total_qty = sum(q for q, _ in filled)
        return total_val / total_qty if total_qty > 0 else 0.0

    @property
    def progress(self):
        executed = sum(1 for s in self.slices if s.executed)
        return executed / len(self.slices) if self.slices else 0.0

    def get_stats(self):
        return {
            'plan_id': self.plan_id,
            'symbol': self.symbol,
            'side': self.side,
            'strategy': self.strategy,
            'total_qty': self.total_qty,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'progress': f"{self.progress:.0%}",
            'slices_total': len(self.slices),
            'slices_filled': sum(1 for s in self.slices if s.executed),
            'completed': self.completed,
            'cancelled': self.cancelled
        }


class SmartExecutionEngine:
    """
    TWAP/VWAP/Iceberg order execution engine.

    Parameters:
        router: ExchangeRouter instance for order placement
        default_exchange: Preferred exchange (None = auto-route to best price)
        telegram_send_fn: Optional function for execution alerts
    """

    def __init__(self, router, default_exchange=None, telegram_send_fn=None):
        self.router = router
        self.default_exchange = default_exchange
        self.telegram_send = telegram_send_fn

        self._active_plans = {}  # plan_id → ExecutionPlan
        self._completed_plans = []
        self._plan_counter = 0
        self._lock = threading.Lock()
        self._threads = {}  # plan_id → Thread

        logging.info("SmartExecutionEngine initialized")

    def twap(self, symbol, side, total_qty, duration_min=None,
             num_slices=None, exchange=None) -> str:
        """
        Time-Weighted Average Price execution.

        Splits total_qty into num_slices equal parts, executed at uniform
        intervals over duration_min minutes.

        Almgren & Chriss (2001): TWAP is optimal when volatility is constant
        and there is no urgency (risk-neutral execution).

        Returns: plan_id string
        """
        duration_min = duration_min or _SE.twap_default_duration_min
        num_slices = num_slices or _SE.twap_default_slices
        interval_sec = (duration_min * 60) / num_slices
        qty_per_slice = total_qty / num_slices
        now = time.time()

        slices = []
        for i in range(num_slices):
            s = ExecutionSlice(
                slice_id=i,
                symbol=symbol,
                side=side,
                qty=qty_per_slice,
                scheduled_time=now + (i * interval_sec)
            )
            slices.append(s)

        plan = self._create_plan(symbol, side, total_qty, 'TWAP', slices)

        logging.info(f"TWAP plan {plan.plan_id}: {total_qty} {symbol} {side} "
                     f"in {num_slices} slices over {duration_min}min")

        self._start_execution(plan, exchange)
        return plan.plan_id

    def vwap(self, symbol, side, total_qty, duration_min=None,
             num_slices=None, exchange=None) -> str:
        """
        Volume-Weighted Average Price execution.

        Distributes slices according to the historical intraday volume profile.
        More volume is placed during high-liquidity hours, less during low.

        Berkowitz et al. (1988): VWAP minimizes implementation shortfall
        by trading proportional to expected volume.

        Returns: plan_id string
        """
        duration_min = duration_min or _SE.vwap_default_duration_min
        num_slices = num_slices or _SE.vwap_default_slices
        now = time.time()
        interval_sec = (duration_min * 60) / num_slices

        volume_profile = np.array(_SE.hourly_volume_profile)

        # Get volume weights for each slice based on when it executes
        weights = []
        for i in range(num_slices):
            exec_time = now + (i * interval_sec)
            hour = datetime.fromtimestamp(exec_time).hour
            weights.append(volume_profile[hour])

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        slices = []
        for i in range(num_slices):
            s = ExecutionSlice(
                slice_id=i,
                symbol=symbol,
                side=side,
                qty=total_qty * weights[i],
                scheduled_time=now + (i * interval_sec)
            )
            slices.append(s)

        plan = self._create_plan(symbol, side, total_qty, 'VWAP', slices)

        logging.info(f"VWAP plan {plan.plan_id}: {total_qty} {symbol} {side} "
                     f"in {num_slices} slices over {duration_min}min")

        self._start_execution(plan, exchange)
        return plan.plan_id

    def iceberg(self, symbol, side, total_qty, visible_qty,
                interval_sec=None, exchange=None) -> str:
        """
        Iceberg order execution.

        Shows only visible_qty at a time. After each fill, waits interval_sec
        before placing the next visible_qty slice.

        Reduces information leakage — other market participants cannot see
        the true order size.

        Returns: plan_id string
        """
        interval_sec = interval_sec or _SE.iceberg_default_interval_sec
        num_slices = math.ceil(total_qty / visible_qty)
        now = time.time()

        slices = []
        remaining = total_qty
        for i in range(num_slices):
            qty = min(visible_qty, remaining)
            s = ExecutionSlice(
                slice_id=i,
                symbol=symbol,
                side=side,
                qty=qty,
                scheduled_time=now + (i * interval_sec)
            )
            slices.append(s)
            remaining -= qty

        plan = self._create_plan(symbol, side, total_qty, 'ICEBERG', slices)

        logging.info(f"ICEBERG plan {plan.plan_id}: {total_qty} {symbol} {side} "
                     f"showing {visible_qty} per slice")

        self._start_execution(plan, exchange)
        return plan.plan_id

    def cancel(self, plan_id):
        """Cancel an active execution plan."""
        with self._lock:
            if plan_id in self._active_plans:
                self._active_plans[plan_id].cancelled = True
                logging.info(f"Cancelled execution plan {plan_id}")
                return True
        return False

    def get_plan_stats(self, plan_id) -> dict:
        """Get stats for a specific plan."""
        with self._lock:
            plan = self._active_plans.get(plan_id)
            if plan:
                return plan.get_stats()
            for p in self._completed_plans:
                if p.plan_id == plan_id:
                    return p.get_stats()
        return {}

    def get_all_stats(self) -> dict:
        """Get overall execution engine statistics."""
        with self._lock:
            return {
                'active_plans': len(self._active_plans),
                'completed_plans': len(self._completed_plans),
                'active': [p.get_stats() for p in self._active_plans.values()],
                'recent_completed': [p.get_stats() for p in self._completed_plans[-5:]]
            }

    def _create_plan(self, symbol, side, total_qty, strategy, slices):
        """Create and register a new execution plan."""
        with self._lock:
            self._plan_counter += 1
            plan_id = f"{strategy}_{self._plan_counter}_{int(time.time())}"
            plan = ExecutionPlan(plan_id, symbol, side, total_qty, strategy, slices)
            self._active_plans[plan_id] = plan
            return plan

    def _start_execution(self, plan, exchange=None):
        """Start a background thread to execute the plan."""
        exchange = exchange or self.default_exchange
        t = threading.Thread(
            target=self._execute_plan,
            args=(plan, exchange),
            daemon=True,
            name=f"Exec-{plan.plan_id}"
        )
        self._threads[plan.plan_id] = t
        t.start()

    def _execute_slice_with_timeout(self, s, exchange):
        """
        Execute a single slice, enforcing a wall-clock timeout.

        Uses a ThreadPoolExecutor future so router.execute() cannot hang
        the entire plan thread indefinitely.

        Returns the result dict, or raises on timeout/error.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.router.execute,
                s.symbol, s.side, s.qty,
                exchange=exchange, order_type='MARKET'
            )
            try:
                return future.result(timeout=_SE.slice_order_timeout_sec)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Slice {s.slice_id} timed out after {_SE.slice_order_timeout_sec}s"
                )

    def _execute_plan(self, plan, exchange=None):
        """
        Execute all slices in the plan according to their schedule.

        Each slice is attempted up to slice_max_retries times with
        exponential backoff. Partial fills are detected and logged.
        A per-slice wall-clock timeout prevents hung exchange calls
        from stalling the entire execution thread.
        """
        for s in plan.slices:
            if plan.cancelled:
                logging.info(f"Plan {plan.plan_id} cancelled at slice {s.slice_id}")
                break

            # Wait until scheduled time
            wait = s.scheduled_time - time.time()
            if wait > 0:
                if plan.cancelled:
                    break
                time.sleep(min(wait, 1.0))
                while time.time() < s.scheduled_time and not plan.cancelled:
                    time.sleep(_SE.schedule_poll_sec)

            if plan.cancelled:
                break

            # Execute slice with retries
            last_exc = None
            for attempt in range(1, _SE.slice_max_retries + 1):
                try:
                    result = self._execute_slice_with_timeout(s, exchange)
                    s.executed = True
                    s.filled_price = result.get('filled_price', 0)
                    s.exchange = exchange or 'auto'
                    s.execution_time = time.time()
                    s.order_result = result

                    # Partial fill detection
                    filled_qty = result.get('filled_qty', s.qty)
                    if filled_qty < s.qty * _SE.partial_fill_threshold:
                        fill_pct = filled_qty / s.qty * 100
                        logging.warning(
                            f"Partial fill on {plan.plan_id} slice {s.slice_id}: "
                            f"got {filled_qty:.4f}/{s.qty:.4f} ({fill_pct:.1f}%) "
                            f"@ {s.filled_price:.2f}"
                        )

                    logging.info(
                        f"  {plan.strategy} slice {s.slice_id}/{len(plan.slices)} "
                        f"(attempt {attempt}): {s.qty:.4f} {s.side} @ {s.filled_price:.2f}"
                    )
                    last_exc = None
                    break  # Success — no more retries needed

                except Exception as e:
                    last_exc = e
                    if attempt < _SE.slice_max_retries:
                        backoff = _SE.slice_retry_backoff_sec * (2 ** (attempt - 1))
                        logging.warning(
                            f"Slice {s.slice_id} attempt {attempt}/{_SE.slice_max_retries} "
                            f"failed: {e} — retrying in {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                    else:
                        logging.error(
                            f"Slice {s.slice_id} FAILED after {_SE.slice_max_retries} attempts: {e}",
                            exc_info=True
                        )

        # Mark complete
        with self._lock:
            plan.completed = True
            self._active_plans.pop(plan.plan_id, None)
            self._completed_plans.append(plan)
            if len(self._completed_plans) > _SE.max_completed_plans:
                self._completed_plans = self._completed_plans[-_SE.max_completed_plans:]

        stats = plan.get_stats()
        failed_slices = sum(1 for s in plan.slices if not s.executed)
        if failed_slices:
            logging.warning(
                f"Plan {plan.plan_id} completed with {failed_slices} failed slice(s): "
                f"{stats['filled_qty']:.4f}/{stats['total_qty']:.4f} filled"
            )
        else:
            logging.info(
                f"Plan {plan.plan_id} complete: {stats['filled_qty']:.4f}/"
                f"{stats['total_qty']:.4f} filled @ avg {stats['avg_fill_price']:.2f}"
            )

        # Telegram alert on completion
        if self.telegram_send:
            try:
                self.telegram_send(
                    f"*{plan.strategy} Complete*\n"
                    f"{plan.symbol} {plan.side}\n"
                    f"Filled: {stats['filled_qty']:.4f}/{stats['total_qty']:.4f}\n"
                    f"Avg Price: {stats['avg_fill_price']:.2f}"
                    + (f"\n⚠️ {failed_slices} slice(s) failed" if failed_slices else "")
                )
            except Exception as e:
                logging.debug(f"Telegram execution alert failed: {e}")
