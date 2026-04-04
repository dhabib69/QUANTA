"""
QUANTA Risk Manager — Institutional-Grade Capital Protection (v11.4)

Provides three layers of protection:
1. Daily drawdown circuit breaker (auto-pause when daily loss exceeds limit)
2. Position concentration limits (max positions, max per-coin exposure)
3. Consecutive loss streak detection (reduces size after losing streaks)

References:
    - Bouchaud & Potters (2003) "Theory of Financial Risk" — drawdown-based risk control
    - Marcos López de Prado (2018) AFML Ch.10 — bet sizing and position limits
    - Tharp (2006) "Trade Your Way to Financial Freedom" — R-multiple risk management

All magic numbers are in quanta_config.py → RiskManagerConfig.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from quanta_config import Config as _Cfg

_RM = _Cfg.risk_manager if hasattr(_Cfg, 'risk_manager') else None


@dataclass
class RiskEvent:
    """Immutable record of a risk event for audit trail."""
    timestamp: float
    event_type: str          # 'circuit_breaker', 'position_limit', 'streak_throttle', 'exposure_limit'
    details: str
    action_taken: str        # 'blocked_trade', 'reduced_size', 'paused_trading', 'resumed_trading'
    balance_at_event: float


class RiskManager:
    """
    Real-time risk manager that wraps PaperTrading to enforce hard limits.

    Usage:
        risk_mgr = RiskManager(paper_trading_instance)

        # Before every trade:
        allowed, reason = risk_mgr.pre_trade_check(symbol, notional, direction)
        if not allowed:
            print(f"Trade blocked: {reason}")
            return

        # After every trade close:
        risk_mgr.on_trade_closed(symbol, pnl, balance)

        # Periodic check (call from tick loop):
        risk_mgr.heartbeat(current_balance)
    """

    def __init__(self, initial_balance=10000.0, flat_all_callback=None):
        # Config (with fallback defaults if RiskManagerConfig not yet in config)
        self.max_daily_drawdown_pct    = getattr(_RM, 'max_daily_drawdown_pct', 3.0)
        self.max_open_positions        = getattr(_RM, 'max_open_positions', 50)
        self.max_single_coin_pct       = getattr(_RM, 'max_single_coin_pct', 15.0)
        self.max_total_exposure_pct    = getattr(_RM, 'max_total_exposure_pct', 5000.0)
        self.consecutive_loss_throttle = getattr(_RM, 'consecutive_loss_throttle', 3)
        self.throttle_size_factor      = getattr(_RM, 'throttle_size_factor', 0.5)
        self.cooldown_after_breaker_min = getattr(_RM, 'cooldown_after_breaker_min', 60)
        self.max_correlation_exposure   = getattr(_RM, 'max_correlation_exposure', 3)
        self.max_risk_per_trade_pct     = getattr(_RM, 'max_risk_per_trade_pct', 2.0)

        # Optional callback: called when circuit breaker fires to close all positions.
        # Set this to paper_trading_instance.flat_all at startup.
        self._flat_all_callback = flat_all_callback

        # State
        self.initial_balance = initial_balance
        self._day_start_balance = initial_balance
        self._day_start_date = datetime.utcnow().date()
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._trading_paused = False
        self._pause_until = None
        self._trade_log = deque(maxlen=1000)  # Rolling trade history
        self._risk_events = deque(maxlen=500)
        self._lock = threading.RLock()  # RLock: allow re-entrant locking (get_size_multiplier called inside on_trade_closed)

        # Tracking
        self._open_positions = {}  # symbol -> notional
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_losses = 0

        logging.info(f"🛡️ RiskManager initialized: "
                     f"max_dd={self.max_daily_drawdown_pct}%, "
                     f"max_pos={self.max_open_positions}, "
                     f"max_coin={self.max_single_coin_pct}%")

    # ─────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────

    def pre_trade_check(self, symbol, notional, direction, current_balance, positions_dict=None):
        """
        Gate check BEFORE opening a trade. Returns (allowed: bool, reason: str).

        Args:
            symbol:          Trading pair (e.g. 'BTCUSDT')
            notional:        USD value of the proposed trade
            direction:       'BULLISH' or 'BEARISH'
            current_balance: Current portfolio balance
            positions_dict:  Current open positions {symbol: {entry, size, direction, ...}}
        """
        with self._lock:
            self._rotate_day_if_needed(current_balance)

            # 1. Circuit breaker — is trading paused?
            if self._trading_paused:
                if self._pause_until and datetime.utcnow() >= self._pause_until:
                    self._trading_paused = False
                    self._pause_until = None
                    self._log_event('circuit_breaker', 'Cooldown expired', 'resumed_trading', current_balance)
                    print("🟢 RISK: Trading resumed after cooldown")
                else:
                    remaining = (self._pause_until - datetime.utcnow()).total_seconds() / 60 if self._pause_until else 0
                    return False, f"Circuit breaker active ({remaining:.0f}min remaining)"

            # 2. Daily drawdown check
            dd_pct = abs(self._daily_pnl / max(self._day_start_balance, 1)) * 100
            if dd_pct >= self.max_daily_drawdown_pct and self._daily_pnl < 0:
                self._trigger_circuit_breaker(current_balance, f"Daily drawdown {dd_pct:.1f}% >= {self.max_daily_drawdown_pct}%")
                return False, f"Daily drawdown limit hit ({dd_pct:.1f}%)"

            # 3. Max open positions check
            n_positions = len(positions_dict) if positions_dict else len(self._open_positions)
            if n_positions >= self.max_open_positions:
                self._log_event('position_limit', f'{n_positions} positions open', 'blocked_trade', current_balance)
                return False, f"Max {self.max_open_positions} positions reached ({n_positions} open)"

            # 4. Single coin exposure check
            coin_exposure_pct = (notional / max(current_balance, 1)) * 100
            if coin_exposure_pct > self.max_single_coin_pct:
                self._log_event('exposure_limit', f'{symbol} {coin_exposure_pct:.1f}%', 'blocked_trade', current_balance)
                return False, f"Single coin exposure {coin_exposure_pct:.1f}% > {self.max_single_coin_pct}%"

            # 5. Total portfolio exposure check
            total_exposure = sum(self._open_positions.values()) + notional
            total_pct = (total_exposure / max(current_balance, 1)) * 100
            if total_pct > self.max_total_exposure_pct:
                self._log_event('exposure_limit', f'Total {total_pct:.1f}%', 'blocked_trade', current_balance)
                return False, f"Total exposure {total_pct:.1f}% > {self.max_total_exposure_pct}%"

            # 6. Same-direction concentration check
            if positions_dict:
                same_dir = sum(1 for p in positions_dict.values() if p.get('direction') == direction)
                if same_dir >= self.max_correlation_exposure:
                    return False, f"Max {self.max_correlation_exposure} same-direction positions ({same_dir} {direction})"

            # 7. Per-trade max risk cap
            # Notional represents the capital at risk for this trade.
            trade_risk_pct = (notional / max(current_balance, 1)) * 100
            if trade_risk_pct > self.max_risk_per_trade_pct:
                self._log_event('per_trade_limit',
                                f'{symbol} risk {trade_risk_pct:.1f}% > cap {self.max_risk_per_trade_pct}%',
                                'blocked_trade', current_balance)
                return False, (f"Per-trade risk {trade_risk_pct:.1f}% exceeds cap "
                               f"{self.max_risk_per_trade_pct}% — reduce position size")

            return True, "OK"

    def get_size_multiplier(self):
        """
        Returns a float [0.25, 1.0] to scale position size based on streak state.

        Called by PaperTrading.open_position() to reduce size after consecutive losses.
        """
        with self._lock:
            if self._consecutive_losses >= self.consecutive_loss_throttle:
                # Exponential decay: 0.5^(n - threshold + 1) but floored at 0.25
                excess = self._consecutive_losses - self.consecutive_loss_throttle + 1
                mult = max(0.25, self.throttle_size_factor ** excess)
                return mult
            return 1.0

    def on_trade_opened(self, symbol, notional):
        """Record that a position was opened."""
        with self._lock:
            self._open_positions[symbol] = notional
            self._daily_trades += 1

    def on_trade_closed(self, symbol, pnl, current_balance):
        """Record that a position was closed. Updates daily P&L and streak."""
        with self._lock:
            self._open_positions.pop(symbol, None)
            self._daily_pnl += pnl

            if pnl >= 0:
                self._consecutive_losses = 0
                self._daily_wins += 1
            else:
                self._consecutive_losses += 1
                self._daily_losses += 1

                if self._consecutive_losses >= self.consecutive_loss_throttle:
                    self._log_event(
                        'streak_throttle',
                        f'{self._consecutive_losses} consecutive losses',
                        'reduced_size',
                        current_balance
                    )
                    print(f"⚠️ RISK: {self._consecutive_losses} consecutive losses — "
                          f"size reduced to {self.get_size_multiplier():.0%}")

            self._trade_log.append({
                'symbol': symbol,
                'pnl': pnl,
                'balance': current_balance,
                'time': time.time(),
                'streak': self._consecutive_losses
            })

            # Check circuit breaker after every losing trade
            dd_pct = abs(self._daily_pnl / max(self._day_start_balance, 1)) * 100
            if dd_pct >= self.max_daily_drawdown_pct and self._daily_pnl < 0:
                self._trigger_circuit_breaker(current_balance, f"Daily drawdown {dd_pct:.1f}%")

    def heartbeat(self, current_balance):
        """Periodic check from the main loop. Rotates day, checks drawdown."""
        with self._lock:
            self._rotate_day_if_needed(current_balance)

    def get_status(self):
        """Return current risk state for dashboard display."""
        with self._lock:
            dd_pct = abs(self._daily_pnl / max(self._day_start_balance, 1)) * 100 if self._daily_pnl < 0 else 0
            return {
                'trading_paused': self._trading_paused,
                'pause_until': self._pause_until.isoformat() if self._pause_until else None,
                'daily_pnl': self._daily_pnl,
                'daily_drawdown_pct': dd_pct,
                'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
                'consecutive_losses': self._consecutive_losses,
                'size_multiplier': self.get_size_multiplier(),
                'open_positions': len(self._open_positions),
                'max_positions': self.max_open_positions,
                'daily_trades': self._daily_trades,
                'daily_wins': self._daily_wins,
                'daily_losses': self._daily_losses,
                'recent_events': list(self._risk_events)[-5:],
            }

    # ─────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────

    def _trigger_circuit_breaker(self, balance, reason):
        """Pause all trading and close all open positions immediately."""
        self._trading_paused = True
        self._pause_until = datetime.utcnow() + timedelta(minutes=self.cooldown_after_breaker_min)
        self._log_event('circuit_breaker', reason, 'paused_trading+flat_all', balance)
        print(f"\n🔴 ══════════════════════════════════════════")
        print(f"🔴 CIRCUIT BREAKER TRIGGERED: {reason}")
        print(f"🔴 Trading paused for {self.cooldown_after_breaker_min} minutes")
        print(f"🔴 Resume at: {self._pause_until.strftime('%H:%M:%S UTC')}")
        print(f"🔴 Daily P&L: ${self._daily_pnl:+.2f}")
        print(f"🔴 ══════════════════════════════════════════\n")

        # CRITICAL: Close all open positions to prevent further loss during cooldown.
        # A paused bot that holds open positions can still lose unlimited capital.
        if self._flat_all_callback is not None:
            try:
                print("🔴 Closing all open positions (circuit breaker flat-all)...")
                self._flat_all_callback()
                print("🔴 All positions closed.")
            except Exception as e:
                logging.error(f"Circuit breaker flat_all failed: {e}", exc_info=True)

    def _rotate_day_if_needed(self, current_balance):
        """Reset daily counters at UTC midnight."""
        today = datetime.utcnow().date()
        if today != self._day_start_date:
            # Log previous day summary
            if self._daily_trades > 0:
                wr = self._daily_wins / max(self._daily_trades, 1) * 100
                logging.info(f"📊 Day {self._day_start_date}: P&L=${self._daily_pnl:+.2f}, "
                             f"{self._daily_trades} trades, {wr:.0f}% WR")

            self._day_start_balance = current_balance
            self._day_start_date = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_wins = 0
            self._daily_losses = 0
            # Don't reset consecutive_losses — that persists across days

    def _log_event(self, event_type, details, action, balance):
        """Add to audit trail."""
        evt = RiskEvent(
            timestamp=time.time(),
            event_type=event_type,
            details=details,
            action_taken=action,
            balance_at_event=balance
        )
        self._risk_events.append(evt)
        logging.warning(f"🛡️ RISK [{event_type}]: {details} → {action} (bal=${balance:.2f})")
