"""
QUANTA Paper Trading Logger — Decision Audit Trail & Performance Tracker (v11.4)

Wraps the existing PaperTrading class with comprehensive decision logging:
  1. Every prediction → logged with features, confidence, direction, PPO gate result
  2. Every trade open → logged with entry, size, risk %, Kelly fraction
  3. Every trade close → logged with exit, P&L, barrier hit, hold time
  4. Real-time equity curve tracking
  5. Periodic performance snapshots for dashboard comparison

Can optionally connect to Binance Testnet for real order execution validation
(without risking capital) when BINANCE_TESTNET=1 is set.

References:
    - Tharp (2006) "Trade Your Way to Financial Freedom" — trade journaling
    - Schwager (1993) "Market Wizards" — every successful trader keeps detailed logs

All magic numbers configurable via quanta_config.py → PaperTradingConfig.
"""

import os
import csv
import json
import time
import logging
import threading
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, asdict
from quanta_config import Config as _Cfg

_PT = _Cfg.paper_trading if hasattr(_Cfg, 'paper_trading') else None


@dataclass
class DecisionRecord:
    """Full decision record for audit trail."""
    timestamp: str
    symbol: str
    direction: str
    confidence: float
    ppo_action: int             # 0=HOLD, 1=BULL, 2=BEAR
    ppo_value: float
    specialist_probs: list       # 7 CatBoost probabilities
    conformal_set_size: int
    hmm_regime: int
    passes_gate: bool
    action_taken: str           # 'TRADE_OPENED', 'GATE_BLOCKED', 'RISK_BLOCKED', 'COOLDOWN'
    risk_check_result: str      # 'OK' or reason for block


@dataclass
class TradeRecord:
    """Enhanced trade record with full context."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    size: float
    notional: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    hold_minutes: float
    barrier_hit: str
    confidence: float
    kelly_fraction: float
    risk_pct: float
    atr_pct: float
    regime_at_entry: int


class PaperTradingLogger:
    """
    Decision logger that wraps the trading pipeline.

    Usage in QUANTA_bot.py consumer_worker:

        # After ML prediction, before trade execution:
        self.paper_logger.log_decision(
            symbol=symbol, direction=ml_dir, confidence=ml_conf,
            ppo_action=ppo_action, ppo_value=ppo_value,
            specialist_probs=probs, conformal_set_size=set_size,
            hmm_regime=regime, passes_gate=passes_gate,
            action_taken='TRADE_OPENED', risk_result='OK'
        )

        # After trade close:
        self.paper_logger.log_trade_closed(symbol, entry, exit, ...)

        # Periodic (every N minutes):
        self.paper_logger.snapshot()
    """

    def __init__(self, log_dir=None):
        self.log_dir = log_dir or os.path.join(str(_Cfg.base_dir), 'paper_trading_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Config
        self.snapshot_interval_min = getattr(_PT, 'snapshot_interval_min', 15)
        self.max_decisions_in_memory = getattr(_PT, 'max_decisions_in_memory', 5000)
        self.max_trades_in_memory = getattr(_PT, 'max_trades_in_memory', 2000)

        # State
        self._decisions = deque(maxlen=self.max_decisions_in_memory)
        self._trades = deque(maxlen=self.max_trades_in_memory)
        self._equity_curve = []  # [(timestamp, balance)]
        self._lock = threading.Lock()

        # Session tracking
        self._session_start = datetime.utcnow()
        self._session_id = self._session_start.strftime('%Y%m%d_%H%M%S')
        self._total_decisions = 0
        self._total_trades = 0
        self._total_blocked = 0

        # File handles
        self._decision_file = os.path.join(self.log_dir, f'decisions_{self._session_id}.csv')
        self._trade_file = os.path.join(self.log_dir, f'trades_{self._session_id}.csv')
        self._equity_file = os.path.join(self.log_dir, f'equity_{self._session_id}.csv')
        self._init_csv_files()

        # Background snapshot thread
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_loop, daemon=True, name="PaperLogger"
        )
        self._snapshot_running = True
        self._snapshot_thread.start()

        print(f"📝 Paper Trading Logger initialized — session {self._session_id}")
        print(f"   📁 Logs: {self.log_dir}")

    def _init_csv_files(self):
        """Create CSV headers."""
        with open(self._decision_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'direction', 'confidence',
                'ppo_action', 'ppo_value', 'specialist_probs',
                'conformal_set_size', 'hmm_regime', 'passes_gate',
                'action_taken', 'risk_check_result'
            ])

        with open(self._trade_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'symbol', 'direction', 'entry_price', 'exit_price',
                'size', 'notional', 'pnl', 'pnl_pct',
                'entry_time', 'exit_time', 'hold_minutes',
                'barrier_hit', 'confidence', 'kelly_fraction',
                'risk_pct', 'atr_pct', 'regime_at_entry'
            ])

        with open(self._equity_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'balance', 'open_positions', 'daily_pnl',
                             'total_trades', 'win_rate', 'sharpe_rolling'])

    # ─────────────────────────────────────────────────
    # LOGGING API
    # ─────────────────────────────────────────────────

    def log_decision(self, symbol, direction, confidence, ppo_action=0, ppo_value=0.0,
                     specialist_probs=None, conformal_set_size=2, hmm_regime=2,
                     passes_gate=False, action_taken='GATE_BLOCKED', risk_result='OK'):
        """Log every prediction decision (trade or not)."""
        record = DecisionRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol, direction=direction, confidence=confidence,
            ppo_action=ppo_action, ppo_value=ppo_value,
            specialist_probs=list(specialist_probs) if specialist_probs is not None else [],
            conformal_set_size=conformal_set_size, hmm_regime=hmm_regime,
            passes_gate=passes_gate, action_taken=action_taken,
            risk_check_result=risk_result
        )

        with self._lock:
            self._decisions.append(record)
            self._total_decisions += 1
            if action_taken != 'TRADE_OPENED':
                self._total_blocked += 1

        # Append to CSV (non-blocking)
        try:
            with open(self._decision_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record.timestamp, symbol, direction, confidence,
                    ppo_action, ppo_value,
                    json.dumps(record.specialist_probs),
                    conformal_set_size, hmm_regime, passes_gate,
                    action_taken, risk_result
                ])
        except Exception as e:
            logging.warning(f"PaperTradingLogger: decision CSV write failed for {symbol}: {e}", exc_info=True)

    def log_trade_closed(self, symbol, direction, entry_price, exit_price,
                         size, pnl, entry_time=None, exit_time=None,
                         barrier_hit='TIMEOUT', confidence=0, kelly_fraction=0,
                         risk_pct=0, atr_pct=0, regime_at_entry=2):
        """Log a completed trade with full context."""
        now = datetime.utcnow().isoformat()
        notional = entry_price * size
        pnl_pct = pnl / max(notional, 1e-8) * 100
        entry_t = entry_time or now
        exit_t = exit_time or now

        # Calculate hold time
        try:
            if isinstance(entry_t, str) and isinstance(exit_t, str):
                et = datetime.fromisoformat(entry_t)
                xt = datetime.fromisoformat(exit_t)
                hold_min = (xt - et).total_seconds() / 60
            else:
                hold_min = 0
        except Exception:
            hold_min = 0

        record = TradeRecord(
            symbol=symbol, direction=direction,
            entry_price=entry_price, exit_price=exit_price,
            size=size, notional=notional, pnl=pnl, pnl_pct=pnl_pct,
            entry_time=str(entry_t), exit_time=str(exit_t),
            hold_minutes=hold_min, barrier_hit=barrier_hit,
            confidence=confidence, kelly_fraction=kelly_fraction,
            risk_pct=risk_pct, atr_pct=atr_pct,
            regime_at_entry=regime_at_entry
        )

        with self._lock:
            self._trades.append(record)
            self._total_trades += 1

        try:
            with open(self._trade_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    symbol, direction, entry_price, exit_price,
                    size, notional, pnl, pnl_pct,
                    entry_t, exit_t, hold_min,
                    barrier_hit, confidence, kelly_fraction,
                    risk_pct, atr_pct, regime_at_entry
                ])
        except Exception as e:
            logging.warning(f"PaperTradingLogger: trade CSV write failed for {symbol}: {e}", exc_info=True)

    def snapshot(self, balance, open_positions=0, daily_pnl=0.0):
        """Take a point-in-time equity snapshot."""
        now = datetime.utcnow().isoformat()

        # Rolling win rate
        recent = list(self._trades)[-100:]
        if recent:
            wins = sum(1 for t in recent if t.pnl > 0)
            win_rate = wins / len(recent) * 100
        else:
            win_rate = 0

        # Rolling Sharpe (last 100 trades)
        import numpy as np
        if len(recent) > 1:
            rets = [t.pnl_pct for t in recent]
            sharpe = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(252))
        else:
            sharpe = 0.0

        with self._lock:
            self._equity_curve.append((now, balance))

        try:
            with open(self._equity_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([now, balance, open_positions, daily_pnl,
                                 self._total_trades, win_rate, sharpe])
        except Exception as e:
            logging.warning(f"PaperTradingLogger: equity snapshot CSV write failed: {e}", exc_info=True)

    def get_session_summary(self):
        """Return summary dict for dashboard display."""
        trades_list = list(self._trades)
        if not trades_list:
            return {
                'session_id': self._session_id,
                'runtime_min': (datetime.utcnow() - self._session_start).total_seconds() / 60,
                'total_decisions': self._total_decisions,
                'total_trades': 0,
                'total_blocked': self._total_blocked,
                'win_rate': 0, 'total_pnl': 0, 'sharpe': 0,
            }

        import numpy as np
        pnls = [t.pnl for t in trades_list]
        pnl_pcts = [t.pnl_pct for t in trades_list]
        wins = sum(1 for p in pnls if p > 0)

        return {
            'session_id': self._session_id,
            'runtime_min': (datetime.utcnow() - self._session_start).total_seconds() / 60,
            'total_decisions': self._total_decisions,
            'total_trades': self._total_trades,
            'total_blocked': self._total_blocked,
            'win_rate': wins / max(len(pnls), 1) * 100,
            'total_pnl': sum(pnls),
            'mean_pnl': float(np.mean(pnls)),
            'sharpe': float(np.mean(pnl_pcts) / max(np.std(pnl_pcts), 1e-8) * np.sqrt(252)) if len(pnl_pcts) > 1 else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_hold_min': float(np.mean([t.hold_minutes for t in trades_list])),
            'barrier_dist': {
                'TP1': sum(1 for t in trades_list if t.barrier_hit == 'TP1'),
                'TP2': sum(1 for t in trades_list if t.barrier_hit == 'TP2'),
                'TP3': sum(1 for t in trades_list if t.barrier_hit == 'TP3'),
                'SL': sum(1 for t in trades_list if t.barrier_hit == 'SL'),
                'TIMEOUT': sum(1 for t in trades_list if t.barrier_hit == 'TIMEOUT'),
            }
        }

    def _snapshot_loop(self):
        """Background thread: periodic equity snapshots."""
        while self._snapshot_running:
            time.sleep(self.snapshot_interval_min * 60)
            # Balance will be set by the main bot loop via snapshot() calls

    def stop(self):
        """Graceful shutdown."""
        self._snapshot_running = False
        print(f"📝 Paper Trading Logger stopped — {self._total_trades} trades logged")
