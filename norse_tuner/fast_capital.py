"""
Shared fast capital replay helpers for Thor-only cached tuning paths.
"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np


BAR_MS_5M = 5 * 60 * 1000


def empty_metrics(realized_atr: np.ndarray | None = None) -> dict:
    arr = np.asarray(realized_atr if realized_atr is not None else [], dtype=np.float64)
    return {
        "final_capital": 0.0,
        "growth_pct": -100.0,
        "max_drawdown_pct": 100.0,
        "trades": 0,
        "pf": 0.0,
        "expectancy": 0.0,
        "calmar": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "realized_atr": arr,
    }


def _get_notional_size(
    equity: float,
    stop_dist_pct: float,
    score: float,
    risk_pct: float,
    compound_mode: str,
    compound_max_loss_pct: float,
    compound_activation_score: float,
) -> float:
    if compound_mode == "asymmetric_target" and score >= compound_activation_score:
        target_loss_usd = equity * (compound_max_loss_pct / 100.0)
    else:
        target_loss_usd = equity * risk_pct
    return target_loss_usd / max(0.0001, stop_dist_pct)


def simulate_fast_thor_metrics(
    *,
    entry_ts: np.ndarray,
    exit_ts: np.ndarray,
    symbol: np.ndarray,
    entry_price: np.ndarray,
    entry_atr: np.ndarray,
    realized_atr: np.ndarray,
    score: np.ndarray,
    stop_atr: float,
    risk_pct: float,
    max_leverage: float,
    capital_cap: float,
    max_concurrent_positions: int,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    compound_mode: str,
    compound_max_loss_pct: float,
    compound_activation_score: float,
) -> dict:
    n_rows = int(len(entry_ts))
    if n_rows <= 0:
        return empty_metrics()

    friction_rate = (float(commission_bps) + float(slippage_bps)) / 10000.0
    max_leverage = max(float(max_leverage), 1e-12)
    capital_cap = max(0.0, float(capital_cap))
    max_concurrent_positions = max(1, int(max_concurrent_positions))
    stop_atr = max(float(stop_atr), 0.1)
    risk_pct = max(0.0, float(risk_pct))

    free_cash = float(initial_capital)
    peak = free_cash
    max_dd = 0.0
    open_margin = 0.0
    open_heap: list[tuple[int, int, dict]] = []
    open_symbols: set[str] = set()
    accepted_realized: list[float] = []
    accepted_net_pnl: list[float] = []
    seq = 0

    def current_equity() -> float:
        return free_cash + open_margin

    def close_due(until_ts: int) -> None:
        nonlocal free_cash, peak, max_dd, open_margin
        while open_heap and open_heap[0][0] <= until_ts:
            _, _, pos = heappop(open_heap)
            open_symbols.discard(pos["symbol"])
            open_margin = max(0.0, open_margin - pos["margin_required"])
            free_cash += pos["margin_required"] + pos["close_cashflow"]
            equity_now = current_equity()
            peak = max(peak, equity_now)
            dd = 0.0 if peak <= 0.0 else (peak - equity_now) / peak
            max_dd = max(max_dd, dd)

    for idx in range(n_rows):
        close_due(int(entry_ts[idx]))

        sym = str(symbol[idx])
        if sym in open_symbols:
            continue
        if len(open_heap) >= max_concurrent_positions:
            continue

        equity_now = current_equity()
        if equity_now <= 0.0 or free_cash <= 0.0:
            break

        entry_px = max(float(entry_price[idx]), 1e-12)
        atr_px = max(float(entry_atr[idx]), 0.0)
        realized = float(realized_atr[idx])
        trade_score = float(score[idx])

        stop_dist_pct = max(0.001, (stop_atr * atr_px) / entry_px)
        total_notional_cap = free_cash * max_leverage
        agent_margin_cap = max(0.0, equity_now * capital_cap - open_margin)
        agent_notional_cap = agent_margin_cap * max_leverage
        notional_by_risk = _get_notional_size(
            equity_now,
            stop_dist_pct,
            trade_score,
            risk_pct,
            compound_mode,
            compound_max_loss_pct,
            compound_activation_score,
        )
        entry_notional = max(0.0, min(notional_by_risk, total_notional_cap, agent_notional_cap))
        if entry_notional <= 10.0:
            continue

        margin_required = entry_notional / max_leverage
        entry_friction = entry_notional * friction_rate
        if margin_required + entry_friction > free_cash:
            continue

        free_cash -= margin_required + entry_friction
        open_margin += margin_required

        gross_return_pct = realized * atr_px / entry_px
        gross_pnl = entry_notional * gross_return_pct
        size_units = entry_notional / entry_px
        exit_price_est = max(1e-12, entry_px + realized * atr_px)
        exit_notional = abs(exit_price_est * size_units)
        exit_friction = exit_notional * friction_rate
        close_cashflow = gross_pnl - exit_friction
        net_pnl = gross_pnl - entry_friction - exit_friction

        accepted_realized.append(realized)
        accepted_net_pnl.append(net_pnl)
        open_symbols.add(sym)
        heappush(
            open_heap,
            (
                int(exit_ts[idx]),
                seq,
                {
                    "symbol": sym,
                    "margin_required": margin_required,
                    "close_cashflow": close_cashflow,
                },
            ),
        )

        equity_now = current_equity()
        peak = max(peak, equity_now)
        dd = 0.0 if peak <= 0.0 else (peak - equity_now) / peak
        max_dd = max(max_dd, dd)
        seq += 1

    close_due(2**63 - 1)

    realized_arr = np.asarray(accepted_realized, dtype=np.float64)
    if realized_arr.size == 0:
        return empty_metrics(realized_arr)

    net_arr = np.asarray(accepted_net_pnl, dtype=np.float64)
    pnl_pos_arr = net_arr[net_arr > 0.0]
    pnl_neg_arr = -net_arr[net_arr < 0.0]

    pnl_pos = float(pnl_pos_arr.sum()) if pnl_pos_arr.size > 0 else 0.0
    pnl_neg = float(pnl_neg_arr.sum()) if pnl_neg_arr.size > 0 else 0.0
    pf = pnl_pos / pnl_neg if pnl_neg > 0.0 else (10.0 if pnl_pos > 0.0 else 0.0)

    avg_win = float(pnl_pos_arr.mean()) if pnl_pos_arr.size > 0 else 0.0
    avg_loss = float(pnl_neg_arr.mean()) if pnl_neg_arr.size > 0 else 0.0

    final_capital = max(0.0, free_cash)
    growth_pct = ((final_capital / float(initial_capital)) - 1.0) * 100.0 if initial_capital > 0 else -100.0
    max_drawdown_pct = float(max_dd * 100.0)
    calmar = growth_pct / max(max_drawdown_pct, 1.0)

    return {
        "final_capital": final_capital,
        "growth_pct": growth_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": int(realized_arr.size),
        "pf": pf,
        "expectancy": float(realized_arr.mean()),
        "calmar": calmar,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "realized_atr": realized_arr,
    }
