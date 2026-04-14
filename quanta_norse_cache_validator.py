"""
Norse cache-paper validator.

Simulates the 3-agent Norse operating set:
- Thor: current Nike v2 breakout logic
- Baldur: Thor-linked bearish top-start detector
- Freya: Thor-linked short-horizon momentum scalp

Outputs:
- NORSE_CACHE_SIM_REPORT.md
- norse_cache_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

from quanta_config import Config
from quanta_nike_live_validator import extract_nike_signals, simulate_nike_exit
from quanta_norse_agents import (
    calc_atr,
    extract_baldur_signals,
    extract_freya_signals,
    measure_baldur_topstart,
    simulate_directional_exit,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "feather_cache"
REPORT_PATH = ROOT / "NORSE_CACHE_SIM_REPORT.md"
CSV_PATH = ROOT / "norse_cache_summary.csv"


def _safe_pf(pos_sum: float, neg_sum: float) -> float:
    if neg_sum >= 0:
        return float("inf")
    return pos_sum / abs(neg_sum)


def _summarize_results(rows: list[dict]) -> dict:
    tp = sum(1 for row in rows if row["label"] == "TP")
    sl = sum(1 for row in rows if row["label"] == "SL")
    timeout = sum(1 for row in rows if row["label"] == "TIMEOUT")
    decided = tp + sl
    realized = [float(row["realized_atr"]) for row in rows]
    pos_sum = sum(x for x in realized if x >= 0)
    neg_sum = sum(x for x in realized if x < 0)
    hold_bars = [int(row["exit_bar"] - row["bar_idx"]) for row in rows if row.get("exit_bar") is not None]
    return {
        "signals": len(rows),
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "decided_accuracy": (tp / decided * 100.0) if decided else 0.0,
        "expectancy_atr": (sum(realized) / len(realized)) if realized else 0.0,
        "weighted_pf": _safe_pf(pos_sum, neg_sum),
        "avg_hold_bars": mean(hold_bars) if hold_bars else 0.0,
        "median_hold_bars": median(hold_bars) if hold_bars else 0.0,
    }


def _format_pf(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.3f}"


def run_cache_validation() -> tuple[pd.DataFrame, str]:
    cfg = Config.events
    files = sorted(CACHE_DIR.glob("*_5m.feather"))

    thor_rows: list[dict] = []
    baldur_rows: list[dict] = []
    freya_rows: list[dict] = []
    baldur_topstart: list[dict] = []
    baldur_warning_count = 0
    overlap_rows: list[dict] = []
    symbols_seen = 0

    for path in files:
        try:
            df = pd.read_feather(path)
        except Exception:
            continue
        if df.empty or len(df) < 80:
            continue

        symbol = path.stem.replace("_5m", "")
        df.attrs["symbol"] = symbol
        symbols_seen += 1

        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        atrs = calc_atr(highs, lows, closes)

        thor_signals = extract_nike_signals(df)
        for sig in thor_signals:
            sig["symbol"] = symbol
            sig["atr"] = float(atrs[int(sig["bar_idx"])]) if int(sig["bar_idx"]) < len(atrs) else 0.0
            sim = simulate_nike_exit(sig, closes, highs, lows, atrs)
            thor_rows.append({
                "agent": "Thor",
                "symbol": symbol,
                "bar_idx": int(sig["bar_idx"]),
                "score": float(sig.get("score", 0.0)),
                "tier": str(sig.get("tier", "")),
                "label": sim["label"],
                "realized_atr": float(sim["realized_atr"]),
                "exit_bar": int(sim["exit_bar"]),
            })

        baldur_warnings, baldur_signals = extract_baldur_signals(df, thor_signals, cfg)
        baldur_warning_count += len(baldur_warnings)
        for sig in baldur_signals:
            sim = simulate_directional_exit(
                sig,
                closes,
                highs,
                lows,
                float(cfg.baldur_tp_atr),
                float(cfg.baldur_sl_atr),
                int(cfg.baldur_max_bars),
            )
            baldur_rows.append({
                "agent": "Baldur",
                "symbol": symbol,
                "bar_idx": int(sig["bar_idx"]),
                "score": float(sig.get("score", 0.0)),
                "label": sim["label"],
                "realized_atr": float(sim["realized_atr"]),
                "exit_bar": int(sim["exit_bar"]),
                "warning_bar_idx": int(sig.get("warning_bar_idx", sig["bar_idx"])),
            })
            topstart = measure_baldur_topstart(
                sig,
                closes,
                lows,
                target_atr=1.0,
                max_bars=int(cfg.baldur_max_bars),
            )
            baldur_topstart.append(topstart)

        freya_signals = extract_freya_signals(df, thor_signals, cfg)
        for sig in freya_signals:
            sim = simulate_directional_exit(
                sig,
                closes,
                highs,
                lows,
                float(cfg.freya_tp_atr),
                float(cfg.freya_sl_atr),
                int(cfg.freya_max_bars),
            )
            freya_rows.append({
                "agent": "Freya",
                "symbol": symbol,
                "bar_idx": int(sig["bar_idx"]),
                "score": float(sig.get("score", 0.0)),
                "label": sim["label"],
                "realized_atr": float(sim["realized_atr"]),
                "exit_bar": int(sim["exit_bar"]),
            })

        thor_bars = {int(row["bar_idx"]) for row in thor_rows if row["symbol"] == symbol}
        for row in freya_rows:
            if row["symbol"] != symbol:
                continue
            overlap_rows.append({
                "symbol": symbol,
                "freya_bar_idx": int(row["bar_idx"]),
                "thor_window_overlap": any(abs(int(row["bar_idx"]) - t) <= int(cfg.thor_context_bars) for t in thor_bars),
            })

    summary_rows = []
    thor_summary = _summarize_results(thor_rows)
    baldur_summary = _summarize_results(baldur_rows)
    freya_summary = _summarize_results(freya_rows)
    for agent, stats in (("Thor", thor_summary), ("Baldur", baldur_summary), ("Freya", freya_summary)):
        summary_rows.append({"agent": agent, **stats})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(CSV_PATH, index=False)

    baldur_successes = [row for row in baldur_topstart if row.get("success")]
    baldur_delay = [int(row["delay_bars"]) for row in baldur_successes if row.get("delay_bars") is not None]
    freya_overlap_rate = (
        sum(1 for row in overlap_rows if row["thor_window_overlap"]) / len(overlap_rows) * 100.0
        if overlap_rows else 0.0
    )

    report = f"""# Norse Cache Simulation Report

## Coverage
- Symbols scanned: `{symbols_seen}`
- Cache files scanned: `{len(files)}`
- Thor signals: `{thor_summary['signals']}`
- Baldur signals: `{baldur_summary['signals']}`
- Freya signals: `{freya_summary['signals']}`

## Agent Outcomes
| Agent | Signals | TP | SL | TIMEOUT | Decided Acc | Weighted PF | Expectancy (ATR) | Avg Hold Bars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | {thor_summary['signals']} | {thor_summary['tp']} | {thor_summary['sl']} | {thor_summary['timeout']} | {thor_summary['decided_accuracy']:.2f}% | {_format_pf(thor_summary['weighted_pf'])} | {thor_summary['expectancy_atr']:+.4f} | {thor_summary['avg_hold_bars']:.2f} |
| Baldur | {baldur_summary['signals']} | {baldur_summary['tp']} | {baldur_summary['sl']} | {baldur_summary['timeout']} | {baldur_summary['decided_accuracy']:.2f}% | {_format_pf(baldur_summary['weighted_pf'])} | {baldur_summary['expectancy_atr']:+.4f} | {baldur_summary['avg_hold_bars']:.2f} |
| Freya | {freya_summary['signals']} | {freya_summary['tp']} | {freya_summary['sl']} | {freya_summary['timeout']} | {freya_summary['decided_accuracy']:.2f}% | {_format_pf(freya_summary['weighted_pf'])} | {freya_summary['expectancy_atr']:+.4f} | {freya_summary['avg_hold_bars']:.2f} |

## Baldur Top-Start Study
- Top warnings generated: `{baldur_warning_count}`
- Confirmed Baldur shorts: `{baldur_summary['signals']}`
- 1 ATR downside reached within `{cfg.baldur_max_bars}` bars: `{(len(baldur_successes) / len(baldur_topstart) * 100.0) if baldur_topstart else 0.0:.2f}%`
- Median delay to first sustained downside leg: `{median(baldur_delay) if baldur_delay else 'n/a'}`

## Thor/Freya Interaction
- Freya signals inside active Thor windows: `{freya_overlap_rate:.2f}%`
- Thor context bars: `{cfg.thor_context_bars}`
- Freya max bars: `{cfg.freya_max_bars}`

## Operating Verdict
- Thor remains the only live-ready Norse agent in this pass if its PF stays aligned with the existing Nike v2 baseline.
- Baldur and Freya should remain observe-only until their cache paper stats are reviewed.
- Legacy Greek specialists can stay loaded for attribution while model-driven live execution is limited to `model_live_specialists = {cfg.model_live_specialists!r}`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    return summary_df, report


if __name__ == "__main__":
    df, report_text = run_cache_validation()
    print(df.to_string(index=False))
    print(f"\nWrote {REPORT_PATH.name} and {CSV_PATH.name}")
