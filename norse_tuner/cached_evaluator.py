"""
norse_tuner/cached_evaluator.py
================================
CachedTrialEvaluator evaluates Thor tuning candidates from norse_pump_mae_rows.csv
using a fast capital replay that approximates the real simulator more closely than
pure ATR compounding.

Cacheable params (CSV-filtered):
  thor_entry_hour_utc, thor_trade_tiers, thor_min_score_trade,
  thor_wave_strength_min, thor_top_risk_max, thor_mae_veto_atr,
  thor_sl_atr, thor_trade_cooldown_bars, baldur_warning_exit_score,
  learned_filter_threshold

Non-cacheable (need stage-2 Numba replay):
  thor_bank_atr, thor_bank_fraction, thor_trail_activate_atr,
  thor_runner_trail_atr, thor_max_bars_pre_bank, thor_max_bars_post_bank
  (their realized exits are baked into the MAE cache)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quanta_config import Config

from .fast_capital import BAR_MS_5M, simulate_fast_thor_metrics


_SL_COLS: dict[float, str] = {
    0.8: "realized_atr_sl_0p8",
    1.0: "realized_atr_sl_1p0",
    1.2: "realized_atr_sl_1p2",
    1.5: "realized_atr_sl_1p5",
    1.8: "realized_atr_sl_1p8",
    2.0: "realized_atr_sl_2p0",
    2.4: "realized_atr_sl_2p4",
    3.0: "realized_atr_sl_3p0",
}
_EXIT_BAR_COLS: dict[float, str] = {
    0.8: "exit_bar_sl_0p8",
    1.0: "exit_bar_sl_1p0",
    1.2: "exit_bar_sl_1p2",
    1.5: "exit_bar_sl_1p5",
    1.8: "exit_bar_sl_1p8",
    2.0: "exit_bar_sl_2p0",
    2.4: "exit_bar_sl_2p4",
    3.0: "exit_bar_sl_3p0",
}
_SL_LEVELS = sorted(_SL_COLS.keys())
_REQUIRED_BASE_COLUMNS = (
    "entry_ts",
    "entry_bar",
    "symbol",
    "entry_price",
    "entry_atr",
    "tier",
    "nike_score",
    "thor_feature_score",
    "hour_of_day_utc",
    "wave_strength_score_at_entry",
    "top_risk_score_at_entry",
    "pre_entry_mae_atr",
    "baldur_warning_score",
    "baldur_warning_fired_bar",
    "baldur_exit_runup_atr",
)


def _parse_allowed_hours(text: str) -> frozenset[int]:
    text = str(text or "").strip()
    if not text:
        return frozenset()
    result: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            result.add(int(part))
    return frozenset(result)


def _parse_allowed_tiers(text: str) -> frozenset[str]:
    return frozenset(p.strip().upper() for p in str(text or "").split(",") if p.strip())


def _nearest_sl(sl_atr: float) -> float:
    return min(_SL_LEVELS, key=lambda k: abs(k - sl_atr))


class CachedTrialEvaluator:
    """
    Walk-forward-aware cached evaluator backed by norse_pump_mae_rows.csv.
    """

    def __init__(
        self,
        mae_csv_path: str | Path,
        n_folds: int = 5,
        burn_in_frac: float = 0.40,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.mae_csv_path = Path(mae_csv_path)
        if not self.mae_csv_path.exists():
            raise FileNotFoundError(f"Norse MAE CSV not found: {self.mae_csv_path}")

        df = pd.read_csv(self.mae_csv_path)
        if df.empty:
            raise ValueError(f"Norse MAE CSV is empty: {self.mae_csv_path}")

        missing = [col for col in _REQUIRED_BASE_COLUMNS if col not in df.columns]
        missing.extend(col for col in _SL_COLS.values() if col not in df.columns)
        missing.extend(col for col in _EXIT_BAR_COLS.values() if col not in df.columns)
        if missing:
            missing_text = ", ".join(sorted(set(missing)))
            raise ValueError(
                "Norse MAE CSV contract is incomplete. "
                f"Missing columns in {self.mae_csv_path.name}: {missing_text}"
            )

        df = df.sort_values("entry_ts").reset_index(drop=True)
        self.n_rows = len(df)
        self.initial_capital = float(initial_capital)
        self.n_folds = int(n_folds)

        self._commission_bps = float(getattr(Config.backtest, "commission_bps", 4.0))
        self._slippage_bps = float(getattr(Config.backtest, "slippage_bps", 2.0))
        self._compound_mode = str(getattr(Config.events, "compound_mode", "asymmetric_target"))
        self._compound_max_loss_pct = float(getattr(Config.events, "compound_max_loss_pct", 3.0))
        self._compound_activation_score = float(getattr(Config.events, "compound_activation_score", 85.0))
        self._default_max_leverage = float(getattr(Config.events, "thor_max_leverage", 5.0))
        self._default_capital_cap = float(getattr(Config.events, "thor_capital_cap", 0.5))
        self._default_max_concurrent_positions = int(getattr(Config.events, "thor_max_concurrent_positions", 3))

        self._entry_ts = df["entry_ts"].to_numpy(dtype=np.int64)
        self._entry_bar = df["entry_bar"].to_numpy(dtype=np.int64)
        self._symbol = df["symbol"].to_numpy(dtype=object)
        self._entry_price = df["entry_price"].to_numpy(dtype=np.float64)
        self._entry_atr = df["entry_atr"].to_numpy(dtype=np.float64)
        self._tier = df["tier"].to_numpy(dtype=object)
        self._score = df["thor_feature_score"].to_numpy(dtype=np.float64)
        self._hour = df["hour_of_day_utc"].to_numpy(dtype=np.int64)
        self._wave = df["wave_strength_score_at_entry"].to_numpy(dtype=np.float64)
        self._top_risk = df["top_risk_score_at_entry"].to_numpy(dtype=np.float64)
        self._mae_veto = df["pre_entry_mae_atr"].to_numpy(dtype=np.float64)
        self._baldur_score = df["baldur_warning_score"].to_numpy(dtype=np.float64)
        self._baldur_bar = df["baldur_warning_fired_bar"].to_numpy(dtype=np.float64)
        self._baldur_runup = df["baldur_exit_runup_atr"].to_numpy(dtype=np.float64)

        self._sl_arrays: dict[float, np.ndarray] = {
            sl: df[col].to_numpy(dtype=np.float64)
            for sl, col in _SL_COLS.items()
        }
        self._exit_bar_arrays: dict[float, np.ndarray] = {
            sl: df[col].to_numpy(dtype=np.int64)
            for sl, col in _EXIT_BAR_COLS.items()
        }

        self._fold_bounds = self._build_folds(self.n_folds, float(burn_in_frac))
        self.learned_proba: dict[float, np.ndarray] = {}

    def _build_folds(self, n: int, burn_in: float) -> list[tuple[int, int]]:
        start = int(self.n_rows * burn_in)
        fold_size = (self.n_rows - start) // n
        bounds = []
        for k in range(n):
            lo = start + k * fold_size
            hi = lo + fold_size if k < n - 1 else self.n_rows
            bounds.append((lo, hi))
        return bounds

    def _filter_mask(self, params: dict, row_mask: Optional[np.ndarray] = None) -> np.ndarray:
        mask = np.ones(self.n_rows, dtype=bool) if row_mask is None else row_mask.copy()

        hours = _parse_allowed_hours(params.get("thor_entry_hour_utc", ""))
        if hours:
            mask &= np.isin(self._hour, list(hours))

        tiers = _parse_allowed_tiers(params.get("thor_trade_tiers", ""))
        if tiers:
            mask &= np.isin(self._tier, list(tiers))

        mask &= self._score >= float(params.get("thor_min_score_trade", 0.0))
        mask &= self._mae_veto <= float(params.get("thor_mae_veto_atr", 999.0))
        mask &= self._wave >= float(params.get("thor_wave_strength_min", 0.0))
        mask &= self._top_risk < float(params.get("thor_top_risk_max", 999.0))

        sl_key = _nearest_sl(float(params.get("thor_sl_atr", 3.0)))
        threshold = float(params.get("learned_filter_threshold", 0.0))
        if threshold > 0.0 and sl_key in self.learned_proba:
            proba = self.learned_proba[sl_key]
            has_pred = ~np.isnan(proba)
            mask &= (~has_pred) | (proba >= threshold)

        return mask

    def _apply_cooldown(self, mask: np.ndarray, params: dict) -> np.ndarray:
        cooldown = int(params.get("thor_trade_cooldown_bars", 0))
        if cooldown <= 0:
            return mask
        kept = mask.copy()
        order = np.lexsort((self._entry_bar, self._symbol))
        last_bar: dict[str, int] = {}
        for idx in order:
            if not kept[idx]:
                continue
            sym = str(self._symbol[idx])
            bar = int(self._entry_bar[idx])
            prev = last_bar.get(sym, -(10**9))
            if bar - prev < cooldown:
                kept[idx] = False
            else:
                last_bar[sym] = bar
        return kept

    def _selected_trade_arrays(self, mask: np.ndarray, params: dict) -> dict[str, np.ndarray]:
        sl_key = _nearest_sl(float(params.get("thor_sl_atr", 3.0)))
        if sl_key not in self._sl_arrays:
            return {
                "entry_ts": np.zeros(0, dtype=np.int64),
                "exit_ts": np.zeros(0, dtype=np.int64),
                "symbol": np.empty(0, dtype=object),
                "entry_price": np.zeros(0, dtype=np.float64),
                "entry_atr": np.zeros(0, dtype=np.float64),
                "realized_atr": np.zeros(0, dtype=np.float64),
                "score": np.zeros(0, dtype=np.float64),
            }

        realized = self._sl_arrays[sl_key].copy()
        exit_bar = self._exit_bar_arrays[sl_key].copy()

        trigger_score = float(params.get("baldur_warning_exit_score", 999.0))
        has_warning = ~np.isnan(self._baldur_bar)
        score_ok = self._baldur_score >= trigger_score
        has_runup = ~np.isnan(self._baldur_runup)
        warning_exit_bar = np.full(self.n_rows, -1, dtype=np.int64)
        warning_exit_bar[has_warning] = self._baldur_bar[has_warning].astype(np.int64) + 1
        override = (
            has_warning
            & score_ok
            & has_runup
            & (warning_exit_bar > self._entry_bar)
            & (warning_exit_bar < exit_bar)
        )
        realized[override] = self._baldur_runup[override]
        exit_bar[override] = warning_exit_bar[override]

        hold_bars = np.maximum(exit_bar - self._entry_bar, 0)
        exit_ts = self._entry_ts + hold_bars.astype(np.int64) * BAR_MS_5M

        return {
            "entry_ts": self._entry_ts[mask],
            "exit_ts": exit_ts[mask],
            "symbol": self._symbol[mask],
            "entry_price": self._entry_price[mask],
            "entry_atr": self._entry_atr[mask],
            "realized_atr": realized[mask],
            "score": self._score[mask],
        }

    def evaluate(
        self,
        params: dict,
        fold_slice: Optional[tuple[int, int]] = None,
    ) -> dict:
        row_mask = None
        if fold_slice is not None:
            lo, hi = fold_slice
            row_mask = np.zeros(self.n_rows, dtype=bool)
            row_mask[lo:hi] = True

        mask = self._filter_mask(params, row_mask)
        mask = self._apply_cooldown(mask, params)
        trades = self._selected_trade_arrays(mask, params)

        return simulate_fast_thor_metrics(
            entry_ts=trades["entry_ts"],
            exit_ts=trades["exit_ts"],
            symbol=trades["symbol"],
            entry_price=trades["entry_price"],
            entry_atr=trades["entry_atr"],
            realized_atr=trades["realized_atr"],
            score=trades["score"],
            stop_atr=float(params.get("thor_sl_atr", 3.0)),
            risk_pct=float(params.get("thor_risk_pct", 0.005)),
            max_leverage=float(params.get("thor_max_leverage", self._default_max_leverage)),
            capital_cap=float(params.get("thor_capital_cap", self._default_capital_cap)),
            max_concurrent_positions=int(
                params.get("thor_max_concurrent_positions", self._default_max_concurrent_positions)
            ),
            initial_capital=self.initial_capital,
            commission_bps=self._commission_bps,
            slippage_bps=self._slippage_bps,
            compound_mode=self._compound_mode,
            compound_max_loss_pct=self._compound_max_loss_pct,
            compound_activation_score=self._compound_activation_score,
        )

    def evaluate_walkforward(self, params: dict) -> list[dict]:
        return [self.evaluate(params, fold_slice=fold_bounds) for fold_bounds in self._fold_bounds]

    def evaluate_full(self, params: dict) -> dict:
        return self.evaluate(params, fold_slice=None)
