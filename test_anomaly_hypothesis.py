"""
QUANTA — Anomaly Hypothesis Tester
===================================
Tests what % of extreme 5m candles are followed by continuation or reversal.
Thresholds tested: 2%, 5%, 10%, 15%, 20%
Lookaheads tested: 1, 3, 6, 12 bars (5m, 15m, 30m, 1h)

Run: python test_anomaly_hypothesis.py
Output: anomaly_hypothesis_results.csv + printed summary
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATHER_CACHE_DIR = "feather_cache"
THRESHOLDS        = [0.02, 0.05, 0.10, 0.15, 0.20]   # 2%, 5%, 10%, 15%, 20%
LOOKAHEAD_BARS    = [1, 3, 6, 12]                      # 5m, 15m, 30m, 1h
MIN_EVENTS        = 5                                   # min events per symbol to include
# ──────────────────────────────────────────────────────────────────────────────


def load_feather(path):
    df = pd.read_feather(path)
    df.columns = [c.lower() for c in df.columns]
    # handle common column name variants
    rename = {}
    for col in df.columns:
        if col in ['o', 'open_price']:   rename[col] = 'open'
        if col in ['h', 'high_price']:   rename[col] = 'high'
        if col in ['l', 'low_price']:    rename[col] = 'low'
        if col in ['c', 'close_price']:  rename[col] = 'close'
        if col in ['v', 'vol']:          rename[col] = 'volume'
    if rename:
        df = df.rename(columns=rename)
    required = ['open', 'high', 'low', 'close', 'volume']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing column: {r}. Found: {list(df.columns)}")
    return df[required].dropna().reset_index(drop=True)


def analyze_symbol(closes, opens, highs, lows, threshold, lookahead):
    results = []
    for i in range(1, len(closes) - lookahead):
        prev_close = closes[i - 1]
        if prev_close == 0:
            continue

        pct_change = (closes[i] - prev_close) / prev_close

        is_bull = pct_change >= threshold
        is_bear = pct_change <= -threshold

        if not is_bull and not is_bear:
            continue

        direction    = 'BULL' if is_bull else 'BEAR'
        future_close = closes[i + lookahead]

        # continuation = move continues in same direction
        # reversal     = move flips direction
        if is_bull:
            continuation = 1 if future_close > closes[i] else 0
        else:
            continuation = 1 if future_close < closes[i] else 0

        reversal = 1 - continuation

        # magnitude of the move after
        future_pct = (future_close - closes[i]) / closes[i] if closes[i] != 0 else 0

        results.append({
            'pct_change':   round(pct_change * 100, 3),
            'direction':    direction,
            'continuation': continuation,
            'reversal':     reversal,
            'future_pct':   round(future_pct * 100, 3),
        })
    return results


def main():
    feather_files = list(Path(FEATHER_CACHE_DIR).glob("*_5m.feather"))
    if not feather_files:
        # try without _5m suffix
        feather_files = list(Path(FEATHER_CACHE_DIR).glob("*.feather"))

    if not feather_files:
        print(f"No feather files found in {FEATHER_CACHE_DIR}/")
        print("Check FEATHER_CACHE_DIR path at top of script.")
        return

    print(f"Found {len(feather_files)} feather files")

    # print columns of first file
    try:
        sample = pd.read_feather(feather_files[0])
        sample.columns = [c.lower() for c in sample.columns]
        print(f"Columns in feather: {list(sample.columns)}\n")
    except Exception as e:
        print(f"Could not read sample file: {e}\n")

    all_records = []

    for threshold in THRESHOLDS:
        for lookahead in LOOKAHEAD_BARS:
            records = []

            for path in feather_files:
                symbol = path.stem.replace("_5m", "")
                try:
                    df      = load_feather(path)
                    closes  = df['close'].values
                    opens   = df['open'].values
                    highs   = df['high'].values
                    lows    = df['low'].values

                    sym_results = analyze_symbol(closes, opens, highs, lows, threshold, lookahead)
                    for r in sym_results:
                        r['symbol'] = symbol
                    records.extend(sym_results)

                except Exception as e:
                    continue

            if not records:
                continue

            df_r = pd.DataFrame(records)
            total        = len(df_r)
            bull_total   = len(df_r[df_r['direction'] == 'BULL'])
            bear_total   = len(df_r[df_r['direction'] == 'BEAR'])
            cont_rate    = df_r['continuation'].mean() * 100
            bull_cont    = df_r[df_r['direction'] == 'BULL']['continuation'].mean() * 100 if bull_total > 0 else 0
            bear_cont    = df_r[df_r['direction'] == 'BEAR']['continuation'].mean() * 100 if bear_total > 0 else 0
            bear_rev     = df_r[df_r['direction'] == 'BEAR']['reversal'].mean() * 100 if bear_total > 0 else 0

            all_records.append({
                'threshold_%':        threshold * 100,
                'lookahead_bars':     lookahead,
                'lookahead_min':      lookahead * 5,
                'total_events':       total,
                'bull_events':        bull_total,
                'bear_events':        bear_total,
                'overall_cont_%':     round(cont_rate, 1),
                'bull_cont_%':        round(bull_cont, 1),
                'bear_cont_%':        round(bear_cont, 1),
                'bear_reversal_%':    round(bear_rev, 1),
            })

    if not all_records:
        print("No events found across any threshold. Check feather file format.")
        return

    df_all = pd.DataFrame(all_records)

    print("=" * 70)
    print("ANOMALY HYPOTHESIS RESULTS")
    print("=" * 70)
    print(df_all.to_string(index=False))

    print("\n" + "=" * 70)
    print("BEAR REVERSAL RATES (your flip hypothesis)")
    print("=" * 70)
    bear_df = df_all[['threshold_%', 'lookahead_min', 'bear_events', 'bear_cont_%', 'bear_reversal_%']]
    print(bear_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for _, row in df_all.iterrows():
        tag = ""
        if row['overall_cont_%'] > 65:
            tag = "✅ STRONG EDGE"
        elif row['overall_cont_%'] > 55:
            tag = "⚠️  WEAK EDGE"
        else:
            tag = "❌ NO EDGE"
        if row['bear_reversal_%'] > 65:
            tag += " | 🔄 BEAR FLIP CONFIRMED"
        print(f"  {row['threshold_%']:.0f}% threshold | {row['lookahead_min']:.0f}min lookahead | "
              f"cont={row['overall_cont_%']}% | bear_flip={row['bear_reversal_%']}% | {tag}")

    df_all.to_csv('anomaly_hypothesis_results.csv', index=False)
    print(f"\nSaved to anomaly_hypothesis_results.csv")


if __name__ == "__main__":
    main()
