# Nike V2 Cache Performance Report

Date: 2026-04-08 05:17 UTC

## Scope

This report benchmarks Nike v2 across the full local `5m` feather cache and compares it to the previous Nike baseline recorded in [NIKE_CACHE_PERFORMANCE_REPORT.md](/C:/Users/habib/QUANTA/NIKE_CACHE_PERFORMANCE_REPORT.md).

Validation gates:
- Top `500` realized spike recall on `same/+1/+2` >= `75.0%`
- Full-cache weighted PF > `1.150`
- Signal count growth <= `35%`, unless PF also improves

## Nike V2 Config Snapshot

- `nike_body_ratio_mult` = `5.0`
- `nike_quiet_body_pct` = `0.5`
- `nike_vol_mult` = `1.5`
- `nike_body_min` = `0.4`
- `nike_immediate_body_ratio_mult` = `8.0`
- `nike_immediate_body_min` = `0.55`
- `nike_immediate_vol_mult` = `2.0`
- `nike_continuation_vol_mult` = `1.0`
- `nike_tier_a_confidence` = `84.0`
- `nike_tier_b_confidence` = `78.0`
- `nike_tier_c_confidence` = `72.0`
- `nike_tier_a_size_mult` = `1.15`
- `nike_tier_b_size_mult` = `1.0`
- `nike_tier_c_size_mult` = `0.75`
- `nike_tier_b_bs_floor` = `0.3`
- `nike_tier_c_bs_floor` = `0.35`
- `nike_bank_atr` = `2.0`
- `nike_bank_fraction` = `0.5`
- `nike_runner_trail_atr` = `1.5`
- `nike_max_bars_pre_bank` = `24`
- `nike_max_bars_post_bank` = `36`

## Rollout Gates

| Gate | Baseline | Nike v2 | Status |
|---|---:|---:|---|
| Top 500 same/+1/+2 recall | 75.00% | 80.80% | PASS |
| Full-cache weighted PF | 1.150 | 1.001 | FAIL |
| Signal count growth | <= 35% unless PF improves | +8.38% | PASS |
| Overall rollout decision | All gates required | REJECT | FAIL |

## Cache Coverage

- `5m` feather files scanned: `230`
- Files used: `230`
- Total `5m` bars scanned: `19,838,110`
- Anomaly events evaluated: `35,621`
- Symbols with Nike signals: `215`
- Total Nike signals: `20,879`

## Anomaly Recall (`same/+1/+2`)

| Bucket | N | Same Bar | +1 Bar | +2 Bar | Same/+1/+2 | Mean Bars To Peak | Median Bars To Peak | Peak <= 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All anomalies | 35621 | 28.07% | 24.93% | 4.80% | 57.80% | 14.76 | 11.00 | 70.42% |
| Run-up > 1% | 19700 | 27.70% | 33.08% | 6.32% | 67.10% | 20.36 | 21.00 | 57.42% |
| Run-up > 3% | 7776 | 29.54% | 35.52% | 6.16% | 71.22% | 22.82 | 25.00 | 49.58% |
| Run-up > 5% | 3896 | 32.29% | 35.99% | 5.67% | 73.95% | 23.94 | 26.00 | 45.17% |
| Run-up > 10% | 1110 | 37.48% | 34.86% | 5.59% | 77.93% | 24.50 | 27.00 | 42.61% |
| Run-up > 20% | 245 | 36.33% | 36.33% | 4.49% | 77.14% | 24.38 | 28.00 | 43.27% |

## Largest Realized Spikes

| Bucket | N | Same Bar | +1 Bar | +2 Bar | Same/+1/+2 | Mean Bars To Peak | Median Bars To Peak | Peak <= 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top 50 run-up | 50 | 42.00% | 38.00% | 0.00% | 80.00% | 25.64 | 28.50 | 40.00% |
| Top 100 run-up | 100 | 35.00% | 38.00% | 4.00% | 77.00% | 24.12 | 28.00 | 45.00% |
| Top 250 run-up | 250 | 36.80% | 36.00% | 4.40% | 77.20% | 24.42 | 28.00 | 43.20% |
| Top 500 run-up | 500 | 38.20% | 36.80% | 5.80% | 80.80% | 24.53 | 28.00 | 42.20% |
| Top 1000 run-up | 1000 | 37.30% | 35.30% | 5.70% | 78.30% | 24.49 | 27.00 | 42.70% |
| Top 2000 run-up | 2000 | 34.60% | 34.80% | 6.20% | 75.60% | 24.61 | 28.00 | 41.35% |

## Nike Signal Outcomes Across Full Cache

- Total Nike signals: `20,879`
- `TP`: `5,516`
- `SL`: `15,232`
- `TIMEOUT`: `131`
- Decided trades (`TP` or `SL`): `20,748`
- Decided accuracy: `26.59%`
- Weighted profit factor: `1.001`
- Expectancy: `+0.0004 ATR` per signal
- Signal count delta vs baseline: `+8.38%`

### Per-Tier Trade Stats

| Tier | Signals | TP | SL | TIMEOUT | Weighted PF | Expectancy (ATR) | Avg Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 10347 | 2699 | 7584 | 64 | 1.061 | +0.0361 | 93.95 |
| B | 8849 | 2441 | 6348 | 60 | 0.980 | -0.0116 | 62.09 |
| C | 1683 | 376 | 1300 | 7 | 0.747 | -0.1564 | 62.19 |

## Top 500 Match Tier Mix

- Tier `A` matches: `222`
- Tier `B` matches: `160`
- Tier `C` matches: `22`

## Largest Missed Winners

- `TANSSIUSDT 2025-11-27 19:00` run-up `73.27%`, peak in `8` bars
- `LEVERUSDT 2025-07-16 15:05` run-up `63.14%`, peak in `30` bars
- `MUSDT 2025-08-30 05:45` run-up `48.69%`, peak in `36` bars
- `ALICEUSDT 2025-10-13 08:00` run-up `47.50%`, peak in `24` bars
- `FHEUSDT 2025-12-07 01:10` run-up `46.27%`, peak in `35` bars
- `HEMIUSDT 2025-08-31 03:00` run-up `45.81%`, peak in `25` bars
- `DOLOUSDT 2025-07-11 00:55` run-up `44.42%`, peak in `22` bars
- `DUSDT 2025-11-07 14:40` run-up `44.01%`, peak in `30` bars
- `TRADOORUSDT 2025-12-30 13:30` run-up `43.30%`, peak in `19` bars
- `ALPHAUSDT 2025-09-22 14:00` run-up `42.60%`, peak in `32` bars

## Tuning Search

Two follow-up searches were run after the first v2 benchmark:

- [nike_v2_tuning_search.csv](/C:/Users/habib/QUANTA/nike_v2_tuning_search.csv): Tier `C` disabled, Tier `B` score thresholds, current `bank_atr=2.0` family.
- [nike_v2_exit_search_wide.csv](/C:/Users/habib/QUANTA/nike_v2_exit_search_wide.csv): broader exit sweep over `bank_atr`, `bank_fraction`, `runner_trail_atr`, and Tier `B` score thresholds.

What those searches showed:

- Tier `C` should not be traded live. It adds very little top-spike recall and has the weakest PF.
- Tier `A` is the only tier with positive standalone PF under the tested Nike v2 exit family.
- Tier `A+B` keeps top-500 `same/+1/+2` recall above `75%`, but none of the tested exit/threshold combinations lifted full-cache PF above the `1.150` baseline.
- The best top-500-passing scenario in the wider search was still below the PF gate:
  - Tier `A+B`, `bank_atr=2.0`, `bank_fraction=0.25`, `runner_trail_atr=1.5`, `max_post=36`
  - top-500 recall `76.4%`
  - weighted PF `1.072`

Rollout decision from that search:

- Tier `A`: live
- Tier `B`: observe-only
- Tier `C`: observe-only

## Bottom Line

- Nike v2 top-500 `same/+1/+2` recall: `80.80%`
- Nike v2 full-cache weighted PF: `1.001`
- Nike v2 signal growth vs baseline: `+8.38%`
- Rollout verdict under the stated gates: `REJECT`
