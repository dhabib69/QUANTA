# Norse Year Paper Simulation Report

## Setup
- Initial paper capital: `$10,000.00`
- Window: last `365` cached days across the local `5m` universe
- Symbols included: `218`
- Optimization target: `maximum final capital with >60% drawdown penalty`
- Stops modeled as: `stop-market`
- Portfolio caps: `Thor 50%`, `Freya 25%`
- Active agents: `Thor`

## Best Parameters
- Thor score floor: `70.0`
- Thor tiers: `A,B`
- Thor stop: `2.40 ATR`
- Thor bank: `3.60 ATR @ 15%`
- Thor trail activate: `1.50 ATR`
- Thor trail gap: `2.50 ATR`
- Thor MAE veto: `6.60 ATR`
- Thor wave-strength min: `50.0`
- Thor top-risk max: `45.0`
- Baldur warning exit score: `62.5`
- Freya score floor: `74.0`
- Freya pyramid wave min: `55.0`
- Freya pyramid top-risk max: `35.0`
- Freya add size: `50%` of parent Thor notional
- Pump material drawdown threshold: `1.00 ATR`

## Portfolio Result
- Final capital: `$16,200.44`
- Growth: `+62.00%`
- Max drawdown: `48.98%`
- Executed trades: `104`
- Liquidation hit: `NO`

## Thor-Only Baseline
- Final capital: `$16,200.44`
- Growth: `+62.00%`
- Max drawdown: `48.98%`
- Executed trades: `104`

## Agent Statistics (Candidate Portfolio)
| Agent | Trades | TP | SL | TIMEOUT | TP Rate | Decided Win Rate | Net PnL ($) | Avg PnL ($) | Profit Factor | Avg Hold Bars | Capital Contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | 104 | 63 | 34 | 7 | 60.58% | 64.95% | +6200.44 | +59.62 | 1.814 | 10.38 | +100.00% |
| Freya | 0 | 0 | 0 | 0 | 0.00% | 0.00% | +0.00 | +0.00 | 0.000 | 0.00 | +0.00% |
| Baldur | 0 | 0 | 0 | 0 | 0.00% | 0.00% | +0.00 | +0.00 | 0.000 | 0.00 | +0.00% |

## Pump Analytics
- Thor pumps recorded: `136`
- Median max run-up: `4.42 ATR`
- Median max drawdown: `2.95 ATR`
- Median time to peak: `3.5` bars
- Median volume decay after peak: `-0.94`


## Baldur / Freya Diagnostics
- Baldur top-warning precision: `0.00%`
- Baldur median delay to downside: `n/a`
- Baldur warnings are used as Thor exit triggers, not standalone shorts.
- Freya blocked by top-risk: `0`

## Tuning Search
- Trials logged: `803`
- Objective score: `4900.22`
- Method: `Optuna TPE + walk-forward CV + bootstrap CI + Numba exit fine-tune`
- Optuna trials: `800`

## Tuning — Walk-forward fold stats
| Fold | Trades | Calmar | Growth% | MaxDD% | PF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 8.10 | +26.60 | 3.29 | 8.255 |
| 1 | 3 | 1.23 | +1.95 | 1.58 | 2.252 |
| 2 | 8 | 5.36 | +8.87 | 1.66 | 5.683 |
| 3 | 43 | 33.59 | +103.65 | 3.09 | 10.169 |
| 4 | 8 | 2.35 | +3.76 | 1.60 | 2.585 |
| **median** | — | **5.36** | — | — | — |

## Tuning — Parameter sensitivity (±20%)
| Param | Value | −20% score | +20% score | Max drop | Brittle? |
| --- | --- | ---: | ---: | ---: | --- |
| thor_min_score_trade | 70.0 | 18.149 | -114.000 | 0.0% | no |
| thor_wave_strength_min | 50.0 | 3.195 | -114.000 | 0.0% | no |
| thor_top_risk_max | 45.0 | -0.593 | -0.593 | 0.0% | no |
| thor_mae_veto_atr | 6.6000000000000005 | -3.756 | -3.076 | 0.0% | no |
| thor_sl_atr | 2.4 | -3.617 | 0.376 | 0.0% | no |
| thor_trade_cooldown_bars | 96 | -0.569 | -0.593 | 0.0% | no |
| baldur_warning_exit_score | 62.5 | -1.440 | 0.690 | 0.0% | no |
| learned_filter_threshold | 0.3 | -0.593 | -0.593 | 0.0% | no |

## Tuning — Optuna study
- Trials completed: `800`
- Best score: `-0.1615`
- Best trial #: `626`

## Tuning — Learned filter (top features by |coef|)

**SL=0.8 ATR**  avg_fold_AUC=0.6023
- `wave_strength_score_at_entry` coef=+0.3676  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2248  ↓profit
- `nike_score` coef=-0.1681  ↓profit
- `weighted_trend` coef=+0.1506  ↑profit
- `bs_prob` coef=-0.0958  ↓profit

**SL=1.0 ATR**  avg_fold_AUC=0.6061
- `wave_strength_score_at_entry` coef=+0.3288  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2452  ↓profit
- `nike_score` coef=-0.2236  ↓profit
- `weighted_trend` coef=+0.1823  ↑profit
- `bs_prob` coef=-0.1115  ↓profit

**SL=1.2 ATR**  avg_fold_AUC=0.6079
- `wave_strength_score_at_entry` coef=+0.2940  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2383  ↓profit
- `nike_score` coef=-0.2207  ↓profit
- `weighted_trend` coef=+0.2170  ↑profit
- `bs_prob` coef=-0.1100  ↓profit

**SL=1.5 ATR**  avg_fold_AUC=0.6127
- `wave_strength_score_at_entry` coef=+0.2700  ↑profit
- `weighted_trend` coef=+0.2675  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2340  ↓profit
- `nike_score` coef=-0.2185  ↓profit
- `bs_prob` coef=-0.1150  ↓profit

**SL=1.8 ATR**  avg_fold_AUC=0.6124
- `wave_strength_score_at_entry` coef=+0.2833  ↑profit
- `weighted_trend` coef=+0.2690  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2296  ↓profit
- `nike_score` coef=-0.2079  ↓profit
- `bs_prob` coef=-0.1203  ↓profit

**SL=2.0 ATR**  avg_fold_AUC=0.6104
- `weighted_trend` coef=+0.2825  ↑profit
- `wave_strength_score_at_entry` coef=+0.2643  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2164  ↓profit
- `nike_score` coef=-0.2001  ↓profit
- `bs_prob` coef=-0.1273  ↓profit

**SL=2.4 ATR**  avg_fold_AUC=0.6150
- `wave_strength_score_at_entry` coef=+0.2886  ↑profit
- `weighted_trend` coef=+0.2818  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2145  ↓profit
- `nike_score` coef=-0.2086  ↓profit
- `bs_prob` coef=-0.1245  ↓profit

**SL=3.0 ATR**  avg_fold_AUC=0.6198
- `wave_strength_score_at_entry` coef=+0.3189  ↑profit
- `weighted_trend` coef=+0.2834  ↑profit
- `pre_impulse_r2_at_entry` coef=-0.2054  ↓profit
- `nike_score` coef=-0.1822  ↓profit
- `bs_prob` coef=-0.1170  ↓profit

## Tuning — Cache vs full-sim reconciliation  ⚠️ DIVERGENCE
| Metric | Cached estimate | Full sim | Delta |
| --- | ---: | ---: | ---: |
| final_capital | $31,404.68 | $16,200.44 | 48.4% |
| max_drawdown_pct | 9.79% | 48.98% | 39.19 pp |

## Output Files
- [`norse_year_paper_summary.csv`](C:/Users/habib/QUANTA/norse_year_paper_summary.csv)
- [`norse_year_paper_trades.csv`](C:/Users/habib/QUANTA/norse_year_paper_trades.csv)
- [`norse_pump_stats.csv`](C:/Users/habib/QUANTA/norse_pump_stats.csv)
- [`norse_pump_ledger.csv`](C:/Users/habib/QUANTA/norse_pump_ledger.csv)
- [`norse_year_tuning_search.csv`](C:/Users/habib/QUANTA/norse_year_tuning_search.csv)
