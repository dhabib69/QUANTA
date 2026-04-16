# Skeptic Report: Norse Year Paper Run 20260412_154336

## Verdict

The reported full-sim result is strong, but it does not yet support an audit-grade claim that Norse or Quanta is "proven best". The strongest defensible statement is narrower:

- The full simulator produced a `+1238.36%` one-year paper result on this run.
- That result came entirely from `Thor`; `Freya` and `Baldur` contributed `0` trades.
- The tuning stack is still selecting on a proxy that materially diverges from the real simulator.

## What I Verified

- `norse_year_paper_summary_20260412_154336.csv` matches the markdown headline:
  - Thor trades: `1997`
  - Thor net PnL: `$123,835.53`
  - Freya trades: `0`
  - Baldur trades: `0`
- `norse_year_paper_trades_20260412_154336.csv` sums to the same total net PnL:
  - total net PnL: `$123,835.53`
- The report's cache-vs-full-sim divergence is real and generated intentionally in code:
  - cached estimate: `$326,242.37`, `8.42%` max drawdown
  - full sim: `$133,835.53`, `26.63%` max drawdown
  - delta: `59.0%` capital gap and `18.21pp` drawdown gap

## Main Skeptical Findings

### 1. The optimizer is still choosing candidates on the wrong surface

The real simulator lives in `quanta_norse_year_sim.py::_evaluate_params(...)` and `_simulate_capital(...)`.

The fast tuner path in `norse_tuner/cached_evaluator.py` does not run the real capital engine. It:

- filters cached MAE rows
- selects a precomputed `realized_atr_sl_*` column
- compounds `pnl_frac = realized * (risk_pct / sl_atr)`
- builds equity by pure cumulative product

This approximation is explicit in code:

- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:927)
- [norse_tuner\cached_evaluator.py](C:\Users\habib\QUANTA\norse_tuner\cached_evaluator.py:321)

That fast path does not model the full simulator's capital behavior, including:

- actual entry notional sizing from current equity
- free-cash availability
- margin reservation and release over overlapping positions
- leverage caps
- per-agent capital caps
- concurrent position limits
- trade rejection when cash is unavailable

Those mechanics are part of the real simulator:

- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:642)
- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:700)
- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:718)

Conclusion: the full-sim result may be valid for this specific run, but the search process is still guided by a materially distorted proxy.

### 2. This is a Thor result, not a validated multi-agent portfolio

The output summary and trade artifacts show:

- Thor: `1997` trades
- Freya: `0`
- Baldur: `0`

So the current evidence supports "Thor had an exceptional paper year", not "the Norse portfolio architecture is fully validated".

### 3. PnL is highly concentrated

Using the trade CSV:

- top 10 trades contributed `$67,067.49` = `54.16%` of total net PnL
- top 20 trades contributed `$86,678.01` = `69.99%` of total net PnL

Top symbols by net PnL are also concentrated:

- `COSUSDT`: `$16,581.73`
- `DODOXUSDT`: `$12,370.68`
- `XPLUSDT`: `$11,427.31`

This does not disqualify the result, but it means the return profile is dependent on a relatively small set of outsized winners.

### 4. The strategy is brittle around important gates

The report's own sensitivity table flags:

- `thor_min_score_trade` as brittle
- `thor_mae_veto_atr` as brittle

That matches the concern from the tuning architecture: a narrow parameter ridge can still look great after a large Optuna search if the search objective is only an approximation of the real system.

### 5. The simulator is more realistic than the cache proxy, but still not fully audit-grade

The full sim does include meaningful realism:

- commission and slippage are charged at entry and exit
- margin is reserved and released
- leverage caps are enforced
- concurrent-position limits are enforced
- agent capital caps are enforced

Relevant code:

- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:718)
- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:720)
- [quanta_norse_year_sim.py](C:\Users\habib\QUANTA\quanta_norse_year_sim.py:733)

But the report still does not prove:

- point-in-time universe construction with no survivorship effects
- market impact realism beyond the configured friction model
- live latency and order-queue effects
- robustness after removing the very largest winners

## Bottom Line

This report is good enough to claim that the current Thor full-sim pipeline found a very strong one-year paper result.

It is not good enough to claim:

- "best bot in the world"
- "portfolio proven"
- "optimizer proven"

The single biggest credibility blocker is the unresolved proxy mismatch between the cache-driven tuner and the real simulator.

## Highest-Value Next Steps

1. Make the optimizer rank on a closer approximation to the real capital engine, or widen full-sim reranking enough that the proxy cannot dominate final selection.
2. Add a concentration stress section: remove top 5, top 10, and top 20 trades and recompute annual outcome.
3. Add a monthly equity / drawdown table and a point-in-time universe note.
4. Separate claims clearly:
   - Thor paper-sim claim
   - Norse portfolio claim
   - live-trading claim
