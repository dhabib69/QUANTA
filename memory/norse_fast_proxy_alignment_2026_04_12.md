# Norse Fast Proxy Alignment (2026-04-12)

- Tightened the Norse fast tuning proxy so it no longer compounds accepted Thor trades as if they were all independent and unconstrained.
- Added shared fast Thor capital replay in `norse_tuner/fast_capital.py`.
- The fast replay now approximates the real Thor capital engine with:
  - free cash and reserved margin
  - leverage caps
  - Thor capital cap
  - max concurrent positions
  - one-open-position-per-symbol rule
  - entry and exit friction
  - asymmetric-target compounding thresholds from config
- Updated `norse_tuner/cached_evaluator.py` to use the shared fast replay plus MAE-cache exit timing (`exit_bar_sl_*`), entry price/ATR, and Baldur early-exit timing.
- Updated `norse_tuner/numba_fine_tune.py` so Stage 2 exit replay and full-sample fast estimates use the same fast capital model instead of pure ATR compounding.
- Verification completed:
  - `python -m py_compile norse_tuner/fast_capital.py norse_tuner/cached_evaluator.py norse_tuner/numba_fine_tune.py`
- Runtime note:
  - I did not run a full Norse tuning or year sim in this turn, so the post-patch cache-vs-full-sim delta is not yet measured.
