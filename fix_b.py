"""Fix B, C, D for QUANTA_bot.py and related files."""
import re, json, os

# ─────────────────────────────────────────────
# FIX B: Unify HMM systems in QUANTA_bot.py
# ─────────────────────────────────────────────
with open('QUANTA_bot.py', 'rb') as f:
    bot = f.read().decode('utf-8')

errors = []

# B1: init _batch_hmm_regimes before the for loop
target = 'regime_mults_matrix = np.ones((len(active_keys), len(batch)))\r\n                                 for _b_idx, _item in enumerate(batch):'
repl   = 'regime_mults_matrix = np.ones((len(active_keys), len(batch)))\r\n                                 _batch_hmm_regimes = []  # FIX B: cache per-item regime for PPO state reuse\r\n                                 for _b_idx, _item in enumerate(batch):'
if target in bot:
    bot = bot.replace(target, repl, 1); print('B1 applied')
else:
    errors.append('B1 not found')

# B2: append regime to cache after regime_idx line
target = 'regime_idx = max(0, min(2, _hmm_regime_val))\r\n                                     for _k_idx, _k in enumerate(active_keys):'
repl   = 'regime_idx = max(0, min(2, _hmm_regime_val))\r\n                                     _batch_hmm_regimes.append(_hmm_regime_val)  # FIX B: store for PPO state\r\n                                     for _k_idx, _k in enumerate(active_keys):'
if target in bot:
    bot = bot.replace(target, repl, 1); print('B2 applied')
else:
    errors.append('B2 not found')

# B3: fallback list in except block
target = 'logging.debug(f"Regime-aware routing failed (falling back to entropy weights): {e}")'
repl   = '_batch_hmm_regimes = [1] * len(batch)  # FIX B: fallback range\r\n                                 logging.debug(f"Regime-aware routing failed (falling back to entropy weights): {e}")'
if target in bot:
    bot = bot.replace(target, repl, 1); print('B3 applied')
else:
    errors.append('B3 not found')

# B4: replace MoE HMM block with cached lookup
# Find Part E2 comment
e2_idx = bot.find("Part E2: Add HMM Regime to RL State")
if e2_idx == -1:
    errors.append('B4: Part E2 not found')
else:
    # Grab region from comment line through end of the if block
    region_start = bot.rfind('\n', 0, e2_idx) + 1   # start of that line
    # End after the except block closing 
    region_end = bot.find('\n\n', e2_idx)
    if region_end == -1:
        region_end = e2_idx + 700
    chunk = bot[region_start:region_end]
    print('B4 region:', repr(chunk[:300]))

    # Build replacement (match actual indentation = 29 spaces based on 'if rl_agent' block)
    pad = '                             '
    new_e2 = (pad + '# \U0001f525 Part E2: Add HMM Regime to RL State\r\n'
              + pad + '# FIX B: reuse per-symbol regime from routing block -- no second HMM call.\r\n'
              + pad + 'hmm_regime = 1  # default: range\r\n'
              + pad + 'try:\r\n'
              + pad + '    if \'_batch_hmm_regimes\' in locals() and idx < len(_batch_hmm_regimes):\r\n'
              + pad + '        hmm_regime = int(_batch_hmm_regimes[idx])\r\n'
              + pad + 'except Exception:\r\n'
              + pad + '    pass')

    # Replace from region_start to "\n\n" (blank line after the old if block)
    bot = bot[:region_start] + new_e2 + '\r\n\r\n' + bot[region_end+2:]
    print('B4 applied')

if errors:
    print('ERRORS:', errors)
else:
    print('All B fixes applied successfully')

with open('QUANTA_bot.py', 'wb') as f:
    f.write(bot.encode('utf-8'))
print(f'bot written ({len(bot)} chars)')
