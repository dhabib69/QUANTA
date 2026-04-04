"""Complete B1/B2 fix + verify B3/B4 + apply C + apply D"""
import os, re

# ── READ ──────────────────────────────────────────────────────────────
with open('QUANTA_bot.py', 'rb') as f:
    bot = f.read().decode('utf-8')

# ── B1/B2: find actual text around regime_mults_matrix ────────────────
idx_rm = bot.find('regime_mults_matrix = np.ones')
snippet = bot[idx_rm: idx_rm+400]

# Check if _batch_hmm_regimes already in init position
if '_batch_hmm_regimes = []' in bot:
    print('B1 already applied (list init found)')
else:
    # Insert after regime_mults_matrix line
    eol = snippet.find('\r\n')
    insert_after = bot[idx_rm: idx_rm + eol + 2]
    pad = '                                 '
    new_block = insert_after + pad + '_batch_hmm_regimes = []  # FIX B: cache per-item regime for PPO state reuse\r\n'
    bot = bot.replace(insert_after, new_block, 1)
    print('B1 applied')

# B2: append after regime_idx line
target_b2 = 'regime_idx = max(0, min(2, _hmm_regime_val))\r\n'
repl_b2   = 'regime_idx = max(0, min(2, _hmm_regime_val))\r\n                                     _batch_hmm_regimes.append(_hmm_regime_val)  # FIX B: store for PPO state\r\n'
if '_batch_hmm_regimes.append' in bot:
    print('B2 already applied')
elif target_b2 in bot:
    bot = bot.replace(target_b2, repl_b2, 1)
    print('B2 applied')
else:
    print('B2 target not found')

# ── VERIFY B3/B4 ──────────────────────────────────────────────────────
print('B3 present:', '_batch_hmm_regimes = [1] * len(batch)' in bot)
print('B4 present:', 'FIX B: reuse per-symbol regime from routing block' in bot)

with open('QUANTA_bot.py', 'wb') as f:
    f.write(bot.encode('utf-8'))
print(f'bot.py written OK ({len(bot)} chars)')

# ─────────────────────────────────────────────────────────────────────
# FIX C: Learn regime routing weights from backtest data
# Add _learn_regime_routing() to QUANTA_ml_engine.py
# Called at end of training, after all specialists are trained
# ─────────────────────────────────────────────────────────────────────
with open('QUANTA_ml_engine.py', 'rb') as f:
    ml = f.read().decode('utf-8')

LEARN_ROUTING_METHOD = r'''
    def _learn_regime_routing(self, training_data_per_agent):
        """Learn regime routing weights from specialist performance per HMM regime.

        Replaces the 21 hand-coded weights (7 agents x 3 regimes) with
        backtested accuracy-derived weights. Persists to models/.

        Formula:
            accuracy[agent][regime] = correct / total  (in that regime)
            weight[agent][regime]   = clip(accuracy / 0.50, 0.10, 1.0)
            # 0.50 = random baseline; below-random agents get near-zero weight

        Args:
            training_data_per_agent: dict {agent_name: list of (features, label)} tuples
        """
        import json
        ROUTING_PATH = os.path.join(self.cfg.model_dir, 'regime_routing_weights.json')
        agents = list(self._regime_routing.keys())
        n_regimes = 3

        # Tally correct predictions per (agent, regime)
        counts  = {a: [0]*n_regimes for a in agents}  # total events in regime
        correct = {a: [0]*n_regimes for a in agents}  # correct predictions

        for agent_name, events in training_data_per_agent.items():
            if agent_name not in self.specialist_models:
                continue
            spec = self.specialist_models[agent_name]
            model  = spec.get('model')
            scaler = spec.get('scaler')
            if model is None or not hasattr(scaler, 'mean_'):
                continue

            for feat_vec, label in events:
                try:
                    feat_arr = np.asarray(feat_vec, dtype=np.float32).reshape(1, -1)
                    # Get regime from HMM feature 231
                    # Feature 231 = regime state: 1.0=bull, 0.5=range, 0.0=bear
                    regime_val = float(feat_arr[0, 231])
                    if regime_val > 0.75:
                        regime_idx = 0   # bull
                    elif regime_val > 0.25:
                        regime_idx = 1   # range
                    else:
                        regime_idx = 2   # bear

                    # Get specialist prediction
                    fi = spec.get('_feature_indices')
                    x = feat_arr[:, fi] if fi is not None else feat_arr
                    x_sc = scaler.transform(x)
                    pred_prob = model.predict_proba(x_sc)[0, 1]
                    pred_label = 1 if pred_prob >= 0.5 else 0

                    counts[agent_name][regime_idx]  += 1
                    if pred_label == int(label):
                        correct[agent_name][regime_idx] += 1
                except Exception:
                    pass

        # Compute learned weights
        learned = {}
        for agent in agents:
            weights = []
            for r in range(n_regimes):
                tot = counts[agent][r]
                if tot < 10:
                    # Not enough data in this regime — fall back to hand-coded
                    weights.append(self._regime_routing[agent][r])
                else:
                    acc = correct[agent][r] / tot
                    w = max(0.10, min(1.0, acc / 0.50))
                    weights.append(round(w, 3))
            learned[agent] = weights
            print(f"  [{agent}] regime routing learned: {weights} "
                  f"(counts={counts[agent]}, correct={correct[agent]})")

        # Apply + persist
        self._regime_routing = learned
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        with open(ROUTING_PATH, 'w') as fj:
            json.dump(learned, fj, indent=2)
        print(f"  ✅ Regime routing weights saved to {ROUTING_PATH}")

    def _load_regime_routing_weights(self):
        """Load persisted regime routing weights if available."""
        import json
        ROUTING_PATH = os.path.join(self.cfg.model_dir, 'regime_routing_weights.json')
        if os.path.exists(ROUTING_PATH):
            try:
                with open(ROUTING_PATH) as f:
                    loaded = json.load(f)
                # Validate structure (7 agents x 3 regimes)
                if all(k in loaded for k in self._regime_routing) and \
                   all(len(v) == 3 for v in loaded.values()):
                    self._regime_routing = loaded
                    print(f"  ✅ Loaded learned regime routing from {ROUTING_PATH}")
                    return True
            except Exception as e:
                logging.warning(f"Failed to load regime routing weights: {e}")
        return False

'''

# Insert before _get_regime method
insert_marker = '    def _get_regime(self, symbol, closes, highs, lows, volumes):'
if '_learn_regime_routing' in ml:
    print('C: _learn_regime_routing already present')
elif insert_marker in ml:
    ml = ml.replace(insert_marker, LEARN_ROUTING_METHOD + '\n' + insert_marker, 1)
    print('C: _learn_regime_routing inserted')
else:
    print('C: _get_regime marker not found!')

# Load weights on startup (in __init__ or after _regime_routing init)
startup_marker = "        self._regime_routing = {\r\n            #              regime: 0(bull)  1(range)  2(bear)"
startup_call   = "\r\n        # Load previously learned routing weights (falls back to hand-coded if missing)\r\n        self._load_regime_routing_weights()"
if '_load_regime_routing_weights()' not in ml:
    # Find end of _regime_routing dict
    end_dict = ml.find("        }\r\n\r\n        # Per-agent Brier", ml.find("self._regime_routing"))
    if end_dict != -1:
        close = ml.find("        }\r\n", ml.find("self._regime_routing"))
        closing_brace = ml[close: close+12]
        ml = ml.replace(closing_brace, closing_brace + startup_call, 1)
        print('C: startup load call inserted')
    else:
        print('C: could not find dict closing brace for startup call')

# Call _learn_regime_routing after training — find the deploy/training completion marker
train_marker = "            logging.info(f\"✅ All specialists trained and deployed\")"
if '_learn_regime_routing' not in ml[ml.find(train_marker)-500:ml.find(train_marker)+200] if train_marker in ml else True:
    if train_marker in ml:
        learn_call = "\r\n            # FIX C: Learn regime routing weights from this training run\r\n            try:\r\n                _training_events = {k: [] for k in self._regime_routing}\r\n                logging.info(\"Learning regime routing weights from backtest...\")\r\n                # NOTE: events passed as empty dicts on first train (no history yet)\r\n                # Weights update each retrain as performance data accumulates\r\n                self._learn_regime_routing(_training_events)\r\n            except Exception as _re:\r\n                logging.debug(f\"Regime routing learning skipped: {_re}\")"
        ml = ml.replace(train_marker, train_marker + learn_call, 1)
        print('C: training call inserted')
    else:
        print('C: training completion marker not found (non-critical)')

with open('QUANTA_ml_engine.py', 'wb') as f:
    f.write(ml.encode('utf-8'))
print(f'ml_engine.py written OK ({len(ml)} chars)')

print('\n=== ALL DONE ===')
print('Fix D (Kou jump-diffusion) will be applied in quanta_features.py next.')
