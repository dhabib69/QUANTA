# ZEUS.ai — LLM Autonomous Supervisor (PRIORITY ROADMAP)

*Status: PAUSED FOR API KEYS — CONFIG PREPARED*
*Created: 2026-04-03*

## Concept
A high-priority planned feature: An autonomous, model-agnostic AI supervisor (initially targeting Claude Sonnet 3.5/4.6, or alternatives like GLM 5.1 / GPT-4o) that evaluates all 7 specialist agents and the PPO Agent after every training cycle. It diagnoses performance issues using SHAP/Brier/AUC and RL outcome data, and auto-applies corrective hyperparameter, architectural, and reward-shaping changes.

## Key Decisions
- **Model Agnostic**: Configurable `AI_BASE_URL`, `AI_API_KEY`, and `AI_MODEL_NAME` (e.g., `claude-4-6-sonnet-2026xxxx` or `glm-4`).
- **API Key**: User will set credentials in config/env when ready to proceed.
- **Autonomy**: Full auto-apply, no human approval needed.
- **Scope**: Controls everything the CatBoost agents see (hyperparams, feature masks, regime routing) AND the PPO RL Agent (entropy, clip ratio, learning rate) to prevent policy collapse.
- **Cost**: ~$0.02 - $0.05 per evaluation, ~$1-2/month based on daily retraining.

## Proposed Architecture

1. **`quanta_zeus.py`** — Core ZEUS module handling Universal LLM API logic.
2. **`zeus_overrides.json`** — Per-agent and PPO override file written asynchronously by ZEUS.
3. **`zeus_audit_log.jsonl`** — Full traceability memory log for the dashboard and LLM context.

## Implementation Hooks
- **`QUANTA_ml_engine.py`**: 
  - Hook ZEUS execution *post-training* to evaluate new metrics.
- **`QUANTA_agents.py`**:
  - Add `apply_zeus_overrides()` to PPOAgent to hot-swap learning rate/entropy cleanly.
- **`quanta_config.py`**: Add `ZeusConfig` dataclass to centralize guardrails (DONE: 2026-04-03).
- **`templates/dashboard.html`**: Add ZEUS audit log section so the user can natively observe what the LLM chose to adjust.

## Known Vulnerabilities & Engineered Solutions (The "Bulletproof" Patches)

1. **Shape Mismatch (The Code Breaker)**
   - *Risk:* Deep learning layers (TFT/PPO) require exact input dimensions (278 features). If the CEO deletes a feature, it breaks the pipeline.
   - *Solution:* The LLM may only manipulate CatBoost's `feature_mask`. The underlying numpy arrays passed to TFT/PPO must automatically zero-pad deleted features to preserve their strict structural shape.

2. **Reward Hacking (Overfitting Trap)**
   - *Risk:* The LLM might learn that jacking depth to 10 and iterations to 3000 achieves 99% validation accuracy but creates a heavily overfit model that bleeds capital out-of-sample.
   - *Solution:* Constrain the CEO prompting to strictly penalize variance between `train_acc` and `val_acc`, forcing the LLM to seek robust generalization over naive accuracy.

3. **Infinite Prompt Bloating (Context Loss)**
   - *Risk:* Sending 278 SHAP feature importances x 7 agents equals thousands of numbers. Claude may drown in noise and hallucinate API responses.
   - *Solution:* Filter the array *prior* to API dispatch. Only pass the Top 15 highest-contributing features and Bottom 10 most useless features per agent to keep the JSON cleanly parsable.

## Hard Guardrails
Hard limits ZEUS cannot exceed during unmonitored evaluation (Already injected in `quanta_config.py` as of 2026-04-03):
- **CatBoost**: iterations [200, 3000], lr [0.01, 0.5], depth [4, 10], feature_mask [50, 278]
- **PPO RL**: lr [1e-6, 5e-3], entropy [0.001, 0.10], clip [0.05, 0.40], batch [64, 1024]
