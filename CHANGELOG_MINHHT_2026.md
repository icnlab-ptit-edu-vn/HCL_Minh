# MinhHT_2026 alignment patch

This patch makes `Minh_2026` the source of truth for the codebase.

Main changes:
- Removed the continuous relay-threshold action `phi` from training and evaluation.
- Reduced the RL state from 14 dimensions to 13 dimensions to match the paper state in Eq. (8).
- Replaced the previous heuristic reward with the paper-aligned reward in Eq. (10):
  `r_t = alpha * alive_ratio - beta * normalized_data_energy - gamma * normalized_control_energy - delta * imbalance`.
- Standardized the action semantics:
  - `0`: keep current topology
  - `1..K`: rebuild topology with a selected cluster count
- Standardized the main baselines in `test3.py`:
  - `ADAPTIVE`
  - `STATIC`
  - `PERIODIC`
  - `ALWAYS_REFRESH`
- Added environment-variable overrides in `config.py` for controlled smoke runs and larger experiments.

Smoke run performed inside the patched repository:
- Training: `EPISODES=1`, `MAX_ROUNDS_PER_EPISODE=10`, checkpoint saved as `models/minhht2026_ep1.pth`
- Evaluation: `EVAL_NUM_RUNS=1`, `EVAL_NUM_ROUNDS=20`, results saved under `results/minhht2026_eval_20260324_045835`

Recommended next step:
Run a full multi-seed experiment campaign with larger episode and round budgets before filling paper tables.
