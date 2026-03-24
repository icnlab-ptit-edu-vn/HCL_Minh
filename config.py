import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


N_NODES = _env_int("N_NODES", 100)
AREA_SIZE = _env_int("AREA_SIZE", 100)
BS_POS = [150, 50]
INITIAL_ENERGY = _env_float("INITIAL_ENERGY", 0.5)

MIN_CLUSTERS = _env_int("MIN_CLUSTERS", 3)
MAX_CLUSTERS = _env_int("MAX_CLUSTERS", 8)
INITIAL_CLUSTERS = _env_int("INITIAL_CLUSTERS", 6)
PERIODIC_REFRESH_INTERVAL = _env_int("PERIODIC_REFRESH_INTERVAL", 10)

DISCRETE_ACTION_DIM = 1 + (MAX_CLUSTERS - MIN_CLUSTERS + 1)
RL_STATE_DIM = 13

CONTROL_UPLINK_BITS = _env_int("CONTROL_UPLINK_BITS", 200)
CONTROL_DOWNLINK_BITS = _env_int("CONTROL_DOWNLINK_BITS", 200)

REWARD_ALPHA = _env_float("REWARD_ALPHA", 1.0)
REWARD_BETA = _env_float("REWARD_BETA", 150.0)
REWARD_GAMMA = _env_float("REWARD_GAMMA", 200.0)
REWARD_DELTA = _env_float("REWARD_DELTA", 1.0)

GAMMA = _env_float("GAMMA", 0.99)
EPISODES = _env_int("EPISODES", 500)
MAX_ROUNDS_PER_EPISODE = _env_int("MAX_ROUNDS_PER_EPISODE", 3000)
SAVE_INTERVAL = _env_int("SAVE_INTERVAL", 10)
PPO_EPOCHS = _env_int("PPO_EPOCHS", 10)
PPO_BATCH_SIZE = _env_int("PPO_BATCH_SIZE", 64)

EVAL_NUM_RUNS = _env_int("EVAL_NUM_RUNS", 10)
EVAL_NUM_ROUNDS = _env_int("EVAL_NUM_ROUNDS", 3000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
