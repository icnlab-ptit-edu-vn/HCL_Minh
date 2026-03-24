from config import *
from env import WSNEnvironment
from rl import H_PPO

DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
env = WSNEnvironment(
    n_nodes=N_NODES,
    area=AREA_SIZE,
    bs=BS_POS,
    initial_energy=INITIAL_ENERGY,
    control_uplink_bits=CONTROL_UPLINK_BITS,
    control_downlink_bits=CONTROL_DOWNLINK_BITS,
    max_cluster_hint=MAX_CLUSTERS,
    reward_alpha=REWARD_ALPHA,
    reward_beta=REWARD_BETA,
    reward_gamma=REWARD_GAMMA,
    reward_delta=REWARD_DELTA,
)
agent = H_PPO(state_dim=RL_STATE_DIM, action_dim_discrete=DISCRETE_ACTION_DIM, device=DEVICE)
print('Integration OK')
