from config import *
from env import WSNEnvironment

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
env.reset()
env.cluster_and_build_chains(num_clusters=INITIAL_CLUSTERS, charge_overhead=False)
state = env.get_slim_state()
print(state.shape)
next_state, reward, done, info = env.step(
    topology_action=0,
    min_clusters=MIN_CLUSTERS,
    max_clusters=MAX_CLUSTERS,
    default_clusters=INITIAL_CLUSTERS,
)
print(next_state.shape, reward, done, info["alive_count"])
