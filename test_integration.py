from env import WSNEnvironment
from rl import H_PPO
from config import *

env = WSNEnvironment(n_nodes=N_NODES, area=AREA_SIZE, bs=BS_POS)
#agent = H_PPO(state_dim=10, device=DEVICE)
#Thiennd: fixed device
DEVICE = 'cuda'
agent = H_PPO(state_dim=10, device=DEVICE)
print('✅ Integration OK')
