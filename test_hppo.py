import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from rl.hppo import H_PPO

# ===== Testing =====
if __name__ == "__main__":
    print("Testing H-PPO...")
    
    # Khởi tạo
    state_dim = 10
    agent = H_PPO(state_dim, action_dim_discrete=6)
    
    # Test select_action
    state = np.random.randn(state_dim)
    delta, phi, log_prob_d, log_prob_c = agent.select_action(state)
    print(f"Delta: {delta}, Phi: {phi:.3f}")
    print(f"Log prob d: {log_prob_d:.3f}, Log prob c: {log_prob_c:.3f}")
    
    # Test update
    trajectories = {
        'states': [np.random.randn(state_dim) for _ in range(100)],
        'actions': [(np.random.randint(0, 6), np.random.rand()) for _ in range(100)],
        'log_probs': [(torch.tensor(0.0), torch.tensor(0.0)) for _ in range(100)],
        'rewards': np.random.randn(100).tolist(),
        'next_states': [np.random.randn(state_dim) for _ in range(100)],
        'dones': [False] * 99 + [True]
    }
    
    metrics = agent.update(trajectories, epochs=3, batch_size=32)
    print(f"Metrics: {metrics}")
    
    # Test save/load
    agent.save('test_hppo.pth')
    agent.load('test_hppo.pth')
    
    print("✅ All tests passed!")

# SỬ DỤNG
# Khởi tạo agent
agent = H_PPO(state_dim=10, action_dim_discrete=6)

# Thiennd: Fixed -> Bổ sung khai báo env từ dòng 338-341
from env import WSNEnvironment
env = WSNEnvironment(n_nodes=100, area=(100, 100), bs=(50, 50))
env.reset()
env.cluster_and_build_chains(num_clusters=5)

# Chọn action
state = env.get_slim_state()
delta, phi, log_prob_d, log_prob_c = agent.select_action(state)

# Thiennd: comment
# Collect trajectories
# trajectories = {
#     'states': [...],
#     'actions': [...],
#     'log_probs': [...],
#     'rewards': [...],
#     'next_states': [...],
#     'dones': [...]
# }

# Update agent
metrics = agent.update(trajectories, epochs=10, batch_size=64)

# Save/Load
agent.save('hppo_model.pth')
agent.load('hppo_model.pth')