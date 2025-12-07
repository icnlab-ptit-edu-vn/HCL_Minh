import torch
from rl import H_PPO
import numpy as np

agent = H_PPO(state_dim=10, device='cpu')

# Test select_action
state = torch.randn(10)
delta, phi, log_prob_d, log_prob_c = agent.select_action(state)

print(f"✅ Delta: {delta} (should be 0-5)")
print(f"✅ Phi: {phi:.4f} (should be 0-1)")
print(f"✅ Log prob d: {log_prob_d:.4f}")
print(f"✅ Log prob c: {log_prob_c:.4f}")

# Thiennd
# Test update
# trajectories = {
#     'states': [torch.randn(10) for _ in range(100)],
#     'actions': [[0, 0.5] for _ in range(100)],
#     'log_probs': [(0.0, 0.0) for _ in range(100)],
#     'rewards': [1.0 for _ in range(100)],
#     'next_states': [torch.randn(10) for _ in range(100)],
#     'dones': [False for _ in range(100)]
# }

# Thiennd
# Khởi tạo
state_dim = 10
agent = H_PPO(state_dim, action_dim_discrete=6)
trajectories = {
    'states': [np.random.randn(state_dim) for _ in range(100)],
    'actions': [(np.random.randint(0, 6), np.random.rand()) for _ in range(100)],
    'log_probs': [(torch.tensor(0.0), torch.tensor(0.0)) for _ in range(100)],
    'rewards': np.random.randn(100).tolist(),
    'next_states': [np.random.randn(state_dim) for _ in range(100)],
    'dones': [False] * 99 + [True]
}

metrics = agent.update(trajectories, epochs=1, batch_size=32)
print(f"✅ Metrics: {metrics}")

# Test save/load
agent.save('test_model.pth')
agent.load('test_model.pth')
print("✅ Save/Load successful")