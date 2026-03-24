import numpy as np
import torch

from config import DISCRETE_ACTION_DIM, RL_STATE_DIM
from rl.hppo import H_PPO

if __name__ == "__main__":
    print("Testing H-PPO...")
    agent = H_PPO(RL_STATE_DIM, action_dim_discrete=DISCRETE_ACTION_DIM)
    state = np.random.randn(RL_STATE_DIM)
    action, phi, log_prob_d, log_prob_c = agent.select_action(state)
    print(f"Action: {action}, Phi: {phi:.3f}")
    print(f"Log prob d: {float(log_prob_d):.3f}, Log prob c: {float(log_prob_c):.3f}")

    trajectories = {
        'states': [np.random.randn(RL_STATE_DIM) for _ in range(100)],
        'actions': [(np.random.randint(0, DISCRETE_ACTION_DIM), np.random.rand()) for _ in range(100)],
        'log_probs': [(torch.tensor(0.0), torch.tensor(0.0)) for _ in range(100)],
        'rewards': np.random.randn(100).tolist(),
        'next_states': [np.random.randn(RL_STATE_DIM) for _ in range(100)],
        'dones': [False] * 99 + [True],
    }

    metrics = agent.update(trajectories, epochs=3, batch_size=32)
    print(f"Metrics: {metrics}")
    agent.save('test_hppo.pth')
    agent.load('test_hppo.pth')
    print("All tests passed!")
