import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .networks import ActorDiscrete, Critic


class H_PPO:
    """Discrete PPO controller aligned with Minh_2026."""

    def __init__(
        self,
        state_dim,
        action_dim_discrete=6,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        c1=0.5,
        c2=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.gae_lambda = float(gae_lambda)
        self.max_grad_norm = float(max_grad_norm)

        self.actor_d = ActorDiscrete(state_dim, action_dim=action_dim_discrete).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.opt_actor_d = optim.Adam(self.actor_d.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.training_step = 0

    def select_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            probs = self.actor_d(state)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.detach().cpu()

    def compute_gae(self, rewards, values, next_values, dones):
        batch_size = rewards.size(0)
        advantages = torch.zeros(batch_size, device=self.device)
        returns = torch.zeros(batch_size, device=self.device)
        gae = 0.0
        for t in reversed(range(batch_size)):
            next_value = next_values[t] if t == batch_size - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def update(self, trajectories, epochs=10, batch_size=64):
        states = torch.as_tensor(np.array(trajectories["states"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(trajectories["actions"]), dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([lp for lp in trajectories["log_probs"]]).to(self.device)
        rewards = torch.as_tensor(trajectories["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(trajectories["next_states"]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(trajectories["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                values_pred = self.critic(batch_states)
                critic_loss = self.c1 * nn.MSELoss()(values_pred, batch_returns)
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                probs = self.actor_d(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.c2 * entropy.mean()

                self.opt_actor_d.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_d.parameters(), self.max_grad_norm)
                self.opt_actor_d.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.mean().item())

        self.training_step += 1
        num_updates = max(1, epochs * int(np.ceil(dataset_size / batch_size)))
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "training_step": self.training_step,
        }

    def save(self, path):
        torch.save(
            {
                "actor_d": self.actor_d.state_dict(),
                "critic": self.critic.state_dict(),
                "opt_actor_d": self.opt_actor_d.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if "actor_d" in checkpoint:
            self.actor_d.load_state_dict(checkpoint["actor_d"])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        if "opt_actor_d" in checkpoint:
            try:
                self.opt_actor_d.load_state_dict(checkpoint["opt_actor_d"])
            except Exception:
                pass
        if "opt_critic" in checkpoint:
            try:
                self.opt_critic.load_state_dict(checkpoint["opt_critic"])
            except Exception:
                pass
        if "training_step" in checkpoint:
            self.training_step = int(checkpoint["training_step"])

    def set_train_mode(self):
        self.actor_d.train()
        self.critic.train()

    def set_eval_mode(self):
        self.actor_d.eval()
        self.critic.eval()
