import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from .networks import ActorDiscrete, ActorContinuous, Critic

class H_PPO:
    """
    Hierarchical Proximal Policy Optimization (H-PPO)
    Kết hợp 2 actors:
    - ActorDiscrete: Chọn delta (số clusters)
    - ActorContinuous: Chọn phi (ngưỡng relay)
    """
    
    def __init__(self, state_dim, action_dim_discrete=6, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01, max_grad_norm=0.5,
                 device='cpu'):
        """
        Args:
            state_dim: Số chiều của state vector
            action_dim_discrete: Số lượng actions rời rạc (số clusters: 3-8)
            lr_actor: Learning rate cho actors
            lr_critic: Learning rate cho critic
            gamma: Discount factor
            epsilon: PPO clipping parameter
            c1: Coefficient cho value loss
            c2: Coefficient cho entropy bonus
            max_grad_norm: Gradient clipping threshold
            device: 'cpu' hoặc 'cuda'
        """
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor_d = ActorDiscrete(state_dim, action_dim=action_dim_discrete).to(self.device)
        self.actor_c = ActorContinuous(state_dim, action_dim=1).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Optimizers
        self.opt_actor_d = optim.Adam(self.actor_d.parameters(), lr=lr_actor)
        self.opt_actor_c = optim.Adam(self.actor_c.parameters(), lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training stats
        self.training_step = 0
        
    def select_action(self, state, deterministic=False):
        """
        Chọn action từ state
        
        Args:
            state: State vector (numpy array hoặc tensor)
            deterministic: Nếu True, chọn action tốt nhất (không sample)
        
        Returns:
            delta: Số clusters (int: 0-5 tương ứng 3-8 clusters)
            phi: Ngưỡng relay (float: 0-1)
            log_prob_d: Log probability của delta
            log_prob_c: Log probability của phi
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            # Discrete action (delta)
            probs = self.actor_d(state)
            dist_d = Categorical(probs)
            
            if deterministic:
                delta = torch.argmax(probs)
            else:
                delta = dist_d.sample()
            
            log_prob_d = dist_d.log_prob(delta)
            
            # Continuous action (phi)
            mean, std = self.actor_c(state)
            dist_c = Normal(mean, std)
            
            if deterministic:
                phi = mean
            else:
                phi = dist_c.sample()
            
            # Clamp phi vào [0, 1]
            phi = torch.clamp(phi, 0.0, 1.0)
            log_prob_c = dist_c.log_prob(phi)
        
        return (delta.item(), 
                phi.item(), 
                log_prob_d.cpu(), 
                log_prob_c.cpu())
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Tính Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Tensor (batch_size,)
            values: Tensor (batch_size,)
            next_values: Tensor (batch_size,)
            dones: Tensor (batch_size,)
        
        Returns:
            advantages: Tensor (batch_size,)
            returns: Tensor (batch_size,)
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros(batch_size, device=self.device)
        returns = torch.zeros(batch_size, device=self.device)
        
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            
            # GAE
            gae = delta + self.gamma * 0.95 * (1 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, trajectories, epochs=10, batch_size=64):
        """
        Cập nhật networks bằng PPO
        
        Args:
            trajectories: Dictionary chứa:
                - states: List of states
                - actions: List of (delta, phi) tuples
                - log_probs: List of (log_prob_d, log_prob_c) tuples
                - rewards: List of rewards
                - next_states: List of next states
                - dones: List of done flags
            epochs: Số epochs để update
            batch_size: Batch size cho mini-batch updates
        
        Returns:
            Dictionary chứa training metrics
        """
        # Chuyển đổi trajectories sang tensors
        states = torch.FloatTensor(np.array(trajectories['states'])).to(self.device)
        
        # Actions: [delta, phi]
        actions = torch.FloatTensor(np.array(trajectories['actions'])).to(self.device)
        deltas = actions[:, 0].long()
        phis = actions[:, 1]
        
        # Old log probs
        old_log_probs = trajectories['log_probs']
        old_log_prob_d = torch.stack([lp[0] for lp in old_log_probs]).to(self.device)
        old_log_prob_c = torch.stack([lp[1] for lp in old_log_probs]).to(self.device)
        
        rewards = torch.FloatTensor(trajectories['rewards']).to(self.device)
        next_states = torch.FloatTensor(np.array(trajectories['next_states'])).to(self.device)
        dones = torch.FloatTensor(trajectories['dones']).to(self.device)
        
        # Tính values và advantages
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Mini-batch updates
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                # Mini-batch data
                batch_states = states[batch_idx]
                batch_deltas = deltas[batch_idx]
                batch_phis = phis[batch_idx]
                batch_old_log_prob_d = old_log_prob_d[batch_idx]
                batch_old_log_prob_c = old_log_prob_c[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # ===== Critic Update =====
                values_pred = self.critic(batch_states)
                critic_loss = self.c1 * nn.MSELoss()(values_pred, batch_returns)
                
                self.opt_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()
                
                # ===== Actor Discrete Update =====
                probs = self.actor_d(batch_states)
                dist_d = Categorical(probs)
                new_log_prob_d = dist_d.log_prob(batch_deltas)
                entropy_d = dist_d.entropy()
                
                ratio_d = (new_log_prob_d - batch_old_log_prob_d).exp()
                surr1_d = ratio_d * batch_advantages
                surr2_d = torch.clamp(ratio_d, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss_d = -torch.min(surr1_d, surr2_d).mean() - self.c2 * entropy_d.mean()
                
                self.opt_actor_d.zero_grad()
                actor_loss_d.backward()
                nn.utils.clip_grad_norm_(self.actor_d.parameters(), self.max_grad_norm)
                self.opt_actor_d.step()
                
                # ===== Actor Continuous Update =====
                mean, std = self.actor_c(batch_states)
                dist_c = Normal(mean, std)
                new_log_prob_c = dist_c.log_prob(batch_phis)
                entropy_c = dist_c.entropy()
                
                ratio_c = (new_log_prob_c - batch_old_log_prob_c).exp()
                surr1_c = ratio_c * batch_advantages
                surr2_c = torch.clamp(ratio_c, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss_c = -torch.min(surr1_c, surr2_c).mean() - self.c2 * entropy_c.mean()
                
                self.opt_actor_c.zero_grad()
                actor_loss_c.backward()
                nn.utils.clip_grad_norm_(self.actor_c.parameters(), self.max_grad_norm)
                self.opt_actor_c.step()
                
                # Accumulate metrics
                total_actor_loss += (actor_loss_d.item() + actor_loss_c.item())
                total_critic_loss += critic_loss.item()
                total_entropy += (entropy_d.mean().item() + entropy_c.mean().item())
        
        self.training_step += 1
        
        num_updates = epochs * (dataset_size // batch_size)
        
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'training_step': self.training_step
        }
    
    def save(self, path):
        """Lưu models"""
        torch.save({
            'actor_d': self.actor_d.state_dict(),
            'actor_c': self.actor_c.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor_d': self.opt_actor_d.state_dict(),
            'opt_actor_c': self.opt_actor_c.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"✅ Model saved to {path}")
    
    # def load(self, path):
    #     """Load models"""
    #     checkpoint = torch.load(path, map_location=self.device)
    #     self.actor_d.load_state_dict(checkpoint['actor_d'])
    #     self.actor_c.load_state_dict(checkpoint['actor_c'])
    #     self.critic.load_state_dict(checkpoint['critic'])
    #     self.opt_actor_d.load_state_dict(checkpoint['opt_actor_d'])
    #     self.opt_actor_c.load_state_dict(checkpoint['opt_actor_c'])
    #     self.opt_critic.load_state_dict(checkpoint['opt_critic'])
    #     self.training_step = checkpoint['training_step']
    #     print(f"✅ Model loaded from {path}")
    
    def load(self, path):
        """Load models (tương thích PyTorch >=2.6 và <2.6)"""
        try:
            # PyTorch >= 2.6: explicit weights_only arg
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch < 2.6: weights_only arg not supported
            checkpoint = torch.load(path, map_location=self.device)

        # Sau khi load, khôi phục states
        # Lưu ý: checkpoint có thể chứa optimizer states; chỉ load keys có trong checkpoint
        if 'actor_d' in checkpoint:
            self.actor_d.load_state_dict(checkpoint['actor_d'])
        if 'actor_c' in checkpoint:
            self.actor_c.load_state_dict(checkpoint['actor_c'])
        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
        # optional: optimizer state if có trong checkpoint
        if 'opt_actor_d' in checkpoint and hasattr(self, 'opt_actor_d'):
            try:
                self.opt_actor_d.load_state_dict(checkpoint['opt_actor_d'])
            except Exception:
                pass
        if 'opt_actor_c' in checkpoint and hasattr(self, 'opt_actor_c'):
            try:
                self.opt_actor_c.load_state_dict(checkpoint['opt_actor_c'])
            except Exception:
                pass
        if 'opt_critic' in checkpoint and hasattr(self, 'opt_critic'):
            try:
                self.opt_critic.load_state_dict(checkpoint['opt_critic'])
            except Exception:
                pass

        # restore training_step if present
        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']
        print(f"✅ Model loaded from {path}")



    def set_train_mode(self):
        """Đặt networks vào training mode"""
        self.actor_d.train()
        self.actor_c.train()
        self.critic.train()
    
    def set_eval_mode(self):
        """Đặt networks vào evaluation mode"""
        self.actor_d.eval()
        self.actor_c.eval()
        self.critic.eval()
