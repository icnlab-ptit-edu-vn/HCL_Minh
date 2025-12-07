import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorDiscrete(nn.Module):
    """
    Actor network cho action space rời rạc
    Output: Phân phối xác suất trên các actions
    """
    def __init__(self, state_dim, action_dim=2, hidden=256, dropout=0.1):
        """
        Args:
            state_dim: Số chiều của state vector
            action_dim: Số lượng actions rời rạc (mặc định 2)
            hidden: Số neurons trong hidden layers
            dropout: Tỷ lệ dropout để tránh overfitting
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),  # Thêm normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, action_dim)
        )
        
        # Khởi tạo weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Khởi tạo weights theo Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: State tensor shape (batch_size, state_dim)
        
        Returns:
            Action probabilities shape (batch_size, action_dim)
        """
        logits = self.net(x)
        # Thêm temperature scaling để điều chỉnh exploration
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state, deterministic=False):
        """
        Lấy action từ state
        
        Args:
            state: State tensor hoặc numpy array
            deterministic: Nếu True, chọn action có xác suất cao nhất
        
        Returns:
            action: Action được chọn
            log_prob: Log probability của action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.forward(state)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()


class ActorContinuous(nn.Module):
    """
    Actor network cho action space liên tục
    Output: Mean và std của Gaussian distribution
    """
    def __init__(self, state_dim, action_dim=1, hidden=256, dropout=0.1, 
                 log_std_min=-20, log_std_max=2):
        """
        Args:
            state_dim: Số chiều của state vector
            action_dim: Số chiều của action (mặc định 1 cho phi)
            hidden: Số neurons trong hidden layers
            dropout: Tỷ lệ dropout
            log_std_min: Giới hạn dưới của log(std)
            log_std_max: Giới hạn trên của log(std)
        """
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads cho mean và log_std
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        
        # Khởi tạo weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Khởi tạo weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: State tensor shape (batch_size, state_dim)
        
        Returns:
            mean: Mean của Gaussian distribution (batch_size, action_dim)
            std: Standard deviation (batch_size, action_dim)
        """
        h = self.net(x)
        
        # Mean: ánh xạ vào [0, 1] bằng sigmoid (cho phi ∈ [0, 1])
        mean = torch.sigmoid(self.mean_head(h))
        
        # Log std: clip để tránh quá lớn hoặc quá nhỏ
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean.squeeze(-1), std.squeeze(-1)
    
    def get_action(self, state, deterministic=False):
        """
        Lấy action từ state
        
        Args:
            state: State tensor hoặc numpy array
            deterministic: Nếu True, trả về mean (không sample)
        
        Returns:
            action: Action được chọn (clipped vào [0, 1])
            log_prob: Log probability của action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.forward(state)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros_like(action)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Clip action vào [0, 1]
                action = torch.clamp(action, 0.0, 1.0)
        
        return action.item(), log_prob.item()
    
    def evaluate_actions(self, state, action):
        """
        Đánh giá log probability và entropy của actions
        (Dùng cho training)
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
        
        Returns:
            log_prob: Log probability (batch_size,)
            entropy: Entropy của distribution (batch_size,)
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class Critic(nn.Module):
    """
    Critic network (Value function)
    Output: Giá trị V(s) của state
    """
    def __init__(self, state_dim, hidden=256, dropout=0.1):
        """
        Args:
            state_dim: Số chiều của state vector
            hidden: Số neurons trong hidden layers
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, 1)
        )
        
        # Khởi tạo weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Khởi tạo weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: State tensor shape (batch_size, state_dim)
        
        Returns:
            value: State value V(s) shape (batch_size,)
        """
        return self.net(x).squeeze(-1)


class DuelingCritic(nn.Module):
    """
    Dueling Critic network (cho DQN-style algorithms)
    Tách Value function và Advantage function
    """
    def __init__(self, state_dim, action_dim, hidden=256, dropout=0.1):
        """
        Args:
            state_dim: Số chiều của state vector
            action_dim: Số lượng actions
            hidden: Số neurons trong hidden layers
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: State tensor (batch_size, state_dim)
        
        Returns:
            Q-values: (batch_size, action_dim)
        """
        features = self.feature(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class EnsembleCritic(nn.Module):
    """
    Ensemble của nhiều Critics để giảm overestimation (dùng cho SAC, TD3)
    """
    def __init__(self, state_dim, num_critics=2, hidden=256, dropout=0.1):
        """
        Args:
            state_dim: Số chiều của state vector
            num_critics: Số lượng critics trong ensemble
            hidden: Số neurons trong hidden layers
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        
        self.critics = nn.ModuleList([
            Critic(state_dim, hidden, dropout) 
            for _ in range(num_critics)
        ])
    
    def forward(self, x):
        """
        Returns:
            List of values from each critic
        """
        return [critic(x) for critic in self.critics]
    
    def min_value(self, x):
        """
        Returns:
            Minimum value across all critics (conservative estimate)
        """
        values = self.forward(x)
        return torch.min(torch.stack(values), dim=0)[0]


# ===== Utility Functions =====

def init_weights(m):
    """Helper function để khởi tạo weights"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def count_parameters(model):
    """Đếm số parameters trong model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    """Lưu model"""
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    """Load model"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ===== Testing =====
if __name__ == "__main__":
    # Test networks
    state_dim = 10
    batch_size = 32
    
    print("="*50)
    print("Testing ActorDiscrete...")
    actor_discrete = ActorDiscrete(state_dim, action_dim=6)
    state = torch.randn(batch_size, state_dim)
    probs = actor_discrete(state)
    print(f"Input shape: {state.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Parameters: {count_parameters(actor_discrete):,}")
    
    print("\n" + "="*50)
    print("Testing ActorContinuous...")
    actor_continuous = ActorContinuous(state_dim)
    mean, std = actor_continuous(state)
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Parameters: {count_parameters(actor_continuous):,}")
    
    print("\n" + "="*50)
    print("Testing Critic...")
    critic = Critic(state_dim)
    value = critic(state)
    print(f"Value shape: {value.shape}")
    print(f"Parameters: {count_parameters(critic):,}")
    
    print("\n" + "="*50)
    print("Testing DuelingCritic...")
    dueling = DuelingCritic(state_dim, action_dim=6)
    q_values = dueling(state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Parameters: {count_parameters(dueling):,}")
    
    print("\n" + "="*50)
    print("Testing EnsembleCritic...")
    ensemble = EnsembleCritic(state_dim, num_critics=3)
    values = ensemble(state)
    min_val = ensemble.min_value(state)
    print(f"Number of critics: {len(values)}")
    print(f"Min value shape: {min_val.shape}")
    print(f"Parameters: {count_parameters(ensemble):,}")
    
    print("\n" + "="*50)
    print("✅ All tests passed!")

# CÁCH DÙNG
# Khởi tạo networks
state_dim = 10
actor = ActorContinuous(state_dim)
critic = Critic(state_dim)

# Forward pass
state = torch.randn(32, state_dim)
mean, std = actor(state)
value = critic(state)

# Get action
action, log_prob = actor.get_action(state[0].numpy(), deterministic=False)

# Evaluate actions (for training)
actions = torch.randn(32, 1)
log_probs, entropy = actor.evaluate_actions(state, actions)

# Save/Load
save_model(actor, 'actor.pth')
actor = load_model(actor, 'actor.pth')
