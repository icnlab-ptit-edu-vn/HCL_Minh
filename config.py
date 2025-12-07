import os
# Sensor Network
N_NODES = 100          # Number of sensor nodes in the network, default 100
AREA_SIZE = 100       # Size of the deployment area (100x100 units), default 100
BS_POS = [150, 50]     # Base Station position coordinates (x=150, y=50)
# RL (Reinforcement Learning)
NUM_CLUSTERS = 6       # Number of clusters for grouping sensor nodes, default 6
GAMMA = 0.99          # Discount factor for future rewards in RL algorithm, default 0.99
EPISODES = 500        # Number of training episodes, default 500
SAVE_INTERVAL = 10   # Interval for saving model checkpoints (every 100 episodes), default 100
# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of current file's directory
MODEL_DIR = os.path.join(BASE_DIR, "models")           # Create path for models directory
os.makedirs(MODEL_DIR, exist_ok=True)                  # Create models directory if it doesn't exist