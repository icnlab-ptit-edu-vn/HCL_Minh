from env import WSNEnvironment

env = WSNEnvironment(n_nodes=100, area=(100, 100), bs=(50, 50))
env.reset()
env.cluster_and_build_chains(num_clusters=5)
state = env.get_slim_state()

print(f"✅ State shape: {state.shape}")  # Should be (10,)
print(f"✅ State: {state}")

reward, done, info = env.transmit_data(phi=0.5)
print(f"✅ Reward: {reward}")
print(f"✅ Done: {done}")
print(f"✅ Info: {info}")