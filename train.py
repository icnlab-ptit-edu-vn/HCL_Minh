import os
import numpy as np
import torch

from config import *
from env.wsn_env import WSNEnvironment
from rl.hppo import H_PPO


def build_env():
    return WSNEnvironment(
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


def build_agent(device):
    return H_PPO(
        state_dim=RL_STATE_DIM,
        action_dim_discrete=DISCRETE_ACTION_DIM,
        gamma=GAMMA,
        device=device,
    )


def save_checkpoint(path, agent, episode, reward):
    torch.save(
        {
            "actor_d": agent.actor_d.state_dict(),
            "critic": agent.critic.state_dict(),
            "opt_actor_d": agent.opt_actor_d.state_dict(),
            "opt_critic": agent.opt_critic.state_dict(),
            "training_step": agent.training_step,
            "episode": episode,
            "reward": reward,
            "config": {
                "state_dim": RL_STATE_DIM,
                "action_dim": DISCRETE_ACTION_DIM,
                "min_clusters": MIN_CLUSTERS,
                "max_clusters": MAX_CLUSTERS,
                "initial_clusters": INITIAL_CLUSTERS,
                "reward_alpha": REWARD_ALPHA,
                "reward_beta": REWARD_BETA,
                "reward_gamma": REWARD_GAMMA,
                "reward_delta": REWARD_DELTA,
            },
        },
        path,
    )


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")
    print(
        f"Episodes={EPISODES}, MaxRounds={MAX_ROUNDS_PER_EPISODE}, "
        f"StateDim={RL_STATE_DIM}, ActionDim={DISCRETE_ACTION_DIM}"
    )

    env = build_env()
    agent = build_agent(device)
    episode_rewards = []
    episode_reconfig_rates = []

    patience = 50
    min_delta = 0.05
    best_reward = -np.inf
    patience_counter = 0

    for ep in range(EPISODES):
        env.reset()
        env.cluster_and_build_chains(num_clusters=INITIAL_CLUSTERS, charge_overhead=False)
        state = env.get_slim_state()
        done = False
        total_reward = 0.0
        step = 0
        reconfig_events = 0
        trajectory = []
        last_info = {
            "alive_ratio": 1.0,
            "cluster_balance_cv": 0.0,
            "data_energy_consumption": 0.0,
            "reconfig_energy_consumption": 0.0,
        }

        agent.set_train_mode()
        while not done and step < MAX_ROUNDS_PER_EPISODE:
            topology_action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(
                topology_action=topology_action,
                min_clusters=MIN_CLUSTERS,
                max_clusters=MAX_CLUSTERS,
                default_clusters=INITIAL_CLUSTERS,
            )

            if info.get("recluster", False):
                reconfig_events += 1

            trajectory.append((state, topology_action, log_prob, reward, next_state, done))
            total_reward += reward
            state = next_state
            last_info = info
            step += 1

        episode_rewards.append(total_reward)
        episode_reconfig_rates.append(reconfig_events / max(step, 1))

        if trajectory:
            states, actions, log_probs, rewards, next_states, dones = zip(*trajectory)
            trajectories = {
                "states": list(states),
                "actions": list(actions),
                "log_probs": list(log_probs),
                "rewards": list(rewards),
                "next_states": list(next_states),
                "dones": list(dones),
            }
            metrics = agent.update(trajectories, epochs=PPO_EPOCHS, batch_size=PPO_BATCH_SIZE)
        else:
            metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        if (ep + 1) % 10 == 0 and len(episode_rewards) >= 10:
            avg_reward = float(np.mean(episode_rewards[-10:]))
            avg_reconfig = float(np.mean(episode_reconfig_rates[-10:]))
            print(
                f"Ep {ep + 1:4d} | AvgReward10 {avg_reward:8.3f} | "
                f"Alive {last_info.get('alive_ratio', 0.0):.2f} | "
                f"Imbalance {last_info.get('cluster_balance_cv', 0.0):.3f} | "
                f"DataE {last_info.get('data_energy_consumption', 0.0):.5f} | "
                f"CtrlE {last_info.get('reconfig_energy_consumption', 0.0):.5f} | "
                f"ReconfigRate {avg_reconfig:.3f} | "
                f"ActorLoss {metrics['actor_loss']:.4f} | CriticLoss {metrics['critic_loss']:.4f}"
            )

            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_counter = 0
                save_checkpoint(
                    os.path.join(MODEL_DIR, "minhht2026_best.pth"),
                    agent,
                    ep,
                    avg_reward,
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at episode {ep + 1}.")
                print(f"Best average reward (last 10 episodes): {best_reward:.3f}")
                break

        if (ep + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(
                os.path.join(MODEL_DIR, f"minhht2026_ep{ep + 1}.pth"),
                agent,
                ep,
                total_reward,
            )

    np.save(os.path.join(MODEL_DIR, "episode_rewards.npy"), np.asarray(episode_rewards, dtype=np.float32))
    np.save(
        os.path.join(MODEL_DIR, "episode_reconfig_rates.npy"),
        np.asarray(episode_reconfig_rates, dtype=np.float32),
    )
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
