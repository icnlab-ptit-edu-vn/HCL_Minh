# train.py
import os
import numpy as np
import torch
from env.wsn_env import WSNEnvironment
from rl.hppo import H_PPO
from config import *


def main():
    # === Xác định thiết bị huấn luyện ===
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Đang huấn luyện trên thiết bị: {DEVICE}")

    # === Khởi tạo môi trường và agent ===
    env = WSNEnvironment(n_nodes=N_NODES, area=AREA_SIZE, bs=BS_POS)
    agent = H_PPO(state_dim=10, gamma=GAMMA, device=DEVICE)
    episode_rewards = []

    # === Cấu hình Early Stopping ===
    patience = 50          # Số episode chờ nếu không cải thiện
    min_delta = 0.5        # Cải thiện tối thiểu để tính là "tốt hơn"
    best_reward = -np.inf
    patience_counter = 0

    # === Vòng lặp huấn luyện chính ===
    for ep in range(EPISODES):
        env.reset()
        env.cluster_and_build_chains(num_clusters=NUM_CLUSTERS)
        state = env.get_slim_state()
        done = False
        total_reward = 0
        step = 0
        trajectory = []

        while not done:
            # Lấy hành động từ agent (Discrete + Continuous)
            delta, phi, log_prob_d, log_prob_c = agent.select_action(state)

            # Nếu delta = 1, tái phân cụm
            if delta == 1:
                env.cluster_and_build_chains(num_clusters=NUM_CLUSTERS)

            # Truyền dữ liệu và nhận reward
            reward, done, info = env.transmit_data(phi=phi)
            next_state = env.get_slim_state()

            # Ghi lại trải nghiệm
            trajectory.append(
                (state, (delta, phi), (log_prob_d, log_prob_c), reward, next_state, done)
            )
            total_reward += reward
            state = next_state
            step += 1

            # Giới hạn tối đa 2000 bước / episode
            if step > 2000:
                done = True

        episode_rewards.append(total_reward)

        # === Cập nhật mô hình RL ===
        if trajectory:
            states, actions, log_probs, rewards, next_states, dones = zip(*trajectory)
            trajectories = {
                "states": list(states),
                "actions": [list(a) for a in actions],
                "log_probs": list(log_probs),
                "rewards": list(rewards),
                "next_states": list(next_states),
                "dones": list(dones),
            }
            agent.update(trajectories, epochs=10, batch_size=64)

        # === Logging mỗi 10 episode ===
        if (ep + 1) % 10 == 0 and len(episode_rewards) >= 50:
            avg_reward = np.mean(episode_rewards[-50:])
            print(
                f"Ep {ep+1:4d} | Avg Reward (last 50): {avg_reward:8.2f} | Alive: {info['alive_ratio']:.2f}"
            )

            # Kiểm tra điều kiện dừng sớm
            if avg_reward > best_reward + min_delta:
                best_reward = avg_reward
                patience_counter = 0
                torch.save(
                    {
                        "actor_d": agent.actor_d.state_dict(),
                        "actor_c": agent.actor_c.state_dict(),
                        "critic": agent.critic.state_dict(),
                        "episode": ep,
                        "reward": avg_reward,
                    },
                    os.path.join(MODEL_DIR, "rl_hcr_best.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered at episode {ep+1}!")
                print(f"   Best avg reward (last 50): {best_reward:.2f}")
                break

        # === Lưu model định kỳ ===
        if (ep + 1) % SAVE_INTERVAL == 0:
            torch.save(
                {
                    "actor_d": agent.actor_d.state_dict(),
                    "actor_c": agent.actor_c.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "episode": ep,
                    "reward": total_reward,
                },
                os.path.join(MODEL_DIR, f"rl_hcr_ep{ep+1}.pth"),
            )

    # === Lưu lịch sử reward ===
    np.save(os.path.join(MODEL_DIR, "episode_rewards.npy"), episode_rewards)
    print("\nHuấn luyện hoàn tất!")


if __name__ == "__main__":
    main()
