 #!/usr/bin/env python3
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import *
from env.wsn_env import WSNEnvironment
from rl.hppo import H_PPO

NUM_RUNS = EVAL_NUM_RUNS
NUM_ROUNDS = EVAL_NUM_ROUNDS


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


def load_checkpoint_if_available(agent):
    candidates = [
        os.path.join(MODEL_DIR, "minhht2026_best.pth"),
        os.path.join(MODEL_DIR, f"minhht2026_ep{EPISODES}.pth"),
        os.path.join(MODEL_DIR, "rl_hcr_best.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                agent.load(path)
                return path
            except Exception:
                pass
    return None


def compute_jain(values):
    a = np.asarray(values, dtype=np.float32)
    a = a[~np.isnan(a)]
    if a.size <= 1:
        return np.nan
    denom = a.size * np.sum(a ** 2)
    if denom <= 0:
        return np.nan
    return float((np.sum(a) ** 2) / denom)


def milestones_from_alive(alive_history, initial_nodes):
    fnd = hnd = lnd = None
    for idx, alive_count in enumerate(alive_history):
        if fnd is None and alive_count < initial_nodes:
            fnd = idx + 1
        if hnd is None and alive_count <= 0.5 * initial_nodes:
            hnd = idx + 1
        if lnd is None and alive_count == 0:
            lnd = idx + 1
    length = len(alive_history)
    return {
        "fnd": length if fnd is None else fnd,
        "hnd": length if hnd is None else hnd,
        "lnd": length if lnd is None else lnd,
    }


def finalize_result(env, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history):
    milestones = milestones_from_alive(alive_history, env.n)
    total_energy = np.asarray(data_energy_history, dtype=np.float32) + np.asarray(control_energy_history, dtype=np.float32)
    return {
        **milestones,
        "alive_history": alive_history,
        "data_energy_history": data_energy_history,
        "control_energy_history": control_energy_history,
        "fairness_history": fairness_history,
        "cluster_history": cluster_history,
        "total_data_energy": float(np.nansum(data_energy_history)),
        "total_control_energy": float(np.nansum(control_energy_history)),
        "total_energy": float(np.nansum(total_energy)),
        "recluster_count": int(np.sum(np.asarray(control_energy_history) > 0.0)),
        "rounds": len(alive_history),
    }


def record_step(env, info, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history):
    alive_history.append(int(info["alive_count"]))
    data_energy_history.append(float(info["data_energy_consumption"]))
    control_energy_history.append(float(info["reconfig_energy_consumption"]))
    cluster_history.append(int(info.get("current_clusters", env.current_num_clusters)))
    fairness_history.append(compute_jain(env.energy[env.alive]))


def run_one_adaptive(env, agent, deterministic=True, max_rounds=NUM_ROUNDS):
    env.reset()
    env.cluster_and_build_chains(num_clusters=INITIAL_CLUSTERS, charge_overhead=False)
    agent.set_eval_mode()

    alive_history, data_energy_history, control_energy_history = [], [], []
    fairness_history, cluster_history = [], []

    for _ in range(max_rounds):
        state = env.get_slim_state()
        topology_action, _ = agent.select_action(state, deterministic=deterministic)
        _, _, done, info = env.step(
            topology_action=topology_action,
            min_clusters=MIN_CLUSTERS,
            max_clusters=MAX_CLUSTERS,
            default_clusters=INITIAL_CLUSTERS,
        )
        record_step(env, info, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history)
        if done:
            break

    return finalize_result(env, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history)


def run_rule_baseline(env, refresh_interval, max_rounds=NUM_ROUNDS):
    env.reset()
    env.cluster_and_build_chains(num_clusters=INITIAL_CLUSTERS, charge_overhead=False)

    alive_history, data_energy_history, control_energy_history = [], [], []
    fairness_history, cluster_history = [], []

    for round_idx in range(max_rounds):
        reconfigured = False
        if refresh_interval is not None and round_idx % refresh_interval == 0:
            env.cluster_and_build_chains(num_clusters=INITIAL_CLUSTERS, charge_overhead=True)
            env.last_recluster_flag = 1.0
            reconfigured = True
        else:
            env.last_recluster_overhead = 0.0
            env.last_control_energy = 0.0
            env.last_recluster_flag = 0.0

        if not reconfigured:
            env.topology_age += 1
        _, done, info = env.transmit_data()
        env._update_topology_descriptors()
        record_step(env, {**info, "current_clusters": env.current_num_clusters}, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history)
        if done:
            break

    return finalize_result(env, alive_history, data_energy_history, control_energy_history, fairness_history, cluster_history)


def pad_with_nan(histories):
    max_len = max(len(h) for h in histories)
    arr = np.full((len(histories), max_len), np.nan, dtype=np.float32)
    for i, history in enumerate(histories):
        arr[i, : len(history)] = np.asarray(history, dtype=np.float32)
    return arr


def summarize(results):
    summary = {}
    for method, runs in results.items():
        summary[method] = {
            "fnd_mean": float(np.mean([r["fnd"] for r in runs])),
            "hnd_mean": float(np.mean([r["hnd"] for r in runs])),
            "lnd_mean": float(np.mean([r["lnd"] for r in runs])),
            "data_energy_mean": float(np.mean([r["total_data_energy"] for r in runs])),
            "control_energy_mean": float(np.mean([r["total_control_energy"] for r in runs])),
            "total_energy_mean": float(np.mean([r["total_energy"] for r in runs])),
            "recluster_count_mean": float(np.mean([r["recluster_count"] for r in runs])),
            "rounds_mean": float(np.mean([r["rounds"] for r in runs])),
        }
    return summary


def plot_results(results, output_dir):
    methods = list(results.keys())
    alive_curves = {m: pad_with_nan([r["alive_history"] for r in runs]) for m, runs in results.items()}
    fairness_curves = {m: pad_with_nan([r["fairness_history"] for r in runs]) for m, runs in results.items()}
    data_curves = {m: pad_with_nan([r["data_energy_history"] for r in runs]) for m, runs in results.items()}
    control_curves = {m: pad_with_nan([r["control_energy_history"] for r in runs]) for m, runs in results.items()}

    plt.figure(figsize=(12, 8))
    for method in methods:
        arr = alive_curves[method]
        x = np.arange(arr.shape[1])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel("Alive nodes")
    plt.title("Alive-node trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alive_nodes.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 8))
    for method in methods:
        arr = fairness_curves[method]
        x = np.arange(arr.shape[1])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Round")
    plt.ylabel("Jain fairness")
    plt.title("Residual-energy fairness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 8))
    for method in methods:
        arr = data_curves[method]
        x = np.arange(arr.shape[1])
        plt.plot(x, np.nancumsum(np.nanmean(arr, axis=0)), label=method)
    plt.xlabel("Round")
    plt.ylabel("Cumulative data energy (J)")
    plt.title("Data-communication energy accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_energy.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 8))
    for method in methods:
        arr = control_curves[method]
        x = np.arange(arr.shape[1])
        plt.plot(x, np.nancumsum(np.nanmean(arr, axis=0)), label=method)
    plt.xlabel("Round")
    plt.ylabel("Cumulative control energy (J)")
    plt.title("Reconfiguration-overhead accumulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "control_energy.png"), dpi=200)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    agent = build_agent(device)
    checkpoint = load_checkpoint_if_available(agent)
    if checkpoint:
        print(f"Loaded checkpoint: {checkpoint}")

    methods = {"ADAPTIVE": [], "STATIC": [], "PERIODIC": [], "ALWAYS_REFRESH": []}
    for _ in tqdm(range(NUM_RUNS), desc="Evaluation runs"):
        methods["ADAPTIVE"].append(run_one_adaptive(build_env(), agent, deterministic=True))
        methods["STATIC"].append(run_rule_baseline(build_env(), refresh_interval=None))
        methods["PERIODIC"].append(run_rule_baseline(build_env(), refresh_interval=PERIODIC_REFRESH_INTERVAL))
        methods["ALWAYS_REFRESH"].append(run_rule_baseline(build_env(), refresh_interval=1))

    summary = summarize(methods)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_DIR, f"minhht2026_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "results.npz"),
        raw_results=methods,
        summary=summary,
        config={
            "num_runs": NUM_RUNS,
            "num_rounds": NUM_ROUNDS,
            "periodic_refresh_interval": PERIODIC_REFRESH_INTERVAL,
            "min_clusters": MIN_CLUSTERS,
            "max_clusters": MAX_CLUSTERS,
            "initial_clusters": INITIAL_CLUSTERS,
        },
    )

    plot_results(methods, output_dir)

    print("\n=== Summary ===")
    for method, stats in summary.items():
        print(
            f"{method:15s} | FND {stats['fnd_mean']:.1f} | HND {stats['hnd_mean']:.1f} | "
            f"LND {stats['lnd_mean']:.1f} | DataE {stats['data_energy_mean']:.4f} | "
            f"CtrlE {stats['control_energy_mean']:.4f} | ReclusterCnt {stats['recluster_count_mean']:.1f}"
        )
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
