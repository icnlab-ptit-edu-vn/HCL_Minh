#!/usr/bin/env python3
"""
test3.py
So sánh 3 phương pháp: RL-HCR, PEG-ABC, LEACH-C
Phiên bản ổn định (robust padding, safe mean/std, debug).
"""

from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Project imports (adjust paths if necessary)
from env.wsn_env import WSNEnvironment
from rl.hppo import H_PPO
from config import *

# If your protocols modules live elsewhere, adjust import paths
from protocols.peg_abc import select_leaders_peg_abc
# LEACH-C helpers are used inside env.setup_leach_c

# ---------------- Configuration ----------------
NUM_RUNS = 10        # số runs độc lập để lấy trung bình / std
NUM_ROUNDS = 20    # số vòng tối đa cho mỗi run
NUM_CLUSTERS = NUM_CLUSTERS if 'NUM_CLUSTERS' in globals() else 6
SAVE_PLOT = "results_comparison_3methods_full_3000.png"

# ---------------- Utility functions ----------------

# def compute_jain(energy_array):
#     if energy_array is None or len(energy_array) == 0:
#         return np.nan  # thay vì 0.0 để không làm tăng fairness giả
#     s = np.sum(energy_array)
#     denom = len(energy_array) * np.sum(energy_array ** 2)
#     return (s ** 2) / denom if denom > 0 else np.nan

def compute_jain(energy_array):
    """
    Trả về Jain's fairness index. Nếu không có node sống hoặc chỉ 1 node sống -> trả np.nan
    để tránh spike giả tạo khi chỉ còn 1 sample.
    """
    if energy_array is None:
        return np.nan
    # Lọc những giá trị hợp lệ
    a = np.asarray(energy_array)
    if a.size <= 1:
        return np.nan
    # Nếu có NaN trong a, loại bỏ chúng
    a = a[~np.isnan(a)]
    if a.size <= 1:
        return np.nan
    s = np.sum(a)
    denom = a.size * np.sum(a ** 2)
    if denom <= 0:
        return np.nan
    return (s ** 2) / denom



def pad_with_nan(list_of_lists):
    """
    Pad list of lists/arrays with np.nan to same length.
    Returns (padded_array (n_runs, max_len), max_len)
    """
    if len(list_of_lists) == 0:
        return np.zeros((0, 0)), 0
    lengths = [len(x) for x in list_of_lists]
    max_len = max(lengths) if lengths else 0
    padded = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, arr in enumerate(list_of_lists):
        if arr is None:
            continue
        a = np.array(arr, dtype=float)
        padded[i, :len(a)] = a
    return padded, max_len

def ensure_global_len(padded, current_max, global_max):
    """Extend padded to global_max columns by appending np.nan columns."""
    if current_max == global_max:
        return padded
    if padded.size == 0:
        return np.full((0, global_max), np.nan)
    extra = np.full((padded.shape[0], global_max - current_max), np.nan, dtype=float)
    return np.concatenate([padded, extra], axis=1)

def safe_nanmean_std(padded):
    """
    Compute mean and std safely from padded (with np.nan).
    Replace columns where all values are nan with zeros (mean=0,std=0).
    """
    if padded.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    nan_cols = np.isnan(mean)
    if np.any(nan_cols):
        mean[nan_cols] = 0.0
        std[nan_cols] = 0.0
    return mean, std

# ---------------- Run functions ----------------
def run_one_rl(env, agent, deterministic=True, max_rounds=NUM_ROUNDS, initial_clusters=NUM_CLUSTERS):
    """Run RL-HCR on env using agent (agent may be untrained)."""
    agent.set_eval_mode()
    env.reset()
    env.cluster_and_build_chains(num_clusters=initial_clusters)

    fnd = hnd = lnd = None
    alive_history = []
    energy_history = []
    fairness_history = []
    cluster_history = []
    phi_history = []
    total_packets = 0
    total_energy = 0.0

    round_idx = 0
    initial_nodes = env.n

    while round_idx < max_rounds:
        state = env.get_slim_state()
        delta, phi, _, _ = agent.select_action(state, deterministic=deterministic)

        try:
            num_clusters = int(delta) + 3
        except Exception:
            num_clusters = initial_clusters

        if num_clusters != len(env.clusters):
            env.cluster_and_build_chains(num_clusters=num_clusters)

        reward, done, info = env.transmit_data(phi=phi)

        alive_count = int(info.get("alive_count", -1))
        if alive_count < 0:
            alive_count = int(info.get("alive_ratio", 0.0) * initial_nodes)

        alive_history.append(alive_count)
        energy_history.append(info.get("energy_consumption", 0.0))
        total_packets += info.get("packets", 0)
        total_energy += info.get("energy_consumption", 0.0)
        cluster_history.append(num_clusters)
        phi_history.append(phi)

        # cv = info.get("cv", 0.0)
        # fairness = 1.0 / (1.0 + cv * cv)

        alive_energy = env.energy[env.alive]
        fairness = compute_jain(alive_energy)

        fairness_history.append(fairness)

        if fnd is None and alive_count < initial_nodes:
            fnd = round_idx
        if hnd is None and alive_count <= initial_nodes * 0.5:
            hnd = round_idx
        if lnd is None and alive_count <= initial_nodes * 0.3:
            lnd = round_idx

        round_idx += 1

        if np.sum(env.alive) == 0:
            break
        if done and np.sum(env.alive) == 0:
            break

    if fnd is None: fnd = round_idx
    if hnd is None: hnd = round_idx
    if lnd is None: lnd = round_idx

    return {
        "fnd": fnd, "hnd": hnd, "lnd": lnd,
        "alive_history": alive_history,
        "energy_history": energy_history,
        "fairness_history": fairness_history,
        "cluster_history": cluster_history,
        "phi_history": phi_history,
        "total_packets": total_packets,
        "total_energy": total_energy,
        "rounds": round_idx
    }

def run_one_peg_abc(env, num_clusters=NUM_CLUSTERS, phi=0.5, max_rounds=NUM_ROUNDS):
    """Run PEG-ABC baseline by re-clustering each round and calling transmit_data."""
    env.reset()
    fnd = hnd = lnd = None
    alive_history = []
    energy_history = []
    fairness_history = []
    total_packets = 0
    total_energy = 0.0

    round_idx = 0
    initial_nodes = env.n

    while round_idx < max_rounds:
        env.cluster_and_build_chains(num_clusters=num_clusters)
        reward, done, info = env.transmit_data(phi=phi)

        alive_count = int(info.get("alive_count", -1))
        if alive_count < 0:
            alive_count = int(info.get("alive_ratio", 0.0) * initial_nodes)

        alive_history.append(alive_count)
        energy_history.append(info.get("energy_consumption", 0.0))
        total_packets += info.get("packets", 0)
        total_energy += info.get("energy_consumption", 0.0)

        # cv = info.get("cv", 0.0)
        # fairness = 1.0 / (1.0 + cv * cv)
        
        alive_energy = env.energy[env.alive]
        fairness = compute_jain(alive_energy)

        fairness_history.append(fairness)

        if fnd is None and alive_count < initial_nodes:
            fnd = round_idx
        if hnd is None and alive_count <= initial_nodes * 0.5:
            hnd = round_idx
        if lnd is None and alive_count <= initial_nodes * 0.3:
            lnd = round_idx

        round_idx += 1
        if np.sum(env.alive) == 0:
            break
        if done and np.sum(env.alive) == 0:
            break

    if fnd is None: fnd = round_idx
    if hnd is None: hnd = round_idx
    if lnd is None: lnd = round_idx

    return {
        "fnd": fnd, "hnd": hnd, "lnd": lnd,
        "alive_history": alive_history,
        "energy_history": energy_history,
        "fairness_history": fairness_history,
        "total_packets": total_packets,
        "total_energy": total_energy,
        "rounds": round_idx
    }

def run_one_leach_c(env, num_clusters=NUM_CLUSTERS, max_rounds=NUM_ROUNDS):
    """Run LEACH-C baseline using env.setup_leach_c and env.transmit_data_leach_c."""
    env.reset()
    # ensure env tracks previous cumulative energy for fallback
    if not hasattr(env, "_prev_total_energy_consumed"):
        env._prev_total_energy_consumed = 0.0

    fnd = hnd = lnd = None
    alive_history = []
    energy_history = []
    fairness_history = []
    total_packets = 0
    total_energy = 0.0

    round_idx = 0
    initial_nodes = env.n

    while round_idx < max_rounds:
        env.setup_leach_c(num_clusters=num_clusters)
        reward, done, info = env.transmit_data_leach_c()

        alive_count = int(info.get("alive_count", -1))
        if alive_count < 0:
            alive_count = int(info.get("alive_ratio", 0.0) * initial_nodes)

        alive_history.append(alive_count)

        # Try to get energy_consumption from info; fallback to env.total_energy_consumed delta
        e_cons = info.get("energy_consumption", None)
        if e_cons is None:
            current_total = getattr(env, "total_energy_consumed", None)
            if current_total is None:
                e_cons = 0.0
            else:
                prev_total = getattr(env, "_prev_total_energy_consumed", 0.0)
                e_cons = current_total - prev_total
                env._prev_total_energy_consumed = current_total
                if e_cons < 0:
                    e_cons = 0.0

        energy_history.append(e_cons)
        total_packets += info.get("packets", 0)
        total_energy += e_cons

        alive_energy = env.energy[env.alive]
        jain = compute_jain(alive_energy)
        fairness_history.append(jain)

        if fnd is None and alive_count < initial_nodes:
            fnd = round_idx
        if hnd is None and alive_count <= initial_nodes * 0.5:
            hnd = round_idx
        if lnd is None and alive_count <= initial_nodes * 0.3:
            lnd = round_idx

        round_idx += 1
        if np.sum(env.alive) == 0:
            break
        if done and np.sum(env.alive) == 0:
            break

    if fnd is None: fnd = round_idx
    if hnd is None: hnd = round_idx
    if lnd is None: lnd = round_idx

    return {
        "fnd": fnd, "hnd": hnd, "lnd": lnd,
        "alive_history": alive_history,
        "energy_history": energy_history,
        "fairness_history": fairness_history,
        "total_packets": total_packets,
        "total_energy": total_energy,
        "rounds": round_idx
    }

# ---------------- Main experiment ----------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"DEVICE = {device}")
    print("Model dir:", MODEL_DIR)

    agent = H_PPO(state_dim=10, device=device)

    model_paths = [
        os.path.join(MODEL_DIR, "rl_hcr_best.pth"),
        os.path.join(MODEL_DIR, f"rl_hcr_ep{EPISODES}.pth")
    ]
    model_loaded = False
    for mp in model_paths:
        if os.path.exists(mp):
            try:
                agent.load(mp)
                print(f"✅ Loaded RL model: {mp}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Warning: failed to load {mp}: {e}")

    if not model_loaded:
        print("⚠️ No RL checkpoint found; RL will use random/untrained policy.")

    # Kết quả
    rl_results = {"fnd": [], "hnd": [], "lnd": [], "alive": [], "energy": [], "fairness": []}
    peg_results = {"fnd": [], "hnd": [], "lnd": [], "alive": [], "energy": [], "fairness": []}
    leach_results = {"fnd": [], "hnd": [], "lnd": [], "alive": [], "energy": [], "fairness": []}

    print(f"Running {NUM_RUNS} runs (each up to {NUM_ROUNDS} rounds)...")
    for run in tqdm(range(NUM_RUNS), desc="Experiments"):
        env_rl = WSNEnvironment(n_nodes=N_NODES, area=AREA_SIZE, bs=BS_POS)
        res_rl = run_one_rl(env_rl, agent, deterministic=True, max_rounds=NUM_ROUNDS, initial_clusters=NUM_CLUSTERS)
        for k in rl_results.keys():
            rl_results[k].append(res_rl[k if k in res_rl else k+"_history"])

        env_peg = WSNEnvironment(n_nodes=N_NODES, area=AREA_SIZE, bs=BS_POS)
        res_peg = run_one_peg_abc(env_peg, num_clusters=NUM_CLUSTERS, phi=0.5, max_rounds=NUM_ROUNDS)
        for k in peg_results.keys():
            peg_results[k].append(res_peg[k if k in res_peg else k+"_history"])

        env_leach = WSNEnvironment(n_nodes=N_NODES, area=AREA_SIZE, bs=BS_POS)
        res_leach = run_one_leach_c(env_leach, num_clusters=NUM_CLUSTERS, max_rounds=NUM_ROUNDS)
        for k in leach_results.keys():
            leach_results[k].append(res_leach[k if k in res_leach else k+"_history"])

        print(f"Run {run+1}/{NUM_RUNS} done: RL={len(res_rl['alive_history'])}, PEG={len(res_peg['alive_history'])}, LEACH={len(res_leach['alive_history'])}")

    # --- Chuẩn bị padding và trung bình ---
    rl_alive_padded, rl_max = pad_with_nan(rl_results["alive"])
    peg_alive_padded, peg_max = pad_with_nan(peg_results["alive"])
    leach_alive_padded, leach_max = pad_with_nan(leach_results["alive"])
    global_max = max(rl_max, peg_max, leach_max)
    rl_alive_padded = ensure_global_len(rl_alive_padded, rl_max, global_max)
    peg_alive_padded = ensure_global_len(peg_alive_padded, peg_max, global_max)
    leach_alive_padded = ensure_global_len(leach_alive_padded, leach_max, global_max)

    rl_energy_padded, rl_e_max = pad_with_nan(rl_results["energy"])
    peg_energy_padded, peg_e_max = pad_with_nan(peg_results["energy"])
    leach_energy_padded, leach_e_max = pad_with_nan(leach_results["energy"])
    rl_energy_padded = ensure_global_len(rl_energy_padded, rl_e_max, global_max)
    peg_energy_padded = ensure_global_len(peg_energy_padded, peg_e_max, global_max)
    leach_energy_padded = ensure_global_len(leach_energy_padded, leach_e_max, global_max)

    rl_f_padded, _ = pad_with_nan(rl_results["fairness"])
    peg_f_padded, _ = pad_with_nan(peg_results["fairness"])
    leach_f_padded, _ = pad_with_nan(leach_results["fairness"])
    rl_f_padded = ensure_global_len(rl_f_padded, rl_f_padded.shape[1] if rl_f_padded.size else 0, global_max)
    peg_f_padded = ensure_global_len(peg_f_padded, peg_f_padded.shape[1] if peg_f_padded.size else 0, global_max)
    leach_f_padded = ensure_global_len(leach_f_padded, leach_f_padded.shape[1] if leach_f_padded.size else 0, global_max)

    # --- Lưu toàn bộ dữ liệu ra file .npz ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/results_3methods_3000_{timestamp}.npz"

    np.savez_compressed(
        save_path,
        rl_fnd=rl_results["fnd"],
        rl_hnd=rl_results["hnd"],
        rl_lnd=rl_results["lnd"],
        peg_fnd=peg_results["fnd"],
        peg_hnd=peg_results["hnd"],
        peg_lnd=peg_results["lnd"],
        leach_fnd=leach_results["fnd"],
        leach_hnd=leach_results["hnd"],
        leach_lnd=leach_results["lnd"],
        rl_alive=rl_alive_padded,
        peg_alive=peg_alive_padded,
        leach_alive=leach_alive_padded,
        rl_energy=rl_energy_padded,
        peg_energy=peg_energy_padded,
        leach_energy=leach_energy_padded,
        rl_fairness=rl_f_padded,
        peg_fairness=peg_f_padded,
        leach_fairness=leach_f_padded,
    )

    print(f"✅ Saved all raw data for plotting to: {save_path}")

    # --- Giữ nguyên phần vẽ biểu đồ ---
    # (phần plotting như trong code của bạn, không cần thay đổi)
    # Sau khi chạy xong bạn có thể vẽ lại từ file .npz mà không cần chạy lại mô phỏng.

if __name__ == "__main__":
    main()