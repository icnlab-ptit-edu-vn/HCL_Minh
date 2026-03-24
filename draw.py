#!/usr/bin/env python3
"""
plot_from_results.py

Usage:
    python plot_from_results.py path/to/results_3methods_....npz
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Utilities ----------------
def safe_load_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    return data, keys

def to_2d_numeric(arr):
    """
    Convert loaded array-like (possibly object/list-of-lists) into 2D numpy array padded with np.nan.
    Returns (padded_array (n_runs, max_len), max_len)
    """
    if arr is None:
        return np.zeros((0,0)), 0
    a = np.array(arr, dtype=object)
    if a.size == 0:
        return np.zeros((0,0)), 0

    # If it's numeric 2D or 1D
    if a.dtype != object:
        a2 = np.array(a, dtype=float)
        if a2.ndim == 1:
            return a2.reshape(1, -1), a2.shape[0]
        elif a2.ndim == 2:
            return a2, a2.shape[1]

    # Otherwise treat as sequence-of-runs
    list_of_runs = []
    for el in a:
        if el is None:
            list_of_runs.append(np.array([], dtype=float))
            continue
        try:
            arr_el = np.array(el, dtype=float)
        except Exception:
            arr_el = np.array(list(el), dtype=float)
        list_of_runs.append(arr_el)

    lengths = [len(x) for x in list_of_runs]
    if len(lengths) == 0:
        return np.zeros((0,0)), 0
    max_len = max(lengths)
    padded = np.full((len(list_of_runs), max_len), np.nan, dtype=float)
    for i, r in enumerate(list_of_runs):
        if r.size > 0:
            padded[i, :r.size] = r
    return padded, max_len

def safe_nanmean_std(padded):
    if padded.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    nan_cols = np.isnan(mean)
    if np.any(nan_cols):
        mean[nan_cols] = 0.0
        std[nan_cols] = 0.0
    return mean, std

# ---------------- Main plotting function ----------------
def plot_from_npz(npz_path, out_dir=None, show_plot=False):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")
    data, keys = safe_load_npz(npz_path)
    print("Loaded keys:", keys)

    def find_key(prefix, variants):
        for v in variants:
            if v in keys:
                return v
        for k in keys:
            if k.lower().startswith(prefix.lower()):
                return k
        return None

    # find keys (milestones)
    rl_fnd_k = find_key("rl_fnd", ["rl_fnd","rl_fnd_list","rl_fnd_arr"])
    rl_hnd_k = find_key("rl_hnd", ["rl_hnd","rl_hnd_list","rl_hnd_arr"])
    rl_lnd_k = find_key("rl_lnd", ["rl_lnd","rl_lnd_list","rl_lnd_arr"])
    peg_fnd_k = find_key("peg_fnd", ["peg_fnd","peg_fnd_list","peg_fnd_arr"])
    peg_hnd_k = find_key("peg_hnd", ["peg_hnd","peg_hnd_list","peg_hnd_arr"])
    peg_lnd_k = find_key("peg_lnd", ["peg_lnd","peg_lnd_list","peg_lnd_arr"])
    leach_fnd_k = find_key("leach_fnd", ["leach_fnd","leach_fnd_list","leach_fnd_arr"])
    leach_hnd_k = find_key("leach_hnd", ["leach_hnd","leach_hnd_list","leach_hnd_arr"])
    leach_lnd_k = find_key("leach_lnd", ["leach_lnd","leach_lnd_list","leach_lnd_arr"])

    # time series keys
    rl_alive_k   = find_key("rl_alive", ["rl_alive","rl_alive_padded","rl_alive_history","rl_alive_arr"])
    peg_alive_k  = find_key("peg_alive", ["peg_alive","peg_alive_padded","peg_alive_history","peg_alive_arr"])
    leach_alive_k= find_key("leach_alive", ["leach_alive","leach_alive_padded","leach_alive_history","leach_alive_arr"])

    rl_energy_k   = find_key("rl_energy", ["rl_energy","rl_energy_padded","rl_energy_history","rl_energy_arr"])
    peg_energy_k  = find_key("peg_energy", ["peg_energy","peg_energy_padded","peg_energy_history","peg_energy_arr"])
    leach_energy_k= find_key("leach_energy", ["leach_energy","leach_energy_padded","leach_energy_history","leach_energy_arr"])

    rl_f_k    = find_key("rl_fairness", ["rl_fairness","rl_f_padded","rl_f_history","rl_f"])
    peg_f_k   = find_key("peg_fairness", ["peg_fairness","peg_f_padded","peg_f_history","peg_f"])
    leach_f_k = find_key("leach_fairness", ["leach_fairness","leach_f_padded","leach_f_history","leach_f"])

    # load scalar lists (milestones)
    def load_scalar_list(key):
        if key is None:
            return None
        arr = np.array(data[key], dtype=float)
        if arr.ndim == 0:
            return [float(arr)]
        return arr.tolist()

    rl_fnd = load_scalar_list(rl_fnd_k)
    rl_hnd = load_scalar_list(rl_hnd_k)
    rl_lnd = load_scalar_list(rl_lnd_k)
    peg_fnd = load_scalar_list(peg_fnd_k)
    peg_hnd = load_scalar_list(peg_hnd_k)
    peg_lnd = load_scalar_list(peg_lnd_k)
    leach_fnd = load_scalar_list(leach_fnd_k)
    leach_hnd = load_scalar_list(leach_hnd_k)
    leach_lnd = load_scalar_list(leach_lnd_k)

    # time-series -> padded 2D
    rl_alive_padded, rl_alive_len     = (np.array([]),0) if rl_alive_k is None   else to_2d_numeric(data[rl_alive_k])
    peg_alive_padded, peg_alive_len   = (np.array([]),0) if peg_alive_k is None  else to_2d_numeric(data[peg_alive_k])
    leach_alive_padded,leach_alive_len= (np.array([]),0) if leach_alive_k is None else to_2d_numeric(data[leach_alive_k])

    rl_energy_padded, rl_energy_len     = (np.array([]),0) if rl_energy_k is None   else to_2d_numeric(data[rl_energy_k])
    peg_energy_padded, peg_energy_len   = (np.array([]),0) if peg_energy_k is None  else to_2d_numeric(data[peg_energy_k])
    leach_energy_padded,leach_energy_len= (np.array([]),0) if leach_energy_k is None else to_2d_numeric(data[leach_energy_k])

    rl_f_padded, rl_f_len     = (np.array([]),0) if rl_f_k is None   else to_2d_numeric(data[rl_f_k])
    peg_f_padded, peg_f_len   = (np.array([]),0) if peg_f_k is None  else to_2d_numeric(data[peg_f_k])
    leach_f_padded,leach_f_len= (np.array([]),0) if leach_f_k is None else to_2d_numeric(data[leach_f_k])

    max_len = max(rl_alive_len, peg_alive_len, leach_alive_len,
                  rl_energy_len, peg_energy_len, leach_energy_len,
                  rl_f_len, peg_f_len, leach_f_len)

    if max_len == 0:
        rounds = np.array([])
    else:
        rounds = np.arange(max_len)

    def pad_to(arr2d, cur_len, max_len):
        if max_len == 0:
            return np.full((0, max_len), np.nan)
        if arr2d.size == 0:
            return np.full((0, max_len), np.nan)
        if cur_len == max_len:
            return arr2d
        rows = arr2d.shape[0]
        extra = np.full((rows, max_len - cur_len), np.nan, dtype=float)
        return np.concatenate([arr2d, extra], axis=1)

    rl_alive_padded = pad_to(rl_alive_padded, rl_alive_len, max_len)
    peg_alive_padded = pad_to(peg_alive_padded, peg_alive_len, max_len)
    leach_alive_padded= pad_to(leach_alive_padded, leach_alive_len, max_len)

    rl_energy_padded = pad_to(rl_energy_padded, rl_energy_len, max_len)
    peg_energy_padded = pad_to(peg_energy_padded, peg_energy_len, max_len)
    leach_energy_padded= pad_to(leach_energy_padded, leach_energy_len, max_len)

    rl_f_padded = pad_to(rl_f_padded, rl_f_len, max_len)
    peg_f_padded = pad_to(peg_f_padded, peg_f_len, max_len)
    leach_f_padded= pad_to(leach_f_padded, leach_f_len, max_len)

    # compute mean/std
    rl_alive_mean, rl_alive_std = safe_nanmean_std(rl_alive_padded)
    peg_alive_mean, peg_alive_std = safe_nanmean_std(peg_alive_padded)
    leach_alive_mean, leach_alive_std = safe_nanmean_std(leach_alive_padded)

    rl_energy_mean, rl_energy_std = safe_nanmean_std(rl_energy_padded)
    peg_energy_mean, peg_energy_std = safe_nanmean_std(peg_energy_padded)
    leach_energy_mean, leach_energy_std = safe_nanmean_std(leach_energy_padded)

    rl_f_mean, rl_f_std = safe_nanmean_std(rl_f_padded)
    peg_f_mean, peg_f_std = safe_nanmean_std(peg_f_padded)
    leach_f_mean, leach_f_std = safe_nanmean_std(leach_f_padded)

    # cumulative energy (J->mJ), treat NaN as 0 for cumulative sum start but we will keep NaN when plotting fairness only
    rl_energy_cum = np.cumsum(np.nan_to_num(rl_energy_mean, nan=0.0)) * 1000.0 if rl_energy_mean.size else np.array([])
    peg_energy_cum = np.cumsum(np.nan_to_num(peg_energy_mean, nan=0.0)) * 1000.0 if peg_energy_mean.size else np.array([])
    leach_energy_cum = np.cumsum(np.nan_to_num(leach_energy_mean, nan=0.0)) * 1000.0 if leach_energy_mean.size else np.array([])

    # prepare output dir
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_dir = out_dir or os.path.dirname(npz_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # --- Plot 1: Alive nodes ---
    plt.figure(figsize=(10,6))
    if rl_alive_mean.size:
        plt.plot(rounds, rl_alive_mean, label="RL-HCR", linewidth=2)
        plt.fill_between(rounds, rl_alive_mean - rl_alive_std, rl_alive_mean + rl_alive_std, alpha=0.2)
    if peg_alive_mean.size:
        plt.plot(rounds, peg_alive_mean, label="PEG-ABC", linewidth=2)
        plt.fill_between(rounds, peg_alive_mean - peg_alive_std, peg_alive_mean + peg_alive_std, alpha=0.2)
    if leach_alive_mean.size:
        plt.plot(rounds, leach_alive_mean, label="LEACH-C", linewidth=2)
        plt.fill_between(rounds, leach_alive_mean - leach_alive_std, leach_alive_mean + leach_alive_std, alpha=0.2)
    plt.title("Alive Nodes Over Time (mean ± std)")
    plt.xlabel("Round"); plt.ylabel("Alive nodes")
    plt.grid(True); plt.legend()
    out1 = os.path.join(out_dir, f"{base}_alive.png")
    plt.tight_layout(); plt.savefig(out1, dpi=300); plt.close()
    print("Saved:", out1)

    # --- Plot 2: Cumulative Energy ---
    plt.figure(figsize=(10,6))
    if rl_energy_cum.size:
        plt.plot(rounds, rl_energy_cum, label="RL-HCR", linewidth=2)
    if peg_energy_cum.size:
        plt.plot(rounds, peg_energy_cum, label="PEG-ABC", linewidth=2)
    if leach_energy_cum.size:
        plt.plot(rounds, leach_energy_cum, label="LEACH-C", linewidth=2)
    plt.title("Cumulative Energy Consumption (mJ)")
    plt.xlabel("Round"); plt.ylabel("Cumulative Energy (mJ)")
    plt.grid(True); plt.legend()
    out2 = os.path.join(out_dir, f"{base}_energy_cumulative.png")
    plt.tight_layout(); plt.savefig(out2, dpi=300); plt.close()
    print("Saved:", out2)

    # --- Plot 3: Fairness (preserve NaN so matplotlib will break lines) ---
    plt.figure(figsize=(10,6))
    # use nan_to_num with nan=np.nan to intentionally keep NaNs (no replacement)
    # (np.nan_to_num(..., nan=np.nan) returns same array but clarifies intent)
    if rl_f_mean.size:
        y = np.nan_to_num(rl_f_mean, nan=np.nan)
        plt.plot(rounds, y, label="RL-HCR", linewidth=2)
    if peg_f_mean.size:
        y = np.nan_to_num(peg_f_mean, nan=np.nan)
        plt.plot(rounds, y, label="PEG-ABC", linewidth=2)
    if leach_f_mean.size:
        y = np.nan_to_num(leach_f_mean, nan=np.nan)
        plt.plot(rounds, y, label="LEACH-C", linewidth=2)
    plt.title("Energy Fairness Over Time (Jain mean) -- NaNs preserved")
    plt.xlabel("Round"); plt.ylabel("Jain's Fairness")
    plt.ylim(-0.05, 1.05)
    plt.grid(True); plt.legend()
    out3 = os.path.join(out_dir, f"{base}_fairness.png")
    plt.tight_layout(); plt.savefig(out3, dpi=300); plt.close()
    print("Saved:", out3)

    # --- Plot 4: Lifetime Milestones bar chart (mean ± std) ---
    def mean_std(lst):
        if lst is None:
            return 0.0, 0.0
        arr = np.array(lst, dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    rl_vals = [mean_std(rl_fnd)[0], mean_std(rl_hnd)[0], mean_std(rl_lnd)[0]]
    rl_err  = [mean_std(rl_fnd)[1], mean_std(rl_hnd)[1], mean_std(rl_lnd)[1]]
    peg_vals = [mean_std(peg_fnd)[0], mean_std(peg_hnd)[0], mean_std(peg_lnd)[0]]
    peg_err  = [mean_std(peg_fnd)[1], mean_std(peg_hnd)[1], mean_std(peg_lnd)[1]]
    leach_vals = [mean_std(leach_fnd)[0], mean_std(leach_hnd)[0], mean_std(leach_lnd)[0]]
    leach_err  = [mean_std(leach_fnd)[1], mean_std(leach_hnd)[1], mean_std(leach_lnd)[1]]

    x = np.arange(3)
    width = 0.25
    plt.figure(figsize=(8,6))
    plt.bar(x - width, rl_vals, width, yerr=rl_err, capsize=5, label='RL-HCR')
    plt.bar(x      , peg_vals, width, yerr=peg_err, capsize=5, label='PEG-ABC')
    plt.bar(x + width, leach_vals, width, yerr=leach_err, capsize=5, label='LEACH-C')
    plt.xticks(x, ['FND','HND','LND'])
    plt.ylabel("Rounds"); plt.title("Lifetime Milestones (mean ± std)")
    plt.grid(axis='y'); plt.legend()
    out4 = os.path.join(out_dir, f"{base}_milestones.png")
    plt.tight_layout(); plt.savefig(out4, dpi=300); plt.close()
    print("Saved:", out4)

    return [out1, out2, out3, out4]

# ---------------- CLI ----------------
if __name__ == "__main__":
    npz_file = "results/results_3methods_3000_20251215_054944.npz"
    out_dir = "results/results_3methods_3000_20251215_054944"
    plot_from_npz(npz_file, out_dir=out_dir, show_plot=False)
