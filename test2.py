import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from env.wsn_env import WSNEnvironment
from rl.hppo import H_PPO
from config import *

class Evaluator:
    """
    Evaluator class để đánh giá và so sánh các phương pháp
    """
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        
    def run_rl_hcr(self, max_rounds=2000, deterministic=True):
        """
        Chạy RL-HCR
        """
        if self.agent is None:
            raise ValueError("Agent is not loaded!")
        
        self.agent.set_eval_mode()
        self.env.reset()
        
        # Khởi tạo clustering
        initial_clusters = 5
        self.env.cluster_and_build_chains(num_clusters=initial_clusters)
        
        # Metrics
        fnd = None
        lnd = None
        hnd = None
        total_packets = 0
        total_energy = 0
        alive_history = []
        energy_history = []
        fairness_history = []
        cluster_history = []
        phi_history = []
        
        round_count = 0
        initial_nodes = self.env.n
        
        while round_count < max_rounds:
            state = self.env.get_slim_state()
            
            # Select action
            delta, phi, _, _ = self.agent.select_action(state, deterministic=deterministic)
            
            # Map delta (0-5) to num_clusters (3-8)
            num_clusters = delta + 3
            
            # Re-cluster if needed
            if delta > 0:
                self.env.cluster_and_build_chains(num_clusters=num_clusters)
            
            # Transmit data
            reward, done, info = self.env.transmit_data(phi=phi)
            
            # Collect metrics
            alive_count = int(info["alive_ratio"] * initial_nodes)
            alive_history.append(alive_count)
            energy_history.append(info.get("energy_consumption", 0))
            total_packets += info.get("packets", 0)
            total_energy += info.get("energy_consumption", 0)
            cluster_history.append(num_clusters)
            phi_history.append(phi)
            
            # Calculate Jain's Fairness Index (FIXED)
            # Jain's Fairness = 1 / (1 + CV^2)
            cv = info.get("cv", 0)
            fairness = 1.0 / (1.0 + cv**2)
            fairness_history.append(fairness)

            # ===== THÊM: Debug mỗi 100 rounds =====
            if round_count % 100 == 0:
                print(f"  [RL-HCR] Round {round_count}: "
                    f"Energy={info.get('energy_consumption', 0):.6f} J, "
                    f"CV={cv:.4f}, Fairness={fairness:.4f}, "
                    f"Alive={alive_count}/{initial_nodes}")
            
            # Check milestones
            if fnd is None and alive_count < initial_nodes:
                fnd = round_count
            
            if hnd is None and alive_count <= initial_nodes * 0.5:
                hnd = round_count
            
            if lnd is None and alive_count <= initial_nodes * 0.3:
                lnd = round_count
            
            round_count += 1
            
            if done:
                break
        
        # Set defaults if not reached
        if fnd is None:
            fnd = round_count
        if hnd is None:
            hnd = round_count
        if lnd is None:
            lnd = round_count
        
        return {
            "fnd": fnd,
            "hnd": hnd,
            "lnd": lnd,
            "total_packets": total_packets,
            "total_energy": total_energy,
            "rounds": round_count,
            "alive_history": alive_history,
            "energy_history": energy_history,
            "fairness_history": fairness_history,
            "cluster_history": cluster_history,
            "phi_history": phi_history,
            "avg_packets_per_round": total_packets / round_count if round_count > 0 else 0,
            "avg_energy_per_round": total_energy / round_count if round_count > 0 else 0
        }

    def run_peg_abc_baseline(self, num_clusters=6, phi=0.5, max_rounds=2000):
        """
        Chạy PEG-ABC baseline (fixed parameters)
        """
        self.env.reset()
        
        # Metrics
        fnd = None
        lnd = None
        hnd = None
        total_packets = 0
        total_energy = 0
        alive_history = []
        energy_history = []
        fairness_history = []
        
        round_count = 0
        initial_nodes = self.env.n
        
        while round_count < max_rounds:
            # Fixed clustering
            self.env.cluster_and_build_chains(num_clusters=num_clusters)
            
            # Transmit data
            reward, done, info = self.env.transmit_data(phi=phi)
            
            # Collect metrics
            alive_count = int(info["alive_ratio"] * initial_nodes)
            alive_history.append(alive_count)
            energy_history.append(info.get("energy_consumption", 0))
            total_packets += info.get("packets", 0)
            total_energy += info.get("energy_consumption", 0)
            
            # Calculate fairness (FIXED)
            cv = info.get("cv", 0)
            fairness = 1.0 / (1.0 + cv**2)
            fairness_history.append(fairness)
            
            # ===== THÊM: Debug mỗi 100 rounds =====
            if round_count % 100 == 0:
                print(f"  [PEG-ABC] Round {round_count}: "
                    f"Energy={info.get('energy_consumption', 0):.6f} J, "
                    f"CV={cv:.4f}, Fairness={fairness:.4f}, "
                    f"Alive={alive_count}/{initial_nodes}")
            
            # Check milestones
            if fnd is None and alive_count < initial_nodes:
                fnd = round_count
            
            if hnd is None and alive_count <= initial_nodes * 0.5:
                hnd = round_count
            
            if lnd is None and alive_count <= initial_nodes * 0.3:
                lnd = round_count
            
            round_count += 1
            
            if done:
                break
        
        # Set defaults
        if fnd is None:
            fnd = round_count
        if hnd is None:
            hnd = round_count
        if lnd is None:
            lnd = round_count
        
        return {
            "fnd": fnd,
            "hnd": hnd,
            "lnd": lnd,
            "total_packets": total_packets,
            "total_energy": total_energy,
            "rounds": round_count,
            "alive_history": alive_history,
            "energy_history": energy_history,
            "fairness_history": fairness_history,
            "avg_packets_per_round": total_packets / round_count if round_count > 0 else 0,
            "avg_energy_per_round": total_energy / round_count if round_count > 0 else 0
        }


def pad_arrays(arr_list, target_length):
    """
    Pad arrays về cùng độ dài
    
    Args:
        arr_list: List of arrays
        target_length: Độ dài mục tiêu
    
    Returns:
        padded_arrays: Numpy array (num_arrays, target_length)
    """
    padded = []
    for arr in arr_list:
        arr = np.array(arr)
        if len(arr) < target_length:
            # Pad với giá trị cuối cùng
            last_val = arr[-1] if len(arr) > 0 else 0
            arr = np.pad(arr, (0, target_length - len(arr)), 
                        mode='constant', constant_values=last_val)
        else:
            arr = arr[:target_length]
        padded.append(arr)
    return np.array(padded)


def plot_comparison(rl_results, baseline_results, save_path="results_comparison.png"):
    """
    Vẽ biểu đồ so sánh
    
    Args:
        rl_results: Dictionary chứa kết quả RL-HCR
        baseline_results: Dictionary chứa kết quả PEG-ABC
        save_path: Đường dẫn lưu biểu đồ
    """
    # Tính độ dài tối đa
    max_len = max(
        max(len(a) for a in rl_results["alive_history"]),
        max(len(a) for a in baseline_results["alive_history"])
    )
    
    # Pad arrays
    rl_alive = pad_arrays(rl_results["alive_history"], max_len)
    baseline_alive = pad_arrays(baseline_results["alive_history"], max_len)
    rl_fairness = pad_arrays(rl_results["fairness_history"], max_len)
    baseline_fairness = pad_arrays(baseline_results["fairness_history"], max_len)
    rl_energy = pad_arrays(rl_results["energy_history"], max_len)
    baseline_energy = pad_arrays(baseline_results["energy_history"], max_len)
    
    # Tính mean và std
    rl_alive_mean = np.mean(rl_alive, axis=0)
    rl_alive_std = np.std(rl_alive, axis=0)
    baseline_alive_mean = np.mean(baseline_alive, axis=0)
    baseline_alive_std = np.std(baseline_alive, axis=0)
    
    rl_fairness_mean = np.mean(rl_fairness, axis=0)
    rl_fairness_std = np.std(rl_fairness, axis=0)
    baseline_fairness_mean = np.mean(baseline_fairness, axis=0)
    baseline_fairness_std = np.std(baseline_fairness, axis=0)
    
    rl_energy_mean = np.mean(rl_energy, axis=0)
    rl_energy_std = np.std(rl_energy, axis=0)
    baseline_energy_mean = np.mean(baseline_energy, axis=0)
    baseline_energy_std = np.std(baseline_energy, axis=0)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Alive Nodes
    ax = axes[0, 0]
    rounds = np.arange(max_len)
    ax.plot(rounds, rl_alive_mean, label="RL-HCR", linewidth=2, color='blue')
    ax.fill_between(rounds, rl_alive_mean - rl_alive_std, rl_alive_mean + rl_alive_std, 
                     alpha=0.2, color='blue')
    ax.plot(rounds, baseline_alive_mean, label="PEG-ABC", linewidth=2, color='red')
    ax.fill_between(rounds, baseline_alive_mean - baseline_alive_std, 
                     baseline_alive_mean + baseline_alive_std, alpha=0.2, color='red')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Alive Nodes", fontsize=12)
    ax.set_title("Network Lifetime", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Fairness
    ax = axes[0, 1]
    ax.plot(rounds, rl_fairness_mean, label="RL-HCR", linewidth=2, color='blue')
    ax.fill_between(rounds, rl_fairness_mean - rl_fairness_std, 
                     rl_fairness_mean + rl_fairness_std, alpha=0.2, color='blue')
    ax.plot(rounds, baseline_fairness_mean, label="PEG-ABC", linewidth=2, color='red')
    ax.fill_between(rounds, baseline_fairness_mean - baseline_fairness_std, 
                     baseline_fairness_mean + baseline_fairness_std, alpha=0.2, color='red')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_title("Energy Fairness", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.05])  # Giới hạn trục y để thấy rõ sự khác biệt
    
    # 3. Cumulative Energy (FIXED)
    ax = axes[1, 0]
    
    # Chuyển đổi sang đơn vị mJ (millijoules) để dễ đọc
    rl_energy_cum = np.cumsum(rl_energy_mean) * 1000  # J -> mJ
    baseline_energy_cum = np.cumsum(baseline_energy_mean) * 1000  # J -> mJ
    
    ax.plot(rounds, rl_energy_cum, label="RL-HCR", linewidth=2, color='blue')
    ax.plot(rounds, baseline_energy_cum, label="PEG-ABC", linewidth=2, color='red')
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Energy (mJ)", fontsize=12)  # Đổi đơn vị
    ax.set_title("Energy Consumption", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Debug: In giá trị để kiểm tra
    print(f"\n🔍 DEBUG Energy Consumption:")
    print(f"  RL-HCR - Min: {rl_energy_cum.min():.6f} mJ, Max: {rl_energy_cum.max():.6f} mJ")
    print(f"  PEG-ABC - Min: {baseline_energy_cum.min():.6f} mJ, Max: {baseline_energy_cum.max():.6f} mJ")
    
    # 4. Bar chart - Summary metrics
    ax = axes[1, 1]
    metrics = ['FND', 'HND', 'LND']
    rl_values = [
        np.mean(rl_results["fnd"]),
        np.mean(rl_results["hnd"]),
        np.mean(rl_results["lnd"])
    ]
    baseline_values = [
        np.mean(baseline_results["fnd"]),
        np.mean(baseline_results["hnd"]),
        np.mean(baseline_results["lnd"])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, rl_values, width, label='RL-HCR', color='blue', alpha=0.7)
    ax.bar(x + width/2, baseline_values, width, label='PEG-ABC', color='red', alpha=0.7)
    ax.set_ylabel('Rounds', fontsize=12)
    ax.set_title('Lifetime Milestones', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ vào '{save_path}'")
    plt.show()


def print_statistics(rl_results, baseline_results):
    """
    In thống kê so sánh
    
    Args:
        rl_results: Dictionary chứa kết quả RL-HCR
        baseline_results: Dictionary chứa kết quả PEG-ABC
    """
    print("\n" + "="*80)
    print("📊 KẾT QUẢ SO SÁNH (Mean ± Std)")
    print("="*80)
    
    metrics = [
        ("FND (First Node Dies)", "fnd"),
        ("HND (Half Nodes Die)", "hnd"),
        ("LND (Last Node Dies)", "lnd"),
        ("Total Packets", "total_packets"),
        ("Total Energy (J)", "total_energy"),
        ("Rounds", "rounds"),
        ("Packets/Round", "avg_packets_per_round"),
        ("Energy/Round (J)", "avg_energy_per_round")
    ]
    
    print(f"{'Metric':<25} {'RL-HCR':<25} {'PEG-ABC':<25} {'Improvement':<15}")
    print("-"*90)
    
    for name, key in metrics:
        rl_mean = np.mean(rl_results[key])
        rl_std = np.std(rl_results[key])
        baseline_mean = np.mean(baseline_results[key])
        baseline_std = np.std(baseline_results[key])
        
        # Calculate improvement
        if baseline_mean != 0:
            if "energy" in key.lower():
                # Lower is better for energy
                improvement = ((baseline_mean - rl_mean) / baseline_mean) * 100
            else:
                # Higher is better for others
                improvement = ((rl_mean - baseline_mean) / baseline_mean) * 100
        else:
            improvement = 0
        
        print(f"{name:<25} {rl_mean:>10.2f} ± {rl_std:<7.2f}  "
              f"{baseline_mean:>10.2f} ± {baseline_std:<7.2f}  "
              f"{improvement:>+6.2f}%")
    
    print("="*80)


def main():
    """
    Main function
    """
    print("="*80)
    print("🧪 Testing RL-HCR vs PEG-ABC")
    print("="*80)
    
    # Load config
    config = {
        'N_NODES': N_NODES,
        'AREA_SIZE': AREA_SIZE,
        'BS_POS': BS_POS,
        'NUM_CLUSTERS': NUM_CLUSTERS,
        'MODEL_DIR': MODEL_DIR,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    
    # Initialize environment
    env = WSNEnvironment(
        n_nodes=config['N_NODES'],
        area=config['AREA_SIZE'],
        bs=config['BS_POS']
    )
    
    # Load agent
    agent = H_PPO(state_dim=10, device=config['DEVICE'])
    
    # Try to load best model first, then latest
    
    print(f"model dir: {config['MODEL_DIR']}")
    model_paths = [
        os.path.join(config['MODEL_DIR'], "rl_hcr_best.pth"),
        os.path.join(config['MODEL_DIR'], f"rl_hcr_ep{EPISODES}.pth")
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"✅ Đã tải model từ: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Lỗi khi load model {model_path}: {e}")
    
    if not model_loaded:
        print("❌ Không tìm thấy model! Vui lòng chạy train.py trước.")
        return
    
    # Create evaluator
    evaluator = Evaluator(env, agent)
    
    # Run experiments
    NUM_RUNS = 30
    print(f"\n🔬 Chạy {NUM_RUNS} experiments...")
    
    rl_results = {
        "fnd": [], "hnd": [], "lnd": [],
        "total_packets": [], "total_energy": [], "rounds": [],
        "alive_history": [], "energy_history": [], "fairness_history": [],
        "avg_packets_per_round": [], "avg_energy_per_round": []
    }
    
    baseline_results = {
        "fnd": [], "hnd": [], "lnd": [],
        "total_packets": [], "total_energy": [], "rounds": [],
        "alive_history": [], "energy_history": [], "fairness_history": [],
        "avg_packets_per_round": [], "avg_energy_per_round": []
    }
    
    for run in tqdm(range(NUM_RUNS), desc="Running experiments"):
        # RL-HCR
        rl_metrics = evaluator.run_rl_hcr(max_rounds=2000, deterministic=True)
        for key in rl_results.keys():
            rl_results[key].append(rl_metrics[key])
        
        # PEG-ABC
        baseline_metrics = evaluator.run_peg_abc_baseline(
            num_clusters=config['NUM_CLUSTERS'],
            phi=0.5,
            max_rounds=2000
        )
        for key in baseline_results.keys():
            baseline_results[key].append(baseline_metrics[key])
    
    # Print statistics
    print_statistics(rl_results, baseline_results)
    
    # Plot comparison
    plot_comparison(rl_results, baseline_results, save_path="/home/icnlab/rl-hcr/results_comparison.png")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"results_{timestamp}.npz")
    
    np.savez(
        results_path,
        rl_fnd=rl_results["fnd"],
        rl_hnd=rl_results["hnd"],
        rl_lnd=rl_results["lnd"],
        rl_packets=rl_results["total_packets"],
        rl_energy=rl_results["total_energy"],
        baseline_fnd=baseline_results["fnd"],
        baseline_hnd=baseline_results["hnd"],
        baseline_lnd=baseline_results["lnd"],
        baseline_packets=baseline_results["total_packets"],
        baseline_energy=baseline_results["total_energy"]
    )
    
    print(f"\n✅ Đã lưu kết quả vào: {results_path}")
    print("="*80)


if __name__ == "__main__":
    main()