import numpy as np

from protocols.peg_abc import select_leaders_peg_abc, build_chain_intra
from protocols.leach_c import select_leaders_leach_c, assign_clusters_leach_c
from utils.energy_model import tx_energy, rx_energy, fusion_energy, control_tx_energy, control_rx_energy


class WSNEnvironment:
    """Overhead-aware clustered WSN environment aligned with Minh_2026."""

    def __init__(
        self,
        n_nodes=100,
        area=100,
        bs=np.array([150, 50]),
        initial_energy=0.5,
        control_uplink_bits=200,
        control_downlink_bits=200,
        max_cluster_hint=8,
        reward_alpha=1.0,
        reward_beta=150.0,
        reward_gamma=200.0,
        reward_delta=1.0,
    ):
        self.n = int(n_nodes)
        self.area = area
        self.bs = np.array(bs, dtype=np.float32)
        self.initial_energy = float(initial_energy)
        self.control_uplink_bits = int(control_uplink_bits)
        self.control_downlink_bits = int(control_downlink_bits)
        self.max_cluster_hint = max(1, int(max_cluster_hint))
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        self.reward_delta = float(reward_delta)
        self.round_count = 0
        self.total_packets_sent = 0
        self.first_node_dead_round = None
        self.half_nodes_dead_round = None
        self.reset()

    def _sample_nodes(self):
        if np.isscalar(self.area):
            width = height = float(self.area)
        else:
            area = np.asarray(self.area, dtype=np.float32).flatten()
            if area.size != 2:
                raise ValueError("area must be a scalar or a length-2 iterable")
            width, height = float(area[0]), float(area[1])
        return np.random.rand(self.n, 2).astype(np.float32) * np.array([width, height], dtype=np.float32)

    def _energy_normalizer(self):
        return max(self.n * self.initial_energy, 1e-8)

    def _normalize_energy(self, energy_value):
        return float(energy_value) / self._energy_normalizer()

    def reset(self):
        self.nodes = self._sample_nodes()
        self.energy = np.full(self.n, self.initial_energy, dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        self.leaders = []
        self.chains = []
        self.clusters = []
        self.current_num_clusters = 0
        self.last_energy_consumption = np.zeros(self.n, dtype=np.float32)
        self.pending_reconfig_overhead = np.zeros(self.n, dtype=np.float32)
        self.total_energy_consumed = 0.0
        self.total_reconfig_overhead = 0.0
        self.recluster_count = 0
        self.last_recluster_overhead = 0.0
        self.last_recluster_flag = 0.0
        self.last_data_energy = 0.0
        self.last_control_energy = 0.0
        self.cluster_balance_cv = 0.0
        self.topology_age = 0
        self.round_count = 0
        self.total_packets_sent = 0
        self.first_node_dead_round = None
        self.half_nodes_dead_round = None
        return self.get_slim_state()

    def get_distances(self):
        dist_matrix = np.linalg.norm(self.nodes[:, None, :] - self.nodes[None, :, :], axis=2)
        dist_to_bs = np.linalg.norm(self.nodes - self.bs, axis=1)
        return dist_matrix, dist_to_bs

    def _cluster_size_cv(self):
        sizes = []
        for cluster in self.clusters:
            alive_members = [idx for idx in cluster if self.alive[idx]]
            if alive_members:
                sizes.append(len(alive_members))
        sizes = np.asarray(sizes, dtype=np.float32)
        if sizes.size <= 1:
            return 0.0
        mean_size = float(np.mean(sizes))
        if mean_size <= 0:
            return 0.0
        return float(np.std(sizes) / (mean_size + 1e-8))

    def _update_topology_descriptors(self):
        self.current_num_clusters = int(sum(1 for cluster in self.clusters if any(self.alive[idx] for idx in cluster)))
        self.cluster_balance_cv = self._cluster_size_cv()

    def _apply_reconfiguration_overhead(self):
        alive_indices = np.where(self.alive)[0]
        if len(alive_indices) == 0:
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0
            return 0.0
        dist_to_bs = np.linalg.norm(self.nodes[alive_indices] - self.bs, axis=1)
        tx_cost = control_tx_energy(dist_to_bs, bits=self.control_uplink_bits)
        rx_cost = np.full(len(alive_indices), control_rx_energy(bits=self.control_downlink_bits), dtype=np.float32)
        control_cost = np.asarray(tx_cost, dtype=np.float32) + rx_cost
        per_node = np.zeros(self.n, dtype=np.float32)
        per_node[alive_indices] = control_cost
        self.energy[alive_indices] -= control_cost
        self.energy = np.maximum(self.energy, 0.0)
        self.alive = self.energy > 0
        self.pending_reconfig_overhead += per_node
        overhead = float(np.sum(control_cost))
        self.total_reconfig_overhead += overhead
        self.last_recluster_overhead = overhead
        self.last_control_energy = overhead
        self.recluster_count += 1
        return overhead

    def cluster_and_build_chains(self, num_clusters=6, charge_overhead=False):
        alive_indices = np.where(self.alive)[0]
        if len(alive_indices) == 0:
            self.leaders = []
            self.chains = []
            self.clusters = []
            self.current_num_clusters = 0
            self.cluster_balance_cv = 0.0
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0
            return 0.0

        num_clusters = max(1, min(int(num_clusters), len(alive_indices)))
        x_coords = self.nodes[alive_indices, 0]
        sorted_idx = alive_indices[np.argsort(x_coords)]
        cluster_size = max(1, len(alive_indices) // num_clusters)

        self.clusters = []
        for i in range(num_clusters):
            start = i * cluster_size
            end = min(start + cluster_size, len(alive_indices)) if i < num_clusters - 1 else len(alive_indices)
            cluster = sorted_idx[start:end].tolist()
            if cluster:
                self.clusters.append(cluster)

        self.leaders = []
        self.chains = []
        dist_matrix, dist_to_bs = self.get_distances()
        for cluster in self.clusters:
            sub_nodes = self.nodes[cluster]
            sub_energy = self.energy[cluster]
            sub_dist_mat = dist_matrix[np.ix_(cluster, cluster)]
            sub_dist_bs = dist_to_bs[cluster]
            try:
                leader_local_list = select_leaders_peg_abc(
                    sub_nodes,
                    sub_energy,
                    sub_dist_mat,
                    sub_dist_bs,
                    num_clusters=1,
                    max_iter=10,
                    num_bees=20,
                )
                leader_local = leader_local_list[0] if len(leader_local_list) > 0 else int(np.argmax(sub_energy))
            except Exception:
                leader_local = int(np.argmax(sub_energy))

            leader_global = cluster[leader_local]
            self.leaders.append(leader_global)

            try:
                local_chain = build_chain_intra(sub_nodes, bs_pos=self.bs)
                global_chain = [cluster[i] for i in local_chain]
                self.chains.append(global_chain)
            except Exception:
                self.chains.append(cluster)

        self.topology_age = 0
        self._update_topology_descriptors()
        if charge_overhead:
            overhead = self._apply_reconfiguration_overhead()
        else:
            overhead = 0.0
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0
        return overhead

    def setup_leach_c(self, num_clusters=6, charge_overhead=False):
        alive_indices = np.where(self.alive)[0]
        if len(alive_indices) == 0:
            self.leaders = []
            self.clusters = []
            self.chains = []
            self.current_num_clusters = 0
            self.cluster_balance_cv = 0.0
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0
            return 0.0

        num_clusters = max(1, min(int(num_clusters), len(alive_indices)))
        sub_nodes = self.nodes[alive_indices]
        sub_energy = self.energy[alive_indices]
        local_leaders = select_leaders_leach_c(sub_nodes, sub_energy, self.bs, num_clusters)
        self.leaders = alive_indices[np.array(local_leaders, dtype=int)].tolist()
        self.clusters = assign_clusters_leach_c(self.nodes, self.leaders, member_indices=alive_indices)
        self.chains = []
        self.topology_age = 0
        self._update_topology_descriptors()
        if charge_overhead:
            overhead = self._apply_reconfiguration_overhead()
        else:
            overhead = 0.0
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0
        return overhead

    def decode_topology_action(self, action, min_clusters=3, max_clusters=8, default_clusters=None):
        action = int(action)
        if action <= 0:
            target = self.current_num_clusters or default_clusters or min_clusters
            return False, int(target)
        target = int(min_clusters + action - 1)
        target = max(min_clusters, min(max_clusters, target))
        return True, target

    def step(self, topology_action, min_clusters=3, max_clusters=8, default_clusters=None):
        if len(self.clusters) == 0:
            bootstrap_clusters = default_clusters or max(min_clusters, min(max_clusters, 6))
            self.cluster_and_build_chains(num_clusters=bootstrap_clusters, charge_overhead=False)

        reconfigure, target_clusters = self.decode_topology_action(
            topology_action,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            default_clusters=default_clusters,
        )

        reconfig_energy = 0.0
        if reconfigure:
            reconfig_energy = self.cluster_and_build_chains(num_clusters=target_clusters, charge_overhead=True)
        else:
            self.last_recluster_overhead = 0.0
            self.last_control_energy = 0.0

        self.last_recluster_flag = float(reconfigure)
        reward, done, info = self.transmit_data()
        if not reconfigure:
            self.topology_age += 1
            self._update_topology_descriptors()

        info.update(
            {
                "recluster": bool(reconfigure),
                "target_clusters": int(target_clusters),
                "current_clusters": int(self.current_num_clusters),
                "topology_age": int(self.topology_age),
                "cluster_balance_cv": float(self.cluster_balance_cv),
                "reconfig_energy": float(reconfig_energy),
                "recluster_count": int(self.recluster_count),
                "reward_terms": {
                    "alive_ratio": float(info.get("alive_ratio", 0.0)),
                    "normalized_data_energy": float(self._normalize_energy(info.get("data_energy_consumption", 0.0))),
                    "normalized_ctrl_energy": float(self._normalize_energy(info.get("reconfig_energy_consumption", 0.0))),
                    "imbalance": float(self.cluster_balance_cv),
                },
            }
        )
        return self.get_slim_state(), reward, done, info

    def _post_round_stats(self, round_energy, packets_sent):
        self.energy = np.maximum(self.energy, 0.0)
        self.alive = self.energy > 0
        self.last_energy_consumption = round_energy.copy()

        total_round_energy = float(np.sum(round_energy))
        reconfig_energy = float(np.sum(self.pending_reconfig_overhead))
        data_energy = float(total_round_energy - reconfig_energy)
        self.last_data_energy = data_energy
        self.last_control_energy = reconfig_energy
        self.total_energy_consumed += total_round_energy
        self.pending_reconfig_overhead.fill(0.0)

        alive_indices = np.where(self.alive)[0]
        alive_count = int(len(alive_indices))
        alive_ratio = alive_count / self.n if self.n > 0 else 0.0

        if alive_count > 0:
            alive_energy = self.energy[alive_indices]
            energy_mean = float(np.mean(alive_energy))
            energy_std = float(np.std(alive_energy))
            energy_variance = float(np.var(alive_energy))
            cv = energy_std / energy_mean if energy_mean > 0 else 0.0
            jain = (np.sum(alive_energy) ** 2) / (alive_count * np.sum(alive_energy ** 2) + 1e-8)
        else:
            energy_mean = energy_std = energy_variance = cv = jain = 0.0

        self._update_topology_descriptors()
        reward = (
            self.reward_alpha * alive_ratio
            - self.reward_beta * self._normalize_energy(data_energy)
            - self.reward_gamma * self._normalize_energy(reconfig_energy)
            - self.reward_delta * self.cluster_balance_cv
        )
        done = alive_count == 0

        self.round_count += 1
        self.total_packets_sent += packets_sent
        if self.first_node_dead_round is None and alive_count < self.n:
            self.first_node_dead_round = self.round_count
        if self.half_nodes_dead_round is None and alive_count <= self.n // 2:
            self.half_nodes_dead_round = self.round_count

        return reward, done, {
            "alive_ratio": alive_ratio,
            "packets": int(packets_sent),
            "energy_consumption": total_round_energy,
            "data_energy_consumption": data_energy,
            "reconfig_energy_consumption": reconfig_energy,
            "energy_variance": energy_variance,
            "energy_std": energy_std,
            "avg_energy": energy_mean,
            "cv": cv,
            "jain": float(jain),
            "alive_count": alive_count,
        }

    def transmit_data(self):
        if not self.leaders:
            self.pending_reconfig_overhead.fill(0.0)
            self.last_data_energy = 0.0
            self.last_control_energy = 0.0
            return 0.0, True, {
                "alive_ratio": 0.0,
                "packets": 0,
                "energy_consumption": 0.0,
                "data_energy_consumption": 0.0,
                "reconfig_energy_consumption": 0.0,
                "energy_variance": 0.0,
                "avg_energy": 0.0,
                "cv": 0.0,
                "alive_count": 0,
                "jain": 0.0,
            }

        round_energy = self.pending_reconfig_overhead.copy()
        packets_sent = 0

        for chain in self.chains:
            if len(chain) <= 1:
                continue
            for i in range(len(chain) - 1):
                current = chain[i]
                next_node = chain[i + 1]
                if not self.alive[current] or not self.alive[next_node]:
                    continue
                dist = np.linalg.norm(self.nodes[current] - self.nodes[next_node])
                e_tx = tx_energy(dist)
                e_rx = rx_energy()
                self.energy[current] -= e_tx
                self.energy[next_node] -= e_rx
                round_energy[current] += e_tx
                round_energy[next_node] += e_rx
                packets_sent += 1

        for leader in self.leaders:
            if not self.alive[leader]:
                continue
            dist_to_bs = np.linalg.norm(self.nodes[leader] - self.bs)
            e_tx = tx_energy(dist_to_bs)
            self.energy[leader] -= e_tx
            round_energy[leader] += e_tx
            packets_sent += 1

        return self._post_round_stats(round_energy=round_energy, packets_sent=packets_sent)

    def transmit_data_leach_c(self):
        if not self.leaders:
            self.pending_reconfig_overhead.fill(0.0)
            self.last_data_energy = 0.0
            self.last_control_energy = 0.0
            return 0.0, True, {
                "packets": 0,
                "alive_ratio": 0.0,
                "alive_count": 0,
                "energy_consumption": 0.0,
                "data_energy_consumption": 0.0,
                "reconfig_energy_consumption": 0.0,
                "cv": 0.0,
                "jain": 0.0,
            }

        round_energy = self.pending_reconfig_overhead.copy()
        total_packets = 0

        for cluster, ch in zip(self.clusters, self.leaders):
            if not self.alive[ch]:
                continue
            for node in cluster:
                if node == ch or not self.alive[node]:
                    continue
                d = np.linalg.norm(self.nodes[node] - self.nodes[ch])
                e_tx = tx_energy(d)
                e_rx = rx_energy()
                round_energy[node] += e_tx
                round_energy[ch] += e_rx
                self.energy[node] -= e_tx
                self.energy[ch] -= e_rx
                total_packets += 1

        for i, ch in enumerate(self.leaders):
            if not self.alive[ch]:
                continue
            d = np.linalg.norm(self.nodes[ch] - self.bs)
            cluster_size = len(self.clusters[i]) if i < len(self.clusters) else 0
            e_fusion = fusion_energy(cluster_size)
            e_tx = tx_energy(d)
            round_energy[ch] += e_tx + e_fusion
            self.energy[ch] -= (e_tx + e_fusion)
            total_packets += 1

        return self._post_round_stats(round_energy=round_energy, packets_sent=total_packets)

    def get_slim_state(self):
        alive_indices = np.where(self.alive)[0]
        if len(alive_indices) == 0:
            return np.zeros(13, dtype=np.float32)

        alive_energy = self.energy[alive_indices]
        alive_ratio = float(len(alive_indices) / self.n)
        alive_leaders = [leader for leader in self.leaders if self.alive[leader]]
        if alive_leaders:
            leader_energy = self.energy[alive_leaders]
            leader_mean = float(np.mean(leader_energy))
            leader_std = float(np.std(leader_energy))
        else:
            leader_mean = leader_std = 0.0

        normalized_cluster_count = float(self.current_num_clusters / max(self.max_cluster_hint, 1))
        normalized_topology_age = float(min(self.topology_age / 50.0, 1.0))
        return np.array(
            [
                float(np.mean(alive_energy)),
                float(np.std(alive_energy)),
                float(np.min(alive_energy)),
                float(np.max(alive_energy)),
                leader_mean,
                leader_std,
                alive_ratio,
                normalized_cluster_count,
                normalized_topology_age,
                float(self.cluster_balance_cv),
                float(self._normalize_energy(self.last_data_energy)),
                float(self._normalize_energy(self.last_control_energy)),
                float(self.last_recluster_flag),
            ],
            dtype=np.float32,
        )

    def get_full_state(self):
        return {
            "nodes": self.nodes.copy(),
            "energy": self.energy.copy(),
            "alive": self.alive.copy(),
            "leaders": self.leaders.copy() if self.leaders else [],
            "clusters": [c.copy() for c in self.clusters],
            "chains": [c.copy() for c in self.chains],
            "round": self.round_count,
            "total_packets": self.total_packets_sent,
            "total_energy_consumed": self.total_energy_consumed,
            "total_reconfig_overhead": self.total_reconfig_overhead,
            "recluster_count": self.recluster_count,
            "current_num_clusters": self.current_num_clusters,
            "topology_age": self.topology_age,
            "cluster_balance_cv": self.cluster_balance_cv,
            "last_data_energy": self.last_data_energy,
            "last_control_energy": self.last_control_energy,
            "first_node_dead": self.first_node_dead_round,
            "half_nodes_dead": self.half_nodes_dead_round,
        }
