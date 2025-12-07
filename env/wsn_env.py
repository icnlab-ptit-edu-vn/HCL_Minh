import numpy as np
from protocols.peg_abc import select_leaders_peg_abc, build_chain_intra
from utils.energy_model import tx_energy, rx_energy, fusion_energy


class WSNEnvironment:
    """
    Môi trường mô phỏng Wireless Sensor Network (WSN)
    Hỗ trợ clustering, chain-based routing và RL optimization
    """
    
    def __init__(self, n_nodes=100, area=100, bs=np.array([150, 50]), initial_energy=0.5):    
        """
        Khởi tạo môi trường WSN
        
        Args:
            n_nodes: Số lượng sensor nodes
            area: Kích thước vùng triển khai (area x area)
            bs: Tọa độ base station [x, y]
            initial_energy: Năng lượng ban đầu của mỗi node (Joules)
        """
        self.n = n_nodes
        self.area = area
        self.bs = np.array(bs, dtype=np.float32)
        self.initial_energy = initial_energy
        
        # Thống kê
        self.round_count = 0
        self.total_packets_sent = 0
        self.first_node_dead_round = None
        self.half_nodes_dead_round = None
        
        self.reset()

    def reset(self):
        """Reset môi trường về trạng thái ban đầu"""
        # Triển khai ngẫu nhiên các nodes
        self.nodes = np.random.rand(self.n, 2) * self.area
        
        # Khởi tạo năng lượng
        self.energy = np.full(self.n, self.initial_energy, dtype=np.float32)
        self.alive = np.ones(self.n, dtype=bool)
        
        # Cấu trúc mạng
        self.leaders = []
        self.chains = []
        self.clusters = []
        
        # Theo dõi năng lượng
        self.last_energy_consumption = np.zeros(self.n, dtype=np.float32)
        self.total_energy_consumed = 0.0
        
        # Reset thống kê
        self.round_count = 0
        self.total_packets_sent = 0
        self.first_node_dead_round = None
        self.half_nodes_dead_round = None
        
        return self.get_slim_state()

    def get_distances(self):
        """
        Tính ma trận khoảng cách giữa các nodes và khoảng cách đến BS
        
        Returns:
            dist_matrix: Ma trận khoảng cách n x n
            dist_to_bs: Mảng khoảng cách đến BS (n,)
        """
        dist_matrix = np.linalg.norm(
            self.nodes[:, None, :] - self.nodes[None, :, :], 
            axis=2
        )
        dist_to_bs = np.linalg.norm(self.nodes - self.bs, axis=1)
        return dist_matrix, dist_to_bs

    def cluster_and_build_chains(self, num_clusters=6):
        """
        Phân cụm nodes và xây dựng chains trong mỗi cluster
        
        Args:
            num_clusters: Số lượng clusters mong muốn
        """
        # Lọc các nodes còn sống
        alive_indices = np.where(self.alive)[0]
        
        if len(alive_indices) == 0:
            self.leaders = []
            self.chains = []
            self.clusters = []
            return
        
        # Điều chỉnh số clusters nếu cần
        num_clusters = min(num_clusters, len(alive_indices))
        
        # Phân cụm dựa trên tọa độ x (geographic clustering)
        x_coords = self.nodes[alive_indices, 0]
        sorted_idx = alive_indices[np.argsort(x_coords)]
        
        cluster_size = len(alive_indices) // num_clusters
        self.clusters = []
        
        for i in range(num_clusters):
            start = i * cluster_size
            if i < num_clusters - 1:
                end = start + cluster_size
            else:
                end = len(alive_indices)  # Cluster cuối lấy tất cả nodes còn lại
            
            cluster = sorted_idx[start:end].tolist()
            if len(cluster) > 0:
                self.clusters.append(cluster)

        # Chọn leaders và xây dựng chains cho mỗi cluster
        self.leaders = []
        self.chains = []
        dist_matrix, dist_to_bs = self.get_distances()

        for cluster in self.clusters:
            if len(cluster) == 0:
                continue
            
            # Lấy thông tin của cluster
            sub_nodes = self.nodes[cluster]
            sub_energy = self.energy[cluster]
            sub_dist_mat = dist_matrix[np.ix_(cluster, cluster)]
            sub_dist_bs = dist_to_bs[cluster]
            
            # Chọn leader bằng PEG-ABC
            try:
                leader_local_list = select_leaders_peg_abc(
                    sub_nodes, 
                    sub_energy, 
                    sub_dist_mat, 
                    sub_dist_bs, 
                    num_clusters=1,
                    max_iter=10,
                    num_bees=20
                )
                
                if len(leader_local_list) == 0:
                    # Fallback: chọn node có năng lượng cao nhất
                    leader_local = np.argmax(sub_energy)
                else:
                    leader_local = leader_local_list[0]
                
                leader_global = cluster[leader_local]
                self.leaders.append(leader_global)
                
            except Exception as e:
                print(f"⚠️ Lỗi chọn leader: {e}")
                # Fallback: chọn node đầu tiên
                leader_global = cluster[0]
                self.leaders.append(leader_global)
            
            # Xây dựng chain trong cluster
            try:
                local_chain = build_chain_intra(sub_nodes, bs_pos=self.bs)
                global_chain = [cluster[i] for i in local_chain]
                self.chains.append(global_chain)
            except Exception as e:
                print(f"⚠️ Lỗi xây dựng chain: {e}")
                # Fallback: chain đơn giản
                self.chains.append(cluster)

    def transmit_data(self, phi=0.5):
        """
        Truyền dữ liệu trong mạng và tính reward
        
        Args:
            phi: Ngưỡng relay (0-1)
        
        Returns:
            reward, done, info
        """
        # if len(self.leaders) == 0:
        #     return 0, True, {
        #         "alive_ratio": 0,
        #         "packets": 0,
        #         "energy_consumption": 0,
        #         "energy_variance": 0,
        #         "avg_energy": 0
        #     }
        
        if not self.leaders:
            return 0, True, {
                "alive_ratio": 0,
                "packets": 0,
                "energy_consumption": 0,
                "energy_variance": 0,
                "avg_energy": 0,
                "cv": 0,
                "alive_count": 0
            }

        
        # Reset energy consumption cho round này
        round_energy = np.zeros(self.n, dtype=np.float32)
        packets_sent = 0
        
        # ===== INTRA-CLUSTER: Truyền trong chain =====
        for chain in self.chains:
            if len(chain) <= 1:
                continue
            
            # Truyền dữ liệu theo chain
            for i in range(len(chain) - 1):
                current = chain[i]
                next_node = chain[i + 1]
                
                if not self.alive[current] or not self.alive[next_node]:
                    continue
                
                # Tính khoảng cách
                dist = np.linalg.norm(self.nodes[current] - self.nodes[next_node])
                
                # Năng lượng truyền và nhận
                e_tx = tx_energy(dist)
                e_rx = rx_energy()
                
                # Tiêu thụ năng lượng
                self.energy[current] -= e_tx
                self.energy[next_node] -= e_rx
                
                round_energy[current] += e_tx
                round_energy[next_node] += e_rx
                
                # Kiểm tra node chết
                if self.energy[current] <= 0:
                    self.alive[current] = False
                    self.energy[current] = 0
                if self.energy[next_node] <= 0:
                    self.alive[next_node] = False
                    self.energy[next_node] = 0
                
                packets_sent += 1
        
        # ===== INTER-CLUSTER: Leaders truyền đến BS =====
        for leader in self.leaders:
            if not self.alive[leader]:
                continue
            
            # Kiểm tra ngưỡng relay
            if self.energy[leader] / self.initial_energy < phi:
                # Năng lượng thấp, tìm relay
                relay = self._find_relay(leader)
                if relay is not None and self.alive[relay]:
                    # Truyền qua relay
                    dist_to_relay = np.linalg.norm(self.nodes[leader] - self.nodes[relay])
                    dist_relay_to_bs = np.linalg.norm(self.nodes[relay] - self.bs)
                    
                    e_tx_1 = tx_energy(dist_to_relay)
                    e_rx_relay = rx_energy()
                    e_tx_2 = tx_energy(dist_relay_to_bs)
                    
                    self.energy[leader] -= e_tx_1
                    self.energy[relay] -= (e_rx_relay + e_tx_2)
                    
                    round_energy[leader] += e_tx_1
                    round_energy[relay] += (e_rx_relay + e_tx_2)
                    
                    if self.energy[leader] <= 0:
                        self.alive[leader] = False
                        self.energy[leader] = 0
                    if self.energy[relay] <= 0:
                        self.alive[relay] = False
                        self.energy[relay] = 0
                    
                    packets_sent += 1
                    continue
            
            # Truyền trực tiếp đến BS
            dist_to_bs = np.linalg.norm(self.nodes[leader] - self.bs)
            e_tx = tx_energy(dist_to_bs)
            
            self.energy[leader] -= e_tx
            round_energy[leader] += e_tx
            
            if self.energy[leader] <= 0:
                self.alive[leader] = False
                self.energy[leader] = 0
            
            packets_sent += 1
        
        # ===== Tính metrics =====
        alive_count = np.sum(self.alive)
        alive_ratio = alive_count / self.n
        
        # Tổng năng lượng tiêu thụ trong round này
        total_round_energy = np.sum(round_energy)
        self.total_energy_consumed += total_round_energy
        
        # Tính energy variance (chỉ tính trên nodes còn sống)
        alive_indices = np.where(self.alive)[0]
        if len(alive_indices) > 0:
            alive_energy = self.energy[alive_indices]
            energy_mean = np.mean(alive_energy)
            energy_variance = np.var(alive_energy)  # Phương sai
            energy_std = np.std(alive_energy)       # Độ lệch chuẩn
            
            # Coefficient of Variation (CV)
            if energy_mean > 0:
                cv = energy_std / energy_mean
            else:
                cv = 0
        else:
            energy_mean = 0
            energy_variance = 0
            cv = 0
        
        # ===== Tính reward =====
        # Reward = packets sent - penalty for energy consumption - penalty for unfairness
        reward = packets_sent * 10.0 - total_round_energy * 1000.0 - cv * 5.0
        
        # ===== Check done =====
        done = (alive_ratio <= 0.3)  # Kết thúc khi 70% nodes chết
        
        # ===== Cập nhật thống kê =====
        self.round_count += 1
        self.total_packets_sent += packets_sent
        
        if self.first_node_dead_round is None and alive_count < self.n:
            self.first_node_dead_round = self.round_count
        
        if self.half_nodes_dead_round is None and alive_count <= self.n // 2:
            self.half_nodes_dead_round = self.round_count
        
        # ===== Info dict =====
        info = {
            "alive_ratio": alive_ratio,
            "packets": packets_sent,
            "energy_consumption": total_round_energy,  # Năng lượng tiêu thụ round này
            "energy_variance": energy_variance,        # Phương sai năng lượng
            "energy_std": energy_std if len(alive_indices) > 0 else 0,
            "avg_energy": energy_mean,
            "cv": cv,  # Coefficient of Variation
            "alive_count": alive_count
        }
        
        return reward, done, info

    def _find_relay(self, leader):
        """
        Tìm relay node cho leader
        
        Args:
            leader: Index của leader node
        
        Returns:
            Index của relay node hoặc None
        """
        # Tìm node còn sống, không phải leader, gần BS hơn
        candidates = []
        leader_dist_to_bs = np.linalg.norm(self.nodes[leader] - self.bs)
        
        for i in range(self.n):
            if i == leader or not self.alive[i]:
                continue
            
            dist_to_bs = np.linalg.norm(self.nodes[i] - self.bs)
            
            # Chỉ chọn node gần BS hơn leader
            if dist_to_bs < leader_dist_to_bs:
                # Tính cost = khoảng cách từ leader đến relay + relay đến BS
                dist_to_relay = np.linalg.norm(self.nodes[leader] - self.nodes[i])
                cost = dist_to_relay + dist_to_bs
                candidates.append((i, cost))
        
        if len(candidates) == 0:
            return None
        
        # Chọn relay có cost thấp nhất
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]


    def get_slim_state(self):
        """
        Trả về state vector đơn giản cho RL agent
        
        Returns:
            state: Numpy array shape (10,) chứa thông tin tóm tắt
        """
        if self.leaders is None or len(self.leaders) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Lọc leaders còn sống
        alive_leaders = [l for l in self.leaders if self.alive[l]]
        
        if len(alive_leaders) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Thông tin năng lượng của leaders
        e = self.energy[alive_leaders]
        c = self.last_energy_consumption[alive_leaders]
        
        state = np.array([
            np.mean(e),           # 0: Năng lượng trung bình leaders
            np.std(e),            # 1: Độ lệch chuẩn năng lượng
            np.min(e),            # 2: Năng lượng min
            np.max(e),            # 3: Năng lượng max
            np.mean(c),           # 4: Tiêu thụ năng lượng trung bình
            np.std(c),            # 5: Độ lệch chuẩn tiêu thụ
            np.min(c),            # 6: Tiêu thụ min
            np.max(c),            # 7: Tiêu thụ max
            len(alive_leaders),   # 8: Số leaders còn sống
            np.mean(self.alive)   # 9: Tỷ lệ nodes sống
        ], dtype=np.float32)
        
        return state
    
    def get_full_state(self):
        """
        Trả về state đầy đủ cho phân tích chi tiết
        
        Returns:
            Dictionary chứa tất cả thông tin về mạng
        """
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
            "first_node_dead": self.first_node_dead_round,
            "half_nodes_dead": self.half_nodes_dead_round
        }
    
    def render(self, mode='human'):
        """
        Hiển thị trạng thái mạng (dùng cho visualization)
        """
        if mode == 'human':
            print(f"\n{'='*50}")
            print(f"Round: {self.round_count}")
            print(f"Alive nodes: {np.sum(self.alive)}/{self.n}")
            print(f"Leaders: {len(self.leaders) if self.leaders else 0}")
            print(f"Total packets sent: {self.total_packets_sent}")
            print(f"Total energy consumed: {self.total_energy_consumed:.4f} J")
            print(f"Avg energy: {np.mean(self.energy[self.alive]):.4f} J" if np.any(self.alive) else "All dead")
            print(f"{'='*50}\n")

    # ========== LEACH-C SUPPORT ==========
    def setup_leach_c(self, num_clusters=6):
        """Thiết lập topology LEACH-C: chọn CH + gán cụm"""
        from protocols.leach_c import select_leaders_leach_c, assign_clusters_leach_c

        # Chọn leaders & assign clusters
        self.leaders = select_leaders_leach_c(self.nodes, self.energy, self.bs, num_clusters)
        self.clusters = assign_clusters_leach_c(self.nodes, self.leaders)
        # Không dùng chains trong LEACH-C
        self.chains = []

    def transmit_data_leach_c(self):
        """
        Mô phỏng 1 vòng LEACH-C:
        - Nút thường gửi đến CH.
        - CH tổng hợp rồi gửi đến BS.
        Trả về reward, done, info (với năng lượng và fairness).
        """
        if not self.leaders or len(self.leaders) == 0:
            return 0.0, True, {
                "packets": 0, "alive_ratio": 0.0,
                "energy_consumption": 0.0, "cv": 0.0, "alive_count": 0
            }

        energy_used = np.zeros(self.n, dtype=np.float32)
        total_packets = 0

        # ===== Bước 1: Nút thường → CH =====
        for cluster, ch in zip(self.clusters, self.leaders):
            for node in cluster:
                if node == ch or not self.alive[node]:
                    continue
                d = np.linalg.norm(self.nodes[node] - self.nodes[ch])
                e_tx = tx_energy(d)
                e_rx = rx_energy()
                energy_used[node] += e_tx
                energy_used[ch] += e_rx
                total_packets += 1

        # ===== Bước 2: CH → BS =====
        for i, ch in enumerate(self.leaders):
            if not self.alive[ch]:
                continue
            d = np.linalg.norm(self.nodes[ch] - self.bs)
            cluster_size = len(self.clusters[i]) if i < len(self.clusters) else 0
            e_fusion = fusion_energy(cluster_size)
            energy_used[ch] += tx_energy(d) + e_fusion
            total_packets += 1

        # ===== Cập nhật năng lượng =====
        prev_energy = self.energy.copy()
        self.energy -= energy_used
        self.energy[self.energy < 0] = 0.0
        self.alive = self.energy > 0

        # ===== Thống kê năng lượng =====
        total_round_energy = np.sum(energy_used)
        self.total_energy_consumed += total_round_energy
        alive_indices = np.where(self.alive)[0]
        alive_count = len(alive_indices)
        alive_ratio = alive_count / self.n if self.n > 0 else 0.0

        if alive_count > 0:
            alive_energy = self.energy[alive_indices]
            energy_mean = np.mean(alive_energy)
            energy_std = np.std(alive_energy)
            energy_variance = np.var(alive_energy)
            cv = energy_std / energy_mean if energy_mean > 0 else 0.0
            jain = (np.sum(alive_energy) ** 2) / (alive_count * np.sum(alive_energy ** 2) + 1e-8)
        else:
            energy_mean = energy_std = energy_variance = cv = jain = 0.0

        # ===== Reward mới =====
        # - Thưởng số packet lớn
        # - Cộng thêm fairness
        # - Cộng thêm 0.2 * alive_ratio để khuyến khích duy trì mạng
        # reward = total_packets + 0.5 * jain + 0.2 * alive_ratio
        reward = total_packets + 0.5 * jain - 0.2 * max(0, 0.8 - jain)

        done = (alive_ratio <= 0.3)

        # ===== Lưu thông tin =====
        info = {
            "packets": total_packets,
            "alive_ratio": alive_ratio,
            "alive_count": alive_count,
            "energy_consumption": total_round_energy,
            "energy_variance": energy_variance,
            "energy_std": energy_std,
            "avg_energy": energy_mean,
            "cv": cv,
            "jain": jain
        }

        # ===== Cập nhật thống kê vòng =====
        self.round_count += 1
        self.total_packets_sent += total_packets
        if self.first_node_dead_round is None and alive_count < self.n:
            self.first_node_dead_round = self.round_count
        if self.half_nodes_dead_round is None and alive_count <= self.n // 2:
            self.half_nodes_dead_round = self.round_count

        return reward, done, info


    def transmit_data_leach_c_old(self):
        """
        Mô phỏng 1 vòng LEACH-C:
        - Mỗi nút (không phải CH) gửi dữ liệu đến CH của mình.
        - Mỗi CH gộp dữ liệu và gửi đến BS.
        """
        if not self.leaders:
            return 0, True, {"packets": 0, "alive_ratio": 0.0}

        energy_used = np.zeros(self.n, dtype=np.float32)
        total_packets = 0

        # Bước 1: Nút thường → CH
        for cluster, ch in zip(self.clusters, self.leaders):
            for node in cluster:
                if node == ch or not self.alive[node]:
                    continue
                d = np.linalg.norm(self.nodes[node] - self.nodes[ch])
                energy_used[node] += tx_energy(d)
                energy_used[ch] += rx_energy()
                total_packets += 1

        # Bước 2: CH → BS
        for i, ch in enumerate(self.leaders):
            if not self.alive[ch]:
                continue
            d = np.linalg.norm(self.nodes[ch] - self.bs)
            # fusion_energy expects cluster size; safe guard index
            cluster_size = len(self.clusters[i]) if i < len(self.clusters) else 1
            energy_used[ch] += tx_energy(d) + fusion_energy(cluster_size)
            total_packets += 1

        # Cập nhật năng lượng
        self.energy -= energy_used
        self.energy = np.maximum(self.energy, 0.0)
        self.alive = self.energy > 0

        # Tính reward / metrics
        alive_energy = self.energy[self.alive]
        if len(alive_energy) == 0:
            jain = 0
        else:
            jain = (np.sum(alive_energy) ** 2) / (len(alive_energy) * np.sum(alive_energy ** 2) + 1e-8)

        #reward = total_packets + 0.5 * jain
        reward = total_packets + 0.5 * jain - 0.2 * max(0, 0.8 - jain)
        done = np.sum(self.alive) <= 0.3 * self.n

        info = {
            "packets": total_packets,
            "alive_ratio": np.mean(self.alive),
            "jain": jain
        }

        # update stats
        self.round_count += 1
        self.total_packets_sent += total_packets

        if self.first_node_dead_round is None and np.sum(self.alive) < self.n:
            self.first_node_dead_round = self.round_count

        if self.half_nodes_dead_round is None and np.sum(self.alive) <= self.n // 2:
            self.half_nodes_dead_round = self.round_count

        return reward, done, info
        # ========== END LEACH-C SUPPORT ==========


# Cách dùng
# Khởi tạo
env = WSNEnvironment(n_nodes=100, area=100, bs=np.array([150, 50]))
# Reset
state = env.reset()
# Chạy 1 round
env.cluster_and_build_chains(num_clusters=6)
reward, done, info = env.transmit_data(phi=0.5)
# Hiển thị
env.render()
# Lấy state đầy đủ
full_state = env.get_full_state()
