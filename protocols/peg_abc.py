import numpy as np
def select_leaders_peg_abc(nodes, energy, dist_matrix, dist_to_bs, num_clusters, max_iter=10, num_bees=40):
    """
    Chọn Cluster Heads sử dụng thuật toán PEG-ABC (Artificial Bee Colony)
    
    Args:
        nodes: Danh sách tọa độ các node
        energy: Mảng năng lượng còn lại của mỗi node
        dist_matrix: Ma trận khoảng cách giữa các node
        dist_to_bs: Khoảng cách từ mỗi node đến base station
        num_clusters: Số lượng cluster heads cần chọn
        max_iter: Số vòng lặp tối đa của ABC
        num_bees: Số lượng ong (solutions) trong quần thể
    
    Returns:
        List các chỉ số của cluster heads được chọn
    """
    n = len(nodes)
    
    # Xử lý trường hợp đặc biệt
    if num_clusters >= n:
        return list(range(n))
    if num_clusters <= 0:
        return []
    
    # Lọc các node còn năng lượng
    alive_nodes = np.where(energy > 0)[0]
    if len(alive_nodes) == 0:
        return list(range(min(num_clusters, n)))
    if len(alive_nodes) < num_clusters:
        return alive_nodes.tolist()
    
    # Tính QV (Quality Value) cho toàn mạng
    DmaxBS, DminBS = dist_to_bs.max(), dist_to_bs.min()
    sum_dist = dist_matrix.sum(axis=1)
    Dmax, Dmin = sum_dist.max(), sum_dist.min()
    Emax, Emin = energy.max(), energy.min()

    # Chuẩn hóa các chỉ số về khoảng [0, 1]
    # Thêm epsilon để tránh chia cho 0
    eps = 1e-8
    norm_dist_bs = (DmaxBS - dist_to_bs) / (DmaxBS - DminBS + eps)  # Gần BS hơn = tốt hơn
    norm_sum_dist = (sum_dist - Dmin) / (Dmax - Dmin + eps)          # Vị trí trung tâm = tốt hơn
    norm_energy = (energy - Emin) / (Emax - Emin + eps)              # Năng lượng cao = tốt hơn
    
    # Quality Value: tổ hợp có trọng số (40% gần BS, 30% trung tâm, 30% năng lượng)
    QV = 0.4 * norm_dist_bs + 0.3 * norm_sum_dist + 0.3 * norm_energy
    
    # Đặt QV = 0 cho các node hết năng lượng
    QV[energy <= 0] = 0

    def random_solution():
        """Tạo giải pháp ngẫu nhiên (chọn CHs)"""
        candidates = alive_nodes.copy()
        if len(candidates) >= num_clusters:
            return np.random.choice(candidates, size=num_clusters, replace=False)
        else:
            return candidates

    def evaluate_solution(sol):
        """Đánh giá fitness của một giải pháp"""
        if len(sol) == 0:
            return 0
        return np.mean(QV[sol])

    # Khởi tạo quần thể ong
    population = [random_solution() for _ in range(num_bees)]
    fitness = np.array([evaluate_solution(sol) for sol in population])
    
    # Tìm giải pháp tốt nhất ban đầu
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Đếm số lần không cải thiện cho mỗi giải pháp (cho Scout bees)
    trial = np.zeros(num_bees, dtype=int)
    limit = num_clusters * 2  # Giới hạn số lần thử

    for it in range(max_iter):
        # ===== WORKER BEES PHASE =====
        # Khai thác các giải pháp hiện tại
        for i in range(num_bees):
            sol = population[i].copy()
            
            # Chọn ngẫu nhiên một vị trí để thay đổi
            idx = np.random.randint(len(sol))
            
            # Tìm các ứng viên thay thế (không trùng với CHs hiện tại)
            candidates = [j for j in alive_nodes if j not in sol]
            
            if not candidates:
                trial[i] += 1
                continue
            
            # Thay thế một CH bằng ứng viên ngẫu nhiên
            new_ch = np.random.choice(candidates)
            sol[idx] = new_ch
            
            # Đánh giá giải pháp mới
            new_fit = evaluate_solution(sol)
            
            # Greedy selection: chỉ chấp nhận nếu tốt hơn
            if new_fit > fitness[i]:
                population[i] = sol
                fitness[i] = new_fit
                trial[i] = 0  # Reset trial counter
                
                # Cập nhật best solution
                if new_fit > best_fitness:
                    best_fitness = new_fit
                    best_solution = sol.copy()
            else:
                trial[i] += 1

        # ===== OBSERVER BEES PHASE =====
        # Chọn giải pháp theo xác suất dựa trên fitness
        prob = fitness / (np.sum(fitness) + eps)
        
        for _ in range(num_bees):
            # Chọn một giải pháp theo xác suất (giải pháp tốt có xác suất cao hơn)
            i = np.random.choice(num_bees, p=prob)
            sol = population[i].copy()
            
            # Thực hiện tương tự Worker bees
            idx = np.random.randint(len(sol))
            candidates = [j for j in alive_nodes if j not in sol]
            
            if not candidates:
                continue
            
            new_ch = np.random.choice(candidates)
            sol[idx] = new_ch
            new_fit = evaluate_solution(sol)
            
            if new_fit > fitness[i]:
                population[i] = sol
                fitness[i] = new_fit
                trial[i] = 0
                
                if new_fit > best_fitness:
                    best_fitness = new_fit
                    best_solution = sol.copy()
            else:
                trial[i] += 1
        
        # ===== SCOUT BEES PHASE =====
        # Thay thế các giải pháp kém bằng giải pháp ngẫu nhiên mới
        for i in range(num_bees):
            if trial[i] >= limit:
                population[i] = random_solution()
                fitness[i] = evaluate_solution(population[i])
                trial[i] = 0

    return best_solution.tolist()


def build_chain_intra(node_coords, bs_pos=None):
    """
    Xây dựng chuỗi intra-cluster sử dụng thuật toán greedy nearest-neighbor
    Tương tự giao thức PEGASIS
    
    Args:
        node_coords: Tọa độ các node trong cluster
        bs_pos: Vị trí base station (tùy chọn)
    
    Returns:
        List các chỉ số node tạo thành chuỗi
    """
    if len(node_coords) == 0:
        return []
    
    node_coords = np.array(node_coords)
    m = len(node_coords)
    
    # Xử lý các trường hợp đơn giản
    if m == 1:
        return [0]
    
    if m == 2:
        if bs_pos is not None:
            bs_pos = np.array(bs_pos)
            d0 = np.linalg.norm(node_coords[0] - bs_pos)
            d1 = np.linalg.norm(node_coords[1] - bs_pos)
            # Node gần BS hơn đặt ở cuối chuỗi (làm leader)
            return [1, 0] if d0 < d1 else [0, 1]
        else:
            return [0, 1]

    # Tính ma trận khoảng cách giữa các node
    dist_matrix = np.linalg.norm(node_coords[:, None] - node_coords, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # Tránh chọn chính nó

    # Xác định điểm bắt đầu: node xa BS nhất
    if bs_pos is not None:
        bs_pos = np.array(bs_pos)
        dist_to_bs = np.linalg.norm(node_coords - bs_pos, axis=1)
        start = np.argmax(dist_to_bs)  # Bắt đầu từ node xa BS nhất
    else:
        # Nếu không có BS, bắt đầu từ node xa trung tâm nhất
        center = np.mean(node_coords, axis=0)
        dist_to_center = np.linalg.norm(node_coords - center, axis=1)
        start = np.argmax(dist_to_center)

    # Xây dựng chuỗi bằng thuật toán greedy: luôn kết nối với node gần nhất chưa thăm
    visited = np.zeros(m, dtype=bool)
    chain = []
    current = start
    visited[current] = True
    chain.append(current)

    for _ in range(m - 1):
        row = dist_matrix[current].copy()
        row[visited] = np.inf  # Loại trừ các node đã thăm
        next_node = np.argmin(row)
        
        # Kiểm tra nếu không còn node nào để kết nối
        if np.isinf(row[next_node]):
            break
        
        visited[next_node] = True
        chain.append(next_node)
        current = next_node

    # Đảm bảo node gần BS nhất ở cuối chuỗi (làm chain leader/CH)
    if bs_pos is not None:
        dist_to_bs = np.linalg.norm(node_coords - bs_pos, axis=1)
        leader = np.argmin(dist_to_bs)  # Node gần BS nhất
        
        if leader != chain[-1]:
            if leader == chain[0]:
                # Nếu leader ở đầu, đảo ngược chuỗi
                chain = chain[::-1]
            else:
                # Nếu leader ở giữa, di chuyển nó ra cuối
                chain.remove(leader)
                chain.append(leader)

    return chain


def calculate_chain_energy(chain, node_coords, energy, bs_pos):
    """
    Tính tổng năng lượng tiêu thụ trong chuỗi
    
    Args:
        chain: List các chỉ số node trong chuỗi
        node_coords: Tọa độ các node
        energy: Năng lượng còn lại của các node
        bs_pos: Vị trí base station
    
    Returns:
        Tổng năng lượng tiêu thụ
    """
    if len(chain) <= 1:
        return 0
    
    from utils.energy_model import tx_energy, rx_energy
    
    total_energy = 0
    node_coords = np.array(node_coords)
    bs_pos = np.array(bs_pos)
    
    # Năng lượng truyền dữ liệu trong chuỗi
    for i in range(len(chain) - 1):
        current = chain[i]
        next_node = chain[i + 1]
        
        # Kiểm tra năng lượng còn lại
        if energy[current] <= 0:
            continue
        
        # Khoảng cách giữa 2 node liên tiếp
        dist = np.linalg.norm(node_coords[current] - node_coords[next_node])
        
        # Năng lượng truyền + nhận
        total_energy += tx_energy(dist) + rx_energy()
    
    # Node cuối cùng (leader) truyền đến BS
    leader = chain[-1]
    if energy[leader] > 0:
        dist_to_bs = np.linalg.norm(node_coords[leader] - bs_pos)
        total_energy += tx_energy(dist_to_bs)
    
    return total_energy


def optimize_chain_order(chain, node_coords, energy, bs_pos, max_attempts=10):
    """
    Tối ưu thứ tự chuỗi bằng local search
    
    Args:
        chain: Chuỗi ban đầu
        node_coords: Tọa độ các node
        energy: Năng lượng còn lại
        bs_pos: Vị trí base station
        max_attempts: Số lần thử tối ưu
    
    Returns:
        Chuỗi đã được tối ưu
    """
    if len(chain) <= 2:
        return chain
    
    best_chain = chain.copy()
    best_energy = calculate_chain_energy(best_chain, node_coords, energy, bs_pos)
    
    for _ in range(max_attempts):
        # Thử hoán đổi 2 node ngẫu nhiên (không phải leader)
        new_chain = best_chain.copy()
        i, j = np.random.choice(len(chain) - 1, size=2, replace=False)
        new_chain[i], new_chain[j] = new_chain[j], new_chain[i]
        
        new_energy = calculate_chain_energy(new_chain, node_coords, energy, bs_pos)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain
