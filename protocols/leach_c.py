# protocols/leach_c.py
import numpy as np

def select_leaders_leach_c(nodes, energy, bs, num_clusters):
    """
    LEACH-C: chọn num_clusters cluster head (CH) dựa trên năng lượng và vị trí.
    Fitness = E_i * (1 / d_i_to_BS) → ưu tiên nút còn nhiều năng lượng và gần BS.
    """
    dist_to_bs = np.linalg.norm(nodes - bs, axis=1)
    # Tránh chia cho 0
    fitness = energy / (dist_to_bs + 1e-8)
    # Chọn num_clusters nút có fitness cao nhất
    leaders = np.argsort(-fitness)[:num_clusters]
    return leaders.tolist()

def assign_clusters_leach_c(nodes, leaders):
    """
    Gán mỗi nút vào cụm có CH gần nhất.
    """
    clusters = [[] for _ in leaders]
    for i, node in enumerate(nodes):
        dists = np.linalg.norm(nodes[leaders] - node, axis=1)
        closest_ch = np.argmin(dists)
        clusters[closest_ch].append(i)
    return clusters
