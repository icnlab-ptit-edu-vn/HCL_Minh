import numpy as np


def select_leaders_leach_c(nodes, energy, bs, num_clusters):
    """
    LEACH-C leader selection.
    Fitness = residual energy / distance-to-BS.
    """
    if len(nodes) == 0 or num_clusters <= 0:
        return []
    num_clusters = max(1, min(int(num_clusters), len(nodes)))
    dist_to_bs = np.linalg.norm(nodes - bs, axis=1)
    fitness = energy / (dist_to_bs + 1e-8)
    leaders = np.argsort(-fitness)[:num_clusters]
    return leaders.tolist()


def assign_clusters_leach_c(nodes, leaders, member_indices=None):
    """
    Assign each member node to its nearest cluster head.

    Args:
        nodes: all node coordinates
        leaders: global indices of cluster heads
        member_indices: optional iterable of node indices to assign.
            If omitted, all nodes are assigned.
    """
    if leaders is None or len(leaders) == 0:
        return []

    if member_indices is None:
        member_indices = np.arange(len(nodes))
    member_indices = np.asarray(member_indices, dtype=int)

    clusters = [[] for _ in leaders]
    leader_coords = nodes[np.asarray(leaders, dtype=int)]
    for node_idx in member_indices:
        dists = np.linalg.norm(leader_coords - nodes[node_idx], axis=1)
        closest_ch = int(np.argmin(dists))
        clusters[closest_ch].append(int(node_idx))
    return clusters
