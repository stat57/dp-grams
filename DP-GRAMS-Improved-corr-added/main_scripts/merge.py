# merge.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity
from main_scripts.ms import mean_shift
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from main_scripts.bandwidth import silverman_bandwidth
from collections import deque

def merge_modes(modes, bandwidth=None, k=1):
    if bandwidth is None:
        bandwidth = silverman_bandwidth(modes)
    threshold = k * bandwidth
    merged = []
    used = np.zeros(len(modes), dtype=bool)

    for i, mode in enumerate(modes):
        if used[i]:
            continue
        cluster = [mode]
        used[i] = True
        for j in range(i + 1, len(modes)):
            if not used[j] and np.linalg.norm(modes[j] - mode) <= threshold:
                cluster.append(modes[j])
                used[j] = True
        merged.append(np.mean(cluster, axis=0))

    print(f"[DEBUG] Merged {len(modes)} modes into {len(merged)} modes with threshold {threshold}")
    return np.array(merged)

def merge_modes_agglomerative(modes, n_clusters, random_state=None):
    """
    Merge modes using Agglomerative Clustering in a deterministic way.

    Parameters
    ----------
    modes : np.ndarray
        Array of shape (n_modes, n_features)
    n_clusters : int
        Number of clusters to merge into
    random_state : int or None
        Seed for reproducibility. Not directly used by AgglomerativeClustering
        (it is deterministic for 'ward' linkage), but kept for API consistency.

    Returns
    -------
    merged : np.ndarray
        Array of merged modes of shape (n_clusters, n_features)
    """
    if len(modes) == 0:
        return np.empty((0, modes.shape[1]))

    # AgglomerativeClustering with 'ward' is deterministic; random_state is unused
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(modes)

    merged = np.array([modes[labels == i].mean(axis=0) for i in range(n_clusters)])
    print(f"[DEBUG] Agglomerative merged {len(modes)} modes into {n_clusters} clusters.")
    return merged

def ms_merge(dp_modes, seed, k=1):
    n = len(dp_modes)
    bw = silverman_bandwidth(dp_modes)
    ms_dp_modes = mean_shift(dp_modes, initial_modes=dp_modes, T=int(np.log(n)),
                            bandwidth=bw, seed=np.random.default_rng(seed))
    return merge_modes(ms_dp_modes, bandwidth=bw, k=k)
   