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

def _dbscan_merge_fallback(modes, alpha=1.5, min_samples=1, verbose=False):
    """Fallback merge using DBSCAN with eps = alpha * median NN distance."""
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:
        if verbose:
            print("[merge_modes_adaptive] sklearn not available for DBSCAN fallback:", e)
        # fallback to simple centroid-of-every-point (no merge)
        return np.array([m.copy() for m in modes])

    if len(modes) <= 1:
        return np.array(modes)

    # median nearest neighbour distance
    dists = squareform(pdist(modes))
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    median_nn = float(np.median(nn))
    eps = float(alpha * median_nn)

    if verbose:
        print(f"[merge_modes_adaptive:DBSCAN] median_nn={median_nn:.4g}, eps={eps:.4g}, alpha={alpha}")

    if median_nn == 0 or eps == 0:
        # degenerate: many identical points -> deduplicate
        uniq = np.unique(modes, axis=0)
        return uniq

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(modes)
    labels = clustering.labels_
    merged = []
    for lbl in np.unique(labels):
        merged.append(modes[labels == lbl].mean(axis=0))
    merged = np.array(merged)
    if verbose:
        print(f"[merge_modes_adaptive:DBSCAN] Merged {len(modes)} -> {len(merged)}")
    return merged

def merge_modes_adaptive(modes, sample_size, p0=.1, seed=None,
                         linkage_method='ward', ratio_threshold=1.3,
                         dbscan_alpha=1.5, min_cluster_size=1, verbose=False):
    """
    Adaptive merge of DP modes, with optional mean-shift preprocessing for large p0.

    Args:
        modes (np.ndarray): array (k x d) of DP raw modes.
        sample_size (int): size of the dataset from which DP modes were generated (n).
        p0 (float): fraction of dataset used for initialization (np0 = sample_size*p0)
        seed (int/Generator, optional): random seed for mean-shift reproducibility
        linkage_method (str): method for hierarchical clustering
        ratio_threshold (float): min relative jump for dendrogram cut
        dbscan_alpha (float): multiplier for median-NN distance in DBSCAN fallback
        min_cluster_size (int): min_samples for DBSCAN
        verbose (bool): print debug info

    Returns:
        np.ndarray: merged mode locations (num_clusters x d)
    """


    modes = np.asarray(modes, dtype=float)
    k = len(modes)
    if k == 0:
        return modes
    if k == 1:
        return modes.copy()

    # -----------------------
    # Step 0: conditional mean-shift for large p0
    # -----------------------
    np0 = p0 * sample_size
    if (p0 >= 0.1 and np0 >= 100) or np0 > sample_size / 2:
        if verbose:
            print(f"[merge_modes_adaptive] np0={np0:.1f} > n/2 -> performing mean-shift on DP modes")
        bandwidth = silverman_bandwidth(modes)
        if verbose:
            print(f"[merge_modes_adaptive] Silverman bandwidth: {bandwidth:.4g}")
        modes = mean_shift(modes, initial_modes=modes, T=int(np.log(k))**1,
                            bandwidth=bandwidth, seed=np.random.default_rng(seed), p=p0)
        if verbose:
            print(f"[merge_modes_adaptive] Mean-shift complete on DP modes")

    # -----------------------
    # Step 1: hierarchical + DBSCAN merge (original logic)
    # -----------------------
    try:
        Z = linkage(modes, method=linkage_method, metric='euclidean')
    except Exception:
        Z = linkage(pdist(modes), method=linkage_method)

    dists = Z[:, 2]  # merge distances

    if np.allclose(dists, 0):
        uniq = np.unique(modes, axis=0)
        if verbose:
            print(f"[merge_modes_adaptive] All merge distances ~0, returning {len(uniq)} unique points")
        return uniq

    # compute relative ratios
    eps_small = 1e-12
    if len(dists) == 1:
        max_ratio = np.inf
        idx = 0
    else:
        ratios = dists[1:] / (dists[:-1] + eps_small)
        idx = int(np.argmax(ratios))
        max_ratio = float(ratios[idx])

    if verbose:
        print(f"[merge_modes_adaptive] largest merge ratio={max_ratio:.4g} at index {idx}")

    if max_ratio >= ratio_threshold:
        cut = 0.5 * (dists[idx] + (dists[idx + 1] if idx + 1 < len(dists) else dists[idx]))
        labels = fcluster(Z, t=cut, criterion='distance')
        merged = np.array([modes[labels == lbl].mean(axis=0) for lbl in np.unique(labels)])
        if verbose:
            print(f"[merge_modes_adaptive] Hierarchical cut used (cut={cut:.4g}). Merged {k} -> {len(merged)}")
        return merged
    else:
        if verbose:
            print("[merge_modes_adaptive] No clear dendrogram gap found -> using DBSCAN fallback")
        return _dbscan_merge_fallback(modes, alpha=dbscan_alpha, min_samples=min_cluster_size, verbose=verbose)

def ms_merge(dp_modes, seed, k=1):
    n = len(dp_modes)
    bw = silverman_bandwidth(dp_modes)
    ms_dp_modes = mean_shift(dp_modes, initial_modes=dp_modes, T=int(np.log(n)),
                            bandwidth=bw, seed=np.random.default_rng(seed))
    return merge_modes(ms_dp_modes, bandwidth=bw, k=k)
   