# dpms_cluster.py 

import numpy as np
from typing import Optional, Tuple
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.dp_grams import dp_grams
from main_scripts.merge import *
from main_scripts.ms import mean_shift

def _child_rng(master):
    seed = int(master.integers(0, 2**31 - 1))
    return np.random.default_rng(seed)

def dpms_private_rr(
    data: np.ndarray,
    epsilon_modes: float,
    epsilon_rr: float,
    delta: float = 1e-5,
    p_seed: float = 0.1,
    use_hdp_filter = False,
    hdp = 0.1,
    bandwidth_multiplier = 1,
    rng: Optional[np.random.Generator] = None,
    k_est: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:

    if rng is None:
        rng = np.random.default_rng()

    n, d = data.shape
    k_for_rr = k_est if k_est is not None else max(1, int(n * p_seed))
    rr_p = np.exp(epsilon_rr) / (np.exp(epsilon_rr) + k_for_rr - 1)
    print(f"[DEBUG] eps_modes={epsilon_modes:.4f}, eps_rr={epsilon_rr:.4f}, rr_p={rr_p:.4f}")

    h = silverman_bandwidth(data) * bandwidth_multiplier

    child = _child_rng(rng)
    # Unpack dp_grams output: h (bandwidth) and dp_modes
    _, dp_modes = dp_grams(
        X=data,
        epsilon=epsilon_modes,
        h=h,
        delta=delta,
        p0=p_seed,
        rng=child, 
        hdp=hdp,
        use_hdp_filter=use_hdp_filter
    )

    print(f"[DEBUG] Number of raw DP modes before merging/refinement: {dp_modes.shape[0]}")

    # Merge modes 
    if k_est is None:
        merged_modes = merge_modes(dp_modes)
        k_final = merged_modes.shape[0]
        print(f"[DEBUG] Refined DP modes into {k_final} clusters (k_est=None)")
    else:
        merged_modes = merge_modes_agglomerative(dp_modes, n_clusters=k_est)
        print(f"[DEBUG] Agglomerative merged DP modes into {k_est} clusters (k_est={k_est})")

    if k_est is None:
        rr_p = np.exp(epsilon_rr) / (np.exp(epsilon_rr) + merged_modes.shape[0] - 1)
        print(f"[DEBUG] Updated RR probability p = {rr_p:.4f} after refinement")

    # Assign labels
    diffs = data[:, None, :] - merged_modes[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    labels_true = np.argmin(dists, axis=1)

    child_rr = _child_rng(rng)
    k = merged_modes.shape[0]
    labels_rr = np.empty(n, dtype=int)
    for i in range(n):
        if child_rr.random() < rr_p:
            labels_rr[i] = labels_true[i]
        else:
            choices = [x for x in range(k) if x != labels_true[i]]
            labels_rr[i] = child_rr.choice(choices)

    return merged_modes, labels_rr, rr_p
