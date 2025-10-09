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
    clip_multiplier = 1,
    rng: Optional[np.random.Generator] = None,
    k_est: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:

    if rng is None:
        rng = np.random.default_rng()

    n, d = data.shape
    k_for_rr = k_est if k_est is not None else max(1, int(n * p_seed))
    rr_p = np.exp(epsilon_rr) / (np.exp(epsilon_rr) + k_for_rr - 1)

    h = silverman_bandwidth(data) * bandwidth_multiplier

    child = _child_rng(rng)
    _, dp_modes = dp_grams(
        X=data,
        epsilon=epsilon_modes,
        h=h,
        delta=delta,
        p0=p_seed,
        rng=child, 
        hdp=hdp,
        use_hdp_filter=use_hdp_filter,
        clip_multiplier=clip_multiplier
    )

    if k_est is None:
        merged_modes = merge_modes(dp_modes)
        k_final = merged_modes.shape[0]
    else:
        merged_modes = merge_modes_agglomerative(dp_modes, n_clusters=k_est)

    if k_est is None:
        rr_p = np.exp(epsilon_rr) / (np.exp(epsilon_rr) + merged_modes.shape[0] - 1)

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
