# ms.py
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.bandwidth import silverman_bandwidth

def mean_shift_step(data, mode, bandwidth):
    kernel = np.exp(-np.linalg.norm(data - mode, axis=1) ** 2 / (2 * bandwidth ** 2))
    return np.sum(kernel[:, np.newaxis] * data, axis=0) / kernel.sum()

def mean_shift(data, initial_modes=None, T=20, bandwidth=None, p=1, seed=None):
    """
    Standard mean shift algorithm with optional deterministic seeding.
    """
    rng = np.random.default_rng(seed)

    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
        print(f"[DEBUG] Bandwidth estimated by Silverman: {bandwidth}")

    if initial_modes is None:
        n_samples = max(1, int(len(data) * p))
        indices = rng.choice(len(data), size=n_samples, replace=False)
        initial_modes = data[indices]
        print(f"[DEBUG] Initial modes randomly selected: {n_samples} points (p={p})")

    modes = initial_modes
    for t in range(T):
        modes = np.array([mean_shift_step(data, mode, bandwidth) for mode in modes])
        print(f"[DEBUG] Iteration {t + 1}/{T} complete.")

    return modes
