import numpy as np
from main_scripts.bandwidth import silverman_bandwidth

def mean_shift_step(data, mode, bandwidth):
    kernel = np.exp(-np.linalg.norm(data - mode, axis=1) ** 2 / (2 * bandwidth ** 2))
    return np.sum(kernel[:, np.newaxis] * data, axis=0) / kernel.sum()

def mean_shift(data, initial_modes=None, T=20, bandwidth=None, p=1, seed=None):
    rng = np.random.default_rng(seed)

    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)

    if initial_modes is None:
        n_samples = max(1, int(len(data) * p))
        indices = rng.choice(len(data), size=n_samples, replace=False)
        initial_modes = data[indices]

    modes = initial_modes
    for _ in range(T):
        modes = np.array([mean_shift_step(data, mode, bandwidth) for mode in modes])

    return modes
