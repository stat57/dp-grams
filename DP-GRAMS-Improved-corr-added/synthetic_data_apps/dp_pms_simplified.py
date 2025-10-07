# dp_pms_simplified.py
import numpy as np
import os, sys, math
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.bandwidth import silverman_bandwidth

def gaussian_kernel_1d(u):
    return np.exp(-0.5 * (u ** 2))

# --------------------------------------------------
# Simplified Differentially Private Partial Mean Shift (DP-PMS)
# --------------------------------------------------
def dp_pms_simple(
    X, Y,
    mesh_points=None,
    epsilon=1.0,
    delta=1e-5,
    T=20,
    rng=None,
    bandwidth=None,
    verbose=True
):
    """
    Simplified DP Partial Mean Shift (debug version).

    Structure mirrors partial_mean_shift but injects DP noise once per iteration:
      - correlated noise across the mesh candidates (using kernel matrix L_base)
      - no separate init noise and no extra independent per-iteration noise

    Args:
        X, Y: arrays of shape (n,)
        mesh_points: initial Y candidates (default: copy of Y)
        epsilon, delta: DP parameters
        T: number of iterations
        rng: np.random.Generator
        bandwidth: kernel bandwidth
        verbose: print progress info

    Returns:
        Y_modes: final shifted Y candidates (1D array same length as mesh_points)
    """

    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    n = len(X)

    if mesh_points is None:
        mesh_points = Y.copy()
    mesh_points = np.asarray(mesh_points).reshape(-1)
    k = len(mesh_points)

    # For this simplified debug version we require mesh_points to be aligned with X
    # (one candidate per sample). If you need different behavior, we can relax this.
    if k != n:
        raise ValueError("dp_pms_simple expects mesh_points length == len(X) for this simplified debug version")

    if bandwidth is None:
        bandwidth = silverman_bandwidth(Y.reshape(-1, 1))

    h = max(1e-3, float(bandwidth))
    h2 = h ** 2

    if verbose:
        print(f"[dp_pms_simple] bandwidth = {h:.6e}, T = {T}")

    # Simple DP noise scale (no advanced accounting)
    C = 1.0 / h / 100#/ n
    sigma = (8 * C / (n * max(1e-12, epsilon))) * np.sqrt(
        T * np.log(2.0 / delta) * np.log(2.5 * n * T / (n * delta))
    )
    if verbose:
        print(f"[dp_pms_simple] sigma = {sigma:.4e}")

    # Build correlated-noise base matrix for the mesh candidates (k x k)
    D2 = cdist(mesh_points.reshape(-1,1), mesh_points.reshape(-1,1), "sqeuclidean")
    K_y = np.exp(-D2 / (2 * h2)) + 1e-8 * np.eye(k)
    try:
        L_base = cholesky(K_y)
    except Exception:
        vals, vecs = eigh(K_y)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    # Initialize candidates deterministically (no init noise)
    Y_new = mesh_points.copy()

    # ------------------------------
    # Iterative updates with correlated noise (once per iteration)
    # ------------------------------
    for t in range(T):
        # mean-shift style update (deterministic part)
        for i in range(k):
            xi = X[i]
            yi = Y_new[i]

            # Gaussian weights for conditional Y|X
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * h2))
            w_sum = np.sum(w)
            if w_sum > 0:
                Y_new[i] = np.sum(w * Y) / w_sum

        # Add correlated Gaussian noise across the k candidates (single addition per iteration)
        if np.isfinite(sigma) and sigma > 0:
            if k > 1:
                G = rng.normal(size=(k, 1))
                noise = (L_base @ G).flatten()  # correlated shape (k,)
                Y_new += sigma * noise
            else:
                # scalar case
                Y_new += rng.normal(0.0, sigma, size=1)

        if verbose and (t % max(1, T // 5) == 0):
            print(f"[dp_pms_simple] iteration {t+1}/{T}")

    return Y_new
