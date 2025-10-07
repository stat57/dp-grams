# dp_pms.py
import numpy as np
import math, sys, os
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.bandwidth import silverman_bandwidth

def gaussian_kernel_1d(u):
    return np.exp(-0.5 * (u ** 2))

# --------------------------------------------------
# Optimized DP-PMS with initialization filtering
# --------------------------------------------------
def dp_pms(
    X, Y,
    mesh_X=None,
    epsilon=1.0,
    delta=1e-5,
    T_override=None,
    rng=None,
    filter_high_density=False,
    hdp=0.8,
    h_x=None,
    h_y=None,
    clip_multiplier=1,
    verbose=True
):
    """
    Differentially Private Partial Mean-Shift (DP-PMS) for Y|X.

    Args:
        X, Y: data arrays of shape (n,)
        mesh_X: optional mesh points for evaluation
        epsilon, delta: DP parameters
        T_override: optional number of iterations
        rng: optional np.random.Generator
        filter_high_density: whether to filter Y candidates by local density
        hdp: threshold for high-density filtering (0..1)
        h_x, h_y: optional bandwidths for X and Y
        clip_multiplier: scales clip
        verbose: print progress

    Returns:
        modes_per_mesh: list of arrays of Y candidates per mesh point
        mesh_X_vals: mesh points
    """

    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same length")
    n = len(X)

    mesh_X_vals = X.copy() if mesh_X is None else np.asarray(mesh_X).reshape(-1)

    # ------------------------------
    # Bandwidths
    # ------------------------------
    if h_x is None:
        h_x = silverman_bandwidth(X.reshape(-1,1))
    if h_y is None:
        h_y = silverman_bandwidth(Y.reshape(-1,1))
    h_x = max(1e-3, h_x) 
    h_y = max(1e-3, h_y) 
    h2 = h_y ** 2

    # ------------------------------
    # High-density filtering
    # ------------------------------
    
    filtered_Y_local_list = []
    if filter_high_density:
        for x0 in mesh_X_vals:
            kx = gaussian_kernel_1d((X - x0) / h_x)
            f_X = np.sum(kx) / n + 1e-12

            # Compute conditional density Y|X
            f_Y_given_X = np.zeros_like(Y)
            for i, y_val in enumerate(Y):
                ky = gaussian_kernel_1d((Y - y_val) / h_y)
                f_Y_given_X[i] = np.sum(kx * ky) / n
            f_Y_given_X /= f_X

            # Always filter by high-density
            mask = f_Y_given_X >= hdp * np.max(f_Y_given_X)
            idx_local = np.nonzero(mask)[0]

            filtered_Y_local_list.append(Y[idx_local])
    else:
        filtered_Y_local_list = [Y.copy() for _ in mesh_X_vals]


    # ------------------------------
    # Iteration count
    # ------------------------------
    T_global = max(1, int(math.ceil(np.log(max(2, n))))) if T_override is None else int(T_override)
    if verbose:
        print(f"[dp_pms] T_global = {T_global}, h_y = {h_y:.6e}, h_x = {h_x:.6e}")

    # per-iteration epsilon
    epsilon_iter = epsilon / (2 * np.sqrt(2 * T_global * np.log(2 / delta)))
    if verbose:
        print(f"[dp_pms] epsilon_iter = {epsilon_iter:.4e}")

    def compute_sigma(C, n_local):
        """
        Compute DP noise scale (sigma) without mini-batching.
        """
        sigma_approx = (8 * C / (n_local * max(1e-12, epsilon_iter))) * np.sqrt(
            T_global * np.log(2.0 / delta) * np.log(2.5 * max(1, n_local) * T_global / (n_local * delta))
        )
        return sigma_approx

    # ------------------------------
    # Iterate mesh points
    # ------------------------------
    modes_per_mesh = []
    sigma_printer = True

    for mi, x0 in enumerate(mesh_X_vals):
        Y_candidates = filtered_Y_local_list[mi].copy()
        n_local = len(Y_candidates)
        if n_local == 0:
            modes_per_mesh.append(np.array([]))
            continue

        k_cand = len(Y_candidates)

        # Precompute X kernel weights
        kx_local = gaussian_kernel_1d((X - x0) / h_x)
        log_kx_local = np.log(kx_local + 1e-300)

        # Correlated noise base
        if k_cand > 1:
            D2 = cdist(Y_candidates.reshape(-1,1), Y_candidates.reshape(-1,1), metric="sqeuclidean")
            K_y = np.exp(-D2 / (2 * h_y**2)) + 1e-8 * np.eye(k_cand)
            try:
                L_base = cholesky(K_y)
            except Exception:
                vals, vecs = eigh(K_y)
                vals[vals < 0] = 0
                L_base = (vecs * np.sqrt(vals)).astype(float)
        else:
            L_base = None

        C = 1 / h_y * clip_multiplier
        sigma = compute_sigma(C, n)
        
        if sigma_printer:
            print(f"[DEBUG] sigma = {sigma}")
            sigma_printer = False

        if verbose and mi % max(1, len(mesh_X_vals)//10) == 0:
            print(f"[dp_pms] Progress: mesh {mi+1}/{len(mesh_X_vals)}")

        # ------------------------------
        # Iterative updates
        # ------------------------------
        for t in range(T_global):
            diffs_matrix = Y[:, None] - Y_candidates[None, :]
            log_ky = -(diffs_matrix ** 2) / (2.0 * h2)
            log_weights = log_kx_local[:, None] + log_ky
            log_Z = logsumexp(log_weights, axis=0) - np.log(max(1, n))
            inv_norm_factors = np.exp(log_weights - log_Z)
            q_i = diffs_matrix * inv_norm_factors

            avg_qs = np.mean(np.clip(q_i, -C, C), axis=0)
            Y_candidates += avg_qs

            # Correlated noise
            if k_cand > 1 and np.isfinite(sigma) and sigma > 0:
                G = rng.normal(size=(k_cand, 1))
                Y_candidates += sigma * (L_base @ G).flatten()
            elif k_cand == 1 and np.isfinite(sigma) and sigma > 0:
                Y_candidates += rng.normal(0.0, sigma, size=1)

        modes_per_mesh.append(Y_candidates)

    return modes_per_mesh, mesh_X_vals
