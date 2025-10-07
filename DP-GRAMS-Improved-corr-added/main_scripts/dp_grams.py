# dp_grams.py 
import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh, eigvalsh

def dp_grams(
    X,                # dataset, shape (n, d)
    epsilon,          # privacy parameter
    delta,            # privacy parameter
    initial_modes=None,
    hdp=0.8,          # high density parameter
    T=None,           # number of iterations; if None, set internally
    m=None,           # minibatch size
    h=None,           # optional bandwidth; if None, calculate internally
    p0=0.1,           # fraction of initial points
    rng=None,         # optional random number generator
    use_hdp_filter=True  # apply HDP density filter only if True
):
    """
    DP-GRAMS: Differentially Private Gradient Ascent-based Mean Shift for Mode Estimation

    Returns:
        h, np.ndarray: bandwidth and final DP modes (k x d)
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape

    if m is None:
        m = int(n / np.log(n))

    if T is None:
        T = int(np.ceil(np.log(n)))
    print(f"[DEBUG] Iterations T = {T}")

    # bandwidth selection -- THIS IS THE THEORETICAL OPTIMAL BANDWIDTHS
    polylog = np.log(2 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
    Kconst = T * d * polylog
    h_non_dp = (np.log(n) / n) ** (1 / (d + 6))
    eps_non = np.sqrt(Kconst) / (n ** (3 / (d + 6)) * np.log(n))
    eps_dp = np.sqrt(Kconst) * n ** (-3 / (d + 6)) * (np.log(n)) ** (-(d + 3) / (d + 6))

    if h is None:
        if epsilon <= eps_non:
            h = (Kconst / (n ** 2 * epsilon ** 2)) ** (1 / (2 * d + 6))
            print(f"[DEBUG] DP regime: h_opt_DP={h:.4e}")
        elif epsilon >= eps_dp:
            h = h_non_dp
            print(f"[DEBUG] Non-DP regime: h_opt_non_DP={h:.4e}")
        else:
            h = h_non_dp
            print(f"[DEBUG] Intermediate regime: h={h:.4e}")
    else:
        print(f"[DEBUG] User provided bandwidth h={h:.4e}")
    h2 = h ** 2

    # initialize starting points
    if initial_modes is None:
        k = max(1, int(n * p0))
        init_indices = rng.choice(n, k, replace=False)
        modes = X[init_indices].copy()
    else:
        modes = np.array(initial_modes).copy()
    print(f"[DEBUG] Initialized {len(modes)} modes")

    # ----------------------------
    # Apply HDP filter only if switch is True
    # ----------------------------
    if use_hdp_filter:
        kde = gaussian_kde(X.T)
        densities = kde(modes.T)
        density_threshold = hdp * densities.max()
        keep_mask = densities >= density_threshold
        modes = modes[keep_mask]
        print(f"[DEBUG] Filtered low-density initial modes: {len(modes)} remain")
    else:
        print(f"[DEBUG] Skipping high-density filter, keeping all {len(modes)} modes")

    # If no modes left, return empty
    k = len(modes)
    if k == 0:
        print("[DEBUG] No initial modes after HDP filter; returning empty array.")
        return h, np.empty((0, d))

    epsilon_iter = epsilon / (2 * np.sqrt(2 * T * np.log(2 / delta)))
    print(f"[DEBUG] Epsilon per iteration: {epsilon_iter:.4e}")

    def compute_sigma(C):
        sigma = (2 * C / m) / np.log(1 + (1 / (m / n)) * (np.exp(epsilon_iter) - 1)) * \
                np.sqrt(2 * np.log(2.5 * m * T / (n * delta)))
        if epsilon <= 1:
            sigma_approx = (8 * C / (n * epsilon)) * np.sqrt(
                T * np.log(2 / delta) * np.log(2.5 * m * T / (n * delta))
            )
            print(f"[DEBUG] Using σ_approx (ε ≤ 1): {sigma_approx:.4e}")
            return sigma_approx
        return sigma

    # -----------------------
    # Build kernel matrix on the k initialization points
    # -----------------------
    def rbf_kernel_matrix(A, B=None, hloc=h):
        if B is None:
            B = A
        D2 = cdist(A, B, metric='sqeuclidean')
        return np.exp(-D2 / (2.0 * hloc * hloc))

    K_modes = rbf_kernel_matrix(modes, modes, hloc=h)
    jitter = 1e-8
    K_modes += jitter * np.eye(k)

    # attempt Cholesky; if it fails, fall back to eigen-decomposition sqrt
    try:
        L_base = cholesky(K_modes)
    except Exception:
        print("[DEBUG] Cholesky failed for K_modes; using eigen-decomposition sqrt.")
        vals, vecs = eigh(K_modes)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)  # shape (k,k)

    # -----------------------
    # Jointly iterate modes with correlated noise across the k modes
    # -----------------------
    modes_curr = modes.copy()   # shape (k, d)
    eta = 1.0
    C =  h ** (1 - d)  # clipping radius
    sigma = compute_sigma(C)
    print(f"[DEBUG] Base sigma (per compute_sigma): {sigma:.4e}")

    for t in range(T):
        # compute avg_grads for each mode (k x d)
        avg_grads = np.zeros_like(modes_curr)
        for i in range(k):
            x = modes_curr[i]
            batch_indices = rng.choice(n, m, replace=False)
            batch = X[batch_indices]

            diffs_full = X - x
            log_weights_full = -np.sum(diffs_full ** 2, axis=1) / (2 * h2)
            log_Z = logsumexp(log_weights_full) - np.log(n)

            diffs = batch - x
            log_weights = -np.sum(diffs ** 2, axis=1) / (2 * h2)
            q_i = diffs * np.exp(log_weights - log_Z)[:, None]

            norms = np.linalg.norm(q_i, axis=1, keepdims=True)
            q_i = q_i * np.minimum(1, C / (norms + 1e-12))

            avg_grads[i] = np.mean(q_i, axis=0)

        # Draw correlated noise: shape (k, d)
        G = rng.normal(size=(k, d))
        noise = sigma * (L_base @ G)  # each dimension gets correlated noise

        # update modes jointly
        modes_curr += eta * (avg_grads + noise)

    final_modes = modes_curr.copy()
    return h, final_modes
