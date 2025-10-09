import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh

def dp_grams(
    X,
    epsilon,
    delta,
    initial_modes=None,
    hdp=0.8,
    T=None,
    m=None,
    h=None,
    p0=0.1,
    rng=None,
    use_hdp_filter=True,
    clip_multiplier = 1.0
):
    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    if m is None:
        m = int(n / np.log(n))
    if T is None:
        T = int(np.ceil(np.log(n)))

    polylog = np.log(2 / delta) * np.log(2.5 * m * max(T, 1) / (n * delta))
    Kconst = T * d * polylog
    h_non_dp = (np.log(n) / n) ** (1 / (d + 6))
    eps_non = np.sqrt(Kconst) / (n ** (3 / (d + 6)) * np.log(n))
    eps_dp = np.sqrt(Kconst) * n ** (-3 / (d + 6)) * (np.log(n)) ** (-(d + 3) / (d + 6))

    if h is None:
        if epsilon <= eps_non:
            h = (Kconst / (n ** 2 * epsilon ** 2)) ** (1 / (2 * d + 6))
        else:
            h = h_non_dp
    h2 = h ** 2

    if initial_modes is None:
        k = max(1, int(n * p0))
        init_indices = rng.choice(n, k, replace=False)
        modes = X[init_indices].copy()
    else:
        modes = np.array(initial_modes).copy()

    if use_hdp_filter:
        kde = gaussian_kde(X.T)
        densities = kde(modes.T)
        density_threshold = hdp * densities.max()
        modes = modes[densities >= density_threshold]

    k = len(modes)
    if k == 0:
        return h, np.empty((0, d))

    epsilon_iter = epsilon / (2 * np.sqrt(2 * T * np.log(2 / delta)))

    def compute_sigma(C):
        sigma = (2 * C / m) / np.log(1 + (1 / (m / n)) * (np.exp(epsilon_iter) - 1)) * \
                np.sqrt(2 * np.log(2.5 * m * T / (n * delta)))
        if epsilon <= 1:
            sigma_approx = (8 * C / (n * epsilon)) * np.sqrt(
                T * np.log(2 / delta) * np.log(2.5 * m * T / (n * delta))
            )
            return sigma_approx
        return sigma

    def rbf_kernel_matrix(A, B=None, hloc=h):
        if B is None:
            B = A
        D2 = cdist(A, B, metric='sqeuclidean')
        return np.exp(-D2 / (2.0 * hloc * hloc))

    K_modes = rbf_kernel_matrix(modes, modes) + 1e-8 * np.eye(k)
    try:
        L_base = cholesky(K_modes)
    except Exception:
        vals, vecs = eigh(K_modes)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    modes_curr = modes.copy()
    eta = 1.0
    C = h ** (1 - d) * clip_multiplier
    sigma = compute_sigma(C)

    for t in range(T):
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

        G = rng.normal(size=(k, d))
        noise = sigma * (L_base @ G)
        modes_curr += eta * (avg_grads + noise)

    final_modes = modes_curr.copy()
    return h, final_modes
