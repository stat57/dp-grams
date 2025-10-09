import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import cholesky, eigh
from main_scripts.bandwidth import silverman_bandwidth

def gaussian_kernel_1d(u):
    return np.exp(-0.5 * (u ** 2))

def dp_pms_simple(
    X, Y,
    mesh_points=None,
    epsilon=1.0,
    delta=1e-5,
    T=20,
    rng=None,
    bandwidth=None,
    verbose=False
):
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    n = len(X)

    if mesh_points is None:
        mesh_points = Y.copy()
    mesh_points = np.asarray(mesh_points).reshape(-1)
    k = len(mesh_points)

    if k != n:
        raise ValueError("dp_pms_simple expects mesh_points length == len(X)")

    if bandwidth is None:
        bandwidth = silverman_bandwidth(Y.reshape(-1, 1))

    h = max(1e-3, float(bandwidth))
    h2 = h ** 2
    C = 1.0 / h / 100
    sigma = (8 * C / (n * max(1e-12, epsilon))) * np.sqrt(
        T * np.log(2.0 / delta) * np.log(2.5 * n * T / (n * delta))
    )

    D2 = cdist(mesh_points.reshape(-1,1), mesh_points.reshape(-1,1), "sqeuclidean")
    K_y = np.exp(-D2 / (2 * h2)) + 1e-8 * np.eye(k)
    try:
        L_base = cholesky(K_y)
    except Exception:
        vals, vecs = eigh(K_y)
        vals[vals < 0] = 0.0
        L_base = (vecs * np.sqrt(vals)).astype(float)

    Y_new = mesh_points.copy()

    for t in range(T):
        for i in range(k):
            xi = X[i]
            yi = Y_new[i]
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * h2))
            w_sum = np.sum(w)
            if w_sum > 0:
                Y_new[i] = np.sum(w * Y) / w_sum

        if np.isfinite(sigma) and sigma > 0:
            if k > 1:
                G = rng.normal(size=(k, 1))
                noise = (L_base @ G).flatten()
                Y_new += sigma * noise
            else:
                Y_new += rng.normal(0.0, sigma, size=1)

    return Y_new
