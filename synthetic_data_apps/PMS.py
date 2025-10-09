import numpy as np
from main_scripts.bandwidth import silverman_bandwidth

def partial_mean_shift(X, Y, mesh_points=None, bandwidth=None, T=20):
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)

    if mesh_points is None:
        mesh_points = Y.copy()

    if bandwidth is None:
        bandwidth = silverman_bandwidth(Y.reshape(-1, 1))

    Y_new = mesh_points.copy()
    for _ in range(T):
        for i in range(n):
            xi = X[i]
            yi = Y_new[i]
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * bandwidth ** 2))
            denom = np.sum(w)
            if denom > 0:
                Y_new[i] = np.sum(w * Y) / denom

    return Y_new
