# PMS.py
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.bandwidth import silverman_bandwidth

# Partial Mean Shift for modal regression
# -------------------------------
def partial_mean_shift(X, Y, mesh_points=None, bandwidth=None, T=20):
    """
    Partial Mean Shift: fix X, update Y along conditional density.
    X: (n,) predictor
    Y: (n,) response
    mesh_points: initial y-values for each X (default: use Y)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)

    if mesh_points is None:
        mesh_points = Y.copy()

    if bandwidth is None:
        bandwidth = silverman_bandwidth(Y.reshape(-1, 1))
        # print(f'[DEBUG] Silverman\'s bandwidth for PMS is {bandwidth}')

    Y_new = mesh_points.copy()
    for t in range(T):
        for i in range(n):
            xi = X[i]
            yi = Y_new[i]
            # Compute Gaussian weights
            w = np.exp(-((X - xi) ** 2 + (Y - yi) ** 2) / (2 * bandwidth ** 2))
            Y_new[i] = np.sum(w * Y) / np.sum(w)

    return Y_new
