# mode_matching_mse.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def mode_matching_mse(true_modes, est_modes, penalty=None):
    """
    Compute MSE between true and estimated modes using optimal matching.

    Args:
        true_modes (np.ndarray): Array of shape (k, d) of true mode coordinates.
        est_modes (np.ndarray): Array of shape (k_hat, d) of estimated mode coordinates.
        penalty (float, optional): Penalty for unmatched modes. If None, extra/missing modes are ignored.

    Returns:
        float: Mean squared error (MSE) between true and estimated modes.
    """
    true_modes = np.atleast_2d(true_modes)
    est_modes = np.atleast_2d(est_modes)

    # If either is empty, return a large MSE or NaN
    if true_modes.size == 0 or est_modes.size == 0:
        print("Warning: true_modes or est_modes is empty")
        return np.nan

    # Compute pairwise squared distances
    dist_matrix = np.sum((true_modes[:, None, :] - est_modes[None, :, :])**2, axis=2)

    # Replace invalid numbers with a large finite value
    if np.isnan(dist_matrix).any() or np.isinf(dist_matrix).any():
        print("Warning: dist_matrix contains NaN/Inf, replacing with large number")
        dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10, neginf=1e10)

    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    total_error = dist_matrix[row_ind, col_ind].sum()

    # Handle unmatched modes if penalty is given
    unmatched = abs(true_modes.shape[0] - est_modes.shape[0])
    if penalty is not None:
        total_error += penalty * unmatched

    # Normalize by max number of modes
    mse = total_error / max(true_modes.shape[0], est_modes.shape[0])
    return mse
