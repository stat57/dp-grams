import numpy as np
from scipy.optimize import linear_sum_assignment

def mode_matching_mse(true_modes, est_modes, penalty=None):
    true_modes = np.atleast_2d(true_modes)
    est_modes = np.atleast_2d(est_modes)

    if true_modes.size == 0 or est_modes.size == 0:
        return np.nan

    dist_matrix = np.sum((true_modes[:, None, :] - est_modes[None, :, :])**2, axis=2)
    dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10, neginf=1e10)

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_error = dist_matrix[row_ind, col_ind].sum()

    if penalty is not None:
        unmatched = abs(true_modes.shape[0] - est_modes.shape[0])
        total_error += penalty * unmatched

    mse = total_error / max(true_modes.shape[0], est_modes.shape[0])
    return mse
