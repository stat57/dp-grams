# bandwidth.py

## This script contains code for computing silverman bandwidth.

import numpy as np
import math

def silverman_bandwidth(data):
    """
    Standard (non-private) Silverman's rule of thumb bandwidth.
    Works for univariate and multivariate data.
    """
    data = np.atleast_2d(data)
    if data.shape[1] == 1:
        return _silverman_univariate(data[:, 0])
    else:
        return _silverman_multivariate(data)

def _silverman_univariate(x):
    n = len(x)
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34)
    return 0.9 * sigma * n ** (-1 / 5)

def _silverman_multivariate(X):
    n, d = X.shape
    cov = np.cov(X.T)
    tr = np.trace(cov)
    h2 = (2 / d) * tr * (4 / ((2 * d + 1) * n)) ** (2 / (d + 4))
    return math.sqrt(h2)
