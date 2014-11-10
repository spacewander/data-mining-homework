import numpy as np
import scipy.spatial.distance as ssd

def pearson_correlation(X, Y):
    """
    X: array of shape
    Y: array of shape
    """
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'correlation', 2)

    return 1 - XY
