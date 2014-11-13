# coding: utf-8

import numpy as np
import scipy.spatial.distance as ssd

def pearson_correlation(X, Y):
    """
    皮尔逊相关系数，大小位于-1到1之间，数值越大相关性越高 链接：
    http://segmentfault.com/q/1010000000094674
    http://zh.wikipedia.org/wiki/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0
    X: array of shape
    Y: array of shape
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'correlation', 2)

    return 1 - XY

def jaccard_coefficient(X, Y):
    """
    The MAE is : 0.925
    The RMSE is : 1.21449578015
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    return np.intersect1d(X, Y).size

def manhattan_distances(X, Y):
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'cityblock')

    return 1.0 - (XY / float(X.shape[1]))
