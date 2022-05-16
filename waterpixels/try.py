from dis import dis
from scipy.spatial.distance import cdist
import numpy as np
from pytictoc import TicToc
from sklearn.metrics import euclidean_distances
import math
a = np.random.rand(1000, 2)
b = np.random.rand(100000, 2)


t = TicToc()
t.tic()
dist = cdist(a, b, "euclidean")
print(dist)
print(dist.shape)
t.toc()

from numba import jit


@jit(nopython=True)
def eudis(a, b):
    x = np.sum(a**2, axis=1)[:, np.newaxis]
    y = np.sum(b**2, axis=1)
    xy = np.dot(a, b.T)
    return np.sqrt(x + y - 2 * xy)


t = TicToc()
t.tic()

dist = eudis(a, b)

print(dist)
print(dist.shape)
t.toc()
