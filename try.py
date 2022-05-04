from scipy.spatial.distance import cdist
import numpy as np
from pytictoc import TicToc
a = np.random.rand(1, 2)
print(a)
b = np.random.rand(6317, 2)


t = TicToc()
t.tic()
dist = cdist(a, b, "cityblock")
print(dist.shape)
t.toc()
# returns an array of shape (25, 50)
x = np.array([[1, 2], [45, 2], [5, 2], [1, 2], [5, 2]])
y = np.flip(x, 1)

print(x.shape)
print(y)
