import numpy as np
from scipy.spatial.distance import cdist

x = np.array([[2, 1], [2, 1]])
y = np.array([[1, 0], [2, 3], [4, 3]])


print(np.array(x).shape)
print(np.array(y).shape)
d = cdist(x, y, 'euclidean')
print(d)
