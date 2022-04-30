from fastdist import fastdist
import numpy as np
from pytictoc import TicToc
a = np.random.rand(154401, 2)
b = np.random.rand(6317, 2)


t = TicToc()
t.tic()
fastdist.matrix_to_matrix_distance(b, a, fastdist.jaccard, "jaccard")

t.toc()
# returns an array of shape (25, 50)
