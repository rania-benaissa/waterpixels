from math import dist
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances

# it computes that for a row


def euclidianDist(a, row, centers):

    return a + euclidean_distances(row, centers).min(axis=1)


# la je dois retourner une image avec le min d * 2/ sigma
def voronoiTesselation(shape, centers, sigma):

    print(len(centers))

    # voronoi diagram
    diagram = np.zeros(shape)

    for i in range(diagram.shape[0]):
        # je reecup les indices d'une ligne i
        row = list(product([i], np.arange(diagram.shape[1])))
        # j'applique ma fonction sur chaque ligne
        diagram[i, :] = np.apply_along_axis(
            euclidianDist, 0, diagram[i, :], row, centers)

    print(diagram)

    cv2.imwrite("voronoi.jpg", (2/sigma) * diagram)

    return (2/sigma) * diagram


# pixels = [[1, 2], [3, 4]]
# centers = [[0, 2], [1, 1], [0, 0]]
# print(euclidean_distances(pixels, centers))

# print(euclidean_distances(pixels, centers).min(axis=1))
comb = list(product([1], [1, 2, 3,
                          10]))
print(comb)
