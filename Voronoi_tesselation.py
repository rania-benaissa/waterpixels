from math import dist
import matplotlib.pyplot as plt
import numpy as np
import cv2
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances


# it computes that for a row


def euclidianDist(row, centers):

    # je reecup les indices d'une ligne i

    # je get les pixels sur les lignes
    row_indices = list(product([euclidianDist.i], np.arange(len(row))))

    euclidianDist.i += 1

    return euclidean_distances(row_indices, centers).min(axis=1)


# la je dois retourner une image avec le min d * 2/ sigma
def voronoiTesselation(shape, centers, sigma):

    # voronoi diagram
    diagram = np.zeros(shape)

    euclidianDist.i = 0

    # j'applique ma fonction sur chaque ligne
    diagram = np.apply_along_axis(
        euclidianDist, 1, diagram, centers)

    cv2.imwrite("voronoi.jpg", diagram)

    return (2/sigma) * diagram
