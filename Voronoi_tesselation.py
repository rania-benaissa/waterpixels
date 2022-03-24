from matplotlib.pyplot import gray
import numpy as np
import cv2
from itertools import product
from sklearn.metrics.pairwise import euclidean_distances


# it computes that for a row


def euclidianDist(row, markers_centers):

    # je reecup les indices d'une ligne i

    # je get les pixels sur les lignes
    row_indices = list(product([euclidianDist.i], np.arange(len(row))))

    # print(row_indices)

    euclidianDist.i += 1

    return euclidean_distances(row_indices, markers_centers).min(axis=1)


# la je dois retourner une image avec le min d * 2/ sigma
def voronoiTesselation(shape, markers, sigma, color=[249, 217, 38][::-1]):

    # voronoi diagram
    diagram = np.zeros(shape)

    gray_diagram = np.zeros(shape[:2])

    euclidianDist.i = 0
    markers_points = []

    for marker in markers:

        for point in marker:
            markers_points.append(point)
    # j'applique ma fonction sur chaque ligne

    # print(np.array(markers_points).shape)

    gray_diagram = np.apply_along_axis(
        euclidianDist, 1, gray_diagram, markers_points)

    gray_diagram = (2/sigma)*gray_diagram

    diagram[:, :, 0] = gray_diagram

    diagram[:, :, 1] = gray_diagram

    diagram[:, :, 2] = gray_diagram

    for point in markers_points:
        # image containing only the selected markers
        diagram[point[0], point[1]] = color

    cv2.imwrite("voronoi.jpg",
                gray_diagram)

    cv2.imwrite("voronoi_with_markers.jpg",
                diagram)

    return gray_diagram
