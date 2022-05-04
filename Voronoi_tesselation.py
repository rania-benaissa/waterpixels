import numpy as np
import cv2
from itertools import product
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances


def euclidianDist(row, markers_centers):

    # je reecup les indices d'une ligne i

    # je get les pixels sur les lignes
    row_indices = list(product([euclidianDist.i], np.arange(len(row))))

    euclidianDist.i += 1

    return euclidean_distances(row_indices, markers_centers).min(axis=1)

# la je dois retourner une image avec le min d * 2/ sigma


def voronoiTesselation(shape, markers, sigma, distType='euclidean', color=[249, 217, 38][::-1]):

    max_dims = 500

    # voronoi diagram
    diagram = np.zeros(shape)

    img = list(product(np.arange(shape[0]), np.arange(shape[1])))

    markers_points = []

    for marker in markers:

        for point in marker:
            markers_points.append(point)

    # calcul pour  chaque marqueur la dist avec tous les elts de l'image

    gray_diagram = []

    if(shape[0] <= max_dims and shape[1] <= max_dims and len(markers_points) < max_dims):

        dist = cdist(np.array(markers_points),
                     np.array(img), distType)

        # on reecup le min sur chaque colonne
        gray_diagram = np.min(dist, axis=0)

        gray_diagram = gray_diagram.reshape(shape[:2])

    else:
        euclidianDist.i = 0
        gray_diagram = np.zeros(shape[:2])
        gray_diagram = np.apply_along_axis(
            euclidianDist, 1, gray_diagram, markers_points)

    gray_diagram = cv2.normalize(
        gray_diagram,  None, 0, 255, cv2.NORM_MINMAX)

    gray_diagram = (2/sigma)*(gray_diagram)

    # juste pour visualiser
    vis_gray_diagram = cv2.normalize(
        gray_diagram,  None, 0, 255, cv2.NORM_MINMAX)

    diagram[:, :, 0] = vis_gray_diagram

    diagram[:, :, 1] = vis_gray_diagram

    diagram[:, :, 2] = vis_gray_diagram

    for point in markers_points:
        # image containing only the selected markers
        diagram[point[0], point[1]] = color

    # cv2.imwrite("voronoi.jpg",
    #             vis_gray_diagram)

    # cv2.imwrite("voronoi_with_markers.jpg",
    #             diagram)

    return gray_diagram, diagram
