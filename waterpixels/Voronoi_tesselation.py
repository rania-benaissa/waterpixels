import numpy as np
import cv2
from itertools import product
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt


def euclidianDist(row, markers_centers):

    # je reecup les indices d'une ligne i

    # je get les pixels sur les lignes
    row_indices = list(product([euclidianDist.i], np.arange(len(row))))

    euclidianDist.i += 1

    return euclidean_distances(row_indices, markers_centers).min(axis=1)

# la je dois retourner une image avec le min d * 2/ sigma


def voronoiTesselation(shape, markers, sigma, distType='euclidean', color=[249, 217, 38][::-1], visu=False):

    binary_img = np.full(shape[:2], 1, np.uint8)

    # voronoi diagram
    diagram = np.zeros(shape, np.uint8)

    markers_points = []

    for marker in markers:

        for point in marker:
            markers_points.append(point)
            binary_img[point[0], point[1]] = 0

    # calcul pour  chaque marqueur la dist avec tous les elts de l'image
    gray_diagram = cv2.distanceTransform(binary_img,
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    gray_diagram = cv2.normalize(
        gray_diagram, None, 0, 255, cv2.NORM_MINMAX)

    gray_diagram = (2 / sigma) * (gray_diagram)

    if(visu):
        # # juste pour visualiser
        vis_gray_diagram = cv2.normalize(
            gray_diagram, None, 0, 255, cv2.NORM_MINMAX)

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
