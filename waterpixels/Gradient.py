from cProfile import label
import numpy as np
import cv2
from pytictoc import TicToc
from skimage.draw import polygon2mask
from matplotlib import pyplot as plt
from numba import jit


def sobelOperator(img, sigma=0, size=3):

    image = img.copy()

    if(sigma != 0):

        image = cv2.GaussianBlur(image, (size, size), sigma, 0)

    g_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=size)
    g_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=size)

    # magnitude
    norm = np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))

    normalized_norm = cv2.normalize(
        norm, None, 0, 255, cv2.NORM_MINMAX)

    return normalized_norm

# only gray lvl

# a morphological gradient is the difference between the dilation and the erosion of a given image.


def morphologicalGradient(image, value=-1, size=3):

    img = image.copy()

    if(value == -1):

        # print("Rect")
        struct_elt = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    if(value == -2):
        # print("Cross")
        struct_elt = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, struct_elt)


def selectMarkers(img, hexaGrid, color=[249, 217, 38][::-1]):
    # t_poly = 0
    # t_minimas = 0

    # t_components = 0

    # t_markers = 0

    # markers of all cells
    markers = []

    #markers_centers = []

    minimas_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for center in hexaGrid.centers:

        # binary image that will contain minimas of a cell
        cell_minimas_image = np.zeros((img.shape), dtype=np.uint8)

        vertices = np.int32(hexaGrid.getHomoHexaVertices(
            center))

        # create mask for my polygon
        mask = np.zeros_like(img)

        cv2.fillPoly(mask, [vertices], (255))

        # get the indices inside the poly
        hex_indices = np.where(mask == 255)

        selected_marker = []

        # selected_center = []

        # this condition treats borders
        if(hex_indices[0] != [] and hex_indices[1] != []):

            # get the local minimum
            poly_idx = img[hex_indices]
            mini = np.amin(poly_idx)

            indices = np.where(poly_idx == mini)

            cell_minimas_image[hex_indices[0][indices], hex_indices[1]
                               [indices]] = 255

            # i'll save the image containing all the minimas
            # minimas_image[hex_indices[0][indices], hex_indices[1]
            #               [indices]] = color

            _, labels = cv2.connectedComponents(
                cell_minimas_image, connectivity=8)

            # i select as a marker the biggest connected component

            counts = np.bincount(labels.ravel())

            # removing background
            counts[0] = -1

            marker_index = np.argmax(counts)

            selected_marker = np.array(np.unravel_index(np.flatnonzero(
                labels == marker_index), labels.shape)).T

        markers.append(selected_marker)

    # for marker in markers:

    #     for point in marker:
    #         # image containing only the selected markers
    #         markers_image[point[0], point[1]] = color

    # cv2.imwrite("image_minimas.jpg", minimas_image)

    # cv2.imwrite("image_markers.jpg", markers_image)

    # cv2.imwrite("image_minimas_with_grid.jpg",
    #             hexaGrid.drawHexaGrid(minimas_image))

    # cv2.imwrite("image_markers_with_grid.jpg",
    #             hexaGrid.drawHexaGrid(markers_image))

    return markers  # , markers_centers
