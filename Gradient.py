from msilib.schema import Component
from os import stat
from matplotlib import markers, patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cv2
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb


def gaussianKernel(sigma):
    """ double -> Array
        return a gaussian kernel of standard deviation sigma
    """
    n2 = np.int64(np.ceil(3*sigma))
    x, y = np.meshgrid(np.arange(-n2, n2+1), np.arange(-n2, n2+1))
    kern = np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()


def SobelOperator(img, sigma=None):

    image = img.copy()

    s_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    s_y = s_x.T

    if(sigma != None):

        image = signal.convolve2d(image, gaussianKernel(sigma), mode='same')

    g_x = signal.convolve2d(image, s_x, mode='same')

    g_y = signal.convolve2d(image, s_y, mode='same')
    # magnitude
    return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))


# only gray lvl

# a morphological gradient is the difference between the dilation and the erosion of a given image.


def morphologicalGradient(image):

    img = image.copy()

    struct_elt = np.ones((4, 4), np.uint8)

    return cv2.dilate(img, struct_elt) - cv2.erode(img, struct_elt)


def selectMarkers(img, hexaGrid, color=[249, 217, 38][::-1]):

    # markers of all cells
    markers = []

    markers_centers = []

    minimas_image = np.full((img.shape[0], img.shape[1], 3), 0, dtype=np.uint8)
    markers_image = np.full((img.shape[0], img.shape[1], 3), 0, dtype=np.uint8)

    for center in hexaGrid.centers:
        # binary image that will contain minimas of a cell
        cell_minimas_image = np.full((img.shape), 0, dtype=np.uint8)

        vertices = np.int32(hexaGrid.getHomoHexaVertices(
            center))

        # create mask for my polygon
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], (255))
        # plt.imshow(mask)
        # plt.show()

        # get the indices inside the poly
        hex_indices = np.where(mask == 255)

        selected_marker = []

        #selected_center = []

        # this condition treats borders
        if(hex_indices[0] != [] and hex_indices[1] != []):
            # get the local minimum
            mini = np.amin(img[hex_indices])

            indices = np.where(img[hex_indices] == mini)

            # image[hex_indices[0][indices], hex_indices[1]
            #       [indices]] = color[::-1]

            cell_minimas_image[hex_indices[0][indices], hex_indices[1]
                               [indices]] = 255

            # i'll save the image containing all the minimas

            minimas_image[hex_indices[0][indices], hex_indices[1]
                          [indices]] = color

            numLabels, labels, _, centers = cv2.connectedComponentsWithStats(
                cell_minimas_image, connectivity=8)

            # plt.imshow(labels)
            # plt.show()

            nb_pixels = 0

            # i select as a marker the biggest connected component

            for i in range(1, numLabels):

                new_marker = np.argwhere(labels == i)

                if(nb_pixels < new_marker.shape[0]):
                    #selected_center = centers[i]
                    nb_pixels = new_marker.shape[0]
                    selected_marker = new_marker

        markers.append(selected_marker)
        # markers_centers.append(selected_center)

    for marker in markers:

        for point in marker:
            # image containing only the selected markers
            markers_image[point[0], point[1]] = color

    cv2.imwrite("image_minimas.jpg", minimas_image)

    cv2.imwrite("image_markers.jpg", markers_image)

    cv2.imwrite("image_minimas_with_grid.jpg",
                hexaGrid.drawHexaGrid(minimas_image))

    cv2.imwrite("image_markers_with_grid.jpg",
                hexaGrid.drawHexaGrid(markers_image))

    return markers  # , markers_centers
