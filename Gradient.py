from os import stat
from matplotlib import patches
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


def computeLocalMinimas(img, hexaGrid, color=[86, 76, 115]):
    #image = np.full((img.shape[0], img.shape[1], 3), 0)

    # connected component of all cells
    cellsComps = []

    for center in hexaGrid.centers:

        binary_image = np.full((img.shape), 0, dtype=np.uint8)

        vertices = np.int32(hexaGrid.getHomoHexaVertices(
            center))

        # connected component of a cell
        cellComponents = []

        # create mask for my polygon
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], (255))

        # get the indices inside the poly
        hex_indices = np.where(mask == 255)

        # this condition treats borders
        if(hex_indices[0] != [] and hex_indices[1] != []):
            # get the local minimum
            mini = np.amin(img[hex_indices])

            indices = np.where(img[hex_indices] == mini)

            # image[hex_indices[0][indices], hex_indices[1]
            #       [indices]] = color[::-1]

            binary_image[hex_indices[0][indices], hex_indices[1]
                         [indices]] = 255

            numLabels, labels, _, _ = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8)

            for i in range(1, numLabels):

                cellComponents.append(np.argwhere(labels == i))

        cellsComps.append(np.array(cellComponents))

    return cellsComps


def selectMarkers(img, hexaGrid, color=[86, 76, 115]):
    # calcule tous les minimas locaux
    localMinimas = computeLocalMinimas(img, hexaGrid)

    # prune local minimas
    # for each cell get the list of components
    for components in localMinimas:

        print("new cell")
        for component in components:

            print(np.array(component).shape)
