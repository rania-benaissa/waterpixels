import numpy as np
import cv2


def SobelOperator(img, sigma=None):

    image = img.copy()

    if(sigma != None):

        image = cv2.GaussianBlur(image, (3, 3), sigma, 0)

    g_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    g_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # magnitude
    norm = np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))

    normalized_norm = cv2.normalize(
        norm,  None, 0, 255, cv2.NORM_MINMAX)

    return normalized_norm

# only gray lvl

# a morphological gradient is the difference between the dilation and the erosion of a given image.


def morphologicalGradient(image):

    img = image.copy()

    struct_elt = np.ones((3, 3), np.uint8)

    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, struct_elt)


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

        # selected_center = []

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
                    # selected_center = centers[i]
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
