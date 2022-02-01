import matplotlib.pyplot as plt
import numpy as np
import cv2


def euclidianDist(p1, p2):

    dists = np.zeros((len(p2)))

    for i in range(len(dists)):

        dists[i] = np.linalg.norm(p1-p2[i])

    return dists


# la je dois retourner une image avec le min d * 2/ sigma


def voronoiTesselation(image, centers, sigma):
    # voronoi diagram
    diagram = np.zeros((image.shape))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            diagram[i, j] = np.min(euclidianDist(
                np.array([i, j]), np.array(centers)))

    diagram = (2/sigma) * diagram

    plt.imshow(diagram, cmap="gray")
    plt.show()

    #cv2.imwrite("voronoi.jpg", diagram)

    # return (2/sigma) * diagram
