from math import dist
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.metrics.pairwise import euclidean_distances


def euclidianDist(p1, p2):

    dists = euclidean_distances([p1], p2)

    return dists[0].min()


# la je dois retourner une image avec le min d * 2/ sigma
def voronoiTesselation(shape, centers, sigma):

    print(len(centers))

    # voronoi diagram
    diagram = np.zeros(shape)
    # height
    for i in range(diagram.shape[0]):
        # width
        for j in range(diagram.shape[1]):

            diagram[i, j] = euclidianDist(
                np.array([i, j]), np.array(centers))

        # diagram = (2/sigma) * diagram

        # plt.imshow(diagram, cmap="gray")
        # plt.show()

        cv2.imwrite("voronoi.jpg", diagram)

        # print(euclidean_distances(
        #     np.array([[243, 413]]), np.array(centers))[0])

        # print(diagram[243, 413])

        # return (2/sigma) * diagram
