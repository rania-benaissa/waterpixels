# importing PIL
import time
import cv2

import matplotlib.pyplot as plt
from Gradient import *

import numpy as np
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation

# parameters
sigma = 40
rho = 2/3

# keep in mind that u get a bgr image
img = cv2.imread("image.jpg")

hexaGrid = HexaGrid(sigma, rho)

hexaGrid.computeCenters(img.shape)

# conversion to gray scale images
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # computing a Sobel operator gradient
#gradient = SobelOperator(gray_img)

gradient = morphologicalGradient(gray_img)
cv2.imwrite("gradient.jpg", gradient)


plt.imshow(np.array(cv2.cvtColor(gradient.astype(
    np.float32), cv2.COLOR_BGR2RGB), np.uint8))
plt.show()

# compute minima


markers_img = computeMinima(gradient, hexaGrid)
#markers_img = hexaGrid.drawHexaGrid(markers_img)

plt.imshow(np.array(cv2.cvtColor(markers_img.astype(
    np.float32), cv2.COLOR_BGR2RGB), np.uint8))
plt.show()


# imageCells = voronoiTesselation(gray_img.shape, centers, sigma)
