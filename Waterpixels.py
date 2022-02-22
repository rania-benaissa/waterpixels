# importing PIL
import time
import cv2
from cv2 import RHO
from matplotlib import markers
import matplotlib.pyplot as plt
from Gradient import SobelOperator, computeMinima

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
gradient = SobelOperator(gray_img)
cv2.imwrite("gradient.jpg", gradient)

# compute minima
markers_img = computeMinima(gradient)


plt.imshow(cv2.cvtColor(markers_img.astype(np.float32), cv2.COLOR_BGR2RGB)/255)
plt.show()


markers_img = hexaGrid.drawHexaGrid(markers_img)


plt.imshow(cv2.cvtColor(markers_img.astype(np.float32), cv2.COLOR_BGR2RGB)/255)
plt.show()

#imageCells = voronoiTesselation(gray_img.shape, centers, sigma)
