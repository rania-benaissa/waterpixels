# importing PIL
import time
import cv2
from cv2 import RHO
from matplotlib import markers
import matplotlib.pyplot as plt
from Gradient import SobelOperator, computeMinimas
from skimage import io
import numpy as np
from Hexa_grid import drawHexaGrid
from Voronoi_tesselation import voronoiTesselation

# parameters
sigma = 40
rho = 2/3

# keep in mind that u get a bgr image
img = cv2.imread("image.jpg")

# conversion to gray scale images


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # computing a Sobel operator gradient
gradient = SobelOperator(gray_img)


cv2.imwrite("gradient.jpg", gradient)


markers_img = computeMinimas(gradient)

plt.imshow(cv2.cvtColor(markers_img.astype(np.float32), cv2.COLOR_BGR2RGB)/255)
plt.show()


markers_img, centers = drawHexaGrid(markers_img, sigma, rho)

plt.imshow(cv2.cvtColor(markers_img.astype(np.float32), cv2.COLOR_BGR2RGB)/255)
plt.show()

#imageCells = voronoiTesselation(gray_img.shape, centers, sigma)
