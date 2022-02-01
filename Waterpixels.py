# importing PIL
import cv2
import matplotlib.pyplot as plt
from Gradient import SobelOperator
from skimage import io
import numpy as np
from Hexa_grid import drawHexaGrid
from Voronoi_tesselation import voronoiTesselation

# img = io.imread('image.jpg')


# # conversion to gray scale image
# gray_img = color.rgb2gray(img)

# # computing a Sobel operator gradient
# grad_img = SobelOperator(gray_img, 0.3)


# plt.imshow(grad_img, cmap='gray')

# plt.show()


img = cv2.imread("image.jpg")


# conversion to grxay scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 40
centers = drawHexaGrid(gray_img, sigma)


voronoiTesselation(gray_img, centers, sigma)
