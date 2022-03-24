from dis import dis
from optparse import Values
import cv2
import matplotlib.pyplot as plt
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
# parameters
sigma = 40

# rho ne doit pas etre egale a 0 control that !
rho = 2/3
# keep in mind that u get a bgr image
img = cv2.imread("image1.jpg")

hexaGrid = HexaGrid(sigma, rho)

hexaGrid.computeCenters(img.shape)

# conversion to gray scale images
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# gradient = morphologicalGradient(gray_img)
# cv2.imwrite("image_morpho_gradient.jpg", gradient)
# print(gradient)
# # # computing a Sobel operator gradient
gradient = SobelOperator(gray_img, 1)


cv2.imwrite("image_gray.jpg", gray_img)
cv2.imwrite("image_gradient.jpg", gradient)

image_grid = hexaGrid.drawHexaGrid(img)

cv2.imwrite("image_grid.jpg", image_grid)

# compute minimas and select markers
markers = selectMarkers(gradient, hexaGrid)

distImage = voronoiTesselation(
    img.shape, markers, sigma)


k = 30

g_reg = gradient + k * distImage


# g_reg = cv2.normalize(
#     g_reg,  None, 0, 255, cv2.NORM_MINMAX)


cv2.imwrite("regularized_gradient.jpg", g_reg)


markers_map = np.zeros_like(g_reg)

for i, marker in enumerate(markers):
    for point in marker:
        markers_map[point[0], point[1]] = i+1


labels = watershed(g_reg, markers_map, watershed_line=True)

indices = np.where(labels == 0)


for i in range(len(indices[0])):

    cv2.circle(img, np.int32([indices[1][i], indices[0][i]]),
               1, [249, 217, 38][::-1], -1, cv2.LINE_AA)


#img[indices[0], indices[1]] = [249, 217, 38][::-1]


cv2.imwrite("waterpixels.jpg", img)
