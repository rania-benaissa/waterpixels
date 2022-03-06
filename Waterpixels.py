import cv2
import matplotlib.pyplot as plt
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
# parameters
sigma = 20

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
cv2.imwrite("image_gradient.jpg", gradient)


# compute minimas and select markers
markers = selectMarkers(gradient, hexaGrid)

distImage = voronoiTesselation(
    img.shape, markers, sigma)


k = 10

g_reg = gradient + k * distImage

cv2.imwrite("regularized_gradient.jpg", cv2.normalize(
    g_reg,  None, 0, 255, cv2.NORM_MINMAX))


markers_map = np.zeros_like(g_reg)

for i, marker in enumerate(markers):
    for point in marker:
        markers_map[point[0], point[1]] = i+1


labels = watershed(g_reg, markers_map, watershed_line=True)

indices = np.where(labels == 0)

img[indices[0], indices[1]] = 255
cv2.imwrite("waterpixels.jpg", img)
