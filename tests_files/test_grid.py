import sys
sys.path.append('../waterpixels/')
from matplotlib import pyplot as plt
from pytictoc import TicToc
import cv2
from waterpixels.Gradient import *
from waterpixels.HexaGrid import HexaGrid
from waterpixels.Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np


def waterPixels(path, g_sigma=-1, sigma=40, rho=2 / 3, k=8, distType='euclidean'):

    # keep in mind that u get a bgr image
    img = cv2.imread(path)

    if(img is not None):

        # t.tic()

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        #t.toc("Hexa grid init")

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        #gradient = morphologicalGradient(gray_img, -1)

        #cv2.imwrite("image_morpho_gradient"+str(count)+".jpg", gradient)

        # t.tic()
        # # # computing a Sobel operator gradient
        gradient = SobelOperator(gray_img, g_sigma)

        #t.toc("gradient computation")

        cv2.imwrite("image_gradient.jpg", gradient)
        # t.tic()
        image_grid = hexaGrid.drawHexaGrid(img)

        #t.toc("Hexa grid drawing")

        #cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers
        # t.tic()
        markers = selectMarkers(gradient, hexaGrid)
        #t.toc("Compute markers")

        # t.tic()

        distImage, visu = voronoiTesselation(
            img.shape, markers, sigma, distType)

        #t.toc("Compute Veronoi tesselations")

        g_reg = gradient + k * (distImage)

        cv2.imwrite("regularized_gradient.jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i + 1

        # t.toc("Regu gradient")
        # t.tic()
        labels = watershed(g_reg, markers_map, watershed_line=True)

        # t.toc("watershed transform")

        # t.tic()

        indices = np.where(labels == 0)

        for i in range(len(indices[0])):

            cv2.circle(img, np.int32([indices[1][i], indices[0][i]]),
                       1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        #t.toc("drawing waterpixels")

        cv2.imwrite("waterpixels.jpg", img)

    return img, visu


    # parameters
sigma = 30
# rho ne doit pas etre egale a 0 control that !
rho = 2 / 3

k = 10
path = "images/image30.jpg"


fig, axs = plt.subplots(2, 3)


im = cv2.imread(path)

steps = [10, 20]

axs[0, 0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original image")
axs[0, 0].axis('off')


for i in range(len(steps)):
    t = TicToc()
    t.tic()
    im, vis = waterPixels(path, 0.7, steps[i], rho, k, "euclidean")
    t.toc()
    # showing image
    axs[0, i + 1].imshow(cv2.cvtColor(
        im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[0, i + 1].set_title(
        "Grid step = " + str(steps[i]))
    axs[0, i + 1].axis('off')

steps = [30, 40, 50]
for i in range(len(steps)):
    t = TicToc()
    t.tic()
    im, vis = waterPixels(path, 0.7, steps[i], rho, k, "euclidean")
    t.toc()
    # showing image
    axs[1, i].imshow(cv2.cvtColor(
        im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[1, i].set_title(
        "Grid step = " + str(steps[i]))
    axs[1, i].axis('off')


fig.tight_layout(pad=3)
plt.show()
fig.savefig('dist_compare.jpg', bbox_inches='tight')
