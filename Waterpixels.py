from pytictoc import TicToc
import cv2
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np


count = 0


def waterPixels(path, g_sigma=-1, sigma=40, rho=2/3, k=8):

    global count

    count += 1

    # keep in mind that u get a bgr image
    img = cv2.imread(path)

    t = TicToc()

    if(img is not None):

        t.tic()

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        t.toc("Hexa grid init")

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        if(g_sigma < 0):

            gradient = morphologicalGradient(gray_img, g_sigma)

            #cv2.imwrite("image_morpho_gradient"+str(count)+".jpg", gradient)
        else:
            t.tic()
            # # # computing a Sobel operator gradient
            gradient = SobelOperator(gray_img, g_sigma)

            t.toc("gradient computation")

            #cv2.imwrite("image_gradient"+str(count)+".jpg", gradient)
        t.tic()
        image_grid = hexaGrid.drawHexaGrid(img)

        t.toc("Hexa grid drawing")

        #cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers
        t.tic()
        markers = selectMarkers(gradient, hexaGrid)
        t.toc("Compute markers")

        t.tic()

        distImage = voronoiTesselation(
            img.shape, markers, sigma)

        t.toc("Compute Veronoi tesselations")

        t.tic()
        g_reg = gradient + k * (distImage)

        cv2.imwrite("regularized_gradient"+str(count)+".jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i+1

        t.toc("Regu gradient")
        t.tic()
        labels = watershed(g_reg, markers_map, watershed_line=True)

        t.toc("watershed transform")

        t.tic()

        indices = np.where(labels == 0)

        for i in range(len(indices[0])):

            cv2.circle(img, np.int32([indices[1][i], indices[0][i]]),
                       1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        t.toc("drawing waterpixels")

        #cv2.imwrite("waterpixels"+str(count)+".jpg", img)

    return img

    # parameters
sigma = 35
# rho ne doit pas etre egale a 0 control that !
rho = 2/3

k = 8

path = "images/image8.jpg"

waterPixels(path, 0.7, sigma, rho, k)
