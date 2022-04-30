import os
import cv2
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def waterPixels(path, g_sigma=0, sigma=40, rho=2/3, k=8):

    global count

    count += 1

    # keep in mind that u get a bgr image
    img = cv2.imread(path)

    if(img is not None):

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        # # # computing a Sobel operator gradient
        gradient = SobelOperator(gray_img, g_sigma)

        #cv2.imwrite("image_gradient"+str(count)+".jpg", gradient)

        image_grid = hexaGrid.drawHexaGrid(img)

        #cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers

        markers = selectMarkers(gradient, hexaGrid)

        distImage = voronoiTesselation(
            img.shape, markers, sigma, 'euclidean')

        g_reg = gradient + k * (distImage)

        #cv2.imwrite("regularized_gradient"+str(count)+".jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i+1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        contours = np.where(labels == 0)

        for i in range(len(contours[0])):

            cv2.circle(img, np.int32([contours[1][i], contours[0][i]]),
                       1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        #cv2.imwrite("waterpixels"+str(count)+".jpg", img)

    return img

    # parameters
sigma = 35
# rho ne doit pas etre egale a 0 control that !
rho = 2/3

k = 8

path = "images/image8.jpg"

waterPixels(path, 0.5, sigma, rho, k)


images = load_images_from_folder("../BSD500/images/test/")


print(images.shape)
