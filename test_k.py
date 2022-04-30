from matplotlib import pyplot as plt
from pytictoc import TicToc
import cv2
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np


def waterPixels(path, g_sigma=-1, sigma=40, rho=2/3, k=8, distType='euclidean'):

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
                markers_map[point[0], point[1]] = i+1

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


sigma = 40
# rho ne doit pas etre egale a 0 control that !
rho = 2/3

# k = 10
path = "images/image15.jpg"


fig = plt.figure()
ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=1, colspan=2)
ax2 = plt.subplot2grid((3, 4), (0, 2), rowspan=1, colspan=2)


im = cv2.imread(path)


ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
ax1.set_title("Original image")
ax1.axis('off')

k = [0, 4, 8, 16]
for i in range(len(k)):

    ax = plt.subplot2grid((3, 4), (1, i), rowspan=2)
    t = TicToc()
    t.tic()
    im, vis = waterPixels(path, 0.3, sigma, rho, k[i], "euclidean")
    t.toc()
    # showing image
    ax.imshow(cv2.cvtColor(
        im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title(
        "k = " + str(k[i]))
    ax.axis('off')

 # showing image
ax2.imshow(cv2.cvtColor(
    vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
ax2.set_title(
    "Voronoi Tesselations with markers")
ax2.axis('off')


fig.tight_layout()
plt.show()
fig.savefig('dist_compare.jpg', bbox_inches='tight')
