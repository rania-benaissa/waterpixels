import sys
sys.path.append('../waterpixels/')
from pytictoc import TicToc
import cv2
from waterpixels.Gradient import *
from waterpixels.HexaGrid import HexaGrid
from waterpixels.Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt


count = 0


def waterPixels(path, g_sigma=-1, sigma=40, rho=2 / 3, k=8):

    global count

    count += 1

    # keep in mind that u get a bgr image
    img = cv2.imread(path)

    t = TicToc()
    t.tic()

    if(img is not None):

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        if(g_sigma < 0):

            gradient = morphologicalGradient(gray_img, g_sigma)

            #cv2.imwrite("image_morpho_gradient"+str(count)+".jpg", gradient)
        else:

            # # # computing a Sobel operator gradient
            gradient = sobelOperator(gray_img, g_sigma)

            #cv2.imwrite("image_gradient"+str(count)+".jpg", gradient)

        image_grid = hexaGrid.drawHexaGrid(img)

        #cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers
        markers = selectMarkers(gradient, hexaGrid)

        distImage, _ = voronoiTesselation(
            img.shape, markers, sigma)

        g_reg = gradient + k * (distImage)

        #cv2.imwrite("regularized_gradient" + str(count) + ".jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i + 1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        indices = np.where(labels == 0)

        for i in range(len(indices[0])):

            cv2.circle(img, np.int32([indices[1][i], indices[0][i]]),
                       1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        #cv2.imwrite("waterpixels"+str(count)+".jpg", img)

    t.toc()
    return img

    # parameters
sigma = 30
# rho ne doit pas etre egale a 0 control that !
rho = 2 / 3

k = 4

fig = plt.figure()
ax1 = plt.subplot2grid((2, 12), (0, 0), rowspan=1, colspan=3)
ax2 = plt.subplot2grid((2, 12), (1, 0), rowspan=1, colspan=4)


g_sigma = [0.4, 0.7, 1.0]

path = "images/image5.jpg"

im = cv2.imread(path)

ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
ax1.set_title("Original image")
ax1.axis('off')


ax2.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
ax2.set_title("Original image")
ax2.axis('off')

type = ["Cross", "Rect"]
for i in range(len(g_sigma)):

    im = waterPixels(path, g_sigma[i], sigma, rho, k)
    ax = plt.subplot2grid((2, 12), (0, (i + 1) * 3), rowspan=1, colspan=3)
    # showing image
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(
        "Sobel with sigma = " + str(g_sigma[i]))
    ax.axis('off')

    if(i < 2):
        # partie gradient morpho
        im = waterPixels(path, -1 - i, sigma, rho, k)
        ax = plt.subplot2grid((2, 12), (1, (i + 1) * 4), rowspan=1, colspan=4)
        # showing image
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        print(-1 - i)
        print(type[-1 - i])
        ax.set_title("Morphological gradient with " +
                     type[-1 - i] + " kernel")
        ax.axis('off')
# sobel
# waterPixels("images/image5.jpg", True, sigma, rho, k)

# waterPixels("images/image5.jpg", False, sigma, rho, k)

fig.tight_layout()
plt.show()
fig.savefig('sobel_morph.jpg', bbox_inches='tight')
