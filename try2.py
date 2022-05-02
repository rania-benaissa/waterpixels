import os
import cv2
from scipy import io
from scipy.spatial.distance import cdist
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt


def load_BSDS(images_folder, ground_folder):
    images = []

    gt_contours = []

    for filename in os.listdir(images_folder):

        # correspondant ground truth filename
        gFilename = filename[:-3]+"mat"

        img = cv2.imread(os.path.join(images_folder, filename))

        data = io.loadmat(os.path.join(ground_folder, gFilename))
        # loading contours
        edge_data = data['groundTruth'][0][0][0][0][1]

        # So need to restore back to 0<x<255
        ground_truth = edge_data * 255

        # je reecupere les contours
        gt_contours.append(np.where(ground_truth == 0))
        images.append(img)

    return images, gt_contours


def plot(x, y, title=None):

    plt.plot(x, y)

    plt.xlabel("Nb. superpixels")
    plt.ylabel("Boundary recall")

    if(title is not None):
        plt.title(title)

    plt.savefig('br.jpg', bbox_inches='tight')
    plt.show()


def boundaryRecall(contours, gt_contours, min_dist=3):

    contours = np.array(contours).T

    gt_contours = np.array(gt_contours).T

    cpt = 0

    for i in range(len(gt_contours)):

        point = [gt_contours[i]]

        dist = cdist(point, contours, "cityblock").min()

        if(dist < min_dist):
            cpt += 1

    return cpt / len(gt_contours)


def waterPixels(img, g_sigma=0, sigma=40, rho=2/3, k=8):

    if(img is not None):

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        # # # computing a Sobel operator gradient
        gradient = SobelOperator(gray_img, g_sigma)

        # cv2.imwrite("image_gradient"+str(count)+".jpg", gradient)

        image_grid = hexaGrid.drawHexaGrid(img)

        # cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers

        markers = selectMarkers(gradient, hexaGrid)

        distImage, _ = voronoiTesselation(
            img.shape, markers, sigma, 'euclidean')

        g_reg = gradient + k * (distImage)

        # cv2.imwrite("regularized_gradient"+str(count)+".jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i+1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        contours = np.where(labels == 0)

        # for i in range(len(contours[0])):

        #     cv2.circle(img, np.int32([contours[1][i], contours[0][i]]),
        #                1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        #cv2.imwrite("waterpixels.jpg", img)

    return img, contours, len(hexaGrid.centers)

    # parameters
sigmas = np.arange(10, 50, 5)

# rho ne doit pas etre egale a 0 control that !
rho = 2/3

k = 8

images, gt_contours = load_BSDS(
    "../BSD500/images/test/", "../BSD500/ground_truth/test/")


images = images[:2]
# print(len(images))
x = np.zeros(len(sigmas))
y = np.zeros(len(sigmas))

for k in range(len(sigmas)):

    for i, image in enumerate(images):

        waterpixels, contours, nbCenters = waterPixels(
            image, 0.5, sigmas[k], rho, k)

        img_ctr = np.zeros(image.shape)

        img_gtCtr = np.zeros(image.shape)

        img_ctr[contours] = 255
        img_gtCtr[gt_contours[i]] = 255

        br = boundaryRecall(contours, gt_contours[i])

        y[k] += br

        # fig, axs = plt.subplots(1, 2)

        # axs[0].imshow(img_ctr)
        # axs[0].set_title("Waterpixels contours")
        # axs[0].axis('off')

        # axs[1].imshow(img_gtCtr)
        # axs[1].set_title("Ground truth contours")
        # axs[1].axis('off')

        # fig.tight_layout()
        # plt.show()
    x[k] = nbCenters
    y[k] /= len(images)


plot(x, y)
