import sys
sys.path.append('../waterpixels/')
import os
import cv2
from scipy import io
from waterpixels.Gradient import *
from waterpixels.HexaGrid import HexaGrid
from waterpixels.Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from pytictoc import TicToc


def labelsToContours(labels, image=None):

    nb_labels = np.unique(labels)
    contours = []

    for label in nb_labels:

        y = labels == label

        y = y.astype('uint8')
        y = morphologicalGradient(y, -3, size=2)

        y = np.array(np.where(y == True)).T

        contours += [c for c in y]

    # this is to get unique contours
    contours = list(set(map(tuple, contours)))
    mask = None
    if(image is not None):

        mask = np.zeros(image.shape, np.uint8)
        for point in np.array(contours):
            x, y = point
            mask[x, y] = 255

    # print(np.array(contours).shape)
    return np.array(contours).T, mask


def load_SBD(images_folder, ground_folder, nb_images=100):

    images = []

    gt_contours = []

    gt_masks = []

    files = os.listdir(images_folder)

    for i in range(nb_images):

        filename = files[i]

        # correspondant ground truth filename
        gFilename = filename[:-3] + "layers.txt"

        img = cv2.imread(os.path.join(images_folder, filename))

        label_img = np.loadtxt(os.path.join(
            ground_folder, gFilename), np.uint8)

        contours, mask = labelsToContours(label_img, img)

        gt_contours.append(np.array(contours))

        images.append(img)

        gt_masks.append(mask)

    return images, gt_contours, gt_masks


def load_BSDS(images_folder, ground_folder):
    images = []

    gt_contours = []

    for filename in os.listdir(images_folder):

        # correspondant ground truth filename
        gFilename = filename[:-3] + "mat"

        img = cv2.imread(os.path.join(images_folder, filename))

        data = io.loadmat(os.path.join(ground_folder, gFilename))
        # loading contours
        edge_data = data['groundTruth'][0][0][0][0][1]

        # je reecupere les contours
        gt_contours.append(np.where(edge_data == 1))

        # img[np.where(edge_data == 1)] = [249, 217, 38][::-1]

        images.append(img)
        # cv2.imwrite("images/"+filename, img)

    return images, gt_contours


def plotTime(x_bsds, time_bsds, x_sbd, time_sbd, title=None):

    plt.rcParams["font.family"] = 'Comic Sans MS'
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    colors = ["#3CBCC7", "#1C96BA", "#0C73B2", "#094C93", "#51BC37"]

    plt.plot(x_bsds, time_bsds, label='WP using BSDS', marker='o',
             color=colors[2])

    plt.plot(x_sbd, time_sbd, label='WP using SBD', marker='o',
             color=colors[4])
    plt.xlabel("Nb. Superpixels", fontsize=14)
    plt.ylabel("Execution time (seconds)", fontsize=14)

    if(title is not None):
        plt.title(title)

    plt.legend()
    plt.savefig('time.jpg', bbox_inches='tight')
    # plt.show()


def waterPixels(img, g_sigma=0, sigma=40, rho=2 / 3, k=8):

    t = TicToc()

    t.tic()
    hexaGrid = HexaGrid(sigma, rho)

    hexaGrid.computeCenters(img.shape)
    t.toc("hexa grid")

    hexaGrid.drawHexaGrid(img)
    # conversion to gray scale images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t.tic()
    # # # computing a Sobel operator gradient
    gradient = sobelOperator(gray_img, g_sigma)
    t.toc("Sobel")

    t.tic()
    markers = selectMarkers(gradient, hexaGrid)
    t.toc("Markers")
    t.tic()
    distImage, _ = voronoiTesselation(
        img.shape, markers, sigma, 'euclidean', visu=True)

    t.toc("voronoi")

    t.tic()

    g_reg = gradient + k * (distImage)

    markers_map = np.zeros_like(g_reg)

    for i, marker in enumerate(markers):
        for point in marker:
            markers_map[point[0], point[1]] = i + 1

    labels = watershed(g_reg, markers_map, watershed_line=True)

    t.toc("watershed")
    indices = np.where(labels == 0)

    for i in range(len(indices[0])):

        cv2.circle(img, np.int32([indices[1][i], indices[0][i]]),
                   1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

    cv2.imwrite("waterpixels.jpg",
                img)
    return len(hexaGrid.centers)

    # parameters
steps = [40]

# rho ne doit pas etre egale a 0 control that !
rho = 2 / 3

print(rho)

k = 4

# images, _ = load_BSDS(
#     "../../BSD500/images/val2/", "../../BSD500/ground_truth/val/")

images = []

images.append(cv2.imread("../images/image1.jpg"))

sigma = 1

t = TicToc()

for j in range(len(steps)):

    for i in range(len(images)):
        t.tic()
        nbCenters = waterPixels(
            images[i], sigma, steps[j], rho, k)

        t.toc("Waterpixels")
