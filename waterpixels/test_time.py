import os
import cv2
from scipy import io
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
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

    hexaGrid = HexaGrid(sigma, rho)

    hexaGrid.computeCenters(img.shape)

    # conversion to gray scale images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # # computing a Sobel operator gradient
    gradient = sobelOperator(gray_img, g_sigma)

    markers = selectMarkers(gradient, hexaGrid)

    distImage, _ = voronoiTesselation(
        img.shape, markers, sigma, 'euclidean')

    g_reg = gradient + k * (distImage)

    markers_map = np.zeros_like(g_reg)

    for i, marker in enumerate(markers):
        for point in marker:
            markers_map[point[0], point[1]] = i + 1

    labels = watershed(g_reg, markers_map, watershed_line=True)

    return len(hexaGrid.centers)

    # parameters
steps = np.arange(10, 50, 5)

# rho ne doit pas etre egale a 0 control that !
rho = 2 / 3

k = 8

images, _ = load_BSDS(
    "../BSD500/images/val/", "../BSD500/ground_truth/val/")


images1, _, _ = load_SBD(
    "../SBD/images/", "../SBD/labels/")
# print(len(images))
x_bsds = np.zeros(len(steps))
x_sbd = np.zeros(len(steps))
time_bsds = np.zeros(len(steps))

time_sbd = np.zeros(len(steps))

sigma = 1

for j in range(len(steps)):

    for i in range(len(images)):

        # waterpixels
        # t = TicToc()
        # t.tic()
        # nbCenters = waterPixels(
        #     images[i], sigma, steps[j], rho, k)

        # time_bsds[j] += t.tocvalue()

        t = TicToc()
        t.tic()
        nbCenters1 = waterPixels(
            images1[i], sigma, steps[j], rho, k)
        print(nbCenters1)
        time_sbd[j] += t.tocvalue()

    #x_bsds[j] = nbCenters
    x_sbd[j] = nbCenters1


time_bsds /= len(images)
time_sbd /= len(images)


plotTime(x_bsds, time_bsds, x_sbd, time_sbd)
