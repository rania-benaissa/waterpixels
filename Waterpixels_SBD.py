import os
import cv2
from scipy.spatial.distance import cdist
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.segmentation import slic


def load_SBD(images_folder, ground_folder):
    images = []

    gt_contours = []

    for filename in os.listdir(images_folder):

        # correspondant ground truth filename
        gFilename = filename[:-3]+"layers.txt"

        img = cv2.imread(os.path.join(images_folder, filename))

        label_img = np.loadtxt(os.path.join(
            ground_folder, gFilename), np.uint8)

        contours = find_boundaries(label_img, mode='outer')

        contours = np.where(contours == True)

        gt_contours.append(np.array(contours))

        images.append(img)

    return images, gt_contours


def plotBR(x_wp, br_wp, x_slic, br_slic, title=None):

    plt.rcParams["font.family"] = 'Comic Sans MS'
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    colors = ["#3CBCC7", "#1C96BA", "#0C73B2", "#094C93", "#51BC37"]

    plt.plot(x_wp, br_wp[0], label='WP with k = 0', marker='o',
             color=colors[0])

    plt.plot(x_wp, br_wp[1], label='WP with k = 4', marker='o',
             color=colors[1])

    plt.plot(x_wp, br_wp[2], label='WP with k = 8', marker='o',
             color=colors[2])

    plt.plot(x_wp, br_wp[3], label='WP with k = 16', marker='o',
             color=colors[3])

    plt.plot(x_slic, br_slic, label='SLICO', marker='o',
             color=colors[4])

    plt.xlabel("Nb. Superpixels", fontsize=14)
    plt.ylabel("Boundary Recall", fontsize=14)

    if(title is not None):
        plt.title(title)

    plt.legend()
    plt.savefig('br.jpg', bbox_inches='tight')
    plt.show()


def plotBR2(x_wp, y_wp, x_slic, y_slic, title=None):

    plt.rcParams["font.family"] = 'Comic Sans MS'
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    colors = ["#3CBCC7", "#1C96BA", "#0C73B2", "#094C93", "#51BC37"]

    plt.plot(x_wp[0], y_wp[0], label='WP with k = 0', marker='o',
             color=colors[0])

    plt.plot(x_wp[1], y_wp[1], label='WP with k = 4', marker='o',
             color=colors[1])

    plt.plot(x_wp[2], y_wp[2], label='WP with k = 8', marker='o',
             color=colors[2])

    plt.plot(x_wp[3], y_wp[3], label='WP with k = 16', marker='o',
             color=colors[3])

    plt.plot(x_slic, y_slic, label='SLICO', marker='o',
             color=colors[4])

    plt.xlabel("Boundary Recall", fontsize=14)
    plt.ylabel("Contour Density", fontsize=14)

    if(title is not None):
        plt.title(title)

    plt.legend()
    plt.savefig('br2.jpg', bbox_inches='tight')
    plt.show()


def boundaryRecall(contours, gt_contours, min_dist=3):

    # print("contours", np.array(contours).shape)

    # print("ground truth", np.array(gt_contours).shape)

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

        # cv2.imwrite("waterpixels.jpg", img)

    return img, contours, len(hexaGrid.centers)


    # parameters
steps = np.arange(45, 50, 5)

# rho ne doit pas etre egale a 0 control that !
rho = 2/3

reg_params = [0, 4, 8, 16]
images, gt_contours = load_SBD(
    "../SBD/images/", "../SBD/labels/")


x_wp = np.zeros(len(steps))
x_slic = np.zeros(len(steps))

cd_wp = np.zeros((4, len(steps)))
cd_slic = np.zeros(len(steps))
br_wp = np.zeros((4, len(steps)))

br_slic = np.zeros(len(steps))


sigma = 0.5

for j in range(len(steps)):

    for i, image in enumerate(images):

        fig, axs = plt.subplots(1, 6)

        for k in range(len(reg_params)):

            waterpixels, contours, nbCenters = waterPixels(
                image, sigma, steps[j], rho, reg_params[k])

            # print(np.array(contours).shape)

            br_wp[k][j] += boundaryRecall(contours, gt_contours[i])

            cd_wp[k][j] += np.array(contours).shape[1] / \
                (image.shape[0]*image.shape[1])

            mask_contours = np.zeros(image.shape, np.uint8)
            mask_contours[contours] = 255
            axs[k].imshow(mask_contours)
            axs[k].set_title("k = " + str(reg_params[k]))
            axs[k].axis('off')

        sp_slic = slic(image, n_segments=nbCenters, slic_zero=True, sigma=sigma,
                       start_label=1)

        slic_contours = find_boundaries(sp_slic, mode='outer')

        slic_contours = np.where(slic_contours == True)

        # print(np.array(slic_contours).shape)

        br_slic[j] += boundaryRecall(slic_contours, gt_contours[i])
        cd_slic[j] += np.array(slic_contours).shape[1] / \
            (image.shape[0]*image.shape[1])

        print("Slic superpixels ", len(np.unique(sp_slic)))

        #mask_gt = np.zeros(image.shape, np.uint8)

        # for point in gt_contours[i].T:
        #     x, y = point
        #     mask_gt[y, x] = 255
        # mask_gt[gt_contours[i]] = 255
        axs[k+1].imshow(mark_boundaries(image, gt_contours[i]))
        axs[k+1].set_title("Ground truth")
        axs[k+1].axis('off')

        axs[k+2].imshow(mark_boundaries(image, sp_slic))
        axs[k+2].set_title("SLIC contours")
        axs[k+2].axis('off')

        fig.tight_layout()
        plt.show()

    x_wp[j] = nbCenters
    x_slic[j] = len(np.unique(sp_slic))


br_wp /= len(images)
br_slic /= len(images)
cd_wp /= len(images)
cd_slic /= len(images)

plotBR(x_wp, br_wp, x_slic, br_slic)


plotBR2(br_wp, cd_wp, br_slic, cd_slic)
