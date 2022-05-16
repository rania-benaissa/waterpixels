from fileinput import filename
import os
import cv2
from Gradient import *
from HexaGrid import HexaGrid
from Voronoi_tesselation import voronoiTesselation
from skimage.segmentation import watershed
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.measure import regionprops


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


def plotGR(x_wp, gr_wp, x_slic, gr_slic, title=None):

    plt.rcParams["font.family"] = 'Comic Sans MS'
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    colors = ["#3CBCC7", "#1C96BA", "#0C73B2", "#094C93", "#51BC37"]

    plt.plot(x_wp, gr_wp[0], label='WP with k = 0', marker='o',
             color=colors[0])

    plt.plot(x_wp, gr_wp[1], label='WP with k = 4', marker='o',
             color=colors[1])

    plt.plot(x_wp, gr_wp[2], label='WP with k = 8', marker='o',
             color=colors[2])

    plt.plot(x_wp, gr_wp[3], label='WP with k = 16', marker='o',
             color=colors[3])

    plt.plot(x_slic, gr_slic, label='SLICO', marker='o',
             color=colors[4])

    plt.xlabel("Nb. Superpixels", fontsize=14)
    plt.ylabel("Global regularity", fontsize=14)

    if(title is not None):
        plt.title(title)

    plt.legend()
    plt.savefig('SBD_gr.jpg', bbox_inches='tight')
    # plt.show()


def SRC(image, label_image):

    regions = regionprops(label_image)

    # print(" regions  =", len(regions))

    src = 0

    for region in regions:

        cc_shape = region.perimeter / region.area
        convex_hull = region.image_convex
        # to get convex hull perimeter
        hull_regions = regionprops(np.array(convex_hull, np.uint8))

        cc_convex = hull_regions[0].perimeter / region.area_convex
        # regularity criteria btw 0 and 1

        if(cc_shape == 0):
            cr = 0
        else:
            cr = cc_convex / cc_shape

        # compute V_xy
        coords_x = region.coords.T[0]
        coords_y = region.coords.T[1]

        std_x = np.std(coords_x)
        std_y = np.std(coords_y)

        # between 0 and 1

        if(std_x == 0 and std_y == 0):
            v_xy = 0
        else:
            v_xy = np.sqrt(np.minimum(std_x, std_y) / np.maximum(std_x, std_y))

        src += region.area * cr * v_xy

    return src / (image.shape[0] * image.shape[1])


def getCenteredShape(shape, region):

    centered_img = np.zeros((shape))

    centroid_x, centroid_y = (
        int(region.centroid_local[0]), int(region.centroid_local[1]))

    dist = (int(shape[0] / 2) - centroid_x,
            int(shape[1] / 2) - centroid_y)

    # binary image
    img_bb = region.image

    # compute the centered image
    for x in range(img_bb.shape[0]):
        for y in range(img_bb.shape[1]):

            centered_img[x + dist[0], y + dist[1]] = img_bb[x, y]

    return centered_img


def getCenteredAvgShape(shape, nb_superpixels, label_image):

    avg_image = np.zeros(shape)

    regions = regionprops(label_image)

    for region in regions:

        centered_img = getCenteredShape(shape, region)

        avg_image += centered_img

    return avg_image / nb_superpixels


def SMF(image, label_image, nb_superpixels):

    shape = (2 * image.shape[0] + 1, 2 * image.shape[1] + 1)
    # get S*
    avg_image = getCenteredAvgShape(
        shape, nb_superpixels, label_image)

    card_avg_img = np.array(np.where(avg_image != 0)).shape[1]

    avg_image /= card_avg_img

    regions = regionprops(label_image)

    smf = 0

    for region in regions:

        # compute S_k*
        centered_img = getCenteredShape(
            shape, region) / region.area

        diff = np.abs(avg_image - centered_img)

        smf += region.area * (diff.sum() / 2)

    return 1 - (smf / (image.shape[0] * image.shape[1]))


def globalReg(image, label_image, nb_superpixels):

    src = SRC(image, label_image)

    smf = SMF(image, label_image, nb_superpixels)

    return src * smf


def waterPixels(img, g_sigma=0, sigma=40, rho=2 / 3, k=8):

    if(img is not None):

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(img.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        # # # computing a Sobel operator gradient
        gradient = sobelOperator(gray_img, g_sigma)

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
                markers_map[point[0], point[1]] = i + 1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        contours = np.where(labels == 0)

        # for i in range(len(contours[0])):

        #     cv2.circle(img, np.int32([contours[1][i], contours[0][i]]),
        #                1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        # cv2.imwrite("waterpixels.jpg", img)

    return img, contours, len(hexaGrid.centers), labels

    # parameters
steps = np.arange(10, 50, 5)

# rho ne doit pas etre egale a 0 control that !
rho = 2 / 3

reg_params = [0, 4, 8, 16]
images, gt_contours, gt_masks = load_SBD(
    "../../SBD/images/", "../../SBD/labels/")


x_wp = np.zeros(len(steps))
x_slic = np.zeros(len(steps))
gr_wp = np.zeros((4, len(steps)))
gr_slic = np.zeros(len(steps))


sigma = 1

for j in range(len(steps)):

    for i, image in enumerate(images):

        #fig, axs = plt.subplots(1, 6)

        for k in range(len(reg_params)):

            waterpixels, contours, nbCenters, labels = waterPixels(
                image, sigma, steps[j], rho, reg_params[k])

            # print(np.array(contours).shape)

            gr_wp[k][j] += globalReg(image, labels, nbCenters)

            # mask_contours = np.zeros(image.shape, np.uint8)
            # mask_contours[contours] = 255
            # axs[k].imshow(mask_contours)
            # axs[k].set_title("k = " + str(reg_params[k]))
            # axs[k].axis('off')

        sp_slic = slic(image, n_segments=nbCenters, slic_zero=True, sigma=sigma,
                       start_label=1)

        slic_contours, mask_slic = labelsToContours(sp_slic, image)

        gr_slic[j] += globalReg(image, sp_slic, len(np.unique(sp_slic)))

        print("Slic superpixels ", len(np.unique(sp_slic)))

        # axs[k+1].imshow(gt_masks[i])
        # axs[k+1].set_title("Ground truth")
        # axs[k+1].axis('off')

        # axs[k+2].imshow(mask_slic)
        # axs[k+2].set_title("SLIC contours")
        # axs[k+2].axis('off')

        # fig.tight_layout()
        # plt.show()

    x_wp[j] = nbCenters
    x_slic[j] = len(np.unique(sp_slic))


gr_wp /= len(images)
gr_slic /= len(images)

plotGR(x_wp, gr_wp, x_slic, gr_slic)
