
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import color
import cv2
# centre = centre de l'hexa
# size = rayon de l hexa (dist entre chaque point et le centre de l'hexa)
# i = corner nbr (6 in total)


def hexPoint(center, size, i):

    degree_angle = 60 * i
    rad_angle = np.deg2rad(degree_angle)
    return (int(center[0] + size * np.cos(rad_angle)),
            int(center[1] + size * np.sin(rad_angle)))


def drawHexa(image, center, size, color=(255, 255, 255)):

    # color = (55, 46, 101)

    thickness = 1

    points = getHexaPoints(center, size)

    for i in range(1, 6):

        image = cv2.line(
            image, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    # last point
    image = cv2.line(image, points[0], points[-1],
                     color, thickness, cv2.LINE_AA)

    return image


def getHexaPoints(center, size):

    points = []

    for i in range(0, 6):

        points.append(hexPoint(center, size, i))

    return points


def isHexInImage(w, h, center, size):

    # car parfois y a de mini pixels qui flood je sais pas si je dois les considerer
    epsilon = 5

    points = getHexaPoints(center, size)

    for point in points:

        x, y = point

        if(x >= 0 and x+epsilon < w and y >= 0 and y+epsilon < h):

            return True
    return False


def drawGrid(image, sigma, color=(255, 255, 255)):
    # color image
    if(len(image.shape) == 3):

        h, w, _ = image.shape
    # grayscale image
    else:
        h, w = image.shape

    change = 1

    centers = []

    size = sigma / np.sqrt(3)

    center = (0, 0)

    while isHexInImage(w, h, center, size):

        while(isHexInImage(w, h, center, size)):

            image = drawHexa(image, center, size)

            centers.append(center)

            cv2.circle(image, (int(center[0]), int(
                center[1])), 0, color, 5)

            # The sqrt(3) comes from sin(60°)
            center = (center[0], center[1] + sigma)

        center = (center[0] + (3/2)*size,
                  change*sigma/2)

        if(change == 0):

            change = 1

        else:

            change = 0

    print(len(centers))

    plt.imshow(image, cmap='gray')

    plt.show()


# img = io.imread('image.jpg')
img = io.imread('lena.jpg')


# conversion to grxay scale image
gray_img = color.rgb2gray(img)


drawGrid(img, 45)
