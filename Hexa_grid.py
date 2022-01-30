from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import cv2
# centre = centre de l'hexa
# size = rayon de l hexa (dist entre chaque point et le centre de l'hexa)
# i = corner nbr (6 in total)


def hex_corner(center, size, i):
    degree_angle = 60 * i
    rad_angle = np.deg2rad(degree_angle)
    return (int(center[0] + size * np.cos(rad_angle)),
            int(center[1] + size * np.sin(rad_angle)))


def drawHexa(image, center, size):

    color = (55, 46, 101)

    thickness = 2

    points = []

    points.append(hex_corner(center, size, 0))

    for i in range(1, 6):

        points.append(hex_corner(center, size, i))

        image = cv2.line(
            image, points[-2], points[-1], color, thickness, cv2.LINE_AA)

    # dernier point
    image = cv2.line(image, points[0], points[-1],
                     color, thickness, cv2.LINE_AA)

    print("Hexa", points, "\n")
    return image


def drawGrid(image, sigma):

    h, w, _ = image.shape

    change = 1

    centers = []

    w_padd = (w % sigma)/2

    h_padd = (h % sigma)/2

    init_x = w_padd + sigma

    init_y = h_padd + sigma

    center = (init_x, init_y)

    centers.append(center)

    for i in range(1, int((w / sigma))):

        for j in range(1, int((h / sigma))):

            print("center", center, "\n")

            image = drawHexa(image, center, sigma)

            # The sqrt(3) comes from sin(60°)
            center = (center[0], center[1] + np.sqrt(3) * sigma)

            centers.append(center)

        center = (init_x + i * 3/2 * sigma,
                  init_y - change * (np.sqrt(3) * sigma)/2)

        if(change == 0):
            change = 1
        else:

            change = 0

    plt.imshow(image, cmap='gray')

    plt.show()


# img = io.imread('image.jpg')
img = io.imread('image.jpg')


# conversion to gray scale image
gray_img = color.rgb2gray(img)


drawGrid(img, 15)
