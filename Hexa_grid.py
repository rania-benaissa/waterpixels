
import numpy as np
import matplotlib.pyplot as plt
import cv2

# centre = centre de l'hexa
# size = rayon de l hexa (dist entre chaque point et le centre de l'hexa)
# i = corner nbr (6 in total)

color = 0

thickness = 1


def hexPoint(center, size, i):

    degree_angle = 60 * i
    rad_angle = np.deg2rad(degree_angle)
    return (int(center[0] + size * np.cos(rad_angle)),
            int(center[1] + size * np.sin(rad_angle)))

# dessine un hexagone


def drawHexa(image, center, size, vertices=None):

    # color = (55, 46, 101)

    if(vertices == None):

        points = getHexaVertices(center, size)

    else:

        points = vertices

    for i in range(6):

        image = cv2.line(
            image, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    return image


def getHexaVertices(center, size):

    points = []

    for i in range(0, 6):

        points.append(hexPoint(center, size, i))

    return points


def isHexInImage(w, h, center, size):

    # car parfois y a de mini pixels qui flood je sais pas si je dois les considerer
    epsilon = 5

    points = getHexaVertices(center, size)

    for point in points:

        x, y = point

        if(x >= 0 and x+epsilon < w and y >= 0 and y+epsilon < h):

            return True
    return False

# the homothty operation


def addMargin(image, centers, rho, size):

    new_vertices = []

    for center in centers:

        vertices = getHexaVertices(center, size)

        new_vertices = []

        for v in vertices:

            x, y = v

            new_x = int(rho * (x-center[0]) + center[0])
            new_y = int(rho * (y-center[1]) + center[1])

            image = cv2.line(
                image, [new_x, new_y], v, color, thickness, cv2.LINE_AA)

            new_vertices.append([new_x, new_y])

        cv2.fillPoly(image, np.int32(
            [[new_vertices], [vertices]]), color)

    return image


def drawHexaGrid(image, sigma, color=(255, 255, 255), rho=2/3):
    # color image
    if(len(image.shape) == 3):

        h, w, _ = image.shape
    # grayscale image
    else:
        h, w = image.shape

    change = 1

    centers = []

    size = sigma / np.sqrt(3)

    center = [0, 0]

    while isHexInImage(w, h, center, size):

        while(isHexInImage(w, h, center, size)):

            image = drawHexa(image, center, size)

            centers.append(center)

            # cv2.circle(image, (int(center[0]), int(
            #     center[1])), 0, color, 3)

            # The sqrt(3) comes from sin(60°)
            center = [center[0], center[1] + sigma]

        center = [center[0] + (3/2)*size,
                  change*sigma/2]

        if(change == 0):

            change = 1

        else:

            change = 0

    cv2.imwrite("grid_image.jpg", image)

    image = addMargin(image, centers, rho, size)

    cv2.imwrite("grid_image_with_margin.jpg", image)
    print("nb centers ", len(centers))

    # plt.imshow(image, cmap='gray')

    # plt.show()

    return centers
