
import numpy as np
import matplotlib.pyplot as plt
import cv2


color = 0

thickness = 1


def hexPoint(center, size, i):
    """Calculates the position of the vertex i of an hexagon.

    Parameters
    ----------
        center : the center of the hexagon.

        size : radius of the hexagon (distance between every vertex and its center).

        i : vertex number.

    Returns:
    ----------
    A tuple (x,y) which represents the position of vertex i in the grid.

    """

    degree_angle = 60 * i
    rad_angle = np.deg2rad(degree_angle)

    return (int(center[0] + size * np.cos(rad_angle)),
            int(center[1] + size * np.sin(rad_angle)))


def drawHexa(image, center, size):
    """Draws an hexagon in an image.

    Parameters
    ----------
        image : the image where the hexagon is drawn.

        center : the center of the hexagon.

        size : the radius of the hexagon (distance between every vertex and its center).

    Returns:
    ----------
    The modified image.

    """

    points = getHexaVertices(center, size)

    for i in range(6):

        image = cv2.line(
            image, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    return image


def getHexaVertices(center, size):
    """Computes the hexagon's vertices.

    Parameters
    ----------
        center : the center of the hexagon.

        size : radius of the hexagon (distance between every vertex and its center).

    Returns:
    ----------
    points : list of vertices positions.

    """
    points = []

    for i in range(0, 6):

        points.append(hexPoint(center, size, i))

    return points


def isHexInImage(w, h, center, size):
    """Check if a hexagon is in an image given its dimensions.

    Parameters
    ----------
        w : width of the image.

        h : height of the image.

        center : the center of the hexagon.

        size : radius of the hexagon (distance between every vertex and its center).

    Returns:
    ----------
    bool : True, if the hexagon is inside the image, False otherwise.

    """

    # dunno if i should consider all pixels
    epsilon = 0

    points = getHexaVertices(center, size)

    for point in points:

        x, y = point

        if(x >= 0 and x+epsilon <= h and y >= 0 and y+epsilon <= w):

            return True
    return False


def addMargin(img, centers, rho, size):
    """Computes homothety centered on hexagons centers.

    Parameters
    ----------
        image : image with an hexagonal grid

        centers : hexagon's centers.

        rho : homothety factor [0,1].

    Returns:
    ----------
    image : the image with a modified grid.

    """
    image = img.copy()

    new_vertices = []

    for center in centers:
        # inversed centers so that it can be drawn
        vertices = getHexaVertices(center[::-1], size)

        new_vertices = []

        for v in vertices:

            x, y = v
            # just inversed the positions so that it can be drawn
            new_x = int(rho * (x-center[1]) + center[1])
            new_y = int(rho * (y-center[0]) + center[0])

            image = cv2.line(
                image, [new_x, new_y], v, color, thickness, cv2.LINE_AA)

            new_vertices.append([new_x, new_y])

        cv2.fillPoly(image, np.int32(
            [[new_vertices], [vertices]]), color)

    return image


def drawHexaGrid(img, sigma, rho=2/3):
    """Draws an hexagonal grid on an image.

    Parameters
    ----------
        image : color or gray-scale image.

        sigma : the grid step.

        rho : homothety factor [0,1].

    Returns:
    ----------
    centers : the grid centers.

    """

    image = img.copy()
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

        # la boucle interne est celle du H
        while(isHexInImage(w, h, center, size)):

            # needed to invert the center
            # coz cv2 works with w *h
            # meanwhile i m workin with h * w
            image = drawHexa(image, center[::-1], size)

            centers.append(center)

            cv2.circle(image, (int(center[1]), int(
                center[0])), 0, color, 3)

            # plt.imshow(image, cmap="gray")
            # plt.show()

            # The sqrt(3) comes from sin(60°) , x =hauteur, y = largeur
            center = [center[0] + sigma, center[1]]

        center = [change*sigma/2, center[1] + (3/2)*size]

        if(change == 0):

            change = 1

        else:

            change = 0

    cv2.imwrite("grid_image.jpg", image)
    # homothety operation
    image = addMargin(image, centers, rho, size)

    cv2.imwrite("grid_image_with_margin.jpg", image)
    print("nb centers ", len(centers))

    return centers
