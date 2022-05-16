
import numpy as np
import matplotlib.pyplot as plt
import cv2


class HexaGrid():

    def __init__(self, sigma, rho, color=[49, 73, 97], thickness=1):
        """Constructor

        Parameters
        ----------
            image : color or gray-scale image.

            sigma : the grid step.

            rho : homothety factor [0,1].

            color : grid color.

            thickness : grid thickness.

        """

        self.sigma = sigma

        self.size = self.sigma / np.sqrt(3)

        self.rho = rho

        self.color = color[::-1]

        self.thickness = thickness

        self.centers = []

    def hexPoint(self, center, i):
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

        return (int(center[0] + self.size * np.cos(rad_angle)),
                int(center[1] + self.size * np.sin(rad_angle)))

    def drawHexa(self, image, center):
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

        image = image.astype(np.uint8)

        points = self.getHexaVertices(center)

        for i in range(6):

            image = cv2.line(
                image, points[i - 1], points[i], self.color, self.thickness, cv2.LINE_AA)

        return image

    def getHexaVertices(self, center):
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

            points.append(self.hexPoint(center, i))

        return points

    def getHomoHexaVertices(self, center):
        """Computes the hexagon's vertices after homothety (inversed centers).

        Parameters
        ----------
            center : the center of the hexagon.


        Returns:
        ----------
        points : list of vertices positions after homothety.

        """

        new_vertices = []

        # inversed centers so that it can be drawn
        vertices = self.getHexaVertices(center[::-1])

        new_vertices = []

        for v in vertices:

            x, y = v
            # just inversed the positions so that it can be drawn
            new_x = int(self.rho * (x - center[1]) + center[1])
            new_y = int(self.rho * (y - center[0]) + center[0])

            new_vertices.append([new_x, new_y])

        return new_vertices

    def isHexInImage(self, w, h, center):
        """Check if a hexagon is in an image given its dimensions.

        Parameters
        ----------
            w : width of the image.

            h : height of the image.

            center : the center of the hexagon.

        Returns:
        ----------
        bool : True, if the hexagon is inside the image, False otherwise.

        """

        # dunno if i should consider all pixels
        epsilon = 10

        points = self.getHexaVertices(center)

        for point in points:

            x, y = point

            if(x >= 0 and x + epsilon <= h and y >= 0 and y + epsilon <= w):

                return True
        return False

    def addMargin(self, img):
        """Computes homothety centered on hexagons centers.

        Parameters
        ----------
            image : image with an hexagonal grid

        Returns:
        ----------
        image : the image with modified grid.

        """

        image = img.copy()

        mask = np.full((img.shape[0], img.shape[1]), 255, np.uint8)

        for center in self.centers:
            # inversed centers so that it can be drawn
            vertices = np.int32(self.getHomoHexaVertices(
                center))

            # create mask for my polygon

            cv2.fillPoly(mask, [vertices], (0))

            # get the indices inside the poly
        grid_indices = np.where(mask == 255)

        if(len(image.shape) == 3):

            image[grid_indices] = self.color
        else:

            image[grid_indices] = np.mean(self.color)

        return image

    def computeCenters(self, dims):
        """Computes centers of an hexagonal grid of a specific image.

        Parameters
        ----------
            dims : color or gray-scale image's dimensions.

        Returns:
        ----------
        centers : the grid centers.

        """

        # color image
        if(len(dims) == 3):

            h, w, _ = dims
        # grayscale image
        else:
            h, w = dims

        change = 1

        centers = []

        center = [0, 0]

        while self.isHexInImage(w, h, center):

            # la boucle interne est celle du H
            while(self.isHexInImage(w, h, center)):

                centers.append(center)

                center = [center[0] + self.sigma, center[1]]

            center = [change * self.sigma / 2, center[1] + (3 / 2) * self.size]

            if(change == 0):

                change = 1

            else:

                change = 0

        #print("nb centers ", len(centers))

        self.centers = centers

    def drawHexaGrid(self, img):
        """Draws an hexagonal grid on an image.

        Parameters
        ----------
            img : color or gray-scale image.


        Returns:
        ----------
        image : image with drawn grids.

        """

        # this is temporary
        grid_image = np.full((img.shape[0], img.shape[1], 3), 0)
        image = img.copy()

        for center in self.centers:
            # needed to invert the center
            # coz cv2 works with w *h
            # meanwhile i m workin with h * w
            image = self.drawHexa(image, center[::-1])

            grid_image = self.drawHexa(grid_image, center[::-1])

            # cv2.circle(image, (int(center[1]), int(
            #     center[0])), 0, color, 3)

            # plt.imshow(image, cmap="gray")
            # plt.show()

        #cv2.imwrite("grid.jpg", grid_image)
        # homothety operation
        image = self.addMargin(image)

        grid_image = self.addMargin(grid_image)

        #cv2.imwrite("grid_with_margin.jpg", grid_image)

        return image
