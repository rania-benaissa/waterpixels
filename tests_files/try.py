from scipy.spatial.distance import cdist
import numpy as np

from itertools import product
import cv2


from pytictoc import TicToc
# a = np.random.rand(1000, 2)
# b = np.random.rand(100000, 2)


# t = TicToc()
# t.tic()
# dist = cdist(a, b, "euclidean")
# print(dist)
# print(dist.shape)
# t.toc()

# from numba import jit


# @jit(nopython=True)
# def eudis(a, b):
#     x = np.sum(a**2, axis=1)[:, np.newaxis]
#     y = np.sum(b**2, axis=1)
#     xy = np.dot(a, b.T)
#     return np.sqrt(x + y - 2 * xy)


# t = TicToc()
# t.tic()

# dist = eudis(a, b)

# print(dist)
# print(dist.shape)
# t.toc()

# import cv2

size = 3

# print("Rect")
# struct_elt = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
# print(struct_elt)
# # print("Ellipse")
# struct_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

# print(struct_elt)
# # print("Cross")
# struct_elt = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
# print(struct_elt)


size = 3

binary_image = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

print(binary_image)

image = np.zeros((5, 5), dtype=np.uint8)

for i in range(size):
    for j in range(size):
        image[j + 1, + i + 1] = (i + 1) + j * 3


print(image)

print(" -----------------")
print(cv2.morphologyEx(image, cv2.MORPH_ERODE, binary_image))

# a = np.array(np.where(binary_image == 1)).T

# img = list(product(np.arange(size), np.arange(size)))
# t = TicToc()
# t.tic()

# dist = cdist(a, img, "euclidean")

# gray_diagram = np.min(dist, axis=0)


# gray_diagram = gray_diagram.reshape((size, size))
# t.toc()
# # print(gray_diagram.shape)


# print(gray_diagram)


# binary_image = 1 - binary_image


# t = TicToc()
# t.tic()
# dist, voronoi = cv2.distanceTransformWithLabels(binary_image,
#                                                 cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
# t.toc()
# print(np.array(dist))


# t = TicToc()
# t.tic()
# dist, voronoi = cv2.distanceTransformWithLabels(binary_image,
#                                                 cv2.DIST_L2, cv2.DIST_MASK_5)
# t.toc()
# print(np.array(dist))
