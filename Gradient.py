import py


# both gray and color
import numpy as np
from scipy import signal


def gaussianKernel(sigma):
    """ double -> Array
        return a gaussian kernel of standard deviation sigma
    """
    n2 = np.int64(np.ceil(3*sigma))
    x, y = np.meshgrid(np.arange(-n2, n2+1), np.arange(-n2, n2+1))
    kern = np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()


def SobelOperator(image, sigma=None):

    s_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    s_y = s_x.T

    if(sigma != None):

        image = signal.convolve2d(image, gaussianKernel(sigma), mode='same')

    g_x = signal.convolve2d(image, s_x, mode='same')

    g_y = signal.convolve2d(image, s_y, mode='same')
    # magnitude
    return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))


# only gray lvl

# a morphological gradient is the difference between the dilation and the erosion of a given image.


def morphological_gradient():
    pass


def computeMinimas(img):

    image = np.zeros((img.shape[0], img.shape[1], 3))
    print(image.shape)
    image[:, :] = [255, 255, 255]

    mini = np.amin(img)

    print(mini)

    indices = np.where(img == mini)

    image[indices[0], indices[1]] = [0, 0, 0]

    return image
