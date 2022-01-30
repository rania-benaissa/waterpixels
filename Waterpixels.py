# importing PIL
import matplotlib.pyplot as plt
from skimage import color
from Gradient import SobelOperator
from skimage import io


img = io.imread('image.jpg')


# conversion to gray scale image
gray_img = color.rgb2gray(img)

# computing a Sobel operator gradient
grad_img = SobelOperator(gray_img, 0.3)


plt.imshow(grad_img, cmap='gray')

plt.show()
