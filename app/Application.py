import sys
sys.path.append('../waterpixels/')
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWidgets import QFileDialog
from skimage.segmentation import watershed
from Interface import Ui_MainWindow  # importing our generated file
from waterpixels.HexaGrid import HexaGrid
from waterpixels.Gradient import sobelOperator, selectMarkers
from waterpixels.Voronoi_tesselation import voronoiTesselation
import numpy as np
import cv2


class mywindow(QtWidgets.QMainWindow):

    def __init__(self):

        super(mywindow, self).__init__()

        # la ou y a les initialisation
        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)
        # here it s to change the font
        self.font = QFont("Century Gothic", 12)

        self.font.setBold(True)

        # set my style sheet

        self.setFont(self.font)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        ############## INITIALISATIONS ###############

        self.ui.loaded_img.setVisible(False)
        self.ui.load_other_img.setVisible(False)

        self.ui.grid_view.setText(str(self.ui.grid_slider.value()))
        self.ui.rho_view.setText(str(self.ui.rho_slider.value() / 100))
        self.ui.k_view.setText(str(self.ui.k_slider.value()))
        self.ui.kernel_size_view.setText(
            str(self.ui.kernel_size_slider.value()))
        self.ui.std_view.setText(
            str(self.ui.std_slider.value()))

        self.ui.visu.setVisible(False)
        self.ui.apply_waterpixels.setEnabled(False)
        self.url = ""

        ###### SLOTS and calls ######

        # menu principal
        self.ui.browserButton.clicked.connect(self.browseImage)
        self.ui.load_other_img.clicked.connect(self.browseImage)
        self.ui.apply_waterpixels.clicked.connect(self.waterpixels)

        self.ui.grid_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 1))
        self.ui.rho_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 2))
        self.ui.k_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 3))
        self.ui.kernel_size_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 4))
        self.ui.std_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 5))

    def browseImage(self):

        # get the image
        selected_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                    'c:/', "Image files (*.jpg *.png)")

        self.url = selected_file[0]

        # if i selected an image gotta choose the recognition sys
        if(self.url != ""):

            self.image = cv2.imread(self.url)

            self.ui.loaded_img.setVisible(True)
            self.ui.browserButton.setVisible(False)
            self.ui.load_other_img.setVisible(True)
            self.ui.apply_waterpixels.setEnabled(True)

            h = self.ui.loaded_img.height()

            w = self.ui.loaded_img.width()
            # affiche l image
            pixmap = QPixmap(self.url)
            self.ui.loaded_img.setPixmap(
                pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))

        else:
            pass

    def updateSlider(self, value, choice):

        if(choice == 1):
            self.ui.grid_view.setText(str(value))

        if(choice == 2):
            self.ui.rho_view.setText(str(float(value / 100)))

        if(choice == 3):
            self.ui.k_view.setText(str(value))

        if(choice == 4):
            self.ui.kernel_size_view.setText(str(value))

        if(choice == 5):
            self.ui.std_view.setText(str(value))

    def waterpixels(self):

        # get values
        k = int(self.ui.k_view.text())
        sigma = int(self.ui.grid_view.text())
        rho = float(self.ui.rho_view.text())

        kernel_size = int(self.ui.kernel_size_view.text())

        g_sigma = 1

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(self.image.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite("image_gray"+str(count)+".jpg", gray_img)

        if(self.ui.sobel_radio.isChecked()):
            print("youhou")

        # # # computing a Sobel operator gradient
        gradient = sobelOperator(gray_img, g_sigma)

        # cv2.imwrite("image_gradient"+str(count)+".jpg", gradient)

        image_grid = hexaGrid.drawHexaGrid(self.image)

        # cv2.imwrite("image_grid.jpg", image_grid)

        # compute minimas and select markers

        markers = selectMarkers(gradient, hexaGrid)

        distImage, _ = voronoiTesselation(
            self.image.shape, markers, sigma, 'euclidean')

        g_reg = gradient + k * (distImage)

        # cv2.imwrite("regularized_gradient"+str(count)+".jpg", g_reg)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i + 1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        contours = np.where(labels == 0)

        print("done")


# APP Starting
app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())
