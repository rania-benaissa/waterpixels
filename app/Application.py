import sys
sys.path.append('../waterpixels/')
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog
from skimage.segmentation import watershed
from Interface import Ui_MainWindow  # importing our generated file
from waterpixels.HexaGrid import HexaGrid
from waterpixels.Gradient import morphologicalGradient, sobelOperator, selectMarkers
from waterpixels.Voronoi_tesselation import voronoiTesselation
import numpy as np
import cv2
from PIL import Image, ImageQt


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

        self.ui.load_other_img.setVisible(False)
        self.ui.reset.setVisible(False)

        self.ui.grid_view.setText(str(self.ui.grid_slider.value()))
        self.ui.rho_view.setText(str(self.ui.rho_slider.value() / 100))
        self.ui.k_view.setText(str(self.ui.k_slider.value()))

        self.ui.std_view.setText(
            str(self.ui.std_slider.value() / 100))

        self.ui.visu.setVisible(False)
        self.ui.apply_waterpixels.setEnabled(False)
        self.url = ""
        self.last_url = ""

        self.gradient = ""
        self.image_grid = " "
        self.voronoi = " "
        ###### SLOTS and calls ######

        # menu principal
        self.ui.browserButton.clicked.connect(self.browseImage)
        self.ui.load_other_img.clicked.connect(self.browseImage)
        self.ui.apply_waterpixels.clicked.connect(self.waterpixels)
        self.ui.reset.clicked.connect(self.reset)

        self.ui.grid_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 1))
        self.ui.rho_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 2))
        self.ui.k_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 3))
        self.ui.std_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 4))

        self.ui.original_img.toggled.connect(lambda: self.showImage())
        self.ui.waterpixels.toggled.connect(self.showWaterpixels)
        self.ui.gradient_img.toggled.connect(
            lambda: self.showImage(self.gradient))
        self.ui.original_hexa.toggled.connect(
            lambda: self.showImage(self.image_grid))
        self.ui.voronoi.toggled.connect(lambda: self.showImage(self.voronoi))

    def reset(self):
        self.ui.load_other_img.setVisible(True)
        self.ui.reset.setVisible(False)
        self.ui.visu.setVisible(False)
        self.ui.parameters.setVisible(True)
        self.ui.apply_waterpixels.setVisible(True)
        self.ui.waterpixels.setChecked(True)
        self.showImage()

    def showWaterpixels(self):

        img = self.image.copy()

        for i in range(len(self.contours[0])):

            cv2.circle(img, np.int32([self.contours[1][i], self.contours[0][i]]),
                       1, [249, 217, 38][::-1], -1, cv2.LINE_AA)

        self.showImage(img)

    def showImage(self, image=""):

        h = self.ui.loaded_img.height()

        w = self.ui.loaded_img.width()

        if(type(image) == str):
            # affiche l image
            pixmap = QPixmap(self.url)
        else:

            image = image.astype(np.uint8)
            if(len(image.shape) == 3):

                image = Image.fromarray(image[:, :, ::-1])
            else:

                image = Image.fromarray(image, 'L')
            qt_img = ImageQt.ImageQt(image)
            pixmap = QtGui.QPixmap.fromImage(qt_img)

        self.ui.loaded_img.setPixmap(
            pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))

    def browseImage(self):

        # get the image
        selected_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                    'c:/', "Image files (*.jpg *.png)")

        self.url = selected_file[0]

        # if i selected an image gotta choose the recognition sys
        if(self.url != ""):

            self.last_url = self.url
            self.image = cv2.imread(self.url)

            self.ui.loaded_img.setVisible(True)
            self.ui.browserButton.setVisible(False)
            self.ui.load_other_img.setVisible(True)
            self.ui.visu.setVisible(False)
            self.ui.parameters.setVisible(True)
            self.ui.apply_waterpixels.setVisible(True)
            self.ui.apply_waterpixels.setEnabled(True)
            # on affiche l'image
            self.showImage()

        else:
            self.url = self.last_url

    def updateSlider(self, value, choice):

        if(choice == 1):
            self.ui.grid_view.setText(str(value))

        if(choice == 2):
            self.ui.rho_view.setText(str(float(value / 100)))

        if(choice == 3):
            self.ui.k_view.setText(str(value))

        if(choice == 4):
            self.ui.std_view.setText(str(float(value / 100)))

    def waterpixels(self):

        # get values
        k = int(self.ui.k_view.text())
        sigma = int(self.ui.grid_view.text())
        rho = float(self.ui.rho_view.text())

        hexaGrid = HexaGrid(sigma, rho)

        hexaGrid.computeCenters(self.image.shape)

        # conversion to gray scale images
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if(self.ui.sobel_radio.isChecked()):
            std = float(self.ui.std_view.text())

            self.gradient = sobelOperator(gray_img, std, 3)

        else:

            if(self.ui.rect.isChecked()):

                form = -1
            else:

                if(self.ui.cross.isChecked()):
                    form = -2

            # computing morphological gradient
            self.gradient = morphologicalGradient(gray_img, form, 3)

        self.image_grid = hexaGrid.drawHexaGrid(self.image)

        # compute minimas and select markers

        markers = selectMarkers(self.gradient, hexaGrid)

        distImage, self.voronoi = voronoiTesselation(
            self.image.shape, markers, sigma, 'euclidean', visu=True)

        g_reg = self.gradient + k * (distImage)

        markers_map = np.zeros_like(g_reg)

        for i, marker in enumerate(markers):
            for point in marker:
                markers_map[point[0], point[1]] = i + 1

        labels = watershed(g_reg, markers_map, watershed_line=True)

        self.contours = np.where(labels == 0)

        print("Waterpixels computation ----> complete")

        # update app
        self.ui.parameters.setVisible(False)
        self.ui.apply_waterpixels.setVisible(False)
        self.ui.visu.setVisible(True)
        self.ui.reset.setVisible(True)
        self.ui.load_other_img.setVisible(True)
        self.showWaterpixels()


# APP Starting
app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())
