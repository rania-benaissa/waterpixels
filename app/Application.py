import sys

import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWidgets import QFileDialog
from skimage import io

from Interface import Ui_MainWindow  # importing our generated file


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
        self.ui.rho_view.setText(str(self.ui.rho_slider.value()/100))
        self.ui.k_view.setText(str(self.ui.k_slider.value()))

        self.ui.visu.setVisible(False)
        self.ui.apply_waterpixels.setEnabled(False)
        self.url = ""

        ###### SLOTS and calls ######

        # menu principal
        self.ui.browserButton.clicked.connect(self.browseImage)
        self.ui.load_other_img.clicked.connect(self.browseImage)

        self.ui.grid_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 1))
        self.ui.rho_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 2))
        self.ui.k_slider.valueChanged.connect(
            lambda value: self.updateSlider(value, 3))

    def browseImage(self):

        # get the image
        selected_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                    'c:/', "Image files (*.jpg *.png)")

        self.url = selected_file[0]

        # if i selected an image gotta choose the recognition sys
        if(self.url != ""):

            self.image = io.imread(self.url)

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
            self.ui.rho_view.setText(str(float(value/100)))

        if(choice == 3):
            self.ui.k_view.setText(str(value))


# APP Starting


app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())
