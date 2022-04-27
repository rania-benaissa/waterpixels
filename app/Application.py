import sys
import cv2

import numpy as np
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

        self.url = ""

        ###### SLOTS and calls ######

        # menu principal
        self.ui.browserButton.clicked.connect(self.browseImage)
        self.ui.load_other_img.clicked.connect(self.browseImage)

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

            # affiche l image
            self.ui.loaded_img.setPixmap(QPixmap(self.url))

        else:
            pass
            # show a message


# APP Starting


app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())
