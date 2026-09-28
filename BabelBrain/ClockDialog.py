import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from Utils.paths import resource_path


class ClockDialog(QDialog):
    def __init__(self, parent=None):
        super(ClockDialog,self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setModal(False)

        self.label = QLabel(self)
        self.movie = QMovie(os.path.join(resource_path(__file__), 'icons8-hourglass.gif'))
        self.label.setMovie(self.movie)
        self.movie.start()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
