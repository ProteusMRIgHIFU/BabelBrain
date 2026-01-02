from PySide6.QtCore import Qt
from PySide6.QtGui import  QMovie

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel)

import platform
import os
import sys
from pathlib import Path

_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class ClockDialog(QDialog):
    def __init__(self, parent=None):
        super(ClockDialog,self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setModal(False)

        self.label = QLabel(self)
        self.movie = QMovie( os.path.join(resource_path(),'icons8-hourglass.gif'))
        self.label.setMovie(self.movie)
        self.movie.start()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
