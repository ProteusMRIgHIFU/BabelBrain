from PySide6.QtWidgets import (QApplication, QWidget,QGridLayout,
                QHBoxLayout,QVBoxLayout,QLineEdit,QDialog,
                QGridLayout, QSpacerItem, QInputDialog, QFileDialog,
                QErrorMessage, QMessageBox)
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPalette, QTextCursor

import os
import sys
from pathlib import Path

import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'GUIComponents'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class ScrollBars(QWidget):
    def __init__(self,parent=None,MainApp=None):
        super(ScrollBars, self).__init__(parent)
        self._MainApp=MainApp
        self.load_ui()


    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(resource_path(), "scrollbars.ui")
        
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.IsppaScrollBar1.valueChanged.connect(self._MainApp.UpdateAcResults)
        self.Widget.IsppaScrollBar1.setEnabled(False)
        self.Widget.IsppaScrollBar2.valueChanged.connect(self._MainApp.UpdateAcResults)
        self.Widget.IsppaScrollBar2.setEnabled(False)

    def set_default_values(self, targetIndex, xvec, yvec):
        self.xcoords = xvec
        self.ycoords = yvec

        self.Widget.IsppaScrollBar1.setEnabled(True)
        self.Widget.IsppaScrollBar2.setEnabled(True)

        self.Widget.IsppaScrollBar1.blockSignals(True)
        self.Widget.IsppaScrollBar2.blockSignals(True)

        self.Widget.IsppaScrollBar1.setMaximum(yvec.shape[0] - 1)
        self.Widget.IsppaScrollBar1.setValue(targetIndex[1])
        self.Widget.IsppaScrollBar2.setMaximum(xvec.shape[0] - 1)
        self.Widget.IsppaScrollBar2.setValue(targetIndex[0])

        self.Widget.IsppaScrollBar1.blockSignals(False)
        self.Widget.IsppaScrollBar2.blockSignals(False)

    def update_labels(self,xind,yind):
        self.Widget.SliceLabel1.setText("Y pos = %3.2f mm" %(self.ycoords[yind]))
        self.Widget.SliceLabel2.setText("X pos = %3.2f mm" %(self.xcoords[xind]))

    def get_scroll_values(self):
        selectedY = self.Widget.IsppaScrollBar1.value()
        selectedX = self.Widget.IsppaScrollBar2.value()
        return selectedY, selectedX