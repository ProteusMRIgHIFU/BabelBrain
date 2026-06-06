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
        # Programmatic form replaces scrollbars.ui.
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QScrollBar, QSizePolicy
        from GUIComponents.TxPanelBase import make_label, LABEL_BLUE

        # Stretch to fill the IsppaScrollBars host so we match the figure width.
        # The host (passed in as our parent) has no layout of its own, so
        # install one here that pins this widget to its full width.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        host = self.parent()
        if host is not None and host.layout() is None:
            host_layout = QVBoxLayout(host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            host_layout.addWidget(self)

        self.Widget = QWidget(self)

        # Two columns, each a horizontal scrollbar with a position label below.
        grid = QGridLayout(self.Widget)
        grid.setContentsMargins(20, 2, 20, 2)
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(2)

        self.Widget.IsppaScrollBar1 = QScrollBar(Qt.Horizontal)
        self.Widget.IsppaScrollBar1.setObjectName("IsppaScrollBar1")
        self.Widget.IsppaScrollBar1.setEnabled(False)

        self.Widget.IsppaScrollBar2 = QScrollBar(Qt.Horizontal)
        self.Widget.IsppaScrollBar2.setObjectName("IsppaScrollBar2")
        self.Widget.IsppaScrollBar2.setEnabled(False)

        self.Widget.SliceLabel1 = make_label(
            "-", name="SliceLabel1", bold=True, color=LABEL_BLUE,
            align=Qt.AlignHCenter | Qt.AlignVCenter)
        self.Widget.SliceLabel2 = make_label(
            "-", name="SliceLabel2", bold=True, color=LABEL_BLUE,
            align=Qt.AlignHCenter | Qt.AlignVCenter)
        for lbl in (self.Widget.SliceLabel1, self.Widget.SliceLabel2):
            f = lbl.font()
            f.setPointSize(11)
            lbl.setFont(f)

        grid.addWidget(self.Widget.IsppaScrollBar1, 0, 0)
        grid.addWidget(self.Widget.IsppaScrollBar2, 0, 1)
        grid.addWidget(self.Widget.SliceLabel1, 1, 0)
        grid.addWidget(self.Widget.SliceLabel2, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        _l = QVBoxLayout(self)
        _l.setContentsMargins(0, 0, 0, 0)
        _l.addWidget(self.Widget)

        self.Widget.IsppaScrollBar1.valueChanged.connect(self._MainApp._showMatplotlibVisualization)
        self.Widget.IsppaScrollBar2.valueChanged.connect(self._MainApp._showMatplotlibVisualization)

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