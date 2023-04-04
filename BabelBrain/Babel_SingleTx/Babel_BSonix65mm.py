# This Python file uses the following encoding: utf-8

from multiprocessing import Process,Queue
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import (QApplication, QWidget,QGridLayout,
                QHBoxLayout,QVBoxLayout,QLineEdit,QDialog,
                QGridLayout, QSpacerItem, QInputDialog, QFileDialog,
                QErrorMessage, QMessageBox)
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPalette, QTextCursor

import numpy as np

from scipy.io import loadmat
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,NavigationToolbar2QT)
import os
import sys
import shutil
from datetime import datetime
import time
import yaml
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from .CalculateFieldProcess import CalculateFieldProcess

import Babel_SingleTx 

import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_SingleTx'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

def DistanceOutPlaneToFocus(FocalLength,Diameter):
    return np.sqrt(FocalLength**2-(Diameter/2)**2)

class Babel_BSonix65mm(Babel_SingleTx.Babel_SingleTx):
    def __init__(self,parent=None,MainApp=None):
        super(Babel_BSonix65mm, self).__init__(parent,MainApp)

    def load_ui(self):
        super(Babel_BSonix65mm, self).load_ui()
        self.Widget.FocalLengthSpinBox.setValue(self.Config['TxFoc'])
        self.Widget.FocalLengthSpinBox.setValue(self.Config['TxDiam'])
        Diameter = self.Widget.DiameterSpinBox.value()

        
    def DefaultConfig(self):
        with open(os.path.join(resource_path(),'defaultBSonix65mm.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        self.Config=config

    @Slot()
    def RunSimulation(self):
        FocalLength = self.Config['TxFoc']*1e3
        Diameter = self.Config['TxDiam']*1e3
        self._FullSolName=self._MainApp._prefix_path+'BSonix65mm_DataForSim.h5' %(FocalLength,Diameter)
        self._WaterSolName=self._MainApp._prefix_path+'BSonix65mm_Water_DataForSim.h5' %(FocalLength,Diameter)

        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)
        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                    "TxMechanicalAdjustmentZ=%3.2f\n" %(Skull['TxMechanicalAdjustmentZ']*1e3)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
                self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
                self.Widget.ZMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentZ']*1e3)
        else:
            bCalcFields = True
        if bCalcFields:
            self._MainApp.Widget.tabWidget.setEnabled(False)
            self.thread = QThread()
            self.worker = RunAcousticSim(self._MainApp,self.thread)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.UpdateAcResults)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)
 
            self.thread.start()
        else:
            self.UpdateAcResults()

   

class RunAcousticSim(Babel_SingleTx.RunAcousticSim):
    #this is pretty much the same as single tx case, just need to specify the suffix for files
    def __init__(self,mainApp,thread):
        super(RunAcousticSim, self).__init__(mainApp,thread)
        self._extrasuffix='BSonix65mm'
