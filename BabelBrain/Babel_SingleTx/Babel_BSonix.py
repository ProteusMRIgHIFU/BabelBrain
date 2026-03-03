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
from PySide6.QtGui import QPalette, QTextCursor, QColor

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
from CalculateFieldProcess import calculate_field_process
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

from trimesh import creation

from .Babel_SingleTx import SingleTx,RunAcousticSim

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

def distance_out_plane_to_focus(FocalLength,Diameter):
    return np.sqrt(FocalLength**2-(Diameter/2)**2)

class BSonix(SingleTx):
    def __init__(self,parent=None,MainApp=None,formfile='formBx.ui'):
        super(BSonix, self).__init__(parent,MainApp,formfile)       

    def load_ui(self,formfile):
        loader = QUiLoader()
        path = os.path.join(resource_path(), formfile)
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        self.Widget.CalculateAcField.clicked.connect(self.run_simulation)
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.update_tx_info)
        self.Widget.TransducerModelcomboBox.currentIndexChanged.connect(self.update_tx_info)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.calculate_mech_adj.clicked.connect(self.calculate_mech_adj)
        self.Widget.calculate_mech_adj.setEnabled(False)
        self.up_load_ui()
        
        
    def default_config(self,cfile='defaultBSonix.yaml'):
        super(BSonix,self).default_config(cfile)

    def notify_generated_mask(self):
        super(BSonix, self).notify_generated_mask()
        self.Widget.SkinDistanceSpinBox.setValue(0.0)

    def get_tx_model(self):
        return "BSonix"+self.Widget.TransducerModelcomboBox.currentText()

    def update_limits(self):
        model=self.get_tx_model()
        FocalLength = self.Config[model]['FocalLength']*1e3
        Diameter = self.Config[model]['TxDiam']*1e3
        DOut=distance_out_plane_to_focus(FocalLength,Diameter)-self.Config[model]['AdjustDistanceSkin']*1e3
        ZMax=DOut-self.Widget.DistanceSkinLabel.property('UserData')
        self._ZMaxSkin = np.round(ZMax,1)
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance']) 
        self.update_distance_labels()

    def get_extra_suffix_ac_fields(self):
        #By default, it returns empty string, useful when dealing with user-specified geometry
        model=self.get_tx_model()
        return model+'_'
    
    def get_export(self):
        Export=super(BSonix,self).get_export()
        Export['TxModel']=self.get_tx_model()
        return Export


    @Slot()
    def run_simulation(self):
        extrasuffix=self.get_extra_suffix_ac_fields()
        model=self.get_tx_model()
        FocalLength = self.Config[model]['FocalLength']*1e3
        Diameter = self.Config[model]['TxDiam']*1e3
        self._prefix=self._MainApp._prefix_path+model
        self._FullSolName=self._prefix+'_DataForSim.h5' 
        self._WaterSolName=self._prefix+'_Water_DataForSim.h5' 
        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)
        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)
            
            DistanceSkin = self._ZMaxSkin - Skull['TxMechanicalAdjustmentZ']*1e3

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                     "DistanceSkin=%3.2f\n" %(DistanceSkin)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
                self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
                self.Widget.SkinDistanceSpinBox.setValue(DistanceSkin)
                if 'zLengthBeyonFocalPoint' in Skull:
                    self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
        else:
            bCalcFields = True
        self._bRecalculated = True
        if bCalcFields:
            self._MainApp.Widget.tabWidget.setEnabled(False)
            self.thread = QThread()
            self.worker = RunAcousticSim(self._MainApp,
                                        extrasuffix,Diameter/1e3,FocalLength/1e3)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.done_ac_sim)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.notify_error)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)
 
            self.thread.start()
            self._MainApp.show_clock_dialog()
        else:
            self.done_ac_sim()
    
    def done_ac_sim(self):
        RADIUS = self.Config['CaseDiameter']/2/1e3 # m dimension of "puck"
        HEIGHT = self.Config['CaseHeight']/1e3 # m
        InitTrans=np.eye(4)
        LocSpot=np.array(np.where(np.flip(self._MainApp._FinalMask,axis=2)==5.0)).flatten()
        spatial_step=np.mean(self._MainApp._MaskData.header.get_zooms())/1e3
        InitTrans[0,3]=LocSpot[0]
        InitTrans[1,3]=LocSpot[1]
        InitTrans[2,3]=LocSpot[2]+HEIGHT/2/spatial_step+(self.Widget.DistanceSkinLabel.property('UserData')+self.Widget.SkinDistanceSpinBox.value())/1e3/spatial_step
        #we create first a cylinder in voxel dimensions
        cylinder = creation.cylinder(radius=RADIUS/spatial_step,height=HEIGHT/spatial_step,sections=20,transform=InitTrans)
        
        affine=self._MainApp._MaskData.affine.copy()
        
        #and we apply the conversion from voxel space to subject space in mm
        cylinder.apply_transform(affine)

        cylinder.export(self._prefix+'_BSonixCase.stl')
        self.update_ac_results()
        


