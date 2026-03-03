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

from _BabelBaseTx import BabelBaseTx

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

class SingleTx(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None,formfile='form.ui'):
        super(SingleTx, self).__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self._bIgnoreUpdate=False
        self._ZMaxSkin = 0.0 # maximum
        self.default_config()
        self.load_ui(formfile)


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
        self.Widget.DiameterSpinBox.valueChanged.connect(self.update_tx_info)
        self.Widget.FocalLengthSpinBox.valueChanged.connect(self.update_tx_info)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.calculate_mech_adj.clicked.connect(self.calculate_mech_adj)
        self.Widget.calculate_mech_adj.setEnabled(False)
        self.up_load_ui()

    def default_config(self,cfile='default.yaml'):
        #Specific parameters for the CTX500 - to be configured later via a yaml

        with open(os.path.join(resource_path(),cfile), 'r') as file:
            config = yaml.safe_load(file)
        self.Config=config

    def notify_generated_mask(self):
        VoxelSize=self._MainApp._MaskData.header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize

        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)

        self.update_limits()

        if self._ZMaxSkin >0:
            self.Widget.SkinDistanceSpinBox.setValue(self._ZMaxSkin) # Tx aligned at the target
        else:
            self.Widget.SkinDistanceSpinBox.setValue(0.0) # Tx aligned at the skin
        self._UnmodifiedZMechanic = 0.0
        
    
    @Slot()
    def update_tx_info(self):
        if self._bIgnoreUpdate:
            return
        self._bIgnoreUpdate=True
        self.update_limits()
        self.update_distance_labels()
        self._bIgnoreUpdate=False 

    def update_distance_labels(self):
        CurDistance=self.Widget.SkinDistanceSpinBox.value()
        if CurDistance<0:
            self.Widget.LabelTissueRemoved.setVisible(True)
        else:
            self.Widget.LabelTissueRemoved.setVisible(False)


    def update_limits(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        DOut=distance_out_plane_to_focus(FocalLength,Diameter)
        self.Widget.DistanceOutplaneLabel.setText('%3.1f' %(DOut))
        ZMax=DOut-self.Widget.DistanceSkinLabel.property('UserData')
        self._ZMaxSkin = np.round(ZMax,1)
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance']) 
        self.update_distance_labels()

    def get_extra_suffix_ac_fields(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        extrasuffix='Foc%03.1f_Diam%03.1f_' %(FocalLength,Diameter)
        return extrasuffix

      
    @Slot()
    def run_simulation(self):
        extrasuffix=self.get_extra_suffix_ac_fields()
        self._FullSolName=self._MainApp._prefix_path+extrasuffix+'DataForSim.h5' 
        self._WaterSolName=self._MainApp._prefix_path+extrasuffix+'Water_DataForSim.h5'
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()

        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)
            
            DistanceSkin = self._ZMaxSkin - Skull['TxMechanicalAdjustmentZ']*1e3

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "FocalLength=%3.2f\n" %(Skull['FocalLength']*1e3)+
                                    "Diameter=%3.2f\n" %(Skull['Aperture']*1e3)+
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
            self.worker.finished.connect(self.update_ac_results)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.notify_error)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)
 
            self.thread.start()

            self._MainApp.show_clock_dialog()
        else:
            self.update_ac_results()

   
    def get_export(self):
        Export=super(SingleTx,self).get_export()
        for k in ['FocalLength','Diameter','XMechanic','YMechanic','SkinDistance']:
            if hasattr(self.Widget,k+'SpinBox'):
                Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

class RunAcousticSim(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp,extrasuffix,Aperture,FocalLength):
        super(RunAcousticSim, self).__init__()
        self._mainApp=mainApp
        self._extrasuffix=extrasuffix
        self._Aperture=Aperture
        self._FocalLength=FocalLength

    def run(self):

        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1WIso'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID']+'_'+self._mainApp.Config['TxSystem']]

        InputSim=self._mainApp._outnameMask

        
        #we can use mechanical adjustments in other directions for final tuning
        
        TxMechanicalAdjustmentX= self._mainApp.AcSim.Widget.XMechanicSpinBox.value()/1e3 #in m
        TxMechanicalAdjustmentY= self._mainApp.AcSim.Widget.YMechanicSpinBox.value()/1e3  #in m
        TxMechanicalAdjustmentZ= (self._mainApp.AcSim._ZMaxSkin - self._mainApp.AcSim.Widget.SkinDistanceSpinBox.value())/1e3  #in m
        ZIntoSkin =0.0
        CurDistance=self._mainApp.AcSim._ZMaxSkin/1e3-TxMechanicalAdjustmentZ
        if CurDistance < 0:
            ZIntoSkin = np.abs(CurDistance)

        Frequencies = [self._mainApp._Frequency]
        basePPW=[self._mainApp._BasePPW]
        T0=time.time()
        kargs={}
        kargs['extrasuffix']=self._extrasuffix
        kargs['ID']=ID
        kargs['deviceName']=deviceName
        kargs['COMPUTING_BACKEND']=COMPUTING_BACKEND
        kargs['basePPW']=basePPW
        kargs['basedir']=basedir
        kargs['Aperture']=self._Aperture
        kargs['FocalLength']=self._FocalLength
        kargs['TxMechanicalAdjustmentZ']=TxMechanicalAdjustmentZ
        kargs['TxMechanicalAdjustmentX']=TxMechanicalAdjustmentX
        kargs['TxMechanicalAdjustmentY']=TxMechanicalAdjustmentY
        kargs['ZIntoSkin']=ZIntoSkin
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs|=self._mainApp.commom_ac_options()

        # Start mask generation as separate process.
        bNoError=True
        queue=Queue()
        T0=time.time()
        fieldWorkerProcess = Process(target=calculate_field_process, 
                                    args=(queue,Target,self._mainApp.Config['TxSystem']),
                                    kwargs=kargs)
        fieldWorkerProcess.start()      
        # progress.
        while fieldWorkerProcess.is_alive():
            time.sleep(0.1)
            while queue.empty() == False:
                cMsg=queue.get()
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg:
                    bNoError=False  
        fieldWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            print(cMsg,end='')
            if '--Babel-Brain-Low-Error' in cMsg:
                bNoError=False
        if bNoError:
            TEnd=time.time()
            TotalTime = TEnd-T0
            print('Total time',TotalTime)
            print("*"*40)
            print("*"*5+" DONE ultrasound simulation.")
            print("*"*40)
            self._mainApp.update_computational_time('ultrasound',TotalTime)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()

