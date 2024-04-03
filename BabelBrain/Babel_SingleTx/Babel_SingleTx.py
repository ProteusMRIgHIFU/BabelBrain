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
from CalculateFieldProcess import CalculateFieldProcess
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

def DistanceOutPlaneToFocus(FocalLength,Diameter):
    return np.sqrt(FocalLength**2-(Diameter/2)**2)

class SingleTx(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None,formfile='form.ui'):
        super(SingleTx, self).__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self._bIgnoreUpdate=False
        self._ZMaxSkin = 0.0 # maximum
        self.DefaultConfig()
        self.load_ui(formfile)


    def load_ui(self,formfile):
        loader = QUiLoader()
        path = os.path.join(resource_path(), formfile)
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()
        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        self.Widget.CalculateAcField.clicked.connect(self.RunSimulation)
        self.Widget.ZMechanicSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.DiameterSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.FocalLengthSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)
        self.up_load_ui()

    def DefaultConfig(self,cfile='default.yaml'):
        #Specific parameters for the CTX500 - to be configured later via a yaml

        with open(os.path.join(resource_path(),cfile), 'r') as file:
            config = yaml.safe_load(file)
        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._MaskData.header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize

        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)

        self.UpdateLimits()
  
        if self._ZMaxSkin>=0:
            self.Widget.ZMechanicSpinBox.setValue(0.0) # Tx aligned at the target
        else:
            self.Widget.ZMechanicSpinBox.setValue(self._ZMaxSkin) #if negative, we push back the Tx as it can't go below this
        self._UnmodifiedZMechanic = 0.0
        
    
    @Slot()
    def UpdateTxInfo(self):
        if self._bIgnoreUpdate:
            return
        self._bIgnoreUpdate=True
        ZMec=self.Widget.ZMechanicSpinBox.value()
        self.UpdateLimits()
        if ZMec > self.Widget.ZMechanicSpinBox.maximum():
            self.Widget.ZMechanicSpinBox.setValue(self.Widget.ZMechanicSpinBox.maximum())
            ZMec=self.Widget.ZMechanicSpinBox.maximum()
        if ZMec < self.Widget.ZMechanicSpinBox.minimum():
            self.Widget.ZMechanicSpinBox.setValue(self.Widget.ZMechanicSpinBox.minimum())
            ZMec=self.Widget.ZMechanicSpinBox.minimum()
        self.UpdateDistanceLabels()
        self._bIgnoreUpdate=False 

    def UpdateDistanceLabels(self):
        ZMec=self.Widget.ZMechanicSpinBox.value()
        CurDistance=self._ZMaxSkin-ZMec
        self.Widget.DistanceTxToSkinLabel.setText('%3.1f' %(CurDistance))
        if CurDistance<0:
            self.Widget.DistanceTxToSkinLabel.setStyleSheet("color: red")
            self.Widget.LabelTissueRemoved.setVisible(True)
        else:
            self.Widget.DistanceTxToSkinLabel.setStyleSheet("color: blue")
            self.Widget.LabelTissueRemoved.setVisible(False)


    def UpdateLimits(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        DOut=DistanceOutPlaneToFocus(FocalLength,Diameter)
        ZMax=DOut-self.Widget.DistanceSkinLabel.property('UserData')
        self._ZMaxSkin = np.round(ZMax,1)
        self.Widget.ZMechanicSpinBox.setMaximum(self._ZMaxSkin+self.Config['MaxNegativeDistance'])
        self.Widget.ZMechanicSpinBox.setMinimum(self._ZMaxSkin-self.Config['MaxDistanceToSkin'])
        self.UpdateDistanceLabels()

    def GetExtraSuffixAcFields(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        extrasuffix='Foc%03.1f_Diam%03.1f_' %(FocalLength,Diameter)
        return extrasuffix

      
    @Slot()
    def RunSimulation(self):
        extrasuffix=self.GetExtraSuffixAcFields()
        self._FullSolName=self._MainApp._prefix_path+extrasuffix+'DataForSim.h5' 
        self._WaterSolName=self._MainApp._prefix_path+extrasuffix+'Water_DataForSim.h5'
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()

        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "FocalLength=%3.2f\n" %(Skull['FocalLength']*1e3)+
                                    "Diameter=%3.2f\n" %(Skull['Aperture']*1e3)+
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
            self.worker.finished.connect(self.UpdateAcResults)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)
 
            self.thread.start()

            self._MainApp.showClockDialog()
        else:
            self.UpdateAcResults()

   
    def GetExport(self):
        Export=super(SingleTx,self).GetExport()
        for k in ['FocalLength','Diameter','XMechanic','YMechanic','ZMechanic']:
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
        TxMechanicalAdjustmentZ= self._mainApp.AcSim.Widget.ZMechanicSpinBox.value()/1e3  #in m
        ZIntoSkin =0.0
        CurDistance=self._mainApp.AcSim._ZMaxSkin/1e3-TxMechanicalAdjustmentZ
        if CurDistance < 0:
            ZIntoSkin = np.abs(CurDistance)

        Frequencies = [self._mainApp.Widget.USMaskkHzDropDown.property('UserData')]
        basePPW=[self._mainApp.Widget.USPPWSpinBox.property('UserData')]
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
        kargs['bUseCT']=self._mainApp.Config['bUseCT']
        kargs['bUseRayleighForWater']=self._mainApp.Config['bUseRayleighForWater']
        kargs['bPETRA'] = False
        if kargs['bUseCT']:
            if self._mainApp.Config['CTType']==3:
                kargs['bPETRA']=True

        # Start mask generation as separate process.
        bNoError=True
        queue=Queue()
        T0=time.time()
        fieldWorkerProcess = Process(target=CalculateFieldProcess, 
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
            self._mainApp.UpdateComputationalTime('ultrasound',TotalTime)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()

