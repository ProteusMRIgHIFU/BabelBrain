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
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

from _BabelBaseTx import BabelBaseTx

import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_H246'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class H246(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None):
        super(H246, self).__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self.DefaultConfig()
        self.load_ui()


    def load_ui(self):
        loader = QUiLoader()
        #path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        path = os.path.join(resource_path(), "form.ui")
        
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        self.Widget.TPODistanceSpinBox.setMinimum(self.Config['MinimalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.setMaximum(self.Config['MaximalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.valueChanged.connect(self.TPODistanceUpdate)
        self.Widget.TPORangeLabel.setText('[%3.1f - %3.1f]' % (self.Config['MinimalTPODistance']*1e3,self.Config['MaximalTPODistance']*1e3))
        self.Widget.CalculatePlanningMask.clicked.connect(self.RunSimulation)
        self.Widget.ShowWaterResultscheckBox.stateChanged.connect(self.UpdateAcResults)

    @Slot()
    def TPODistanceUpdate(self,value):
        self._ZSteering =self.Widget.TPODistanceSpinBox.value()/1e3
        print('ZSteering',self._ZSteering*1e3)

    def DefaultConfig(self):
        #Specific parameters for the H246 - to be configured later via a yaml

        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
        with open(os.path.join(resource_path(),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H246 configuration:")
        print(config)

        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._DataMask.header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize

        self.Widget.TPODistanceSpinBox.setValue(np.round(DistanceFromSkin,1))
        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)

        self.TPODistanceUpdate(0)

    @Slot()
    def RunSimulation(self):
        self._FullSolName=self._MainApp._prefix_path+'DataForSim.h5'
        self._WaterSolName=self._MainApp._prefix_path+'Water_DataForSim.h5'

        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)
        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)
            TPO=Skull['ZSteering']

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "ZSteering=%3.2f\n" %(TPO*1e3)+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.TPODistanceSpinBox.setValue(TPO*1e3)
                self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
                self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
        else:
            bCalcFields = True
        self._bRecalculated = True
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

    def NotifyError(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()
   
    def GetExport(self):
        Export={}
        for k in ['TPODistance','XMechanic','YMechanic']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

class RunAcousticSim(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp,thread):
        super(RunAcousticSim, self).__init__()
        self._mainApp=mainApp
        self._thread=thread

    def run(self):

        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1W'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID']+'_'+self._mainApp.Config['TxSystem']]

        InputSim=self._mainApp._outnameMask

        #we can use mechanical adjustments in other directions for final tuning
        TxMechanicalAdjustmentX= self._mainApp.AcSim.Widget.XMechanicSpinBox.value()/1e3 #in m
        TxMechanicalAdjustmentY= self._mainApp.AcSim.Widget.YMechanicSpinBox.value()/1e3  #in m
        TxMechanicalAdjustmentZ= self._mainApp.AcSim.Widget.ZMechanicSpinBox.value()/1e3  #in m

        ###############
        TPODistance=self._mainApp.AcSim.Widget.TPODistanceSpinBox.value()/1e3  #Add here the final adjustment)
        ##############

        print('Ideal Distance to program in TPO : ', TPODistance*1e3)


        ZSteering=TPODistance
        print('ZSteering',ZSteering*1e3)

        Frequencies = [self._mainApp.Widget.USMaskkHzDropDown.property('UserData')]
        basePPW=[self._mainApp.Widget.USPPWSpinBox.property('UserData')]
        T0=time.time()
        kargs={}
        kargs['ID']=ID
        kargs['deviceName']=deviceName
        kargs['COMPUTING_BACKEND']=COMPUTING_BACKEND
        kargs['basePPW']=basePPW
        kargs['basedir']=basedir
        kargs['TxMechanicalAdjustmentZ']=TxMechanicalAdjustmentZ
        kargs['TxMechanicalAdjustmentX']=TxMechanicalAdjustmentX
        kargs['TxMechanicalAdjustmentY']=TxMechanicalAdjustmentY
        kargs['ZSteering']=ZSteering
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['bUseCT']=self._mainApp.Config['bUseCT']
        
        # Start mask generation as separate process.
        queue=Queue()
        fieldWorkerProcess = Process(target=CalculateFieldProcess, 
                                    args=(queue,Target),
                                    kwargs=kargs)
        fieldWorkerProcess.start()      
        # progress.
        T0=time.time()
        bNoError=True
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
            print('Total time',TEnd-T0)
            print("*"*40)
            print("*"*5+" DONE ultrasound simulation.")
            print("*"*40)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()




if __name__ == "__main__":
    app = QApplication([])
    widget = H246()
    widget.show()
    sys.exit(app.exec_())
