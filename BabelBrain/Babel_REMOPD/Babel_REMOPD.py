# This Python file uses the following encoding: utf-8
from multiprocessing import Process,Queue
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import (QApplication, QWidget,QGridLayout,
                QHBoxLayout,QVBoxLayout,QLineEdit,QDialog,
                QGridLayout, QSpacerItem, QInputDialog, QFileDialog,
                QErrorMessage, QMessageBox)
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread,Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPalette, QTextCursor

import numpy as np

from scipy.io import loadmat
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,NavigationToolbar2QT)

#import cv2 as cv
import os
import sys
import platform
from datetime import datetime
import time
import yaml
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

from CalculateFieldProcess import CalculateFieldProcess

from _BabelBasePhasedArray import BabelBasePhaseArray

_IS_MAC = platform.system() == 'Darwin'
def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_REMOPD'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class REMOPD(BabelBasePhaseArray): 
    def __init__(self,parent=None,MainApp=None):
        super().__init__(parent=parent,MainApp=MainApp,formfile=os.path.join(resource_path(), "form.ui"))

    def load_ui(self,formfile):
        loader = QUiLoader()
        path = os.fspath(formfile)
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)

        self.Widget.XSteeringSpinBox.setMinimum(self.Config['MinimalXSteering']*1e3)
        self.Widget.XSteeringSpinBox.setMaximum(self.Config['MaximalXSteering']*1e3)
        self.Widget.YSteeringSpinBox.setMinimum(self.Config['MinimalYSteering']*1e3)
        self.Widget.YSteeringSpinBox.setMaximum(self.Config['MaximalYSteering']*1e3)
        self.Widget.ZSteeringSpinBox.setMinimum(self.Config['MinimalZSteering']*1e3)
        self.Widget.ZSteeringSpinBox.setMaximum(self.Config['MaximalZSteering']*1e3)
        self.Widget.ZSteeringSpinBox.setValue(self.Config['DefaultZSteering']*1e3)
        
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance'])  
        self.Widget.SkinDistanceSpinBox.setValue(0.0)

        self.Widget.RefocusingcheckBox.stateChanged.connect(self.EnableRefocusing)
        self.Widget.CalculateAcField.clicked.connect(self.RunSimulation)
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.UpdateDistanceFromSkin)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)
        self.up_load_ui()
        
    @Slot()
    def UpdateDistanceFromSkin(self):
        self._bIgnoreUpdate=True
        CurDistance=self.Widget.SkinDistanceSpinBox.value()
        if CurDistance<0:
            self.Widget.LabelTissueRemoved.setVisible(True)
        else:
            self.Widget.LabelTissueRemoved.setVisible(False)

    def DefaultConfig(self):
        #Specific parameters for the REMOPD - to be configured later via a yaml

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("REMOPD configuration:")
        print(config)

        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._MaskData.header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize
        
        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)


    @Slot()
    def RunSimulation(self):
        #we create an object to do a dryrun to recover filenames
        dry=RunAcousticSim(self._MainApp,bDryRun=True)
        FILENAMES = dry.run()
        
        self._FullSolName=FILENAMES['FilesSkull']
        self._WaterSolName=FILENAMES['FilesWater']

        bCalcFields=False
        bPrexistingFiles=True
        for sskull,swater in zip(self._FullSolName,self._WaterSolName):
            if not(os.path.isfile(sskull) and os.path.isfile(swater)):
                bPrexistingFiles=False
                break
            
        if bPrexistingFiles:
            #we can use the first entry, this is valid for all files in the list
            Skull=ReadFromH5py(self._FullSolName[0])
            XSteering=Skull['XSteering']
            YSteering=Skull['YSteering']
            ZSteering=Skull['ZSteering']
            if 'RotationZ' in Skull:
                RotationZ=Skull['RotationZ']
            else:
                RotationZ=0.0
                
            DistanceSkin =  -Skull['TxMechanicalAdjustmentZ']*1e3

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "XSteering=%3.2f\n" %(XSteering*1e3)+
                                    "YSteering=%3.2f\n" %(YSteering*1e3)+
                                    "ZSteering=%3.2f\n" %(ZSteering*1e3)+
                                    "ZRotation=%3.2f\n" %(RotationZ)+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                    "DistanceSkin=%3.2f\n" %(DistanceSkin)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.XSteeringSpinBox.setValue(XSteering*1e3)
                self.Widget.YSteeringSpinBox.setValue(YSteering*1e3)
                self.Widget.ZSteeringSpinBox.setValue(ZSteering*1e3)
                self.Widget.ZRotationSpinBox.setValue(RotationZ)
                self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'])
                self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
                TxSet = Skull['TxSet']
                if type(TxSet) is bytes:
                    TxSet=TxSet.decode("utf-8")
                index = self.Widget.SelTxSetDropDown.findText(TxSet, Qt.MatchFixedString)
                if index >= 0:
                    self.Widget.SelTxSetDropDown.setCurrentIndex(index)
                self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
                self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
                self.Widget.SkinDistanceSpinBox.setValue(DistanceSkin)
        else:
            bCalcFields = True
        self._bRecalculated = True
        if bCalcFields:
            self._MainApp.Widget.tabWidget.setEnabled(False)
            self.thread = QThread()
            self.worker = RunAcousticSim(self._MainApp)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.EndSimulation)
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
        Export=super(REMOPD,self).GetExport()
        Export['Refocusing']=self.Widget.RefocusingcheckBox.isChecked()
        def dict_to_string(d, separator=', ', equals_sign='='):
            return separator.join(f'{key}:{value*1000.0}' for key, value in d.items())
        # if self._MultiPoint is not None:
        #     st =''
        #     for e in self._MultiPoint:
        #         st+='[%s] ' % dict_to_string(e)
        #     Export['MultiPoint']=st
        # else:
        #     self._MultiPoint ='N/A'
         
        for k in ['XSteering','YSteering','ZSteering','ZRotation','XMechanic','YMechanic','ZMechanic']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export
    
    def EnableMultiPoint(self,MultiPoint):
        pass #we disable multipoint for the time being

class RunAcousticSim(QObject):

    finished = Signal(object)
    endError = Signal()

    def __init__(self,mainApp,bDryRun=False):
        super(RunAcousticSim, self).__init__()
        self._mainApp=mainApp
        self._bDryRun=bDryRun

    def run(self):
        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1WIso'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID']+'_'+self._mainApp.Config['TxSystem']]

        InputSim=self._mainApp._outnameMask

        bRefocus = self._mainApp.AcSim.Widget.RefocusingcheckBox.isChecked()
        #we can use mechanical adjustments in other directions for final tuning
        if not bRefocus:
            TxMechanicalAdjustmentX= self._mainApp.AcSim.Widget.XMechanicSpinBox.value()/1e3 #in m
            TxMechanicalAdjustmentY= self._mainApp.AcSim.Widget.YMechanicSpinBox.value()/1e3  #in m
            TxMechanicalAdjustmentZ= -self._mainApp.AcSim.Widget.SkinDistanceSpinBox.value()/1e3  #in m

        else:
            TxMechanicalAdjustmentX=0
            TxMechanicalAdjustmentY=0
            TxMechanicalAdjustmentZ=0
        ###############
        XSteering=self._mainApp.AcSim.Widget.XSteeringSpinBox.value()/1e3 
        YSteering=self._mainApp.AcSim.Widget.YSteeringSpinBox.value()/1e3  
        ZSteering=self._mainApp.AcSim.Widget.ZSteeringSpinBox.value()/1e3  
        ##############
        RotationZ=self._mainApp.AcSim.Widget.ZRotationSpinBox.value()
        TxSet = self._mainApp.AcSim.Widget.SelTxSetDropDown.currentText()

        Frequencies = [self._mainApp.Widget.USMaskkHzDropDown.property('UserData')]

        basePPW=[self._mainApp.Widget.USPPWSpinBox.property('UserData')]
        ZIntoSkin =0.0
        if TxMechanicalAdjustmentZ > 0:
            ZIntoSkin = np.abs(TxMechanicalAdjustmentZ)
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
        kargs['XSteering']=XSteering
        kargs['YSteering']=YSteering
        kargs['ZSteering']=ZSteering
        kargs['RotationZ']=RotationZ
        kargs['RotationZ']=RotationZ
        kargs['TxSet']=TxSet
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['bDoRefocusing']=bRefocus
        kargs['bUseCT']=self._mainApp.Config['bUseCT']
        kargs['CTMapCombo']=self._mainApp.Config['CTMapCombo']
        kargs['bUseRayleighForWater']=self._mainApp.Config['bUseRayleighForWater']
        kargs['bPETRA'] = False
        kargs['bDryRun'] = self._bDryRun
        kargs['ZIntoSkin'] = ZIntoSkin
            
        if kargs['bUseCT']:
            if self._mainApp.Config['CTType']==3:
                kargs['bPETRA']=True

        
        queue=Queue()
        if self._bDryRun == False:
            #in real run, we run this in background
            # Start mask generation as separate process.
            fieldWorkerProcess = Process(target=CalculateFieldProcess, 
                                        args=(queue,Target,self._mainApp.Config['TxSystem']),
                                        kwargs=kargs)
            fieldWorkerProcess.start()      
                
            # progress.
            T0=time.time()
            bNoError=True
            OutFiles=None
            while fieldWorkerProcess.is_alive():
                time.sleep(0.1)
                while queue.empty() == False:
                    cMsg=queue.get()
                    if type(cMsg) is str:
                        print(cMsg,end='')
                        if '--Babel-Brain-Low-Error' in cMsg:
                            bNoError=False
                    else:
                        assert(type(cMsg) is dict)
                        OutFiles=cMsg
            fieldWorkerProcess.join()
            while queue.empty() == False:
                cMsg=queue.get()
                if type(cMsg) is str:
                    print(cMsg,end='')
                    if '--Babel-Brain-Low-Error' in cMsg:
                        bNoError=False
                else:
                    assert(type(cMsg) is dict)
                    OutFiles=cMsg
            if bNoError:
                TEnd=time.time()
                TotalTime = TEnd-T0
                print('Total time',TotalTime)
                print("*"*40)
                print("*"*5+" DONE ultrasound simulation.")
                print("*"*40)
                self._mainApp.UpdateComputationalTime('ultrasound',TotalTime)
                self.finished.emit(OutFiles)
            else:
                print("*"*40)
                print("*"*5+" Error in execution.")
                print("*"*40)
                self.endError.emit()
        else:
            #in dry run, we just recover the filenames
            return CalculateFieldProcess(queue,Target,self._mainApp.Config['TxSystem'],**kargs)


if __name__ == "__main__":
    app = QApplication([])
    widget = REMOPD()
    widget.show()
    sys.exit(app.exec_())
