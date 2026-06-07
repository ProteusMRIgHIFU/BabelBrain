# This Python file uses the following encoding: utf-8

from multiprocessing import Process,Queue
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QVBoxLayout
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread
from PySide6.QtUiTools import QUiLoader

import numpy as np

import os
import sys
import time
import yaml
from BabelViscoFDTD.H5pySimple import ReadFromH5py
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
        from Babel_H246.H246Form import H246Form
        self.Widget = H246Form(self)

        _l = QVBoxLayout(self)
        _l.setContentsMargins(0, 0, 0, 0)
        _l.addWidget(self.Widget)

        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        self.Widget.TPODistanceSpinBox.setMinimum(self.Config['MinimalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.setMaximum(self.Config['MaximalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.valueChanged.connect(self.TPODistanceUpdate)
        self.Widget.TPORangeLabel.setText('[%3.1f - %3.1f]' % (self.Config['MinimalTPODistance']*1e3,self.Config['MaximalTPODistance']*1e3))
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance'])  
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.UpdateDistanceFromSkin)
        self.Widget.CalculateAcField.clicked.connect(self.RunSimulation)
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

    @Slot()
    def TPODistanceUpdate(self,value):
        self._ZSteering =self.Widget.TPODistanceSpinBox.value()/1e3

    def DefaultConfig(self):
        #Specific parameters for the H246 - to be configured later via a yaml

        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
        with open(os.path.join(resource_path(),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H246 configuration:")
        print(config)

        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._MaskNib.header.get_zooms()[0]
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
            
            DistanceSkin =  -Skull['TxMechanicalAdjustmentZ']*1e3

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "Z distance=%3.2f\n" %(TPO*1e3)+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                    "DistanceSkin=%3.2f\n" %(DistanceSkin)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.TPODistanceSpinBox.setValue(TPO*1e3)
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
            self.worker = RunAcousticSim(self._MainApp)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.UpdateAcResults)
            self.worker.finished.connect(self._MainApp.SendTelemetry)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self._MainApp.SendTelemetry)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)
 
            self.worker.logTelemetry.connect(self._MainApp.LogTelemetry)

            self.thread.start()
            self._MainApp.showClockDialog()
        else:
            self.UpdateAcResults()

   
    def GetExport(self):
        Export=super(H246,self).GetExport()
        for k in ['TPODistance','XMechanic','YMechanic','SkinDistance']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

class RunAcousticSim(QObject):

    finished = Signal()
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self,mainApp):
        super(RunAcousticSim, self).__init__()
        self._mainApp=mainApp

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
        TxMechanicalAdjustmentZ= -self._mainApp.AcSim.Widget.SkinDistanceSpinBox.value()/1e3  #in m

        ###############
        TPODistance=self._mainApp.AcSim.Widget.TPODistanceSpinBox.value()/1e3  #Add here the final adjustment)
        ##############

        print('Ideal Distance to program in TPO : ', TPODistance*1e3)


        ZSteering=TPODistance
        print('ZSteering',ZSteering*1e3)

        Frequencies = [self._mainApp._Frequency]
        basePPW=[self._mainApp._BasePPW]
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
        kargs['ZSteering']=ZSteering
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['ZIntoSkin'] = ZIntoSkin
        kargs|=self._mainApp.CommomAcOptions()
        # Start mask generation as separate process.
        queue=Queue()
        fieldWorkerProcess = Process(target=CalculateFieldProcess, 
                                    args=(queue,Target,self._mainApp.Config['TxSystem']),
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
                if 'CTS:' in cMsg:
                    self.logTelemetry.emit(cMsg)
                if '--Babel-Brain-Low-Error' in cMsg:
                    self.logTelemetry.emit("CTS:L1:S2: "+cMsg)
                    bNoError=False 
        fieldWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            print(cMsg,end='')
            if 'CTS:' in cMsg:
                self.logTelemetry.emit(cMsg)
            if '--Babel-Brain-Low-Error' in cMsg:
                self.logTelemetry.emit("CTS:L1:S2: "+cMsg)
                bNoError=False 
        if bNoError:
            TEnd=time.time()
            TotalTime = TEnd-T0
            print('Total time',TotalTime)
            print("*"*40)
            print("*"*5+" DONE ultrasound simulation.")
            print("*"*40)
            self.logTelemetry.emit("CTS:L2:S2: TOTAL TIME " + str(TotalTime))
            self._mainApp.UpdateComputationalTime('ultrasound',TotalTime)
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
