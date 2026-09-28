# This Python file uses the following encoding: utf-8

import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

from babel_transducers.transducer_templates.babel_base_tx import BabelBaseTx
from CalculateFieldProcess import CalculateFieldProcess
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

class FlatAnnularArrayTx(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None,tx_config_file=None,step_2_form=None):
        super().__init__(parent,MainApp,tx_config_file,step_2_form)
        self.static_canvas=None

    def _WirePanel(self):
        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)
        self.Widget.TPODistanceSpinBox.setMinimum(self.Config['MinimalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.setMaximum(self.Config['MaximalTPODistance']*1e3)
        self.Widget.TPODistanceSpinBox.valueChanged.connect(self.TPODistanceUpdate)
        self.Widget.TPORangeLabel.setText('[%3.1f - %3.1f]' % (self.Config['MinimalTPODistance']*1e3,self.Config['MaximalTPODistance']*1e3))
        self.Widget.SkinDistanceSpinBox.setMaximum(self.Config['MaxDistanceToSkin'])
        self.Widget.SkinDistanceSpinBox.setMinimum(-self.Config['MaxNegativeDistance'])
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.UpdateDistanceFromSkin)
        
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)
        
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

    def NotifyGeneratedMask(self):
        self._SyncActiveTrajectoryFromMainApp()
        DistanceFromSkin = self.CalculateDistanceFromSkin()
        self.Widget.TPODistanceSpinBox.setValue(np.round(DistanceFromSkin,1))
        self.TPODistanceUpdate(0)
    

    @Slot()
    def _ResolveSimulationFilenames(self):
        self._FullSolName=self._MainApp._prefix_path[self._TrajectoryNumber]+'DataForSim.h5'
        self._WaterSolName=self._MainApp._prefix_path[self._TrajectoryNumber]+'Water_DataForSim.h5'
        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)

    def _PromptReuseOrRecalc(self):
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
            return True
        self.Widget.TPODistanceSpinBox.setValue(TPO*1e3)
        self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
        self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
        self.Widget.SkinDistanceSpinBox.setValue(DistanceSkin)
        if 'zLengthBeyonFocalPoint' in Skull:
            self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
        return False

    def _CreateAcousticWorker(self):
        return RunAcousticSim(self._MainApp)


    def GetExport(self):
        Export = super().GetExport()
        for k in ['TPODistance','XMechanic','YMechanic','SkinDistance']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

class RunAcousticSim(QObject):

    finished = Signal()
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self,mainApp):
        super().__init__()
        self._mainApp=mainApp

    def run(self):

        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1WIso'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID'][self._mainApp.AcSim._TrajectoryNumber]+'_'+self._mainApp.Config['TxSystem']]

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
        kargs['is_custom_tx'] = self._mainApp.Config['is_custom_tx']
        # if self._mainApp.Config['is_custom_tx']:
        #     kargs['geometry_type'] = self._mainApp.AcSim.Config['geometry_type']
        # else:
        #     kargs['geometry_type'] = self._mainApp.Config['TxType']
        kargs['geometry_type'] = self._mainApp.Config['TxType']
        kargs['coordinate_system'] = self._mainApp.AcSim.Config.get('coordinate_system','cartesian')
        kargs['Aperture'] = self._mainApp.AcSim.Config['TxDiam']
        kargs['COMPUTING_BACKEND']=COMPUTING_BACKEND
        kargs['basePPW']=basePPW
        kargs['basedir']=basedir
        kargs['TxMechanicalAdjustmentZ']=TxMechanicalAdjustmentZ
        kargs['TxMechanicalAdjustmentX']=TxMechanicalAdjustmentX
        kargs['TxMechanicalAdjustmentY']=TxMechanicalAdjustmentY
        kargs['ZSteering']=ZSteering
        kargs['Frequencies']=Frequencies
        kargs['InDiameters']=np.array(self._mainApp.AcSim.Config['InDiameters'])
        kargs['OutDiameters']=np.array(self._mainApp.AcSim.Config['OutDiameters'])
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
    widget = FlatAnnularArrayTx()
    widget.show()
    sys.exit(app.exec_())
