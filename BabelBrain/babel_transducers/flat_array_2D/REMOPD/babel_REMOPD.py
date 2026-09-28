# This Python file uses the following encoding: utf-8
import os
import sys
import time
from multiprocessing import Process,Queue

import numpy as np
from PySide6.QtWidgets import QApplication

from babel_transducers.flat_array_2D.REMOPD.REMOPD_form import REMOPDForm
from babel_transducers.transducer_templates import babel_flat_array_2D_tx
from CalculateFieldProcess import CalculateFieldProcess
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars
from Utils.paths import resource_path


class REMOPD(babel_flat_array_2D_tx.FlatArray2DTx): 
    def __init__(self, parent=None, MainApp=None):
        config_file = os.path.join(resource_path(__file__), "default.yaml")
        super().__init__(parent,MainApp,config_file,REMOPDForm)
    
    @property
    def FlipSteeringY(self):
        yflip = self._MainApp.Config.get('TrajectoryType') == 'brainsight'
        return yflip

    def _WirePanel(self):
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
        
        self.Widget.SkinDistanceSpinBox.valueChanged.connect(self.UpdateDistanceFromSkin)
        self.Widget.LabelTissueRemoved.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)
        if self._MainApp.Config['TrajectoryType']!='localite':
            self.Widget.ApplyFeasibleTraj.clicked.connect(self.ApplyFeasibleTrajectory)
        else:
            self.Widget.ApplyFeasibleTraj.setVisible(False)

    def _CreateAcousticWorker(self):
        return RunAcousticSim(self._MainApp)

class RunAcousticSim(babel_flat_array_2D_tx.RunAcousticSim):

    def __init__(self,mainApp,bDryRun=False):
        super().__init__(mainApp,bDryRun)

    def run(self):
        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1WIso'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID'][self._mainApp.AcSim._TrajectoryNumber]+'_'+self._mainApp.Config['TxSystem']]

        InputSim = self._mainApp._outnameMask

        bRefocus = self._mainApp.AcSim.Widget.RefocusingcheckBox.isChecked()
        #we can use mechanical adjustments in other directions for final tuning
        TxMechanicalAdjustmentZ= -self._mainApp.AcSim.Widget.SkinDistanceSpinBox.value()/1e3  #in m
        if not bRefocus:
            TxMechanicalAdjustmentX= self._mainApp.AcSim.Widget.XMechanicSpinBox.value()/1e3 #in m
            TxMechanicalAdjustmentY= self._mainApp.AcSim.Widget.YMechanicSpinBox.value()/1e3  #in m
        else:
            TxMechanicalAdjustmentX=0
            TxMechanicalAdjustmentY=0
        ###############
        XSteering=self._mainApp.AcSim.Widget.XSteeringSpinBox.value()/1e3 
        YSteering=self._mainApp.AcSim.Widget.YSteeringSpinBox.value()/1e3  
        ZSteering=self._mainApp.AcSim.Widget.ZSteeringSpinBox.value()/1e3  
        ##############
        RotationZ=self._mainApp.AcSim.Widget.ZRotationSpinBox.value()
        TxSet = self._mainApp.AcSim.Widget.SelTxSetDropDown.currentText()

        Frequencies = [self._mainApp._Frequency]

        basePPW=[self._mainApp._BasePPW]
        ZIntoSkin =0.0
        if TxMechanicalAdjustmentZ > 0:
            ZIntoSkin = np.abs(TxMechanicalAdjustmentZ)
        T0=time.time()

        kargs={}
        kargs['ID']=ID
        kargs['deviceName']=deviceName
        kargs["is_custom_tx"] = self._mainApp.Config["is_custom_tx"]
        # if self._mainApp.Config['is_custom_tx']:
        #     kargs['geometry_type'] = self._mainApp.AcSim.Config['geometry_type']
        # else:
        #     kargs['geometry_type'] = self._mainApp.Config['TxType']
        kargs['geometry_type'] = self._mainApp.Config['TxType']
        kargs["elements"] = self._mainApp.AcSim.Config["elements"]
        kargs["num_elements"] = self._mainApp.AcSim.Config["num_elements"]
        kargs["element_size"] = self._mainApp.AcSim.Config["element_size"]
        kargs["distance_outplane"] = self._mainApp.AcSim.Config[
            "NaturalOutPlaneDistance"
        ]
        kargs["Aperture"] = self._mainApp.AcSim.Config["TxDiam"]
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
        kargs['TxSet']=TxSet
        # GUI Y stays as typed; flip Y in the solver for Brainsight trajectories.
        # kargs['bFlipSteeringY']= self._mainApp.AcSim.FlipSteeringY
        # if kargs['bFlipSteeringY']:
        #     print('Flipping Y Steering for brainsight operation for REMOPD')
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['bDoRefocusing']=bRefocus
        kargs['bDryRun'] = self._bDryRun
        kargs['ZIntoSkin'] = ZIntoSkin
        kargs|=self._mainApp.CommomAcOptions()

        
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
                        if 'CTS:' in cMsg:
                            self.logTelemetry.emit(cMsg)
                        if '--Babel-Brain-Low-Error' in cMsg:
                            self.logTelemetry.emit("CTS:L1:S2: "+cMsg)
                            bNoError=False
                    else:
                        assert(type(cMsg) is dict)
                        OutFiles=cMsg
            fieldWorkerProcess.join()
            while queue.empty() == False:
                cMsg=queue.get()
                if type(cMsg) is str:
                    print(cMsg,end='')
                    if 'CTS:' in cMsg:
                        self.logTelemetry.emit(cMsg)
                    if '--Babel-Brain-Low-Error' in cMsg:
                        self.logTelemetry.emit("CTS:L1:S2: "+cMsg)
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
                self.logTelemetry.emit("CTS:L2:S2: TOTAL TIME " + str(TotalTime))
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
