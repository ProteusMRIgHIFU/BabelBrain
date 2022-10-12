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

#import cv2 as cv
import os
import sys
import shutil
from datetime import datetime
import time
import yaml
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py

from .CalculateFieldProcess import CalculateFieldProcess

class H317(QWidget):
    def __init__(self,parent=None,MainApp=None):
        super(H317, self).__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self.DefaultConfig()
        self.load_ui()


    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.ZSteeringSpinBox.setMinimum(self.Config['MinimalZSteering']*1e3)
        self.Widget.ZSteeringSpinBox.setMaximum(self.Config['MaximalZSteering']*1e3)
        self.Widget.ZSteeringSpinBox.setValue(0.0)

        self.Widget.DistanceConeToFocusSpinBox.setMinimum(self.Config['MinimalDistanceConeToFocus']*1e3)
        self.Widget.DistanceConeToFocusSpinBox.setMaximum(self.Config['MaximalDistanceConeToFocus']*1e3)
        self.Widget.DistanceConeToFocusSpinBox.setValue(self.Config['DefaultDistanceConeToFocus']*1e3)

        self.Widget.ZSteeringSpinBox.valueChanged.connect(self.ZSteeringUpdate)
        self.Widget.RefocusingcheckBox.stateChanged.connect(self.EnableRefocusing)
        self.Widget.CalculatePlanningMask.clicked.connect(self.RunSimulation)

    @Slot()
    def ZSteeringUpdate(self,value):
        self._ZSteering =self.Widget.ZSteeringSpinBox.value()/1e3
        print('ZSteering',self._ZSteering*1e3)

    @Slot()
    def EnableRefocusing(self,value):
        bRefocus =self.Widget.CalculatePlanningMask.isChecked()
        self.Widget.XMechanicSpinBox.setEnabled(bRefocus)
        self.Widget.YMechanicSpinBox.setEnabled(bRefocus)
        self.Widget.ZMechanicSpinBox.setEnabled(bRefocus)

    def DefaultConfig(self):
        #Specific parameters for the H317 - to be configured later via a yaml

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H317 configuration:")
        print(config)

        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._DataMask.header.get_zooms()[0]*1e-3
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize

        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin*1e3))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)

        self.ZSteeringUpdate(0)

    @Slot()
    def RunSimulation(self):
        self._FullSolName=self._MainApp._prefix_path+'DataForSim.h5'
        self._WaterSolName=self._MainApp._prefix_path+'Water_DataForSim.h5'

        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)
        bCalcFields=False
        if os.path.isfile(self._FullSolName) and os.path.isfile(self._WaterSolName):
            Skull=ReadFromH5py(self._FullSolName)
            ZSteering=Skull['ZSteering']
            if 'RotationZ' in Skull:
                RotationZ=Skull['RotationZ']
            else:
                RotationZ=0.0

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "ZSteering=%3.2f\n" %(ZSteering*1e3)+
                                    "ZRotation=%3.2f\n" %(RotationZ)+
                                    "TxMechanicalAdjustmentX=%3.2f\n" %(Skull['TxMechanicalAdjustmentX']*1e3)+
                                    "TxMechanicalAdjustmentY=%3.2f\n" %(Skull['TxMechanicalAdjustmentY']*1e3)+
                                    "TxMechanicalAdjustmentZ=%3.2f\n" %(Skull['TxMechanicalAdjustmentZ']*1e3)+
                                    "Do you want to recalculate?\nSelect No to reload",
                QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
            else:
                self.Widget.ZSteeringSpinBox.setValue(ZSteering*1e3)
                self.Widget.ZRotationSpinBox.setValue(RotationZ)
                self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'])
                if 'DistanceConeToFocus' in Skull:
                    self.Widget.DistanceConeToFocusSpinBox.setValue(Skull['DistanceConeToFocus']*1e3)
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

    def NotifyError(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()

    @Slot()
    def UpdateAcResults(self):
        #this will generate a modified trajectory file
         self._MainApp.Widget.tabWidget.setEnabled(True)
         self._MainApp.ThermalSim.setEnabled(True)
         Water=ReadFromH5py(self._WaterSolName)
         Skull=ReadFromH5py(self._FullSolName)
         if self._MainApp._bInUseWithBrainsight:
            if Skull['bDoRefocusing']:
                #we update the name to be loaded in BSight
                self._MainApp._BrainsightInput=self._MainApp._prefix_path+'FullElasticSolutionRefocus.nii.gz'
            with open(self._MainApp._BrainsightSyncPath+os.sep+'Output.txt','w') as f:
                f.write(self._MainApp._BrainsightInput) 
         self._MainApp.ExportTrajectory(CorX=Skull['AdjustmentInRAS'][0],
                                        CorY=Skull['AdjustmentInRAS'][1],
                                        CorZ=Skull['AdjustmentInRAS'][2])

         LocTarget=Skull['TargetLocation']
         print(LocTarget)

         if Skull['bDoRefocusing']:
             SelP='p_amp_refocus'
         else:
             SelP='p_amp'

         for d in [Skull]:
             for t in [SelP,'MaterialMap']:
                 d[t]=np.ascontiguousarray(np.flip(d[t],axis=2))

         for d in [Water]:
             for t in ['p_amp','MaterialMap']:
                 d[t]=np.ascontiguousarray(np.flip(d[t],axis=2))

         DistanceToTarget=self.Widget.DistanceSkinLabel.property('UserData')*1e3

         Water['z_vec']*=1e3
         Skull['z_vec']*=1e3
         Skull['x_vec']*=1e3
         Skull['y_vec']*=1e3
         Skull['MaterialMap'][Skull['MaterialMap']==3]=2
         Skull['MaterialMap'][Skull['MaterialMap']==4]=3

         DensityMap=Water['Material'][:,0][Water['MaterialMap']]
         SoSMap=    Water['Material'][:,1][Water['MaterialMap']]
         IWater=Water['p_amp']**2/2/DensityMap/SoSMap/1e4

         DensityMap=Skull['Material'][:,0][Skull['MaterialMap']]
         SoSMap=    Skull['Material'][:,1][Skull['MaterialMap']]
         ISkull=Skull[SelP]**2/2/DensityMap/SoSMap/1e4

         IntWaterLocation=IWater[LocTarget[0],LocTarget[1],LocTarget[2]]
         IntSkullLocation=ISkull[LocTarget[0],LocTarget[1],LocTarget[2]]

         ISkull/=IWater[Skull['MaterialMap']==3].max()
         IWater/=IWater[Skull['MaterialMap']==3].max()


         Factor=IWater[Skull['MaterialMap']==3].max()/ISkull[Skull['MaterialMap']==3].max()
         print('*'*40+'\n'+'*'*40+'\n'+'Correction Factor for Isppa',Factor,'\n'+'*'*40+'\n'+'*'*40+'\n')
         print('*'*40+'\n'+'*'*40+'\n'+'Correction Factor for Isppa (location)',IntWaterLocation/IntSkullLocation,'\n'+'*'*40+'\n'+'*'*40+'\n')
         
         ISkull[Skull['MaterialMap']!=3]=0
         self._figAcField=Figure(figsize=(14, 12))

         if self.static_canvas is not None:
             self._layout.removeItem(self._layout.itemAt(0))
             self._layout.removeItem(self._layout.itemAt(0))
         else:
             self._layout = QVBoxLayout(self.Widget.AcField_plot1)

         self.static_canvas = FigureCanvas(self._figAcField)
         self._layout.addWidget(self.static_canvas)
         toolbar=NavigationToolbar2QT(self.static_canvas,self)
         self._layout.addWidget(toolbar)
         self._layout.addWidget(self.static_canvas)
         static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)

         dz=np.diff(Skull['z_vec']).mean()
         Zvec=Skull['z_vec'].copy()
         Zvec-=Zvec[LocTarget[2]]
         Zvec+=DistanceToTarget#+self.Widget.ZSteeringSpinBox.value()
         XX,ZZ=np.meshgrid(Skull['x_vec'],Zvec)
         self._imContourf1=static_ax1.contourf(XX,ZZ,ISkull[:,LocTarget[1],:].T*Factor,np.arange(2,22,2)/20,cmap=plt.cm.jet)
         h=plt.colorbar(self._imContourf1,ax=static_ax1)
         h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
         static_ax1.contour(XX,ZZ,Skull['MaterialMap'][:,LocTarget[1],:].T,[0,1,2,3], cmap=plt.cm.gray)
         static_ax1.set_aspect('equal')
         static_ax1.set_xlabel('X mm')
         static_ax1.set_ylabel('Z mm')
         static_ax1.invert_yaxis()
         static_ax1.plot(0,DistanceToTarget,'+y',markersize=18)

         YY,ZZ=np.meshgrid(Skull['y_vec'],Zvec)

         self._imContourf2=static_ax2.contourf(YY,ZZ,ISkull[LocTarget[0],:,:].T*Factor,np.arange(2,22,2)/20,cmap=plt.cm.jet)
         h=plt.colorbar(self._imContourf1,ax=static_ax2)
         h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
         static_ax2.contour(YY,ZZ,Skull['MaterialMap'][LocTarget[0],:,:].T,[0,1,2,3], cmap=plt.cm.gray)
         static_ax2.set_aspect('equal')
         static_ax2.set_xlabel('Y mm')
         static_ax2.set_ylabel('Z mm')
         static_ax2.invert_yaxis()
         static_ax2.plot(0,DistanceToTarget,'+y',markersize=18)
         self._figAcField.set_facecolor(np.array(self.Widget.palette().color(QPalette.Window).getRgb())/255)
         self._figAcField.set_tight_layout(True)

         #f.set_title('MAIN SIMULATION RESULTS')


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
        basedir,ID=os.path.split(os.path.split(self._mainApp._Mat4Brainsight)[0])
        basedir+=os.sep
        Target=[self._mainApp._ID]

        InputSim=self._mainApp._outnameMask

        bRefocus = self._mainApp.AcSim.Widget.RefocusingcheckBox.isChecked()
        #we can use mechanical adjustments in other directions for final tuning
        if not bRefocus:
            TxMechanicalAdjustmentX= self._mainApp.AcSim.Widget.XMechanicSpinBox.value()/1e3 #in m
            TxMechanicalAdjustmentY= self._mainApp.AcSim.Widget.YMechanicSpinBox.value()/1e3  #in m
            TxMechanicalAdjustmentZ= self._mainApp.AcSim.Widget.ZMechanicSpinBox.value()/1e3  #in m

        else:
            TxMechanicalAdjustmentX=0
            TxMechanicalAdjustmentY=0
            TxMechanicalAdjustmentZ=0
        ###############
        ZSteering=self._mainApp.AcSim.Widget.ZSteeringSpinBox.value()/1e3  #Add here the final adjustment)
        XSteering=1e-6
        ##############
        RotationZ=self._mainApp.AcSim.Widget.ZRotationSpinBox.value()

        print('ZSteering',ZSteering*1e3)
        print('RotationZ',RotationZ)

        Frequencies = [self._mainApp.Widget.USMaskkHzDropDown.property('UserData')]
        basePPW=[self._mainApp.Widget.USPPWSpinBox.property('UserData')]
        T0=time.time()

        DistanceConeToFocus=self._mainApp.AcSim.Widget.DistanceConeToFocusSpinBox.value()/1e3

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
        kargs['ZSteering']=ZSteering
        kargs['RotationZ']=RotationZ
        kargs['Frequencies']=Frequencies
        kargs['bDoRefocusing']=bRefocus
        kargs['DistanceConeToFocus']=DistanceConeToFocus
        kargs['bUseCT']=self._mainApp._bUseCT

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
    widget = H317()
    widget.show()
    sys.exit(app.exec_())
