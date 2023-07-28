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
from .CalculateFieldProcess import CalculateFieldProcess
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

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

class SingleTx(QWidget):
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
        self.Widget.CalculatePlanningMask.clicked.connect(self.RunSimulation)
        self.Widget.ZMechanicSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.DiameterSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.FocalLengthSpinBox.valueChanged.connect(self.UpdateTxInfo)
        self.Widget.ShowWaterResultscheckBox.stateChanged.connect(self.UpdateAcResults)
        self.Widget.LabelTissueRemoved.setVisible(False)

    def DefaultConfig(self,cfile='default.yaml'):
        #Specific parameters for the CTX500 - to be configured later via a yaml

        with open(os.path.join(resource_path(),cfile), 'r') as file:
            config = yaml.safe_load(file)
        self.Config=config

    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._DataMask.header.get_zooms()[0]
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
        
    
    @Slot()
    def UpdateTxInfo(self):
        if self._bIgnoreUpdate:
            return
        self._bIgnoreUpdate=True
        self.UpdateLimits()
        ZMec=self.Widget.ZMechanicSpinBox.value()
        if ZMec > self.Widget.ZMechanicSpinBox.maximum():
            self.ZMechanicSpinBox.setValue(self.Widget.ZMechanicSpinBox.maximum())
            ZMec=self.Widget.ZMechanicSpinBox.maximum()
        
        CurDistance=self._ZMaxSkin-ZMec
        self.Widget.DistanceTxToSkinLabel.setText('%3.1f' %(CurDistance))
        if CurDistance<0:
            self.Widget.DistanceTxToSkinLabel.setStyleSheet("color: red")
            self.Widget.LabelTissueRemoved.setVisible(True)
        else:
            self.Widget.DistanceTxToSkinLabel.setStyleSheet("color: blue")
            self.Widget.LabelTissueRemoved.setVisible(False)
            
        self._bIgnoreUpdate=False 


    def UpdateLimits(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        DOut=DistanceOutPlaneToFocus(FocalLength,Diameter)
        ZMax=DOut-self.Widget.DistanceSkinLabel.property('UserData')
        self._ZMaxSkin = np.round(ZMax,1)
        self.Widget.ZMechanicSpinBox.setMaximum(self._ZMaxSkin+self.Config['MaxNegativeDistance'])
      
    @Slot()
    def RunSimulation(self):
        FocalLength = self.Widget.FocalLengthSpinBox.value()
        Diameter = self.Widget.DiameterSpinBox.value()
        extrasuffix='Foc%03.1f_Diam%03.1f_' %(FocalLength,Diameter)
        self._FullSolName=self._MainApp._prefix_path+extrasuffix+'DataForSim.h5' 
        self._WaterSolName=self._MainApp._prefix_path+extrasuffix+'Water_DataForSim.h5'
        self._MainApp._BrainsightInput=self._MainApp._prefix_path+extrasuffix+'FullElasticSolution.nii.gz'

        print('FullSolName',self._FullSolName)
        print('WaterSolName',self._WaterSolName)
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
        else:
            bCalcFields = True
        self._bRecalculated = True
        if bCalcFields:
            self._MainApp.Widget.tabWidget.setEnabled(False)
            self.thread = QThread()
            self.worker = RunAcousticSim(self._MainApp,self.thread,
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
        else:
            self.UpdateAcResults()

    def NotifyError(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()

    @Slot()
    def UpdateAcResults(self):
        if self._bRecalculated:
            #this will generate a modified trajectory file
            if self.Widget.ShowWaterResultscheckBox.isEnabled()== False:
                self.Widget.ShowWaterResultscheckBox.setEnabled(True)
            self._MainApp.Widget.tabWidget.setEnabled(True)
            self._MainApp.ThermalSim.setEnabled(True)
            Water=ReadFromH5py(self._WaterSolName)
            Skull=ReadFromH5py(self._FullSolName)
            if self._MainApp._bInUseWithBrainsight:
                with open(self._MainApp._BrainsightSyncPath+os.sep+'Output.txt','w') as f:
                    f.write(self._MainApp._BrainsightInput)    
            self._MainApp.ExportTrajectory(CorX=Skull['AdjustmentInRAS'][0],
                                        CorY=Skull['AdjustmentInRAS'][1],
                                        CorZ=Skull['AdjustmentInRAS'][2])

            LocTarget=Skull['TargetLocation']
            print(LocTarget)

            for d in [Water,Skull]:
                for t in ['p_amp','MaterialMap']:
                    d[t]=np.ascontiguousarray(np.flip(d[t],axis=2))

            DistanceToTarget=self.Widget.DistanceSkinLabel.property('UserData')
            dx=  np.mean(np.diff(Skull['x_vec']))

            Water['z_vec']*=1e3
            Skull['z_vec']*=1e3
            Skull['x_vec']*=1e3
            Skull['y_vec']*=1e3
            Skull['MaterialMap'][Skull['MaterialMap']==3]=2
            Skull['MaterialMap'][Skull['MaterialMap']==4]=3

            IWater=Water['p_amp']**2/2/Water['Material'][0,0]/Water['Material'][0,1]

            DensityMap=Skull['Material'][:,0][Skull['MaterialMap']]
            SoSMap=    Skull['Material'][:,1][Skull['MaterialMap']]

            ISkull=Skull['p_amp']**2/2/Skull['Material'][4,0]/Skull['Material'][4,1]

            IntWaterLocation=IWater[LocTarget[0],LocTarget[1],LocTarget[2]]
            IntSkullLocation=ISkull[LocTarget[0],LocTarget[1],LocTarget[2]]
            
            ISkull[Skull['MaterialMap']!=3]=0
            cxr,cyr,czr=np.where(ISkull==ISkull.max())
            cxr=cxr[0]
            cyr=cyr[0]
            czr=czr[0]

            EnergyAtFocusSkull=ISkull[:,:,czr].sum()*dx**2

            cxr,cyr,czr=np.where(IWater==IWater.max())
            cxr=cxr[0]
            cyr=cyr[0]
            czr=czr[0]

            EnergyAtFocusWater=IWater[:,:,czr].sum()*dx**2

            print('EnergyAtFocusWater',EnergyAtFocusWater,'EnergyAtFocusSkull',EnergyAtFocusSkull)
            
            Factor=EnergyAtFocusWater/EnergyAtFocusSkull
            print('*'*40+'\n'+'*'*40+'\n'+'Correction Factor for Isppa',Factor,'\n'+'*'*40+'\n'+'*'*40+'\n')
            
            ISkull/=ISkull.max()
            IWater/=IWater.max()

            dz=np.diff(Skull['z_vec']).mean()
            Zvec=Skull['z_vec'].copy()
            Zvec-=Zvec[LocTarget[2]]
            Zvec+=DistanceToTarget
            XX,ZZ=np.meshgrid(Skull['x_vec'],Zvec)
            self._XX = XX
            self._ZZX = ZZ
            YY,ZZ=np.meshgrid(Skull['y_vec'],Zvec)
            self._YY = YY
            self._ZZY = ZZ

            self.Widget.IsppaScrollBars.set_default_values(LocTarget,Skull['x_vec']-Skull['x_vec'][LocTarget[0]],Skull['y_vec']-Skull['y_vec'][LocTarget[1]])

            self._Water = Water
            self._IWater = IWater
            self._Skull = Skull
            self._ISkull = ISkull
            self._DistanceToTarget = DistanceToTarget
        
            if hasattr(self,'_figAcField'):
                    children = []
                    for i in range(self._layout.count()):
                        child = self._layout.itemAt(i).widget()
                        if child:
                            children.append(child)
                    for child in children:
                        child.deleteLater()
                    delattr(self,'_figAcField')
                    self.Widget.AcField_plot1.repaint()
        
        if self.Widget.ShowWaterResultscheckBox.isChecked():
            Field=self._IWater
        else:
            Field=self._ISkull

        SelY, SelX = self.Widget.IsppaScrollBars.get_scroll_values()

        if hasattr(self,'_figAcField'):
            if hasattr(self,'_imContourf1'):
                for c in [self._imContourf1,self._imContourf2,self._contour1,self._contour2]:
                    for coll in c.collections:
                        coll.remove()
                del self._imContourf1
                del self._imContourf2
                del self._contour1
                del self._contour2

            self._imContourf1=self._static_ax1.contourf(self._XX,self._ZZX,Field[:,SelY,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            self._contour1 = self._static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)

            self._imContourf2=self._static_ax2.contourf(self._YY,self._ZZY,Field[SelX,:,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            self._contour2 = self._static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)

            self._figAcField.canvas.draw_idle()
        else:
            self._figAcField=Figure(figsize=(14, 12))

            if not hasattr(self,'_layout'):
                self._layout = QVBoxLayout(self.Widget.AcField_plot1)

            self.static_canvas = FigureCanvas(self._figAcField)
            toolbar=NavigationToolbar2QT(self.static_canvas,self)
            self._layout.addWidget(toolbar)
            self._layout.addWidget(self.static_canvas)
            static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)
            self._static_ax1 = static_ax1
            self._static_ax2 = static_ax2

            self._imContourf1=static_ax1.contourf(self._XX,self._ZZX,Field[:,SelY,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax1)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour1 = static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax1.set_aspect('equal')
            static_ax1.set_xlabel('X mm')
            static_ax1.set_ylabel('Z mm')
            static_ax1.invert_yaxis()
            static_ax1.plot(0,self._DistanceToTarget,'+y',markersize=18)

            self._imContourf2=static_ax2.contourf(self._YY,self._ZZY,Field[SelX,:,:].T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax2)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour2 = static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax2.set_aspect('equal')
            static_ax2.set_xlabel('Y mm')
            static_ax2.set_ylabel('Z mm')
            static_ax2.invert_yaxis()
            static_ax2.plot(0,self._DistanceToTarget,'+y',markersize=18)

        self._figAcField.set_facecolor(np.array(self.Widget.palette().color(QPalette.Window).getRgb())/255)
        self._figAcField.set_tight_layout(True)

        #f.set_title('MAIN SIMULATION RESULTS')
        self.Widget.IsppaScrollBars.update_labels(SelX, SelY)
        self._bRecalculated = False
   
    def GetExport(self):
        Export={}
        for k in ['FocalLength','Diameter','XMechanic','YMechanic','ZMechanic']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

class RunAcousticSim(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp,thread,extrasuffix,Aperture,FocalLength):
        super(RunAcousticSim, self).__init__()
        self._mainApp=mainApp
        self._thread=thread
        self._extrasuffix=extrasuffix
        self._Aperture=Aperture
        self._FocalLength=FocalLength

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

        # Start mask generation as separate process.
        bNoError=True
        queue=Queue()
        T0=time.time()
        fieldWorkerProcess = Process(target=CalculateFieldProcess, 
                                    args=(queue,Target),
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

