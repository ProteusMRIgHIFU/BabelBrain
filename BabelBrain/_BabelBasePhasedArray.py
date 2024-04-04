'''
Base Class for Tx concave phase array GUI, not to be instantiated directly
'''

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
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

from CalculateFieldProcess import CalculateFieldProcess
import platform 

from _BabelBaseTx import BabelBaseTx


class BabelBasePhaseArray(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None,formfile=None):
        super().__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self._MultiPoint = None #if None, the default is to run one single focal point
        self.DefaultConfig()
        self.load_ui(formfile)


    def load_ui(self,formfile):
        loader = QUiLoader()
        path =  formfile
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget =loader.load(ui_file, self)
        ui_file.close()

        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)

        for spinbox,ID in zip([self.Widget.XSteeringSpinBox,
                               self.Widget.YSteeringSpinBox,
                               self.Widget.ZSteeringSpinBox],
                               ['X','Y','Z']):
            
            spinbox.setMinimum(self.Config['Minimal'+ID+'Steering']*1e3)
            spinbox.setMaximum(self.Config['Maximal'+ID+'Steering']*1e3)
            spinbox.setValue(0.0)

        self.Widget.DistanceConeToFocusSpinBox.setMinimum(self.Config['MinimalDistanceConeToFocus']*1e3)
        self.Widget.DistanceConeToFocusSpinBox.setMaximum(self.Config['MaximalDistanceConeToFocus']*1e3)
        self.Widget.DistanceConeToFocusSpinBox.setValue(self.Config['DefaultDistanceConeToFocus']*1e3)
        
        self.Widget.MultifocusLabel.setVisible(False)
        self.Widget.SelCombinationDropDown.setVisible(False)
        while self.Widget.SelCombinationDropDown.count()>0:
            self.Widget.SelCombinationDropDown.removeItem(0)
        self.Widget.SelCombinationDropDown.addItem('ALL') # Add this will cover the case of single focus
        
        self.Widget.ZSteeringSpinBox.valueChanged.connect(self.ZSteeringUpdate)
        self.Widget.RefocusingcheckBox.stateChanged.connect(self.EnableRefocusing)
        self.Widget.CalculateAcField.clicked.connect(self.RunSimulation)
        self.Widget.ZMechanicSpinBox.setVisible(False) #for these Tx, we disable ZMechanic as this is controlled by the distance cone to focus
        self.Widget.ZMechaniclabel.setVisible(False)
        self.Widget.CalculateMechAdj.clicked.connect(self.CalculateMechAdj)
        self.Widget.CalculateMechAdj.setEnabled(False)
        self.up_load_ui()
        
       
    @Slot()
    def ZSteeringUpdate(self,value):
        self._ZSteering =self.Widget.ZSteeringSpinBox.value()/1e3

    @Slot()
    def EnableRefocusing(self,value):
        bRefocus =self.Widget.RefocusingcheckBox.isChecked()
        self.Widget.XMechanicSpinBox.setEnabled(not bRefocus)
        self.Widget.YMechanicSpinBox.setEnabled(not bRefocus)
        self.Widget.ZMechanicSpinBox.setEnabled(not bRefocus)

    def DefaultConfig(self):
        #Specific parameters for the Tx - to be configured later via a yaml
        #to be defined by child classess
        raise NotImplementedError("This needs to be defined by the child class")

        
    def NotifyGeneratedMask(self):
        VoxelSize=self._MainApp._MaskData.header.get_zooms()[0]
        TargetLocation =np.array(np.where(self._MainApp._FinalMask==5.0)).flatten()
        LineOfSight=self._MainApp._FinalMask[TargetLocation[0],TargetLocation[1],:]
        StartSkin=np.where(LineOfSight>0)[0].min()
        DistanceFromSkin = (TargetLocation[2]-StartSkin)*VoxelSize

        self.Widget.DistanceSkinLabel.setText('%3.2f'%(DistanceFromSkin))
        self.Widget.DistanceSkinLabel.setProperty('UserData',DistanceFromSkin)
        self._UnmodifiedZMechanic = 0.0
        self.ZSteeringUpdate(0)

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

            ret = QMessageBox.question(self,'', "Acoustic sim files already exist with:.\n"+
                                    "XSteering=%3.2f\n" %(XSteering*1e3)+
                                    "YSteering=%3.2f\n" %(YSteering*1e3)+
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
                self.Widget.XSteeringSpinBox.setValue(XSteering*1e3)
                self.Widget.YSteeringSpinBox.setValue(YSteering*1e3)
                self.Widget.ZSteeringSpinBox.setValue(ZSteering*1e3)
                self.Widget.ZRotationSpinBox.setValue(RotationZ)
                self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'])
                if 'DistanceConeToFocus' in Skull:
                    self.Widget.DistanceConeToFocusSpinBox.setValue(Skull['DistanceConeToFocus']*1e3)
                if 'zLengthBeyonFocalPoint' in Skull:
                    self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
                self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
                self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
                self.Widget.ZMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentZ']*1e3)
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
        Export=super().GetExport()
        Export['Refocusing']=self.Widget.RefocusingcheckBox.isChecked()
        def dict_to_string(d, separator=', ', equals_sign='='):
            return separator.join(f'{key}:{value*1000.0}' for key, value in d.items())
        if self._MultiPoint is not None:
            st =''
            for e in self._MultiPoint:
                st+='[%s] ' % dict_to_string(e)
            Export['MultiPoint']=st
        else:
            self._MultiPoint ='N/A'
         
        for k in ['ZSteering','ZRotation','DistanceConeToFocus','XMechanic','YMechanic','ZMechanic']:
            if hasattr(self.Widget,k+'SpinBox'):
                Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export

    @Slot()
    def EndSimulation(self,OutFiles):
        assert(type(OutFiles['FilesSkull']) is list)
        assert(type(OutFiles['FilesWater']) is list)
        
        if self._MultiPoint is None:
            assert(len(OutFiles['FilesSkull'])==1)
            assert(len(OutFiles['FilesWater'])==1)
        else:
            assert(len(OutFiles['FilesSkull'])==len(self._MultiPoint))
            assert(len(OutFiles['FilesSkull'])==len(self._MultiPoint))
            
        self.UpdateAcResults()
        
    @Slot()
    def UpdateAcResults(self):
        self._MainApp.SetSuccesCode()
        self.Widget.CalculateMechAdj.setEnabled(True)
        #We overwrite the base class method
        if self._bRecalculated:
            self._MainApp.hideClockDialog()
            self._AcResults =[]
            #this will generate a modified trajectory file
            if self.Widget.ShowWaterResultscheckBox.isEnabled()== False:
                self.Widget.ShowWaterResultscheckBox.setEnabled(True)
            if self.Widget.HideMarkscheckBox.isEnabled()== False:
                self.Widget.HideMarkscheckBox.setEnabled(True)
            self._MainApp.Widget.tabWidget.setEnabled(True)
            self._MainApp.ThermalSim.setEnabled(True)
            
            for fwater,fskull in zip(self._WaterSolName,self._FullSolName):
                Skull=ReadFromH5py(fskull)
                Water=ReadFromH5py(fwater)
    
                if Skull['bDoRefocusing']:
                    SelP='p_amp_refocus'
                else:
                    SelP='p_amp'

                for t in [SelP,'MaterialMap']:
                    Skull[t]=np.ascontiguousarray(np.flip(Skull[t],axis=2))

                for t in ['p_amp','MaterialMap']:
                    Water[t]=np.ascontiguousarray(np.flip(Water[t],axis=2))
                Water['p_amp'][:,:,0]=0.0
                
                entry={'Skull':Skull,'Water':Water}
                
                self._AcResults.append(entry)
            
            Water=self._AcResults[0]['Water']
            Skull=self._AcResults[0]['Skull']
            
            if Skull['bDoRefocusing']:
                SelP='p_amp_refocus'
            else:
                SelP='p_amp'
            
            if self._MainApp.Config['bInUseWithBrainsight']:
                if Skull['bDoRefocusing']:
                    #we update the name to be loaded in BSight
                    self._MainApp._BrainsightInput=self._MainApp._prefix_path+'FullElasticSolutionRefocus_Sub_NORM.nii.gz'
                else:
                    self._MainApp._BrainsightInput=self._MainApp._prefix_path+'FullElasticSolution_Sub_NORM.nii.gz'
            self.ExportStep2Results(Skull)

            LocTarget=Skull['TargetLocation']
            print(LocTarget)

            DistanceToTarget=self.Widget.DistanceSkinLabel.property('UserData')

            Water['z_vec']*=1e3
            Skull['z_vec']*=1e3
            Skull['x_vec']*=1e3
            Skull['y_vec']*=1e3
            Skull['MaterialMap'][Skull['MaterialMap']==3]=2
            Skull['MaterialMap'][Skull['MaterialMap']==4]=3

            DensityMap=Skull['Material'][:,0][Skull['MaterialMap']]
            SoSMap=    Skull['Material'][:,1][Skull['MaterialMap']]
            
            self._ISkullCol=[]
            self._IWaterCol=[]
            sz=self._AcResults[0]['Water']['p_amp'].shape
            AllSkull=np.zeros((sz[0],sz[1],sz[2],len(self._AcResults)))
            AllWater=np.zeros((sz[0],sz[1],sz[2],len(self._AcResults)))
            for n,entry in enumerate(self._AcResults):
                ISkull=entry['Skull'][SelP]**2/2/DensityMap/SoSMap/1e4
                ISkull[Skull['MaterialMap']!=3]=0
                IWater=entry['Water']['p_amp']**2/2/Water['Material'][0,0]/Water['Material'][0,1]
                
                AllSkull[:,:,:,n]=ISkull
                AllWater[:,:,:,n]=IWater
                ISkull/=ISkull.max()
                IWater/=IWater.max()
                
                self._ISkullCol.append(ISkull)
                self._IWaterCol.append(IWater)
            #now we add the max projection of fields, we add it at the top
            AllSkull=AllSkull.max(axis=3)
            AllSkull[Skull['MaterialMap']!=3]=0
            AllSkull/=AllSkull.max()
            AllWater=AllWater.max(axis=3)
            AllWater/=AllWater.max()
            
            self._ISkullCol.insert(0,AllSkull)
            self._IWaterCol.insert(0,AllWater)
            
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

            self._Skull = Skull
            
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
                
                
        
        SelY, SelX = self.Widget.IsppaScrollBars.get_scroll_values()

        if hasattr(self.Widget,'SelCombinationDropDown'):
            IWater = self._IWaterCol[self.Widget.SelCombinationDropDown.currentIndex()]
            ISkull = self._ISkullCol[self.Widget.SelCombinationDropDown.currentIndex()]
        else:
            IWater = self._IWaterCol[0]
            ISkull = self._ISkullCol[0]
        #we need to declare these for compatibility for parent functions
        self._IWater = IWater
        self._ISkull = ISkull

        if self.Widget.ShowWaterResultscheckBox.isChecked():
            sliceXZ=IWater[:,SelY,:]
            sliceYZ = IWater[SelX,:,:]
        else:
            sliceXZ=ISkull[:,SelY,:]
            sliceYZ =ISkull[SelX,:,:]

        if hasattr(self,'_figAcField'):
            if hasattr(self,'_imContourf1'):
                for c in [self._imContourf1,self._imContourf2,self._contour1,self._contour2]:
                    for coll in c.collections:
                        coll.remove()
                del self._imContourf1
                del self._imContourf2
                del self._contour1
                del self._contour2

            self._imContourf1=self._static_ax1.contourf(self._XX,self._ZZX,sliceXZ.T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            self._contour1 = self._static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)

            self._imContourf2=self._static_ax2.contourf(self._YY,self._ZZY,sliceYZ.T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
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

            self._imContourf1=static_ax1.contourf(self._XX,self._ZZX,sliceXZ.T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax1)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour1 = static_ax1.contour(self._XX,self._ZZX,self._Skull['MaterialMap'][:,SelY,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax1.set_aspect('equal')
            static_ax1.set_xlabel('X mm')
            static_ax1.set_ylabel('Z mm')
            static_ax1.invert_yaxis()
            self._marker1,=static_ax1.plot(0,self._DistanceToTarget,'+k',markersize=18)
                
            self._imContourf2=static_ax2.contourf(self._YY,self._ZZY,sliceYZ.T,np.arange(2,22,2)/20,cmap=plt.cm.jet)
            h=plt.colorbar(self._imContourf1,ax=static_ax2)
            h.set_label('$I_{\mathrm{SPPA}}$ (normalized)')
            self._contour2 = static_ax2.contour(self._YY,self._ZZY,self._Skull['MaterialMap'][SelX,:,:].T,[0,1,2,3], cmap=plt.cm.gray)
            static_ax2.set_aspect('equal')
            static_ax2.set_xlabel('Y mm')
            static_ax2.set_ylabel('Z mm')
            static_ax2.invert_yaxis()
            self._marker2,=static_ax2.plot(0,self._DistanceToTarget,'+k',markersize=18)
        
        self._figAcField.set_facecolor(np.array(self.Widget.palette().color(QPalette.Window).getRgb())/255)

        mc=[0.0,0.0,0.0,1.0]
        if self.Widget.HideMarkscheckBox.isChecked():
             mc[3] = 0.0
        self._marker1.set_markerfacecolor(mc)
        self._marker2.set_markerfacecolor(mc)

        self.Widget.IsppaScrollBars.update_labels(SelX, SelY)
        self._bRecalculated = False
    
    def EnableMultiPoint(self,MultiPoint):
        self.Widget.MultifocusLabel.setVisible(True)
        self.Widget.SelCombinationDropDown.setVisible(True)
        while self.Widget.SelCombinationDropDown.count()>0:
            self.Widget.SelCombinationDropDown.removeItem(0)
        self.Widget.SelCombinationDropDown.addItem('ALL')
        print('MultiPoint',MultiPoint)
        for c in MultiPoint:
            self.Widget.SelCombinationDropDown.addItem('X:%2.1f Y:%2.1f Z:%2.1f' %(c['X']*1e3,c['Y']*1e3,c['Z']*1e3))
        self._MultiPoint = MultiPoint
        self.Widget.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateAcResults)


class RunAcousticSim(QObject):

    finished = Signal(object)
    endError = Signal()

    def __init__(self,mainApp,bDryRun=False):
        super().__init__()
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
            TxMechanicalAdjustmentZ= self._mainApp.AcSim.Widget.ZMechanicSpinBox.value()/1e3  #in m

        else:
            TxMechanicalAdjustmentX=0
            TxMechanicalAdjustmentY=0
            TxMechanicalAdjustmentZ=0
        ###############
        ZSteering=self._mainApp.AcSim.Widget.ZSteeringSpinBox.value()/1e3  #Add here the final adjustment)
        XSteering=self._mainApp.AcSim.Widget.XSteeringSpinBox.value()/1e3
        YSteering=self._mainApp.AcSim.Widget.YSteeringSpinBox.value()/1e3

        if XSteering==0.0 and YSteering==0.0 and ZSteering==0.0:
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
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['bDoRefocusing']=bRefocus
        kargs['DistanceConeToFocus']=DistanceConeToFocus
        kargs['bUseCT']=self._mainApp.Config['bUseCT']
        kargs['bUseRayleighForWater']=self._mainApp.Config['bUseRayleighForWater']
        kargs['bPETRA'] = False
        kargs['MultiPoint'] =self._mainApp.AcSim._MultiPoint
        kargs['bDryRun'] = self._bDryRun
            
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

