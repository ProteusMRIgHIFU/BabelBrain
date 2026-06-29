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
import nibabel

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
from GUIComponents.AppStyle import style_nav_toolbar

from CalculateFieldProcess import CalculateFieldProcess
import platform 

from _BabelBaseTx import BabelBaseTx


class BabelBasePhaseArray(BabelBaseTx):
    def __init__(self,parent=None,MainApp=None,formtype=None):
        super().__init__(parent)
        self.static_canvas=None
        self._MainApp=MainApp
        self._MultiPoint = None #if None, the default is to run one single focal point
        self.DefaultConfig()
        self.load_ui(formtype)


    def load_ui(self,formtype):
        # the concrete form class is passed in via `formtype` (a class, or a
        # legacy path resolved by suffix); stash it for _CreateForm and build the
        # per-trajectory tabs.
        self._formtype = formtype
        self._setupTrajectoryTabs()

    def _CreateForm(self):
        formtype = self._formtype
        if isinstance(formtype, type):
            form_cls = formtype
        else:
            # Backwards compat: map old .ui paths to the new form classes.
            if "Babel_DomeTx" in str(formtype):
                from Babel_DomeTx.DomeTxForm import DomeTxForm as form_cls
            elif "Babel_H317" in str(formtype):
                from Babel_H317.H317Form import H317Form as form_cls
            else:
                raise ValueError(f"No programmatic form mapping for {formtype}")
        return form_cls(self)

    def _WirePanel(self):
        self.Widget.IsppaScrollBars = WidgetScrollBars(parent=self.Widget.IsppaScrollBars,MainApp=self)

        for spinbox,ID in zip([self.Widget.XSteeringSpinBox,
                               self.Widget.YSteeringSpinBox,
                               self.Widget.ZSteeringSpinBox],
                               ['X','Y','Z']):

            spinbox.setMinimum(self.Config['Minimal'+ID+'Steering']*1e3)
            spinbox.setMaximum(self.Config['Maximal'+ID+'Steering']*1e3)
            spinbox.setValue(0.0)

        if hasattr(self.Widget,'DistanceConeToFocusSpinBox'):
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
        
        if hasattr(self.Widget,'DistanceConeToFocusSpinBox'):
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
        if hasattr( self.Widget,'ZMechanicSpinBox'):
            self.Widget.ZMechanicSpinBox.setEnabled(not bRefocus)

    def DefaultConfig(self):
        #Specific parameters for the Tx - to be configured later via a yaml
        #to be defined by child classess
        raise NotImplementedError("This needs to be defined by the child class")

        
    def NotifyGeneratedMask(self):
        self._SyncActiveTrajectoryFromMainApp()
        self.CalculateDistanceFromSkin()
        self._UnmodifiedZMechanic = 0.0
        self.ZSteeringUpdate(0)

    @Slot()
    def _ResolveSimulationFilenames(self):
        #we create an object to do a dryrun to recover filenames
        dry=RunAcousticSim(self._MainApp,bDryRun=True)
        FILENAMES = dry.run()

        self._FullSolName=FILENAMES['FilesSkull']
        self._WaterSolName=FILENAMES['FilesWater']

    def _ExistingSimulationFiles(self):
        for sskull,swater in zip(self._FullSolName,self._WaterSolName):
            if not(os.path.isfile(sskull) and os.path.isfile(swater)):
                return False
        return True

    def _PromptReuseOrRecalc(self):
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
            return True
        self.Widget.XSteeringSpinBox.setValue(XSteering*1e3)
        self.Widget.YSteeringSpinBox.setValue(YSteering*1e3)
        self.Widget.ZSteeringSpinBox.setValue(ZSteering*1e3)
        self.Widget.ZRotationSpinBox.setValue(RotationZ)
        try:
            self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'])
        except:
            self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'].astype(int))
        if 'DistanceConeToFocus' in Skull and hasattr(self.Widget,'DistanceConeToFocusSpinBox'):
            self.Widget.DistanceConeToFocusSpinBox.setValue(Skull['DistanceConeToFocus']*1e3)
        if 'zLengthBeyonFocalPoint' in Skull:
            self.Widget.MaxDepthSpinBox.setValue(Skull['zLengthBeyonFocalPoint']*1e3)
        self.Widget.XMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentX']*1e3)
        self.Widget.YMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentY']*1e3)
        self.Widget.ZMechanicSpinBox.setValue(Skull['TxMechanicalAdjustmentZ']*1e3)
        return False

    def _CreateAcousticWorker(self):
        return RunAcousticSim(self._MainApp)

    def _SimulationFinishedSlot(self):
        return self.EndSimulation


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
    
    def GetExtraDataForThermal(self):
        ExtraValues=super().GetExtraDataForThermal()
        ExtraValues['DistanceConeToFocus']=self._LastDistanceConeToFocus
        return ExtraValues         

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

    # Phased arrays use a larger figure and select the displayed field from the
    # multifocus column set; the per-trajectory tab/render plumbing is inherited
    # from BabelBaseTx.
    def _AcResultFigure(self):
        return Figure(figsize=(14, 12))

    def _GetActiveFields(self, panel):
        if hasattr(self.Widget, 'SelCombinationDropDown'):
            i = self.Widget.SelCombinationDropDown.currentIndex()
            if i < 0:
                i = 0
        else:
            i = 0
        return panel['IWaterCol'][i], panel['ISkullCol'][i]

    def _LoadAcResultData(self, panel):
        '''Read all steering results for this trajectory and stash the field columns.'''
        AcResults = []
        for fwater, fskull in zip(self._WaterSolName, self._FullSolName):
            Skull = ReadFromH5py(fskull)
            Water = ReadFromH5py(fwater)

            if 'SDR' in Skull and hasattr(self.Widget, 'SDRLabel'):
                self._SDR = Skull['SDR']
                self.Widget.SDRLabel.setText('%0.2f' % (Skull['SDR']))
                panel['SDR'] = Skull['SDR']

            if Skull['bDoRefocusing']:
                SelP = 'p_amp_refocus'
            else:
                SelP = 'p_amp'

            keys = [SelP, 'MaterialMap']
            if 'AirMask' in Skull:
                keys.append('AirMask')
            for t in keys:
                Skull[t] = np.ascontiguousarray(np.flip(Skull[t], axis=2))

            for t in ['p_amp', 'MaterialMap']:
                Water[t] = np.ascontiguousarray(np.flip(Water[t], axis=2))
            Water['p_amp'][:, :, 0] = 0.0

            AcResults.append({'Skull': Skull, 'Water': Water})

        Water = AcResults[0]['Water']
        Skull = AcResults[0]['Skull']

        if Skull['bDoRefocusing']:
            SelP = 'p_amp_refocus'
        else:
            SelP = 'p_amp'

        if self._MainApp.Config['bInUseWithBrainsight']:
            if Skull['bDoRefocusing']:
                #we update the name to be loaded in BSight
                self._MainApp._BrainsightInput = self._MainApp._prefix_path[self._TrajectoryNumber] + 'FullElasticSolutionRefocus_Sub_NORM.nii.gz'
            else:
                self._MainApp._BrainsightInput = self._MainApp._prefix_path[self._TrajectoryNumber] + 'FullElasticSolution_Sub_NORM.nii.gz'
        self.ExportStep2Results(Skull)

        LocTarget = Skull['TargetLocation']
        print(LocTarget)

        DistanceToTarget = self.Widget.DistanceSkinLabel.property('UserData')
        if 'DistanceConeToFocus' in Skull:
            self._LastDistanceConeToFocus = Skull['DistanceConeToFocus']
        else:
            self._LastDistanceConeToFocus = 0.0

        Water['z_vec'] *= 1e3
        Skull['z_vec'] *= 1e3
        Skull['x_vec'] *= 1e3
        Skull['y_vec'] *= 1e3

        DensityMap = Skull['Material'][:, 0][Skull['MaterialMap']]
        SoSMap = Skull['Material'][:, 1][Skull['MaterialMap']]

        Skull['MaterialMap'][Skull['MaterialMap'] == 3] = 2
        Skull['MaterialMap'][Skull['MaterialMap'] == 4] = 3

        ISkullCol = []
        IWaterCol = []
        sz = AcResults[0]['Water']['p_amp'].shape
        AllSkull = np.zeros((sz[0], sz[1], sz[2], len(AcResults)))
        AllWater = np.zeros((sz[0], sz[1], sz[2], len(AcResults)))
        for n, entry in enumerate(AcResults):
            ISkull = entry['Skull'][SelP] ** 2 / 2 / DensityMap / SoSMap / 1e4
            if not self._MainApp.Config['bForceHomogenousMedium']:
                ISkull[Skull['MaterialMap'] < 3] = 0
            IWater = entry['Water']['p_amp'] ** 2 / 2 / Water['Material'][0, 0] / Water['Material'][0, 1]

            AllSkull[:, :, :, n] = ISkull
            AllWater[:, :, :, n] = IWater
            ISkull /= ISkull.max()
            IWater /= IWater.max()

            ISkullCol.append(ISkull)
            IWaterCol.append(IWater)
        #now we add the max projection of fields, we add it at the top
        AllSkull = AllSkull.max(axis=3)
        AllSkull /= AllSkull.max()
        AllWater = AllWater.max(axis=3)
        AllWater /= AllWater.max()

        ISkullCol.insert(0, AllSkull)
        IWaterCol.insert(0, AllWater)

        Zvec = Skull['z_vec'].copy()
        Zvec -= Zvec[LocTarget[2]]
        Zvec += DistanceToTarget
        XX, ZZX = np.meshgrid(Skull['x_vec'], Zvec)
        YY, ZZY = np.meshgrid(Skull['y_vec'], Zvec)

        # Keep self._AcResults pointing at the active trajectory for any external use.
        self._AcResults = AcResults
        panel.update({
            'AcResults': AcResults, 'Skull': Skull,
            'ISkullCol': ISkullCol, 'IWaterCol': IWaterCol,
            'XX': XX, 'ZZX': ZZX, 'YY': YY, 'ZZY': ZZY,
            'DistanceToTarget': DistanceToTarget, 'LocTarget': LocTarget,
            'xvec': Skull['x_vec'] - Skull['x_vec'][LocTarget[0]],
            'yvec': Skull['y_vec'] - Skull['y_vec'][LocTarget[1]],
            'FullSolName': self._FullSolName, 'WaterSolName': self._WaterSolName,
            'LastDistanceConeToFocus': self._LastDistanceConeToFocus,
        })

    @Slot()
    def UpdateAcResults(self):
        self._MainApp.SetSuccesCode()
        self.Widget.CalculateMechAdj.setEnabled(True)
        if self._bRecalculated:
            self._MainApp.ThermalSim.setEnabled(True)
            self._MainApp.hideClockDialog()
            self._MainApp.Widget.tabWidget.setEnabled(True)
        self._showMatplotlibVisualization()
        if self._MultiPoint:
            NiftiSkull=nibabel.load(self._FullSolName[0].split('__Steer')[0]+'_FullElasticSolution_Sub_NORM.nii.gz')
            NiftiWater=nibabel.load(self._FullSolName[0].split('__Steer')[0]+'_Water_FullElasticSolution_Sub_NORM.nii.gz')
        else:
            NiftiSkull=nibabel.load(self._FullSolName[0].split('_DataForSim.h5')[0]+'_FullElasticSolution_Sub_NORM.nii.gz')
            NiftiWater=nibabel.load(self._FullSolName[0].split('_DataForSim.h5')[0]+'_Water_FullElasticSolution_Sub_NORM.nii.gz')

        self._MainApp.UpdateNiftiAcResults(NiftiSkull,NiftiWater,self._TrajectoryNumber)

        
    
    def EnableMultiPoint(self,MultiPoint):
        print('MultiPoint',MultiPoint)
        # Apply the multifocus dropdown to every trajectory tab's form.
        for form in self._Widgets:
            form.MultifocusLabel.setVisible(True)
            form.SelCombinationDropDown.setVisible(True)
            while form.SelCombinationDropDown.count()>0:
                form.SelCombinationDropDown.removeItem(0)
            form.SelCombinationDropDown.addItem('ALL')
            for c in MultiPoint:
                form.SelCombinationDropDown.addItem('X:%2.1f Y:%2.1f Z:%2.1f' %(c['X']*1e3,c['Y']*1e3,c['Z']*1e3))
            form.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateAcResults)
        self._MultiPoint = MultiPoint


class RunAcousticSim(QObject):

    finished = Signal(object)
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self,mainApp,bDryRun=False):
        super().__init__()
        self._mainApp=mainApp
        self._bDryRun=bDryRun

    def run(self):
        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']
        basedir,ID=os.path.split(os.path.split(self._mainApp.Config['T1WIso'])[0])
        basedir+=os.sep
        Target=[self._mainApp.Config['ID'][self._mainApp.AcSim._TrajectoryNumber]+'_'+self._mainApp.Config['TxSystem']]

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

        Frequencies = [self._mainApp._Frequency]
        basePPW=[self._mainApp._BasePPW]
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
        kargs['Frequencies']=Frequencies
        kargs['zLengthBeyonFocalPointWhenNarrow']=self._mainApp.AcSim.Widget.MaxDepthSpinBox.value()/1e3
        kargs['bDoRefocusing']=bRefocus
        if hasattr(self._mainApp.AcSim.Widget,'DistanceConeToFocusSpinBox'):
            DistanceConeToFocus=self._mainApp.AcSim.Widget.DistanceConeToFocusSpinBox.value()/1e3
            kargs['DistanceConeToFocus']=DistanceConeToFocus
        kargs['MultiPoint'] =self._mainApp.AcSim._MultiPoint
        kargs['bDryRun'] = self._bDryRun
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

