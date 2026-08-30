# This Python file uses the following encoding: utf-8
from multiprocessing import Process,Queue
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QFileDialog
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread,Qt
from PySide6.QtUiTools import QUiLoader


import numpy as np


#import cv2 as cv
import os
import sys
import platform
import time
import yaml
from BabelViscoFDTD.H5pySimple import ReadFromH5py
from GUIComponents.ScrollBars import ScrollBars as WidgetScrollBars

import nibabel

from CalculateFieldProcess import CalculateFieldProcess

from _BabelBasePhasedArray import BabelBasePhaseArray
from ConvMatTransform import ReadTrajectoryBrainsight
from TranscranialModeling.BabelIntegrationREMOPD import DeviceFrameSteering

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

def _brainsight_origin_ras_mm(path):
    '''Origin (Loc X/Y/Z) of a Brainsight trajectory export, in NIfTI RAS mm.'''
    R = ReadTrajectoryBrainsight(path)
    if getattr(R, 'ndim', 2) == 3:
        R = R[:, :, 0]
    return np.asarray(R[:3, 3], dtype=float)

def mechanical_xy_from_feasible_ras_mm(mask_path, ras_mm):
    '''Raw domain X/Y (mm) from a feasible RAS point to the intended target.

    Same math as Sam's CalcRayXYDistance.py: intended = label 5 in the
    Step 1 *BabelViscoInput.nii.gz (the trajectory BabelBrain was launched
    with). Keeps that intended point at the center of Step 2/3.
    '''
    if not mask_path or 'BabelViscoInput.nii.gz' not in os.path.basename(mask_path):
        raise ValueError('Step 1 mask (*BabelViscoInput.nii.gz) is not available.')
    if not os.path.isfile(mask_path):
        raise ValueError('Run Step 1 first so the simulation mask exists.')

    # Sam's CalcRayXYDistance.py, verbatim (label 5 = intended target).
    ras = np.array([float(ras_mm[0]), float(ras_mm[1]), float(ras_mm[2]), 1], dtype=float).reshape((4, 1))
    inb = nibabel.load(mask_path)
    zoom = inb.header.get_zooms()
    data = inb.get_fdata().astype(int)
    TargetIJK = np.array(np.where(data == 5)).flatten()
    if TargetIJK.size < 3:
        raise ValueError('Intended target (label 5) was not found in the Step 1 mask.')
    inv_affine = np.linalg.inv(inb.affine)
    AffIJK = np.round(np.dot(inv_affine, ras)).flatten()[:3]
    DiffIJK = AffIJK - TargetIJK
    return float(DiffIJK[0] * zoom[0]), float(DiffIJK[1] * zoom[1])

class REMOPD(BabelBasePhaseArray): 
    def __init__(self,parent=None,MainApp=None):
        super().__init__(parent=parent,MainApp=MainApp,formtype=os.path.join(resource_path(), "."))

    # Inherits BabelBasePhaseArray.load_ui (-> _setupTrajectoryTabs); only the
    # form and its wiring differ.
    def _CreateForm(self):
        from Babel_REMOPD.REMOPDForm import REMOPDForm
        return REMOPDForm(self)

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
        # Sam: Y-flip and this action only when BabelBrain is called from Brainsight.
        b_brainsight = bool(self._MainApp.Config.get('bInUseWithBrainsight'))
        self.Widget.ApplyFeasibleTraj.setVisible(b_brainsight)
        if b_brainsight:
            self.Widget.ApplyFeasibleTraj.clicked.connect(self.ApplyFeasibleTrajectory)
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
        self._SyncActiveTrajectoryFromMainApp()
        DistanceFromSkin = self.CalculateDistanceFromSkin()
        self.Widget.ZSteeringSpinBox.setValue(np.round(DistanceFromSkin,1))

    @Slot()
    def ApplyFeasibleTrajectory(self):
        '''Park the array on a Brainsight feasible pose; steer back to intended.

        remopd/feasible-traj: intended trajectory (already loaded) stays label 5.
        The user picks the feasible Brainsight .txt (no retyped RAS). Mechanical
        X/Y slide the array to that pose; steering is the opposite offset so the
        electronic focus stays on intended. DeviceFrameSteering is an involution,
        so the same map fills the GUI when Brainsight Y is flipped in the solver.
        '''
        if not bool(self._MainApp.Config.get('bInUseWithBrainsight')):
            return
        start = ''
        mat4 = self._MainApp.Config.get('Mat4Trajectory') or ''
        if mat4 and os.path.isfile(mat4):
            start = os.path.dirname(mat4)
        elif self._MainApp.Config.get('OutputFilesPath'):
            start = self._MainApp.Config['OutputFilesPath']
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Brainsight feasible trajectory',
            start,
            'Brainsight trajectory (*.txt);;All files (*)')
        if not path:
            return
        try:
            masks = getattr(self._MainApp, '_outnameMask', None)
            idx = getattr(self, '_TrajectoryNumber', 0)
            if not masks:
                raise ValueError('Run Step 1 first so the simulation mask exists.')
            mask_path = masks[idx]
            ras = _brainsight_origin_ras_mm(path)
            mech_x, mech_y = mechanical_xy_from_feasible_ras_mm(mask_path, ras)
            # Domain steer that keeps the focus on label 5 after the array slides.
            gui_x, gui_y = DeviceFrameSteering(-mech_x, -mech_y, flip_y=True)
        except Exception as e:
            QMessageBox.critical(self, 'Apply feasible trajectory', str(e))
            return

        self.Widget.XMechanicSpinBox.setValue(np.round(mech_x, 1))
        self.Widget.YMechanicSpinBox.setValue(np.round(mech_y, 1))
        self.Widget.XSteeringSpinBox.setValue(np.round(gui_x, 1))
        self.Widget.YSteeringSpinBox.setValue(np.round(gui_y, 1))

        xmin = self.Widget.XSteeringSpinBox.minimum()
        xmax = self.Widget.XSteeringSpinBox.maximum()
        ymin = self.Widget.YSteeringSpinBox.minimum()
        ymax = self.Widget.YSteeringSpinBox.maximum()
        warn = []
        if not (xmin <= gui_x <= xmax) or not (ymin <= gui_y <= ymax):
            warn.append('Steering is outside the allowed range and was clamped.')
        mxmin = self.Widget.XMechanicSpinBox.minimum()
        mxmax = self.Widget.XMechanicSpinBox.maximum()
        mymin = self.Widget.YMechanicSpinBox.minimum()
        mymax = self.Widget.YMechanicSpinBox.maximum()
        if not (mxmin <= mech_x <= mxmax) or not (mymin <= mech_y <= mymax):
            warn.append('Mechanical X/Y are outside the allowed range and were clamped.')
        extra = ('\n\n' + ' '.join(warn)) if warn else ''
        QMessageBox.information(
            self,
            'Apply feasible trajectory',
            'Mechanical X, Y (mm): %0.1f, %0.1f\n'
            'Steering X, Y (mm): %0.1f, %0.1f\n\n'
            'Mechanical slides the array to the feasible pose. '
            'Steering is the opposite offset so the focus stays on the intended target.'
            '%s' % (mech_x, mech_y, gui_x, gui_y, extra))

    @Slot()
    def _ResolveSimulationFilenames(self):
        #we create an object to do a dryrun to recover filenames
        dry=RunAcousticSim(self._MainApp,bDryRun=True)
        FILENAMES = dry.run()

        self._FullSolName=FILENAMES['FilesSkull']
        self._WaterSolName=FILENAMES['FilesWater']

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
            return True
        self.Widget.XSteeringSpinBox.setValue(XSteering*1e3)
        self.Widget.YSteeringSpinBox.setValue(YSteering*1e3)
        self.Widget.ZSteeringSpinBox.setValue(ZSteering*1e3)
        self.Widget.ZRotationSpinBox.setValue(RotationZ)
        try:
            self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'])
        except:
            self.Widget.RefocusingcheckBox.setChecked(Skull['bDoRefocusing'].astype(int))
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
        return False

    def _CreateAcousticWorker(self):
        return RunAcousticSim(self._MainApp)

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
         
        for k in ['XSteering','YSteering','ZSteering','ZRotation','XMechanic','YMechanic','SkinDistance']:
            Export[k]=getattr(self.Widget,k+'SpinBox').value()
        return Export
    
    def EnableMultiPoint(self,MultiPoint):
        pass #we disable multipoint for the time being

class RunAcousticSim(QObject):

    finished = Signal(object)
    endError = Signal()
    logTelemetry = Signal(str)

    def __init__(self,mainApp,bDryRun=False):
        super(RunAcousticSim, self).__init__()
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
        kargs['XSteering']=XSteering
        kargs['YSteering']=YSteering
        kargs['ZSteering']=ZSteering
        kargs['RotationZ']=RotationZ
        kargs['TxSet']=TxSet
        # GUI Y stays as typed; the solver flips Y only in Brainsight mode.
        kargs['bFlipSteeringY']=bool(self._mainApp.Config.get('bInUseWithBrainsight'))
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
