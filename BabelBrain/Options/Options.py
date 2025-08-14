# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QDialog,QFileDialog,QStyle,QMessageBox
from PySide6.QtGui import QValidator
from PySide6.QtCore import Slot, Qt,QTimer

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_Dialog
import platform
import os
from pathlib import Path

from multiprocessing import Process,Queue
import time


from Calibration.TxCalibration import RUN_FITTING_Process 
from ClockDialog import ClockDialog
from CreateSingleVoxelMask import create_single_voxel_mask
from ConvMatTransform import (
    ReadTrajectoryBrainsight,
    itk_to_BSight,
    read_itk_affine_transform,
    templateBSight,
    BSight_to_itk,
    templateSlicer
)

import yaml
import glob
import subprocess
import traceback
from functools import partial
from scipy.io import loadmat

_IS_MAC = platform.system() == 'Darwin'


def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.join(os.path.split(Path(__file__))[0],'..')

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = os.path.join(Path(__file__).parent,'..')

    return bundle_dir

def connect_folder_button(parent,button, line_edit, title):
    """Helper to connect a button to folder selection behavior."""
    button.clicked.connect(partial(select_folder, parent,line_edit, title))
    button.setIcon(button.style().standardIcon(QStyle.SP_DirOpenIcon))

def connect_file_button(parent,button, line_edit, title,filemask):
    """Helper to connect a button to file selection behavior."""
    button.clicked.connect(partial(select_file, parent,line_edit, title,filemask))
    button.setIcon(button.style().standardIcon(QStyle.SP_FileIcon))

@Slot()
def select_folder(parent,line_edit, title):
    """Generic folder selection slot."""
    bdir = line_edit.text()
    if not os.path.isdir(bdir):
        bdir = os.getcwd()

    folder = QFileDialog.getExistingDirectory(parent, title, bdir)
    if folder:
        line_edit.setText(folder)
        line_edit.setCursorPosition(len(folder))

@Slot()
def select_file(parent,line_edit, title,filemask):
    """Generic file selection slot."""
    bdir = line_edit.text()
    if not os.path.isdir(bdir):
        bdir = os.getcwd()

    file = QFileDialog.getOpenFileName(parent, title, bdir, filemask)[0]
    if len(file)>0:
        line_edit.setText(file)
        line_edit.setCursorPosition(len(file))

class PlanTUSTxConfig(object):
    def __init__(self, max_distance, 
                 min_distance, 
                 transducer_diameter, 
                 max_angle, 
                 plane_offset,
                 additional_offset, 
                 focal_distance_list, 
                 flhm_list,
                 IDTarget="",
                 fsl_path="/Users/spichardo/fsl/share/fsl/bin",
                 connectome_path="/Applications/wb_view.app/Contents/usr/bin",
                 freesurfer_path="/Applications/freesurfer/7.4.1/bin"):

        # Maximum and minimum focal depth of transducer (in mm)
        self.max_distance = max_distance
        self.min_distance = min_distance

        # Aperture diameter (in mm)
        self.transducer_diameter = transducer_diameter

        # Maximum allowed angle for tilting of TUS transducer (in degrees)
        self.max_angle = max_angle

        # Offset between radiating surface and exit plane of transducer (in mm)
        self.plane_offset = plane_offset

        # Additional offset between skin and exit plane of transducer (in mm;
        # e.g., due to addtional gel/silicone pad)
        self.additional_offset = additional_offset

        # Focal distance and corresponding FLHM values (both in mm) according to, e.g.,
        # calibration report
        self.focal_distance_list = focal_distance_list
        self.flhm_list = flhm_list
        self.fsl_path = fsl_path
        self.connectome_path = connectome_path
        self.freesurfer_path = freesurfer_path
        self.IDTarget = IDTarget

    def ExportYAML(self,fname):
        txconfig = {
            "max_distance": self.max_distance,
            "min_distance": self.min_distance,
            "transducer_diameter": self.transducer_diameter,
            "max_angle": self.max_angle,
            "plane_offset": self.plane_offset,
            "additional_offset": self.additional_offset,
            "focal_distance_list": self.focal_distance_list,
            "flhm_list": self.flhm_list,
            "fsl_path": self.fsl_path,
            "connectome_path": self.connectome_path,
            "freesurfer_path": self.freesurfer_path,
            "IDTarget": self.IDTarget
        }

        with open(fname, "w") as file:
            yaml.dump(txconfig, file)


class OptionalParams(object):
    def __init__(self,AllTransducers):
        self._DefaultAdvanced={}
        
        self._DefaultAdvanced['bApplyBOXFOV']=False
        self._DefaultAdvanced['FOVDiameter']=200.0
        self._DefaultAdvanced['FOVLength']=400.0
        self._DefaultAdvanced['bForceUseBlender']=False
        self._DefaultAdvanced['ElastixOptimizer']='AdaptiveStochasticGradientDescent'
        self._DefaultAdvanced['TrabecularProportion']=0.8
        self._DefaultAdvanced['CTX_500_Correction']='Original'
        self._DefaultAdvanced['CTX_250_Correction']='Original'
        self._DefaultAdvanced['DPX_500_Correction']='Original'
        self._DefaultAdvanced['PetraNPeaks']=2
        self._DefaultAdvanced['PetraMRIPeakDistance']=50
        self._DefaultAdvanced['bInvertZTE']=False
        self._DefaultAdvanced['bDisableCTMedianFilter']=False
        self._DefaultAdvanced['bGeneratePETRAHistogram']=False
        self._DefaultAdvanced['BaselineTemperature']=37.0
        self._DefaultAdvanced['PETRASlope']=-2929.6
        self._DefaultAdvanced['PETRAOffset']=3274.9
        self._DefaultAdvanced['ZTESlope']=-2085.0
        self._DefaultAdvanced['ZTEOffset']=2329.0
        self._DefaultAdvanced['bSaveStress']=False
        self._DefaultAdvanced['bSaveDisplacement']=False
        self._DefaultAdvanced['bSegmentBrainTissue']=False
        self._DefaultAdvanced['SimbNINBSRoot']='...'
        self._DefaultAdvanced['PlanTUSRoot']='...'
        self._DefaultAdvanced['FSLRoot']='...'
        self._DefaultAdvanced['ConnectomeRoot']='...'
        self._DefaultAdvanced['FreeSurferRoot']='...'
        self._DefaultAdvanced['LimitBHTEIterationsPerProcess']=100
        self._DefaultAdvanced['bForceHomogenousMedium']=False
        self._DefaultAdvanced['HomogenousMediumValues']={}
        self._DefaultAdvanced['HomogenousMediumValues']['Density']=1000.0
        self._DefaultAdvanced['HomogenousMediumValues']['LongSoS']=1500.0
        self._DefaultAdvanced['HomogenousMediumValues']['LongAtt']= 5.0
        self._DefaultAdvanced['HomogenousMediumValues']['ShearSoS'] = 0.0
        self._DefaultAdvanced['HomogenousMediumValues']['ShearAtt'] = 0.0 
        self._DefaultAdvanced['HomogenousMediumValues']['ThermalConductivity'] = 0.5
        self._DefaultAdvanced['HomogenousMediumValues']['SpecificHeat'] = 3583.0
        self._DefaultAdvanced['HomogenousMediumValues']['Perfusion'] = 555.0
        self._DefaultAdvanced['HomogenousMediumValues']['Absorption'] = 0.85
        self._DefaultAdvanced['HomogenousMediumValues']['InitTemperature'] = 37.0
        self._DefaultAdvanced['bForceNoAbsorptionSkullScalp']=False
        self._DefaultAdvanced['TxOptimizedWeights']={}
        for tx in AllTransducers:
            self._DefaultAdvanced['TxOptimizedWeights'][tx]=''
        for k,v in self._DefaultAdvanced.items():
            setattr(self,k,v)

    def keys(self):
        return list(self._DefaultAdvanced.keys())
class AdvancedOptions(QDialog):
    def __init__(self,
                 currentConfig,
                 TxConfig,
                 defaultValues,
                 AllTransducers,
                parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Advanced Options")
        self.ui.ContinuepushButton.clicked.connect(self.Continue)
        self.ui.CancelpushButton.clicked.connect(self.Cancel)
        self.ui.ResetpushButton.clicked.connect(self.ResetToDefaults)
        self.ui.tabWidget.setCurrentIndex(0)
        self._TxConfig = TxConfig
        self._currentConfig = currentConfig


        self.defaultValues = defaultValues
        curvalues = OptionalParams(AllTransducers)
        self._AllTransducers = AllTransducers
        self._TxSystem = currentConfig['TxSystem']
        for k in defaultValues.keys():
            setattr(curvalues,k,currentConfig[k])
        self.SetValues(curvalues)

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        buttons = [
                (self.ui.SimNIBSRootpushButton, self.ui.SimbNINBSRootlineEdit, "Select SimNIBS Root Folder"),
                (self.ui.PlanTUSRootpushButton, self.ui.PlanTUSRootlineEdit, "Select PlanTUS Root Folder"),
                (self.ui.ConnectomeRootpushButton, self.ui.ConnectomeRootlineEdit, "Select Connectome Root Folder"),
                (self.ui.FreeSurferRootpushButton, self.ui.FreeSurferRootlineEdit, "Select FreeSurfer Root Folder"),
                (self.ui.FSLRootpushButton, self.ui.FSLRootlineEdit, "Select FSL Root Folder")
            ]

        for button, line_edit, title in buttons:
            connect_folder_button(self,button, line_edit, title)
            
        buttons = [
                (self.ui.TxOptimizedWeightspushButton, self.ui.TxOptimizedWeightsLineEdit, "Select Tx Optimized Weights File","HDF5 (*.h5 *.hdf5)"),
                (self.ui.YAMLCalibrationpushButton, self.ui.YAMLCalibrationLineEdit, "Select YAML file with calibration input fields","YAML files (*.yaml *.yml)"),
               ]
        for button, line_edit, title, mask in buttons:
            connect_file_button(self,button, line_edit, title, mask)
    
        self.ui.RUNPlanTUSpushButton.clicked.connect(self.RUNPlanTUS)

        self.ui.ExecuteCalibrationButton.clicked.connect(self.ExecuteCalibration)

        self.CalQueue = None
        self.CalProcess = None
        self.Caltimer = QTimer(self)
        self.Caltimer.timeout.connect(self.check_queue)
        self.CaltimerTUSPlan = QTimer(self)
        self.CaltimerTUSPlan.timeout.connect(self.check_queue_TUSPlan)
        self._WorkingDialog = ClockDialog(self)
        

    @Slot()
    def ExecuteCalibration(self):
        """Execute the calibration with the current parameters"""
        if not self.ui.YAMLCalibrationLineEdit.text():
            msgBox = QMessageBox()
            msgBox.setText("Please select the YAML file with calibration input fields.")
            msgBox.exec()
            self.ui.ExcelAcousticProfilesLineEdit.setFocus()
            return
        if not os.path.isfile(self.ui.YAMLCalibrationLineEdit.text()):
            msgBox = QMessageBox()
            msgBox.setText("The indicated YAML file with the with calibration input fields does not exist.")
            msgBox.exec()
            self.ui.ExcelAcousticProfilesLineEdit.setFocus()
            return
        
        self.RUN_FITTING_Parallel(self._TxConfig,
              self.ui.YAMLCalibrationLineEdit.text(),
              deviceName=self._currentConfig['ComputingDevice'],
              COMPUTING_BACKEND=self._currentConfig['ComputingBackend'])

    def RUN_FITTING_Parallel(self,TxConfig, YAMLConfigFilename, deviceName='M3', COMPUTING_BACKEND=3):
        """
        Run the fitting process in parallel using multiprocessing.
        """
        queue=Queue()
        self.CalQueue=queue
        fieldWorkerProcess = Process(target=RUN_FITTING_Process, 
                                            args=(queue,
                                                TxConfig,
                                                YAMLConfigFilename,
                                                deviceName,
                                                COMPUTING_BACKEND))
        
        self.CalProcess=fieldWorkerProcess
        self.T0Cal=time.time()
        fieldWorkerProcess.start()     
        self.Caltimer.start(100)
        mainWindowCenter = self.geometry().center()

        self._WorkingDialog.move(
            mainWindowCenter.x() - 50,
            mainWindowCenter.y() - 50
        )
        self._WorkingDialog.show()
        self.setEnabled(False)

    def check_queue(self):
            
        # progress.
        
        bNoError=True
        bDone=False
        SSINow=None
        while self.CalQueue and not self.CalQueue.empty():
            cMsg=self.CalQueue.get()
            if type(cMsg) is str:
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg:
                    bNoError=False
                    self.Caltimer.stop()
                    self.CalProcess.join()
                    bDone=True
            else:
                assert(type(cMsg) is dict)
                SSINow=cMsg['SSINow']
                SSI_BFGS=cMsg['SSI_BFGS']
                self.Caltimer.stop()
                self.CalProcess.join()
                bDone=True
                
        if bDone:
            self.setEnabled(True)
            if bNoError:
                TEnd=time.time()
                TotalTime = TEnd-self.T0Cal
                print('Total time',TotalTime)
                print("*"*40)
                print("*"*5+" DONE Calibration.")
                print("*"*40)
                self._WorkingDialog.hide()
                msgBox = QMessageBox()
                msgBox.setText("Calibration executed with the following results:\n"+
                            "SSI uncorrected: {:4.3f}\nSSI after fitting: {:4.3f}".format(SSINow,SSI_BFGS)+
                                "\n\nPLEASE check plots in output directory specied in YAML file")
                msgBox.exec()
            else:
                print("*"*40)
                print("*"*5+" Error in execution of the calibration process.")
                print("*"*40)    
        
    def SetValues(self,values):

        sel=self.ui.ElastixOptimizercomboBox.findText(values.ElastixOptimizer)
        if sel==-1:
            raise ValueError('The elastix optimizer is not available in the GUI -'+values.ElastixOptimizer )
        
        self.ui.ManualFOVcheckBox.toggled.connect(self.EnableManualFOV)
        
        self.ui.ElastixOptimizercomboBox.setCurrentIndex(sel)
        self.ui.ForceBlendercheckBox.setChecked(values.bForceUseBlender)
        self.ui.ManualFOVcheckBox.setChecked(values.bApplyBOXFOV)
        self.ui.FOVDiameterSpinBox.setValue(values.FOVDiameter)
        self.ui.FOVLengthSpinBox.setValue(values.FOVLength)
        self.ui.TrabecularProportionSpinBox.setValue(values.TrabecularProportion)
        self.ui.PetraNPeaksSpinBox.setValue(values.PetraNPeaks)
        self.ui.PetraMRIPeakDistancespinBox.setValue(values.PetraMRIPeakDistance)
        self.ui.InvertZTEcheckBox.setChecked(values.bInvertZTE)
        self.ui.DisableCTMedianFiltercheckBox.setChecked(values.bDisableCTMedianFilter)
        self.ui.GeneratePETRAHistogramcheckBox.setChecked(values.bGeneratePETRAHistogram)
        self.ui.BaselineTemperatureSpinBox.setValue(values.BaselineTemperature)
        self.ui.LimitBHTEIterationsPerProcessSpinBox.setValue(values.LimitBHTEIterationsPerProcess)
        self.ui.PETRASlopeSpinBox.setValue(values.PETRASlope)
        self.ui.PETRAOffsetSpinBox.setValue(values.PETRAOffset)
        self.ui.ZTESlopeSpinBox.setValue(values.ZTESlope)
        self.ui.ZTEOffsetSpinBox.setValue(values.ZTEOffset)
        self.ui.SaveStresscheckBox.setChecked(values.bSaveStress)
        self.ui.SaveDisplacementcheckBox.setChecked(values.bSaveDisplacement)
        self.ui.SegmentBrainTissuecheckBox.setChecked(values.bSegmentBrainTissue)
        self.ui.SimbNINBSRootlineEdit.setText(values.SimbNINBSRoot)
        self.ui.SimbNINBSRootlineEdit.setCursorPosition(len(values.SimbNINBSRoot))
        self.ui.PlanTUSRootlineEdit.setText(values.PlanTUSRoot)
        self.ui.PlanTUSRootlineEdit.setCursorPosition(len(values.PlanTUSRoot))
        self.ui.FSLRootlineEdit.setText(values.FSLRoot)
        self.ui.FSLRootlineEdit.setCursorPosition(len(values.FSLRoot))
        self.ui.ConnectomeRootlineEdit.setText(values.ConnectomeRoot)
        self.ui.ConnectomeRootlineEdit.setCursorPosition(len(values.ConnectomeRoot))
        self.ui.FreeSurferRootlineEdit.setText(values.FreeSurferRoot)
        self.ui.FreeSurferRootlineEdit.setCursorPosition(len(values.FreeSurferRoot))

        # sel=self.ui.CTX500CorrectioncomboBox.findText(values.CTX_500_Correction)
        # if sel==-1:
        #     raise ValueError('The CTX 500 correction choice is not available in the GUI -'+values.CTX_500_Correction )
        # self.ui.CTX500CorrectioncomboBox.setCurrentIndex(sel)

        self.ui.bForceHomogenousMediumcheckBox.setChecked(values.bForceHomogenousMedium)
        self.ui.HomogenousDensitySpinBox.setValue(values.HomogenousMediumValues['Density'])
        self.ui.HomogenousLSoSSpinBox.setValue(values.HomogenousMediumValues['LongSoS'])
        self.ui.HomogenousLAttSpinBox.setValue(values.HomogenousMediumValues['LongAtt'])
        self.ui.HomogenousSSoSSpinBox.setValue(values.HomogenousMediumValues['ShearSoS'])
        self.ui.HomogenousSAttSpinBox.setValue(values.HomogenousMediumValues['ShearAtt'])
        self.ui.HomogenousThermCondSpinBox.setValue(values.HomogenousMediumValues['ThermalConductivity'])
        self.ui.HomogenousSpecHeatSpinBox.setValue(values.HomogenousMediumValues['SpecificHeat'])
        self.ui.HomogenousPerfusionSpinBox.setValue(values.HomogenousMediumValues['Perfusion'])
        self.ui.HomogenousAbsorptionSpinBox.setValue(values.HomogenousMediumValues['Absorption'])
        self.ui.HomogenousInitTempSpinBox.setValue(values.HomogenousMediumValues['InitTemperature'])
        self.ui.bForceNoAbsorptionSkullScalpcheckBox.setChecked(values.bForceNoAbsorptionSkullScalp)
        self.ui.TxWeightLabel.setText("Optimized Weights for Transducer: " +  self._TxSystem)
        self.ui.TxOptimizedWeightsLineEdit.setText(values.TxOptimizedWeights[self._TxSystem])
        
    @Slot()
    def ResetToDefaults(self):
        self.SetValues(self.defaultValues)

    @Slot()
    def EnableManualFOV(self,value):
        self.ui.grpManualFOV.setEnabled(value)
        
    @Slot()
    def Continue(self):
        self.NewValues=OptionalParams(self._AllTransducers)

        self.NewValues.FOVDiameter=self.ui.FOVDiameterSpinBox.value()
        self.NewValues.FOVLength=self.ui.FOVLengthSpinBox.value()
        self.NewValues.bForceUseBlender=self.ui.ForceBlendercheckBox.isChecked()
        self.NewValues.ElastixOptimizer=self.ui.ElastixOptimizercomboBox.currentText()
        self.NewValues.TrabecularProportion=self.ui.TrabecularProportionSpinBox.value()
        # self.NewValues.CTX_500_Correction=self.ui.CTX500CorrectioncomboBox.currentText()
        self.NewValues.PetraNPeaks=self.ui.PetraNPeaksSpinBox.value()
        self.NewValues.PetraMRIPeakDistance=self.ui.PetraMRIPeakDistancespinBox.value()
        self.NewValues.bInvertZTE=self.ui.InvertZTEcheckBox.isChecked()
        self.NewValues.bDisableCTMedianFilter=self.ui.DisableCTMedianFiltercheckBox.isChecked()
        self.NewValues.bGeneratePETRAHistogram=self.ui.GeneratePETRAHistogramcheckBox.isChecked()
        self.NewValues.BaselineTemperature=self.ui.BaselineTemperatureSpinBox.value()
        self.NewValues.LimitBHTEIterationsPerProcess=self.ui.LimitBHTEIterationsPerProcessSpinBox.value()
        self.NewValues.PETRASlope=self.ui.PETRASlopeSpinBox.value()
        self.NewValues.PETRAOffset=self.ui.PETRAOffsetSpinBox.value()
        self.NewValues.ZTESlope=self.ui.ZTESlopeSpinBox.value()
        self.NewValues.ZTEOffset=self.ui.ZTEOffsetSpinBox.value()
        self.NewValues.bSaveStress=self.ui.SaveStresscheckBox.isChecked()
        self.NewValues.bSaveDisplacement=self.ui.SaveDisplacementcheckBox.isChecked()
        self.NewValues.bSegmentBrainTissue=self.ui.SegmentBrainTissuecheckBox.isChecked()
        self.NewValues.SimbNINBSRoot=self.ui.SimbNINBSRootlineEdit.text()
        self.NewValues.PlanTUSRoot=self.ui.PlanTUSRootlineEdit.text()
        self.NewValues.FSLRoot=self.ui.FSLRootlineEdit.text()
        self.NewValues.ConnectomeRoot=self.ui.ConnectomeRootlineEdit.text()
        self.NewValues.FreeSurferRoot=self.ui.FreeSurferRootlineEdit.text()
        if self.NewValues.bSegmentBrainTissue:
            if not os.path.isdir(self.NewValues.SimbNINBSRoot):
                msgBox = QMessageBox()
                msgBox.setDetailedText("'The SimNIBS root folder does not exist -'+self.NewValues.SimbNINBSRoot")
                msgBox.exec()
                return
                
        self.NewValues.bForceHomogenousMedium=self.ui.bForceHomogenousMediumcheckBox.isChecked()
        
        self.NewValues.HomogenousMediumValues['Density']             = self.ui.HomogenousDensitySpinBox.value()        
        self.NewValues.HomogenousMediumValues['LongSoS']             = self.ui.HomogenousLSoSSpinBox.value()      
        self.NewValues.HomogenousMediumValues['LongAtt']             = self.ui.HomogenousLAttSpinBox.value()      
        self.NewValues.HomogenousMediumValues['ShearSoS']            = self.ui.HomogenousSSoSSpinBox.value()      
        self.NewValues.HomogenousMediumValues['ShearAtt']            = self.ui.HomogenousSAttSpinBox.value()      
        self.NewValues.HomogenousMediumValues['ThermalConductivity'] = self.ui.HomogenousThermCondSpinBox.value() 
        self.NewValues.HomogenousMediumValues['SpecificHeat']        = self.ui.HomogenousSpecHeatSpinBox.value()  
        self.NewValues.HomogenousMediumValues['Perfusion']           = self.ui.HomogenousPerfusionSpinBox.value() 
        self.NewValues.HomogenousMediumValues['Absorption']          = self.ui.HomogenousAbsorptionSpinBox.value()
        self.NewValues.HomogenousMediumValues['InitTemperature']     = self.ui.HomogenousInitTempSpinBox.value()  
        self.NewValues.bForceNoAbsorptionSkullScalp=self.ui.bForceNoAbsorptionSkullScalpcheckBox.isChecked()
        self.NewValues.TxOptimizedWeights[self._TxSystem] = self.ui.TxOptimizedWeightsLineEdit.text()

        self.accept()

    @Slot()
    def Cancel(self):
        self.done(-1)

    @Slot()
    def RUNPlanTUS(self):
        '''
        Run the external PlanTUS script
        '''
        MainApp=self.parent()
        BabelTxConfig=MainApp.AcSim.Config
        SelFreq=MainApp.Widget.USMaskkHzDropDown.property('UserData')
        TrajectoryType=MainApp.Config['TrajectoryType']
        Mat4Trajectory=MainApp.Config['Mat4Trajectory']

        PlanTUSRoot=self.ui.PlanTUSRootlineEdit.text()
        SimbNINBSRoot=self.ui.SimbNINBSRootlineEdit.text()
        FSLRoot=self.ui.FSLRootlineEdit.text()
        ConnectomeRoot=self.ui.ConnectomeRootlineEdit.text()
        FreeSurferRoot=self.ui.FreeSurferRootlineEdit.text()

        if TrajectoryType =='brainsight':
            RMat=ReadTrajectoryBrainsight(Mat4Trajectory)
        else:
            inMat=read_itk_affine_transform(Mat4Trajectory)
            RMat = itk_to_BSight(inMat)

        #we will reuse to recover the center of the trajectory
        self._RMat = RMat

        # Create a new PlanTUSTxConfig object with the current values
        plan_tus_config = PlanTUSTxConfig(
            transducer_diameter=BabelTxConfig['TxDiam']*1e3,
            min_distance=BabelTxConfig['MinimalTPODistance']*1e3,
            max_distance=BabelTxConfig['MaximalTPODistance']*1e3,
            max_angle=10.0, #we keep it constant for the time being
            plane_offset=(BabelTxConfig['FocalLength']-BabelTxConfig['NaturalOutPlaneDistance'])*1e3,
            additional_offset=MainApp.AcSim.Widget.SkinDistanceSpinBox.value(),
            focal_distance_list=BabelTxConfig['PlanTUS'][SelFreq]['FocalDistanceList'],
            flhm_list=BabelTxConfig['PlanTUS'][SelFreq]['FHMLList'],
            IDTarget=MainApp.Config['ID'],
            fsl_path=FSLRoot,
            connectome_path=ConnectomeRoot,
            freesurfer_path=FreeSurferRoot
        )
  

        t1Path=MainApp.Config['T1W']

        basepath=os.path.split(t1Path)[0]
        TxConfigName = basepath + os.sep + "PlanTUSTxConfig.yaml"
        # Export the configuration to a YAML file
        plan_tus_config.ExportYAML(TxConfigName)
        
        mshPath=glob.glob(MainApp.Config['simbnibs_path'] + os.sep + "*.msh")[0]
        maskPath=t1Path.replace('.nii.gz','_PlanTUSMask.nii.gz')

        create_single_voxel_mask(t1Path, RMat[:3,3], maskPath)

        scriptbase=os.path.join(resource_path(),"ExternalBin"+os.sep+"PlanTUS"+os.sep)
        queue=Queue()
        self.CalQueue=queue

        fieldWorkerProcess = Process(target=RunPlanTUSBackground, 
                                            args=(queue,
                                                scriptbase,
                                                SimbNINBSRoot,
                                                PlanTUSRoot,
                                                t1Path,
                                                mshPath,
                                                maskPath,
                                                TxConfigName))
        
        self.CalProcess=fieldWorkerProcess
        self.T0Cal=time.time()
        fieldWorkerProcess.start()     
        self.CaltimerTUSPlan.start(100)
        mainWindowCenter = self.geometry().center()

        self._WorkingDialog.move(
            mainWindowCenter.x() - 50,
            mainWindowCenter.y() - 50
        )
        self._WorkingDialog.show()
        self.setEnabled(False)
        

    def check_queue_TUSPlan(self):

        # progress.
        
        bNoError=True
        bDone=False
        while self.CalQueue and not self.CalQueue.empty():
            cMsg=self.CalQueue.get()
            if type(cMsg) is str:
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg\
                   or '--Babel-Brain-Success' in cMsg:
                    if '--Babel-Brain-Low-Error' in cMsg:
                        bNoError=False
                    self.CaltimerTUSPlan.stop()
                    self.CalProcess.join()
                    bDone=True
                
        if bDone:
            self.setEnabled(True)
            self._WorkingDialog.hide()
            if bNoError:
                TEnd=time.time()
                TotalTime = TEnd-self.T0Cal
                print('Total time',TotalTime)
                print("*"*40)
                print("*"*5+" DONE PlanTUS.")
                print("*"*40)
                t1Path=self.parent().Config['T1W']
                basepath=os.path.split(t1Path)[0]+os.sep+'PlanTUS'

                #we look for new trajectory files
                trajFiles=glob.glob(basepath+os.sep+'**'+os.sep+'*Localite.mat',recursive=True)
                if len(trajFiles)>0:
                    for trajFile in trajFiles:
                        id = self.parent().Config['ID']+'_PlanTUS'
                        transform=loadmat(trajFile)['position_matrix']
                        TT=transform.copy()
                        # we need to convert the transform to the correct format
                        TT[:3,0] = -transform[0:3,1]
                        TT[:3,1] = transform[0:3,2] 
                        TT[:3,2] = -transform[0:3,0]
                        transform=TT 
                        print("Found trajectory file:", trajFile)
                        # we will reuse to recover the center of the trajectory
                        outString=templateBSight.format(m0n0=transform[0,0],
                                m0n1=transform[1,0],
                                m0n2=transform[2,0],
                                m1n0=transform[0,1],
                                m1n1=transform[1,1],
                                m1n2=transform[2,1],
                                m2n0=transform[0,2],
                                m2n1=transform[1,2],
                                m2n2=transform[2,2],
                                X=self._RMat[0,3],
                                Y=self._RMat[1,3],
                                Z=self._RMat[2,3],
                                name=id)
                        foutnameBSight = trajFile.split('Localite.mat')[0] + 'BSight.txt'
                        with open(foutnameBSight, 'w') as f:
                            f.write(outString)

                        transform = BSight_to_itk(transform)
                        outString=templateSlicer.format(m0n0=transform[0,0],
                                        m0n1=transform[1,0],
                                        m0n2=transform[2,0],
                                        m1n0=transform[0,1],
                                        m1n1=transform[1,1],
                                        m1n2=transform[2,1],
                                        m2n0=transform[0,2],
                                        m2n1=transform[1,2],
                                        m2n2=transform[2,2],
                                        X=self._RMat[0,3],
                                        Y=self._RMat[1,3],
                                        Z=self._RMat[2,3])
                        foutnameSlicer = trajFile.split('Localite.mat')[0] + 'Slicer.txt'
                        with open(foutnameSlicer, 'w') as f:
                            f.write(outString)

                    ret = QMessageBox.question(self,'', "Do you want to use the\n PlanTUS to update the trajectory? ",QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        TrajectoryType=self.parent().Config['TrajectoryType']
                        if TrajectoryType =='brainsight':
                            ext='*BSight.txt'
                        else:
                            ext='*Slicer.txt'
                        fname = QFileDialog.getOpenFileName(self, "Select txt file with calibration input fields",basepath, "Text files ("+ext+")")[0]
                        if len(fname)>0:
                            self.parent().Config['Mat4Trajectory'] = fname
            else:
                print("*"*40)
                print("*"*5+" Error in execution of PlanTUS.")
                print("*"*40)


def RunPlanTUSBackground(queue,
                        scriptbase,
                        SimbNINBSRoot,
                        PlanTUSRoot,
                        t1Path,
                        mshPath,
                        maskPath,
                        TxConfigName):
    class InOutputWrapper(object):
       
        def __init__(self, queue, stdout=True):
            self.queue=queue
            if stdout:
                self._stream = sys.stdout
                sys.stdout = self
            else:
                self._stream = sys.stderr
                sys.stderr = self
            self._stdout = stdout

        def write(self, text):
            self.queue.put(text)

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def __del__(self):
            try:
                if self._stdout:
                    sys.stdout = self._stream
                else:
                    sys.stderr = self._stream
            except AttributeError:
                pass

    stdout = InOutputWrapper(queue,True)
  
    try:
        if sys.platform == 'linux' or _IS_MAC:
            if sys.platform == 'linux':
                shell='bash'
                path_script =scriptbase+"run_linux.sh"
            elif _IS_MAC:
                shell='zsh'
                path_script = scriptbase+"run_mac.sh"

            print("Starting PlanTUS")
            if _IS_MAC:
                cmd ='source "'+path_script + '" "' + SimbNINBSRoot + '" "' + PlanTUSRoot + '" "' + t1Path + '" "' + mshPath +'" "' + maskPath + '" "'+TxConfigName+'"'
                print(cmd)
                result = os.system(cmd)
            else:
                result = subprocess.run(
                        [shell,
                        path_script,
                        SimbNINBSRoot,
                        PlanTUSRoot,
                        t1Path,
                        mshPath,
                        maskPath,
                        TxConfigName], capture_output=True, text=True
                )
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                result=result.returncode 
        else:
            path_script = os.path.join(resource_path(),"ExternalBin/PlanTUS/run_win.bat")
            
            print("Starting PlanTUS")
            result = subprocess.run(
                    [path_script,
                    SimbNINBSRoot,
                    PlanTUSRoot,
                    t1Path,
                    mshPath,
                    maskPath,
                    TxConfigName,
                    ], capture_output=True, text=True,shell=True,
            )
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            result=result.returncode 
        print("PlanTUS Finished")
        print("--Babel-Brain-Success")
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))
    
