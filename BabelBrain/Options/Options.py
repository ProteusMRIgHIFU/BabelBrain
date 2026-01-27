# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QDialog,QFileDialog,QStyle,QMessageBox
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
import yaml
from glob import glob

from functools import partial

from Calibration.TxCalibration import RUN_FITTING_Process 
from Calibration.ViewResults import PlotViewerCalibration
from ClockDialog import ClockDialog

from PlanTUSViewer.RunPlanTUS import RUN_PLAN_TUS
from BabelViscoFDTD.H5pySimple import SaveToH5py, ReadFromH5py



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
        return True
    return False


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
        self._DefaultAdvanced['bExtractAirRegions']=True
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
                (self.ui.YAMLCalibrationpushButton, self.ui.YAMLCalibrationLineEdit, "Select YAML file with input fields to run calibration","YAML files (*.yaml *.yml)"),
               ]
        for button, line_edit, title, mask in buttons:
            connect_file_button(self,button, line_edit, title, mask)
        
        self.ui.TxOptimizedWeightspushButton.clicked.connect(self.VerifyCalibrationFile)
    
        self.ui.RUNPlanTUSpushButton.clicked.connect(self.RUNPlanTUS)

        TxSystem = self.parent().Config['TxSystem']

        self.ui.ExecuteCalibrationButton.clicked.connect(self.ExecuteCalibration)

        if TxSystem not in ['CTX_500', 'CTX_250', 'CTX_250_2ch', 'DPX_500', 'DPXPC_300', 'R15287', 'R15473']:
            self.ui.frameCalibration.setEnabled(False)
        self.CalQueue = None
        self.CalProcess = None
        self.Caltimer = QTimer(self)
        self.Caltimer.timeout.connect(self.check_queue)

        BabelTxConfig=self.parent().AcSim.Config
       
        if 'MinimalTPODistance' in BabelTxConfig or\
           'MinimalZSteering' in BabelTxConfig or\
           'BSonix35mm' in BabelTxConfig or\
           'DomeTx' in TxSystem: 
            self.ui.FocalLengthLabel.setEnabled(False)
            self.ui.DiameterLabel.setEnabled(False)
            self.ui.FocalLengthSpinBox.setEnabled(False)
            self.ui.DiameterSpinBox.setEnabled(False)
        if 'DomeTx' in TxSystem: 
            self.ui.SkinDistanceSpinBox.setEnabled(False)
            self.ui.RUNPlanTUSpushButton.setEnabled(False)
        if 'MinimalZSteering' in BabelTxConfig and 'FocalLength' in BabelTxConfig and TxSystem not in ['DomeTx'] :
            self.ui.DistanceTxLabel.setText('Distance Cone to\nFocus (mm)')
            self.ui.SkinDistanceSpinBox.setMinimum(self.parent().AcSim.Widget.DistanceConeToFocusSpinBox.minimum())
            self.ui.SkinDistanceSpinBox.setMaximum(self.parent().AcSim.Widget.DistanceConeToFocusSpinBox.maximum())
            self.ui.SkinDistanceSpinBox.setValue(self.parent().AcSim.Widget.DistanceConeToFocusSpinBox.value())

        self._WorkingDialog = ClockDialog(self)

    @Slot()
    def VerifyCalibrationFile(self):

        if not select_file(self.ui.TxOptimizedWeightspushButton, self.ui.TxOptimizedWeightsLineEdit, "Select Tx Optimized Weights File","HDF5 (*.h5 *.hdf5)"):
            return

        calfile=self.ui.TxOptimizedWeightsLineEdit.text()
        def ShowError(msg):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(msg)
            msgBox.exec()

        bError=False
        
        if not os.path.isfile(calfile):
            ShowError(f"Calibration file does not exist\n{calfile}")
            bError=True
        else:
            RefData={'TxConfig':self._TxConfig}
            try:
                Data=ReadFromH5py(calfile)
                for k in ['TxConfig']:
                    if Data[k]!=RefData[k]:
                            ShowError(f"Calibration file does not match current Tx definition for field {k}:\n{Data[k]} vs {RefData[k]}")
                            bError=True
            except:
                ShowError(f"Unable to load calibration file!!")
                bError=True
        if bError:
            self.ui.TxOptimizedWeightsLineEdit.setText("")

    @Slot()
    def ExecuteCalibration(self):
        """Execute the calibration with the current parameters"""
        if not self.ui.YAMLCalibrationLineEdit.text():
            msgBox = QMessageBox()
            msgBox.setText("Please select the YAML file with calibration input fields.")
            msgBox.exec()
            self.ui.YAMLCalibrationLineEdit.setFocus()
            return
        if not os.path.isfile(self.ui.YAMLCalibrationLineEdit.text()):
            msgBox = QMessageBox()
            msgBox.setText("The indicated YAML file with the with calibration input fields does not exist.")
            msgBox.exec()
            self.ui.YAMLCalibrationLineEdit.setFocus()
            return
        
        yamlFile=self.ui.YAMLCalibrationLineEdit.text()
        with open(yamlFile,'r') as f:
            inputInfo=yaml.safe_load(f)
            if inputInfo['Device']!=self._TxSystem:
                msgBox = QMessageBox()
                msgBox.setText(f"The Device field in the YAML file ({inputInfo['Device']})\ndoes not match the current transducer in BabelBrain:{self._TxSystem}.")
                msgBox.exec()
                self.ui.YAMLCalibrationLineEdit.setFocus()
                return
            
        #we delete output files
        rootnamepath=inputInfo['OutputResultsPath']
        files=sorted(glob(os.path.join(rootnamepath,'Plots-AcProfiles*.pdf')))+\
                      sorted(glob(os.path.join(rootnamepath,'Plots-Acplanes*.pdf')))+\
                      glob(os.path.join(rootnamepath,'Plots-weight.pdf'))
        for f in files:
            os.remove(f)
        
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
                yamlFile=self.ui.YAMLCalibrationLineEdit.text()
                with open(yamlFile,'r') as f:
                    inputInfo=yaml.safe_load(f)
                rootnamepath=inputInfo['OutputResultsPath']
                files=sorted(glob(os.path.join(rootnamepath,'Plots-AcProfiles*.pdf')))+\
                      sorted(glob(os.path.join(rootnamepath,'Plots-Acplanes*.pdf')))+\
                      [os.path.join(rootnamepath,'Plots-weight.pdf')]

                res=PlotViewerCalibration(files).exec()
                if res==QDialog.Accepted:
                    calfile=os.path.join(rootnamepath,'CALIBRATION.h5')
                    self.ui.TxOptimizedWeightsLineEdit.setText(calfile)
                    self.ui.TxOptimizedWeightsLineEdit.setCursorPosition(len(calfile))
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
        self.ui.ExtractAirRegionscheckBox.setChecked(values.bExtractAirRegions)
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
        if self._TxSystem in ['CTX_500', 'CTX_250', 'CTX_250_2ch', 'DPX_500', 'DPXPC_300', 'R15287', 'R15473']:
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
        self.NewValues.bExtractAirRegions=self.ui.ExtractAirRegionscheckBox.isChecked()
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
        R=RUN_PLAN_TUS(self.parent(),self)
        R.Execute()
    
