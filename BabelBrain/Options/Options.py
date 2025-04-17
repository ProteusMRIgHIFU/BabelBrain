# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QDialog,QFileDialog,QStyle,QMessageBox
from PySide6.QtCore import Slot, Qt

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_Dialog
import platform
import os
from pathlib import Path
import re


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

class OptionalParams(object):
    def __init__(self):
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

        for k,v in self._DefaultAdvanced.items():
            setattr(self,k,v)

    def keys(self):
        return list(self._DefaultAdvanced.keys())
class AdvancedOptions(QDialog):
    def __init__(self,
                 currentConfig,
                 defaultValues,
                parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Advanced Options")
        self.ui.ContinuepushButton.clicked.connect(self.Continue)
        self.ui.CancelpushButton.clicked.connect(self.Cancel)
        self.ui.ResetpushButton.clicked.connect(self.ResetToDefaults)
        self.ui.tabWidget.setCurrentIndex(0)

        self.defaultValues = defaultValues
        curvalues = OptionalParams()
        for k in defaultValues.keys():
            setattr(curvalues,k,currentConfig[k])
        self.SetValues(curvalues)

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)   
        
        self.ui.SimNIBSRootpushButton.clicked.connect(self.SelectSimNIBSRoot)
        self.ui.SimNIBSRootpushButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        
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
        
        sel=self.ui.CTX500CorrectioncomboBox.findText(values.CTX_500_Correction)
        if sel==-1:
            raise ValueError('The CTX 500 correction choice is not available in the GUI -'+values.CTX_500_Correction )
        self.ui.CTX500CorrectioncomboBox.setCurrentIndex(sel)

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
        
    @Slot()
    def SelectSimNIBSRoot(self):
        """Select the SimNIBS root folder"""
        bdir=self.ui.SimbNINBSRootlineEdit.text()
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select SimNIBS Root Folder",bdir)    
        
        if folder:
            self.ui.SimbNINBSRootlineEdit.setText(folder)
            self.ui.SimbNINBSRootlineEdit.setCursorPosition(len(folder))
        
    @Slot()
    def ResetToDefaults(self):
        self.SetValues(self.defaultValues)

    @Slot()
    def EnableManualFOV(self,value):
        self.ui.grpManualFOV.setEnabled(value)
        
    @Slot()
    def Continue(self):
        self.NewValues=OptionalParams()

        self.NewValues.FOVDiameter=self.ui.FOVDiameterSpinBox.value()
        self.NewValues.FOVLength=self.ui.FOVLengthSpinBox.value()
        self.NewValues.bForceUseBlender=self.ui.ForceBlendercheckBox.isChecked()
        self.NewValues.ElastixOptimizer=self.ui.ElastixOptimizercomboBox.currentText()
        self.NewValues.TrabecularProportion=self.ui.TrabecularProportionSpinBox.value()
        self.NewValues.CTX_500_Correction=self.ui.CTX500CorrectioncomboBox.currentText()
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

        self.accept()

    @Slot()
    def Cancel(self):
        self.done(-1)
         
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    widget = AdvancedOptions()
    widget.show()
    sys.exit(app.exec())
