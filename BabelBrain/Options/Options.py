# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QDialog,QFileDialog,QMessageBox
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


class AdvanceOptions(QDialog):
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

        self.defaultValues = defaultValues
        kargs={}
        for k in self.defaultValues:
            kargs[k]=currentConfig[k]

        self.SetValues(**kargs)

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)   
        
    def SetValues(self,
                 ElastixOptimizer='AdaptiveStochasticGradientDescent',
                 bForceUseBlender=False,
                 bApplyBOXFOV=False,
                 FOVDiameter=200.0,
                 FOVLength=400.0,
                 TrabecularProportion=0.8):
        
        sel=self.ui.ElastixOptimizercomboBox.findText(ElastixOptimizer)
        if sel==-1:
            raise ValueError('The elastix optimizer is not available in the GUI -'+ElastixOptimizer )
        
        self.ui.ManualFOVcheckBox.toggled.connect(self.EnableManualFOV)
        
        self.ui.ElastixOptimizercomboBox.setCurrentIndex(sel)
        self.ui.ForceBlendercheckBox.setChecked(bForceUseBlender)
        self.ui.ManualFOVcheckBox.setChecked(bApplyBOXFOV)
        self.ui.FOVDiameterSpinBox.setValue(FOVDiameter)
        self.ui.FOVLengthSpinBox.setValue(FOVLength)
        self.ui.TrabecularProportionSpinBox.setValue(TrabecularProportion)

    @Slot()
    def ResetToDefaults(self):
        self.SetValues(**self.defaultValues)

    @Slot()
    def EnableManualFOV(self,value):
        self.ui.grpManualFOV.setEnabled(value)
        
    @Slot()
    def Continue(self):
        self.accept()

    @Slot()
    def Cancel(self):
        self.done(-1)
         
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    widget = AdvanceOptions()
    widget.show()
    sys.exit(app.exec())
