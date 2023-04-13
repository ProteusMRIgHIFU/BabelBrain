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

class SelFiles(QDialog):
    def __init__(self, parent=None,Trajectory='',T1W='',
                    SimbNIBS='',CTType=0,CoregCT=0,CT='',
                    SimbNIBSType=0,TrajectoryType=0,
                    GPU='CPU',
                    Backend='Metal'):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        with open(os.path.join(resource_path(),'version.txt'), 'r') as f:
            version=f.readlines()[0]
        self.setWindowTitle("BabelBrain V"+version + " - Select input files ...")
        self.ui.SelTrajectorypushButton.clicked.connect(self.SelectTrajectory)
        self.ui.SelT1WpushButton.clicked.connect(self.SelectT1W)
        self.ui.SelCTpushButton.clicked.connect(self.SelectCT)
        self.ui.SelSimbNIBSpushButton.clicked.connect(self.SelectSimbNIBS)
        self.ui.SelTProfilepushButton.clicked.connect(self.SelectThermalProfile)
        self.ui.ContinuepushButton.clicked.connect(self.Continue)
        self.ui.CTTypecomboBox.currentIndexChanged.connect(self.selCTType)
        self.ui.CancelpushButton.clicked.connect(self.Cancel)

        if len(Trajectory)>0:
            self.ui.TrajectorylineEdit.setText(Trajectory)
            self.ui.TrajectorylineEdit.setCursorPosition(len(Trajectory))
        if len(T1W)>0:
            self.ui.T1WlineEdit.setText(T1W)
            self.ui.T1WlineEdit.setCursorPosition(len(T1))
        if len(SimbNIBS)>0:
            self.ui.SimbNIBSlineEdit.setText(SelectSimbNIBS)
            self.ui.SimbNIBSlineEdit.setCursorPosition(len(SelectSimbNIBS))
        if len(CT)>0:
            self.ui.CTlineEdit.setText(CT)
            self.ui.CTlineEdit.setCursorPosition(len(CT))
        self.ui.CTTypecomboBox.setCurrentIndex(CTType)
        self.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimbNIBSType)
        self.ui.TrajectoryTypecomboBox.setCurrentIndex(TrajectoryType)
        self.ui.CoregCTcomboBox.setCurrentIndex(CoregCT)

        self._GPUs=self.GetAvailableGPUs()

        if len(self._GPUs)==0: #only CPU
            msgBox = QMessageBox()
            msgBox.setText("No GPUs were detected!\BabelBrain can't run without a GPU\nfor simulations")
            msgBox.exec()
            sys.exit(0)

        for dev in self._GPUs:
            self.ui.ComputingEnginecomboBox.addItem(dev[0] + ' -- ' + dev[1])
        for sel,dev in enumerate(self._GPUs):
            if GPU in dev[0] and (GPU=='CPU' or Backend in dev[1]):
                self.ui.ComputingEnginecomboBox.setCurrentIndex(sel)
                break

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

    def SelectComputingEngine(self,GPU='CPU',Backend=''):
        for sel,dev in enumerate(self._GPUs):
            if GPU in dev[0] and (GPU=='CPU' or Backend in dev[1]):
                self.ui.ComputingEnginecomboBox.setCurrentIndex(sel)
                break
    def SelectTxSystem(self,TxSystem='CTX_500'):
        index = self.ui.TransducerTypecomboBox.findText(TxSystem)
        if index >=0:
            self.ui.TransducerTypecomboBox.setCurrentIndex(index)

    def GetSelectedComputingEngine(self):
        index = self.ui.ComputingEnginecomboBox.currentIndex()
        return self._GPUs[index]
            

    def GetAvailableGPUs(self):
        AllDevices=[]
        if 'Darwin' in platform.system():
            from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_METAL import ListDevices
            devices=ListDevices()
            print('Available Metal Devices',devices)
            for dev in devices:
                AllDevices.append([dev,'Metal'])
        else:
            #we try to import CUDA and OpenCL in Win/Linux systems, if it fails, it means some drivers are not correctly installed
            try:
                from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_CUDA import ListDevices
                devices=ListDevices()
                print('Available CUDA Devices',devices)
                for dev in devices:
                    AllDevices.append([dev,'CUDA'])
            except:
                pass
            try:
                from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_OPENCL import ListDevices
                devices=ListDevices()
                print('Available OPENCL Devices',devices)
                for dev in devices:
                    AllDevices.append([dev,'OpenCL'])
            except:
                pass 
        return AllDevices
    
    def ValidTrajectory(self):
        fTraj = self.ui.TrajectorylineEdit.text()

        if not os.path.isfile(fTraj):
            self.msgDetails = "Trajectory file was not specified"
            return False

        with open(fTraj) as f:
            lines = f.readlines()
        lines= str(lines).lower()

        if self.ui.TrajectoryTypecomboBox.currentIndex() == 0: # Brainsight
            if re.search("brainsight",lines):
                return True
            else:
                self.msgDetails = "Selected trajectory file is not a Brainsight file"
                return False
        else: # Slicer
            if re.search("(?<!bra)insight",lines): #insight, but not brainsight in text
                return True
            else:
                self.msgDetails = "Selected trajectory file is not a Slicer file"
                return False
            
    def ValidSimNIBS(self):
        folderSimNIBS = self.ui.SimbNIBSlineEdit.text()

        if not os.path.isdir(folderSimNIBS):
            self.msgDetails = "SimNIBS Directory was not specified"
            return False

        files =  os.listdir(folderSimNIBS)
        files = str(files).lower()

        if self.ui.SimbNIBSTypecomboBox.currentIndex() == 0: # Charm
            if "charm" in files:
                return True
            else:
                self.msgDetails = "Selected SimbNIBS folder was not Charm generated"
                return False
        else: # Headreco
            if "headreco" in files:
                return True
            else:
                self.msgDetails = "Selected SimbNIBS folder was not Headreco generated"
                return False

    @Slot()
    def SelectTrajectory(self):
        fTraj=QFileDialog.getOpenFileName(self,
            "Select trajectory", os.getcwd(), "Trajectory (*.txt)")[0]
        if len(fTraj)>0:
            self.ui.TrajectorylineEdit.setText(fTraj)
            self.ui.TrajectorylineEdit.setCursorPosition(len(fTraj))

    @Slot()
    def SelectT1W(self):
        fT1W=QFileDialog.getOpenFileName(self,
            "Select T1W", os.getcwd(), "Nifti (*.nii *.nii.gz)")[0]
        if len(fT1W)>0:
            self.ui.T1WlineEdit.setText(fT1W)
            self.ui.T1WlineEdit.setCursorPosition(len(T1fT1W))

    @Slot()
    def SelectCT(self):
        fCT=QFileDialog.getOpenFileName(self,
            "Select CT", os.getcwd(), "Nifti (*.nii *.nii.gz)")[0]
        if len(fCT)>0:
            self.ui.CTlineEdit.setText(fCT)
            self.ui.CTlineEdit.setCursorPosition(len(fCT))

    @Slot()
    def SelectThermalProfile(self):
        fThermalProfile=QFileDialog.getOpenFileName(self,"Select thermal profile",os.getcwd(),"yaml (*.yaml)")[0]
        if len(fThermalProfile)>0:
            print('fThermalProfile',fThermalProfile)
            self.ui.ThermalProfilelineEdit.setText(fThermalProfile)
            self.ui.fThermalProfile.setCursorPosition(len(fThermalProfile))

    @Slot()
    def SelectSimbNIBS(self):
        fSimbNIBS=QFileDialog.getExistingDirectory(self,"Select SimbNIBS directory",
                    os.getcwd())
        if len(fSimbNIBS)>0:
            self.ui.SimbNIBSlineEdit.setText(fSimbNIBS)
            self.ui.SimbNIBSlineEdit.setCursorPosition(len(fSimbNIBS))

    @Slot()
    def selCTType(self,value):
        bv = value >0
        self.ui.CTlineEdit.setEnabled(bv)
        self.ui.SelCTpushButton.setEnabled(bv)
        self.ui.CoregCTlabel.setEnabled(bv)
        self.ui.CoregCTcomboBox.setEnabled(bv)

    @Slot()
    def Continue(self):
        self.msgDetails = ""
        if not self.ValidTrajectory() or\
           not self.ValidSimNIBS() or\
           not os.path.isfile(self.ui.T1WlineEdit.text()) or\
           (self.ui.CTTypecomboBox.currentIndex()>0 and not os.path.isfile(self.ui.CTlineEdit.text())) or\
           not os.path.isfile(self.ui.ThermalProfilelineEdit.text()) :
            msgBox = QMessageBox()
            msgBox.setText("Please indicate valid entries")
            msgBox.setDetailedText(self.msgDetails)
            msgBox.exec()
        else:
            self.accept()

    @Slot()
    def Cancel(self):
        sys.exit(0)
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    widget = SelFiles()
    widget.show()
    sys.exit(app.exec())
