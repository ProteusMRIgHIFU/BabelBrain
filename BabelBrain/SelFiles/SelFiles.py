# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QDialog,QFileDialog,QMessageBox
from PySide6.QtCore import Slot

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_Dialog

import os


class SelFiles(QDialog):
    def __init__(self, parent=None,Trajectory='',T1W='',SimbNIBS=''):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Select input file data...")
        self.ui.SelTrajectorypushButton.clicked.connect(self.SelectTrajectory)
        self.ui.SelT1WpushButton.clicked.connect(self.SelectT1W)
        self.ui.SelSimbNIBSpushButton.clicked.connect(self.SelectSimbNIBS)
        self.ui.SelTProfilepushButton.clicked.connect(self.SelectThermalProfile)
        self.ui.ContinuepushButton.clicked.connect(self.Continue)

        if len(Trajectory)>0:
            self.ui.TrajectorylineEdit.setText(Trajectory)
        if len(T1W)>0:
            self.ui.T1WlineEdit.setText(T1W)
        if len(SimbNIBS)>0:
            self.ui.SimbNIBSlineEdit.setText(T1W)

    @Slot()
    def SelectTrajectory(self):
        fTraj=QFileDialog.getOpenFileName(self,
            "Select trajectory", os.getcwd(), "Trajectory (*.txt)")[0]
        if len(fTraj)>0:
            self.ui.TrajectorylineEdit.setText(fTraj)

    @Slot()
    def SelectT1W(self):
        fT1W=QFileDialog.getOpenFileName(self,
            "Select T1W", os.getcwd(), "Nifti (*.nii *.nii.gz)")[0]
        if len(fT1W)>0:
            self.ui.T1WlineEdit.setText(fT1W)

    @Slot()
    def SelectThermalProfile(self):
        fThermalProfile=QFileDialog.getOpenFileName(self,"Select thermal profile",os.getcwd(),"yaml (*.yaml)")[0]
        if len(fThermalProfile)>0:
            print('fThermalProfile',fThermalProfile)
            self.ui.ThermalProfilelineEdit.setText(fThermalProfile)

    @Slot()
    def SelectSimbNIBS(self):
        fSimbNIBS=QFileDialog.getExistingDirectory(self,"Select SimbNIBS directory",
                    os.getcwd())
        self.ui.SimbNIBSlineEdit.setText(fSimbNIBS)


    @Slot()
    def Continue(self):
        if not os.path.isfile(self.ui.TrajectorylineEdit.text()) or\
           not os.path.isfile(self.ui.T1WlineEdit.text()) or\
           not os.path.isdir(self.ui.SimbNIBSlineEdit.text()) or\
           not os.path.isfile(self.ui.ThermalProfilelineEdit.text()) :
            msgBox = QMessageBox()
            msgBox.setText("Please indicate valid entries")
            msgBox.exec()
        else:
            self.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = SelFiles()
    widget.show()
    sys.exit(app.exec())
