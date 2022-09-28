# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
from multiprocessing import Process,Queue

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

import os
import sys
import shutil
from datetime import datetime
import time
import yaml
from IntegrationBrainsightTW.CalculateTemperatureEffects import CalculateTemperatureEffects,GetThermalOutName
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from .CalculateThermalProcess import CalculateThermalProcess

import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return Path(__file__)

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_Thermal_SingleFocus'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

class Babel_Thermal(QWidget):
    def __init__(self,parent=None,MainApp=None):
        super(Babel_Thermal, self).__init__(parent)
        self._MainApp=MainApp
        self._ThermalResults=[]
        self.static_canvas=None
        self.DefaultConfig()
        self.load_ui()

    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(resource_path(), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget = loader.load(ui_file, self)
        ui_file.close()

        self.Widget.CalculateThermal.clicked.connect(self.RunSimulation)

        while self.Widget.SelCombinationDropDown.count()>0:
            self.Widget.SelCombinationDropDown.removeItem(0)

        for c in self.Config['AllDC_PRF_Duration']:
            self.Widget.SelCombinationDropDown.addItem('%3.1fs %3.1f%% %3.1fHz' %(c['Duration'],c['DC']*100,c['PRF']))

        self.Widget.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateThermalResults)
        self.Widget.IsppaSpinBox.valueChanged.connect(self.UpdateThermalResults)
        self.Widget.SelCombinationDropDown.setEnabled(False)
        self.Widget.SelCombinationDropDown.setEnabled(False)

    def DefaultConfig(self):
        #Specific parameters for the thermal simulation - to be configured  via a yaml

        with open(self._MainApp._ThermalProfile, 'r') as file:
            config = yaml.safe_load(file)
        print("Thermal configuration:")
        print(config)

        self.Config=config


    @Slot()
    def RunSimulation(self):
        bCalcFields=False
        BaseField=self._MainApp.AcSim._FullSolName

        PrevFiles=[]
        for combination in self.Config['AllDC_PRF_Duration']:
            ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                    combination['DC'],
                                                    self.Config['BaseIsppa'],
                                                    combination['PRF'])+'.h5'

            if os.path.isfile(ThermalName):
                PrevFiles.append(ThermalName)


        if len(PrevFiles)==len(self.Config['AllDC_PRF_Duration']):
            ret = QMessageBox.question(self,'', "Thermal sim files already exist\n" +
                                "Do you want to recalculate?\nSelect No to reload",
            QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcFields=True
        else:
            bCalcFields = True
        if bCalcFields:
            self._ThermalResults=[]

            self.thread = QThread()
            self.worker = RunThermalSim(self._MainApp,self.thread)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.UpdateThermalResults)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)

            self.thread.start()
            self._MainApp.Widget.tabWidget.setEnabled(False)
            print('thermal sim thread initiated')
        else:
            self.UpdateThermalResults()

    def NotifyError(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()

    @Slot()
    def UpdateThermalResults(self):
        self._MainApp.Widget.tabWidget.setEnabled(True)
        self._MainApp.ThermalSim.setEnabled(True)

        self.Widget.SelCombinationDropDown.setEnabled(True)
        self.Widget.SelCombinationDropDown.setEnabled(True)
        BaseField=self._MainApp.AcSim._FullSolName
        if len(self._ThermalResults)==0:
            for combination in self.Config['AllDC_PRF_Duration']:
                ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                        combination['DC'],
                                                        self.Config['BaseIsppa'],
                                                        combination['PRF'])+'.h5'

                self._ThermalResults.append(ReadFromH5py(ThermalName))


        SaveDict=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        SelIsspa=self.Widget.IsppaSpinBox.value()

        self._figIntThermalFields=Figure(figsize=(14, 12))
        if self.static_canvas is not None:
            self._layout.removeItem(self._layout.itemAt(0))
            self._layout.removeItem(self._layout.itemAt(0))
        else:
            self._layout = QVBoxLayout(self.Widget.AcField_plot1)

        self.static_canvas = FigureCanvas(self._figIntThermalFields)
        self._layout.addWidget(self.static_canvas)
        toolbar=NavigationToolbar2QT(self.static_canvas,self)
        self._layout.addWidget(toolbar)
        self._layout.addWidget(self.static_canvas)
        static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)

        xf=SaveDict['x_vec']
        zf=SaveDict['z_vec']

        SkinZ=np.array(np.where(SaveDict['MaterialMap']==1)).T.min(axis=0)[1]
        zf-=zf[SkinZ]


        IsppaRatio=SelIsspa/self.Config['BaseIsppa']
        PresRatio=np.sqrt(IsppaRatio)

        AdjustedIsspa = SelIsspa/SaveDict['RatioLosses']

        DensityMap=SaveDict['MaterialList']['Density'][SaveDict['MaterialMap']]
        SoSMap=    SaveDict['MaterialList']['SoS'][SaveDict['MaterialMap']]
        IntensityMap=SaveDict['p_map']**2/2/DensityMap/SoSMap/1e4*IsppaRatio
        Tmap=(SaveDict['MonitorSlice']-37.0)*IsppaRatio+37.0

        TargetLocation=SaveDict['TargetLocation']
        self._IntensityIm=static_ax1.imshow(IntensityMap.T,extent=[xf.min(),xf.max(),zf.max(),zf.min()],
                 cmap=plt.cm.jet,vmax=SelIsspa)
        static_ax1.plot(xf[TargetLocation[0]],zf[TargetLocation[1]],'k+',markersize=18)
        static_ax1.set_title('Isppa (W/cm$^2$)')
        plt.colorbar(self._IntensityIm,ax=static_ax1)


        XX,ZZ=np.meshgrid(xf,zf)
        static_ax1.contour(XX,ZZ,SaveDict['MaterialMap'].T,[0,1,2,3], cmap=plt.cm.gray)

        static_ax1.set_ylabel('Distance from skin (mm)')

        self._ThermalIm=static_ax2.imshow(Tmap.T,
                 extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet,vmin=37)
        static_ax2.plot(xf[TargetLocation[0]],zf[TargetLocation[1]],'k+',markersize=18)
        static_ax2.set_title('Temperature ($^{\circ}$C)')
#        static_ax2.set_yticklabels([])
        plt.colorbar(self._ThermalIm,ax=static_ax2)
        static_ax2.contour(XX,ZZ,SaveDict['MaterialMap'].T,[0,1,2,3], cmap=plt.cm.gray)


        self._figIntThermalFields.set_tight_layout(True)

        self._figIntThermalFields.set_facecolor(np.array(self.palette().color(QPalette.Window).getRgb())/255)

        DutyCycle=self.Config['AllDC_PRF_Duration'][self.Widget.SelCombinationDropDown.currentIndex()]['DC']

        self.Widget.IsppaWaterLabel.setProperty('UserData',AdjustedIsspa)
        self.Widget.IsppaWaterLabel.setText('%4.2f' % self.Widget.IsppaWaterLabel.property('UserData'))

        self.Widget.IsptaLabel.setProperty('UserData',SelIsspa*DutyCycle)
        self.Widget.IsptaLabel.setText('%4.2f' % self.Widget.IsptaLabel.property('UserData'))

        self.Widget.MILabel.setProperty('UserData',SaveDict['MI']*PresRatio)
        self.Widget.TIBrainLabel.setProperty('UserData',SaveDict['TI']*IsppaRatio)
        self.Widget.TICLabel.setProperty('UserData',SaveDict['TIC']*IsppaRatio)
        self.Widget.TISkinLabel.setProperty('UserData',SaveDict['TIS']*IsppaRatio)
        self.Widget.AdjustRASLabel.setProperty('UserData',SaveDict['AdjustmentInRAS'])
        for obj in [self.Widget.MILabel,self.Widget.TIBrainLabel,self.Widget.TICLabel,self.Widget.TISkinLabel]:
            obj.setText('%3.2f' % obj.property('UserData'))
        self.Widget.AdjustRASLabel.setText(np.array2string(self.Widget.AdjustRASLabel.property('UserData'),
                                                           formatter={'float_kind':lambda x: "%3.2f" % x}))


class RunThermalSim(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp,thread):
         super(RunThermalSim, self).__init__()
         self._mainApp=mainApp
         self._thread=thread

    def run(self):

        case=self._mainApp.AcSim._FullSolName
        print('Calculating thermal maps for configurations\n',self._mainApp.ThermalSim.Config['AllDC_PRF_Duration'])
        T0=time.time()
        kargs={}
        kargs['deviceName']=self._mainApp.Config['ComputingDevice']
        kargs['COMPUTING_BACKEND']=self._mainApp.Config['ComputingBackend']
        kargs['Isppa']=self._mainApp.ThermalSim.Config['BaseIsppa']
        kargs['sel_p']='p_amp'
        if self._mainApp.Config['TxSystem'] =='H317':
            if self._mainApp.AcSim.Widget.RefocusingcheckBox.isChecked():
                 kargs['sel_p']='p_amp_refocus'
        
       

        # Start mask generation as separate process.
        queue=Queue()
        fieldWorkerProcess = Process(target=CalculateThermalProcess, 
                                    args=(queue,case,self._mainApp.ThermalSim.Config['AllDC_PRF_Duration']),
                                    kwargs=kargs)
        fieldWorkerProcess.start()      
        # progress.
        T0=time.time()
        bNoError=True
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
            print("*"*5+" DONE thermal simulation.")
            print("*"*40)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()

if __name__ == "__main__":
    app = QApplication([])
    widget = Babel_Thermal()
    widget.show()
    sys.exit(app.exec_())
