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
from ThermalModeling.CalculateTemperatureEffects import GetThermalOutName
from BabelViscoFDTD.H5pySimple import ReadFromH5py, SaveToH5py
from .CalculateThermalProcess import CalculateThermalProcess
import pandas as pd
import platform
_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_Thermal_SingleFocus'
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

def RCoeff(Temperature):
    R = np.ones(Temperature.shape)*0.25
    R[Temperature>=43]=0.5
    return R

class Babel_Thermal(QWidget):
    def __init__(self,parent=None,MainApp=None):
        super(Babel_Thermal, self).__init__(parent)
        self._MainApp=MainApp
        self._ThermalResults=[]
        self.static_canvas=None
        self.DefaultConfig()
        self.load_ui()
        self._LastTMap=-1

    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(resource_path(), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget = loader.load(ui_file, self)
        ui_file.close()

        self.Widget.CalculateThermal.clicked.connect(self.RunSimulation)
        self.Widget.ExportSummary.clicked.connect(self.ExportSummary)

        while self.Widget.SelCombinationDropDown.count()>0:
            self.Widget.SelCombinationDropDown.removeItem(0)

        for c in self.Config['AllDC_PRF_Duration']:
            self.Widget.SelCombinationDropDown.addItem('%3.1fs-On %3.1fs-Off %3.1f%% %3.1fHz' %(c['Duration'],c['DurationOff'],c['DC']*100,c['PRF']))

        self.Widget.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateThermalResults)
        self.Widget.IsppaSpinBox.valueChanged.connect(self.UpdateThermalResults)
        self.Widget.IsppaScrollBar.valueChanged.connect(self.UpdateThermalResults)
        self.Widget.IsppaScrollBar.setEnabled(False)
        self.Widget.SelCombinationDropDown.setEnabled(False)
        self.Widget.IsppaSpinBox.setEnabled(False)

        self.Widget.LocMTB.clicked.connect(self.LocateMTB)
        self.Widget.LocMTB.setEnabled(False)
        self.Widget.LocMTC.clicked.connect(self.LocateMTC)
        self.Widget.LocMTC.setEnabled(False)
        self.Widget.LocMTS.clicked.connect(self.LocateMTS)
        self.Widget.LocMTS.setEnabled(False)

        for l in [self.Widget.label_13,self.Widget.label_14,self.Widget.label_15]:
            l.setText(l.text()+' ('+"\u2103"+'):')


    def DefaultConfig(self):
        #Specific parameters for the thermal simulation - to be configured  via a yaml

        with open(self._MainApp.Config['ThermalProfile'], 'r') as file:
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
                                                    combination['DurationOff'],
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
        
        self._bRecalculated=True
        self._ThermalResults=[]
        if bCalcFields:
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
    def UpdateThermalResults(self,bUpdatePlot=True,OverWriteIsppa=None):
        self._MainApp.Widget.tabWidget.setEnabled(True)
        self._MainApp.ThermalSim.setEnabled(True)
        self.Widget.ExportSummary.setEnabled(True)
        self.Widget.SelCombinationDropDown.setEnabled(True)
        self.Widget.IsppaScrollBar.setEnabled(True)
        self.Widget.IsppaSpinBox.setEnabled(True)
        self.Widget.LocMTS.setEnabled(True)
        self.Widget.LocMTC.setEnabled(True)
        self.Widget.LocMTB.setEnabled(True)

        BaseField=self._MainApp.AcSim._FullSolName
        if len(self._ThermalResults)==0:
            self._LastTMap=-1
            for combination in self.Config['AllDC_PRF_Duration']:
                ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                        combination['DurationOff'],
                                                        combination['DC'],
                                                        self.Config['BaseIsppa'],
                                                        combination['PRF'])+'.h5'

                self._ThermalResults.append(ReadFromH5py(ThermalName))
                if self._MainApp.Config['bUseCT']:
                    self._ThermalResults[-1]['MaterialMap'][self._ThermalResults[-1]['MaterialMap']>=3]=3
            DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
            self._xf=DataThermal['x_vec']
            self._zf=DataThermal['z_vec']
            SkinZ=np.array(np.where(DataThermal['MaterialMap']==1)).T.min(axis=0)[1]
            self._zf-=self._zf[SkinZ]
        
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]

        Loc=DataThermal['TargetLocation']

        if self._LastTMap==-1:
            self.Widget.IsppaScrollBar.setMaximum(DataThermal['MaterialMap'].shape[1]-1)
            self.Widget.IsppaScrollBar.setValue(Loc[1])
            self.Widget.IsppaScrollBar.setEnabled(True)
            
        self._LastTMap=self.Widget.SelCombinationDropDown.currentIndex()
            
        if OverWriteIsppa is None:
            SelIsppa=self.Widget.IsppaSpinBox.value()
        else:
            SelIsppa=OverWriteIsppa

        xf=self._xf
        zf=self._zf

       
        SelY=self.Widget.IsppaScrollBar.value()

 

        IsppaRatio=SelIsppa/self.Config['BaseIsppa']

        PresRatio=np.sqrt(IsppaRatio)

        AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']

        
        DutyCycle=self.Config['AllDC_PRF_Duration'][self.Widget.SelCombinationDropDown.currentIndex()]['DC']

        self.Widget.IsppaWaterLabel.setProperty('UserData',AdjustedIsspa)
        self.Widget.IsppaWaterLabel.setText('%4.2f' % self.Widget.IsppaWaterLabel.property('UserData'))

        self.Widget.IsptaLabel.setProperty('UserData',SelIsppa*DutyCycle)
        self.Widget.IsptaLabel.setText('%4.2f' % self.Widget.IsptaLabel.property('UserData'))
        AdjustedTemp=((DataThermal['TemperaturePoints']-37)*IsppaRatio+37)
        DoseUpdate=np.sum(RCoeff(AdjustedTemp)**(43.0-AdjustedTemp),axis=1)*DataThermal['dt']/60
        
        self.Widget.MILabel.setProperty('UserData',DataThermal['MI']*PresRatio)
        self.Widget.MTBLabel.setProperty('UserData',DataThermal['TI']*IsppaRatio+37)
        self.Widget.MTCLabel.setProperty('UserData',DataThermal['TIC']*IsppaRatio+37)
        self.Widget.MTSLabel.setProperty('UserData',DataThermal['TIS']*IsppaRatio+37)

        self.Widget.CEMSkinLabel.setProperty('UserData',DoseUpdate[0])
        self.Widget.CEMBrainLabel.setProperty('UserData',DoseUpdate[1])
        self.Widget.CEMSkullLabel.setProperty('UserData',DoseUpdate[2])
     
        self.Widget.AdjustRASLabel.setProperty('UserData',DataThermal['AdjustmentInRAS'])
        for obj in [self.Widget.MILabel,self.Widget.MTBLabel,
                    self.Widget.MTCLabel,self.Widget.MTSLabel]:
                obj.setText('%3.2f' % obj.property('UserData'))
        for obj in [self.Widget.CEMBrainLabel,self.Widget.CEMSkullLabel,
                    self.Widget.CEMSkinLabel]:
                obj.setText('%4.1G' % obj.property('UserData'))

        self.Widget.AdjustRASLabel.setText(np.array2string(self.Widget.AdjustRASLabel.property('UserData'),
                                               formatter={'float_kind':lambda x: "%3.2f" % x}))

        if self._bRecalculated:
            XX,ZZ=np.meshgrid(xf,zf)
            self._XX=XX
            self._ZZ=ZZ
            
        if bUpdatePlot:
            DensityMap=DataThermal['MaterialList']['Density'][DataThermal['MaterialMap'][:,SelY,:]]
            SoSMap=    DataThermal['MaterialList']['SoS'][DataThermal['MaterialMap'][:,SelY,:]]
            IntensityMap=(DataThermal['p_map'][:,SelY,:]**2/2/DensityMap/SoSMap/1e4*IsppaRatio).T
            if 'ZIntoSkinPixels' in DataThermal:
                IntensityMap[DataThermal['ZIntoSkinPixels'],:]=0
            else:
                IntensityMap[0,:]=0
            Tmap=(DataThermal['TempEndFUS'][:,SelY,:]-37.0)*IsppaRatio+37.0

            if self._MainApp.Config['bUseCT']:
                crlims=[0,1,2]
            else:
                crlims=[0,1,2,3]

            if self._bRecalculated and hasattr(self,'_figIntThermalFields'):
                children = []
                for i in range(self._layout.count()):
                    child = self._layout.itemAt(i).widget()
                    if child:
                        children.append(child)
                for child in children:
                    child.deleteLater()
                delattr(self,'_figIntThermalFields')
                # self._layout.deleteLater()
                

            if hasattr(self,'_figIntThermalFields'):
                self._IntensityIm.set_data(IntensityMap)
                self._IntensityIm.set(clim=[IntensityMap.min(),IntensityMap.max()])
                self._ThermalIm.set_data(Tmap.T)
                self._ThermalIm.set(clim=[37,Tmap.max()])
                if hasattr(self,'_contour1'):
                    for c in [self._contour1,self._contour2]:
                        for coll in c.collections:
                            coll.remove()
                    del self._contour1
                    del self._contour2
                self._contour1=self._static_ax1.contour(self._XX,self._ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)
                self._contour2=self._static_ax2.contour(self._XX,self._ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)
                while len(self._ListMarkers)>0:
                    obj= self._ListMarkers.pop()
                    obj.remove()
                self._figIntThermalFields.canvas.draw_idle()
            else:
                self._ListMarkers=[]
                if not hasattr(self,'_layout'):
                    self._layout = QVBoxLayout(self.Widget.AcField_plot1)

                self._figIntThermalFields=Figure(figsize=(14, 12))
                self.static_canvas = FigureCanvas(self._figIntThermalFields)
                toolbar=NavigationToolbar2QT(self.static_canvas,self)
                self._layout.addWidget(toolbar)
                self._layout.addWidget(self.static_canvas)
                static_ax1,static_ax2 = self.static_canvas.figure.subplots(1,2)
                self._static_ax1=static_ax1
                self._static_ax2=static_ax2

                self._IntensityIm=static_ax1.imshow(IntensityMap,extent=[xf.min(),xf.max(),zf.max(),zf.min()],
                        cmap=plt.cm.jet)
                static_ax1.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18)
                static_ax1.set_title('Isppa (W/cm$^2$)')
                plt.colorbar(self._IntensityIm,ax=static_ax1)

                self._contour1=static_ax1.contour(self._XX,self._ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)

                static_ax1.set_ylabel('Distance from skin (mm)')

                self._ThermalIm=static_ax2.imshow(Tmap.T,
                        extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet,vmin=37)
                static_ax2.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18)
                static_ax2.set_title('Temperature ($^{\circ}$C)')

                plt.colorbar(self._ThermalIm,ax=static_ax2)
                self._contour2=static_ax2.contour(XX,ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)

                # self._figIntThermalFields.set_tight_layout(True)

                self._figIntThermalFields.set_facecolor(np.array(self.palette().color(QPalette.Window).getRgb())/255)

            self._bRecalculated=False

            yf=DataThermal['y_vec']
            yf-=yf[Loc[1]]

            for k,kl in zip(['mSkin','mBrain','mSkull'],['MTS','MTB','MTC']):
                if SelY == DataThermal[k][1]:
                    self._ListMarkers.append(self._static_ax2.plot(xf[DataThermal[k][0]],
                                    zf[DataThermal[k][2]],'wx',markersize=12)[0])
                    self._ListMarkers.append(self._static_ax2.text(xf[DataThermal[k][0]]-5,
                                    zf[DataThermal[k][2]]+5,kl,color='w',fontsize=10))
            self.Widget.SliceLabel.setText("Y pos = %3.2f mm" %(yf[self.Widget.IsppaScrollBar.value()]))

    @Slot()
    def LocateMTB(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mBrain'][1])
    @Slot()
    def LocateMTC(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mSkull'][1])
    @Slot()
    def LocateMTS(self):
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        self.Widget.IsppaScrollBar.setValue(DataThermal['mSkin'][1])

    @Slot()
    def ExportSummary(self):
        DefaultPath=os.path.split(self._MainApp.Config['T1W'])[0]
        outCSV=QFileDialog.getSaveFileName(self,"Select export CSV file",DefaultPath,"csv (*.csv)")[0]
        if len(outCSV)==0:
            return
        
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        DataToExport={}
        #we recover specifics of main app and acoustic simulation
        for obj in [self._MainApp,self._MainApp.AcSim]:
            Export=obj.GetExport()
            DataToExport= DataToExport | Export
        DataToExport['AdjustRAS']=self.Widget.AdjustRASLabel.property('UserData')
        
        pd.DataFrame.from_dict(data=DataToExport, orient='index').to_csv(outCSV, header=False)
        currentIsppa=self.Widget.IsppaSpinBox.value()
        currentCombination=self.Widget.SelCombinationDropDown.currentIndex()
        #now we create new Table to export safety metrics based on timing options and Isppa
        print('self.Widget.SelCombinationDropDown.count()',self.Widget.SelCombinationDropDown.count())
        for n in range(self.Widget.SelCombinationDropDown.count()):
            self.Widget.SelCombinationDropDown.setCurrentIndex(n)
            DataToExport['TimingExposure']=self.Widget.SelCombinationDropDown.currentText()
            print('self.Widget.SelCombinationDropDown.currentText()',self.Widget.SelCombinationDropDown.currentText())
            with open(outCSV,'a') as f:
                f.write('*'*80+'\n')
                f.write('TimingExposure,'+self.Widget.SelCombinationDropDown.currentText()+'\n')
                f.write('*'*80+'\n')
                
            DataToExport={}
            DataToExport['Isppa']=np.arange(0.5,self.Widget.IsppaSpinBox.maximum()+0.5,0.5)
            for v in DataToExport['Isppa']:
                self.UpdateThermalResults(bUpdatePlot=False,OverWriteIsppa=v)
                for k in ['IsppaWater','MI','Ispta',
                            ['MTB','MTBLabel'],['MTS','MTSLabel'],
                            'MTC','CEMBrain','CEMSkin','CEMSkull']:
                    if type(k) is list:
                        if k[0] not in DataToExport:
                            DataToExport[k[0]]=[]
                    else:
                        if k not in DataToExport:
                            DataToExport[k]=[]
                    if type(k) is list: 
                        obj=getattr(self.Widget,k[1])
                        DataToExport[k[0]].append(obj.property('UserData'))
                    else:
                        obj=getattr(self.Widget,k+'Label')
                        DataToExport[k].append(obj.property('UserData'))
                
            pd.DataFrame.from_dict(data=DataToExport).to_csv(outCSV,mode='a',index=False)
        if currentCombination !=self.Widget.SelCombinationDropDown.currentIndex():
            self.Widget.SelCombinationDropDown.setCurrentIndex(currentCombination) #this will refresh
        else:
            self.UpdateThermalResults(bUpdatePlot=True,OverWriteIsppa=currentIsppa)
        
        


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

        kargs['TxSystem']=self._mainApp.Config['TxSystem']
        if kargs['TxSystem'] in ['CTX_500','Single','H246','BSonix']:
            kargs['sel_p']='p_amp'
        else:
            bRefocus = self._mainApp.AcSim.Widget.RefocusingcheckBox.isChecked()
            if bRefocus:
                kargs['sel_p']='p_amp_refocus'
            else:
                kargs['sel_p']='p_amp'


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
