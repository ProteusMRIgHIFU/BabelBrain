# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
from multiprocessing import Process,Queue

from PySide6.QtWidgets import (QApplication, QWidget,QGridLayout,
                QHBoxLayout,QVBoxLayout,QLineEdit,QDialog,QTextEdit,
                QGridLayout, QSpacerItem, QInputDialog, QFileDialog,QFrame,
                QErrorMessage, QMessageBox,QDialogButtonBox,QLabel,QTableWidgetItem)
from PySide6.QtCore import QFile,Slot,QObject,Signal,QThread,Qt
from PySide6 import QtCore,QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPalette, QTextCursor,QColor


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
import nibabel

_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS) / 'Babel_Thermal'
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
        self._bMultiPoint = False
        self.bDisableUpdate=False
        self.static_canvas=None
        self.load_ui()
        self.DefaultConfig()
        self._LastTMap=-1

    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(resource_path(), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget = loader.load(ui_file, self)
        ui_file.close()

        self.Widget.SelectProfile.clicked.connect(self.SelectProfile)
        self.Widget.SelectProfile.setStyleSheet("color: green")
        self.Widget.CalculateThermal.clicked.connect(self.RunSimulation)
        self.Widget.CalculateThermal.setStyleSheet("color: red")
        self.Widget.ExportSummary.clicked.connect(self.ExportSummary)
        self.Widget.ExportThermalMap.clicked.connect(self.ExportThermalMap)

        self.Widget.SelCombinationDropDown.currentIndexChanged.connect(self.UpdateSelCombination)
        self.Widget.IsppaSpinBox.valueChanged.connect(self.UpdateThermalResults)
        self.Widget.IsppaScrollBar.valueChanged.connect(self.UpdateThermalResults)
        self.Widget.HideMarkscheckBox.stateChanged.connect(self.HideMarkChange)
        self.Widget.IsppaScrollBar.setEnabled(False)
        self.Widget.SelCombinationDropDown.setEnabled(False)
        self.Widget.IsppaSpinBox.setEnabled(False)

        self.Widget.LocMTB.clicked.connect(self.LocateMTB)
        self.Widget.LocMTB.setEnabled(False)
        self.Widget.LocMTC.clicked.connect(self.LocateMTC)
        self.Widget.LocMTC.setEnabled(False)
        self.Widget.LocMTS.clicked.connect(self.LocateMTS)
        self.Widget.LocMTS.setEnabled(False)

        # for l in [self.Widget.label_13,self.Widget.label_14,self.Widget.label_15,self.Widget.label_22]:
        #     l.setText(l.text()+' ('+"\u2103"+'):')

        Ids=['Isppa at target (W/cm2):',
             'Req. Isppa water (W/cm2):',
             'Ispta (W/cm2):',
             'Ispta at target (W/cm2):',
             'Adjustment in RAS T1W space:',
             'Max. temp. target ('+"\u2103"+') - CEM43:',
             'Max. temp. brain ('+"\u2103"+') - CEM43:',
             'Max. temp. skin ('+"\u2103"+') - CEM43:',
             'Max. temp. skull ('+"\u2103"+') - CEM43:',
             'Mechanical index:',
             'Distance from MTB to MTT (mm):']
        bg_color = self.Widget.tableWidget.parent().palette().color(self.Widget.backgroundRole())
        text_color = self.Widget.tableWidget.parent().palette().color(self.Widget.foregroundRole())
        table_palette = self.Widget.tableWidget.palette()
        table_palette.setColor(QPalette.Base, bg_color)
        self.Widget.tableWidget.setPalette(table_palette)
        if 'Windows' in platform.system():
            
            # self.Widget.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
            self.Widget.tableWidget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            # self.Widget.tableWidget.verticalHeader().setDefaultSectionSize(5)
        for n,v in enumerate(Ids):
            item=QTableWidgetItem(v)
            item.setTextAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
            if 'Windows' in platform.system():
                font = item.font()
                font.setPointSize(8)
                item.setFont(font)
            else:
                font = item.font()
                font.setPointSize(12)
                item.setFont(font)

            self.Widget.tableWidget.setItem(n,0,item)
        if 'Windows' in platform.system():
            self.Widget.tableWidget.setColumnWidth(0,180)
            self.Widget.tableWidget.setColumnWidth(1,self.Widget.tableWidget.width()-180)
        else:
            self.Widget.tableWidget.setColumnWidth(0,220)
            self.Widget.tableWidget.setColumnWidth(1,self.Widget.tableWidget.width()-220)
        if 'Windows' in platform.system():
            self.Widget.tableWidget.verticalHeader().setDefaultSectionSize(5)
        else:
            self.Widget.tableWidget.verticalHeader().setDefaultSectionSize(25)
        self.Widget.tableWidget.setFrameShape(QFrame.NoFrame)

    def DefaultConfig(self):
        #Specific parameters for the thermal simulation - to be configured  via a yaml
        with open(self._MainApp.Config['ThermalProfile'], 'r') as file:
            config = yaml.safe_load(file)
            print("Thermal configuration:")
            print(config)
            self.Config=config
            self.bDisableUpdate=True

            while self.Widget.SelCombinationDropDown.count()>0:
                self.Widget.SelCombinationDropDown.removeItem(0)

            for c in self.Config['AllDC_PRF_Duration']:
                self.Widget.SelCombinationDropDown.addItem('%3.1fs-On %3.1fs-Off %3.1f%% %3.1fHz' %(c['Duration'],c['DurationOff'],c['DC']*100,c['PRF']))
            self.bDisableUpdate=False

    def EnableMultiPoint(self):
        self._bMultiPoint=True

    @Slot()
    def SelectProfile(self):
        fThermalProfile=QFileDialog.getOpenFileName(self,"Select thermal profile",os.getcwd(),"yaml (*.yaml)")[0]
        if len(fThermalProfile)>0:
            if self._MainApp.UpdateThermalProfile(fThermalProfile):
                self.Widget.SelectProfile.setProperty('UserData',fThermalProfile)  
                self.DefaultConfig()  
                self.RunSimulation()
                

    @Slot()
    def RunSimulation(self):
        bCalcFields=False
        
        BaseField=self._MainApp.AcSim._FullSolName
        
        if type(BaseField) is list:
            BaseField=BaseField[0]

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
            self.worker = RunThermalSim(self._MainApp)
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
            self._MainApp.showClockDialog()
        else:
            self.UpdateThermalResults()

    def NotifyError(self):
        self._MainApp.hideClockDialog()
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()

    @Slot()
    def HideMarkChange(self,val):
        self.UpdateThermalResults()

    @Slot()
    def UpdateSelCombination(self):
        self.UpdateThermalResults()

    @Slot()
    def UpdateThermalResults(self,bUpdatePlot=True,OverWriteIsppa=None):
        if self.bDisableUpdate:
            return
        self._MainApp.Widget.tabWidget.setEnabled(True)
        self.Widget.ExportSummary.setEnabled(True)
        self.Widget.ExportThermalMap.setEnabled(True)
        self.Widget.SelCombinationDropDown.setEnabled(True)
        self.Widget.IsppaScrollBar.setEnabled(True)
        self.Widget.IsppaSpinBox.setEnabled(True)
        self.Widget.LocMTS.setEnabled(True)
        self.Widget.LocMTC.setEnabled(True)
        self.Widget.LocMTB.setEnabled(True)
        if self.Widget.HideMarkscheckBox.isEnabled()== False:
            self.Widget.HideMarkscheckBox.setEnabled(True)

        BaseField=self._MainApp.AcSim._FullSolName
        if type(BaseField) is list:
            BaseField=BaseField[0]
            
        if len(self._ThermalResults)==0:
            self._MainApp.hideClockDialog()
            self._NiftiThermalNames=[]
            self._LastTMap=-1
            for combination in self.Config['AllDC_PRF_Duration']:
                ThermalName=GetThermalOutName(BaseField,combination['Duration'],
                                                        combination['DurationOff'],
                                                        combination['DC'],
                                                        self.Config['BaseIsppa'],
                                                        combination['PRF'])+'.h5'
                self._NiftiThermalNames.append(os.path.splitext(ThermalName)[0])
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

        if self._bMultiPoint:
            AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']
            AdjustedIsspaStDev = np.std(AdjustedIsspa)
            AdjustedIsspa=np.mean(AdjustedIsspa)
        else:
            AdjustedIsspa = SelIsppa/DataThermal['RatioLosses']
                
        DutyCycle=self.Config['AllDC_PRF_Duration'][self.Widget.SelCombinationDropDown.currentIndex()]['DC']

        def NewItem(str,data,color="blue"):
            item=QTableWidgetItem(str)
            item.setData(QtCore.Qt.UserRole,data)
            item.setForeground(QColor(color))
            # Set the font style to bold
            font = item.font()
            font.setBold(True)
            if 'Windows' in platform.system():
                font.setPointSize(8)
            else:
                font.setPointSize(12)
            item.setFont(font)
            return item

        IsppaTarget = DataThermal['p_map'][Loc[0],Loc[1],Loc[2]]**2/2/\
                      DataThermal['MaterialList']['Density'][DataThermal['MaterialMap'][Loc[0],Loc[1],Loc[2]]]/\
                      DataThermal['MaterialList']['SoS'][DataThermal['MaterialMap'][Loc[0],Loc[1],Loc[2]]]/\
                      1e4*IsppaRatio
        self.Widget.tableWidget.setItem(0,1,NewItem('%4.2f' % IsppaTarget,IsppaTarget))

        if self._bMultiPoint:
            self.Widget.tableWidget.setItem(1,1,NewItem('%4.2f (%4.2f)' % (AdjustedIsspa,AdjustedIsspaStDev),AdjustedIsspa))
        else:
            self.Widget.tableWidget.setItem(1,1,NewItem('%4.2f' % AdjustedIsspa,AdjustedIsspa))
            
        self.Widget.tableWidget.setItem(2,1,NewItem('%4.2f' % (SelIsppa*DutyCycle),SelIsppa*DutyCycle))

        self.Widget.tableWidget.setItem(3,1,NewItem('%4.2f' % (IsppaTarget*DutyCycle),IsppaTarget*DutyCycle))

        self.Widget.tableWidget.setItem(4,1,NewItem(np.array2string(DataThermal['AdjustmentInRAS'],
                                               formatter={'float_kind':lambda x: "%3.2f" % x}),DataThermal['AdjustmentInRAS']))

        AdjustedTemp=((DataThermal['TemperaturePoints']-37)*IsppaRatio+37)
        DoseUpdate=np.trapz(RCoeff(AdjustedTemp)**(43.0-AdjustedTemp),dx=DataThermal['dt'],axis=1)/60
   
        MTT=(DataThermal['TempEndFUS'][Loc[0],Loc[1],Loc[2]]-37.0)*IsppaRatio+37.0
        MTTCEM=DoseUpdate[3] if len(DoseUpdate)==4 else DoseUpdate[1]
        self.Widget.tableWidget.setItem(5,1,NewItem('%3.1f - %4.1G' % (MTT,MTTCEM),[MTT,MTTCEM],"red" if MTT >= 39 else "blue"))

        MTB=DataThermal['TI']*IsppaRatio+37.0
        MTBCEM=DoseUpdate[1]
        self.Widget.tableWidget.setItem(6,1,NewItem('%3.1f - %4.1G' % (MTB,MTBCEM),[MTB,MTBCEM],"red" if MTB >= 39 else "blue"))
        
        MTS=DataThermal['TIS']*IsppaRatio+37.0
        MTSCEM=DoseUpdate[0]
        self.Widget.tableWidget.setItem(7,1,NewItem('%3.1f - %4.1G' % (MTS,MTSCEM),[MTS,MTSCEM],"red" if MTS >= 39 else "blue"))

        MTC=DataThermal['TIC']*IsppaRatio+37.0
        MTCCEM=DoseUpdate[2]
        self.Widget.tableWidget.setItem(8,1,NewItem('%3.1f - %4.1G' % (MTC,MTCCEM),[MTC,MTCCEM],"red" if MTC >= 39 else "blue"))

        MI=DataThermal['MI']*PresRatio
        self.Widget.tableWidget.setItem(9,1,NewItem('%3.1f ' % (MI),MI,"red" if MI > 1.9 else "blue"))
    
        Distance_MTB_MTT = np.linalg.norm(DataThermal['mBrain']-Loc)*(xf[1]-xf[0])
        self.Widget.tableWidget.setItem(10,1,NewItem('%3.1f ' % (Distance_MTB_MTT),Distance_MTB_MTT))

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
                self._marker1=static_ax1.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18)[0]
                static_ax1.set_title('Isppa (W/cm$^2$)')
                plt.colorbar(self._IntensityIm,ax=static_ax1)

                self._contour1=static_ax1.contour(self._XX,self._ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)

                static_ax1.set_ylabel('Distance from skin (mm)')

                self._ThermalIm=static_ax2.imshow(Tmap.T,
                        extent=[xf.min(),xf.max(),zf.max(),zf.min()],cmap=plt.cm.jet,vmin=37)
                self._marker2=static_ax2.plot(xf[Loc[0]],zf[Loc[2]],'k+',markersize=18,)[0]
                static_ax2.set_title('Temperature ($^{\circ}$C)')

                plt.colorbar(self._ThermalIm,ax=static_ax2)
                self._contour2=static_ax2.contour(XX,ZZ,DataThermal['MaterialMap'][:,SelY,:].T,crlims, cmap=plt.cm.gray)

                # self._figIntThermalFields.set_tight_layout(True)

                self._figIntThermalFields.set_facecolor(np.array(self.palette().color(QPalette.Window).getRgb())/255)

            self._bRecalculated=False
            mc = [0.0,0.0,0.0,1.0]
            if self.Widget.HideMarkscheckBox.isChecked():
                mc[3]=0.0 
            self._marker1.set_markerfacecolor(mc)
            self._marker2.set_markerfacecolor(mc)

            yf=DataThermal['y_vec']
            yf-=yf[Loc[1]]
            if not self.Widget.HideMarkscheckBox.isChecked():
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
        DataToExport['AdjustRAS']=self.Widget.tableWidget.item(4,1).data(QtCore.Qt.UserRole)
        
        pd.DataFrame.from_dict(data=DataToExport, orient='index').to_csv(outCSV, header=False)
        currentIsppa=self.Widget.IsppaSpinBox.value()
        currentCombination=self.Widget.SelCombinationDropDown.currentIndex()
        #now we create new Table to export safety metrics based on timing options and Isppa
        for n in range(self.Widget.SelCombinationDropDown.count()):
            self.Widget.SelCombinationDropDown.setCurrentIndex(n)
            with open(outCSV,'a') as f:
                f.write('*'*80+'\n')
                f.write('TimingExposure,'+self.Widget.SelCombinationDropDown.currentText()+'\n')
                f.write('*'*80+'\n')

            DataToExport={}
            DataToExport['Distance from MTB to MTT:']=self.Widget.tableWidget.item(10,1).data(QtCore.Qt.UserRole)
            pd.DataFrame.from_dict(data=DataToExport, orient='index').to_csv(outCSV,mode='a', header=False)
                
            DataToExport={}
            DataToExport['Isppa']=np.arange(0.5,self.Widget.IsppaSpinBox.maximum()+0.5,0.5)
            for v in DataToExport['Isppa']:
                self.UpdateThermalResults(bUpdatePlot=False,OverWriteIsppa=v)
                collection = ['Isppa target']
                source = [0,1]
                if self._bMultiPoint:
                    collection+=['Isppa water (avg)','Isppa water (std)']
                    source+=[1]
                else:
                    collection+=['Isppa water']
                collection+=['Mechanical index',
                            'Ispta',
                            'Ispta target',
                            'Max. temp. target',
                            'Max. temp. brain',
                            'Max. temp. skin',
                            'Max. temp. skull',
                            'CEM target',
                            'CEM brain',
                            'CEM skin',
                            'CEM skull']
                source+=[9,2,3,5,6,7,8,5,6,7,8]
                for k,index in zip(collection,source):
                    if k not in DataToExport:
                        DataToExport[k]=[]
                   
                    data=self.Widget.tableWidget.item(index,1).data(QtCore.Qt.UserRole)
                    if 'temp.' in k:
                        data=data[0]
                    elif 'CEM' in k:
                        data=data[1]
                    elif k in ['Isppa water (avg)','Isppa water']:
                        data=np.mean(data)
                    elif k == 'Isppa water (std)':
                        data=np.std(data)
                    DataToExport[k].append(data)
                
            pd.DataFrame.from_dict(data=DataToExport).to_csv(outCSV,mode='a',index=False)
        if currentCombination !=self.Widget.SelCombinationDropDown.currentIndex():
            self.Widget.SelCombinationDropDown.setCurrentIndex(currentCombination) #this will refresh
        else:
            self.UpdateThermalResults(bUpdatePlot=True,OverWriteIsppa=currentIsppa)
        
    @Slot()
    def ExportThermalMap(self):
        OutName=self._NiftiThermalNames[self.Widget.SelCombinationDropDown.currentIndex()]
        SelIsppa=self.Widget.IsppaSpinBox.value()
        IsppaRatio=SelIsppa/self.Config['BaseIsppa']
        BasePath = OutName.split('_DataForSim')[0]
        
        OutName = OutName.replace('_DataForSim','')
        OutName = OutName.split('-Isppa')[0] + ('_Isppa_%2.1fW' % (SelIsppa)).replace('.','p') + '-PRF' + OutName.split('-PRF')[1]
        OutName+='.nii.gz'
        print(OutName)
        
        suffix='_FullElasticSolution_Sub_NORM.nii.gz'
        if self._MainApp.Config['TxSystem'] not in ['CTX_500','CTX_250','Single','H246','BSonix']:
            if self._MainApp.AcSim.Widget.RefocusingcheckBox.isChecked():
                suffix='_FullElasticSolutionRefocus_Sub_NORM.nii.gz'
        BasePath+=suffix
        nidata = nibabel.load(BasePath)
        DataThermal=self._ThermalResults[self.Widget.SelCombinationDropDown.currentIndex()]
        Tmap=(DataThermal['TempEndFUS']-37.0)*IsppaRatio+37.0
        Tmap=np.flip(Tmap,axis=2)
        nii=nibabel.Nifti1Image(Tmap,affine=nidata.affine)
        nii.to_filename(OutName)
        #If running with Brainsight, we save the path of thermal map
        if self._MainApp.Config['bInUseWithBrainsight']:
            with open(self._MainApp.Config['Brainsight-ThermalOutput'],'w') as f:
                f.write(OutName)
        txt = "Thermal map file\n" + os.path.basename(OutName) +'\nsaved at:\n '+os.path.dirname(OutName)
        maxL=np.max([len(os.path.basename(OutName)), len(os.path.dirname(OutName))])
        msgBox = DialogShowText(txt,"Saved thermal map")
        msgBox.exec()



class DialogShowText(QDialog):
    def __init__(self, text,title,parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok 

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel(text)
        message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message.setStyleSheet(f"qproperty-alignment: {int(Qt.AlignmentFlag.AlignCenter)};")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    

class RunThermalSim(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp):
         super(RunThermalSim, self).__init__()
         self._mainApp=mainApp

    def run(self):

        case=self._mainApp.AcSim._FullSolName
        print('Calculating thermal maps for configurations\n',self._mainApp.ThermalSim.Config['AllDC_PRF_Duration'])
        T0=time.time()
        kargs={}
        kargs['deviceName']=self._mainApp.Config['ComputingDevice']
        kargs['COMPUTING_BACKEND']=self._mainApp.Config['ComputingBackend']
        kargs['Isppa']=self._mainApp.ThermalSim.Config['BaseIsppa']

        kargs['TxSystem']=self._mainApp.Config['TxSystem']
        if kargs['TxSystem'] in ['CTX_500','CTX_250','Single','H246','BSonix']:
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
            TotalTime = TEnd-T0
            print('Total time',TotalTime)
            print("*"*40)
            print("*"*5+" DONE thermal simulation.")
            print("*"*40)
            self._mainApp.UpdateComputationalTime('thermal',TotalTime)
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
