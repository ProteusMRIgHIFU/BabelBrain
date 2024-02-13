# This Python file uses the following encoding: utf-8
'''
BabelBrain: Application for the planning and delivery of LIFU to be triggered from Brainsight
ABOUT:
    author        - Samuel Pichardo
    date          - July 16, 2022
    last update   - July 16, 2022
'''
import argparse
import multiprocessing
import os
import platform
import shutil
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import SimpleITK as sitk
import nibabel
import numpy as np
import yaml
from PySide6.QtCore import QFile, QObject, QThread, Qt, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QPalette, QTextCursor, QMovie
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QLabel
)
from linetimer import CodeTimer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from nibabel import processing
from superqt import QLabeledDoubleRangeSlider

from CalculateMaskProcess import CalculateMaskProcess
from ConvMatTransform import (
    BSight_to_itk,
    GetIDTrajectoryBrainsight,
    ReadTrajectoryBrainsight,
    GetBrainSightHeader,
    itk_to_BSight,
    templateSlicer,
    read_itk_affine_transform,
)
from SelFiles.SelFiles import SelFiles


multiprocessing.freeze_support()
if sys.platform =='linux':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass




sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))



#from qtrangeslider import   QLabeledDoubleRangeSlider




_IS_MAC = platform.system() == 'Darwin'

def resource_path():  # needed for bundling
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if not _IS_MAC:
        return os.path.split(Path(__file__))[0]

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent

    return bundle_dir

def get_text_values(initial_texts, parent=None, title="", label=""):
    '''
    Simple Input dialog to ask for multiple inputs
    '''
    dialog = QInputDialog()
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)
    dialog.show()
    # hide default QLineEdit
    dialog.findChild(QLineEdit).hide()

    editors = []
    for i, text in enumerate(initial_texts, start=1):
        editor = QLineEdit(text=text)
        dialog.layout().insertWidget(i, editor)
        editors.append(editor)

    ret = dialog.exec() == QDialog.Accepted
    return ret, [editor.text() for editor in editors]

###################################################################

ReturnCodes={}
ReturnCodes['SUCCES']=0
ReturnCodes['CANCEL_OR_INCOMPLETE']=1
ReturnCodes['ERROR_DOMAIN']=2
ReturnCodes['ERROR_ACOUSTICS']=3

##########################

class OutputWrapper(QObject):
    outputWritten = Signal(object, object)

    def __init__(self, parent, stdout=True):
        super().__init__(parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        self._stream.write(text)
        self.outputWritten.emit(text, self._stdout)

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

#Pointers to functions

GetSmallestSOS=None

_LastSelConfig=str(Path.home())+os.sep+os.path.join('.config','BabelBrain','lastselection.yaml')

def GetLatestSelection():
    res=None
    if os.path.isfile(_LastSelConfig):
        with open(_LastSelConfig,'r') as f:
            try:
                res=yaml.safe_load(f)
            except BaseException as e:
                print('Unable to load previous selection')
                print(e)
                res=None
        try:
            if res is not None:
                if not os.path.isdir(res['simbnibs_path']) or not os.path.isfile(res['T1W']) or not os.path.isfile(res['Mat4Trajectory'])\
                or not os.path.isfile(res['ThermalProfile']):
                    print('Ignoring config as files and dir may not exist anymore\n',res)
                    res=None
        except:
            res = None
    return res

def GetInputFromBrainsight():
    res=None
    PathMat4Trajectory  = _BrainsightSyncPath + os.sep +'Input_Target.txt'
    PathT1W             = _BrainsightSyncPath + os.sep +'Input_Anatomical.txt'
    Pathsimbnibs_path   = _BrainsightSyncPath + os.sep +'Input_SegmentationsPath.txt'
    PathToSaveResults   = _BrainsightSyncPath + os.sep +'SimulationOutputPath.txt'


    if os.path.isfile(PathMat4Trajectory) and \
        os.path.isfile(PathT1W) and \
        os.path.isfile(Pathsimbnibs_path):
        res={}
        with open (PathT1W,'r') as f:
            l=f.readlines()[0].strip()
        res['T1W']=l

        res['outputfiles_path']=None

        if os.path.isfile(PathToSaveResults):
            with open (PathToSaveResults,'r') as f:
                l=f.readlines()[0].strip()
            if len(l)>0:
                res['outputfiles_path']=l   

        ID=GetIDTrajectoryBrainsight(PathMat4Trajectory)
        header =  GetBrainSightHeader(PathMat4Trajectory)
        if header['Version']=='13':
            # EndWithError("Version 13 of export trajectory not supported.\nEnding BabelBrain execution")
            pass
        else:
            if header['Version']!='14':
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("Version of export trajectory not officially supported.\nBabelBrain will continue but issues may occur")
                msgBox.exec()
            if 'NIfTI' not in header['Coordinate system']:
                EndWithError("BabelBrain only supports Nifti convention for trajectory")
            InputT1 = nibabel.load(res['T1W'])
            qformcode=int(InputT1.header['qform_code'])
            sformcode=int(InputT1.header['sform_code'])
            bValidT1W=False
            #we validate agreement between header of Brainsight trajectory and qform and sform codes in the Nifti file
            if header['Coordinate system'] == 'NIfTI:Q:Scanner':
                bValidT1W = qformcode==1
            elif header['Coordinate system'] == 'NIfTI:Q:Aligned':
                bValidT1W = qformcode==2
            elif header['Coordinate system'] == 'NIfTI:Q:MNI-152':
                bValidT1W = qformcode==3
            elif header['Coordinate system'] == 'NIfTI:Q:Talairach':
                bValidT1W = qformcode==4
            elif header['Coordinate system'] == 'NIfTI:Q:Other-Template':
                bValidT1W = qformcode==5
            elif header['Coordinate system'] == 'NIfTI:S:Scanner':
                bValidT1W = sformcode==1 
            elif header['Coordinate system'] == 'NIfTI:S:Aligned':
                bValidT1W = sformcode==2 
            elif header['Coordinate system'] == 'NIfTI:S:MNI-152':
                bValidT1W = sformcode==3 
            elif header['Coordinate system'] == 'NIfTI:S:Talairach':
                bValidT1W = sformcode==4 
            elif header['Coordinate system'] == 'NIfTI:S:Other-Template':
                bValidT1W = sformcode==5 
            if not bValidT1W:
                EndWithError("Header coordinate system ("+header['Coordinate system'] +") does not match T1W Nifti header\n qformcode and sformcode [%i,%i]" %(qformcode,sformcode))
                        

        if res['outputfiles_path'] is not None:
            outpath = res['outputfiles_path'] 
        else:
            outpath = os.path.dirname(res['T1W']) #+ os.sep + 'Babel' + os.sep + ID 
        
        
        #for the time being, we need the trajectory to be next to T1w
        RPath=outpath+os.sep+ID+'.txt'
        assert(shutil.copyfile(PathMat4Trajectory,RPath))

        print('ID,RPath',ID,RPath)

        res['Mat4Trajectory']=RPath
        
        with open (Pathsimbnibs_path,'r') as f:
            l=f.readlines()[0].strip()
        res['simbnibs_path']=l
        
         
        
        if not os.path.isdir(res['simbnibs_path']) or not os.path.isfile(res['T1W']) or not os.path.isfile(res['Mat4Trajectory']):
                print('Ignoring Brainsight config as files and dir may not exist anymore\n',res)
                res=None
    else:
        EndWithError("Incomplete Brainsight input files at\n" + BabelBrain._BrainsightSyncPath)

    ofiles =[_BrainsightSyncPath+ os.sep+'Output.txt',
            _BrainsightSyncPath+ os.sep+'Output_TargetModified.txt',
            _BrainsightSyncPath+ os.sep+'Output_Thermal.txt']
    for fpath in ofiles:
        if os.path.isfile(fpath):
            os.remove(fpath)
        
    return res,header

def EndWithError(msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(msg)
        msgBox.exec()
        raise SystemError(msg)
########################
class ClockDialog(QDialog):
    def __init__(self, parent=None):
        super(ClockDialog,self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setModal(False)

        self.label = QLabel(self)
        self.movie = QMovie( os.path.join(resource_path(),'icons8-hourglass.gif'))
        self.label.setMovie(self.movie)
        self.movie.start()

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

_BrainsightSyncPath = str(Path.home()) + os.sep + '.BabelBrainSync'
class BabelBrain(QWidget):
    '''
    Main LIFU Control application

    '''

    def __init__(self,widget,bInUseWithBrainsight=False,AltOutputFilesPath=None):
        super(BabelBrain, self).__init__()
        #This file will store the last config selected

        simbnibs_path=widget.ui.SimbNIBSlineEdit.text()
        T1W=widget.ui.T1WlineEdit.text()
        CT_or_ZTE_input=widget.ui.CTlineEdit.text()
        bUseCT=widget.ui.CTTypecomboBox.currentIndex()>0
        CTType=widget.ui.CTTypecomboBox.currentIndex()
        Mat4Trajectory=widget.ui.TrajectorylineEdit.text()
        ThermalProfile=widget.ui.ThermalProfilelineEdit.text()
        if widget.ui.SimbNIBSTypecomboBox.currentIndex()==0:
            SimbNIBSType ='charm'
        else:
            SimbNIBSType ='headreco'
        if widget.ui.TrajectoryTypecomboBox.currentIndex()==0:
            TrajectoryType ='brainsight'
        else:
            TrajectoryType ='slicer'
        
        self.Config={}
        ComputingDevice,Backend =widget.GetSelectedComputingEngine()
        if ComputingDevice=='CPU':
            ComputingBackend=0
        elif Backend=='CUDA':
            ComputingBackend=1
        elif Backend=='OpenCL':
            ComputingBackend=2
        elif Backend=='Metal':
            ComputingBackend=3

        self.Config['ComputingBackend']=ComputingBackend
        self.Config['ComputingDevice']=ComputingDevice
        self.Config['TxSystem']=widget.ui.TransducerTypecomboBox.currentText()

        self.Config['simbnibs_path']=simbnibs_path
        self.Config['SimbNIBSType']=SimbNIBSType
        self.Config['TrajectoryType']=TrajectoryType
        self.Config['Mat4Trajectory']=Mat4Trajectory
        self.Config['ThermalProfile']=ThermalProfile
        self.Config['T1W']=T1W
        self.Config['bUseCT']=bUseCT
        self.Config['CTType']=widget.ui.CTTypecomboBox.currentIndex()
        self.Config['CoregCT_MRI']=widget.ui.CoregCTcomboBox.currentIndex()
        self.Config['CT_or_ZTE_input']=CT_or_ZTE_input
        self.Config['ID'] = os.path.splitext(os.path.split(self.Config['Mat4Trajectory'])[1])[0]

        #filenames when saving results for Brainsight
        self.Config['bInUseWithBrainsight']= bInUseWithBrainsight #this will be use to sync input and output with Brainsight
        self.Config['BrainsightSyncPath']  = _BrainsightSyncPath
        self.Config['Brainsight-Output']   = _BrainsightSyncPath+ os.sep+'Output.txt'
        self.Config['Brainsight-Target']   = _BrainsightSyncPath+ os.sep+'Output_TargetModified.txt'
        self.Config['Brainsight-ThermalOutput']  = _BrainsightSyncPath+ os.sep+'Output_Thermal.txt'

        if bInUseWithBrainsight:
            #if we are running from Brainsight
            for k in ['Brainsight-Output','Brainsight-Target','Brainsight-ThermalOutput']:
                fpath = self.Config[k]
                if os.path.isfile(fpath):
                    os.remove(fpath)

        if AltOutputFilesPath is not None:
            self.Config['OutputFilesPath']=AltOutputFilesPath
        else:
            self.Config['OutputFilesPath']=os.path.dirname(self.Config['T1W'])#+os.sep+'Babel'+os.sep+self.Config['ID']

        
        self.Config['T1WIso']= self.Config['OutputFilesPath']+os.sep+os.path.split(self.Config['T1W'])[1].replace('.nii.gz','-isotropic.nii.gz')
        with open(os.path.join(resource_path(),'version.txt'), 'r') as f:
            self.Config['version'] =f.readlines()[0]

        self.SaveLatestSelection()


        self.load_ui()
        self.InitApplication()
        self.static_canvas=None

        # Set default figure text color, works for both light and dark mode
        FIGTEXTCOLOR = np.array(self.palette().color(QPalette.WindowText).getRgb())/255.0
        plt.rcParams['text.color'] = FIGTEXTCOLOR
        plt.rcParams['axes.labelcolor'] = FIGTEXTCOLOR
        plt.rcParams['xtick.color'] = FIGTEXTCOLOR
        plt.rcParams['ytick.color'] = FIGTEXTCOLOR

        self._WorkingDialog = ClockDialog(self)
        self.moveTimer = QTimer(self)
        self.moveTimer.setSingleShot(True)
        self.moveTimer.timeout.connect(self.centerClockDialog)
        self.moveTimer.setInterval(500) 

        self.RETURN_CODE = ReturnCodes['CANCEL_OR_INCOMPLETE']

    def showClockDialog(self):
        self.centerClockDialog()
        # Show the dialog
        self._WorkingDialog.show()

    def hideClockDialog(self):
        self._WorkingDialog.hide()

    def centerClockDialog(self):
        # Calculate and set the new position for the dialog
        mainWindowCenter = self.geometry().center()
        dialogWidth = self._WorkingDialog.width()
        dialogHeight = self._WorkingDialog.height()
        self._WorkingDialog.move(
            mainWindowCenter.x() - 50,
            mainWindowCenter.y() - 50
        )

    def moveEvent(self, event):
        super().moveEvent(event)
        # Re-center the dialog when the main window moves
        if self._WorkingDialog.isVisible():
            self.moveTimer.start() # the timer will make the move of the wait dialog less clunky

    def SaveLatestSelection(self):
        if not os.path.isfile(_LastSelConfig):
            try:
                os.makedirs(os.path.split(_LastSelConfig)[0],exist_ok=True)
            except BaseException as e:
                print('Unable to save selection')
                print(e)
                return
        if os.path.isdir(os.path.split(_LastSelConfig)[0]):
            with open(_LastSelConfig,'w') as f:
                try:
                    res=yaml.safe_dump(self.Config,f)
                except BaseException as e:
                    print('Unable to save selection')
                    print(e)


    def load_ui(self):
        global GetSmallestSOS
        loader = QUiLoader()
        #path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        path = os.path.join(resource_path(), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        self.Widget = loader.load(ui_file, self)
        ui_file.close()
        ## THIS WILL BE LOADED DYNAMICALLY in function of the active Tx
        import BabelDatasetPreps as DataPreps

        from TranscranialModeling.BabelIntegrationBASE import GetSmallestSOS
        if self.Config['TxSystem'] =='Single':
            from Babel_SingleTx.Babel_SingleTx import SingleTx as WidgetAcSim
        elif self.Config['TxSystem'] =='CTX_500':
            from Babel_CTX500.Babel_CTX500 import CTX500 as WidgetAcSim
        elif self.Config['TxSystem'] =='H317':
            from Babel_H317.Babel_H317 import H317 as WidgetAcSim
        elif self.Config['TxSystem'] =='H246':
            from Babel_H246.Babel_H246 import H246 as WidgetAcSim
        elif self.Config['TxSystem'] =='BSonix':
            from Babel_SingleTx.Babel_BSonix import BSonix as WidgetAcSim
        else:
            EndWithError("TX system " + self.Config['TxSystem'] + " is not yet supported")

        from Babel_Thermal_SingleFocus.Babel_Thermal import Babel_Thermal as WidgetThermal

        new_tab = WidgetAcSim(parent=self.Widget.tabWidget,MainApp=self)
        grid_tab = QGridLayout(new_tab)
        grid_tab.setSpacing(1)
        new_tab.setLayout(grid_tab)
        new_tab.tab_name_private = "AcSim"
        self.Widget.tabWidget.addTab(new_tab, "Step 2 - Ac Sim")
        new_tab.setEnabled(False)
        self.AcSim=new_tab

        new_tab = WidgetThermal(parent=self.Widget.tabWidget,MainApp=self)
        grid_tab = QGridLayout(new_tab)
        grid_tab.setSpacing(1)
        new_tab.setLayout(grid_tab)
        new_tab.tab_name_private = "ThermalSim"
        self.Widget.tabWidget.addTab(new_tab, "Step 3 - Thermal Sim")
        new_tab.setEnabled(False)
        self.ThermalSim=new_tab

        slider= QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0.05, 1.0)
        slider.setValue((0.1, 0.6))
        ZTE=self.Widget.CTZTETabs.widget(0)
        LayRange=ZTE.findChildren(QVBoxLayout)[0]
        LayRange.addWidget(slider)
        self.Widget.ZTERangeSlider=slider
        self.Widget.setStyleSheet("QTabBar::tab::disabled {width: 0; height: 0; margin: 0; padding: 0; border: none;} ")
        print("self.Config['CTType']",self.Config['CTType'])
        if self.Config['bUseCT'] == False:
            self.Widget.CTZTETabs.hide()
        elif self.Config['CTType'] not in [2,3]:
            self.Widget.CTZTETabs.setTabEnabled(0,False)
        elif self.Config['CTType']==3: #PETRA, we change the label
            self.Widget.CTZTETabs.setTabText(0,"PETRA")
            ZTE.findChild(QLabel,"RangeLabel").setText("Normalized PETRA Range")
        self.Widget.HUTreshold=self.Widget.CTZTETabs.widget(1).findChildren(QDoubleSpinBox)[0]

        # self.Widget.TransparencyScrollBar.sliderReleased.connect(self.UpdateTransparency)
        self.Widget.TransparencyScrollBar.valueChanged.connect(self.UpdateTransparency)
        self.Widget.TransparencyScrollBar.setEnabled(False)
        self.Widget.HideMarkscheckBox.stateChanged.connect(self.HideMarks)
        
        
    
    @Slot()
    def handleOutput(self, text, stdout):
        color = self.palette().color(QPalette.WindowText)
        self.Widget.outputTerminal.moveCursor(QTextCursor.End)
        self.Widget.outputTerminal.setTextColor(color if stdout else self._err_color)
        self.Widget.outputTerminal.insertPlainText(text)
        self.Widget.outputTerminal.setTextColor(color)

    def InitApplication(self):
        '''
        Initialization of GUI controls using configuration information

        '''
     
        while self.Widget.USMaskkHzDropDown.count()>0:
            self.Widget.USMaskkHzDropDown.removeItem(0)

        for f in self.AcSim.Config['USFrequencies']:
            self.Widget.USMaskkHzDropDown.insertItem(0, '%i'%(f/1e3))

        if self.Config['TxSystem']=='Single': #for the single Tx , we use 500 kHz as default
            sel=self.Widget.USMaskkHzDropDown.findText('500')
            self.Widget.USMaskkHzDropDown.setCurrentIndex(sel)

        self.setWindowTitle('BabelBrain V'+self.Config['version'] +' - ' + self.Config['ID'] + ' - ' + self.Config['TxSystem'] +
                            ' - ' + os.path.split(self.Config['ThermalProfile'])[1].split('.yaml')[0])
        self.Widget.IDLabel.setText(self.Config['ID'])
        self.Widget.TXLabel.setText(self.Config['TxSystem'])
        self.Widget.ThermalProfileLabel.setText(os.path.split(self.Config['ThermalProfile'])[1].split('.yaml')[0])

        #we connect callbacks
        self.Widget.CalculatePlanningMask.clicked.connect(self.GenerateMask)

        self.Widget.USMaskkHzDropDown.currentIndexChanged.connect(self.UpdateParamsMaskFloat)
        self.Widget.USPPWSpinBox.valueChanged.connect(self.UpdateParamsMaskFloat)



        #Then we update the GUI and control parameters
        self.UpdateMaskParameters()

        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.handleOutput)
#        stderr = OutputWrapper(self, False)
#        stderr.outputWritten.connect(self.handleOutput)

    def UpdateMaskParameters(self):
        '''
        Update of GUI elements and parameters to be used in LFIU
        '''
        self.Widget.USMaskkHzDropDown.setProperty('UserData',float(self.Widget.USMaskkHzDropDown.currentText())*1e3)

        for obj in [self.Widget.USPPWSpinBox]:
            obj.setProperty('UserData',obj.value())



    @Slot(float)
    def UpdateParamsMaskFloat(self, newvalue):
        self.UpdateMaskParameters()


    @Slot()
    def GenerateMask(self):
        '''
        This function will produce the mask required for simulation
        '''
        Frequency=  self.Widget.USMaskkHzDropDown.property('UserData')
        BasePPW=self.Widget.USPPWSpinBox.property('UserData')

        self._prefix= self.Config['ID'] + '_' + self.Config['TxSystem'] +'_%ikHz_%iPPW_' %(int(Frequency/1e3),BasePPW)
        basedir = self.Config['OutputFilesPath']
        if not os.path.isdir(basedir):
            try:
                os.makedirs(basedir)
            except:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("Unable to create directory to save results at:\n" + basedir)
                msgBox.exec()
                raise
                
        self._prefix_path=basedir+os.sep+self._prefix
        self._outnameMask=self._prefix_path+'BabelViscoInput.nii.gz'
        
        print('outname',self._outnameMask)
        self._T1W_resampled_fname=self._outnameMask.split('BabelViscoInput.nii.gz')[0]+'T1W_Resampled.nii.gz'
        bCalcMask=False
        if os.path.isfile(self._outnameMask) and os.path.isfile(self._T1W_resampled_fname):
            ret = QMessageBox.question(self,'', "Mask file already exists.\nDo you want to recalculate?\nSelect No to reload", QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcMask=True
        else:
            bCalcMask = True

        if bCalcMask:
            #We run the Background
            self.thread = QThread()
            self.worker = RunMaskGeneration(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.UpdateMask)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)

            self.thread.start()
            self.Widget.tabWidget.setEnabled(False)
            self.showClockDialog()

        else:
            self.UpdateMask()

    #this will modify the coordinates of the trajectory
    def ExportTrajectory(self,CorX=0.0,CorY=0.0,CorZ=0.0):
        newFName=os.path.join(self.Config['OutputFilesPath'],'_mod_'+os.path.split(self.Config['Mat4Trajectory'])[1])
            
        if self.Config['TrajectoryType']=='brainsight':
            OrigTraj=ReadTrajectoryBrainsight(self.Config['Mat4Trajectory'])
            OrigTraj[0,3]-=CorX
            OrigTraj[1,3]-=CorY
            OrigTraj[2,3]-=CorZ
            with open(self.Config['Mat4Trajectory'],'r') as f:
                allLines=f.readlines()
            for n,l in enumerate(allLines):
                if 'Created by: ' in l:
                    allLines[n] = '# Created by: BabelBrain ' + self.Config['version'] +'\n' 
                if l[0]!='#':
                    break
            LastLine=l.split('\t')
            LastLine[0]='_mod_'+LastLine[0]
            LastLine[1]='%4.3f' %(OrigTraj[0,3])
            LastLine[2]='%4.3f' %(OrigTraj[1,3])
            LastLine[3]='%4.3f' %(OrigTraj[2,3])
            allLines[n]='\t'.join(LastLine)
            with open(newFName,'w') as f:
                f.writelines(allLines)
        else:
            inMat=read_itk_affine_transform(self.Config['Mat4Trajectory'])
            OrigTraj = itk_to_BSight(inMat)
            OrigTraj[0,3]+=CorX
            OrigTraj[1,3]+=CorY
            OrigTraj[2,3]+=CorZ
            transform = BSight_to_itk(OrigTraj)
            outString=templateSlicer.format(m0n0=transform[0,0],
                                        m0n1=transform[1,0],
                                        m0n2=transform[2,0],
                                        m1n0=transform[0,1],
                                        m1n1=transform[1,1],
                                        m1n2=transform[2,1],
                                        m2n0=transform[0,2],
                                        m2n1=transform[1,2],
                                        m2n2=transform[2,2],
                                        X=transform[0,3],
                                        Y=transform[1,3],
                                        Z=transform[2,3])
        
            with open(newFName,'w') as f:
                f.write(outString)
        return newFName



    def UpdateAcousticTab(self):
        self.AcSim.NotifyGeneratedMask()

    def NotifyError(self):
        self.SetErrorDomainCode()
        self.hideClockDialog()
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText("There was an error in execution -\nconsult log window for details")
        msgBox.exec()

    def UpdateMask(self):
        '''
        Refresh mask
        '''
        self.hideClockDialog()
        if self.Widget.HideMarkscheckBox.isEnabled()== False:
            self.Widget.HideMarkscheckBox.setEnabled(True)
        self.Widget.tabWidget.setEnabled(True)
        self.AcSim.setEnabled(True)
        Data=nibabel.load(self._outnameMask)
        FinalMask=Data.get_fdata()
        FinalMask=np.flip(FinalMask,axis=2)
        T1W=nibabel.load(self._T1W_resampled_fname)
        T1WData=T1W.get_fdata()
        T1WData=np.flip(T1WData,axis=2)
        self._T1WData=T1WData
        
        self._MaskData=Data
        if self.Config['bUseCT']:
            self._CTnib=nibabel.load(self._prefix_path+'CT.nii.gz')
            CTData=np.flip(self._CTnib.get_fdata(),axis=2)
        
        self._FinalMask=FinalMask
        voxSize=Data.header.get_zooms()
        x_vec=np.arange(Data.shape[0])*voxSize[0]
        x_vec-=x_vec.mean()
        y_vec=np.arange(Data.shape[1])*voxSize[1]
        y_vec-=y_vec.mean()
        z_vec=np.arange(Data.shape[2])*voxSize[2]
        z_vec-=z_vec.mean()
        LocFocalPoint=np.array(np.where(FinalMask==5)).flatten()
        self._LocFocalPoint=LocFocalPoint
        CMapXZ=FinalMask[:,LocFocalPoint[1],:].T.copy()
        CMapYZ=FinalMask[LocFocalPoint[0],:,:].T.copy()
        CMapXY=FinalMask[:,:,LocFocalPoint[2]].T.copy()
        if self.Config['bUseCT']:
            CMapXZ[CMapXZ==2]=3
            CMapYZ[CMapYZ==2]=3
            CMapXY[CMapXY==2]=3
        
        sm=plt.cm.ScalarMappable(cmap='gray')
        alpha=self.Widget.TransparencyScrollBar.value()/100.0
        T1WXZ=sm.to_rgba(T1WData[:,LocFocalPoint[1],:].T,alpha=alpha)
        T1WYZ=sm.to_rgba(T1WData[LocFocalPoint[0],:,:].T,alpha=alpha)
        T1WXY=sm.to_rgba(T1WData[:,:,LocFocalPoint[2]].T,alpha=alpha)

        sr=['y:','w:']

        plt.rcParams['font.size']=8
        extentXZ=[x_vec.min(),x_vec.max(),z_vec.max(),z_vec.min()]
        extentYZ=[y_vec.min(),y_vec.max(),z_vec.max(),z_vec.min()]
        extentXY=[x_vec.min(),x_vec.max(),y_vec.max(),y_vec.min()]

        CTMaps=[None,None,None]
        if self.Config['bUseCT']:
            CTMapXZ=CTData[:,LocFocalPoint[1],:].T
            CTMapYZ=CTData[LocFocalPoint[0],:,:].T
            CTMapXY=CTData[:,:,LocFocalPoint[2]].T
            CTMaps=[CTMapXZ,CTMapYZ,CTMapXY]

        if hasattr(self,'_figMasks'):
            for im,imTW,imCT,CMap,T1WMap,CTMap,extent in zip(self._imMasks,
                                    self._imT1W,
                                    self._imCtMasks,
                                    [CMapXZ,CMapYZ,CMapXY],
                                    [T1WXZ,T1WYZ,T1WXY],
                                    CTMaps,
                                    [extentXZ,extentYZ,extentXY]):
                imTW.set_data(T1WMap)
                im.set_data(CMap)
                if CTMap is not None:
                    Zm = np.ma.masked_where((CMap !=2) &(CMap!=3) , CTMap)
                    imCT.set_data(Zm)
                im.set_extent(extent)
            self._figMasks.canvas.draw_idle()
        else:
            
            self._imMasks=[]
            self._imT1W=[]
            self._imCtMasks=[]
            self._markers=[]

            self._figMasks = Figure(figsize=(18, 6))

            self._layout = QVBoxLayout(self.Widget.USMask)

            self.static_canvas = FigureCanvas(self._figMasks)
            
            toolbar=NavigationToolbar2QT(self.static_canvas,self)
            self._layout.addWidget(toolbar)
            self._layout.addWidget(self.static_canvas)

            axes=self.static_canvas.figure.subplots(1,3)
            self._axes=axes

            for CMap,T1WMap,CTMap,extent,static_ax,vec1,vec2,c1,c2 in zip([CMapXZ,CMapYZ,CMapXY],
                                    [T1WXZ,T1WYZ,T1WXY],
                                    CTMaps,
                                    [extentXZ,extentYZ,extentXY],
                                    axes,
                                    [x_vec,y_vec,x_vec],
                                    [z_vec,z_vec,y_vec],
                                    [LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[0]],
                                    [LocFocalPoint[2],LocFocalPoint[2],LocFocalPoint[1]]):


                self._imMasks.append(static_ax.imshow(CMap,cmap=cm.jet,extent=extent,aspect='equal'))
                if CTMap is not None:
                    Zm = np.ma.masked_where((CMap !=2) &(CMap!=3) , CTMap)
                    self._imCtMasks.append(static_ax.imshow(Zm,cmap=cm.gray,extent=extent,aspect='equal'))
                else:
                    self._imCtMasks.append(None)
                self._imT1W.append(static_ax.imshow(T1WMap,extent=extent,aspect='equal')) 
                self._markers.append(static_ax.plot(vec1[c1],vec2[c2],'+y',markersize=14)[0])
            self._figMasks.set_facecolor(np.array(self.palette().color(QPalette.Window).getRgb())/255)
        self.UpdateAcousticTab()
        self.Widget.TransparencyScrollBar.setEnabled(True)

    @Slot()
    def HideMarks(self,v):
        mc=[0.75, 0.75, 0.0,1.0]
        if self.Widget.HideMarkscheckBox.isChecked():
            mc[3] = 0.0
        for m in self._markers:
            m.set_markerfacecolor(mc)
            m.set_markeredgecolor(mc)
        self._figMasks.canvas.draw_idle()
    
    @Slot()
    def UpdateTransparency(self):
        alpha=self.Widget.TransparencyScrollBar.value()/100.0
        sm=plt.cm.ScalarMappable(cmap='gray')
        T1WXZ=sm.to_rgba(self._T1WData[:,self._LocFocalPoint[1],:].T,alpha=alpha)
        T1WYZ=sm.to_rgba(self._T1WData[self._LocFocalPoint[0],:,:].T,alpha=alpha)
        T1WXY=sm.to_rgba(self._T1WData[:,:,self._LocFocalPoint[2]].T,alpha=alpha)
        for im,T1WMap in zip(self._imT1W,
                                    [T1WXZ,T1WYZ,T1WXY]):
            im.set_data(T1WMap)
        self._figMasks.canvas.draw_idle()
            

    def GetExport(self):
        ExtraConfig ={}
        ExtraConfig['PPW']=self.Widget.USPPWSpinBox.property('UserData')
        if self.Config['bUseCT']:
            ExtraConfig['HUThreshold']=self.Widget.HUTreshold.value()
            if self.Config['CTType'] in [2,3]: #ZTE or PETRA
                ExtraConfig['ZTERange']=self.Widget.ZTERangeSlider.value()
        return self.Config | ExtraConfig
    
    ##
    def SetSuccesCode(self):
        self.RETURN_CODE = ReturnCodes['SUCCES']

    def SetErrorDomainCode(self):
        self.RETURN_CODE = ReturnCodes['ERROR_DOMAIN']

    def SetErrorAcousticsCode(self):
        self.RETURN_CODE = ReturnCodes['ERROR_ACOUSTICS']
        
    def EnableMultiPoint(self,MultiPoint):
        # triggered by the thermal pane if the profile file includes multi point focal points
        self.AcSim.EnableMultiPoint(MultiPoint)

class RunMaskGeneration(QObject):

    finished = Signal()
    endError = Signal()

    def __init__(self,mainApp):
        super(RunMaskGeneration, self).__init__()
        self._mainApp=mainApp

    def run(self):
        """Long-running task."""

        print("*"*40)
        print("*"*5+" Calculating mask.. BE PATIENT... it can take a couple of minutes...")
        print("*"*40)

        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']

        Widget=self._mainApp.Widget

        T1WIso= self._mainApp.Config['T1WIso']
        T1W= self._mainApp.Config['T1W']

        Frequency=  Widget.USMaskkHzDropDown.property('UserData')
        SmallestSoS= GetSmallestSOS(Frequency,bShear=True)

        BasePPW=Widget.USPPWSpinBox.property('UserData')
        SpatialStep=np.round(SmallestSoS/Frequency/BasePPW*1e3,3) #step of mask to reconstruct , mm
        print("Frequency, SmallestSoS, BasePPW,SpatialStep",Frequency, SmallestSoS, BasePPW,SpatialStep)

        prefix=self._mainApp._prefix
        print("Config['Mat4Trajectory']",self._mainApp.Config['Mat4Trajectory'])

        #first we ensure we have isotropic scans at 1 mm required to get affine matrix at 1.0 mm isotropic
        new_spacing = [1.0, 1.0, 1.0]
        preT1=sitk.ReadImage(T1W)
        getMin=sitk.MinimumMaximumImageFilter()
        getMin.Execute(preT1)
        minval=getMin.GetMinimum()
        original_spacing = preT1.GetSpacing()
        original_size = preT1.GetSize()
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
        preT1=sitk.Resample(preT1, 
                        new_size, 
                        sitk.Transform(),
                        sitk.sitkLinear,
                        preT1.GetOrigin(), 
                        new_spacing,
                        preT1.GetDirection(), 
                        minval,
                        preT1.GetPixelID())
        sitk.WriteImage(preT1, T1WIso)

        kargs={}
        kargs['SimbNIBSDir']=self._mainApp.Config['simbnibs_path']
        kargs['SimbNIBSType']=self._mainApp.Config['SimbNIBSType']
        kargs['CoregCT_MRI']=self._mainApp.Config['CoregCT_MRI']
        kargs['TrajectoryType']=self._mainApp.Config['TrajectoryType']
        kargs['Mat4Trajectory']=self._mainApp.Config['Mat4Trajectory'] #Path to trajectory file
        kargs['T1Source_nii']=T1W
        kargs['T1Conformal_nii']=T1WIso
        kargs['nIterationsAlign']=10
        kargs['SpatialStep']=SpatialStep
        kargs['InitialAligment']='HF'
        kargs['Location']=[0,0,0] #This coordinate will be ignored
        kargs['prefix']=prefix
        kargs['bPlot']=False
        kargs['bAlignToSkin']=True
        if self._mainApp.Config['bUseCT']:
            kargs['CT_or_ZTE_input']=self._mainApp.Config['CT_or_ZTE_input']
            kargs['CTType']=self._mainApp.Config['CTType']
            if kargs['CTType'] in [2,3]:
                kargs['ZTERange']=self._mainApp.Widget.ZTERangeSlider.value()
            kargs['HUThreshold']=self._mainApp.Widget.HUTreshold.value()
        # Start mask generation as separate process.
        queue=Queue()
        maskWorkerProcess = Process(target=CalculateMaskProcess, 
                                    args=(queue,
                                         COMPUTING_BACKEND,
                                         deviceName),
                                    kwargs=kargs)
        maskWorkerProcess.start()      
        # progress.
        T0=time.time()
        bNoError=True
        while maskWorkerProcess.is_alive():
            time.sleep(0.1)
            while queue.empty() == False:
                cMsg=queue.get()
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg:
                    bNoError=False

        maskWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            print(cMsg,end='')
            if '--Babel-Brain-Low-Error' in cMsg:
                bNoError=False
        if bNoError:
            TEnd=time.time()
            print('Total time',TEnd-T0)
            print("*"*40)
            print("*"*5+" DONE calculating mask.")
            print("*"*40)
            self.finished.emit()
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()
def main():
    if os.getenv('FSLDIR') is None:
        os.environ['FSLDIR']='/usr/local/fsl'
        os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
        os.environ['PATH']=os.environ['PATH']+':'+'/usr/local/fsl/bin'

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(10)
    
    parser = MyParser(prog='BabelBrain', usage='python %(prog)s.py [options]',description='Run BabelBrain simulation',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-bInUseWithBrainsight', action='store_true')

    args = parser.parse_args()

    app = QApplication([])
 
    selwidget = SelFiles()
    
    prevConfig=GetLatestSelection()
    
    if prevConfig is not None:
        selwidget.ui.SimbNIBSlineEdit.setText(prevConfig['simbnibs_path'])
        selwidget.ui.T1WlineEdit.setText(prevConfig['T1W'])
        selwidget.ui.TrajectorylineEdit.setText(prevConfig['Mat4Trajectory'])
        selwidget.ui.ThermalProfilelineEdit.setText(prevConfig['ThermalProfile'])
        if 'CT_or_ZTE_input' in prevConfig:
            selwidget.ui.CTlineEdit.setText(prevConfig['CT_or_ZTE_input'])
            selwidget.ui.CTTypecomboBox.setCurrentIndex(prevConfig['CTType'])
        if 'SimbNIBSType' in prevConfig:
            SimbNIBSType=prevConfig['SimbNIBSType']
            if SimbNIBSType =='charm':
                SimbNIBSTypeint=0
            else:
                SimbNIBSTypeint=1
            selwidget.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimbNIBSTypeint)
        if 'TrajectoryType' in prevConfig:
            TrajectoryType=prevConfig['TrajectoryType']
            if TrajectoryType =='brainsight':
                TrajectoryTypeint=0
            else:
                TrajectoryTypeint=1
            selwidget.ui.TrajectoryTypecomboBox.setCurrentIndex(TrajectoryTypeint)
        if 'CoregCT_MRI' in prevConfig:
            selwidget.ui.CoregCTcomboBox.setCurrentIndex(prevConfig['CoregCT_MRI'])
        if 'ComputingBackend' in prevConfig:
            if prevConfig['ComputingBackend']==0:
                Backend=''
                GPU='CPU'
            else:
                GPU=prevConfig['ComputingDevice']
                if prevConfig['ComputingBackend']==1:
                    Backend='CUDA'
                elif prevConfig['ComputingBackend']==2:
                    Backend='OpenCL'
                elif prevConfig['ComputingBackend']==3:
                    Backend='Metal'

            selwidget.SelectComputingEngine(GPU=GPU,Backend=Backend)

        if 'TxSystem' in prevConfig:
            selwidget.SelectTxSystem(prevConfig['TxSystem'])

    AltOutputFilesPath=None
    if args.bInUseWithBrainsight:
        Brainsight,header=GetInputFromBrainsight()
        assert(Brainsight is not None)
        selwidget.ui.SimbNIBSlineEdit.setText(Brainsight['simbnibs_path'])
        selwidget.ui.T1WlineEdit.setText(Brainsight['T1W'])
        selwidget.ui.TrajectorylineEdit.setText(Brainsight['Mat4Trajectory'])
        selwidget.ui.TrajectoryTypecomboBox.setCurrentIndex(0)
        AltOutputFilesPath=Brainsight['outputfiles_path']

    icon = QIcon(os.path.join(resource_path(),'Proteus-Alciato-logo.png'))
    app.setWindowIcon(icon)


    ret=selwidget.exec()
    if ret ==-1:
        sys.exit(ReturnCodes['CANCEL_OR_INCOMPLETE'])
    
    widget = BabelBrain(selwidget,
                        bInUseWithBrainsight=args.bInUseWithBrainsight,
                        AltOutputFilesPath=AltOutputFilesPath)
    widget.show()
    retcode=app.exec()
    if (retcode==0):
        sys.exit(widget.RETURN_CODE)
    else:
        return retcode

if __name__ == "__main__":

    main()
