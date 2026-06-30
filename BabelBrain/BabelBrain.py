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
import logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(name)s - %(levelname)s - %(asctime)s.%(msecs)03d - %(message)s',
#                     datefmt='%H:%M:%S')
import os
import platform
import re
import shutil
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
import psutil
import cpuinfo

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))

import SimpleITK as sitk
import nibabel
import numpy as np
import importlib
import yaml
from PySide6.QtCore import QFile, QObject, QThread, Qt, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPalette, QTextCursor, QMovie, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QLabel
)
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from GUIComponents.AppStyle import style_nav_toolbar
from nibabel import processing
from superqt import QLabeledDoubleRangeSlider

from CalculateMaskProcess import CalculateMaskProcess
from CTZTEProcessing import ConfirmPseudoCT
from ConvMatTransform import (
    BSight_to_itk,
    GetIDTrajectoryBrainsight,
    ReadTrajectoryBrainsight,
    GetBrainSightHeader,
    itk_to_BSight,
    templateSlicer,
    read_itk_affine_transform,
)
from SelFiles.SelFiles import SelFiles,ValidThermalProfile

from Options.Options import AdvancedOptions, OptionalParams
from ClockDialog import ClockDialog
from GUIComponents.nifti_viewer import NiftiViewerWindow

from Telemetry.Telemetry import send_telemetry
from datetime import datetime, timezone


multiprocessing.freeze_support()
if sys.platform =='linux':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


bINUSE_INSIDE_BRAINSIGHT = False




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
        if bINUSE_INSIDE_BRAINSIGHT == False:
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

_LocationInstallID = str(Path.home())+os.sep+os.path.join('.config','BabelBrain','installation.id')

_date_session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

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
    return res

def SaveTelemetryLevelToConfig(level):
    '''
    Persist the telemetry level into the user's lastselection.yaml so the
    first-launch consent dialog is not shown again. Used before the BabelBrain
    main widget is constructed (which is what normally owns SaveLatestSelection).
    '''
    try:
        os.makedirs(os.path.split(_LastSelConfig)[0], exist_ok=True)
    except BaseException as e:
        print('Unable to save telemetry level')
        print(e)
        return
    cfg = GetLatestSelection() or {}
    cfg['TelemetryLevel'] = int(level)
    try:
        with open(_LastSelConfig, 'w') as f:
            yaml.safe_dump(cfg, f)
    except BaseException as e:
        print('Unable to save telemetry level')
        print(e)

def _styled_dialog_parent():
    """Return a stylesheet-carrying widget so QMessageBoxes raised from
    module-level helpers (no `self` in scope) still inherit the compact app
    font. The programmatic forms (MainForm, Tx panels) carry the stylesheet
    under objectName "Widget"; fall back to the active window itself."""
    app = QApplication.instance()
    win = app.activeWindow() if app else None
    if win is None:
        return None
    return win.findChild(QWidget, "Widget") or win


def GetInputFromBrainsight():
    '''
    Reads and validates input files exported from Brainsight for use in BabelBrain.

    Returns
    -------
    tuple
        res : dict or None
            Dictionary with paths and configuration if all files are valid, otherwise None.
        header : dict
            Header information from the Brainsight trajectory file.

    Raises
    ------
    SystemError
        If required input files are missing or invalid.
    '''
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
            if header['Version'] not in ['14','15']:
                msgBox = QMessageBox(_styled_dialog_parent())
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
    '''
    Display an error message and raise a SystemError.

    Parameters
    ----------
    msg : str
        The error message to display.
    '''
    msgBox = QMessageBox(_styled_dialog_parent())
    msgBox.setIcon(QMessageBox.Critical)
    msgBox.setText(msg)
    msgBox.exec()
    raise SystemError(msg)

def save_T1W_iso(T1W_fname,T1WIso_fname,new_spacing=[1.0,1.0,1.0]):
    '''
    Resample a T1-weighted MRI image to isotropic voxel spacing and save to file.

    Parameters
    ----------
    T1W_fname : str
        Path to the input T1-weighted image.
    T1WIso_fname : str
        Path to save the isotropic resampled image.
    new_spacing : list of float, optional
        Desired voxel spacing (default is [1.0, 1.0, 1.0]).
    '''
    preT1=sitk.ReadImage(T1W_fname)
    getMin=sitk.MinimumMaximumImageFilter()
    getMin.Execute(preT1)
    minval=getMin.GetMinimum()
    original_spacing = preT1.GetSpacing()
    original_size = preT1.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    preT1=sitk.Resample(preT1, 
                    new_size, 
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    preT1.GetOrigin(),
                    new_spacing,
                    preT1.GetDirection(), 
                    minval,
                    preT1.GetPixelID())
    sitk.WriteImage(preT1, T1WIso_fname)


_BrainsightSyncPath = str(Path.home()) + os.sep + '.BabelBrainSync'

######################

class BabelBrain(QWidget):
    '''
    Main TUS simulation application

    '''

    def __init__(self,widget,bInUseWithBrainsight=False,AltOutputFilesPath=None):
        super(BabelBrain, self).__init__()
        #This file will store the last config selected

        self._AllTransducers = widget.GetAllTransducers()

        simbnibs_path=widget.ui.SimbNIBSlineEdit.text()
        T1W=widget.ui.T1WlineEdit.text()
        CT_or_ZTE_input=widget.ui.CTlineEdit.text()
        bUseCT=widget.ui.CTTypecomboBox.currentIndex()>0
        CTType=widget.ui.CTTypecomboBox.currentIndex()
        CTMapCombo = widget._dfCTParams.iloc[widget.ui.CTMappingcomboBox.currentIndex()].name
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
        
        prevConfig = GetLatestSelection()
        if prevConfig is None:
            self.Config={}
        else:
            self.Config=prevConfig
        ComputingDevice,Backend =widget.GetSelectedComputingEngine()
        if ComputingDevice=='CPU':
            ComputingBackend=0
        elif Backend=='CUDA':
            ComputingBackend=1
        elif Backend=='OpenCL':
            ComputingBackend=2
        elif Backend=='Metal':
            ComputingBackend=3
        elif Backend=='MLX':
            ComputingBackend=4

        self.Config['bUseRayleighForWater']= True
        self.Config['ComputingBackend']=ComputingBackend
        self.Config['ComputingDevice']=ComputingDevice
        self.Config['TxSystem']=widget.ui.TransducerTypecomboBox.currentText()

        self.Config['simbnibs_path']=simbnibs_path
        self.Config['SimbNIBSType']=SimbNIBSType
        self.Config['TrajectoryType']=TrajectoryType
        self.Config['Mat4Trajectory']=Mat4Trajectory
        self.Config['OrigMat4Trajectory']=Mat4Trajectory
        self.Config['ThermalProfile']=ThermalProfile
        self.Config['T1W']=T1W
        self.Config['bUseCT']=bUseCT
        self.Config['CTType']=CTType
        self.Config['CoregCT_MRI']=widget.ui.CoregCTcomboBox.currentIndex()
        self.Config['CT_or_ZTE_input']=CT_or_ZTE_input
        self.Config['CTMapCombo']=CTMapCombo
        self.Config['NumberTransducers']=1 
        if self.Config['TrajectoryType']=='brainsight':
            ID=ReadTrajectoryBrainsight(self.Config['Mat4Trajectory'],bGetID=True)[1]
            if type(ID) is list:
                self.Config['NumberTransducers']=len(ID) #multiple devices
            else:
                ID=[ID] #we enforce a list of 1 ID to simpliy processing
            self.Config['ID'] = ID
        else:
            #for 3DSlicer, we will limit to only one - the time we found a better strategy
            self.Config['ID'] = [os.path.splitext(os.path.split(self.Config['Mat4Trajectory'])[1])[0]]
            

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

        
        self.Config['T1WIso'] = self.Config['OutputFilesPath'] + os.sep + re.sub(r'\.nii(\.gz)?$', '', os.path.split(self.Config['T1W'])[1]) + '-isotropic.nii.gz'
        with open(os.path.join(resource_path(),'version-gui.txt'), 'r') as f:
            self.Config['version'] =f.readlines()[0]

        self.Config['MultiPoint']=''
        self.Config['EnableMultiPoint']=False
        if widget.ui.MultiPointTypecomboBox.currentIndex()==1:
            self.Config['EnableMultiPoint']=True    
            self.Config['MultiPoint']=widget.ui.MultiPointlineEdit.text()
            
        #default values for advanced features 

        self._DefaultOptions=OptionalParams(self._AllTransducers)  
        
        for k in self._DefaultOptions.keys():
            if k not in self.Config:
                self.Config[k]=getattr(self._DefaultOptions,k)
            elif k=='TxOptimizedWeights':
                #we check if we are missing a Tx
                for tx in self._AllTransducers:
                    if tx not in self.Config['TxOptimizedWeights']:
                        self.Config['TxOptimizedWeights'][tx]=getattr(self._DefaultOptions,k)[tx]
            
        self.Config['AdvancedParamsFile']=self.Config['OutputFilesPath']+os.sep+os.path.split(self.Config['T1W'])[1].split('.nii')[0]+'-AdvancedParams.yaml'
        self.SaveLatestSelection()

        self.load_ui()
        #in case of multipoint , we prepared 
        if len(self.Config['MultiPoint'].strip())>0 and\
            self.Config['EnableMultiPoint']:
            with open(self.Config['MultiPoint'],'r') as f:
                profile=yaml.safe_load(f)
            for n in range(len(profile['MultiPoint'])):
                #we convert to mm
                for k in ['X','Y','Z']:
                    profile['MultiPoint'][n][k]=profile['MultiPoint'][n][k] * 1e-3 #in mm
            self.AcSim.EnableMultiPoint(profile['MultiPoint'])
            self.ThermalSim.EnableMultiPoint()
        self.InitApplication()
        
        # Set default figure text color, works for both light and dark mode
        FIGTEXTCOLOR = np.array(self.palette().color(QPalette.WindowText).getRgb())/255.0
        plt.rcParams['text.color'] = FIGTEXTCOLOR
        plt.rcParams['axes.labelcolor'] = FIGTEXTCOLOR
        plt.rcParams['xtick.color'] = FIGTEXTCOLOR
        plt.rcParams['ytick.color'] = FIGTEXTCOLOR

        # Error text color for outputTerminal (red on both light and dark backgrounds)
        self._err_color = QColor(Qt.red)

        self._WorkingDialog = ClockDialog(self)
        self.moveTimer = QTimer(self)
        self.moveTimer.setSingleShot(True)
        self.moveTimer.timeout.connect(self.centerClockDialog)
        self.moveTimer.setInterval(500) 

        self.RETURN_CODE = ReturnCodes['CANCEL_OR_INCOMPLETE']

        self._TrackingTime={'Calculation time domain':0.0,
                            'Calculation time ultrasound':0.0,
                            'Calculation time thermal':0.0}
        self._NiftiCT = None
        self._NiftiAirMask = None

        self._TelmetryMsgs=[]
        self._TimeStart = time.time()

        self.LogTelemetry('CTS:L1: BabelBrain started')
        self.LogTelemetry('CTS:L1: OS: '+ platform.platform())

        mem = psutil.virtual_memory()
        info = cpuinfo.get_cpu_info()
        print(f'CPU: {info['brand_raw']}')
        print(f'Total memory:{mem.total / 2**30:.2f} GB')
        self.LogTelemetry(f'CTS:L1: CPU: {info['brand_raw']}')
        self.LogTelemetry(f'CTS:L1: Memory: {mem.total / 2**30:.2f} GB')
        self.LogTelemetry(f'CTS:L1: GPU: {Backend} {ComputingDevice}')
        self.LogTelemetry(f'CTS:L4: Device: {self.Config['TxSystem']}')
        self.SendTelemetry()
        
        
    def showEvent(self, event):
        super().showEvent(event)
        self.centerOnScreen()

    def closeEvent(self, event):
        self.LogTelemetry('CTS:L1: BabelBrain closing')
        self.SendTelemetry(waittocomplete=True)
        if hasattr(self,'_vtk_visualization'):
            self._vtk_visualization.close()
        super().closeEvent(event)  # let the default close logic run

    def centerOnScreen(self):
        # Get the screen geometry where the window is currently shown
        screen = self.screen().geometry()
        # Get the window geometry (including title bar, etc.)
        frame = self.frameGeometry()
        # Move the center of the frame to the screen center
        frame.moveCenter(screen.center())
        # Move the top-left point of the window to match
        self.move(frame.topLeft())
        
    def bHasTxWeights(self):
        '''
        Returns True if the current transducer has optimized weights
        '''
        return len(self.Config['TxOptimizedWeights'][self.Config['TxSystem']])>0
        
    def showClockDialog(self):
        self.centerClockDialog()
        # Show the dialog
        self._WorkingDialog.show()

    def hideClockDialog(self):
        self._WorkingDialog.hide()

    def centerClockDialog(self):
        # Calculate and set the new position for the dialog
        mainWindowCenter = self.geometry().center()
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

    def AddConfigInformation(self,key,entry):
        self.Config[key]=entry
        self.SaveLatestSelection()

    def load_ui(self):
        global GetSmallestSOS

        # Top-level form is now built programmatically — see MainForm.py.
        # Widget attribute names (tabWidget, IDLabel, CTZTETabs, USMask, …)
        # are preserved so the rest of this file references them unchanged.
        from MainForm import BabelBrainMainForm
        self.Widget = BabelBrainMainForm(self)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.Widget)

        ## THIS WILL BE LOADED DYNAMICALLY in function of the active Tx
        import BabelDatasetPreps as DataPreps

        from TranscranialModeling.BabelIntegrationBASE import GetSmallestSOS
        if self.Config['TxSystem'] =='CTX_500':
            idimport = 'CTX500'
            ibsub=idimport
        elif self.Config['TxSystem'] =='CTX_500':
            idimport = 'CTX500'
            ibsub=idimport
        elif self.Config['TxSystem'] =='CTX_250':
            idimport = 'CTX250'
            ibsub=idimport
        elif self.Config['TxSystem'] =='CTX_250_2ch':
            idimport = 'CTX250_2ch'
            ibsub=idimport
        elif self.Config['TxSystem'] =='DPX_500':
            idimport = 'DPX500'
            ibsub=idimport
        elif self.Config['TxSystem'] =='DPXPC_300':
            idimport = 'DPXPC300'
            ibsub=idimport
        elif self.Config['TxSystem'] =='Single':
            idimport = 'SingleTx'
            ibsub=idimport
        elif self.Config['TxSystem'] =='BSonix':
            idimport = 'SingleTx'
            ibsub='BSonix'
        else:
            idimport = self.Config['TxSystem']
            ibsub=idimport
        try:
            WidgetAcSim = importlib.import_module(f"Babel_{idimport}.Babel_{ibsub}").__dict__[ibsub]
        except ImportError:
            EndWithError("TX system " + self.Config['TxSystem'] + " is not yet supported")

        from Babel_Thermal.Babel_Thermal import Babel_Thermal as WidgetThermal

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
        # Insert before the layout's trailing stretch so the slider aligns to the
        # top of the tab (like the CT tab's HU threshold), not the bottom.
        LayRange.insertWidget(LayRange.count()-1, slider)
        self.Widget.ZTERangeSlider=slider
        # NB: do not call self.Widget.setStyleSheet() here — it would replace
        # (not extend) the compact _FORM_QSS applied in BabelBrainMainForm,
        # which already includes the QTabBar::tab::disabled rule.
        print("self.Config['CTType']",self.Config['CTType'])
        if self.Config['bUseCT'] == False:
            self.Widget.CTZTETabs.hide()
        elif self.Config['CTType'] not in [2,3]:
            self.Widget.CTZTETabs.setTabEnabled(0,False)
        if self.Config['CTType']==3: #PETRA, we change the label
            print('doing density selection')
            self.Widget.CTZTETabs.setTabText(0,"PETRA")
            ZTE.findChild(QLabel,"RangeLabel").setText("Normalized PETRA Range")
        elif self.Config['CTType']==4: #Density, we change a little labels and limits
            self.Widget.CTZTETabs.setTabText(1,"Density")
            self.Widget.CTZTETabs.widget(1).findChild(QLabel,"HULabel").setText("Density threshold")
            self.Widget.HUThresholdSpinBox.setMinimum(1050)
            self.Widget.HUThresholdSpinBox.setMaximum(3000)
            self.Widget.HUThresholdSpinBox.setValue(1200)
            
        self.Widget.HUTreshold=self.Widget.CTZTETabs.widget(1).findChildren(QDoubleSpinBox)[0]

        # self.Widget.TransparencyScrollBar.sliderReleased.connect(self.UpdateTransparency)
        self.Widget.TransparencyScrollBar.valueChanged.connect(self.UpdateTransparency)
        self.Widget.TransparencyScrollBar.setEnabled(False)
        self.Widget.HideMarkscheckBox.stateChanged.connect(self.HideMarks)

        if self.Config['TxSystem'] =='Single':
            USMaskkHzDropDown = self.Widget.USMaskkHzDropDown
            USMaskkHzDropDown.setEditable(True)
            USMaskkHzDropDown.lineEdit().textChanged.connect(self.StartManualMaskFrequency)
            USMaskkHzDropDown.lineEdit().editingFinished.connect(self.UpdateManualMaskFrequency)


    @Slot()
    def StartManualMaskFrequency(self,txt):
        try:
            value = float(txt)
            if value <200 or value > 1000:
                self.Widget.CalculatePlanningMask.setEnabled(False)
            else:
                self.Widget.CalculatePlanningMask.setEnabled(True)
        except ValueError:
            self.Widget.CalculatePlanningMask.setEnabled(False)

    @Slot()
    def UpdateManualMaskFrequency(self):
        try:
            value = float(self.Widget.USMaskkHzDropDown.currentText())
            if value <200 or value > 1000:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid frequency in kHz.")
                return
            self.UpdateMaskParameters()
            self.Widget.CalculatePlanningMask.setEnabled(True)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid frequency in kHz.")
            self.Widget.USMaskkHzDropDown.setFocus()

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

        self.UpdateWindowTitle()

        #we connect callbacks
        self.Widget.CalculatePlanningMask.clicked.connect(self.GenerateMask)

        self.Widget.USMaskkHzDropDown.currentIndexChanged.connect(self.UpdateFrequencyFloat)
        self.Widget.USPPWSpinBox.valueChanged.connect(self.UpdateParamsMaskFloat)

        if self.AcSim.Config['USFrequencies'][0]<=350e3: #we set default to 9 PPW for low frequencies
            self.Widget.USPPWSpinBox.setValue(9)
        
        self.Widget.AdvancedOptions.clicked.connect(self.ShowAdvancedOptions)

        #Then we update the GUI and control parameters
        self.UpdateFrequencyFloat(0)
        self.UpdateMaskParameters()

        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.handleOutput)
#        stderr = OutputWrapper(self, False)
#        stderr.outputWritten.connect(self.handleOutput)

        self.Widget.vtkVisualizationqPushButton.clicked.connect(self.OpenVTKVisualization)
        self.Widget.vtkVisualizationqPushButton.setEnabled(False)

    def UpdateWindowTitle(self):
        IDTitle='+'.join(self.Config['ID'])
        self.setWindowTitle('BabelBrain V'+
                            self.Config['version'] +' - ' + 
                            IDTitle + ' - ' + 
                            self.Config['TxSystem'] + ' - ' + 
                            os.path.split(self.Config['ThermalProfile'])[1].split('.yaml')[0])
        self.Widget.IDLabel.setText(IDTitle)
        self.Widget.TXLabel.setText(self.Config['TxSystem'])
        self.Widget.ThermalProfileLabel.setText(os.path.split(self.Config['ThermalProfile'])[1].split('.yaml')[0])
    
    def UpdateThermalProfile(self,fThermalProfile):
        bValid,msg = ValidThermalProfile(fThermalProfile)
        if not bValid:
            print('bValid,msg',bValid,msg)
            msgBox = QMessageBox(self.Widget)
            msgBox.setText("Please indicate valid entries in the profile")
            msgBox.setDetailedText(msg)
            msgBox.exec()
            return False
        else:
            self.Config['ThermalProfile']=fThermalProfile
            self.UpdateWindowTitle()
            self.SaveLatestSelection()
            return True

    def UpdateMaskParameters(self):
        '''
        Update of GUI elements and parameters to be used in TUS
        '''
        self.Widget.USMaskkHzDropDown.setProperty('UserData',float(self.Widget.USMaskkHzDropDown.currentText())*1e3)

        for obj in [self.Widget.USPPWSpinBox]:
            obj.setProperty('UserData',obj.value())

        if self.Config['TxSystem']=='DomeTx':
            #for the DomeTx we have only some limited PPW
            if float(self.Widget.USMaskkHzDropDown.currentText())!=220:
                self.Widget.USPPWSpinBox.setValue(6)
                self.Widget.USPPWSpinBox.setMinimum(6)
                self.Widget.USPPWSpinBox.setMaximum(6)
            else:
                self.Widget.USPPWSpinBox.setMinimum(6)
                self.Widget.USPPWSpinBox.setMaximum(12)
                self.Widget.USPPWSpinBox.setValue(9)

    @Slot()
    def ShowAdvancedOptions(self):
        
        options = AdvancedOptions(self.Config,
                                 self.AcSim.Config,
                                 self._DefaultOptions,
                                 self._AllTransducers,
                                 parent=self)
        ret=options.exec()
        if hasattr(options,'NewValues'):
            for k in options.NewValues.keys():
                self.Config[k]=getattr(options.NewValues,k)
            self.SaveLatestSelection()

    @Slot(float)
    def UpdateFrequencyFloat(self, newvalue):
        if float(self.Widget.USMaskkHzDropDown.currentText())>350:
            self.Widget.USPPWSpinBox.setValue(6)
        else:
            self.Widget.USPPWSpinBox.setValue(9)
        self.UpdateMaskParameters()

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
        self._Frequency =Frequency
        self._BasePPW =BasePPW

    
        #we prepare paths
        self._prefix=[ p + '_' + self.Config['TxSystem'] +'_%ikHz_%iPPW_' %(int(Frequency/1e3),BasePPW) for p in self.Config['ID']]
        self._merged_prefix = '+'.join(self.Config['ID']) + self.Config['TxSystem'] +'_%ikHz_%iPPW_' %(int(Frequency/1e3),BasePPW)
        self._prefix_path=[self.Config['OutputFilesPath']+os.sep+p for p in self._prefix]
        self._merged_prefix_path = self.Config['OutputFilesPath']+os.sep+self._merged_prefix
        self._outnameMask=[p+'BabelViscoInput.nii.gz' for p in self._prefix_path]
        self._T1W_resampled_fname=[p.split('BabelViscoInput.nii.gz')[0]+'T1W_Resampled.nii.gz' for p in self._outnameMask]
        
        self._MaskNib =[None]*len(self.Config['ID'])
        self._T1WNib  =[None]*len(self.Config['ID'])
        self._NiftiCT =[None]*len(self.Config['ID'])
        self._NiftiAirMask =[None]*len(self.Config['ID'])
        self.FinalMaskRaw =[None]*len(self.Config['ID'])
        self._FinalMask =[None]*len(self.Config['ID'])
        self._T1WDataRaw =[None]*len(self.Config['ID'])
        self._NiftiSkull =[None]*len(self.Config['ID'])
        self._NiftiWater =[None]*len(self.Config['ID'])

        combinedID='+'.join(self.Config['ID'])
        self._trackingtimefile = self.Config['OutputFilesPath']+os.sep+combinedID+'_%ikHz_%iPPW_' %(int(Frequency/1e3),BasePPW)+'ExecutionTimes.yml'

        self._TrajectoryNumber=0
        #we run first trajectory, most cases it will be only one
        self.ExecuteTrajectory()

    def ExecuteTrajectory(self):
        
        basedir = self.Config['OutputFilesPath']
        if not os.path.isdir(basedir):
            try:
                os.makedirs(basedir)
            except:
                msgBox = QMessageBox(self.Widget)
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setText("Unable to create directory to save results at:\n" + basedir)
                msgBox.exec()
                raise
                
        if not os.path.isfile(self._trackingtimefile):
            self.UpdateComputationalTime('domain',0.0) #this will initalize the trackig file
        
        print('outname',self._outnameMask[self._TrajectoryNumber])
        bCalcMask=False
        if os.path.isfile(self._outnameMask[self._TrajectoryNumber]) and os.path.isfile(self._T1W_resampled_fname[self._TrajectoryNumber]):
            # Parent to self.Widget (the styled MainForm) so the dialog inherits
            # the compact _FORM_QSS; self is the top-level app and carries no
            # stylesheet of its own.
            ret = QMessageBox.question(self.Widget,'', "Mask file already exists.\nDo you want to recalculate?\nSelect No to reload", QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                bCalcMask=True
        else:
            bCalcMask = True

        if bCalcMask:
            #We run the Background
            # Reset the chaining flag; VerifyResults sets it True only on a
            # successful run that should be followed by another trajectory.
            self._bRunNextTrajectory = False
            self.thread = QThread()
            self.worker = RunMaskGeneration(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.VerifyResults)
            self.worker.finished.connect(self.SendTelemetry)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # Chain the next trajectory only once the thread's event loop has
            # actually exited (thread.finished), never from worker.finished —
            # at that earlier point thread.quit() is still queued and the
            # thread is alive, so starting a new run would deadlock/crash.
            self.thread.finished.connect(self._StartNextTrajectory)


            self.worker.endError.connect(self.NotifyError)
            self.worker.endError.connect(self.SendTelemetry)
            self.worker.endError.connect(self.thread.quit)
            self.worker.endError.connect(self.worker.deleteLater)

            self.worker.logTelemetry.connect(self.LogTelemetry)

            self.thread.start()
            self.Widget.tabWidget.setEnabled(False)
            self.showClockDialog()

        else:
            self.UpdateMask()
            # Reload path: no worker thread is spawned, so advance to the next
            # trajectory directly.
            if self._TrajectoryNumber<len(self.Config['ID'])-1:
                self._TrajectoryNumber+=1
                self.ExecuteTrajectory()


    #this will modify the coordinates of the trajectory
    def ExportTrajectory(self,CorX=0.0,CorY=0.0,CorZ=0.0,Ntraj=0):
        newFName=os.path.join(self.Config['OutputFilesPath'],'_mod_'+os.path.split(self.Config['Mat4Trajectory'])[1])
            
        if self.Config['TrajectoryType']=='brainsight':
            OrigTraj=ReadTrajectoryBrainsight(self.Config['Mat4Trajectory'])
            if len(OrigTraj.shape)==3:
                OrigTraj=OrigTraj[:,:,Ntraj]
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
            OrigTraj[0,3]-=CorX
            OrigTraj[1,3]-=CorY
            OrigTraj[2,3]-=CorZ
            transform = BSight_to_itk(OrigTraj)
            transform[:3,:3]=transform[:3,:3].T
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
        if 'BABEL_PYTEST' not in os.environ:
            msgBox = QMessageBox(self.Widget)
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("There was an error in execution -\nconsult log window for details")
            msgBox.exec()
        else:
            #this will unblock for PyTest
            self.testing_error = True
            self.Widget.tabWidget.setEnabled(True)

    def VerifyResults(self,output_files):
        self.hideClockDialog()
        if self.Config['bUseCT']:
            if self.Config['CTType'] in [2,3]:
                if not ConfirmPseudoCT(output_files['pCTfname']):
                    self.UpdateMask(bDeleteOnly=True)
                    return
        self.UpdateMask()
        # Only flag that the next trajectory should run; the actual launch is
        # deferred to _StartNextTrajectory, fired by thread.finished once the
        # current worker thread has fully stopped.
        if self._TrajectoryNumber<len(self.Config['ID'])-1:
            self._bRunNextTrajectory = True

    def _StartNextTrajectory(self):
        '''
        Invoked from thread.finished (current worker thread fully stopped).
        Carries over to the next trajectory if VerifyResults flagged one.
        '''
        if getattr(self,'_bRunNextTrajectory',False):
            self._bRunNextTrajectory = False
            self._TrajectoryNumber += 1
            self.ExecuteTrajectory()

    def _ensurePlanningTabs(self):
        '''
        Build (or rebuild) the per-trajectory tab container inside USMask.

        One tab is created per entry in self.Config['ID'], using the ID as the
        tab title. Each tab owns its own matplotlib figure/canvas; the artists
        and per-trajectory state are stored in self._trajPanels[i]. With a single
        trajectory (the common case) the tab bar and pane frame are hidden so the
        view matches the original single-panel look.
        '''
        IDs = list(self.Config['ID'])
        if not hasattr(self,'_planningTabs'):
            outer = QVBoxLayout(self.Widget.USMask)
            outer.setContentsMargins(0, 0, 0, 0)
            self._planningTabs = QTabWidget(self.Widget.USMask)
            # Keep trajectory names fully visible: never elide the labels, and
            # let the bar scroll (rather than squeeze tabs into "...") when many
            # trajectories don't fit the available width.
            tab_bar = self._planningTabs.tabBar()
            tab_bar.setElideMode(Qt.ElideNone)
            tab_bar.setExpanding(False)
            self._planningTabs.setUsesScrollButtons(True)
            outer.addWidget(self._planningTabs)
            self._trajPanels = []
            self._trajPanelIDs = None

        if self._trajPanelIDs != IDs:
            # Trajectory set changed — discard old tabs and rebuild from scratch.
            while self._planningTabs.count():
                w = self._planningTabs.widget(0)
                self._planningTabs.removeTab(0)
                if w is not None:
                    w.deleteLater()
            self._trajPanels = []
            for tid in IDs:
                tab = QWidget()
                lay = QVBoxLayout(tab)
                lay.setContentsMargins(0, 0, 0, 0)
                self._planningTabs.addTab(tab, str(tid))
                self._trajPanels.append({'layout': lay, 'figure': None})
            self._trajPanelIDs = IDs

        # Single trajectory (the common case): hide the tab bar and drop the pane
        # frame so Step 1 looks like the original single-panel view.  Several
        # trajectories: show the tab bar with only a top line under the tab row.
        if len(IDs) == 1:
            self._planningTabs.tabBar().setVisible(False)
            self._planningTabs.setStyleSheet("QTabWidget::pane { border: 0px; }")
        else:
            self._planningTabs.tabBar().setVisible(True)
            self._planningTabs.setStyleSheet(
                "QTabWidget::pane { border: 0px; border-top: 1px solid palette(mid); }")

    def _showMatplotlibVisualization(self,bDeleteOnly:bool):
        AirMask=None
        if self.Config['bUseCT']:
            CTData=np.flip(self._NiftiCT[self._TrajectoryNumber].get_fdata(dtype=np.float32),axis=2)
            if self.Config['bExtractAirRegions'] and os.path.exists(self._prefix_path[self._TrajectoryNumber]+'AirRegions.nii.gz'):
                AirMask=np.flip(self._NiftiAirMask[self._TrajectoryNumber].get_fdata(dtype=np.float32),axis=2)
        
        FinalMask=np.flip(self.FinalMaskRaw[self._TrajectoryNumber],axis=2)
        T1WData=np.flip(self._T1WDataRaw[self._TrajectoryNumber],axis=2)
        voxSize=self._T1WNib[self._TrajectoryNumber].header.get_zooms()
        x_vec=np.arange(self._T1WNib[self._TrajectoryNumber].shape[0])*voxSize[0]
        x_vec-=x_vec.mean()
        y_vec=np.arange(self._T1WNib[self._TrajectoryNumber].shape[1])*voxSize[1]
        y_vec-=y_vec.mean()
        z_vec=np.arange(self._T1WNib[self._TrajectoryNumber].shape[2])*voxSize[2]
        z_vec-=z_vec.mean()
        LocFocalPoint=np.array(np.where(FinalMask==5)).flatten()
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
        AirMaps=[None,None,None]
        if self.Config['bUseCT']:
            CTMaps=[CTData[:,LocFocalPoint[1],:].T,
                    CTData[LocFocalPoint[0],:,:].T,
                    CTData[:,:,LocFocalPoint[2]].T]
            if AirMask is not None:
                AirMaps=[AirMask[:,LocFocalPoint[1],:].T,
                        AirMask[LocFocalPoint[0],:,:].T,
                        AirMask[:,:,LocFocalPoint[2]].T]
                
        # Render this trajectory's plots into its own tab so multiple
        # trajectories can be visualized side by side. The active trajectory
        # (self._TrajectoryNumber) selects which tab is (re)populated.
        self._ensurePlanningTabs()
        panel = self._trajPanels[self._TrajectoryNumber]

        # Clear any previous content (toolbar + canvas) from this tab.
        while (child := panel['layout'].takeAt(0)) is not None:
            w = child.widget()
            if w is not None:
                w.deleteLater()
        panel['figure'] = None

        if bDeleteOnly:
            self.AcSim.setEnabled(False)
            self.ThermalSim.setEnabled(False)
            return
        self._imMasks=[]
        self._imT1W=[]
        self._imCtMasks=[]
        self._markers=[]

        self._figMasks = Figure(figsize=(18, 6))

        self.static_canvas = FigureCanvas(self._figMasks)

        toolbar=style_nav_toolbar(NavigationToolbar2QT(self.static_canvas,self))
        panel['layout'].addWidget(toolbar)
        panel['layout'].addWidget(self.static_canvas)

        axes=self.static_canvas.figure.subplots(1,3)
        # Shrink the plotting area on the right (keeping the default left margin
        # so the leftmost subplot's y labels aren't clipped). This shifts the
        # subplots slightly left — more centered — and reserves room on the
        # right so the legend anchored past the last subplot isn't clipped.
        self._figMasks.subplots_adjust(right=0.86, wspace=0.22)
        self._axes=axes

        for CMap,T1WMap,CTMap,AirMap,extent,static_ax,vec1,vec2,c1,c2 in zip([CMapXZ,CMapYZ,CMapXY],
                                [T1WXZ,T1WYZ,T1WXY],
                                CTMaps,
                                AirMaps,
                                [extentXZ,extentYZ,extentXY],
                                axes,
                                [x_vec,y_vec,x_vec],
                                [z_vec,z_vec,y_vec],
                                [LocFocalPoint[0],LocFocalPoint[1],LocFocalPoint[0]],
                                [LocFocalPoint[2],LocFocalPoint[2],LocFocalPoint[1]]):

            if self._bSegmentedBrain :
                vmaxMask=8
            else:
                vmaxMask=5
            self._imMasks.append(static_ax.imshow(CMap,cmap=cm.jet,vmin=0,vmax=vmaxMask,extent=extent,interpolation='none',aspect='equal'))
            if CTMap is not None:
                Zm = np.ma.masked_where((CMap !=2) &(CMap!=3) , CTMap)
                self._imCtMasks.append(static_ax.imshow(Zm,cmap=cm.gray,extent=extent,aspect='equal'))               
            else:
                self._imCtMasks.append(None)

            if AirMap is not None:
                Zm = np.ma.masked_where(AirMap==0 , AirMap)
                cmap = ListedColormap(['black', (223/255,199/255,224/255,1.0)]) 
                self._imCtMasks.append(static_ax.imshow(Zm,cmap=cmap,vmin=0,vmax=1,extent=extent,aspect='equal'))
            self._imT1W.append(static_ax.imshow(T1WMap,extent=extent,aspect='equal')) 
            self._markers.append(static_ax.plot(vec1[c1],vec2[c2],'+y',markersize=14)[0])
        im = self._imMasks[-1]
        if self.Config['bUseCT']:
            if self._bSegmentedBrain :
                values =[1,4,6,7,8]
                legends  = ['scalp','brain-n.s','white m.','gray m.','CSF']
                colors =[(0.0, 0.3, 1.0, 1.0), 
                        (0.4863,  1.0,  0.4745,   1.0),
                        (1.0,  0.5804,   0.0,  1.0),
                        (1.0,  0.1137,   0.0,  1.0),
                        (0.4980, 0.0,   0.0,    1.0)]
            else:
                values =[1,4]
                legends  = ['scalp','brain']
                colors =[(0.0, 0.3, 1.0, 1.0), 
                     (1.0, 0.40740740740740755,0.0, 1.0)]
            #we use manual color asignation 
            
        else:
            if self._bSegmentedBrain :
                values =[1,2,3,4,6,7,8]
                legends  = ['scalp','cort.','trab.','brain-n.s','white m.','gray m.','CSF']
                colors = [(0.0, 0.3, 1.0, 1.0), 
                        (0.0, 0.5020, 1.0, 1.0),
                        (0.0824,  1.0,  0.8824, 1.0),
                        (0.4863,  1.0,  0.4745,   1.0),
                        (1.0,  0.5804,   0.0,  1.0),
                        (1.0,  0.1137,   0.0,  1.0),
                        (0.4980, 0.0,   0.0,    1.0)]
                
            else:
                values =[1,2,3,4]
                legends  = ['scalp','cort.','trab.','brain']
                colors = [(0.0, 0.0, 1.0, 1.0), 
                      (0.16129032258064513, 1.0, 0.8064516129032259, 1.0), 
                      (0.8064516129032256, 1.0, 0.16129032258064513, 1.0), 
                      (1.0, 0.40740740740740755, 0.0, 1.0)]
                
        if AirMask is not None:
            values.append(values[-1]+1)
            legends.append('Air')
            colors.append((223/255,199/255,224/255,1.0))
            		
            #we use manual color asignation 
                
        patches = [ mpatches.Patch(color=colors[i], label=legends[i] ) for i in range(len(values)) ]
        leg=axes[-1].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0. )
        self._figMasks.set_facecolor(self._BackgroundColorFigures)
        leg.get_frame().set_facecolor(self._BackgroundColorFigures)
        self.Widget.TransparencyScrollBar.setEnabled(True)
        if not self.Widget.HideMarkscheckBox.isEnabled():
            self.Widget.HideMarkscheckBox.setEnabled(True)

        # Persist this tab's artists/state so HideMarks and UpdateTransparency
        # can update every trajectory's plots, not just the active one.
        panel['figure']=self._figMasks
        panel['canvas']=self.static_canvas
        panel['axes']=axes
        panel['imMasks']=self._imMasks
        panel['imT1W']=self._imT1W
        panel['imCtMasks']=self._imCtMasks
        panel['markers']=self._markers
        panel['T1WData']=T1WData
        panel['LocFocalPoint']=LocFocalPoint

        # Honor the current marker-visibility setting on the freshly built plots
        # (transparency already applied above via the live scrollbar value).
        self._applyMarkerVisibility(panel)

        # Bring the just-computed trajectory's tab to the front.
        self._planningTabs.setCurrentIndex(self._TrajectoryNumber)

    def UpdateMask(self, bDeleteOnly=False):
        self.hideClockDialog()
        self.Widget.tabWidget.setEnabled(True)
        self.AcSim.setEnabled(True)
        try:
            Data=nibabel.load(self._outnameMask[self._TrajectoryNumber])
        except:
            raise ValueError("BabelViscoInput file does not exist. This is most likely due to a crash related to high PPW, please explore using lower PPW")
        self.FinalMaskRaw[self._TrajectoryNumber]=Data.get_fdata(dtype=np.float32)
        self._FinalMask[self._TrajectoryNumber] = np.flip(self.FinalMaskRaw[self._TrajectoryNumber],axis=2)

        self._bSegmentedBrain = np.max(self.FinalMaskRaw[self._TrajectoryNumber])>5

        T1W=nibabel.load(self._T1W_resampled_fname[self._TrajectoryNumber])
        self._T1WDataRaw[self._TrajectoryNumber]=T1W.get_fdata(dtype=np.float32)
        
        self._MaskNib[self._TrajectoryNumber]=Data
        self._T1WNib[self._TrajectoryNumber]=T1W

        if self.Config['bUseCT']:
            self._NiftiCT[self._TrajectoryNumber]=nibabel.load(self._prefix_path[self._TrajectoryNumber]+'CT.nii.gz')
            AllBoneHU = np.load(self._prefix_path[self._TrajectoryNumber]+'CT-cal.npz')['UniqueHU']
            CTData=AllBoneHU[self._NiftiCT[self._TrajectoryNumber].get_fdata().astype(int)].astype(np.float32)
            self._NiftiCT[self._TrajectoryNumber]=nibabel.Nifti1Image(CTData,affine=self._NiftiCT[self._TrajectoryNumber].affine,header=self._NiftiCT[self._TrajectoryNumber].header)               
            if self.Config['bExtractAirRegions'] and os.path.exists(self._prefix_path[self._TrajectoryNumber]+'AirRegions.nii.gz'):
                self._NiftiAirMask[self._TrajectoryNumber]=nibabel.load(self._prefix_path[self._TrajectoryNumber]+'AirRegions.nii.gz')
            
        self._BackgroundColorFigures=np.array(get_color_at(self.Widget.tabWidget,10,10))/255
        self._showMatplotlibVisualization(bDeleteOnly)
        if self._TrajectoryNumber == len(self.Config['ID'])-1:
            self.Widget.vtkVisualizationqPushButton.setEnabled(True)
        if hasattr(self,'_vtk_visualization'):
            self._UpdateVTKDomain()
        self.UpdateAcousticTab()
        self.AcSim._txTabs.setCurrentIndex(0) #we default always to first tab


    @Slot()
    def OpenVTKVisualization(self):
         # --- create / re-create the VTK viewer ---
        if not hasattr(self,'_vtk_visualization'):
            self._vtk_visualization = NiftiViewerWindow(trajectories=self.Config['ID'])
            self._vtk_visualization.resize(1580, 500)
            self._vtk_visualization.show()
            self._vtk_visualization.setWindowTitle("VTK NIfTI Viewer — Multi-Volume")
            self._vtk_visualization.closed.connect(self._closingVtkVisualization)
            self._UpdateVTKDomain(bFullyPopulate=True)
        else:
            self._vtk_visualization.raise_()
            self._vtk_visualization.activateWindow()

    def _UpdateVTKDomain(self,bFullyPopulate):

        if bFullyPopulate:
            lTraj=np.arange(len(self.Config['ID']))
        else:
            lTraj=[self._TrajectoryNumber]
        for n in lTraj:
            mask_nib=self._MaskNib[n]
            t1w_nib = self._T1WNib[n]

            # Focal-point voxel (label == 5 in the mask)
            mask_array = mask_nib.get_fdata(dtype=np.float32)
            focal_voxel = np.array(np.where(mask_array == 5)).flatten()
          
            viewer =self._vtk_visualization.viewer[n]
        
            viewer.load_base(mask_nib,
                                                focal_voxel,
                                                'Tissue Type',
                                                tissue_label=True)
            viewer.add_overlay(t1w_nib,'T1W',use_percentile=True,id='T1W')
            viewer._on_cmap_changed(0,"TissueLabel")
            viewer._layer_panel._rows[1]._opacity_slider.setValue(50)
            if self._NiftiCT[n]:
                viewer.add_overlay(self._NiftiCT[n],'CT',id='CT')
                viewer._layer_panel._rows[-1]._opacity_slider.setValue(50)
                viewer._layer_panel._rows[-1]._cutoff_edit.setText('1')
                viewer._layer_panel._rows[-1]._on_cutoff_changed()
            if self._NiftiAirMask[n]:
                viewer.add_overlay(self._NiftiAirMask[n],'Air',id='Air')
                viewer._layer_panel._rows[-1]._opacity_slider.setValue(50)
                viewer._layer_panel._rows[-1]._cutoff_edit.setText('1')
                viewer._layer_panel._rows[-1]._on_cutoff_changed()
                viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(4)

            if hasattr(self,'_NiftiSkull'):
                self._UpdateVTKAcResults(n)
            if hasattr(self,'_NiftiTemperature'):
                self._UpdateVTKThermal()
        if hasattr(self,'_NiftiMergeAc'):
            self._UpdateVTKMergedAcResults()   

    def UpdateNiftiAcResults(self,NiftiSkull,NiftiWater,NTraj):
        self._NiftiSkull[NTraj]=NiftiSkull
        self._NiftiWater[NTraj]=NiftiWater
        self._UpdateVTKAcResults(NTraj)

    def UpdateNiftiTemperatureResults(self,NiftiIntensity,NiftiTemperature):
        self._NiftiIntensity=NiftiIntensity
        self._NiftiTemperature=NiftiTemperature
        self._UpdateVTKThermal()
    
    def _UpdateVTKAcResults(self,NTraj):
        if not hasattr(self,'_vtk_visualization'):
            return
        if self._NiftiSkull[NTraj] is None:
            return
        # We remove previous entries if already available
        viewer =self._vtk_visualization.viewer[NTraj]
        for id in ['Skull','Water']:
            for n,row in enumerate(viewer._layer_panel._rows):
                if row._id == id:
                    viewer._on_remove_requested(n)
                    break
        viewer.add_overlay(self._NiftiSkull[NTraj],'Skull',id='Skull')
        viewer._layer_panel._rows[-1]._opacity_slider.setValue(100)
        viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(5)
        viewer._layer_panel._rows[-1]._cutoff_edit.setText('0.25')
        viewer._layer_panel._rows[-1]._on_cutoff_changed()
        viewer.add_overlay(self._NiftiWater[NTraj],'Water',id='Water')
        viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(5)
        viewer._layer_panel._rows[-1]._cutoff_edit.setText('0.25')
        viewer._layer_panel._rows[-1]._on_cutoff_changed()
        viewer._layer_panel._rows[-1]._eye_btn.setChecked(False)
        #we select skull for default windowing
        viewer._layer_panel._rows[-2]._wl_btn.click()

    def _UpdateVTKThermal(self):
        if not hasattr(self,'_vtk_visualization'):
            return
        # We remove previous entries if already available
        for id in ['Intensity','Temperature']:
            for n,row in enumerate(self._vtk_visualization.viewer._layer_panel._rows):
                if row._id == id:
                    self._vtk_visualization.viewer._on_remove_requested(n)
                    break
        # We hide  pressure fields
        for id in ['Water','Skull']:
            for n,row in enumerate(self._vtk_visualization.viewer._layer_panel._rows):
                if row._id == id:
                    row._eye_btn.setChecked(False)
                    break
        self._vtk_visualization.viewer.add_overlay(self._NiftiIntensity,'Intensity',id='Intensity')
        self._vtk_visualization.viewer._layer_panel._rows[-1]._opacity_slider.setValue(100)
        self._vtk_visualization.viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(5)
        self._vtk_visualization.viewer._layer_panel._rows[-1]._cutoff_edit.setText('0.1')
        self._vtk_visualization.viewer._layer_panel._rows[-1]._on_cutoff_changed()
        self._vtk_visualization.viewer.add_overlay(self._NiftiTemperature,'Temperature',id='Temperature')
        self._vtk_visualization.viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(5)
        self._vtk_visualization.viewer._layer_panel._rows[-1]._cutoff_edit.setText('37.05')
        self._vtk_visualization.viewer._layer_panel._rows[-1]._on_cutoff_changed()
        self._vtk_visualization.viewer._layer_panel._rows[-1]._eye_btn.toggle()

        self._vtk_visualization.viewer._layer_panel._rows[-2]._wl_btn.click()

    def UpdateNiftiMergedAcResults(self,NiftiMergeAc):
        self._NiftiMergeAc=NiftiMergeAc
        self._UpdateVTKMergedAcResults()

    def _UpdateVTKMergedAcResults(self):
        if not hasattr(self,'_vtk_visualization'):
            return
        # We remove previous entries if already available
        for viewer in self._vtk_visualization.viewer:
            for id in ['MergedAc']:
                for n,row in enumerate(viewer._layer_panel._rows):
                    if row._id == id:
                        viewer._on_remove_requested(n)
                        break
            viewer.add_overlay(self._NiftiMergeAc,'MergedAc',id='MergedAc')
            viewer._layer_panel._rows[-1]._opacity_slider.setValue(100)
            viewer._layer_panel._rows[-1]._cmap_combo.setCurrentIndex(5)
            viewer._layer_panel._rows[-1]._cutoff_edit.setText('0.25')
            viewer._layer_panel._rows[-1]._on_cutoff_changed()

    @Slot()
    def _closingVtkVisualization(self):
        delattr(self,'_vtk_visualization')

    def _applyMarkerVisibility(self,panel):
        mc=[0.75, 0.75, 0.0,1.0]
        if self.Widget.HideMarkscheckBox.isChecked():
            mc[3] = 0.0
        for m in panel['markers']:
            m.set_markerfacecolor(mc)
            m.set_markeredgecolor(mc)

    @Slot()
    def HideMarks(self,v):
        # Apply to every trajectory tab so the toggle stays consistent.
        for panel in getattr(self,'_trajPanels',[]):
            if panel.get('figure') is None:
                continue
            self._applyMarkerVisibility(panel)
            panel['figure'].canvas.draw_idle()

    def _applyTransparency(self,panel):
        alpha=self.Widget.TransparencyScrollBar.value()/100.0
        sm=plt.cm.ScalarMappable(cmap='gray')
        T1WData=panel['T1WData']
        loc=panel['LocFocalPoint']
        T1WXZ=sm.to_rgba(T1WData[:,loc[1],:].T,alpha=alpha)
        T1WYZ=sm.to_rgba(T1WData[loc[0],:,:].T,alpha=alpha)
        T1WXY=sm.to_rgba(T1WData[:,:,loc[2]].T,alpha=alpha)
        for im,T1WMap in zip(panel['imT1W'],
                                    [T1WXZ,T1WYZ,T1WXY]):
            im.set_data(T1WMap)

    @Slot()
    def UpdateTransparency(self):
        # Apply to every trajectory tab so the slider affects all plots.
        for panel in getattr(self,'_trajPanels',[]):
            if panel.get('figure') is None:
                continue
            self._applyTransparency(panel)
            panel['figure'].canvas.draw_idle()
            
          
    def GetExport(self):
        ExtraConfig ={}
        ExtraConfig['Frequency']=self._Frequency
        ExtraConfig['PPW']=self._BasePPW
        if self.Config['bUseCT']:
            if self.Config['CTType'] in [1,2,3]:
                ExtraConfig['HUThreshold']=self.Widget.HUTreshold.value()
            else:
                ExtraConfig['DensityThreshold']=self.Widget.HUTreshold.value()
            if self.Config['CTType'] in [2,3]: #ZTE or PETRA
                ExtraConfig['ZTERange']=self.Widget.ZTERangeSlider.value()
        with open(self._trackingtimefile,'r') as f:
            self._TrackingTime=yaml.load(f,yaml.SafeLoader)
        return self.Config | ExtraConfig | self._TrackingTime
    
    ##
    def SetSuccesCode(self):
        self.RETURN_CODE = ReturnCodes['SUCCES']

    def SetErrorDomainCode(self):
        self.RETURN_CODE = ReturnCodes['ERROR_DOMAIN']

    def SetErrorAcousticsCode(self):
        self.RETURN_CODE = ReturnCodes['ERROR_ACOUSTICS']
        
    def UpdateComputationalTime(self,step,steptime):
        if os.path.isfile(self._trackingtimefile):
            with open(self._trackingtimefile,'r') as f:
                self._TrackingTime=yaml.load(f,yaml.SafeLoader)
        if step == 'domain':
            self._TrackingTime['Calculation time domain']=steptime
        elif step == 'ultrasound':
            self._TrackingTime['Calculation time ultrasound']=steptime
        elif step == 'thermal':
            self._TrackingTime['Calculation time thermal']=steptime
        else:
            raise ValueError('type of step to track time not valid -'+step)
        with open(self._trackingtimefile,'w') as f:
            yaml.dump(self._TrackingTime,f,yaml.SafeDumper)

    def CommomAcOptions(self):
        kargs={}
        kargs['bUseCT']=self.Config['bUseCT']
        kargs['CTMapCombo']=self.Config['CTMapCombo']
        kargs['bUseRayleighForWater']=self.Config['bUseRayleighForWater']
        kargs['bSaveStress']=self.Config['bSaveStress']
        kargs['bSaveDisplacement']=self.Config['bSaveDisplacement']
        kargs['bForceHomogenousMedium']=self.Config['bForceHomogenousMedium']
        kargs['HomogenousMediumValues']=self.Config['HomogenousMediumValues']
        kargs['bExtractAirRegions']=self.Config['bExtractAirRegions']   
        kargs['bPETRA'] = False
        if kargs['bUseCT']:
            if self.Config['CTType']==3:
                kargs['bPETRA']=True
            elif self.Config['CTType']==4:
                kargs['bDensity']=True
        kargs['OptimizedWeightsFile']=self.Config['TxOptimizedWeights'][self.Config['TxSystem']]
        return kargs
    
    def LogTelemetry(self,entry):
        msgLevel = int(entry.split(':')[1][1])
        if self.Config['TelemetryLevel'] >= msgLevel:
            self._TelmetryMsgs.append({'time':time.time()-self._TimeStart,'event':entry})

    def SendTelemetry(self,waittocomplete=False):
        #we will break in segments of 
        send_telemetry('BabelBrain log',
                       idpath=_LocationInstallID,
                       session_date=_date_session,
                       APP_VERSION=self.Config['version'].rstrip(),
                       data=self._TelmetryMsgs,
                       waittocomplete=waittocomplete)
        self._TelmetryMsgs=[] #we clean the list

    def AllAcFieldsDone(self):
        return all(x is not None for x in self._NiftiSkull)
        

def get_color_at(widget, x,y):
    pixmap = QPixmap(widget.size())
    widget.render(pixmap)
    return pixmap.toImage().pixelColor(x,y).getRgb()


class RunMaskGeneration(QObject):
    '''
    Worker class for running mask generation in a separate process.
    '''
    finished = Signal(object)
    endError = Signal()

    logTelemetry = Signal(str)

    def __init__(self,mainApp):
        super(RunMaskGeneration, self).__init__()
        self._mainApp=mainApp

    def run(self):
        '''
        Execute the mask generation process in a separate process.

        Emits
        -----
        finished : Signal
            Emitted when mask generation completes successfully.
        endError : Signal
            Emitted if an error occurs during mask generation.
        '''

        print("*"*40)
        print("*"*5+" Calculating mask.. BE PATIENT... it can take a couple of minutes...")
        print("*"*40)

        deviceName=self._mainApp.Config['ComputingDevice']
        COMPUTING_BACKEND=self._mainApp.Config['ComputingBackend']

        Widget=self._mainApp.Widget

        T1WIso= self._mainApp.Config['T1WIso']
        T1W= self._mainApp.Config['T1W']

        Frequency=  self._mainApp._Frequency
        SmallestSoS= GetSmallestSOS(Frequency,bShear=True)

        BasePPW=self._mainApp._BasePPW
        SpatialStep=np.round(SmallestSoS/Frequency/BasePPW*1e3,3) #step of mask to reconstruct , mm
        self.logTelemetry.emit(f"CTS:L3:S1: Frequency={Frequency} PPW={BasePPW}")
        print("Frequency, SmallestSoS, BasePPW,SpatialStep",Frequency, SmallestSoS, BasePPW,SpatialStep)

        prefix=self._mainApp._prefix[self._mainApp._TrajectoryNumber]
        TrajectoryNumber=self._mainApp._TrajectoryNumber
        print("Config['Mat4Trajectory']",self._mainApp.Config['Mat4Trajectory'])

        #first we ensure we have isotropic scans at 1 mm required to get affine matrix at 1.0 mm isotropic
        save_T1W_iso(T1W,T1WIso)

        kargs={}
        kargs['SimbNIBSDir']=self._mainApp.Config['simbnibs_path']
        kargs['SimbNIBSType']=self._mainApp.Config['SimbNIBSType']
        kargs['CoregCT_MRI']=self._mainApp.Config['CoregCT_MRI']
        kargs['TrajectoryType']=self._mainApp.Config['TrajectoryType']
        kargs['Mat4Trajectory']=self._mainApp.Config['Mat4Trajectory'] #Path to trajectory file
        kargs['T1Source_nii']=T1W
        kargs['T1Conformal_nii']=T1WIso
        kargs['SpatialStep']=SpatialStep
        kargs['Location']=[0,0,0] #This coordinate will be ignored
        kargs['prefix']=prefix
        kargs['TrajectoryNumber']=TrajectoryNumber
        kargs['bPlot']=False
        if self._mainApp.Config['bUseCT']:
            kargs['CT_or_ZTE_input']=self._mainApp.Config['CT_or_ZTE_input']
            kargs['CTType']=self._mainApp.Config['CTType']
            if kargs['CTType'] in [2,3]:
                kargs['ZTERange']=Widget.ZTERangeSlider.value()
            if kargs['CTType'] in [1,2,3]:
                kargs['HUThreshold']=Widget.HUTreshold.value()
            else:
                assert(kargs['CTType']==4)
                kargs['DensityThreshold']=Widget.HUTreshold.value()
                kargs['RegionAirCT']=[0.01, 10] # air density values
            
        def ValidParam(k):
            #here we screen out parameters that are irrelevant for Step 1
            if '_Correction' not in k and k not in ['BaselineTemperature','bSaveStress',
                                                    'bSaveDisplacement','LimitBHTEIterationsPerProcess',
                                                    'bForceHomogenousMedium','HomogenousMediumValues',
                                                    'bForceNoAbsorptionSkullScalp',
                                                    'TxOptimizedWeights',
                                                    'PlanTUSRoot',
                                                    'ConnectomeRoot',
                                                    'TelemetryLevel',
                                                    'NumberTransducers']:
                return True
            else:
                return False
        #advanced parameters
        for k in self._mainApp._DefaultOptions.keys():
            if ValidParam(k):
                kargs[k]=self._mainApp.Config[k] 
        
        bForceFullRecalculation = False
        if os.path.isfile(self._mainApp.Config['AdvancedParamsFile']):
            print('Advance params file',self._mainApp.Config['AdvancedParamsFile'])
            with open(self._mainApp.Config['AdvancedParamsFile'],'r') as f:
                PrevParams=yaml.load(f,yaml.SafeLoader)
            bForceFullRecalculation=False
            for k in self._mainApp._DefaultOptions.keys():
                if ValidParam(k): 
                    if k not in PrevParams: #if a new parameter was added in a new release, we force recalculations
                        bForceFullRecalculation=True
                        print('PrevParamsFile - Parameter',k,'is not present in prevparams')
                        break
            if not bForceFullRecalculation:
                for k in PrevParams:
                    if ValidParam(k) and k in kargs: 
                       if kargs[k] != PrevParams[k]: #if a parameter changed, we force recalculations
                                print('PrevParamsFile - Parameter',k,'is differemt',kargs[k],PrevParams[k])
                                bForceFullRecalculation=True
                                break
        else:
            #in case no file of params have been saved, we compare with defaults, which is compatible with previous releases of BabelBrain
            for k in self._mainApp._DefaultOptions.keys():
                if ValidParam(k):
                    if kargs[k] != getattr(self._mainApp._DefaultOptions,k): #if a parameter is different from default, we force recalculations
                        print('Defaults - Parameter',k,'is differemt',kargs[k],getattr(self._mainApp._DefaultOptions,k))
                        bForceFullRecalculation=True
                        break
        kargs['bForceFullRecalculation']=bForceFullRecalculation
            
        # now we save the parameters for future comparison
        NewParams={}
        for k in self._mainApp._DefaultOptions.keys():
            if ValidParam(k):
                NewParams[k]=kargs[k]
            
        with open(self._mainApp.Config['AdvancedParamsFile'],'w') as f:
            yaml.safe_dump(NewParams,f)

        
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
                if type(cMsg) is str:
                    print(cMsg,end='')
                    if 'CTS:' in cMsg:
                        self.logTelemetry.emit(cMsg)
                    if '--Babel-Brain-Low-Error' in cMsg:
                        bNoError=False
                        self.logTelemetry.emit("CTS:L2:S1: "+cMsg)
                elif type(cMsg) is dict:
                    output_files=cMsg
                else:
                    print('WARNING: Unknown type of message from thread:', type(cMsg))

        maskWorkerProcess.join()
        while queue.empty() == False:
            cMsg=queue.get()
            if type(cMsg) is str:
                print(cMsg,end='')
                if 'CTS:' in cMsg:
                    self.logTelemetry.emit(cMsg)
                if '--Babel-Brain-Low-Error' in cMsg:
                    bNoError=False
                    self.logTelemetry.emit("CTS:L2:S1: "+cMsg)
            elif type(cMsg) is dict:
                output_files=cMsg
            else:
                print('WARNING: Unknown type of message from thread:', type(cMsg))
        if bNoError:
            TEnd=time.time()
            TotalTime = TEnd-T0
            print('Total time',TotalTime)
            print("*"*40)
            print("*"*5+" DONE calculating mask.")
            print("*"*40)
            self.logTelemetry.emit("CTS:L2:S1: TOTAL TIME " + str(TotalTime))
            self._mainApp.UpdateComputationalTime('domain',TotalTime)
            self.finished.emit(output_files)
        else:
            print("*"*40)
            print("*"*5+" Error in execution.")
            print("*"*40)
            self.endError.emit()
def _apply_color_scheme(app):
    """Apply an application-level stylesheet that keeps text readable in dark mode.

    Qt enters "stylesheet mode" for any widget that has even a partial stylesheet
    (including empty ones from Qt Designer .ui files).  In that mode, unspecified
    properties no longer inherit from the system palette, so text can stay dark on
    dark backgrounds.  Setting palette(window-text) at the *application* level has
    the lowest CSS specificity, so it provides a safe default while intentionally
    colored widgets (blue accent labels, green/red status buttons) override it.
    """
    bg = app.palette().color(QPalette.Window)
    is_dark = bg.lightness() < 128
    if is_dark:
        app.setStyleSheet(
            "QWidget { color: palette(window-text); }"
            " QLabel { background: transparent; }"
        )
    else:
        app.setStyleSheet("")


def main():
    '''
    Main entry point for the BabelBrain application.
    Handles argument parsing, GUI initialization, and application execution.
    '''
    global bINUSE_INSIDE_BRAINSIGHT
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
    _apply_color_scheme(app)
    # Re-apply if the user switches dark/light mode while the app is running.
    # colorSchemeChanged is available from Qt 6.5; guard for older installs.
    try:
        QGuiApplication.styleHints().colorSchemeChanged.connect(
            lambda: _apply_color_scheme(app)
        )
    except AttributeError:
        pass

    selwidget = SelFiles()

    prevConfig=GetLatestSelection()

    # First-launch telemetry consent. Shown before any input dialog so the
    # user's choice is recorded once and reused for every subsequent run.
    if prevConfig is None or 'TelemetryLevel' not in prevConfig:
        from Telemetry.TelemetryConsentDialog import TelemetryConsentDialog
        consent = TelemetryConsentDialog()
        consent.exec()
        SaveTelemetryLevelToConfig(consent.selected_level())
        if prevConfig is None:
            prevConfig = {}
        prevConfig['TelemetryLevel'] = consent.selected_level()

    bFullPrevConfig=False

    if prevConfig is not None:
        bFullPrevConfig = 'simbnibs_path' in prevConfig

    if prevConfig is not None and bFullPrevConfig:
        selwidget.ui.SimbNIBSlineEdit.setText(prevConfig['simbnibs_path'])
        selwidget.ui.T1WlineEdit.setText(prevConfig['T1W'])
        selwidget.ui.TrajectorylineEdit.setText(prevConfig['Mat4Trajectory'])
        selwidget.ui.ThermalProfilelineEdit.setText(prevConfig['ThermalProfile'])
        if 'CT_or_ZTE_input' in prevConfig:
            selwidget.ui.CTlineEdit.setText(prevConfig['CT_or_ZTE_input'])
            selwidget.ui.CTTypecomboBox.setCurrentIndex(prevConfig['CTType'])
        if 'CTMapCombo' in prevConfig:
            selwidget.ui.CTMappingcomboBox.setCurrentIndex(selwidget._dfCTParams.index.get_loc(tuple(prevConfig['CTMapCombo'])))
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
                Backend=''
                if prevConfig['ComputingBackend']==1:
                    Backend='CUDA'
                elif prevConfig['ComputingBackend']==2:
                    Backend='OpenCL'
                elif prevConfig['ComputingBackend']==3:
                    Backend='Metal'
                elif prevConfig['ComputingBackend']==4:
                    Backend='MLX'
                if len(Backend)>0:
                    selwidget.SelectComputingEngine(GPU=GPU,Backend=Backend)

        if 'TxSystem' in prevConfig:
            selwidget.SelectTxSystem(prevConfig['TxSystem'])
        if 'MultiPoint' in prevConfig:
            if prevConfig['EnableMultiPoint']:
                selwidget.ui.MultiPointTypecomboBox.setCurrentIndex(1)
            if len(prevConfig['MultiPoint'].strip())>0:
                selwidget.ui.MultiPointlineEdit.setText(prevConfig['MultiPoint'])
                
    AltOutputFilesPath=None
    if args.bInUseWithBrainsight:
        bINUSE_INSIDE_BRAINSIGHT = True
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
