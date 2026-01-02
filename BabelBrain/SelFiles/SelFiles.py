# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QDialog,QFileDialog,QMessageBox,QStyle
from PySide6.QtCore import Slot, Qt,QAbstractTableModel

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_Dialog
import platform
import os
from pathlib import Path
import re
import yaml
import sys

sys.path.append(os.path.abspath('../'))
                
sys.path.append(os.path.abspath('../../'))

from TranscranialModeling.BabelIntegrationBASE import SpeedofSoundWebbDataset
    

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

ListTxSteering=['H317','I12378','ATAC','R15148','R15646','IGT64_500','H301','DomeTx']

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return   Qt.AlignmentFlag.AlignCenter

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
            
ORIGINAL_BABELBRAIN_SELECTION={'real CT':19,'ZTE':19,'PETRA':7}

def ValidThermalProfile(fProf):
    msgDetails=None
    try:
        with open(fProf,'r') as f:
            profile=yaml.safe_load(f)
    except:
        msgDetails = "Invalid profile YAML file"
        return False,msgDetails
        
    if 'BaseIsppa' not in profile:
        msgDetails = "BaseIsppa entry must be in YAML file"
        return False,msgDetails
    
    if type(profile['BaseIsppa']) is not float:
        msgDetails = "BaseIsppa must be a single float"
        return False,msgDetails
    
    if 'AllDC_PRF_Duration' not in profile:
        msgDetails = "AllDC_PRF_Duration entry must be in YAML file"
        return False,msgDetails
    
    if type(profile['AllDC_PRF_Duration']) is not list:
        msgDetails = "AllDC_PRF_Duration must be a list"
        return False,msgDetails
    
    for n,entry in enumerate(profile['AllDC_PRF_Duration']):
        if type(entry) is not dict:
            msgDetails = "entry %i in AllDC_PRF_Duration must be a dictionary" % (n)
            return False,msgDetails
        for k in ['DC','PRF','Duration','DurationOff']:
            if k not in entry:
                msgDetails = "entry %i in AllDC_PRF_Duration must have a key %s" % (n,k)
                return False,msgDetails
            if type(entry[k]) is not float:
                msgDetails = "key %s in entry %i of AllDC_PRF_Duration must be float" % (k,n)
                return False,msgDetails
        if 'Repetitions' in entry:
            if type(entry['Repetitions']) is not int:
                msgDetails = "key Repetitions in entry %i of AllDC_PRF_Duration must be integer" % (n)
                return False,msgDetails
            if entry['Repetitions'] <1:
                msgDetails = "key Repetitions in entry %i of AllDC_PRF_Duration must be larger or equal than 1" % (n)
                return False,msgDetails
        if 'NumberGroupedSonications' in entry:
            if type(entry['NumberGroupedSonications']) is not int:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be integer" % (n)
                return False,msgDetails
            if entry['NumberGroupedSonications'] <1:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be larger than 1" % (n)
                return False,msgDetails
            if 'PauseBetweenGroupedSonications' not in entry:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be present if NumberGroupedSonications is specified" % (n)
                return False,msgDetails
        if 'PauseBetweenGroupedSonications' in entry:
            if type(entry['PauseBetweenGroupedSonications']) is not float:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be float" % (n)
                return False,msgDetails
            if entry['PauseBetweenGroupedSonications'] <0.0:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be larger than 0.0" % (n)
                return False,msgDetails
            if 'NumberGroupedSonications' not in entry:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be present if PauseBetweenGroupedSonications is specified" % (n)
                return False,msgDetails
        for k in entry:
            if k not in ['DC','PRF','Duration','DurationOff','Repetitions','NumberGroupedSonications','PauseBetweenGroupedSonications']:
                msgDetails = "key %s in entry %i of AllDC_PRF_Duration is unknown. It must be either 'DC', 'PRF', 'Duration',  'DurationOff', 'Repetitions', 'NumberGroupedSonications' or 'PauseBetweenGroupedSonications'" % (k,n)
                return False,msgDetails
    return True,msgDetails

class SelFiles(QDialog):
    def __init__(self, parent=None,Trajectory='',T1W='',
                    SimbNIBS='',CTType=0,CoregCT=1,CT='',
                    SimbNIBSType=0,TrajectoryType=0,
                    GPU='CPU',
                    Backend='Metal',
                    defaultCTMap=ORIGINAL_BABELBRAIN_SELECTION['real CT']):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        with open(os.path.join(resource_path(),'version-gui.txt'), 'r') as f:
            version=f.readlines()[0]
        self.setWindowTitle("BabelBrain V"+version + " - Select input files ...")
        self.ui.SelTrajectorypushButton.clicked.connect(self.SelectTrajectory)
        self.ui.SelT1WpushButton.clicked.connect(self.SelectT1W)
        self.ui.SelCTpushButton.clicked.connect(self.SelectCT)
        self.ui.SelSimbNIBSpushButton.clicked.connect(self.SelectSimbNIBS)
        self.ui.SelTProfilepushButton.clicked.connect(self.SelectThermalProfile)
        self.ui.ContinuepushButton.clicked.connect(self.Continue)
        self.ui.CTTypecomboBox.currentIndexChanged.connect(self.SelectCTType)
        self.ui.MultiPointTypecomboBox.currentIndexChanged.connect(self.SelectMultiPoint)
        self.ui.TransducerTypecomboBox.currentIndexChanged.connect(self.SelectTransducer)
        self.ui.SelMultiPointProfilepushButton.clicked.connect(self.SelectMultiPointProfile)
        self.ui.CancelpushButton.clicked.connect(self.Cancel)
                
        self.ui.SelTrajectorypushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelT1WpushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelCTpushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelTProfilepushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelSimbNIBSpushButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
                                        

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
        self.ui.ResetCTMapOriginalpushButton.clicked.connect(self.ResetOriginalCTCombo)


        self._GPUs=self.GetAvailableGPUs()

        if len(self._GPUs)==0: #only CPU
            msgBox = QMessageBox()
            msgBox.setText("No GPUs were detected!\nBabelBrain can't run without a GPU\nfor simulations")
            msgBox.exec()
            sys.exit(0)

        for dev in self._GPUs:
            self.ui.ComputingEnginecomboBox.addItem(dev[0] + ' -- ' + dev[1])
        for sel,dev in enumerate(self._GPUs):
            if GPU in dev[0] and (GPU=='CPU' or Backend in dev[1]):
                self.ui.ComputingEnginecomboBox.setCurrentIndex(sel)
                break

        df = SpeedofSoundWebbDataset()
        for index, row in df.iterrows():
            self.ui.CTMappingcomboBox.addItem(', '.join(index))
        self._dfCTParams=df
        self.ui.CTMappingcomboBox.setCurrentIndex(defaultCTMap)

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

    def GetAllTransducers(self):
        """
        Returns a list of all transducers available in BabelBrain
        """
        return [self.ui.TransducerTypecomboBox.itemText(i) for i in range(self.ui.TransducerTypecomboBox.count())]

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
                # AllDevices.append([dev,'MLX']) #we disable this for the time being until MLX fixes their support to large arrays
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
            
    def ValidThermalProfile(self):
        fProf = self.ui.ThermalProfilelineEdit.text()
        retValue, self.msgDetails = ValidThermalProfile(fProf)
        return retValue
    
    def ValidateMultiPointProfile(self):
        selTx=self.ui.TransducerTypecomboBox.currentText()
        if  selTx not in ListTxSteering:
            return True
        if self.ui.MultiPointTypecomboBox.currentIndex() ==0:
            return True
        
        fProf = self.ui.MultiPointlineEdit.text()

        if not os.path.isfile(fProf):
            self.msgDetails = "Profile file was not specified"
            return False

        try:
            with open(fProf,'r') as f:
                profile=yaml.safe_load(f)
        except:
            self.msgDetails = "Invalid profile YAML file"
            return False
        if 'MultiPoint' not in profile:
            self.msgDetails = "YAML file missing 'MultiPoint' entry"
            return False
        selTx=self.ui.TransducerTypecomboBox.currentText()
        if selTx not in ListTxSteering:
            self.msgDetails = "MultiPoint in profile can only be specified with a phased array-type transducer"
            return False
        if type(profile['MultiPoint']) is not list:
            self.msgDetails = "MultiPoint must be a list" 
            return False
        for n,entry in enumerate(profile['MultiPoint']):
            if type(entry) is not dict:
                self.msgDetails = "entry %i in MultiPoint must be a dictionary" % (n)
                return False
            for k in ['X','Y','Z']:
                if k not in entry:
                    self.msgDetails = "entry %i in MultiPoint must have a key %s" % (n,k)
                    return False
                if type(entry[k]) is not float:
                    self.msgDetails = "key %s in entry %i of MultiPoint must be float" % (k,n)
                    return False
        return True
            # we convert to mm
    
    @Slot()
    def SelectTrajectory(self):
        curfile=self.ui.TrajectorylineEdit.text()
        bdir=os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fTraj=QFileDialog.getOpenFileName(self,
            "Select trajectory", bdir, "Trajectory (*.txt)")[0]
        if len(fTraj)>0:
            self.ui.TrajectorylineEdit.setText(fTraj)
            self.ui.TrajectorylineEdit.setCursorPosition(len(fTraj))

    @Slot()
    def SelectT1W(self):
        curfile=self.ui.T1WlineEdit.text()
        bdir=os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fT1W=QFileDialog.getOpenFileName(self,
            "Select T1W",bdir, "Nifti (*.nii *.nii.gz)")[0]
        if len(fT1W)>0:
            self.ui.T1WlineEdit.setText(fT1W)
            self.ui.T1WlineEdit.setCursorPosition(len(fT1W))

    @Slot()
    def SelectCT(self):
        curfile=self.ui.CTlineEdit.text()
        bdir=os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fCT=QFileDialog.getOpenFileName(self,
            "Select CT", bdir, "Nifti (*.nii *.nii.gz)")[0]
        if len(fCT)>0:
            self.ui.CTlineEdit.setText(fCT)
            self.ui.CTlineEdit.setCursorPosition(len(fCT))

    @Slot()
    def SelectThermalProfile(self):
        curfile=self.ui.ThermalProfilelineEdit.text()
        bdir=os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fThermalProfile=QFileDialog.getOpenFileName(self,"Select thermal profile",bdir,"yaml (*.yaml)")[0]
        if len(fThermalProfile)>0:
            print('fThermalProfile',fThermalProfile)
            self.ui.ThermalProfilelineEdit.setText(fThermalProfile)

    @Slot()
    def SelectMultiPointProfile(self):
        curfile=self.ui.MultiPointlineEdit.text()
        bdir=os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fMultiPointProfile=QFileDialog.getOpenFileName(self,"Select multi point profile",bdir,"yaml (*.yaml)")[0]
        if len(fMultiPointProfile)>0:
            print('fMultiPointProfile',fMultiPointProfile)
            self.ui.MultiPointlineEdit.setText(fMultiPointProfile)

    @Slot()
    def SelectSimbNIBS(self):
        bdir=self.ui.SimbNIBSlineEdit.text()
        if not os.path.isdir(bdir):
            bdir=os.getcwd()
        fSimbNIBS=QFileDialog.getExistingDirectory(self,"Select SimbNIBS directory",
                    bdir)
        if len(fSimbNIBS)>0:
            self.ui.SimbNIBSlineEdit.setText(fSimbNIBS)
            self.ui.SimbNIBSlineEdit.setCursorPosition(len(fSimbNIBS))

    @Slot()
    def SelectCTType(self,value):
        bv = value >0
        self.ui.CTlineEdit.setEnabled(bv)
        self.ui.SelCTpushButton.setEnabled(bv)
        self.ui.CoregCTlabel.setEnabled(bv)
        self.ui.CoregCTlabel_2.setEnabled(bv)
        self.ui.CoregCTlabel_3.setEnabled(bv)
        self.ui.CoregCTcomboBox.setEnabled(bv)
        self.ui.CTMappingcomboBox.setEnabled(bv)
        self.ui.ResetCTMapOriginalpushButton.setEnabled(bv)
        self.ResetOriginalCTCombo()
    
    @Slot()
    def SelectMultiPoint(self,value):
        bv = value >0
        self.ui.MultiPointlineEdit.setEnabled(bv)
        self.ui.SelMultiPointProfilepushButton.setEnabled(bv)

    @Slot()
    def SelectTransducer(self,value):
        selTx=self.ui.TransducerTypecomboBox.currentText()
        bv = selTx in ListTxSteering
        if not bv:
            self.ui.MultiPointTypecomboBox.setCurrentIndex(0)
        self.ui.MultiPointTypecomboBox.setEnabled(bv)

    @Slot()
    def ResetOriginalCTCombo(self):
        if self.ui.CTTypecomboBox.currentText() != 'NO':
            if self.ui.CTTypecomboBox.currentText() in ORIGINAL_BABELBRAIN_SELECTION:
                self.ui.CTMappingcomboBox.setCurrentIndex(ORIGINAL_BABELBRAIN_SELECTION[ self.ui.CTTypecomboBox.currentText()])
        
        
    @Slot()
    def Continue(self):
        self.msgDetails = ""
        if not self.ValidTrajectory() or\
           not self.ValidSimNIBS() or\
           not self.ValidThermalProfile() or\
           not self.ValidateMultiPointProfile() or\
           not os.path.isfile(self.ui.T1WlineEdit.text()) or\
           (self.ui.CTTypecomboBox.currentIndex()>0 and not os.path.isfile(self.ui.CTlineEdit.text())):
            msgBox = QMessageBox()
            msgBox.setText("Please indicate valid entries")
            print(self.msgDetails)
            msgBox.setDetailedText(self.msgDetails)
            msgBox.exec()
        else:
            self.accept()

    @Slot()
    def Cancel(self):
        self.done(-1)
         
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    widget = SelFiles()
    widget.show()
    sys.exit(app.exec())
