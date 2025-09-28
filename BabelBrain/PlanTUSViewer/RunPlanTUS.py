import sys

from PySide6.QtWidgets import QDialog,QFileDialog,QStyle,QMessageBox,QVBoxLayout
from PySide6.QtCore import QTimer,QObject

import platform
import os
from pathlib import Path

from multiprocessing import Process,Queue
import time

import yaml
import glob
import subprocess
import traceback
import shutil
import nibabel
import numpy as np

from scipy.io import loadmat

from ClockDialog import ClockDialog
from CreateVoxelMask import create_target_mask
from ConvMatTransform import (
    ReadTrajectoryBrainsight,
    itk_to_BSight,
    read_itk_affine_transform,
    templateBSight,
    BSight_to_itk,
    templateSlicer
)

from PlanTUSViewer.PlanTUSViewer import MultiGiftiViewerWidget,FinalResultViewer

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


def AcousticAxisONeil(frequency,aperture,focal_length,c=1500.0,step=0.05):
    #Old O'Neil, still unbeatable analytical formula for spherical shells
    k = np.pi*2*frequency/c #wavenumber
    l = c/frequency
    a = aperture/2
    A=focal_length
    h=A-np.sqrt(A**2-a**2)
    z=np.arange(0,2*focal_length,l*step) #twice the focal spot should be enough
    B=np.sqrt((z-h)**2+a**2)
    M=(B+z)/2
    delta=B-z
    E=2/(1-z/A)
    P=E*np.sin(k*delta/2)
    return h,z,np.abs(P)
    
def FindTPOEquivalent(frequency,aperture,focal_length):
    #first we calculate the acoustic axis with an analytical formula
    h,zo,po=AcousticAxisONeil(frequency,aperture,focal_length)
    peaks, _ = find_peaks(po, height=None, threshold=None, distance=None, prominence=None)
    #we find the peak closer to the natural focus
    closer_peak=np.argmin(np.abs(zo[peaks]-focal_length))
    end_half=po[peaks[closer_peak]:]>=0.5*po[peaks[closer_peak]]
    end_half=np.where(np.diff(end_half))[0]
    if len(end_half)>0:
        end_half=end_half[0]+1
        # print('e',po[peaks[closer_peak]:][end_half]/po[peaks[closer_peak]])
        end_half=zo[peaks[closer_peak]:][end_half]
    else: #we use the last point, 
        print('Warning, unable to find end of half peak pressure length')
        end_half=zo[peaks[closer_peak]:][-1]
    
    
    beg_half=po[:peaks[closer_peak]]>=0.5*po[peaks[closer_peak]]
    beg_half=np.where(np.diff(beg_half))[0]
    if len(beg_half)>0:
        beg_half=beg_half[-1]
        # print('b',po[:peaks[closer_peak]][beg_half]/po[peaks[closer_peak]])
        beg_half=zo[:peaks[closer_peak]][beg_half]
    else: #we use the last point, 
        print('Warning, unable to find begining of half peak pressure length')
        beg_half=zo[:peaks[closer_peak]][0]
    FLHM=end_half-beg_half
    TPOequivalent=zo[peaks[closer_peak]]-h #we made this relative to the outplane
    msg=f'''
analytical estimation of TPO location for single element Tx with:
  Diameter = {aperture}
  Focal length = {focal_length}
  Frequency = {frequency}
  TPO value (relative to outplane) = {TPOequivalent}
  FLHM = {FLHM}
      '''
    print(msg)
    return h,TPOequivalent,FLHM

class PlanTUSTxConfig(object):
    def __init__(self, max_distance, 
                 min_distance, 
                 transducer_diameter, 
                 max_angle, 
                 plane_offset,
                 additional_offset, 
                 focal_distance_list, 
                 flhm_list,
                 IDTarget="",
                 fsl_path="/Users/spichardo/fsl/share/fsl/bin",
                 connectome_path="/Applications/wb_view.app/Contents/usr/bin",
                 freesurfer_path="/Applications/freesurfer/7.4.1/bin",
                 bUseGenericTransducerModel=False):

        # Maximum and minimum focal depth of transducer (in mm)
        self.max_distance = max_distance
        self.min_distance = min_distance

        # Aperture diameter (in mm)
        self.transducer_diameter = transducer_diameter

        # Maximum allowed angle for tilting of TUS transducer (in degrees)
        self.max_angle = max_angle

        # Offset between radiating surface and exit plane of transducer (in mm)
        self.plane_offset = plane_offset

        # Additional offset between skin and exit plane of transducer (in mm;
        # e.g., due to addtional gel/silicone pad)
        self.additional_offset = additional_offset

        # Focal distance and corresponding FLHM values (both in mm) according to, e.g.,
        # calibration report
        self.focal_distance_list = focal_distance_list
        self.flhm_list = flhm_list
        self.fsl_path = fsl_path
        self.connectome_path = connectome_path
        self.freesurfer_path = freesurfer_path
        self.IDTarget = IDTarget
        self.bUseGenericTransducerModel = bUseGenericTransducerModel

    def ExportYAML(self,fname):
        txconfig = {
            "max_distance": self.max_distance,
            "min_distance": self.min_distance,
            "transducer_diameter": self.transducer_diameter,
            "max_angle": self.max_angle,
            "plane_offset": self.plane_offset,
            "additional_offset": self.additional_offset,
            "focal_distance_list": self.focal_distance_list,
            "flhm_list": self.flhm_list,
            "fsl_path": self.fsl_path,
            "connectome_path": self.connectome_path,
            "freesurfer_path": self.freesurfer_path,
            "IDTarget": self.IDTarget,
            "bUseGenericTransducerModel": self.bUseGenericTransducerModel
        }

        with open(fname, "w") as file:
            yaml.dump(txconfig, file)


class RUN_PLAN_TUS(QObject):
    def __init__(self, MainApp, OptionsDlg):
        super().__init__(OptionsDlg)
        self.MainApp = MainApp # reference to main BabelBrain object
        self.OptionsDlg = OptionsDlg #reference to options dialog that is calling this class
        self.CalQueue = None
        self.CalProcess = None
        self.CaltimerTUSPlan = QTimer(self)
        self.CaltimerTUSPlan.timeout.connect(self.check_queue_TUSPlan)
        self._WorkingDialog = ClockDialog(OptionsDlg)
        self.select_vortex=-1

    def Execute(self):
        '''
        Run the external PlanTUS script
        '''
        BabelTxConfig=self.MainApp.AcSim.Config
        TxSystem = self.MainApp.Config['TxSystem']
        SelFreq=self.MainApp.Widget.USMaskkHzDropDown.property('UserData')
        TrajectoryType=self.MainApp.Config['TrajectoryType']
        Mat4Trajectory=self.MainApp.Config['Mat4Trajectory']

        PlanTUSRoot=self.OptionsDlg.ui.PlanTUSRootlineEdit.text()
        SimbNINBSRoot=self.OptionsDlg.ui.SimbNINBSRootlineEdit.text()
        FSLRoot=self.OptionsDlg.ui.FSLRootlineEdit.text()
        ConnectomeRoot=self.OptionsDlg.ui.ConnectomeRootlineEdit.text()
        FreeSurferRoot=self.OptionsDlg.ui.FreeSurferRootlineEdit.text()

        if TrajectoryType =='brainsight':
            RMat=ReadTrajectoryBrainsight(Mat4Trajectory)
        else:
            inMat=read_itk_affine_transform(Mat4Trajectory)
            RMat = itk_to_BSight(inMat)

        #we will reuse to recover the center of the trajectory
        self._RMat = RMat
        if 'CTX' in TxSystem or 'DPX' in TxSystem:
            bUseGenericTransducerModel = False
        else:
            bUseGenericTransducerModel = True

        if 'MinimalTPODistance' in BabelTxConfig: #ring-based Txs
            min_distance=BabelTxConfig['MinimalTPODistance']*1e3
            max_distance=BabelTxConfig['MaximalTPODistance']*1e3
            if 'FocalLength' in BabelTxConfig:
                plane_offset=(BabelTxConfig['FocalLength']-BabelTxConfig['NaturalOutPlaneDistance'])*1e3
            else:
                plane_offset=0.0
            additional_offset=self.MainApp.AcSim.Widget.SkinDistanceSpinBox.value()
            transducer_diameter=BabelTxConfig['TxDiam']*1e3
            focal_distance_list=BabelTxConfig['PlanTUS'][SelFreq]['FocalDistanceList']
            flhm_list=BabelTxConfig['PlanTUS'][SelFreq]['FHMLList']
            
        elif 'MinimalZSteering' in BabelTxConfig: #phased arrays
            Diameter=BabelTxConfig['TxDiam']
            if 'TxFoc' in BabelTxConfig: #shell Tx
                Focus=BabelTxConfig['TxFoc']
                plane_offset=(Focus-np.sqrt(Focus**2-Diameter**2/4))*1e3
                max_distance_plane =Focus*1e3-plane_offset 
                assert(max_distance_plane>=BabelTxConfig['MaximalDistanceConeToFocus'])
                plane_offset=Focus-BabelTxConfig['MaximalDistanceConeToFocus'] 
                additional_offset=BabelTxConfig['MaximalDistanceConeToFocus']-self.MainApp.AcSim.Widget.DistanceConeToFocusSpinBox.value()
            else: #flat array as the REMOPD
                plane_offset=0.0
                additional_offset=self.MainApp.AcSim.Widget.SkinDistanceSpinBox.value()
            transducer_diameter=BabelTxConfig['TxDiam']*1e3
            focal_distance_list=BabelTxConfig['PlanTUS'][SelFreq]['FocalDistanceList']
            flhm_list=BabelTxConfig['PlanTUS'][SelFreq]['FHMLList']
            min_distance=np.min(focal_distance_list)
            max_distance=np.max(focal_distance_list)
        elif 'BSonix35mm' in BabelTxConfig: #BSonixTx
            transducer_diameter=BabelTxConfig['CaseDiameter']
            plane_offset=0.0
            additional_offset=self.MainApp.AcSim.Widget.SkinDistanceSpinBox.value()
            seldevice = self.MainApp.AcSim.Widget.GetTxModel()
            focal_distance_list=BabelTxConfig['PlanTUS'][SelFreq][seldevice]['FocalDistanceList']
            flhm_list=BabelTxConfig['PlanTUS'][SelFreq][seldevice]['FHMLList']
            min_distance=np.min(focal_distance_list)
            max_distance=np.max(focal_distance_list)
        else: # single element Tx
            FocalLength = self.MainApp.AcSim.FocalLengthSpinBox.value()
            transducer_diameter = self.MainApp.AcSim.DiameterSpinBox.value()
            #we use analytical formula of acoustic axis from the O'Neil paper
            plane_offset,TPOequivalent,FLHM=FindTPOEquivalent(SelFreq,transducer_diameter*1e-3,FocalLength*1e-3)
            plane_offset*=1e3
            focal_distance_list = [TPOequivalent*1e3]
            flhm_list = [FLHM*1e3]
            min_distance=TPOequivalent*1e3
            max_distance=min_distance
            
        additional_offset=np.max([0.0,additional_offset]) #negative values is for special cases to "invade" the scalp

        # Create a new PlanTUSTxConfig object with the current values
        plan_tus_config = PlanTUSTxConfig(
            transducer_diameter=float(transducer_diameter),
            min_distance=float(min_distance),
            max_distance=float(max_distance),
            max_angle=10.0, #we keep it constant for the time being
            plane_offset=float(plane_offset),
            additional_offset=float(additional_offset),
            focal_distance_list=focal_distance_list,
            flhm_list=flhm_list,
            IDTarget=self.MainApp.Config['ID'],
            fsl_path=FSLRoot,
            connectome_path=ConnectomeRoot,
            freesurfer_path=FreeSurferRoot,
            bUseGenericTransducerModel=bUseGenericTransducerModel
        )
  

        t1Path=self.MainApp.Config['T1W']
        voxel_radius=2/np.array(nibabel.load(t1Path).header.get_zooms()) #2mm radius
        raddi=tuple(np.ceil(voxel_radius).astype(int))

        basepath=os.path.split(t1Path)[0]
        TxConfigName = basepath + os.sep + "PlanTUSTxConfig.yaml"
        # Export the configuration to a YAML file
        plan_tus_config.ExportYAML(TxConfigName)
        
        mshPath=glob.glob(self.MainApp.Config['simbnibs_path'] + os.sep + "*.msh")[0]
        maskPath=Mat4Trajectory.replace('.txt','_PlanTUSMask.nii.gz')
        self.PlanOutputPath=os.path.split(mshPath)[0]+os.sep+'PlanTUS'+os.sep+maskPath.split(os.sep)[-1].replace('.nii.gz','')
        print('self.PlanOutputPath', self.PlanOutputPath)

        create_target_mask(t1Path, RMat[:3,3], maskPath,raddi=raddi)

        scriptbase=os.path.join(resource_path(),"ExternalBin"+os.sep+"PlanTUS"+os.sep)
        queue=Queue()
        self.CalQueue=queue

        self.planTUSargs=(queue,
                         scriptbase,
                         SimbNINBSRoot,
                         PlanTUSRoot,
                         t1Path,
                         mshPath,
                         maskPath,
                         TxConfigName)
        
        #we check if files are already generated, in case the user may just want to refine the location

        if os.path.isfile(self.PlanOutputPath+os.sep+'skin.surf.gii'):
            ret = QMessageBox.question(self.OptionsDlg,'', "PlanTUS results already exist for this target.\nDo you want to recalculate?\nSelect No to reload", QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.No:
                if not self.showTUSPlanViewer():
                    return #we stop here
                print('Generating trajectory for ID', self.select_vortex)
                self.GenerateTrajectory(self.select_vortex)
                return

        print('Starting PlanTUS for target', self.MainApp.Config['ID'])
        self.RunningTUSPlan = True

        fieldWorkerProcess = Process(target=RunPlanTUSBackground, 
                                            args=self.planTUSargs)
        
        self.CalProcess=fieldWorkerProcess
        self.T0Cal=time.time()
        fieldWorkerProcess.start()     
        self.CaltimerTUSPlan.start(100)
        mainWindowCenter = self.OptionsDlg.geometry().center()

        self._WorkingDialog.move(
            mainWindowCenter.x() - 50,
            mainWindowCenter.y() - 50
        )
        self._WorkingDialog.show()
        self.OptionsDlg.setEnabled(False)


    def callBackAfterGenTrajectory(self, select_vortex):
        print("Callback after generating trajectory for cell id:", select_vortex)
        self.select_vortex=select_vortex
        self.dlgResultsPlanTUS.accept()
        # progress.

    def GenerateTrajectory(self, select_vortex):

        self.RunningTUSPlan = False
        fieldWorkerProcess = Process(target=RunPlanTUSBackground, 
                                            args=self.planTUSargs,
                                            kwargs={'runOnlyTrajectory': select_vortex})
        
        self.CalProcess=fieldWorkerProcess
        self.T0Cal=time.time()
        fieldWorkerProcess.start()     
        self.CaltimerTUSPlan.start(100)
        mainWindowCenter = self.OptionsDlg.geometry().center()

        self._WorkingDialog.move(
            mainWindowCenter.x() - 50,
            mainWindowCenter.y() - 50
        )
        self._WorkingDialog.show()
        self.OptionsDlg.setEnabled(False)

    def check_queue_TUSPlan(self):

        # progress.
        
        bNoError=True
        bDone=False
        while self.CalQueue and not self.CalQueue.empty():
            cMsg=self.CalQueue.get()
            if type(cMsg) is str:
                print(cMsg,end='')
                if '--Babel-Brain-Low-Error' in cMsg\
                   or '--Babel-Brain-Success' in cMsg:
                    if '--Babel-Brain-Low-Error' in cMsg:
                        bNoError=False
                    self.CaltimerTUSPlan.stop()
                    self.CalProcess.join()
                    bDone=True
                
        if bDone:
            self.OptionsDlg.setEnabled(True)
            self._WorkingDialog.hide()
            if bNoError:
                TEnd=time.time()
                TotalTime = TEnd-self.T0Cal
                if self.RunningTUSPlan:
                    print('Total time',TotalTime)
                    print("*"*40)
                    print("*"*5+" DONE PlanTUS.")
                    print("*"*40)
                    if not self.showTUSPlanViewer():
                        return

                    print('Generating trajectory for ID', self.select_vortex)

                    self.GenerateTrajectory(self.select_vortex)
                    #now we re run to generate the trajectory
                    return

                #this means we are completing the trajectory
                print('Total time',TotalTime)
                print("*"*40)
                print("*"*5+" DONE Trajectory generation.")
                print("*"*40)
                print('looking for trajectories')
                
                basepath=self.PlanOutputPath            

                #we look for new trajectory files
                trajFiles=glob.glob(basepath+os.sep+'**'+os.sep+'*Localite.mat',recursive=True)
                if len(trajFiles)>0:
                    assert(len(trajFiles)==1)
                    for trajFile in trajFiles:
                        id = self.MainApp.Config['ID']+'_PlanTUS'
                        transform=loadmat(trajFile)['position_matrix']
                        TT=transform.copy()
                        # we need to convert the transform to the correct format
                        TT[:3,0] = -transform[0:3,1]
                        TT[:3,1] = transform[0:3,2] 
                        TT[:3,2] = -transform[0:3,0]
                        transform=TT 
                        print("Found trajectory file:", trajFile)
                        # we will reuse to recover the center of the trajectory
                        outString=templateBSight.format(m0n0=transform[0,0],
                                m0n1=transform[1,0],
                                m0n2=transform[2,0],
                                m1n0=transform[0,1],
                                m1n1=transform[1,1],
                                m1n2=transform[2,1],
                                m2n0=transform[0,2],
                                m2n1=transform[1,2],
                                m2n2=transform[2,2],
                                X=self._RMat[0,3],
                                Y=self._RMat[1,3],
                                Z=self._RMat[2,3],
                                name=id)
                        foutnameBSight = trajFile.split('Localite.mat')[0] + 'BSight.txt'
                        with open(foutnameBSight, 'w') as f:
                            f.write(outString)

                        transform = BSight_to_itk(transform)
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
                                        X=self._RMat[0,3],
                                        Y=self._RMat[1,3],
                                        Z=self._RMat[2,3])
                        foutnameSlicer = trajFile.split('Localite.mat')[0] + 'Slicer.txt'
                        with open(foutnameSlicer, 'w') as f:
                            f.write(outString)

                        bdir,sfile = os.path.split(trajFile)
                        tfile = sfile.replace('position_matrix','transducer').replace('_Localite.mat','.surf.gii')
                        self.showFinalResults(bdir+os.sep+tfile)

                    ret = QMessageBox.question(self.OptionsDlg,'', "Do you want to use the\n PlanTUS results to update the trajectory? ",QMessageBox.Yes | QMessageBox.No)
                    if ret == QMessageBox.Yes:
                        TrajectoryType=self.MainApp.Config['TrajectoryType']
                        if TrajectoryType =='brainsight':
                            ext='*BSight.txt'
                            lastfoutname=foutnameBSight
                        else:
                            ext='*Slicer.txt'
                            lastfoutname=foutnameSlicer

                        finalfname=self.MainApp.Config['Mat4Trajectory'].split('.txt')[0]+'_PlanTUS.txt'
                                
    
                        if len(trajFiles)>1:
                            fname = QFileDialog.getOpenFileName(self.OptionsDlg, "Select txt file with calibration input fields",basepath, "Text files ("+ext+")")[0]
                            if len(fname)>0:
                                shutil.copy(fname, finalfname)
                                self.MainApp.Config['Mat4Trajectory'] = finalfname
                                self.MainApp.Config['ID'] = id
                                self.MainApp.UpdateWindowTitle()
                        else:
                            fname = lastfoutname
                            shutil.copy(fname, finalfname)
                            self.MainApp.Config['Mat4Trajectory'] = finalfname
                            self.MainApp.Config['ID'] = id
                            self.MainApp.UpdateWindowTitle()
            else:
                print("*"*40)
                print("*"*5+" Error in execution of PlanTUS.")
                print("*"*40)

    def showFinalResults(self,TxResultSurface):
        DlgResults=QDialog(self.OptionsDlg)
        DlgResults.setWindowTitle("Trajectory Results")

        layout = QVBoxLayout()
        DlgResults.setLayout(layout)

        gifti_files = []
        gifti_files.append(self.PlanOutputPath+os.sep+'skin.surf.gii')
        gifti_files.append(TxResultSurface)
        
        widget = FinalResultViewer(gifti_files)
        layout.addWidget(widget)
        DlgResults.resize(600, 600)
        DlgResults.exec()

    def showTUSPlanViewer(self):
        DlgResults=QDialog(self.OptionsDlg)
        DlgResults.setWindowTitle("PlanTUS Results")

        layout = QVBoxLayout()
        DlgResults.setLayout(layout)

        gifti_files = []
        gifti_files.append((self.PlanOutputPath+os.sep+'skin.surf.gii',
                            self.PlanOutputPath+os.sep+'distances_skin.func.gii',
                            self.PlanOutputPath+os.sep+'distances_skin_thresholded.func.gii',
                            [0,100],
                            'Distance'))
        gifti_files.append((self.PlanOutputPath+os.sep+'skin.surf.gii',
                            self.PlanOutputPath+os.sep+'target_intersection_skin.func.gii',
                            self.PlanOutputPath+os.sep+'distances_skin_thresholded.func.gii',
                            [0,5],
                            'Target Intersection'))
        gifti_files.append((self.PlanOutputPath+os.sep+'skin.surf.gii',
                            self.PlanOutputPath+os.sep+'angles_skin.func.gii',
                            self.PlanOutputPath+os.sep+'distances_skin_thresholded.func.gii',
                            [0,20],
                            'Angle'))
        gifti_files.append((self.PlanOutputPath+os.sep+'skin.surf.gii',
                            self.PlanOutputPath+os.sep+'skin_skull_angles_skin.func.gii',
                            self.PlanOutputPath+os.sep+'distances_skin_thresholded.func.gii',
                            [0,20],
                            'Skin-Skull Angle'))

        widget = MultiGiftiViewerWidget(gifti_files,MaxViews=4,callBackAfterGenTrajectory=self.callBackAfterGenTrajectory)
        layout.addWidget(widget)
        self.dlgResultsPlanTUS = DlgResults
        DlgResults.resize(1700, 700)
        return self.dlgResultsPlanTUS.exec()


def RunPlanTUSBackground(queue,
                        scriptbase,
                        SimbNINBSRoot,
                        PlanTUSRoot,
                        t1Path,
                        mshPath,
                        maskPath,
                        TxConfigName,
                        runOnlyTrajectory=-1):
    class InOutputWrapper(object):
       
        def __init__(self, queue, stdout=True):
            self.queue=queue
            if stdout:
                self._stream = sys.stdout
                sys.stdout = self
            else:
                self._stream = sys.stderr
                sys.stderr = self
            self._stdout = stdout

        def write(self, text):
            self.queue.put(text)

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

    stdout = InOutputWrapper(queue,True)
  
    try:
        if sys.platform == 'linux' or _IS_MAC:
            if sys.platform == 'linux':
                shell='bash'
                path_script =scriptbase+"run_linux.sh"
            elif _IS_MAC:
                shell='zsh'
                path_script = scriptbase+"run_mac.sh"

            print("Starting PlanTUS")
            if _IS_MAC:
                cmd ='source "'+path_script + '" "' + SimbNINBSRoot + '" "' + PlanTUSRoot + '" "' + t1Path + '" "' + mshPath +'" "' + maskPath + '" "'+TxConfigName+'"'
                if runOnlyTrajectory>-1:
                    cmd += ' --do_only_trajectory '+str(runOnlyTrajectory)
                print(cmd)
                result = os.system(cmd)
            else:
                args= [shell,
                        path_script,
                        SimbNINBSRoot,
                        PlanTUSRoot,
                        t1Path,
                        mshPath,
                        maskPath,
                        TxConfigName]
                if runOnlyTrajectory>-1:
                    args.append('--do_only_trajectory')
                    args.append(str(runOnlyTrajectory))
                result = subprocess.run(args, capture_output=True, text=True)
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                result=result.returncode 
        else:
            path_script = os.path.join(resource_path(),"ExternalBin/PlanTUS/run_win.bat")
            
            print("Starting PlanTUS")
            args= [path_script,
                    SimbNINBSRoot,
                    PlanTUSRoot,
                    t1Path,
                    mshPath,
                    maskPath,
                    TxConfigName]
            if runOnlyTrajectory>-1:
                    args.append('--do_only_trajectory')
                    args.append(str(runOnlyTrajectory))
            result = subprocess.run(args, capture_output=True, text=True,shell=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            result=result.returncode 
        print("PlanTUS Finished")
        print("--Babel-Brain-Success")
    except BaseException as e:
        print('--Babel-Brain-Low-Error')
        print(traceback.format_exc())
        print(str(e))
    